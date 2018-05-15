/* malloc + free for shared memory */

#include <error.h>
#include <pthread.h>
#include <signal.h>
#include <stdalign.h>

#include "smalloc.h"
#include "mem.h"

pthread_mutex_t stackl;
pthread_mutex_t heapl; /* just lock the entire heap for now */

void fn_start() {
  pthread_mutex_lock(&stackl);
  /* Push current stack pointer onto stack. */
  *((uintptr_t*)ebp) = ebp;
  esp = ebp + sizeof(void*);
  ebp = esp;
  pthread_mutex_unlock(&stackl);
}

/* free everything in this function */
void fn_end() {
  pthread_mutex_lock(&stackl);
  /* Free everything up to base pointer and restore base pointer. */
  esp = ebp - sizeof(void*);
  ebp = *((uintptr_t*)(esp));
  pthread_mutex_unlock(&stackl);
}

/* Allocate the requested number of bytes.
 * Unlike salloca, smalloc'd data is not freed on fn_end.
 */
void* smalloc(size_t nbytes) {
  pthread_mutex_lock(&heapl);

  fprintf(stderr, "getting cur block\n");
  memblock* cur_block = (memblock*)(heap);
  fprintf(stderr, "getting next block\n");
  fprintf(stderr, "header size %lu\n", (cur_block->header).data_size);
  memblock* next_block = (memblock*)((char*)heap + (cur_block->header).data_size);
  fprintf(stderr, "got blocks\n");
  (next_block->header).prev = cur_block;
  (next_block->header).next = NULL;
  (next_block->header).data_size = nbytes;
  next_block->data = NULL;

  heap = next_block;
    
  pthread_mutex_unlock(&heapl);

  return &(next_block->data);
}

/* For now, just allocate a new block. */
void* srealloc(void* data, size_t nbytes) {
  return smalloc(nbytes);
}

bool is_aligned(uintptr_t addr) {
  return (PGADDR(addr) == addr);
}

/* Allocate the requested number of bytes, aligned to hugepage boundary.
 * Unlike salloca, smalloc'd data is not freed on fn_end.
 */
void* smalloc_aligned(size_t nbytes) {
  pthread_mutex_lock(&heapl);

  fprintf(stderr, "getting cur block\n");
  memblock* cur_block = (memblock*)(heap);
  fprintf(stderr, "getting next block\n");
  fprintf(stderr, "header size %lu\n", (cur_block->header).data_size);

  // get projected location for next data allocation
  uintptr_t next_block_addr = (uintptr_t)((char*)(cur_block) + sizeof(memhdr) + (cur_block->header).data_size);
  uintptr_t next_data_addr = next_block_addr + sizeof(memhdr);

  uintptr_t pgnum = (uintptr_t)PGADDR_TO_HUGEPG_NUM(next_data_addr);
  uintptr_t aligned_addr = PGNUM_TO_PGADDR(pgnum);
  if ( aligned_addr < next_data_addr ) pgnum+= HUGEPG_PAGES; // not page aligned - align data to next (huge)page boundary
  aligned_addr = PGNUM_TO_PGADDR(pgnum);
  if ( (aligned_addr - next_block_addr) < sizeof(memhdr) ) { // can't fit header in remaining space
    pgnum+= HUGEPG_PAGES;
    aligned_addr = PGNUM_TO_PGADDR(pgnum);
  }

  memblock* next_block = (memblock*)(aligned_addr - sizeof(memhdr)); // page aligned block
  fprintf(stderr, "got blocks\n");
  (cur_block->header).next = next_block;
  (next_block->header).prev = cur_block;
  (next_block->header).next = NULL;
  (next_block->header).data_size = nbytes;
  next_block->data = NULL;
  
  heap = next_block;
    
  pthread_mutex_unlock(&heapl);

  return &(next_block->data);
}

/* Free smalloc'd data.
 * Returns 0 on success, -1 if the requested pointer is not a valid pointer to free (?).
 * TODO: make free blocks available for reuse.
 */
int sfree(void* ptr) {
  pthread_mutex_lock(&heapl);

  memblock* req_block = (memblock*)((char*)ptr - sizeof(memhdr)); /* TODO: not sure this subtraction works. */
  memblock* prev_block = (req_block->header).prev;
  memblock* next_block = (req_block->header).next;

  if ( prev_block != NULL ) {
    (prev_block->header).next = next_block;
  }
  if ( next_block != NULL ) {
    (next_block->header).prev = prev_block;
  }
    
  pthread_mutex_unlock(&heapl);

  return 0;
}

/* Allocate the requested number of bytes. If we ran out of space, try to request more. 
 * Unlike alloca, does not automatically clear on function exit!
 * Call fn_end to clear the current stack frame.
 */
void* salloca(size_t nbytes) {
  pthread_mutex_lock(&stackl);
  
  uintptr_t ret = esp;
  esp += nbytes;

  if ( esp > (uintptr_t)((char*)stack + max_stack_size) ) {
    /* TODO out of stack space; request more! */
    esp = (uintptr_t)ret;
    pthread_mutex_unlock(&stackl);
    return NULL;
  }

  pthread_mutex_unlock(&stackl);
  return (void*)ret;
}

/* Get the size of a file from s3 and allocate enough memory for it (on the heap). 
 */
void* init_s3_file(char* s3_hostname, char* s3_file) {
  size_t s3_size = 100; //get_s3_size(s3_hostname, s3_file);
  void* data_ptr = smalloc(s3_size);
  return data_ptr;
}

/* store location of heap, to use in calls to malloc */
int init_heap(void* start, size_t max_size) {
  pthread_mutex_init(&heapl, NULL);
  pthread_mutex_lock(&heapl);

  memblock* m = (memblock*)(start);
  (m->header).prev = NULL;
  (m->header).next = NULL;
  m->data = NULL;
  
  heap = start;
  max_heap_size = max_size;

  pthread_mutex_unlock(&heapl);
  return 0;
}

void free_heap() {
  pthread_mutex_destroy(&heapl);
}

/* store initial location of stack, to use in calls to alloca */
int init_stack(void* start, size_t max_size) {
  pthread_mutex_init(&stackl, NULL);

  pthread_mutex_lock(&stackl);
  stack = start;
  ebp = (uintptr_t)(stack);
  esp = (uintptr_t)(stack);
  max_stack_size = max_size;

  pthread_mutex_unlock(&stackl);
  return 0;
}

void free_stack() {
  pthread_mutex_destroy(&stackl);
}