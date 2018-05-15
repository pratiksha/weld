#ifndef SMALLOC_H
#define SMALLOC_H

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

struct memblock;

typedef struct memhdr {
  struct memblock* prev;
  struct memblock* next;
  size_t data_size;
} memhdr;

/* Heap blocks. */
typedef struct memblock {
  memhdr header;
  void*  data;
} memblock;

static void* heap;
static size_t max_heap_size;
static void* stack;
static size_t max_stack_size;
static uintptr_t ebp; /* "base pointer"  */
static uintptr_t esp; /* "stack pointer" */

void fn_start();
void fn_end();

void* smalloc(size_t nbytes);
void* srealloc(void* data, size_t nbytes);
void* smalloc_aligned(size_t nbytes);
void* salloca(size_t nbytes);

int   sfree(void* ptr);

void* init_s3_file(char* s3_hostname, char* s3_file);

int   init_heap(void* start, size_t max_size);
void  free_heap();

int   init_stack(void* start, size_t max_size);
void  free_stack();

#endif // SMALLOC_H