#ifndef _MEM_H_
#define _MEM_H_

#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

#define PG_WRITE 0x2
#define PG_PRESENT 0x1
#define PG_BITS 12
#define PG_SIZE (1 << (PG_BITS)) /* 4096 bytes */
#define HUGEPG_BITS 8
#define HUGEPG_PAGES (1 << (HUGEPG_BITS)) /* 256 pages per hugepage */
#define HUGEPG_SIZE (1 << (PG_BITS + HUGEPG_BITS)) /* about 1 MB worth of 4096-byte pages */

// #define NUM_PAGES 20000000L /* about 2.12 gigabytes of (4096-byte) pages */
#define NUM_PAGES 1000L

// Convert address to the start of the page.
static inline uintptr_t PGADDR(uintptr_t addr) {
  return addr & ~(PG_SIZE - 1);
}

// Convert page number to address of start of page.
static inline uintptr_t PGNUM_TO_PGADDR(uintptr_t pgnum) {
  return pgnum << PG_BITS;
}

static inline uintptr_t PGNUM_TO_HUGEPG_NUM(uintptr_t pgnum) {
  return (pgnum >> HUGEPG_BITS) << HUGEPG_BITS;
}

static inline uintptr_t PGADDR_TO_HUGEPG_NUM(uintptr_t addr) {
  return (addr >> (PG_BITS + HUGEPG_BITS)) << HUGEPG_BITS;
}

// Convert page address to the page number.
static inline uintptr_t PGADDR_TO_PGNUM(uintptr_t addr) {
  return addr >> PG_BITS;
}
 
#endif  // _MEM_H_
