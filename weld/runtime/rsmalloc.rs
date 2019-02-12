//! Interface to Clamor shared memory allocator for use in the Weld runtime.

use runtime::Ptr;

#[link(name="smalloc")]
extern "C" {
    pub fn smalloc(nbytes: u64) -> Ptr;
    pub fn srealloc(pointer: Ptr, nbytes: u64) -> Ptr;
    pub fn sfree(pointer: Ptr) -> i32;
}
