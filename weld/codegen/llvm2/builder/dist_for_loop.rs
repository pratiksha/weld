//! Code generation for a distributed for loop.
//!
//! Shards input data appropriately.
//! Generates RPCs (as UDFs) to distribute a serialized function to workers,
//! and then merges the results of the RPCs as necessary based on the provided builder type.
//!
//! Note that this backend assumes distributed shared memory,
//! i.e. arguments are not serialized being passed to the RPC; only addresses are passed.
//! Hence, all arguments passed to the RPC must be allocated on the globally-accessible heap.


extern crate llvm_sys;

use std::ffi::CString;

use ast::*;
use ast::Type::*;
use ast::IterKind::*;
use error::*;
use runtime::WeldRuntimeErrno;
use sir::*;

use self::llvm_sys::{LLVMIntPredicate, LLVMLinkage};
use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use codegen::llvm2::llvm_exts::*;
use codegen::llvm2::llvm_exts::LLVMExtAttribute::*;
use codegen::llvm2::{SIR_FUNC_CALL_CONV, LLVM_VECTOR_WIDTH};

use super::{CodeGenExt, FunctionContext, LlvmGenerator};

/// C function to compute shard sizes.
const PARTITION_FUNC: &str = "partition_sizes";

/// C RPC function.
const DISPATCH_FUNC: &str = "dispatch_one";

/// An internal trait for generating parallel For loops.
pub trait DistForLoopGenInternal {
    /// Entry point to generating a distributed for loop.
    ///
    /// This is the only function in the trait that should be called -- all other methods are
    /// helpers. 
    unsafe fn gen_dist_for_internal(&mut self,
                               ctx: &mut FunctionContext,
                               output: &Symbol,
                               distfor: &DistForData) -> WeldResult<()>; 
    /// Generates code to divide the input iters into partitions.
    /// 
    /// The number of partitions is determined by the number of workers, which is determined by a config option.
    unsafe fn gen_partitions(&mut self,
                             ctx: &mut FunctionContext,
                             func: &SirFunction,
                             distfor: &DistForData) -> WeldResult<LLVMValueRef>;
    /// Generates code to pack arguments to send in the RPC.
    unsafe fn gen_args(&mut self,
                       ctx: &mut FunctionContext,
                       func: &SirFunction,
                       distfor: &DistForData) -> WeldResult<()>;
    /// Generates code to call an external C function that executes the inner loop and merge the results.
    unsafe fn gen_rpc(&mut self,
                      ctx: &mut FunctionContext,
                      func: &String,
                      distfor: &DistForData) -> WeldResult<()>;
}

impl DistForLoopGenInternal for LlvmGenerator {
    /// Entry point to generating a distributed for loop.
    unsafe fn gen_dist_for_internal(&mut self,
                                    ctx: &mut FunctionContext,
                                    output: &Symbol,
                                    distfor: &DistForData) -> WeldResult<()> {
        if !self.conf.distribute {
            compile_err!("Distribute must be enabled in order to use DistFor");
        }

        Ok(())
    }

    /// Generates code to compute sizes of partitions for the input iters.
    /// 
    /// The number of partitions is determined by the number of workers,
    /// which is determined by a config option.
    /// 
    /// TODO: Currently assumes/requires all input vectors to be of the same length,
    /// but this can be checked in generated code.
    unsafe fn gen_partitions(&mut self,
                             ctx: &mut FunctionContext,
                             func: &SirFunction,
                             data_size: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        let run = ctx.get_run();
        
        // allocate an array to store the {start, size} pairs
        let res_tys = vec![self.i64_type(), self.i64_type()];
        let elt_type = LLVMStructType(res_tys.as_mut_ptr(), res_tys.len() as u32, 0);
        let elt_size = self.size_of(elt_type);
        let nelts = self.i32(self.conf.nworkers);
        let alloc_size = LLVMBuildMul(ctx.builder, elt_size, nelts, c_str!(""));
        let output_bytes = self.intrinsics.call_weld_run_malloc(ctx.builder,
                                                                run, alloc_size, None);
        let output_ptr_ty = LLVMPointerType(elt_type, 0);

        // cast raw bytes to correct array type
        let ret_array = LLVMBuildBitCast(ctx.builder, output_bytes, output_ptr_ty, c_str!(""));

        // input args: data size, nworkers, increment
        let mut arg_tys = vec![self.i64_type(), self.i32_type(), self.i64_type(), output_ptr_ty];
        let final_type = LLVMStructType(arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let size = self.size_of_bytes(final_type) as i64;
        let ll_size = self.i64(size);
        let input_bytes = self.intrinsics.call_weld_run_malloc(ctx.builder,
                                                               run, ll_size, None);

        // call external function to create partitions
        self.intrinsics.add(PARTITION_FUNC, output_ptr_ty, &mut arg_tys);

        // TODO make min increment a conf option, or argument to DistFor
        let mut arg_values = vec![data_size, self.conf.nworkers, self.i64(1), ret_array];

        let _ = self.intrinsics.call(ctx.builder, PARTITION_FUNC, &mut arg_values);

        Ok(ret_array);
    }

    /// Generates code to pack arguments to send in the RPC.
    /// 
    /// The order of arguments is: [iters, args].
    unsafe fn gen_args(&mut self,
                       ctx: &mut FunctionContext,
                       func: &SirFunction,
                       distfor: &DistForData) -> WeldResult<LLVMValueRef> {
        // Allocate arrays with enough space for all the required args (addresses).
        let nworkers = self.conf.nworkers;

        // compute size required to store all args
        // TODO: can check size equality here
        let mut arg_tys = vec![];
        let mut data_size: Option<LLVMValueRef> = None;
        for iter in distfor.data.iter() {
            let iter_ty = ctx.sir_function.symbol_type(iter.data)?;
            match iter_ty {
                Vector(ref elem_ty) => {
                    arg_tys.push(self.llvm_type(iter_ty)?);

                    use codegen::llvm2::vector::VectorExt;
                    data_size = Some(VectorExt::gen_size(self, ctx.builder, iter_ty, iter.data))?;
                }
                DistVec(ref elem_ty) => {
                    arg_tys.push(self.llvm_type(Vector(elem_ty))?); // received type will be Vector
                }
                _ => return compile_err!("Iter in DistFor must be Vector or DistVec: {}", iter_ty)
            }
        }

        let sizes = match data_size {
            Some(ref size) => {
                // at least one iter needs to be sharded. Generate shards.
                gen_partitions(ctx, func, size)
            }
            None => {
                // all iters were already distributed. This value will never be used
                self.void_type()
            }
        };
    
        for arg in distfor.args.iter() {
            arg_tys.push(self.llvm_type(ctx.sir_function.symbol_type(arg)?)?);
        }

        let final_type = LLVMStructType(arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let run = ctx.get_run();
        let size = self.size_of_bytes(final_type) as i64;
        let ll_size = self.i64(size);

        // allocate space to store args for each worker
        let arg_ptrs = vec![];
        for i in 0..nworkers {
            let bytes = self.intrinsics.call_weld_run_malloc(ctx.builder,
                                                             run, ll_size, None);
            let arg_ptr = LLVMBuildBitCast(ctx.builder, bytes,
                                           LLVMPointerType(final_type, 0), c_str!(""));
            arg_ptrs.push(arg_ptr);
        }

        for (i, iter) in distfor.data.iter().enumerate() {
            match iter.kind {
                ScalarIter if iter.start.is_some() => {
                    // Start/end/stride are supported, but they have to be propagated correctly to the dispatched function.
                    unimplemented!()
                }
                ScalarIter => {
                    let iter_ty = ctx.sir_function.symbol_type(iter.data)?;
                    for j in 0..nworkers {
                        let args = LLVMBuildStructGEP(ctx.builder,
                                                      arg_ptrs,
                                                      j,
                                                      c_str!(""));
                        let insert_ptr = LLVMBuildStructGEP(ctx.builder,
                                                            args,
                                                            i as u32,
                                                            c_str!(""));

                        let insert_val = match iter_ty {
                            Vector(ref elem) => {
                                // Not sharded. Create a slice.
                                use codegen::llvm2::vector::VectorExt;

                                let shard = LLVMBuildStructGEP(ctx.builder,
                                                               sizes, j as u32, c_str!(""));
                                let start = LLVMBuildStructGEP(ctx.builder,
                                                               shard, 0, c_str!(""));
                                let size = LLVMBuildStructGEP(ctx.builder,
                                                              shard, 1, c_str!(""));
                                
                                let slice = VectorExt::gen_slice(self, ctx.builder, iter.data,
                                                                 start, size)?;
                                slice
                            }
                            
                            DistVec(ref elem) => {
                                // Already sharded, so just push the jth shard to each worker.
                                use codegen::llvm2::dist_vector::DistVectorExt;
                                DistVectorExt::gen_vec_at(self, iter.data, j)?
                            }
                        };

                        LLVMBuildStore(ctx.builder, insert_val, insert_ptr);
                    }
                }
                RangeIter => {
                    unimplemented!()
                }
                _ => return compile_err!("DistFor requires ScalarIter or RangeIter: {}", iter.kind)
            }
        }

        // Now that we have the data shards, append any other args accessed in the computation.
        let base_idx = distfor.data.len();
        for i in 0..nworkers {
            // Get the ith args vector.
            let args_i = LLVMBuildStructGEP(ctx.builder,
                                            arg_ptrs,
                                            i,
                                            c_str!(""));

            // Append the remaining args.
            for (j, arg) in distfor.args.iter().enumerate() {
                let insert_idx = j + base_idx;
                let arg_pointer = LLVMBuildStructGEP(ctx.builder,
                                                     args_i,
                                                     insert_idx as u32,
                                                     c_str!(""));
                let value = self.load(ctx.builder, ctx.get_value(arg)?)?;
                LLVMBuildStore(ctx.builder, value, arg_pointer);
            }
        }

        Ok(arg_ptrs)
    }
    
    /// Generates code to call an external C RPC that dispatches the inner loop, and retrieves and merges the results.
    unsafe fn gen_rpc(&mut self,
                      ctx: &mut FunctionContext,
                      func: &String,
                      args: &Vec<LLVMValueRef>,
                      distfor: &DistForData) -> WeldResult<()> {
        let output_pointer = ctx.get_value(output)?;
        let return_ty = self.llvm_type(ctx.sir_function.symbol_type(output)?)?;
        let mut arg_tys = vec![];

        // The RPC takes an i8* and outputs an i8*, so we have to pack the args into a struct
        // and bitcast the struct to i8 before passing to the RPC.
        for arg in args.iter() {
            arg_tys.push(self.llvm_type(ctx.sir_function.symbol_type(arg)?)?);
        }
        
        let final_type = LLVMStructType(arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let run = ctx.get_run();
        let size = self.size_of_bytes(final_type) as i64;
        let ll_size = self.i64(size);
        let bytes = self.intrinsics.call_weld_run_malloc(ctx.builder,
                                                         run, ll_size, None);
        let arg_struct_ptr = LLVMBuildBitCast(ctx.builder, bytes,
                                              LLVMPointerType(final_type, 0), c_str!(""));
        
        for (i, arg) in args.iter().enumerate() {
            let arg_pointer = LLVMBuildStructGEP(ctx.builder,
                                                 arg_struct_ptr,
                                                 i as u32,
                                                 c_str!(""));
            let value = self.load(ctx.builder, ctx.get_value(arg)?)?;
            LLVMBuildStore(ctx.builder, value, arg_pointer);
        }
        
        let output_ty = LLVMPointerType(return_ty, 0);
        let input_cast_ptr = LLVMBuildBitCast(ctx.builder, arg_struct_ptr, self.void_pointer_type(), c_str!(""));
        let output_cast_ptr = LLVMBuildBitCast(ctx.builder, output_pointer, self.void_pointer_type(), c_str!(""));
        
        // input and output are i8 pointers
        let mut i8_tys = vec![self.void_pointer_type(), self.void_pointer_type()]; 
        
        let fn_ret_ty = self.void_type();
        self.intrinsics.add(DISPATCH_FUNC, fn_ret_ty, &mut i8_tys);
        
        let mut cast_args = vec![input_cast_ptr, output_cast_ptr];
        let _ = self.intrinsics.call(ctx.builder, DISPATCH_FUNC, &mut cast_args)?;
        
        Ok(())
    }
}