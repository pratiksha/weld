//! Extensions to generate _distributed_ vectors. 
//!
//! Distributed vectors are partitioned vectors that are located on the cluster.
//! They are represented as an ordered sequence of contiguous vectors.
//! Each constituent vector must be composed of the same types, but the vectors need not be the
//! same length. The distributed vector is always initialized with a list of allocated vectors
//! and is immutable once constructed, although the constitutent vectors _may_ not be immutable.
//! (In general, however, they will be generated from a Res, and therefore will actually be immutable.)
//! 
//! This module provides a wrapper interface for methods and utilities on distributed vector types. Other
//! modules use it for distributed-vector-related functionality or operators over distributed vectors.

extern crate llvm_sys;

use std::ffi::CString;

use ast::Type;
use error::*;

use super::llvm_exts::*;
use super::llvm_exts::LLVMExtAttribute::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMIntPredicate::*;

use super::LLVM_VECTOR_WIDTH;
use super::CodeGenExt;
use super::LlvmGenerator;
use super::intrinsic::Intrinsics;
use super::vector;

/// Index of the pointer into the vector data structure.
pub const POINTER_INDEX: u32 = 0;
/// Index of the size into the vector data structure.
pub const NVECS_INDEX: u32 = 1;

/// Extensions for generating methods on distributed vectors.
///
/// This provides convenience wrappers for calling methods on distributed vectors.
/// The `vector_type` is the distributed vector type (not the element type).
pub trait DistVectorExt {
    unsafe fn gen_new(&mut self,
                      builder: LLVMBuilderRef,
                      vector_type: &Type,
                      size: LLVMValueRef,
                      run: LLVMValueRef) -> WeldResult<LLVMValueRef>;
    unsafe fn gen_flatten(&mut self,
                          builder: LLVMBuilderRef,
                          dvec_type: &Type,
                          dvec: LLVMValueRef,
                          run: LLVMValueRef) -> WeldResult<LLVMValueRef>;

    /// This function returns a pointer to the ith *element* in the distributed vector,
    /// as opposed to the ith *vector*.
    ///
    /// The index is computed by considering the vectors in the order they are stored
    /// as though they are one contiguous vector. For example, if a distributed vector D
    /// is composed of one vector v1 of length 2 and v2 of length 3, then index D[3]
    /// resolves to v2[1].
    unsafe fn gen_at(&mut self,
                     builder: LLVMBuilderRef,
                     vector_type: &Type,
                     vec: LLVMValueRef,
                     index: LLVMValueRef) -> WeldResult<LLVMValueRef>;

    unsafe fn gen_size(&mut self,
                       builder: LLVMBuilderRef,
                       vector_type: &Type,
                       dvec: LLVMValueRef) -> WeldResult<LLVMValueRef>;

    unsafe fn gen_nvecs(&mut self,
                        builder: LLVMBuilderRef,
                        vector_type: &Type,
                        dvec: LLVMValueRef) -> WeldResult<LLVMValueRef>;
}

impl DistVectorExt for LlvmGenerator {
    unsafe fn gen_new(&mut self,
                      builder: LLVMBuilderRef,
                      vector_type: &Type,
                      size: LLVMValueRef,
                      run: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if let Type::DistVec(ref elem_type) = *vector_type {
            let mut methods = self.vectors.get_mut(elem_type).unwrap();
            methods.gen_new(builder, &mut self.intrinsics, run, size)
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_flatten(&mut self,
                        builder: LLVMBuilderRef,
                        vector_type: &Type,
                        vector: LLVMValueRef,
                        run: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if let Type::Vector(ref elem_type) = *vector_type {
            let mut methods = self.vectors.get_mut(elem_type).unwrap();
            methods.gen_flatten(builder, &mut self.intrinsics, vector, run)
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_at(&mut self,
                     builder: LLVMBuilderRef,
                     vector_type: &Type,
                     vector: LLVMValueRef,
                     index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if let Type::Vector(ref elem_type) = *vector_type {
            let mut methods = self.vectors.get_mut(elem_type).unwrap();
            methods.gen_at(builder, vector, index)
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_size(&mut self,
                       builder: LLVMBuilderRef,
                       vector_type: &Type,
                       vector: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if let Type::Vector(ref elem_type) = *vector_type {
            let mut methods = self.vectors.get_mut(elem_type).unwrap();
            methods.gen_size(builder, vector)
        } else {
            unreachable!()
        }
    }
}

/// A vector type and its associated methods.
pub struct DistVector {
    pub dvec_ty: LLVMTypeRef,
    pub name: String,
    pub vec_ty: LLVMTypeRef,
    pub elem_ty: LLVMTypeRef,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    new: Option<LLVMValueRef>,
    flatten: Option<LLVMValueRef>,
    at: Option<LLVMValueRef>,
    vec_at: Option<LLVMValueRef>,
    nvecs: Option<LLVMValueRef>,
    size: Option<LLVMValueRef>,
    slice: Option<LLVMValueRef>,
}

impl CodeGenExt for DistVector {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
}

impl DistVector {
    /// Define a new distributed vector type with the given element type.
    ///
    /// This function only inserts a definition for the vector, but does not generate any new code.
    pub unsafe fn define<T: AsRef<str>>(name: T,
                                        vec_ty: LLVMTypeRef, // the type of the constituent vec pointers
                                        elem_ty: LLVMTypeRef, 
                                        context: LLVMContextRef,
                                        module: LLVMModuleRef) -> DistVector {

        let c_name = CString::new(name.as_ref()).unwrap();

        // [vec_ty, nvecs]
        let mut layout = [LLVMPointerType(vec_ty, 0), LLVMInt64TypeInContext(context)];
        let dvec = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(dvec, layout.as_mut_ptr(), layout.len() as u32, 0);
        DistVector {
            name: c_name.into_string().unwrap(),
            context: context,
            module: module,
            dvec_ty: dvec,
            vec_ty: vec_ty,
            elem_ty: elem_ty,
            new: None,
            flatten: None,
            at: None,
            vec_at: None,
            nvecs: None,
            size: None,
            slice: None,
        }
    }

    /// Generates the `new` method on vectors and calls it.
    ///
    /// The new method allocates a buffer of size exactly `nvecs`
    /// and initializes it with the provided pointers (`vecs`).
    pub unsafe fn gen_new(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &mut Intrinsics,
                          run: LLVMValueRef,
                          vecs: Vec<LLVMValueRef>) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let mut arg_tys = [self.i64_type(), self.run_handle_type()];
            let ret_ty = self.vector_ty;

            let name = format!("{}.new", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let size = LLVMGetParam(function, 0);
            let elem_size = self.size_of(self.elem_ty);
            let alloc_size = LLVMBuildMul(builder, elem_size, size, c_str!("size"));
            let run = LLVMGetParam(function, 1);
            let bytes = intrinsics.call_weld_run_malloc(builder, run, alloc_size, Some(c_str!("bytes")));
            let elements = LLVMBuildBitCast(builder, bytes, LLVMPointerType(self.elem_ty, 0), c_str!("elements"));
            let mut result = LLVMGetUndef(self.vector_ty);

            let elem_size = self.size_of(self.vec_ty);
            let nelts = self.i32(vecs.len());
            let alloc_size = LLVMBuildMul(builder, elem_size, nelts, c_str!("size"));
            let run = LLVMGetParam(function, 1);
            let bytes = intrinsics.call_weld_run_malloc(builder, run, alloc_size, Some(c_str!("bytes")));
            let elements = LLVMBuildBitCast(builder, bytes, LLVMPointerType(self.vec_ty, 0), c_str!("elements"));

            // push all the elements
            for (i, vec) in vecs.iter().enumerate() {
                let insert_ptr = LLVMBuildStructGEP(ctx.builder,
                                                    elements,
                                                    i as u32,
                                                    c_str!(""));
                LLVMBuildStore(ctx.builder, vec.clone(), insert_ptr);
            }
            
            result = LLVMBuildInsertValue(builder, result, elements, POINTER_INDEX, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, nelts, NVECS_INDEX, c_str!(""));
            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }
        
        let mut args = [size, run];

        Ok(LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `flatten` method on vectors and calls it.
    ///
    /// The flatten method shallow-copies the vector elements into a local, contiguous vector.
    unsafe fn gen_flatten(&mut self,
                          builder: LLVMBuilderRef,
                          dvec_type: &Type,
                          dvec: LLVMValueRef,
                          run: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.flatten.is_none() {
            let mut arg_tys = [self.vector_ty, self.run_handle_type()];
            let ret_ty = self.vector_ty;

            let name = format!("{}.flatten", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let vector = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            let elem_size = self.size_of(self.elem_ty);
            let size = LLVMBuildExtractValue(builder, vector, SIZE_INDEX, c_str!("")); 
            let alloc_size = LLVMBuildMul(builder, elem_size, size, c_str!("size"));

            let dst_bytes = intrinsics.call_weld_run_malloc(builder, run, alloc_size, Some(c_str!("")));
            let source_bytes = LLVMBuildExtractValue(builder, vector, POINTER_INDEX, c_str!("")); 
            let source_bytes = LLVMBuildBitCast(builder, source_bytes, self.void_pointer_type(), c_str!(""));
            let _ = intrinsics.call_memcpy(builder, dst_bytes, source_bytes, alloc_size);

            let elements = LLVMBuildBitCast(builder, dst_bytes, LLVMPointerType(self.elem_ty, 0), c_str!(""));
            let result = LLVMBuildInsertValue(builder,
                                           LLVMGetUndef(self.vector_ty),
                                           elements, POINTER_INDEX, c_str!(""));
            let result = LLVMBuildInsertValue(builder, result, size, SIZE_INDEX, c_str!(""));
            LLVMBuildRet(builder, result);

            self.flatten = Some(function);
            LLVMDisposeBuilder(builder);
        }
        
        let mut args = [vector, run];
        return Ok(LLVMBuildCall(builder, self.flatten.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `at` method on distributed vectors and calls it.
    ///
    /// This function returns a pointer to the ith *element* in the distributed vector,
    /// as opposed to the ith *vector*.
    ///
    /// The index is computed by considering the vectors in the order they are stored
    /// as though they are one contiguous vector. For example, if a distributed vector D
    /// is composed of one vector v1 of length 2 and v2 of length 3, then index D[3]
    /// resolves to v2[1].
    ///
    /// TODO: index out of bounds check.
    pub unsafe fn gen_at(&mut self,
                         builder: LLVMBuilderRef,
                         dvec: LLVMValueRef,
                         index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        use self::vector::VectorExt;
        
        if self.at.is_none() {
            let mut arg_tys = [self.dvec_ty, self.i64_type()];
            let ret_ty = LLVMPointerType(self.elem_ty, 0);

            let name = format!("{}.at", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnFunction(self.context, function, &[AlwaysInline]);

            let dvec = LLVMGetParam(function, 0);
            let index = LLVMGetParam(function, 1);
            let pointer = LLVMBuildExtractValue(builder, dvec, POINTER_INDEX, c_str!(""));
            let value_pointer = LLVMBuildGEP(builder, pointer, [index].as_mut_ptr(), 1, c_str!(""));
            LLVMBuildRet(builder, value_pointer);

            let loop_block = LLVMAppendBasicBlockInContext(self.context, function,  c_str!(""));
            let ret_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
            let inc_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
            let done_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));

            let nvecs = self.gen_nvecs(builder, dvec);

            // pseudocode:
            // total = 0
            // for (i, vec) in vecs.enumerate():
            //    if (total + vec.size) > index:
            //        return vecs[i][index-total]
            //    total += vec.size
            
            // Index variable.
            let phi_i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
            let phi_total = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));

            // Get the ith vector's size.
            let vec_i = self.gen_vec_at(builder, dvec, phi_i);
            let size_i = VectorExt::gen_size(self, builder, vec_i);

            let sum_size = LLVMBuildNSWAdd(builder, phi_total, size_i, c_str!(""));
            let size_check = LLVMBuildICmp(builder, LLVMIntSGT, sum_size, index, c_str!(""));
            LLVMBuildCondBr(builder, size_check, ret_block, inc_block);

            LLVMPositionBuilderAtEnd(builder, ret_block);
            // total + vec.size > index -- return element
            let inner_idx = LLVMBuildNSWSub(builder, index, phi_total, c_str!(""));
            let elt_pointer = VectorExt::gen_at(self, builder, inner_idx);
            LLVMBuildRet(builder, elt_pointer);

            // update counters
            LLVMPositionBuilderAtEnd(builder, inc_block);
            let updated_i = LLVMBuildNSWAdd(builder, phi_i, self.i64(1), c_str!(""));
            let updated_total = LLVMBuildNSWAdd(builder, phi_total, size_i, c_str!(""));
            let bound_check = LLVMBuildICmp(builder, LLVMIntEQ, updated_i, nvecs, c_str!(""));
            LLVMBuildCondBr(builder, bound_check, done_block, loop_block);

            // control flow
            let mut blocks = [entry_block, loop_block];
            let mut values = [self.i64(0), updated_i];
            let mut total_values = [self.i64(0), updated_total];
            LLVMAddIncoming(phi_i, values.as_mut_ptr(),
                            blocks.as_mut_ptr(), values.len() as u32);
            LLVMAddIncoming(phi_total, total_values.as_mut_ptr(),
                            blocks.as_mut_ptr(), total_values.len() as u32);

            LLVMPositionBuilderAtEnd(builder, done_block);

            // Reached the end of the vector list, so index is out of range -- return null pointer.
            LLVMBuildRet(builder, self.void_pointer());

            self.at = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dvec, index];
        Ok(LLVMBuildCall(builder, self.at.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `vec_at` method on distributed vectors and calls it.
    ///
    /// This method returns a pointer to the constituent vector at the requested index.
    pub unsafe fn gen_vec_at(&mut self,
                             builder: LLVMBuilderRef,
                             dvec: LLVMValueRef,
                             index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.vec_at.is_none() {
            let mut arg_tys = [self.vector_ty, self.i64_type()];
            let ret_ty = LLVMPointerType(self.elem_ty, 0);

            let name = format!("{}.vec_at", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnFunction(self.context, function, &[AlwaysInline]);

            let dvec = LLVMGetParam(function, 0);
            let index = LLVMGetParam(function, 1);
            let pointer = LLVMBuildExtractValue(builder, dvec, POINTER_INDEX, c_str!(""));
            let value_pointer = LLVMBuildGEP(builder, pointer, [index].as_mut_ptr(), 1, c_str!(""));
            LLVMBuildRet(builder, value_pointer);

            self.vec_at = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dvec, index];
        Ok(LLVMBuildCall(builder, self.vec_at.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generate the `slice` method on distributed vectors and calls it.
    pub unsafe fn gen_slice(&mut self,
                            builder: LLVMBuilderRef,
                            vector: LLVMValueRef,
                            index: LLVMValueRef,
                            size: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        unimplemented!()
    }

    /// Generates the `size` method on distributed vectors and calls it.
    ///
    /// This returns the sum of the sizes (equivalently, the capacities) of the constituent vectors.
    pub unsafe fn gen_size(&mut self,
                           builder: LLVMBuilderRef,
                           dvec: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.size.is_none() {
            let mut arg_tys = [self.vector_ty];
            let ret_ty = self.i64_type();

            let name = format!("{}.size", self.name);
            let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnFunction(self.context, function, &[AlwaysInline]);

            let dvec = LLVMGetParam(function, 0);
            let nvecs = self.gen_nvecs(builder, dvec)?;
            
            //let loop_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
            let done_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));

            //LLVMPositionBuilderAtEnd(builder, loop_block);

            // Index variable.
            let phi_i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
            let vec_tuple = self.gen_at(builder, ty, dvec, phi_i)?;
            
            LLVMPositionBuilderAtEnd(builder, loop_block);
            let size = LLVMBuildExtractValue(builder, vector, SIZE_INDEX, c_str!(""));

            LLVMBuildRet(builder, size);

            self.size = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [vector];
        Ok(LLVMBuildCall(builder, self.size.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `nvecs` method on distributed vectors and calls it.
    ///
    /// This returns the number of constituent vectors.
    pub unsafe fn gen_nvecs(&mut self,
                            builder: LLVMBuilderRef,
                            dvec: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.nvecs.is_none() {
            let mut arg_tys = [self.dvec_ty];
            let ret_ty = self.i64_type();

            let name = format!("{}.nvecs", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnFunction(self.context, function, &[AlwaysInline]);

            let dvec = LLVMGetParam(function, 0);
            let nvecs = LLVMBuildExtractValue(builder, dvec, NVECS_INDEX, c_str!(""));
            LLVMBuildRet(builder, nvecs);

            self.nvecs = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dvec];
        Ok(LLVMBuildCall(builder, self.nvecs.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }
}
