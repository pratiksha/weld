//! Transform mergers on vec[T] so that they behave appropriately on sharded vec[vec[T]].

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use conf::ParsedConf;
use error::*;
use ast::constructors;

use optimizer::transforms::distribute::distribute::SHARDED_ANNOTATION;
use optimizer::transforms::distribute::code_util;

/// Merge a vector of shards into a vec[vec[T]]. In order to do this, the builder is modified to build
/// a vector of the original vectors.
pub fn gen_merge_appender(result_iter: &Iter, result_ty: Type,
                          builder: &Expr) -> WeldResult<Expr> {
    let new_kind: BuilderKind = if let Builder(ref bk, _) = builder.ty {
        match bk {
            BuilderKind::Appender(ref ty) => {
                BuilderKind::Appender(Box::new(Vector(ty.clone())))
            },
            _ => bk.clone()
        }
    } else {
        return compile_err!("Non-appender passed to appender merge in distribute\n");
    };

    // Change the builder to create a vec[vec[T]] if the builder was an Appender.
    let new_builder = constructors::newbuilder_expr(new_kind, None)?;

    // Create the loop.
    let params = code_util::new_loop_params(&new_builder.ty, &result_ty, builder);
    let merge_expr = code_util::simple_merge_expr(&params[0], &params[2]);
    let merge_func = constructors::lambda_expr(params, merge_expr).unwrap();
    let mut loop_expr = constructors::for_expr(vec![result_iter.clone()],
                                        new_builder,
                                        merge_func, false).unwrap();
    print!("loop type: {}\n", &loop_expr.ty);

    // Set the "sharded" annotations so we know that this vec[vec[T]] should be treated as a vec[T].
    loop_expr.annotations.set_bool(SHARDED_ANNOTATION, true);
        
    Ok(loop_expr)
}

/// Merge the results of distributed merger computations into a single result.
pub fn gen_merge_merger(result_iter: &Iter, result_ty: Type,
                        builder: &Expr) -> WeldResult<Expr> {
    // Create the loop.
    let params = code_util::new_loop_params(&builder.ty, &result_ty, builder);
    let merge_expr = code_util::simple_merge_expr(&params[0], &params[2]);
    let merge_func = constructors::lambda_expr(params, merge_expr).unwrap();
    
    let loop_expr = constructors::for_expr(vec![result_iter.clone()],
                                    builder.clone(),
                                    merge_func, false).unwrap();
    Ok(loop_expr)
}

/// Merge a vector of locally computed dictionaries.
/// The builder should be the same as the non-distributed builder.
/// The outer loop is over dictionaries and the inner loop is over (key, value) pairs within a dictionary.
/// The (key, value) pairs are merged into the outermost builder.
/// result_iter should be an iterator over the shards returned in the result.
/// TODO: this is currently incorrect
/* pub fn gen_merge_dicts(result_iter: &Iter, result_ty: Type,
                       builder: &Expr) -> WeldResult<Expr> {
    let inner_params = code_util::new_loop_params(&builder.ty, &result_ty, builder);
    let outer_params = code_util::new_loop_params(&builder.ty, &Vector(Box::new(result_ty.clone())), builder);

    // convert dictionary to a vector
    // lookup_expr used to get dictionary from pointer
    let dict_vec = constructors::tovec_expr(constructors::lookup_expr(
        constructors::ident_from_param(outer_params[2].clone()).unwrap(),
        constructors::zero_i64_literal().unwrap()).unwrap())?;

    // iterator over elements within each dict
    let element_iter = Iter { data: Box::new(dict_vec),
                              start: None, end: None, stride: None,
                              kind: IterKind::ScalarIter,
                              strides: None, shape: None };

    // merge using key, value
    let inner_merge = code_util::simple_merge_expr(&inner_params[0], &inner_params[2]);
    let inner_lambda = constructors::lambda_expr(inner_params, inner_merge)?;
    let inner_for = constructors::for_expr(vec![element_iter], // dictionary converted to vector
                                    constructors::ident_from_param(outer_params[0].clone()).unwrap(), // outer builder
                                    inner_lambda, true)?; // merge into outer builder
    let outer_lambda = constructors::lambda_expr(outer_params, inner_for)?;
    let outer_for = constructors::for_expr(vec![result_iter.clone()], builder.clone(), outer_lambda, false)?;
    Ok(outer_for)
} */

/// Merge a vector of locally computed vecmerger.
/// The builder should be the same as the non-distributed builder.
/// The outer loop is over vectors and the inner loop is over elements of the vector.
/// The elements are merged into the outermost builder by index.
/// result_iter should be an iterator over the shards returned in the result.
pub fn gen_merge_vecmerger(result_iter: &Iter, result_ty: Type,
                           builder: &Expr) -> WeldResult<Expr> {
    let inner_ty = if let Vector(ref elem_ty) = result_ty {
        elem_ty.clone()
    } else {
        return compile_err!("Found non-Vector in vecmerger");
    };
    
    let inner_params = code_util::new_loop_params(&builder.ty, &inner_ty, builder); // vec[f64]
    let outer_params = code_util::new_loop_params(&builder.ty, &result_ty.clone(), builder);

    // iterator over elements in each vector
    let element = constructors::ident_from_param(outer_params[2].clone())?;
    let element_iter = Iter { data: Box::new(element),
                              start: None, end: None, stride: None,
                              kind: IterKind::ScalarIter,
                              strides: None, shape: None };

    // merge using key, value
    let merge_idx = constructors::ident_from_param(inner_params[1].clone())?;
    let merge_elt = constructors::ident_from_param(inner_params[2].clone())?;
    let merge_struct = constructors::makestruct_expr(vec![merge_idx, merge_elt])?;
    let builder_ident = constructors::ident_from_param(inner_params[0].clone()).unwrap();

    let inner_merge = constructors::merge_expr(builder_ident, merge_struct)?;
    let inner_lambda = constructors::lambda_expr(inner_params, inner_merge)?;
    let inner_for = constructors::for_expr(vec![element_iter], 
                                           constructors::ident_from_param(outer_params[0].clone()).unwrap(), // outer vecmerger
                                    inner_lambda, true)?; // merge into outer builder
    let outer_lambda = constructors::lambda_expr(outer_params, inner_for)?;
    let outer_for = constructors::for_expr(vec![result_iter.clone()], builder.clone(), outer_lambda, false)?;
    Ok(outer_for)
}
