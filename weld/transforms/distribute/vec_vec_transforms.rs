//! Utility functions for moving between operations on vec[T] and vec[vec[T]].
//! Used in the `distribute` transform for operations on shards.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use conf::ParsedConf;
use error::*;
use exprs;
use pretty_print::*;
use util::SymbolGenerator;

#[cfg(test)]
use parser::*;
#[cfg(test)]
use type_inference::*;

const LOOKUP_SYM: &str = "lookup_index";

/// Transform a Lookup on a vec[T] into an equivalent Lookup on a vec[vec[T]].
pub fn transform_lookup(expr: &mut Expr<Type>) -> WeldResult<Expr<Type>> {
    if let Lookup { ref data, ref index } = expr.kind {
        if data.annotations.sharded() {
            if let Vector(ref ty) = data.ty {
                if let Vector(ref ty) = **ty {
                    let mut sym_gen = SymbolGenerator::from_expression(expr);

                    let data_sym = sym_gen.new_symbol("data");
                    let data_ident = exprs::ident_expr(data_sym.clone(), data.ty.clone())?;
                    
                    /* call UDF to compute the new index */
                    let new_idx_expr = exprs::cudf_expr(LOOKUP_SYM.to_string(),
                                                        vec![data_ident.clone()],
                                                        Struct(vec![Scalar(ScalarKind::I64),
                                                                    Scalar(ScalarKind::I64)])).unwrap();
                    let idx_sym = sym_gen.new_symbol("idx");
                    let idx_ident = exprs::ident_expr(idx_sym.clone(), new_idx_expr.ty.clone())?;
                    
                    let shard_expr = exprs::lookup_expr(data_ident.clone(),
                                                        exprs::getfield_expr(idx_ident.clone(), 0)?)?;
                    let elt_expr = exprs::lookup_expr(shard_expr,
                                                      exprs::getfield_expr(idx_ident.clone(), 1)?)?;

                    /* don't compute index twice */
                    let idx_let = exprs::let_expr(idx_sym.clone(), new_idx_expr, elt_expr)?;
                    let data_let = exprs::let_expr(data_sym.clone(), (**data).clone(), idx_let)?;
                    return Ok(data_let)
                }
            }
        }
    }

    // else invalid or not sharded -- do nothing
    return Ok(expr.clone())
}

/// Wrap a vec[vec[T]] in a loop that flattens it into a vec[T].
/// Used when materializing a top-level Result.
pub fn flatten_vec(expr: &mut Expr<Type>) -> WeldResult<Expr<Type>> {
    let mut sym_gen = SymbolGenerator::from_expression(expr);

    if let Res { ref builder } = expr.kind {
        /* this Res will be a sharded vector when it is materialized */
        print!("type: {}\n", print_type(&expr.ty));
        if expr.annotations.sharded() {
            if let Vector(ref ty) = expr.ty {
                if let Vector(ref ty) = (**ty) {
                    // iterate over shards
                    let outer_appender = exprs::newbuilder_expr(BuilderKind::Appender((*ty).clone()), None)?;
                    let outer_params = vec![Parameter { name: sym_gen.new_symbol("b2"),
                                                        ty: outer_appender.ty.clone(),
                    },
                                            Parameter {
                                                name: sym_gen.new_symbol("i2"),
                                                ty: Scalar(ScalarKind::I64),
                                            },
                                            Parameter {
                                                name: sym_gen.new_symbol("e2"),
                                                ty: (**ty).clone()
                                            }];
                    
                    
                    // iterate over elements
                    let inner_params = vec![Parameter { name: sym_gen.new_symbol("b1"),
                                                        ty: outer_appender.ty.clone(),
                    },
                                          Parameter {
                                              name: sym_gen.new_symbol("i1"),
                                              ty: Scalar(ScalarKind::I64),
                                          },
                                          Parameter {
                                              name: sym_gen.new_symbol("e1"),
                                              ty: (**ty).clone()
                                          }];
                    

                    let shard_ident = exprs::ident_from_param(outer_params[1].clone())?;
                    let element_iter = Iter { data: Box::new(shard_ident.clone()),
                                              start: None, end: None, stride: None,
                                              kind: IterKind::ScalarIter,
                                              strides: None, shape: None };
                    
                    let inner_merge = exprs::merge_expr(exprs::ident_from_param(inner_params[0].clone()).unwrap(),
                                                        exprs::ident_from_param(inner_params[2].clone()).unwrap())?;
                    let inner_lambda = exprs::lambda_expr(inner_params, inner_merge)?;
                    let inner_for = exprs::for_expr(vec![element_iter],
                                                    exprs::ident_from_param(outer_params[0].clone()).unwrap(),
                                                    inner_lambda, true)?;
                    let outer_lambda = exprs::lambda_expr(outer_params, inner_for)?;

                    let shard_iter = Iter { data: Box::new(expr.clone()),
                                            start: None, end: None, stride: None,
                                            kind: IterKind::ScalarIter,
                                            strides: None, shape: None };
                    let outer_for = exprs::for_expr(vec![shard_iter], outer_appender.clone(), outer_lambda, false)?;
                    return Ok(outer_for)
                }
            }
        }
    }

    return Ok(expr.clone())
}

pub fn flatten_toplevel_func(expr: &mut Expr<Type>) -> WeldResult<Expr<Type>> {
    print!("in flatten: {}\n", print_typed_expr(&expr));
    if let Lambda { ref mut params, ref mut body } = expr.kind {
        if let Res { ref builder } = body.kind {
            let new_res = flatten_vec(&mut (**body).clone())?;
            return Ok(exprs::lambda_expr(params.clone(), new_res.clone())?);
        }
    }

    return Ok(expr.clone())
}