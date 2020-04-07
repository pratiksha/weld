//! Utility functions for moving between operations on vec[T] and vec[vec[T]].
//! Used in the `distribute` transform for operations on shards.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use conf::ParsedConf;
use error::*;
use ast::constructors;
use util::SymbolGenerator;

use optimizer::transforms::distribute::distribute;
use optimizer::transforms::distribute::distribute::get_sharded;
use optimizer::transforms::distribute::distribute::SHARDED_ANNOTATION;

const LOOKUP_SYM: &str = "lookup";
const SLICE_SYM: &str = "slice";

/// Transform a Lookup on a vec[T] into an equivalent Lookup on a vec[vec[T]].
/// Precondition: data field of Lookup is a sharded vector.
pub fn gen_distributed_lookup(expr: &Expr)
                        -> WeldResult<Expr> {
    if let Lookup { ref data, ref index } = expr.kind {
        let mut sym_gen = SymbolGenerator::from_expression(expr);
        
        let data_sym = sym_gen.new_symbol("data");
        let data_ident = constructors::ident_expr(data_sym.clone(), Vector(Box::new(data.ty.clone())))?;
        let index_sym = sym_gen.new_symbol("index");
        let index_ident = constructors::ident_expr(index_sym.clone(), index.ty.clone())?;
        
        /* call UDF to compute the new index */
        let new_idx_expr = constructors::cudf_expr(LOOKUP_SYM.to_string(),
                                                   vec![data_ident.clone(),
                                                        index_ident.clone()],
                                                   Struct(vec![Scalar(ScalarKind::I64),
                                                               Scalar(ScalarKind::I64)])).unwrap();
        let index_let = constructors::let_expr(index_sym.clone(), (**index).clone(),
                                               new_idx_expr.clone())?;
        
        let idx_sym = sym_gen.new_symbol("idx");
        let idx_ident = constructors::ident_expr(idx_sym.clone(), new_idx_expr.ty.clone())?;
        
        let shard_expr = constructors::lookup_expr(data_ident.clone(),
                                                   constructors::getfield_expr(idx_ident.clone(), 0)?)?;
        let elt_expr = constructors::lookup_expr(shard_expr,
                                                 constructors::getfield_expr(idx_ident.clone(), 1)?)?;
        
        /* don't compute index twice */
        let idx_let = constructors::let_expr(idx_sym.clone(), index_let, elt_expr)?;
        let data_let = constructors::let_expr(data_sym.clone(), (**data).clone(), idx_let)?;
        return Ok(data_let)
    }

    // else invalid or not sharded -- do nothing
    return Ok(expr.clone())
}

/*/// Transform a Slice on a vec[T] into an equivalent Slice returning vec[vec[T]].
pub fn transform_slice(expr: &mut Expr) -> WeldResult<Expr> {
    if let Slice { ref data, ref index, ref size } = expr.kind {
        if (&data).annotations.get_bool(SHARDED_ANNOTATION) {
            if let Vector(ref ty) = data.ty {
                if let Vector(ref ty) = **ty {
                    let mut sym_gen = SymbolGenerator::from_expression(expr);

                    let data_sym = sym_gen.new_symbol("data");
                    let data_ident = constructors::ident_expr(data_sym.clone(), data.ty.clone())?;
                    
                    /* call UDF to compute the new index */
                    let new_idx_expr = constructors::cudf_expr(SLICE_SYM.to_string(),
                                                               vec![data_ident.clone()],
                                                               Struct(vec![Scalar(ScalarKind::I64),
                                                                           Scalar(ScalarKind::I64)])).unwrap();
         
                }
            }
        }
    }
}*/

/// Wrap a vec[vec[T]] in a loop that flattens it into a vec[T].
/// Used when materializing a top-level Result.
pub fn flatten_vec(expr: &mut Expr) -> WeldResult<Expr> {
    let mut sym_gen = SymbolGenerator::from_expression(expr);

    if let Res { ref builder } = expr.kind {
        /* this Res will be a sharded vector when it is materialized */
        print!("type: {}\n", &expr.ty);
        if get_sharded(&expr) {
            if let Vector(ref shard_ty) = expr.ty {
                if let Vector(ref ty) = (**shard_ty) {
                    // iterate over shards
                    let outer_appender = constructors::newbuilder_expr(BuilderKind::Appender((*ty).clone()), None)?;
                    let outer_params = vec![Parameter { name: sym_gen.new_symbol("b2"),
                                                        ty: outer_appender.ty.clone(),
                    },
                                            Parameter {
                                                name: sym_gen.new_symbol("i2"),
                                                ty: Scalar(ScalarKind::I64),
                                            },
                                            Parameter {
                                                name: sym_gen.new_symbol("e2"),
                                                ty: (**shard_ty).clone()
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
                    

                    let shard_ident = constructors::ident_from_param(outer_params[2].clone())?;
                    let element_iter = Iter { data: Box::new(shard_ident.clone()),
                                              start: None, end: None, stride: None,
                                              kind: IterKind::ScalarIter,
                                              strides: None, shape: None };
                    
                    let inner_merge = constructors::merge_expr(constructors::ident_from_param(inner_params[0].clone()).unwrap(),
                                                        constructors::ident_from_param(inner_params[2].clone()).unwrap())?;
                    let inner_lambda = constructors::lambda_expr(inner_params, inner_merge)?;
                    println!("for 1");
                    let inner_for = constructors::for_expr(vec![element_iter],
                                                           constructors::ident_from_param(outer_params[0].clone()).unwrap(),
                                                           //inner_lambda, true)?; // TODO vectorization
                                                           inner_lambda, false)?;
                    let outer_lambda = constructors::lambda_expr(outer_params, inner_for)?;

                    let shard_iter = Iter { data: Box::new(expr.clone()),
                                            start: None, end: None, stride: None,
                                            kind: IterKind::ScalarIter,
                                            strides: None, shape: None };
                    println!("for 2");
                    let outer_for = constructors::for_expr(vec![shard_iter], outer_appender.clone(), outer_lambda, false)?;
                    return Ok(outer_for)
                }
            }
        }
     }

    return Ok(expr.clone())
}

pub fn flatten_toplevel_func(expr: &mut Expr) -> WeldResult<Expr> {
    let mut print_conf = PrettyPrintConfig::new();
    print_conf.show_types = true;
    print!("in flatten: {}\n", expr.pretty_print_config(&print_conf));
    if let Lambda { ref mut params, ref mut body } = expr.kind {
        println!("got lambda");
        if let Res { ref builder } = body.kind {
            println!("got res");
            let new_res = flatten_vec(&mut (**body).clone())?;
            return Ok(constructors::lambda_expr(params.clone(), new_res.clone())?);
        }
    }

    return Ok(expr.clone())
}
