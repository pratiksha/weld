//! Transform a local expression into a distributed expression using RPC calls.
//! Serialize the distributed code to a string and replace with a UDF that calls the RPC.

use std::collections::HashSet;

use annotations::*;
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

const SHARD_SYM: &str = "shard_data";
const DISPATCH_SYM: &str = "dispatch";
const DISPATCH_ONE_SYM: &str = "dispatch_one";

/// Get names and types of all Idents accessed in this computation.
fn get_parameters(e: &Expr<Type>) -> HashSet<Parameter<Type>> {
    let mut syms: HashSet<Parameter<Type>> = HashSet::new();
    let mut defs: HashSet<Parameter<Type>> = HashSet::new(); // keep track of symbols that are *defined in* the subprogram -- these will not be arguments

    e.traverse(&mut |ref e| {
        if let Ident(ref sym) = e.kind {
            syms.insert(Parameter{name: (*sym).clone(), ty: e.ty.clone()});
        } else if let Let { ref name, .. } = e.kind {
            defs.insert(Parameter{name: (*name).clone(), ty: e.ty.clone()});
        } else if let Lambda { ref params, .. } = e.kind {
            for p in params.iter() {
                defs.insert(Parameter{name: (p.name).clone(), ty: p.ty.clone()});
            }
        }
    });

    syms.difference(&defs).cloned().collect() // return all elements that are accessed but not defined
}

/// Generate Weld Slices of input iters from {start, size} shard structs.
/// Create argument structs containing slices and copies of all other pointer args.
/// The pointer to each argument struct will be passed umodified through `dispatch` to the subprogram,
/// so the order of the arguments should be the same as the order in the subprogram parameters.
fn shard_to_args_func(shard_ty: Type,
                      iter_idents: Vec<Expr<Type>>,
                      is_sharded: bool,
                      pointer_idents: Vec<Expr<Type>>,
                      ctx: &Expr<Type>) -> WeldResult<(Expr<Type>, Vec<Parameter<Type>>, Type, Expr<Type>)> {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);
    
    let element_param = Parameter {
                          name: sym_gen.new_symbol("e"), /* elements are slice parameters */
                          ty: shard_ty.clone()
                      };
    let element_ident = exprs::ident_expr(element_param.name.clone(), element_param.ty.clone())?;

    let mut struct_vec = vec![];
    let mut input_params = vec![]; // record order of parameters to pass back to dispatch
    for name in iter_idents.iter() {
        if is_sharded {
            struct_vec.push(element_ident.clone());
        } else {
            let slice = exprs::slice_expr((*name).clone(), /* slice each input iter. */
                                          exprs::getfield_expr(element_ident.clone(), 0)?, /* slice start index */
                                          exprs::getfield_expr(element_ident.clone(), 1)?  /* slice size */)?;
            struct_vec.push(slice);
        }
        if let Ident(ref sym) = name.kind {
            input_params.push(Parameter{ name: sym.clone(), ty: name.ty.clone() });
        } else {
            return compile_err!("Non-ident iters not allowed in distribute");
        }
    }
    
    for name in pointer_idents.iter() {
        if iter_idents.contains(name) { /* already got this param from shard */
            continue;
        }
        
        struct_vec.push(name.clone());
        if let Ident(ref sym) = name.kind {
            input_params.push(Parameter{ name: sym.clone(), ty: name.ty.clone() });
        } else {
            return compile_err!("Non-ident args not allowed in distribute");
        }
    }
    

    let ret_struct = exprs::makestruct_expr(struct_vec).unwrap(); /* make a struct of all args */
    let ret_val = exprs::makevector_expr(vec![ret_struct]).unwrap(); /* make a vec to hold pointer */
    let builder = exprs::newbuilder_expr(BuilderKind::Appender(Box::new(ret_val.ty.clone())),
                                         None).unwrap();
    // Create new params for the loop.
    let params = vec![Parameter { name: sym_gen.new_symbol("b"),
                                  ty: builder.ty.clone(),
                      },
                      Parameter {
                          name: sym_gen.new_symbol("i"),
                          ty: Scalar(ScalarKind::I64),
                      },
                      element_param.clone()];

    let body = exprs::merge_expr(
        exprs::ident_from_param(params[0].clone()).unwrap(),
        ret_val.clone()).unwrap();

    let func = exprs::lambda_expr(params, body.clone())?;   /* function that generates a struct of the args */
    
    Ok((func, input_params, body.ty, builder.clone()))
}

/// Generate the UDF to call the dispatch function, which takes a vector of arguments as input.
fn generate_dispatch_func(subprog: &Expr<Type>,
                          body_ty: Type,
                          args_list: &Expr<Type>) -> WeldResult<Expr<Type>> {
    let code = exprs::literal_expr(LiteralKind::StringLiteral(print_typed_expr_without_indent(&subprog)))?;
    
    let dispatch_expr = if let Vector(_) = args_list.ty {
        exprs::cudf_expr(DISPATCH_SYM.to_string(),
                                             vec![code,
                                                  args_list.clone()],
                                             Vector(Box::new(Vector(Box::new(body_ty.clone()))))).unwrap()
    } else {
        return compile_err!("args of dispatch must be a materialized Vector");
    };
     
    Ok(dispatch_expr)
}

/// Generate the UDF to call the dispatch function, which takes a vector of arguments as input.
fn gen_dispatch_one(subprog: &Expr<Type>,
                    body_ty: Type,
                    index_expr: Expr<Type>,
                    args_expr: Expr<Type>) -> WeldResult<Expr<Type>> {
    let code = exprs::literal_expr(LiteralKind::StringLiteral(print_typed_expr_without_indent(&subprog)))?;

    /* Result returned is a struct of {worker ID, pointer to result data}. */
    let res_struct_ty = Struct(vec![Scalar(ScalarKind::I64),
                                    Vector(Box::new(body_ty.clone()))]);
    let dispatch_expr = exprs::cudf_expr(DISPATCH_ONE_SYM.to_string(),
                                         vec![code,
                                              index_expr.clone(), /* param referencing iteration idx */
                                              args_expr.clone()], /* param referencing arg */
                                         res_struct_ty).unwrap();
    Ok(dispatch_expr)
}

/// Traverse the expression and recursively set the 'sharded' annotation wherever the variable `name` is used.
/// This annotation indicates that a vector of vectors is a set of shards of a single vector rather than a normal nested vector.
/// TODO: currently assumes an entire sharded vector will not be passed as an argument to a Lambda.
fn set_sharded(expr: &mut Expr<Type>, name: String) {
    let mut names = vec![name];
    
    expr.traverse_mut(&mut |ref mut e| {
        if let Ident(ref sym) = e.kind {
            if names.contains(&sym.name) {
                e.annotations.set_sharded(true);
                names.push(sym.name.clone());
            }
        } else if let Let { ref name, ref mut value, .. } = e.kind {
            let mut set = false;
            if let Ident(ref sym) = value.kind {
                if names.contains(&sym.name) {
                    set = true;
                    names.push(sym.name.clone());
                }
            }

            if set {
                value.annotations.set_sharded(true);
            }
        }
    });
}

fn gen_zero_keyfunc(element_ty: Type, ctx: &Expr<Type>) -> WeldResult<Expr<Type>> {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);
    
    let param = Parameter {
        name: sym_gen.new_symbol("e"),
        ty: element_ty.clone()
    };
    let element_ident = exprs::ident_from_param(param.clone())?;
    let lookup = exprs::getfield_expr(element_ident.clone(), 0)?;
    let lookup_func = exprs::lambda_expr(vec![param], lookup)?;
    Ok(lookup_func)
}

/// Sort {key, value} pairs by key, and return only the values.
fn gen_sorted_values_by_key(vec: &Expr<Type>, ctx: &Expr<Type>) -> WeldResult<Expr<Type>> {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);

    let vec_ty = if let Vector(ref ty) = vec.ty {
        (**ty).clone()
    } else {
        return compile_err!("Cannot sort a non-Vector: {}\n", print_type(&vec.ty));
    };
    
    let elem_ty = if let Vector(ref ty) = vec.ty {
        if let Struct(ref tys) = **ty {
            if tys.len() < 2 { return compile_err!("Cannot strip keys from length 1 struct\n"); };
            tys[1].clone()
        } else {
            return compile_err!("Cannot strip keys from non-struct\n");
        }
    } else {
        return compile_err!("Cannot sort a non-Vector: {}\n", print_type(&vec.ty));
    };

    let sorted_results = exprs::sort_expr(vec.clone(),
                                          gen_zero_keyfunc(vec_ty.clone(), vec).unwrap())?;

    let appender = exprs::newbuilder_expr(BuilderKind::Appender(Box::new(elem_ty.clone())), None)?;
    let lookup_params = vec![Parameter { name: sym_gen.new_symbol("b"),
                                         ty: appender.ty.clone(),
    },
                             Parameter {
                                 name: sym_gen.new_symbol("i"),
                                 ty: Scalar(ScalarKind::I64),
                             },
                             Parameter {
                                 name: sym_gen.new_symbol("e"),
                                 ty: vec_ty.clone()
                             }];

    let b_ident = exprs::ident_from_param(lookup_params[0].clone()).unwrap();
    let e_ident = exprs::ident_from_param(lookup_params[2].clone()).unwrap();
    print!("generating merge here\n");
    let merge = exprs::merge_expr(b_ident,
                                  exprs::getfield_expr(e_ident, 1)?)?;
    let element_iter = Iter { data: Box::new(sorted_results),
                              start: None, end: None, stride: None,
                              kind: IterKind::ScalarIter,
                              strides: None, shape: None };
    Ok(exprs::for_expr(vec![element_iter], appender, merge, false).unwrap())
}

/// Merge a vector of shards into a vec[vec[T]]. In order to do this, the builder is modified to build
/// a vector of the original vectors.
pub fn gen_merge_appender(result_iter: &Iter<Type>, result_ty: Type,
                          builder: &Expr<Type>) -> WeldResult<Expr<Type>> {
    let mut sym_gen = SymbolGenerator::from_expression(builder);

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
    let new_builder = exprs::newbuilder_expr(new_kind, None)?;

    // Create new params for the loop.
    let params = vec![Parameter { name: sym_gen.new_symbol("b"),
                                  ty: new_builder.ty.clone(),
    },
                      Parameter {
                          name: sym_gen.new_symbol("i"),
                          ty: Scalar(ScalarKind::I64),
                      },
                      Parameter {
                          name: sym_gen.new_symbol("e"),
                          ty: Vector(Box::new(result_ty.clone())), // result of dispatch will be vec containing a vec of evaluations
                      }];

    let merge_expr = exprs::merge_expr(
        exprs::ident_from_param(params[0].clone()).unwrap(),
        exprs::lookup_expr(exprs::ident_from_param(params[2].clone()).unwrap(),
                           exprs::zero_i64_literal().unwrap())
            .unwrap()).unwrap();
    let merge_func = exprs::lambda_expr(params, merge_expr).unwrap();
    
    let mut loop_expr = exprs::for_expr(vec![result_iter.clone()],
                                    new_builder,
                                    merge_func, false).unwrap();

    // Set the "sharded" annotation so we know that this vec[vec[T]] should be treated as a vec[T].
    loop_expr.annotations.set_sharded(true);
        
    Ok(loop_expr)
}

/// Merge the results of distributed merger computations into a single result.
pub fn gen_merge_merger(result_iter: &Iter<Type>, result_ty: Type,
                        builder: &Expr<Type>) -> WeldResult<Expr<Type>> {
    let mut sym_gen = SymbolGenerator::from_expression(builder);

    // Create new params for the loop.
    let params = vec![Parameter { name: sym_gen.new_symbol("b"),
                                  ty: builder.ty.clone(),
    },
                      Parameter {
                          name: sym_gen.new_symbol("i"),
                          ty: Scalar(ScalarKind::I64),
                      },
                      Parameter {
                          name: sym_gen.new_symbol("e"),
                          ty: Vector(Box::new(result_ty.clone())), // result of dispatch will be vec containing a single merged evaluation
                      }];

    let merge_expr = exprs::merge_expr(
        exprs::ident_from_param(params[0].clone()).unwrap(),
        exprs::lookup_expr(exprs::ident_from_param(params[2].clone()).unwrap(),
                           exprs::zero_i64_literal().unwrap())
            .unwrap()).unwrap();
    let merge_func = exprs::lambda_expr(params, merge_expr).unwrap();
    
    let loop_expr = exprs::for_expr(vec![result_iter.clone()],
                                    builder.clone(),
                                    merge_func, false).unwrap();
    Ok(loop_expr)
}

/// Merge a vector of locally computed dictionaries. The builder should be the same as the non-distributed builder.
/// The outer loop is over dictionaries and the inner loop is over (key, value) pairs within a dictionary.
/// The (key, value) pairs are merged into the outermost builder.
/// result_iter should be an iterator over the shards returned in the result.
pub fn gen_merge_dicts(result_iter: &Iter<Type>, result_ty: Type,
                       builder: &Expr<Type>) -> WeldResult<Expr<Type>> {
    let mut sym_gen = SymbolGenerator::from_expression(builder);

    let inner_params = vec![Parameter { name: sym_gen.new_symbol("b2"),
                                        ty: builder.ty.clone(),
    },
                            Parameter {
                                name: sym_gen.new_symbol("i2"),
                                ty: Scalar(ScalarKind::I64),
                            },
                            Parameter {
                                name: sym_gen.new_symbol("e2"),
                                ty: Vector(Box::new(result_ty.clone())), // result of dispatch will be dict of single evaluation of subprogram
                            }];
    
    let outer_params = vec![Parameter { name: sym_gen.new_symbol("b"),
                                        ty: builder.ty.clone(),
    },
                            Parameter {
                                name: sym_gen.new_symbol("i"),
                                ty: Scalar(ScalarKind::I64),
                            },
                            Parameter {
                                name: sym_gen.new_symbol("e"),
                                ty: Vector(Box::new(result_ty.clone())), // result of dispatch will be vec containing a dict of evaluations
                            }];

    // convert dictionary to a vector
    // lookup_expr used to get dictionary from pointer
    let dict_vec = exprs::tovec_expr(exprs::lookup_expr(exprs::ident_from_param(outer_params[2].clone()).unwrap(),
                                                        exprs::zero_i64_literal().unwrap()).unwrap())?;

    // iterator over elements within each dict
    let element_iter = Iter { data: Box::new(dict_vec),
                              start: None, end: None, stride: None,
                              kind: IterKind::ScalarIter,
                              strides: None, shape: None };

    // merge using key, value
    let inner_merge = exprs::merge_expr(exprs::ident_from_param(inner_params[0].clone()).unwrap(),
                                        exprs::ident_from_param(inner_params[2].clone()).unwrap())?;
    let inner_lambda = exprs::lambda_expr(inner_params, inner_merge)?;
    let inner_for = exprs::for_expr(vec![element_iter], // dictionary converted to vector
                                    exprs::ident_from_param(outer_params[0].clone()).unwrap(), // outer builder
                                    inner_lambda, true)?; // merge into outer builder
    let outer_for = exprs::for_expr(vec![result_iter.clone()], builder.clone(), inner_for, false)?;
    Ok(outer_for)
}

// fn add_args(iter_idents: Vec<Expr<Type>>,
//             param_idents: Vec<Expr<Type>>,
//             ctx: &Expr<Type>) -> WeldResult<Option<Expr<Type>>> {
//     for name in param_idents.iter() {
//         if iter_idents.contains(name) { /* already got this param from shard */
//             continue;
//         }
        
//         struct_vec.push(name.clone());
//         if let Ident(ref sym) = name.kind {
//             input_params.push(Parameter{ name: sym.clone(), ty: name.ty.clone() });
//         } else {
//             return compile_err!("Non-ident args not allowed in distribute");
//         }
//     }
    
    
// }

/// Convert a For into a distributed For.
/// Generate UDF to dispatch RPCs for this function.
/// Shard data according to number of available workers.
/// Keep track of variables that aren't sharded, and pass to workers.
pub fn gen_distributed_loop(e: &Expr<Type>, nworkers_conf: &i32) -> WeldResult<Option<Expr<Type>>> {
    print!("in distribute: {}\n", print_expr(&e));
    if let For { ref iters, ref builder, ref func } = e.kind {
        let mut len_exprs   = vec![]; // lengths of each iter in For, for sharding. Should all be equal
        let mut iter_idents = vec![]; // Idents for each iter

        // keep track of whether the iters are sharded.
        // TODO: currently if one iter is sharded, they must all be sharded.
        let mut sharded = false;

        print!("getting iters\n");
        // get the vectors out that we want to shard
        for it in iters.iter() {
            print!("iter...\n");
            if let Vector(_) = (*it).data.ty { // for now, only works if all iters are Vectors and Idents
                if let Ident(_) = (*it).data.kind {
                    let len = exprs::length_expr(*it.data.clone())?;
                    len_exprs.push(len.clone());
                    iter_idents.push((*(*it).data).clone());
                    if (*it).data.annotations.sharded() {
                        sharded = true;
                    }
                } else {
                    print!("Not distributing: iters are not Idents\n");
                    return Ok(None);
                }
            } else { // abort
                print!("Not distributing: iters are not Vectors\n");
                return Ok(None);
            }
        }

        /* create args lists for dispatch */
        print!("generating args\n");

        /* Collect any remaining params that are not in the list of iters. */
        let params: Vec<Parameter<Type>> = get_parameters(e).into_iter().collect();
        let mut param_idents = vec![];
        for p in params.iter() {
            let param_ident = exprs::ident_expr(p.name.clone(), p.ty.clone())?;
            if iter_idents.contains(&param_ident) { // TODO does this work?
                continue;
            }
            param_idents.push(param_ident);
        }
        
        // each dispatch call requires the input vector as well as any auxiliary input arguments,
        // so we generate a loop over the shards that slices the input iters
        // and also adds the (remaining) input args in a struct
        // and then passes the whole thing to a dispatch call,
        // before calling dispatch and stitching the result back using a merge.
        print!("generating shards\n");
        let mut sym_gen = SymbolGenerator::from_expression(e);
        // shard returns a list of {start index, size} structs.
        let shard_elem_ty = Struct(vec![Scalar(ScalarKind::I64),
                                        Scalar(ScalarKind::I64)]);

        let nworkers = exprs::literal_expr(LiteralKind::I32Literal(nworkers_conf.clone())).unwrap();
        let increment = exprs::one_i64_literal().unwrap();

        let shard_expr = exprs::cudf_expr(SHARD_SYM.to_string(),
                                          vec![len_exprs[0].clone(),
                                               nworkers.clone(),
                                               increment.clone()],
                                          Vector(Box::new(shard_elem_ty.clone())))?;
        let shard_name = sym_gen.new_symbol("shard"); // don't copy shard data
        let shard_ident = exprs::ident_expr(shard_name.clone(), shard_expr.ty.clone())?;

        print!("generating slices\n");

        // function that generates slices of input iters from a shard, plus adds other required args
        let (slice_function, input_params, ret_ty, args_builder) = shard_to_args_func(shard_elem_ty.clone(),
                                                                                      iter_idents,
                                                                                      sharded,
                                                                                      param_idents,
                                                                                      e).unwrap();
        let shards_iter = Iter {
            data: Box::new(shard_ident),
            start: None, end: None, stride: None,
            kind: IterKind::ScalarIter,
            strides: None, shape: None
        };

        print!("generating args loop\n");
        /* convert all shards -> slices + args */
        let args_loop = exprs::for_expr(vec![shards_iter],
                                        args_builder.clone(),
                                        slice_function, false).unwrap();
        let shard_let = exprs::let_expr(shard_name, shard_expr, args_loop).unwrap();
        let args_res = exprs::result_expr(shard_let).unwrap(); /* materialize the entire list of nworkers args */

        let args_iter = Iter {
            data: Box::new(args_res),
            start: None, end: None, stride: None,
            kind: IterKind::ScalarIter,
            strides: None, shape: None
        };

        print!("generating subprog\n");
        /* create Lambda for subprogram */
        let subprog_body = exprs::result_expr(e.clone()).unwrap();
        let subprog = exprs::lambda_expr(input_params.clone(),
                                         subprog_body.clone()).unwrap(); // as long as iters are all Idents, we can just reuse the expr

        /* Create a loop to dispatch to all workers. */

        /* Build a vector of returned results. */
        let res_struct_ty = Struct(vec![Scalar(ScalarKind::I64),
                                        Vector(Box::new(subprog_body.ty.clone()))]);
        let dispatch_result_builder = exprs::newbuilder_expr(
            BuilderKind::Appender(Box::new(res_struct_ty.clone())), None)?;
        let dispatch_result_params = vec![Parameter { name: sym_gen.new_symbol("b"),
                                          ty: dispatch_result_builder.ty.clone(),
        },
                              Parameter {
                                  name: sym_gen.new_symbol("i"),
                                  ty: Scalar(ScalarKind::I64),
                              },
                              Parameter {
                                  name: sym_gen.new_symbol("e"),
                                  ty: ret_ty.clone(), // TODO is this right?
                              },
            ];

        print!("generating dispatch\n");
        let dispatch_func = gen_dispatch_one(&subprog,
                                             subprog_body.ty.clone(),
                                             exprs::ident_from_param(
                                                 dispatch_result_params[1].clone()).unwrap(),
                                             exprs::ident_from_param(
                                                 dispatch_result_params[2].clone()).unwrap()).unwrap();
        print!("generating dispatch 1\n");

        let dispatch_merge = exprs::merge_expr(exprs::ident_from_param(
            dispatch_result_params[0].clone()).unwrap(),
                                               dispatch_func)?;
        print!("generating dispatch 2\n");
        let dispatch_loop = exprs::for_expr(vec![args_iter], dispatch_result_builder,
                                            dispatch_merge, false)?;

        /* Sort result pointers by worker ID so there is a canonical ordering. */
        let sorted_results = exprs::result_expr(gen_sorted_values_by_key(
            &exprs::result_expr(dispatch_loop).unwrap(), e)?)?;
        
        print!("generating loop\n");
        /* Merge result of dispatch. */
        let result_iter = Iter { data: Box::new(sorted_results),
                                 start: None, end: None, stride: None,
                                 kind: IterKind::ScalarIter,
                                 strides: None, shape: None };

        let merge_loop: Expr<Type> = if let Builder(ref bk, _) = builder.ty {
            match bk {
                BuilderKind::Appender(..) => {
                    print!("merge appender\n");
                    gen_merge_appender(&result_iter, subprog_body.ty, builder).unwrap()
                },
                BuilderKind::Merger(..) => {
                    print!("merge merger\n");
                    gen_merge_merger(&result_iter, subprog_body.ty, builder).unwrap()
                },
                BuilderKind::DictMerger(..) => {
                    print!("merge dictionary\n");
                    gen_merge_dicts(&result_iter, subprog_body.ty, builder).unwrap()
                },
                _ => {
                    print!("Not distributing: builder kind not supported\n");
                    return Ok(None);
                }
            } 
        } else {
            return compile_err!("Non-builder found in for expression\n");
        };
        
        print!("returning loop... {}\n", print_expr(&merge_loop));

        return Ok(Some(merge_loop));
    } else { // abort
        print!("Not distributing: not a For\n");
        return Ok(None);
    }
}

/// Convert top-level For loops in this expr into distributed For loops.
/// Transform any vector operations on a distributed vector appropriately to be compatible with the new distributed input.
pub fn distribute(expr: &mut Expr<Type>, nworkers_conf: &i32) {
    //let transformed = vec![];

    expr.transform_and_continue_res(&mut |ref mut e| {
        if let For { ref iters, ref builder, ref func } = e.kind {
            let ret = gen_distributed_loop(e, nworkers_conf).unwrap();
            match ret {
                Some(e) => return Ok((Some(e), false)),
                None => return Ok((None, true))
            }
        }

        return Ok((None, true))
        // if let Lookup { ref data, ref index } = e.kind {
        //     if let Res { ref builder } = data.kind {
        //         let dist_builder = gen_distributed_loop(*builder, nworkers_conf);
        //     }
        // } else if let Res { ref builder } = e.kind {
        //     let dist_builder = gen_distributed_loop(*builder, nworkers_conf);
            
        // } else if let Let { ref name, ref value, ref body } = e.kind {
        //     if let Res { ref builder } = body.kind {
        //         let dist_builder = gen_distributed_loop(*builder, nworkers_conf);
        //         let annotated_name = name.clone();
        //         if let BuilderKind::Appender(_) = builder.kind {
        //             annotated_name.annotations.set_distributed(); // wherever this name is used, the loop should be over a vec[vec[T]]
        //         }
        //         let replace_let = exprs::let_expr(annotated_name, value.clone(), exprs::result_expr(dist_builder)?)?;
        //     }
        // }
    });
}

/// Parse and perform type inference on an expression.
#[cfg(test)]
fn typed_expr(code: &str) -> TypedExpr {
    let mut e = parse_expr(code).unwrap();
    assert!(infer_types(&mut e).is_ok());
    e.to_typed().unwrap()
}

#[test]
fn distribute_test() {
    //let code = "|z:i32, x:vec[i32], y:vec[i32]| result(for(zip(x, y), merger[i32, +], |b,i,e|merge(b, e.$0 + z)))";
    let code = "|x:vec[i32]| result(for(x, merger[i32, +], |b,i,e|merge(b, e)))";
    let mut e = typed_expr(code);
    distribute(&mut e, &1);
    print!("{}\n", print_typed_expr(&e));
}

#[test]
fn distribute_appender_test() {
    //let code = "|z:i32, x:vec[i32], y:vec[i32]| result(for(zip(x, y), merger[i32, +], |b,i,e|merge(b, e.$0 + z)))";
    let code = "|x:vec[i32]| result(for(x, appender[i32], |b,i,e|merge(b, e)))";
    let mut e = typed_expr(code);
    distribute(&mut e, &1);
    print!("{}\n", print_typed_expr(&e));
}