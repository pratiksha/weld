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
                      pointer_idents: Vec<Expr<Type>>,
                      ctx: &Expr<Type>) -> WeldResult<(Expr<Type>, Vec<Parameter<Type>>, Type, Expr<Type>)> {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);
    
    let element_param = Parameter {
                          name: sym_gen.new_symbol("e"), /* elements are indices */
                          ty: shard_ty.clone()
                      };
    let element_ident = exprs::ident_expr(element_param.name.clone(), element_param.ty.clone())?;

    let mut struct_vec = vec![];
    let mut input_params = vec![]; // record order of parameters to pass back to dispatch
    for name in iter_idents.iter() {
        let slice = exprs::slice_expr((*name).clone(), /* slice each input iter. */
                                      exprs::getfield_expr(element_ident.clone(), 0)?, /* slice start index */
                                      exprs::getfield_expr(element_ident.clone(), 1)?  /* slice size */)?;
        struct_vec.push(slice);
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

/// Generate UDF to dispatch RPCs for this function.
/// Shard data according to number of available workers.
/// Keep track of variables that aren't sharded, and pass to workers.
pub fn distribute(expr: &mut Expr<Type>, nworkers_conf: &i32) {
    let nworkers = exprs::literal_expr(LiteralKind::I32Literal(nworkers_conf.clone())).unwrap();
    let increment = exprs::literal_expr(LiteralKind::I64Literal(1)).unwrap();
    
    expr.transform_and_continue_res(&mut |ref mut e| {
        print!("in distribute: {}\n", print_expr(&e));
        if let For { ref iters, ref builder, ref func } = e.kind {
            let mut len_exprs = vec![]; // lengths of each iter in For, for sharding. Should all be equal
            let mut iter_idents = vec![]; // Idents for each iter

            print!("getting iters\n");
            // get the vectors out that we want to shard
            for it in iters.iter() {
                print!("iter...\n");
                if let Vector(_) = (*it).data.ty { // for now, only works if all iters are Vectors and Idents
                    if let Ident(_) = (*it).data.kind {
                        let len = exprs::length_expr(*it.data.clone())?;
                        len_exprs.push(len.clone());
                        iter_idents.push((*(*it).data).clone());
                        // types.push(*vec_ty.clone());
                    } else {
                        print!("Not distributing: iters are not Idents\n");
                        return Ok((None, true));
                    }
                } else { // abort
                    print!("Not distributing: iters are not Vectors\n");
                    return Ok((None, true));
                }
            }

            /* create args lists for dispatch */
            print!("generating args\n");
            let params: Vec<Parameter<Type>> = get_parameters(e).into_iter().collect();
            let mut param_idents = vec![];
            for p in params.iter() {
                param_idents.push(exprs::ident_expr(p.name.clone(), p.ty.clone())?);
            }

            print!("generating shards\n");
            let mut sym_gen = SymbolGenerator::from_expression(e);
            // shard returns a list of {start index, size} structs.
            let shard_elem_ty =  Struct(vec![Scalar(ScalarKind::I64),
                                             Scalar(ScalarKind::I64)]);
            let shard_expr = exprs::cudf_expr(SHARD_SYM.to_string(),
                                              vec![len_exprs[0].clone(),
                                                   nworkers.clone(),
                                                   increment.clone()],
                                              Vector(Box::new(shard_elem_ty.clone())))?;
            let shard_name = sym_gen.new_symbol("shard"); // don't copy shard data
            let shard_ident = exprs::ident_expr(shard_name.clone(), shard_expr.ty.clone())?;

            print!("generating slices\n");
            // function that generates slices of input iters from shards, plus adds other required args
            let (slice_function, input_params, ret_ty, args_builder) = shard_to_args_func(shard_elem_ty.clone(),
                                                                                          iter_idents,
                                                                                          param_idents,
                                                                                          e).unwrap();
            let shards_iter = Iter {
                data: Box::new(shard_ident),
                start: None, end: None, stride: None,
                kind: IterKind::ScalarIter,
                strides: None, shape: None
            };

            print!("generating args loop\n");
            /* convert all shards -> args */
            let args_loop = exprs::for_expr(vec![shards_iter],
                                            args_builder.clone(),
                                            slice_function, false).unwrap();
            let shard_let = exprs::let_expr(shard_name, shard_expr, args_loop).unwrap();
            let args_res = exprs::result_expr(shard_let).unwrap();
            
            print!("generating subprog\n");
            /* create Lambda for subprogram */
            let subprog_body = exprs::result_expr(e.clone()).unwrap();
            let subprog = exprs::lambda_expr(input_params.clone(),
                                             subprog_body.clone()).unwrap(); // as long as iters are all Idents, we can just reuse the expr
            print!("generating dispatch\n");
            let dispatch_func = generate_dispatch_func(&subprog,
                                                       subprog_body.ty.clone(),
                                                       &args_res).unwrap();

            print!("generating loop\n");
            /* merge result of dispatch */
            let result_iter = Iter { data: Box::new(dispatch_func),
                                     start: None,
                                     end: None,
                                     stride: None,
                                     kind: IterKind::ScalarIter,
                                     strides: None,
                                     shape: None };

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
                                  ty: Vector(Box::new(subprog_body.ty.clone())), // result of dispatch will be vec of single evaluation of subprogram
                              }];
            let merge_expr = exprs::merge_expr(
                exprs::ident_from_param(params[0].clone()).unwrap(),
                exprs::lookup_expr(exprs::ident_from_param(params[2].clone()).unwrap(),
                                   exprs::literal_expr(LiteralKind::I64Literal(0)).unwrap())
                    .unwrap()).unwrap();
            let merge_func = exprs::lambda_expr(params, merge_expr).unwrap();
            
            let loop_expr = exprs::for_expr(vec![result_iter],
                                            (**builder).clone(),
                                            merge_func, false).unwrap(); 
            print!("returning loop... {}\n", print_expr(&loop_expr));
            return Ok((Some(loop_expr), false));
        } else { // abort
            print!("Not distributing: not a For\n");
            return Ok((None, true));
        }
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