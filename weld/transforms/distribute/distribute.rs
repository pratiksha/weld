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

use transforms::distribute::code_util;
use transforms::distribute::dispatch;
use transforms::distribute::iters_to_idents::*;
use transforms::distribute::mergers::*;
use transforms::distribute::shard::*;
use transforms::distribute::sort::*;
use transforms::distribute::vec_vec_transforms::*;

#[cfg(test)]
use parser::*;
#[cfg(test)]
use type_inference::*;

/* Names of required C UDFs, implemented in Clamor. */
const SHARD_SYM: &str = "shard_data";
const DISPATCH_SYM: &str = "dispatch";
const DISPATCH_ONE_SYM: &str = "dispatch_one";

/// Information about vectors that will be inputs to subprograms.
pub struct vec_info {
    pub ident: Expr<Type>,
    pub is_sharded: bool
}

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

/// Convert a For into a distributed For.
/// Generate UDF to dispatch RPCs for this function.
/// Shard data according to number of available workers.
/// Keep track of variables that aren't sharded, and pass to workers.
pub fn gen_distributed_loop(e: &Expr<Type>, nworkers_conf: &i32) -> WeldResult<Option<Expr<Type>>> {
    print!("in distribute: {}\n", print_typed_expr(&e));
    if let For { ref iters, ref builder, ref func } = e.kind {
        print!("getting iters\n");
        let mut iter_data = vec![];
        let mut len_opt: Option<Expr<Type>> = None;
        for it in iters.iter() {
            if let Ident(_) = (*it).data.kind {
                let is_sharded = (*it).data.annotations.sharded().clone();
                if is_sharded {
                    // already sharded. it type is a vec[vec[T]]
                    if let Vector(ref ty) = (*it).data.ty {
                        iter_data.push(vec_info{ ident: *(*it).data.clone(),
                                                 is_sharded: true
                        });
                    }
                } else {
                    // will need to create shards
                    // create new symbols for subprogram input parameters
                    len_opt = Some(exprs::length_expr(*(*it).data.clone()).unwrap());
                    iter_data.push(vec_info { ident: *(*it).data.clone(),
                                              is_sharded: false
                    });
                }
            } else {
                return compile_err!("non-Ident found in Iter");
            }
        }

        let dummy_len = exprs::literal_expr(LiteralKind::I32Literal(nworkers_conf.clone())).unwrap();
        let len_expr = if let Some(ref expr) = len_opt {
            expr
        } else {
            &dummy_len
        };
        
        /* create Lambda for subprogram */
        print!("generating subprog\n");
        let mut subprog_iter_idents: Vec<Expr<Type>> = iter_data.iter().map(|ref x| x.ident.clone()).collect();
        let mut subprog_iters = subprog_iter_idents.iter().map(|x| code_util::simple_iter(x.clone())).collect();

        /* TODO propagate vectorize */
        let subprog_body = exprs::result_expr(
            exprs::for_expr(subprog_iters, (**builder).clone(), (**func).clone(), false)?)?;

        /* create args lists for dispatch */
        print!("generating args\n");

        /* Collect any remaining params that are not in the list of iters. */
        let params: Vec<Parameter<Type>> = get_parameters(&subprog_body).into_iter().collect();
        let mut param_idents = vec![];
        for p in params.iter() {
            let param_ident = exprs::ident_expr(p.name.clone(), p.ty.clone())?;
            if subprog_iter_idents.contains(&param_ident) {
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
        let (args_res, input_params) = dispatch::gen_args_loop(&iter_data,
                                                               len_expr,
                                                               &param_idents,
                                                               nworkers_conf,
                                                               e).unwrap();
        let args_iter = code_util::simple_iter(args_res);
        let subprog = exprs::lambda_expr(input_params,
                                         subprog_body.clone()).unwrap();

        /* Create a loop to dispatch to all workers. 
         * Result of the dispatch loop is a vec of structs of {worker ID, pointer to result data}. */
        let dispatch_loop = dispatch::gen_dispatch_loop(args_iter, &subprog, &subprog_body.ty, e).unwrap();

        print!("generating sort\n");
        /* Sort result pointers by worker ID so there is a canonical ordering. */
        let sorted_results = exprs::result_expr(gen_sorted_values_by_key(
            &exprs::result_expr(dispatch_loop).unwrap(), e)?)?;
        print!("sorted type: {}\n", print_type(&sorted_results.ty));
        
        print!("generating loop\n");
        /* Finally, merge result of dispatch. */
        let result_iter = code_util::simple_iter(sorted_results);

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
        
        print!("returning loop... {}\n", print_typed_expr(&merge_loop));

        return Ok(Some(merge_loop));
    } else { // abort
        print!("Not distributing: not a For\n");
        return Ok(None);
    }
}

/// Convert top-level For loops in this expr into distributed For loops.
/// Transform any vector operations on a distributed vector appropriately to be compatible with the new distributed input.
pub fn distribute(expr: &mut Expr<Type>, nworkers_conf: &i32) {
    /* First transform-up any loops that need to be sharded. */
    expr.transform_up(&mut |ref mut e| {
        print!("transform up: {}\n", print_typed_expr_without_indent(&e));
        if let For { ref iters, ref builder, ref func } = e.kind {
            /* Check that iters are all Idents. If not, wrap in an Ident before distributing. */
            let (ident_loop, new_symbols) = iters_to_idents(e).unwrap();
            let mut dist_loop = gen_distributed_loop(&ident_loop, nworkers_conf).unwrap().unwrap();
            for (sym, value) in new_symbols.iter() {
                dist_loop = exprs::let_expr((*sym).clone(), (*value).clone(), dist_loop).unwrap();
            }
            return Some(dist_loop);
        } else if let Lookup { ref data, ref index } = e.kind {
            print!("got lookup\n");
            if let Res { ref builder } = data.kind {
                print!("got res\n");
                if builder.annotations.sharded() {
                    let mut new_e = e.clone();
                    new_e.annotations.set_sharded(true);
                    return Some(new_e);
                }
            }
        } else if let Res { ref builder } = e.kind {
            print!("got res\n");
            if builder.annotations.sharded() {
                print!("got sharded\n");
                let mut new_e = exprs::result_expr((**builder).clone()).unwrap(); /* update the type */
                new_e.annotations.set_sharded(true);
                print!("result type: {}\n", print_type(&new_e.ty));
                print!("result builder type: {}\n", print_type(&builder.ty));
                return Some(new_e);
            } 
        } else if let Let { ref name, ref value, ref body } = e.kind {
            print!("got let\n");
            if let Res { ref builder } = body.kind {
                print!("got res\n");
                if let For { ref iters, ref builder, ref func } = builder.kind {
                    print!("got for\n");
                    if builder.annotations.sharded() {
                        print!("sharded\n");
                        if let Builder(ref bk, _) = builder.ty {
                            print!("got builder\n");
                            if let BuilderKind::Appender(_) = bk {
                                print!("got appender\n");
                                let mut replace_let = e.clone(); // TODO sharded should be propagated to Idents using the name as well
                                replace_let.annotations.set_sharded(true);
                                return Some(replace_let);
                            }
                        }
                    }
                }
            }
        }

        return None;
    });

    /* Finally, flatten the topmost result that will be returned to the calling program. */
    expr.transform_once(&mut |ref mut e| Some(flatten_toplevel_func(e).unwrap()));
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