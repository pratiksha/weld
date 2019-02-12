//! Transform a local expression into a distributed expression using RPC calls.
//! Serialize the distributed code to a string and replace with a UDF that calls the RPC.

use std::collections::HashSet;

use ast::*;
use ast::constructors;
use ast::ExprKind::*;
use ast::Type::*;
use conf::ParsedConf;
use error::*;
use fnv::FnvHashMap;
use util::SymbolGenerator;

use optimizer::transforms::distribute::code_util;
use optimizer::transforms::distribute::dispatch;
use optimizer::transforms::distribute::iters_to_idents::*;
use optimizer::transforms::distribute::mergers::*;
use optimizer::transforms::distribute::shard::*;
use optimizer::transforms::distribute::sort::*;
use optimizer::transforms::distribute::vec_vec_transforms::*;

pub const SHARDED_ANNOTATION: &str = "sharded"; // Marks non-contiguous data
pub const DISTRIBUTE_ANNOTATION: &str = "distribute"; // Marks outer loops that will be distributed

/// Information about vectors that will be inputs to subprograms.
pub struct vec_info {
    pub ident: Expr,
    pub original_ident: Expr,
    pub is_sharded: bool
}

/// Get names and types of all Idents accessed in this computation.
fn get_parameters(e: &Expr) -> HashSet<Parameter> {
    let mut syms: HashSet<Parameter> = HashSet::new();
    let mut defs: HashSet<Parameter> = HashSet::new(); // keep track of symbols that are *defined in* the subprogram -- these will not be arguments

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

/// Convert a For into a distributed For.
/// Generate UDF to dispatch RPCs for this function.
/// Shard data according to number of available workers.
/// Keep track of variables that aren't sharded, and pass to workers.
pub fn gen_distributed_loop(e: &Expr, nworkers_conf: &i32) -> WeldResult<Option<Expr>> {
    let mut print_conf = PrettyPrintConfig::new();
    print_conf.show_types = true;
    print!("in distribute: {}\n", e.pretty_print_config(&print_conf));

    if let For { ref iters, ref builder, ref func } = e.kind {
        print!("getting iters\n");
        let mut iter_data = vec![];
        let mut iter_params = vec![];
        let mut subprog_idents = vec![];
        
        let mut len_opt: Option<Expr> = None;
        for it in iters.iter() {
            if let Ident(ref sym) = (*it).data.kind {
                let is_sharded = (&(*it).data).annotations.get_bool(SHARDED_ANNOTATION);
                if is_sharded {
                    // already sharded. it type is a vec[vec[T]]
                    if let Vector(ref ty) = (*it).data.ty {
                        let (subprog_sym, subprog_ident) = code_util::new_sym_and_ident("new_iter", ty, &e);
                        iter_data.push(vec_info{ ident: subprog_ident.clone(),
                                                 original_ident: *(*it).data.clone(),
                                                 is_sharded: true
                        });

                        iter_params.push(Parameter{name: subprog_sym, ty: (**ty).clone()});
                        subprog_idents.push(subprog_ident.clone());
                    }
                } else {
                    // will need to create shards
                    len_opt = Some(constructors::length_expr(*(*it).data.clone()).unwrap());
                    iter_data.push(vec_info { ident: *(*it).data.clone(),
                                              original_ident: *(*it).data.clone(),
                                              is_sharded: false
                    });

                    iter_params.push(Parameter{name: (*sym).clone(), ty: (*it).data.ty.clone()});
                    subprog_idents.push(*(*it).data.clone());
                }
            } else {
                return compile_err!("non-Ident found in Iter");
            }
        }

        let dummy_len = constructors::one_i64_literal().unwrap();
        let len_expr = if let Some(ref expr) = len_opt {
            expr
        } else {
            &dummy_len
        };
        
        /* create Lambda for subprogram */
        print!("generating subprog\n");
        let mut subprog_iter_idents: Vec<Expr> = iter_data.iter().map(|ref x| x.ident.clone()).collect();
        let mut subprog_iters = subprog_idents.iter().map(|x| code_util::simple_iter(x.clone())).collect();

        /* TODO propagate vectorize */
        let subprog_body = constructors::result_expr(
            constructors::for_expr(subprog_iters, (**builder).clone(), (**func).clone(), false)?)?;

        /* create args lists for dispatch */
        print!("generating args\n");

        /* Collect any remaining params that are not in the list of iters. */
        let params: Vec<Parameter> = get_parameters(&subprog_body).into_iter().collect();
        let mut param_idents = vec![];
        for p in params.iter() {
            let param_ident = constructors::ident_expr(p.name.clone(), p.ty.clone())?;
            if iter_params.contains(&p) {
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
        let subprog = constructors::lambda_expr(input_params,
                                         subprog_body.clone()).unwrap();

        /* Create a loop to dispatch to all workers. 
         * Result of the dispatch loop is a vec of structs of {worker ID, pointer to result data}. */
        let dispatch_loop = dispatch::gen_dispatch_loop(args_iter, &subprog, &subprog_body.ty, e).unwrap();

        print!("generating sort\n");
        /* Sort result pointers by worker ID so there is a canonical ordering. */
        let sorted_results = constructors::result_expr(gen_sorted_values_by_key(
            &constructors::result_expr(dispatch_loop).unwrap(), e)?)?;
        print!("sorted type: {}\n", &sorted_results.ty);
        
        print!("generating loop\n");
        /* Finally, merge result of dispatch. */
        let result_iter = code_util::simple_iter(sorted_results);

        let merge_loop: Expr = if let Builder(ref bk, _) = builder.ty {
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
        
        print!("returning loop... {}\n", merge_loop.pretty_print());

        return Ok(Some(merge_loop));
    } else { // abort
        print!("Not distributing: not a For\n");
        return Ok(None);
    }
}

pub fn annotate_distribute(e: &Expr) -> Expr {
    let mut new_expr = e.clone();
    new_expr.annotations.set_bool(DISTRIBUTE_ANNOTATION);
    new_expr
}

/// Returns true if any subexpression has the `distribute` annotation.
pub fn contains_distribute(expr: &Expr) -> bool {
    let mut ret = false;
    expr.traverse(&mut |ref e| {
        if e.annotations.get_bool(DISTRIBUTE_ANNOTATION) {
            ret = true;
        }
    });
    ret
}

/// Returns `true` iff `expr` only accesses vectors in `computed`.
pub fn check_idents(expr: &Expr, computed: &FnvHashMap<Symbol, bool>) -> bool {
    let mut ret = true;
    expr.traverse(&mut |ref e| {
        if let Ident(ref sym) = e.kind {
            match e.ty {
                Vector(_) => {
                    if computed.get(&sym) == None {
                        println!("could not find sym: {}", sym.name());
                        ret = false;
                    }
                },
                _ => {}
            }
        }
    });
    ret
}

/// Distribute any top-level loops that access only Idents in `computed`.
/// Return the names of any idents that are computed by the new distributed loops
/// (i.e., vectors that may now be sharded).
pub fn distribute_transform(expr: &mut Expr, computed: &mut FnvHashMap<Symbol, bool>, nworkers_conf: &i32) {
    expr.transform_and_continue(&mut |ref mut e| {
        if e.annotations.get_bool(DISTRIBUTE_ANNOTATION) {
            if let For { ref iters, ref builder, ref func } = e.kind {
                let mut all_computed = true;
                for iter in iters.iter() {
                    all_computed &= check_idents(&(*iter.data), computed);
                }
                all_computed &= check_idents(builder, computed);
                all_computed &= check_idents(func, computed);

                if !all_computed {
                    return (None, true)
                }

                /* Check that iters are all Idents. If not, wrap in an Ident before distributing. */
                let (ident_loop, new_symbols) = iters_to_idents(e).unwrap();
                let mut dist_loop = gen_distributed_loop(&ident_loop, nworkers_conf).unwrap().unwrap();
                for (sym, value) in new_symbols.iter() {
                    dist_loop = constructors::let_expr((*sym).clone(),
                                                       (*value).clone(), dist_loop).unwrap();
                }
                return (Some(dist_loop), false);
            }
        }

        (None, true)
    });
}

/// Propagate the type effects of sharding from children upwards to parents.
fn propagate_sharded(e: &mut Expr, computed: &mut FnvHashMap<Symbol, bool>) -> Option<Expr> {
    if let Lookup { ref data, ref index } = e.kind {
        if let Res { ref builder } = data.kind {
            let mut new_e = e.clone();
            new_e.apply_bool_annotation(&builder, SHARDED_ANNOTATION);
            return Some(new_e)
        }
    } else if let Res { ref builder } = e.kind {
        /* The annotation has to be updated, and the builder will also have a new type. */
        let is_sharded = (&builder).annotations.get_bool(SHARDED_ANNOTATION);
        if is_sharded {
            let mut new_e = constructors::result_expr((**builder).clone()).unwrap(); /* update the type */
            new_e.apply_bool_annotation(&builder, SHARDED_ANNOTATION);
            return Some(new_e);
        } 
    } else if let Let { ref name, ref value, ref body } = e.kind {
        if let Res { ref builder } = value.kind {
            println!("got res");
            if let For { ref iters, ref builder, ref func } = builder.kind {
                println!("got for");
                if let Builder(ref bk, _) = builder.ty {
                    print!("got builder\n");
                    if let BuilderKind::Appender(_) = bk {
                        print!("got appender\n");
                        let mut replace_let = e.clone();
                        replace_let.apply_bool_annotation(&builder, SHARDED_ANNOTATION);
                        computed.insert(name.clone(), true);
                        return Some(replace_let)
                    }
                }
            }
        }
    }

    None
}

/// Convert top-level For loops in this expr into distributed For loops.
/// Transform any vector operations on a distributed vector appropriately to be compatible with the new distributed input.
/// TODO iters_to_idents.
pub fn distribute(expr: &mut Expr, nworkers_conf: &i32) {
    /* First annotate all top-level For loops to be distributed. */
    expr.transform_and_continue(&mut |ref mut e| {
        if let For { ref iters, ref builder, ref func } = e.kind {
            return (Some(annotate_distribute(&e)), false);
        }

        return (None, true)
    });
    
    /* Start by distributing loops only over Idents that are not themselves
       the product of distributed loops, because those might become sharded vectors.
       After each pass, we can add any Idents whose type (vec[T] or vec[vec[T]] or otherwise) we now know,
       and distribute any loops involving those inputs in the next pass.
       Iterate until no top-level distribute loops remain. */
    let mut computed = FnvHashMap::default();
    if let Lambda { ref params, ref body } = expr.kind {
        for p in params.iter() {
            computed.insert(p.name.clone(), false); // we can assume inputs are contiguous in memory
        }
    }

    while contains_distribute(&expr) {
        /* Distribute as many loops as we can. */
        let mut print_conf = PrettyPrintConfig::new();
        print_conf.show_types = true;
        println!("distributing... {}", expr.pretty_print_config(&print_conf));
        distribute_transform(expr, &mut computed, nworkers_conf);

        /* Now some Idents might be sharded. Propagate the sharded annotation upwards... */
        println!("propagate up...");
        expr.transform_up(&mut |ref mut e| {
            propagate_sharded(e, &mut computed)
        });

        println!("propagate down...");
        /* ... and then downwards to children. */
        expr.transform_and_continue(&mut |ref mut e| {
            if let Ident(ref sym) = e.kind {
                let annotation_set = e.annotations.get_bool(SHARDED_ANNOTATION);
                let is_sharded = match computed.get(&sym) {
                    Some(value) => { *value },
                    None => false
                };
                if !annotation_set && is_sharded {
                    let mut annotated = e.clone();
                    annotated.ty = Vector(Box::new(e.ty.clone()));
                    annotated.annotations.set_bool(SHARDED_ANNOTATION);
                    return (Some(annotated), true);
                }
            }

            (None, true)
        });
    }

    println!("done, flatten....");
    /* Finally, flatten the topmost result that will be returned to the calling program. */
    expr.transform_once(&mut |ref mut e| Some(flatten_toplevel_func(e).unwrap()));
}
