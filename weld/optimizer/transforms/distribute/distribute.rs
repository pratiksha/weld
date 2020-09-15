//! Transform a local expression into a distributed expression using RPC calls.
//! Serialize the distributed code to a string and replace with a UDF that calls the RPC.

use std::collections::HashSet;

use ast::*;
use ast::constructors;
use ast::ExprKind::*;
use ast::BuilderKind::*;
use ast::IterKind::*;
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

/// The known state of an Ident that is derived from a distributed loop.
/// Note that we don't use Option<bool> because this is used as a value in a hashmap,
/// which would also return None (instead of unset) if the value was not present.
#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub enum IdentState {
    Unset,
    Sharded,
    Contiguous
}

/// Information about vectors that will be inputs to subprograms.
pub struct vec_info {
    pub ident: Expr,
    pub original_ident: Expr,
    pub is_sharded: bool
}

/// Get names and types of all Idents accessed in this computation.
fn get_parameters(e: &Expr) -> HashSet<Parameter> {
    let mut syms: HashSet<Symbol> = HashSet::new();
    let mut defs: HashSet<Symbol> = HashSet::new(); // keep track of symbols that are defined in the subprogram itself
    let mut types = FnvHashMap::default();
    
    e.traverse(&mut |ref e| {
        if let Ident(ref sym) = e.kind {
            syms.insert((*sym).clone());
            types.insert((*sym).clone(), e.ty.clone());
        } else if let Let { ref name, .. } = e.kind {
            defs.insert((*name).clone());
        } else if let Lambda { ref params, .. } = e.kind {
            for p in params.iter() {
                defs.insert((p.name).clone());
            }
        }
    });

    let final_syms: HashSet<Symbol> = syms.difference(&defs).cloned().collect(); // all elements that are accessed but not defined
    let mut typed_syms: HashSet<Parameter> = HashSet::new();
    for sym in final_syms.iter() {
        typed_syms.insert(Parameter { name: (*sym).clone(), ty: (*types.get(&sym).unwrap()).clone() });
    }
 
    typed_syms
}

/// Convert a For into a distributed For.
/// Generate UDF to dispatch RPCs for this function.
/// Shard data according to number of available workers.
/// Keep track of variables that aren't sharded, and pass to workers.
pub fn gen_distributed_loop(e: &mut Expr,
                            ident_states: &mut FnvHashMap<Symbol, IdentState>,
                            partitions_conf: &i32) -> WeldResult<Option<Expr>> {
    let mut print_conf = PrettyPrintConfig::new();
    print_conf.show_types = true;
    //print!("in distribute: {}\n", e.pretty_print_config(&print_conf));

    if let For { ref iters, ref builder, ref func } = e.kind {
        //print!("getting iters\n");
        let mut iter_data = vec![];
        let mut iter_params = vec![];
        let mut subprog_idents = vec![];
        
        let mut len_opt: Option<Expr> = None;
        for (i, it) in iters.iter().enumerate() {
            for it in iters.iter() {
                if it.kind != ScalarIter {
                    return compile_err!("Got non-ScalarIter in distribute");
                }
            }
            
            if let Ident(ref sym) = (*it).data.kind {
                let is_sharded = ident_states.get(&sym);
                
                if is_sharded == Some(&IdentState::Sharded) {
                    // already sharded. it type is a vec[vec[T]]
                    if let Vector(ref ty) = (*it).data.ty {
                        let (subprog_sym, subprog_ident) =
                            code_util::new_sym_and_ident(format!("new_iter__{}", i).as_ref(), &(*it).data.ty.clone(), &e);
                        iter_data.push(vec_info { ident: subprog_ident.clone(),
                                                  original_ident: *(*it).data.clone(),
                                                  is_sharded: true
                        });

                        iter_params.push(Parameter { name: subprog_sym, ty: (**ty).clone() });
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
        //print!("generating subprog\n");
        let mut subprog_iter_idents: Vec<Expr> = iter_data.iter().map(|ref x| x.ident.clone()).collect();
        let mut subprog_iters = subprog_idents.iter().map(|x| code_util::simple_iter(x.clone())).collect();

        /* TODO propagate vectorize */
        let subprog_body = constructors::result_expr(
            constructors::for_expr(subprog_iters, (**builder).clone(), (**func).clone(), false)?)?;

        /* create args lists for dispatch */
        //print!("generating args\n");

        /* Collect any remaining params that are not in the list of iters. */
        let params: Vec<Parameter> = get_parameters(&subprog_body).into_iter().collect();
        let mut param_idents = vec![];
        for p in params.iter() {
            let is_sharded = ident_states.get(&p.name);
            let ty = if is_sharded == Some(&IdentState::Sharded) {
                Vector(Box::new(p.ty.clone()))
            } else {
                p.ty.clone()
            };
            
            let param_ident = constructors::ident_expr(p.name.clone(), ty)?;
            if iter_params.contains(&p) | subprog_iter_idents.contains(&param_ident) {
                continue;
            }
            param_idents.push(param_ident);
        }

        // each dispatch call requires the input vector as well as any auxiliary input arguments,
        // so we generate a loop over the shards that slices the input iters
        // and also adds the (remaining) input args in a struct
        // and then passes the whole thing to a dispatch call,
        // then stitches the result back using a merge.
        //print!("generating shards\n");
        let (args_res, input_params) = dispatch::gen_args_loop(&iter_data,
                                                               len_expr,
                                                               &param_idents,
                                                               partitions_conf,
                                                               e).unwrap();
        let mut subprog = constructors::lambda_expr(input_params,
                                                    subprog_body.clone()).unwrap();
        lookup_transform(&mut subprog, ident_states); // Transform any distributed lookups in the subexpression
        force_ident_types(&mut subprog, ident_states);
        
        //print!("generating dispatch loop\n");
        /* Create a loop to dispatch to all workers. 
         * Result of the dispatch loop is a vec of structs of {worker ID, pointer to result data}. */
        // let dispatch_loop = dispatch::gen_dispatch_loop(args_iter, &subprog, &subprog_body.ty, e).unwrap();
        let dispatch_loop = dispatch::gen_dispatch_all(&subprog, &subprog_body.ty, args_res, e).unwrap();

        // print!("generating sort\n");
        // /* Sort result pointers by worker ID so there is a canonical ordering. */
        // let sorted_results = constructors::result_expr(gen_sorted_values_by_key(
        //     &constructors::result_expr(dispatch_loop).unwrap(), e)?)?;
        // print!("sorted type: {}\n", &sorted_results.ty);

        //print!("dispatch loop result\n");
        // let results = constructors::result_expr(dispatch_loop)?;
        let results = dispatch_loop;

        let (result_sym, result_ident) = code_util::new_sym_and_ident("result",
                                                                      &(results.ty),
                                                                      &e);
        //let result_let = constructors::let_expr(result_sym, results,  )
        
        //print!("generating loop\n");
        /* Finally, merge result of dispatch. */
        let result_iter = code_util::simple_iter(result_ident.clone());

        let merge_loop: Expr = if let Builder(ref bk, _) = builder.ty {
            match bk {
                BuilderKind::Appender(..) => {
                    //print!("merge appender\n");
                    gen_merge_appender(&result_iter, subprog_body.ty, builder).unwrap()
                }, 
                BuilderKind::Merger(..) => {
                    //print!("merge merger\n");
                    gen_merge_merger(&result_iter, subprog_body.ty, &result_ident, builder).unwrap()
                },
                BuilderKind::DictMerger(..) => {
                    print!("merge dictionary\n");
                    gen_merge_dictmerger(&result_iter, subprog_body.ty, builder).unwrap()
                }, 
                BuilderKind::VecMerger(..) => {
                    //print!("merge vecmerger\n");
                    gen_merge_vecmerger(&result_iter, subprog_body.ty, builder).unwrap()
                },
                _ => {
                   // print!("Not distributing: builder kind not supported\n");
                    return Ok(None);
                }
            } 
        } else {
            return compile_err!("Non-builder found in for expression\n");
        };
        
        //print!("returning loop... {}\n", merge_loop.pretty_print());

        return Ok(Some(merge_loop));
    } else { // abort
        print!("Not distributing: not a For\n");
        return Ok(None);
    }
}

pub fn annotate_distribute(e: &Expr) -> Expr {
    let mut new_expr = e.clone();
    let cur = new_expr.annotations.get_bool(DISTRIBUTE_ANNOTATION);
    match cur {
        Some(x) => {
            match x {
                false => { return new_expr; } // If explicitly told not to distribute, don't change the annotation.
                _ => {}
            }
        }
        _ => {}
    }
    new_expr.annotations.set_bool(DISTRIBUTE_ANNOTATION, true);
    new_expr
}

pub fn annotate_sharded(e: &Expr) -> Expr {
    let mut new_expr = e.clone();
    new_expr.annotations.set_bool(SHARDED_ANNOTATION, true);
    new_expr
}

pub fn get_sharded(e: &Expr) -> bool {
    match e.annotations.get_bool(SHARDED_ANNOTATION) {
        Some(value) => value,
        None => false
    }
}

pub fn should_distribute(e: &Expr) -> bool {
    match e.annotations.get_bool(DISTRIBUTE_ANNOTATION) {
        Some(x) => x,
        None => false
    }
}

/// Returns true if any subexpression has the `distribute` annotation.
pub fn contains_distribute(expr: &Expr) -> bool {
    let mut ret = false;
    expr.traverse(&mut |ref e| {
        ret |= match e.annotations.get_bool(DISTRIBUTE_ANNOTATION) {
            None => false,
            Some(value) => value
        }
    });

    ret
}

/// Returns true if any subexpression is a distributed Appender
/// (the only builder that will return a result sharded on the cluster).
pub fn contains_sharded(expr: &Expr) -> bool {
    let mut ret = false;
    expr.traverse(&mut |ref e| {
        if let For { ref iters, ref builder, ref func } = e.kind {
            if let Builder(ref bk, _) = e.ty {
                if let Appender(ref elem) = bk {
                    ret |= match e.annotations.get_bool(DISTRIBUTE_ANNOTATION) {
                        None => false,
                        Some(value) => value
                    }
                }
            }
        }
    });

    ret
}

/// Transform any lookups containing a distributed input.
pub fn lookup_transform(expr: &mut Expr, ident_states: &mut FnvHashMap<Symbol, IdentState>) {
    expr.transform_and_continue(&mut |ref mut e| {
        if should_distribute(&e) {
            if let Lookup { ref data, ref index } = e.kind {
                if let Ident(ref sym) = data.kind {
                    let sharded = ident_states.get(&sym);
                    println!("Checking {}", sym.name());
                    match sharded {
                        None => {},
                        Some(x) => {
                            println!("Sharded state for {}: {:?}", sym.name(), x);
                            if *x == IdentState::Sharded {
                                let dist_lookup = gen_distributed_lookup(&e).unwrap();
                                //println!("Returning from lookup: {}\n", &dist_lookup.pretty_print());
                                return (Some(dist_lookup), false); // TODO
                            }
                        }
                    }
                } else if get_sharded(&data) {
                    let dist_lookup = gen_distributed_lookup(&e).unwrap();
                    //println!("Returning from lookup: {}\n", &dist_lookup.pretty_print());
                    return (Some(dist_lookup), false); // TODO
                }
            }
        } 

        if let Lookup { ref data, ref index } = e.kind {
            // if there was a Distribute annotation, safe to remove it now
            let mut replace = e.clone();
            if !(replace.annotations.get(DISTRIBUTE_ANNOTATION).is_none()) {
                replace.annotations.remove(DISTRIBUTE_ANNOTATION);
            }
            return (Some(replace), false);
        }
        
        return (None, true);
    });
}

/// Distribute any top-level loops annotated with `distribute`.
/// ident_states tracks any Idents that correspond to a sharded vector.
pub fn distribute_transform(expr: &mut Expr,
                            ident_states: &mut FnvHashMap<Symbol, IdentState>,
                            partitions_conf: &i32) {
    expr.transform_or_continue(&mut |ref mut e| {
        if should_distribute(&e) {
            if let For { ref iters, ref builder, ref func } = e.kind {
                /* Check that iters are all Idents. If not, wrap in an Ident before distributing. */
                let (mut ident_loop, new_symbols) = iters_to_idents(e).unwrap();
                let mut dist_loop = gen_distributed_loop(&mut ident_loop, ident_states,
                                                         partitions_conf).unwrap().unwrap();
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

/// Set all the potentially-sharded Idents to Unknown type and re-infer them.
fn force_ident_types(expr: &mut Expr, ident_states: &mut FnvHashMap<Symbol, IdentState>) {
    expr.transform_and_continue(&mut |ref mut e| {
        if let Ident(ref sym) = e.kind {
            if ident_states.get(&sym) == Some(&IdentState::Sharded) {
                e.ty = Unknown;
            }
        }

        (None, true)
    });

    let mut print_conf = PrettyPrintConfig::new();
    print_conf.show_types = true;
    print!("after erased types: {}\n", expr.pretty_print_config(&print_conf));

    expr.infer_types().unwrap();
}

/// Reconstruct some expressions on sharded arguments in order to force type correctness.
fn propagate_types(e: &mut Expr, ident_states: &mut FnvHashMap<Symbol, IdentState>)
                   -> WeldResult<Option<Expr>> {
    match e.kind {
        Let { ref name, ref value, ref body } => {
            let mut replace_let = constructors::let_expr(name.clone(),
                                                         (**value).clone(),
                                                         (**body).clone())?;
            replace_let.annotations = e.annotations.clone();
            return Ok(Some(replace_let));
        },
        If { ref cond, ref on_true, ref on_false } => {
            let sharded_t = get_sharded(&on_true);
            let sharded_f = get_sharded(&on_false);

            if !(sharded_t && sharded_f) {
                return compile_err!("on_true and on_false must match");
            }

            let mut replace_if = constructors::if_expr((**cond).clone(),
                                                       (**on_true).clone(),
                                                       (**on_false).clone())?;
            replace_if.annotations = e.annotations.clone();
            return Ok(Some(replace_if));
        },
        Select { ref cond, ref on_true, ref on_false } => {
            let sharded_t = get_sharded(&on_true);
            let sharded_f = get_sharded(&on_false);

            if !(sharded_t && sharded_f) {
                return compile_err!("on_true and on_false must match");
            }

            let mut replace_if = constructors::select_expr((**cond).clone(),
                                                           (**on_true).clone(),
                                                           (**on_false).clone())?;
            replace_if.annotations = e.annotations.clone();
            return Ok(Some(replace_if));
        },
        Res { ref builder } => {
            let mut replace_res = constructors::result_expr((**builder).clone())?;
            replace_res.annotations = e.annotations.clone();
            return Ok(Some(replace_res));
        },
        _ => {
            return Ok(None);
        }
    }
}
    
/// Propagate the `distribute` and `sharded` annotations upwards to parents.
fn propagate_annotations(e: &mut Expr, ident_states: &mut FnvHashMap<Symbol, IdentState>)
                         -> WeldResult<Option<Expr>> {
    match e.kind {
        Ident(ref sym) => {
            // These are tracked in ident_states.
        }
        Lambda { ref params, ref body } => {
            let mut replace = e.clone();
            //replace.apply_bool_annotation(&body, SHARDED_ANNOTATION);

            // TODO: We currently assume Lambdas never return sharded types.
            // Adding this capability requires adding an annotation to Type,
            // so that we can propagate the return type of the Lambda rather than
            // the body type.
            
            return Ok(Some(replace));
        },
        Lookup { ref data, ref index } => {
            return Ok(Some(annotate_distribute(&e)));
        },
        OptLookup { ref data, ref index } => {
            let sharded = get_sharded(&data);
            if sharded {
                unimplemented!()
            }
        },
        Let { ref name, ref value, ref body } => {
            let mut replace = e.clone();
            replace.apply_bool_annotation(&value, SHARDED_ANNOTATION);

            let new_annot = get_sharded(&value);
            match new_annot {
                true => {
                    ident_states.insert(name.clone(), IdentState::Sharded);
                },
                false => {
                    ident_states.insert(name.clone(), IdentState::Contiguous);
                }
            };
            
            return Ok(Some(replace));
        },
        Length { ref data } => {
            let sharded = get_sharded(&data);
            if sharded {
                unimplemented!()
            }
        },
        If { ref cond, ref on_true, ref on_false } => {
            // If is special because on_true and on_false must either both be sharded
            // or both be contiguous in order for the transforms to be correct.
            let sharded_t = get_sharded(&on_true);
            let sharded_f = get_sharded(&on_false);

            if !(sharded_t && sharded_f) {
                return compile_err!("on_true and on_false must match");
            }

            let mut replace = e.clone();
            replace.apply_bool_annotation(&on_true, SHARDED_ANNOTATION);
            return Ok(Some(replace));
        },
        Iterate { ref initial, ref update_func } => {
            let sharded = get_sharded(&initial) || get_sharded(&update_func); // this probably shouldn't happen...
            if sharded {
                println!("Unimplemented: {}", update_func.pretty_print());
                unimplemented!()
            }
        }
        Select { ref cond, ref on_true, ref on_false } => {
            // Select is special because on_true and on_false must either both be sharded
            // or both be contiguous in order for the transforms to be correct.
            let sharded_t = get_sharded(&on_true);
            let sharded_f = get_sharded(&on_false);

            if !(sharded_t && sharded_f) {
                return compile_err!("on_true and on_false must match");
            }

            let mut replace = e.clone();
            replace.apply_bool_annotation(&on_true, SHARDED_ANNOTATION);
            return Ok(Some(replace));
        },
        Slice { ref data, ref index, ref size } => {
            let sharded = get_sharded(&data);
            if sharded {
                unimplemented!()
            }
        },
        Sort { ref data, ref cmpfunc } => {
            let sharded = get_sharded(&data);
            if sharded {
                unimplemented!()
            }
        },
        Res { ref builder } => {
            let sharded = get_sharded(&builder);
            if sharded {
                /* update the type */
                let mut replace = constructors::result_expr((**builder).clone()).unwrap(); 
                replace.apply_bool_annotation(&builder, SHARDED_ANNOTATION);
                return Ok(Some(replace));
            }
        },
        CUDF { ref sym_name, ref args, ref return_ty } => {
            // We assume the CUDF itself never returns a sharded vector,
            // although a CUDF could be called on elements of a sharded vector.
        },
        Serialize(_) | Deserialize { .. } | NewBuilder(_) => {
            // pass
        },
        For { ref iters, ref builder, ref func } => {
            // any top-level Fors should already have the correct annotations
        },
        Merge { ref builder, ref value } => {
            // We assume we never merge a sharded vector as a value.
        },
        GetField { .. } | Literal(_) | Not(_) | Negate(_) | Broadcast(_) | BinOp { .. } |
        UnaryOp { .. } | Cast { .. } | ToVec { .. } | MakeStruct { .. } |
        MakeVector { .. } | Zip { .. } | GetField { .. } => {
            // these either only operate on scalars/structs, or will never return a distributed vector
        }
            
        _ => {
            println!("Could not find {}", e.pretty_print());
            unimplemented!() // keep this around so we don't forget any exprs
        } 
    }

    return Ok(Some(e.clone()));
}

/// Convert top-level For loops in this expr into distributed For loops.
/// Transform any vector operations on a distributed vector appropriately to be compatible with the new distributed input.
/// TODO iters_to_idents.
pub fn distribute(expr: &mut Expr, partitions_conf: &i32) -> WeldResult<()> {
    /* First annotate all top-level For loops to be distributed. */
    let mut print_conf = PrettyPrintConfig::new();
    print_conf.show_types = true;
    
    expr.transform_and_continue(&mut |ref mut e| {
        if let For { ref iters, ref builder, ref func } = e.kind {
            // Distribute annotation indicates that this loop should be transformed.
            for it in iters.iter() {
                if it.kind != ScalarIter {
                    // Can't distribute.
                    // Also, don't try to distribute anything below this,
                    // as we might end up merging several sharded vectors into
                    // a top-level vector using this loop.
                    return (None, false);
                }
            }
            
            let mut ret = annotate_distribute(&e);

            if let Builder(ref bk, _) = e.ty {
                if let Appender(_) = bk {
                    // Sharded annotation indicates that this loop
                    // will return a non-contiguous vector.
                    if should_distribute(&ret) { // If distribute was explicitly false, not sharded.
                        ret = annotate_sharded(&ret);
                    }
                }
            }

            return (Some(ret), false);
        }

        return (None, true)
    });

    /* Annotate any expressions derived from a distributed loop. 
     * Keep track of Let expressions that correspond to sharded vectors. */
    let mut ident_states = FnvHashMap::default();
    expr.transform_up_res(&mut |ref mut e| {
        return propagate_annotations(e, &mut ident_states);
    });
    
    /* Distribute one `distribute`-annotated loop at a time.
     * Iterate until no top-level `distribute` loops remain. */
    let mut i = 0;
    while contains_distribute(&expr) {
        /* Distribute as many loops as we can. */
        println!("distributing... iter {}", expr.pretty_print_config(&print_conf));

        distribute_transform(expr, &mut ident_states, partitions_conf);
        lookup_transform(expr, &mut ident_states);

        expr.transform_up_res(&mut |ref mut e| {
            return propagate_types(e, &mut ident_states);
        });

        i += 1;
    }

    println!("done, flatten.... expr {}", expr.pretty_print());
    /* Finally, flatten the topmost result that will be returned to the calling program. */
    expr.transform_once(&mut |ref mut e| Some(flatten_toplevel_func(e).unwrap()));
    println!("flattened expr: {}", expr.pretty_print());

    force_ident_types(expr, &mut ident_states);
    
    Ok(())
}
