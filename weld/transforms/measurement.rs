//! Code generation to make data-dependent transform decisions.
//!
//! Inserts code to measure selectivity of conditionals on data and dynamically decide
//! whether or not to predicate.

use std::collections::HashSet;

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use cost_params::*;
use error::*;
use exprs;
use util::SymbolGenerator;

use super::predication::*;

#[cfg(test)]
use parser::*;
#[cfg(test)]
use type_inference::*;

use pretty_print::*;

/// Merge 1 or 0 based on the value of cond.
fn generate_measurement_func(cond: &Expr<Type>,
                             elem_ty: &Type,
                             params_original: &Vec<Parameter<Type>>,
                             ctx: &Expr<Type>) -> WeldResult<(Expr<Type>,
                                                              Expr<Type>)> {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);
    
    let bk = BuilderKind::Merger(Box::new(Scalar(ScalarKind::F64)), BinOpKind::Add);
    let builder = exprs::newbuilder_expr(bk, None)?;

    // Create new params for the loop.
    let params = vec![Parameter {
                          name: sym_gen.new_symbol("b"),
                          ty: builder.ty.clone(),
                      },
                      Parameter {
                          name: sym_gen.new_symbol("i"),
                          ty: Scalar(ScalarKind::I64),
                      },
                      Parameter {
                          name: sym_gen.new_symbol("e"),
                          ty: elem_ty.clone()
                      }];

    let builder_name = exprs::ident_expr(params[0].name.clone(), params[0].ty.clone())?;

    let f64_1 = exprs::literal_expr(LiteralKind::F64Literal(1f64.to_bits()))?;
    let on_true = exprs::merge_expr(builder_name.clone(),
                                    f64_1.clone())?;
    let f64_2 = exprs::literal_expr(LiteralKind::F64Literal(0f64.to_bits()))?;
    let on_false = exprs::merge_expr(builder_name.clone(),
                                     f64_2.clone())?;

    // Substitute in new params.
    let mut cond_new = cond.clone();
    for i in 0..params.len() {
        cond_new.substitute(&params_original[i].name.clone(),
        &exprs::ident_expr(params[i].name.clone(), params[i].ty.clone())?);
    }

    let body = exprs::if_expr(cond_new.clone(), on_true, on_false)?;
    let func = exprs::lambda_expr(params, body)?;
    Ok((builder, func)) // need to pass builder to for loop
}

/// Check that this expr is either an If or an If nested inside some Lets.
pub fn can_measure(e: &Expr<Type>) -> bool {
    if let If { ref cond, .. } = e.kind {
        return true;
    } else if let Let { ref name, ref value, ref body } = e.kind {
        return can_measure(body);
    }

    false
}

/// Generate code to measure selectivity of first k rows.
/// Copies input iterator data -- vectors should be refactored into Let expressions as preprocessing.
pub fn measure_selectivity(e: &Expr<Type>, k: i64) -> WeldResult<Option<Expr<Type>>> {
    if let For { ref iters, ref builder, ref func } = e.kind {
        let mut measure_iters = vec![];
        let mut niters_exprs = vec![]; // TODO doing some extra copying to avoid an Option
        let mut types = vec![];

        //print!("got for!\n");
        let mut sym_gen = SymbolGenerator::from_expression(e);
        for it in iters.iter() {
            if let Vector(ref vec_ty) = (*it).data.ty { // only works if all inputs are Vectors
                // length should be min of vector length and requested length
                let niters_expr = exprs::binop_expr(BinOpKind::Min,
                                                    exprs::literal_expr(
                                                        LiteralKind::I64Literal(k))?,
                                                    exprs::length_expr(*it.data.clone())?)?;
                niters_exprs.push(niters_expr.clone());
                
                let measure_iter = Iter {
                    data: Box::new(*it.data.clone()),
                    start: Some(Box::new(exprs::literal_expr(LiteralKind::I64Literal(0))?)),
                    end: Some(Box::new(niters_expr.clone())),
                    stride: Some(Box::new(exprs::literal_expr(LiteralKind::I64Literal(1))?)),
                    kind: IterKind::ScalarIter
                };
                
                measure_iters.push(measure_iter);
                types.push(*vec_ty.clone());
            }
        }

        //print!("done making iters!\n");
        if let Lambda { ref params, ref body } = func.kind {
            //print!("got lambda!\n");
            if let If { ref cond, .. } = body.kind {
                //print!("got if! types: {}\n", types.len());
                let (builder, func) = if types.len() == 1 {
                    generate_measurement_func(cond, &types[0], params, e)?
                } else {
                    generate_measurement_func(cond, &Struct(types), params, e)?
                };
                
                //print!("got func! {}\n", print_expr(&func));
                let mut measure_loop = exprs::for_expr(measure_iters,
                                                       builder.clone(),
                                                       func, false)?; // don't vectorize for now
                //print!("got loop! {}\n", print_expr(&measure_loop));
                let mut res = exprs::binop_expr(BinOpKind::Divide, // normalize to get selectivity
                                                exprs::result_expr(measure_loop)?,
                                                exprs::cast_expr(ScalarKind::F64,
                                                             niters_exprs[0].clone())?)?; 
                //print!("got res! {}\n", print_expr(&res));
                return Ok(Some(res))
            }
        }
    }
    
    return Ok(None)
}

/// Get accesses (GetField, Lookup, Ident) into the data referred to by it_name, removing duplicates.
/// Returns list of accesses along with element sizes.
/// TODO make duplicate removal faster.
fn get_accesses(e: &Expr<Type>, it_name: &Symbol) -> Vec<(Expr<Type>, u32)> {
    let mut exprs: Vec<Expr<Type>> = vec![];
    let mut sizes: Vec<u32> = vec![];
    
    e.traverse_early_stop(&mut |ref e| {
        if let GetField { ref expr, ref index } = e.kind {
            if let Ident(ref sym) = expr.kind {
                if sym == it_name && !(exprs.contains(e)) {
                    // TODO expr.ty?
                    exprs.push((*e).clone());
                    sizes.push(expr.ty.bits().unwrap());
                    return true;
                }
            }
        } else if let Lookup { ref data, ref index } = e.kind {
            if let Ident(ref sym) = data.kind {
                if sym == it_name && !(exprs.contains(e)) {
                    exprs.push((*e).clone());
                    sizes.push(data.ty.bits().unwrap());
                    return true;
                }
            }
        } else if let Ident(ref sym) = e.kind {
            if sym == it_name && !(exprs.contains(e)) {
                exprs.push((*e).clone());
                sizes.push(e.ty.bits().unwrap());
                return true;
            }
        }
        
        false
    });

    // TODO remove extra clone
    exprs.iter().zip(sizes).map(|x| ((*x.0).clone(), x.1)).collect()
}

/// Generate code to compute memory cost for a computed selectivity value.
/// selectivity should be an Ident to avoid recomputing expression.
///
/// Pr(line accessed) = 1 - (1-s)^{line width}
/// Pr(sequential access) = Pr(line access)^2
/// Pr(random access) = Pr(access) - Pr(sequential access)
fn memory_cost(selectivity: &Expr<Type>,
               block_size: &Expr<Type>) -> WeldResult<Option<(Expr<Type>, Expr<Type>)>> {
    let line_access_expr = exprs::binop_expr(BinOpKind::Pow,
                                             exprs::binop_expr(BinOpKind::Subtract,
                                                               exprs::literal_expr(
                                                                   LiteralKind::F64Literal(
                                                                       1f64.to_bits()))?,
                                                               selectivity.clone())?,
                                             block_size.clone())?;
    let access_expr = exprs::binop_expr(BinOpKind::Subtract,
                                        exprs::literal_expr(LiteralKind::F64Literal(1f64.to_bits()))?,
                                        line_access_expr)?;
    let sequential_expr = exprs::binop_expr(BinOpKind::Pow, access_expr.clone(),
                                            exprs::literal_expr(LiteralKind::F64Literal(2f64.to_bits()))?)?;
    let random_expr = exprs::binop_expr(BinOpKind::Subtract, access_expr.clone(),
                                        sequential_expr.clone())?;

    Ok(Some((sequential_expr, random_expr)))
}

pub fn load_cost(e: &Expr<Type>,
                 selectivity: &Expr<Type>,
                 vectorized: bool) -> WeldResult<Option<Expr<Type>>> {
    if let Lambda { ref params, ref body } = e.kind {
        //print!("lambda2\n");
        if !(can_predicate(body)) {
            return Ok(None);
        }

        if params.len() != 3 {
            return Ok(None); // need lambda in the form |b,i,e|
        }

        let ref it_name = params[2].name;
        
        // Expression is of the form if(cond, merge(b, e), b)
        if let If { ref cond, ref on_true, ref on_false } = body.kind {
            //print!("if\n");
            // get conditional accesses from cond
            let mut cond_accesses = get_accesses(cond, it_name);
            //print!("cond accesses\n");
            // get body accesses from on_true
            let mut merge_accesses = vec![];
            if let Merge { ref builder, ref value } = on_true.kind {
                merge_accesses = get_accesses(on_true, it_name);
            }
            //print!("merge accesses\n");
            // accesses that only occur in body and not cond
            let mut body_accesses = vec![];
            for exp in merge_accesses {
                //print!("{}\n", print_expr(&exp.0));
                if !cond_accesses.contains(&exp) {
                    body_accesses.push(exp.clone());
                }
            }
            
            if vectorized {
                // Vectorized, so all accesses look sequential.
                //print!("vectorized\n");
                let mut ret: f64 = 0.0;
                for val in cond_accesses.iter() {
                    ret += mem_cost_sequential(val.1);
                }

                for val in body_accesses.iter() {
                    ret += mem_cost_sequential(val.1);
                }

                ret *= VEC_CONSTANT;
                
                //print!("returning expr\n");
                return Ok(Some(exprs::literal_expr(
                    LiteralKind::F64Literal(ret.to_bits()))?));
            } else {
                // Unvectorized, so conditional accesses look sequential and
                // body accesses look random.
                // Note that with short-circuiting, selectivity also applies in conditional.

                //print!("unvectorized\n");
                let mut seq_cost: f64 = 0.0;
                for val in cond_accesses.iter() {
                    seq_cost += mem_cost_sequential(val.1);
                }

                let mut ret = exprs::literal_expr(LiteralKind::F64Literal(
                    seq_cost.to_bits()))?;
                for val in body_accesses.iter() {
                    //print!("val...\n");
                    let cost_expr = exprs::literal_expr(LiteralKind::F64Literal(
                        mem_cost_random().to_bits()))?;
                    ret = exprs::binop_expr(BinOpKind::Add,
                                            ret.clone(),
                                            cost_expr.clone())?;
                }
                //print!("returning unvec expr\n");
                ret = exprs::binop_expr(BinOpKind::Multiply,
                                        selectivity.clone(),
                                        ret.clone())?;
                return Ok(Some(ret));
            }
        }
    }
    
    Ok(None)
}

/// Estimated processing cost for arithmetic expression.
fn processing_cost(e: &Expr<Type>) -> WeldResult<Option<f64>> {
    let mut total_cost: f64 = 0.0;
    e.traverse(&mut |ref e| {
        total_cost += 1.0;
    });

    Ok(Some(total_cost))
}

/// Cost of expression when predicated and vectorized.
pub fn predicated_cost(e: &mut Expr<Type>) -> WeldResult<Option<f64>> {
    if !(can_predicate(e)) {
        return Ok(None);
    }

    if let If { ref cond, ref on_true, ref on_false } = e.kind {
        let cond_cost = processing_cost(cond)?.unwrap();
        let vectorized_cond_cost = cond_cost / 5.0;
        let body_cost = processing_cost(on_true)?.unwrap();
        let vectorized_body_cost = body_cost / 5.0;

        return Ok(Some(vectorized_cond_cost + vectorized_body_cost));
    }

    return Ok(None);
}

pub fn generate_measurement_branch(e: &mut Expr<Type>) {
    //println!("in generate");
    e.transform_and_continue_res(&mut |ref mut e| {
        //print!("transforming: {}\n", print_expr(e));
        let mut data_clones = vec![];
        let mut names = vec![];
        let mut new_iters = vec![];
        let mut sym_gen = SymbolGenerator::from_expression(e);

        if let For { ref iters, ref builder, ref func } = e.kind {
            //print!("got for! {}\n", print_expr(e));
            
            /* generate new names for all iters to avoid copying */
            for it in iters.iter() {
                //print!("in iter loop\n");
                let name = sym_gen.new_symbol("d"); // don't copy the data
                let data_ident = exprs::ident_expr(name.clone(),
                                                   (*it).data.ty.clone())?;
                //print!("got ident! {}\n", print_expr(&data_ident));
                data_clones.push(*(*it).data.clone());
                names.push(name.clone());
                
                let mut new_iter = it.clone();
                new_iter.data = Box::new(data_ident.clone());
                new_iters.push(new_iter);
            }

            let mut unpredicated = exprs::for_expr(new_iters.clone(),
                                                   *builder.clone(), *func.clone(), false)?;

            //print!("got unpredicated! {}\n", print_expr(e));
            if let For { ref iters, ref builder, ref func } = unpredicated.kind {
                //println!("got for! {}\n", print_expr(e));
                if let Lambda { ref params, ref body } = func.kind {
                    //print!("got lambda! {}\n", print_expr(e));
                    if !(should_be_predicated(body) && can_predicate(body)) {
                        //print!("not predicating in branch! {}", print_expr(body));
                        return Ok((None, false));
                    }

                    //print!("generating measurement! {}\n", print_expr(e));

                    // insert measurement code
                    let measure_code = measure_selectivity(&unpredicated, 200)?.unwrap();
                    let sel_name = sym_gen.new_symbol("s");
                    let sel_ident = exprs::ident_expr(sel_name.clone(), measure_code.ty.clone())?;

                    // TODO handle None
                    //print!("branched cost\n");
                    let branched_cost = load_cost(func, &sel_ident, false)?.unwrap();
                    //print!("load cost\n");
                    let pred_cost = load_cost(func, &sel_ident, true)?.unwrap();

                    //print!("calling predicate! {}\n", print_expr(e));
                    let pred_body = generate_predicated_expr(body)?.unwrap();
                    let predicated = exprs::for_expr(iters.clone(), *builder.clone(),
                                                     exprs::lambda_expr(params.clone(),
                                                                        pred_body)?,
                                                     false)?;
                    
                    //print!("got predicate! {}\n", print_expr(&predicated));
                    let mut cond = exprs::let_expr(sel_name, measure_code.clone(),
                                                   exprs::binop_expr(BinOpKind::GreaterThan,
                                                                     branched_cost, pred_cost)?)?;

                    let mut branch = exprs::if_expr(cond.clone(),
                                                    predicated.clone(),
                                                    unpredicated.clone())?;

                    //print!("got branch! {}\n", print_expr(&branch));
                    for (name, data) in names.iter().zip(data_clones.iter()) {
                        branch = exprs::let_expr((*name).clone(),
                                                 (*data).clone(),
                                                 branch)?;
                    }
                    return Ok((Some(branch), false));
                }
            }
        }
        return Ok((None, true));
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
fn cond_test() {
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| @(predicate:true)if(e>0, merge(b,e), b)))";
    let mut e = typed_expr(code);
    generate_measurement_branch(&mut e);
    //print!("{}\n", print_typed_expr(&e).as_str());

    //let expected = "|v:vec[i32]|result(for(v:vec[i32],merger[i32,+],|b:merger[i32,+],i:i64,e:i32|merge(b:merger[i32,+],select((e:i32>0),e:i32,0))))";
    //assert_eq!(print_typed_expr_without_indent(&typed_e.unwrap()).as_str(),
    //           expected);
    
}

#[test]
fn cost_test() {
    let code = "|v:vec[i32], d:vec[i32]| result(for(zip(d, v), merger[i32,+], |b,i,e| if(e.$0>0, merge(b,e.$0+e.$1), b)))";
    let mut e = typed_expr(code);
    generate_measurement_branch(&mut e);
    //print!("{}\n", print_typed_expr(&e).as_str());

    //let expected = "|v:vec[i32]|result(for(v:vec[i32],merger[i32,+],|b:merger[i32,+],i:i64,e:i32|merge(b:merger[i32,+],select((e:i32>0),e:i32,0))))";
    //assert_eq!(print_typed_expr_without_indent(&typed_e.unwrap()).as_str(),
    //           expected);
    
}
