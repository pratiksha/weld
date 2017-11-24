//! Code generation to make data-dependent transform decisions.
//!
//! Inserts code to measure selectivity of conditionals on data and dynamically decide
//! whether or not to predicate.

use std::collections::HashSet;

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use error::*;
use exprs;
use util::SymbolGenerator;

use super::predication::*;

#[cfg(test)]
use parser::*;
#[cfg(test)]
use type_inference::*;

use pretty_print::*;

/// merge 1 or 0 based on the value of cond.
fn generate_measurement_func(cond: &Expr<Type>,
                             elem_ty: &Type,
                             params_original: &Vec<Parameter<Type>>,
                             ctx: &Expr<Type>) -> WeldResult<(Expr<Type>,
                                                              Expr<Type>)> {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);
    
//    print!("in generate func! {}\n", print_expr(cond));
    let bk = BuilderKind::Merger(Box::new(Scalar(ScalarKind::F64)), BinOpKind::Add);
    let builder = exprs::newbuilder_expr(bk, None)?;

    // Create new params for the loop.
//    print!("type in measure: {}\n", print_type(elem_ty));
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
    
//    print!("got cond! {}\n", print_expr(&cond_new));
    let body = exprs::if_expr(cond_new.clone(), on_true, on_false)?;
//    print!("got branch! {}\n", print_expr(&body));

    let func = exprs::lambda_expr(params, body)?;
//    print!("finished func! {}\n", print_expr(&func));
    Ok((builder, func)) // need to pass builder to for loop
}

/// Generate code to measure selectivity of first k rows.
/// Copies input iterator data -- vectors should be refactored into Let expressions as preprocessing.
pub fn measure_selectivity(e: &Expr<Type>, k: i64) -> WeldResult<Option<Expr<Type>>> {
    if let For { ref iters, ref builder, ref func } = e.kind {
        let mut measure_iters = vec![];
        let mut niters_exprs = vec![]; // TODO doing some extra copying to avoid an Option
        let mut types = vec![];

//        print!("got for!\n");
        let mut sym_gen = SymbolGenerator::from_expression(e);
        for it in iters.iter() {
            if let Vector(ref vec_ty) = (*it).data.ty { // only works if all inputs are Vectors
                if let Lambda { ref params, ref body } = func.kind {
                    if let If { ref cond, .. } = body.kind {
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
            }
        }

//        print!("done making iters!\n");
        if let Lambda { ref params, ref body } = func.kind {
//            print!("got lambda!\n");
            if let If { ref cond, .. } = body.kind {
//                print!("got if! types: {}\n", types.len());
                let (builder, func) = if types.len() == 1 {
                    generate_measurement_func(cond, &types[0], params, e)?
                } else {
                    generate_measurement_func(cond, &Struct(types), params, e)?
                };
                
//                print!("got func! {}\n", print_expr(&func));
                let mut measure_loop = exprs::for_expr(measure_iters,
                                                       builder.clone(),
                                                       func, false)?; // don't vectorize for now
//                print!("got loop! {}\n", print_expr(&measure_loop));
                let mut res = exprs::binop_expr(BinOpKind::Divide, // normalize to get selectivity
                                                exprs::result_expr(measure_loop)?,
                                                exprs::cast_expr(ScalarKind::F64,
                                                             niters_exprs[0].clone())?)?; 
//                print!("got res! {}\n", print_expr(&res));
                return Ok(Some(res))
            }
        }
    }
    
    return Ok(None)
}

/// Get accesses (GetField, Lookup, Ident) into the data referred to by it_name, removing duplicates.
/// TODO make duplicate removal faster.
fn get_accesses(e: &Expr<Type>, it_name: &Symbol) -> Vec<Expr<Type>> {
    let mut exprs: Vec<Expr<Type>> = vec![];
    
    e.traverse_early_stop(&mut |ref e| {
        if let GetField { ref expr, ref index } = e.kind {
            if let Ident(ref sym) = expr.kind {
                if sym == it_name && !(exprs.contains(e)) {
                    exprs.push((*e).clone());
                    return true;
                }
            }
        } else if let Lookup { ref data, ref index } = e.kind {
            if let Ident(ref sym) = data.kind {
                if sym == it_name && !(exprs.contains(e)) {
                    exprs.push((*e).clone());
                    return true;
                }
            }
        } else if let Ident(ref sym) = e.kind {
            if sym == it_name && !(exprs.contains(e)) {
                exprs.push((*e).clone());
                return true;
            }
        }
        
        false
    });

    exprs
}

pub fn load_cost(e: &mut Expr<Type>) -> WeldResult<Option<f64>> {
    if let Lambda { ref params, ref body } = e.kind {
        if let Res { ref builder } = body.kind {
            if let For { ref iters, ref builder, ref func } = builder.kind { 
                if let Lambda { ref params, ref body } = func.kind {
                    if !(can_predicate(body)) {
                        return Ok(None);
                    }

                    if params.len() != 3 {
                        return Ok(None); // need lambda in the form |b,i,e|
                    }

                    let ref it_name = params[2].name;
                    
                    // Expression is of the form if(cond, merge(b, e), b)
                    if let If { ref cond, ref on_true, ref on_false } = body.kind {
                        // get accesses from cond
                        let mut cond_accesses = get_accesses(cond, it_name);
                        
                        // get accesses from on_true
                        let mut merge_accesses: Vec<Expr<Type>> = vec![];
                        if let Merge { ref builder, ref value } = on_true.kind {
                            merge_accesses = get_accesses(on_true, it_name);
                        }

                        let mut diff_accesses = vec![];
                        for exp in merge_accesses {
                            print!("{}\n", print_expr(&exp));
                            if !cond_accesses.contains(&exp) {
                                diff_accesses.push(exp.clone());
                            }
                        }

                        for exp in diff_accesses {
                            print!("{}\n", print_expr(&exp));
                        }
                        
                        return Ok(Some(0.0));
                    }
                }
            }
        }
    }
    
    Ok(None)
}

pub fn generate_measurement_branch(e: &mut Expr<Type>) {
//    println!("in generate");
//    print!("transforming: {}\n", print_expr(e));
    e.transform_and_continue_res(&mut |ref mut e| {
        let mut data_clones = vec![];
        let mut names = vec![];
        let mut new_iters = vec![];

        if let For { ref iters, ref builder, ref func } = e.kind {
            let mut sym_gen = SymbolGenerator::from_expression(e);

            /* generate new names for all vectors to avoid copying */
            for it in iters.iter() {
                if let Vector(ref vec_ty) = (*it).data.ty { // only works if all inputs are Vectors
                    if let Lambda { ref params, ref body } = func.kind {
                        if let If { ref cond, .. } = body.kind {
                            let name = sym_gen.new_symbol("d"); // don't copy the data
                            let data_ident = exprs::ident_expr(name.clone(),
                                                               (*it).data.ty.clone())?;
                            data_clones.push(*(*it).data.clone());
                            names.push(name.clone());

                            let mut new_iter = it.clone();
                            new_iter.data = Box::new(data_ident.clone());
                            new_iters.push(new_iter);
                        }
                    }
                }
            }

            let mut unpredicated = exprs::for_expr(new_iters.clone(),
                                                   *builder.clone(), *func.clone(), false)?;

            if let For { ref iters, ref builder, ref func } = unpredicated.kind {
                //            println!("got for! {}\n", print_expr(e));
                if let Lambda { ref params, ref body } = func.kind {
                    //                print!("got lambda! {}\n", print_expr(e));
                    if !(should_be_predicated(body)) {
                        //                    print!("not predicating in branch! {}", print_expr(body));
                        return Ok((None, false));
                    }

                    //                print!("generating measurement! {}\n", print_expr(e));

                    // insert measurement code
                    let measure_code = measure_selectivity(&unpredicated, 200)?.unwrap();
                    let threshold = exprs::literal_expr(LiteralKind::F64Literal((0.02f64).to_bits()))?;
                    //                print!("calling predicate! {}\n", print_expr(e));
                    let pred_body = generate_predicated_expr(body)?.unwrap();
                    let predicated = exprs::for_expr(iters.clone(), *builder.clone(),
                                                     exprs::lambda_expr(params.clone(),
                                                                        pred_body)?,
                                                     false)?;
                    
                    //                print!("got predicate! {}\n", print_expr(&predicated));
                    let mut branch = exprs::if_expr(
                        exprs::binop_expr(BinOpKind::GreaterThan, measure_code, threshold)?,
                        predicated.clone(),
                        unpredicated.clone())?;
                    //                print!("got branch! {}\n", print_expr(&branch));
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
    print!("{}\n", print_typed_expr(&e).as_str());

    //let expected = "|v:vec[i32]|result(for(v:vec[i32],merger[i32,+],|b:merger[i32,+],i:i64,e:i32|merge(b:merger[i32,+],select((e:i32>0),e:i32,0))))";
    //assert_eq!(print_typed_expr_without_indent(&typed_e.unwrap()).as_str(),
    //           expected);
    
}

#[test]
fn cost_test() {
    let code = "|v:vec[i32], d:vec[i32]| result(for(zip(d, v), merger[i32,+], |b,i,e| if(e.$0>0, merge(b,e.$0+e.$1), b)))";
    let mut e = typed_expr(code);
    let cost = load_cost(&mut e).unwrap().unwrap();
    print!("{}\n", cost);

    //let expected = "|v:vec[i32]|result(for(v:vec[i32],merger[i32,+],|b:merger[i32,+],i:i64,e:i32|merge(b:merger[i32,+],select((e:i32>0),e:i32,0))))";
    //assert_eq!(print_typed_expr_without_indent(&typed_e.unwrap()).as_str(),
    //           expected);
    
}
