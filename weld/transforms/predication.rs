//! Predicates expressions in the AST.
//!
//!

use std::collections::HashSet;

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use error::*;
use exprs;
use util::SymbolGenerator;

#[cfg(test)]
use parser::*;
#[cfg(test)]
use type_inference::*;

use pretty_print::*;

pub fn can_predicate(e: &Expr<Type>) -> bool {
    if let If { ref cond, ref on_true, ref on_false } = e.kind {
        if let Merge { ref builder, ref value } = on_true.kind {
            if let Ident(ref name) = on_false.kind {
                if let Ident(ref name2) = builder.kind {
                    if name == name2 {
                        if let Builder(ref bk, _) = builder.ty {
                            print!("can predicate\n");
                            return true;
                        }
                    }
                }
            }
        }
    }

    print!("can't predicate\n");
    false
}

pub fn generate_predicated_expr(e: &Expr<Type>) -> WeldResult<Option<Expr<Type>>> {
    if !(should_be_predicated(e)) {
//        print!("not predicating! {}\n", print_expr(e));
        return Ok(None);
    }

//    print!("predicating now! {}\n", print_expr(e));
    // Predication for a value merged into a merger. This pattern checks for if(cond, merge(b, e), b).
    if let If { ref cond, ref on_true, ref on_false } = e.kind {
        if let Merge { ref builder, ref value } = on_true.kind {
            if let Ident(ref name) = on_false.kind {
                if let Ident(ref name2) = builder.kind {
                    if name == name2 {
                        if let Builder(ref bk, _) = builder.ty {
                            // Merge in the identity element if the predicate fails
                            // (effectively merging in nothing)
                            let (ty, op) = match *bk {
                                BuilderKind::Merger(ref ty, ref op) => (ty, op),
                                BuilderKind::DictMerger(_, ref ty2, ref op) => (ty2, op),
                                BuilderKind::VecMerger(ref ty, ref op) => (ty, op),
                                _ => {
                                    return Ok(None);
                                }
                            };

                            let identity = get_id_element(ty.as_ref(), op)?;
                            match identity {
                                Some(x) => {
                                    match *bk {
                                        BuilderKind::Merger(_, _)
                                            | BuilderKind::VecMerger(_, _) => {
                                                /* Change if(cond, merge(b, e), b) => 
                                                merge(b, select(cond, e, identity). */
                                                let expr = exprs::merge_expr(*builder.clone(),
                                                                             exprs::select_expr(
                                                                                 *cond.clone(),
                                                                                 *value.clone(), x)?)?;
                                                return Ok(Some(expr));
                                                
                                            },
                                        BuilderKind::DictMerger(_, _, _) => {
                                            /* For dictmerger, need to match identity element 
                                            back to the key. */
                                            let sel_expr = make_select_for_kv(*cond.clone(),
                                                                              *value.clone(),
                                                                              x)?;
                                            return Ok(sel_expr); 
                                        }
                                        _ => {
                                            return Ok(None);
                                        }
                                    }
                                }
                                None => {
                                    return Ok(None);
                                }
                            };

                        }
                    }
                }
            }
        }
    }
    Ok(None)
}
    
/// Predicate an `If` expression by checking for if(cond, merge(b, e), b) and transforms it to merge(b, select(cond, e, identity)).
pub fn predicate(e: &mut Expr<Type>) {
    e.transform_and_continue_res(&mut |ref mut e| {
        if !(should_be_predicated(e)) {
            return Ok((None, true));
        }

        return Ok((generate_predicated_expr(e)?, true));
    });
}

pub fn should_be_predicated(e: &Expr<Type>) -> bool {
    true
    //e.annotations.predicate()
}

fn get_id_element(ty: &Type, op: &BinOpKind) -> WeldResult<Option<Expr<Type>>> {
    let ref sk = match *ty {
        Scalar(sk) => sk,
        _ => {
            return Ok(None);
        }
    };

    /* Dummy element to merge when predicate fails. */
    let identity = match *op {
        BinOpKind::Add => {
            match *sk {
                ScalarKind::I8 => exprs::literal_expr(LiteralKind::I8Literal(0))?,
                ScalarKind::I32 => exprs::literal_expr(LiteralKind::I32Literal(0))?,
                ScalarKind::I64 => exprs::literal_expr(LiteralKind::I64Literal(0))?,
                ScalarKind::F32 => exprs::literal_expr(LiteralKind::F32Literal(0f32.to_bits()))?,
                ScalarKind::F64 => exprs::literal_expr(LiteralKind::F64Literal(0f64.to_bits()))?,
                _ => {
                    return Ok(None);
                }
            }
        }
        BinOpKind::Multiply => {
            match *sk {
                ScalarKind::I8 => exprs::literal_expr(LiteralKind::I8Literal(1))?,
                ScalarKind::I32 => exprs::literal_expr(LiteralKind::I32Literal(1))?,
                ScalarKind::I64 => exprs::literal_expr(LiteralKind::I64Literal(1))?,
                ScalarKind::F32 => exprs::literal_expr(LiteralKind::F32Literal(1f32.to_bits()))?,
                ScalarKind::F64 => exprs::literal_expr(LiteralKind::F64Literal(1f64.to_bits()))?,
                _ => {
                    return Ok(None);
                }
            }
        }
        _ => {
            return Ok(None);
        }
    };
    Ok(Some(identity))
}

fn make_select_for_kv(cond:  Expr<Type>,
                      kv:    Expr<Type>,
                      ident: Expr<Type>) -> WeldResult<Option<Expr<Type>>> {
    let mut sym_gen = SymbolGenerator::from_expression(&kv);
    let name = sym_gen.new_symbol("k");
    
    let kv_struct = exprs::ident_expr(name.clone(), kv.ty.clone())?;
    let kv_ident = exprs::makestruct_expr(vec![exprs::getfield_expr(kv_struct.clone(), 0)?, ident])?; // use the original key and the identity as the value
    
    let sel = exprs::select_expr(cond, kv_struct, kv_ident)?;
    let le = exprs::let_expr(name, kv, sel)?; /* avoid copying key */
    return Ok(Some(le))
}