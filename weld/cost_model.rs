use std::str;
use std::slice;

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::Type::*;
use super::cost_params::*;
use super::parser::*;
use super::type_inference::*;
use super::transforms::predication::*;
use error::*;

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

pub fn load_cost(e: &Expr<Type>,
                 selectivity: f64,
                 vectorized: bool) -> WeldResult<Option<f64>> {
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
                return Ok(Some(ret));
            } else {
                // Unvectorized, so conditional accesses look sequential and
                // body accesses look random.
                // Note that with short-circuiting, selectivity also applies in conditional.

                //print!("unvectorized\n");
                let mut seq_cost: f64 = 0.0;
                for val in cond_accesses.iter() {
                    seq_cost += mem_cost_sequential(val.1);
                }

                for val in body_accesses.iter() {
                    //print!("val...\n");
                    seq_cost += mem_cost_random();
                }
                //print!("returning unvec expr\n");
                return Ok(Some(seq_cost * selectivity));
            }
        }
    }
    
    Ok(None)
}


/// An in memory representation of a Weld vector.
#[derive(Clone)]
#[allow(dead_code)]
#[repr(C)]
pub struct WeldVec<T> {
    data: *const T,
    len: i64,
}

#[no_mangle]
pub unsafe extern "C" fn weld_rt_cost_model(selectivity: f64, branched: *const WeldVec<u8>, vectorized: bool) ->
    WeldResult<f64> {
        let result = (*branched).clone();
        let branched_str = str::from_utf8(slice::from_raw_parts(result.data, result.len as usize)).unwrap();
        let mut branched_prog = parse_expr(branched_str).unwrap();
        infer_types(&mut branched_prog)?;
        let cost = load_cost(&branched_prog.to_typed()?, selectivity, vectorized)?;
        Ok(cost.unwrap())
}