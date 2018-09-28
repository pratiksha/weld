//! Utility transform for wrapping non-Ident data in an Iter in an Ident to make the Iter suitable to
//! distribute.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use error::*;
use exprs;
use util::SymbolGenerator;

use transforms::distribute::code_util;

pub fn iters_to_idents(e: &Expr<Type>) -> WeldResult<(Expr<Type>, Vec<(Symbol, Expr<Type>)>)> {
    let mut new_symbols = vec![];
    let mut new_iters   = vec![];
    
    if let For { ref iters, ref builder, ref func } = e.kind {
        for it in iters.iter() {
            if let Ident(_) = (*it).data.kind {
                new_iters.push(it.clone());
            } else {
                let (new_sym, new_ident) = code_util::new_sym_and_ident("iter_sym", &(*it).data.ty, e);
                new_iters.push(
                    Iter {
                        data: Box::new(new_ident.clone()),
                        start: it.start.clone(), end: it.end.clone(), stride: it.stride.clone(),
                        kind: it.kind.clone(), strides: it.strides.clone(), shape: it.shape.clone()
                    });
                new_symbols.push((new_sym, (*(*it).data).clone()));
            }
        }

        let mut is_vectorizable = true;
        for it in new_iters.iter() {
            if let Vector(ref ty) = it.data.ty {
                if let Simd(ref sk) = (**ty) {
                    continue;
                } else {
                    is_vectorizable = false;
                }
            }
        }

        /* TODO propagate vectorized */
        let mut new_loop = exprs::for_expr(new_iters, (**builder).clone(), (**func).clone(), is_vectorizable)?;
        new_loop.annotations.set_sharded(e.annotations.sharded());
        Ok((new_loop, new_symbols))
    } else {
        compile_err!("iters_to_idents: non-For passed to iters_to_idents")
    }
}