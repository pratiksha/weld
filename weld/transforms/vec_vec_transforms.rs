//! Utility functions for moving between operations on vec[T] and vec[vec[T]].
//! Used in the `distribute` transform for operations on shards.

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

/// Transform a loop over a vec[T] (marked with @distribute) into a loop over a vec[vec[T]].
fn flatten_vec_vec_for(e: &Expr<Type>) -> WeldResult<Expr<Type>> {
    if let For { ref iters, ref builder, .. } = e.kind {
        if it.len() != 1 { /* TODO zip not supported yet */
            return Ok(None)
        }

        for it in iters.iter() {
            if it.annotations.distributed() {
                // this was a vec[T] in the AST, but has been transformed into a vec[vec[T]] via a dispatch call elsewhere
                // so transform the loop accordingly
                new_for = exprs::for_expr()?;
            }            
        }
    }

    Ok(None)
}

/// Parse and perform type inference on an expression.
#[cfg(test)]
fn typed_expr(code: &str) -> TypedExpr {
    let mut e = parse_expr(code).unwrap();
    assert!(infer_types(&mut e).is_ok());
    e.to_typed().unwrap()
}

#[test]
fn vec_iter_test() {
    let code = "|x:vec[vec[i32]]| result(for((@distribute:true)x, merger[i32, +], |b,i,e|merge(b, e)))";
    let mut e = typed_expr(code);
    distribute(&mut e, &1);
    print!("{}\n", print_typed_expr(&e));
}