//! Automatically extract an outer For loop to distribute on the cluster,
//! and convert the For into a DistFor.

use ast::Expr;
use ast::Parameter;
use ast::Type;

use ast::ExprKind::*;
use ast::LiteralKind::*;
use ast::ScalarKind::*;

use ast::constructors;

use error::WeldResult;

use std::collections::HashSet;

#[cfg(test)]
use tests::*;

fn get_parameters(e: &Expr) -> Vec<Parameter> {
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

    syms.difference(&defs).cloned().into_iter().collect() // return all elements that are accessed but not defined
}

/// Used to construct a DistFor from a For.
/// The iters and builder in the original For remain the iters and builder in the DistFor.
/// func will be applied in batch in the DistFor.
pub fn distfor_expr(inner_for: &Expr) -> WeldResult<Expr> {
    if let For { ref iters, ref builder, ref func } = inner_for.kind {
        if let NewBuilder(_) = builder.kind {
            // Wrap the For in a lambda, which is the function that will be executed on the remote machine.
            // The arguments to the lambda are the iters followed by any additional params.
            // The iters will be zipped together when the For loop is executed and passed to `func`.
            // Currently we only support loops where all iter data vectors are Idents.

            let params = vec![];
            let args = vec![];

            for iter in iters.iter() {
                if let Ident(ref sym) = iter.data.kind {
                    let iter_param = Parameter {name: sym.name.clone(), ty: iter.data.ty.clone() };
                    args.push(constructors::ident_expr(iter_param.name.clone(), iter_param.ty.clone())?);
                    params.push(iter_param);
                } else {
                    return compile_err!("Only Ident iters can be distributed: {}", iter.data.kind);
                }
            }

            // Collect a list of any other Idents accessed (but not defined) in the For.
            let all_params = get_parameters(&inner_for);
            for p in all_params.iter() {
                args.push(constructors::ident_expr(p.name.clone(), p.ty.clone())?);
                if !(params.contains(&p)) {
                    params.push(p);
                }
            }

            // Wrap in the Lambda.
            let for_func = constructors::lambda_expr(all_params, inner_for.clone())?;

            // Serialize the program to string.
            use ast::{PrettyPrint, PrettyPrintConfig};
            
            let mut print_conf = PrettyPrintConfig::new()
                .show_types(true)
                .should_indent(false);
            let code = for_func.pretty_print_config(&print_conf);

            // DistFor always implicitly materializes the result, unlike For, which always returns a builder.
            let final_ty = if let Appender(ref elem) = builder.kind {
                DistVector(elem.clone())
            } else {
                builder.result_type()
            };
            
            constructors::new_expr(DistFor {
                iters: iters.iter().cloned().collect(),
                args: args,
                builder: builder.clone(),
                func: code,
            },
                                   final_ty)
        } else {
            return compile_err!("Can only construct DistFor with top-level NewBuilder: {}", builder.kind.name());
        }
    } else {
        return compile_err!("DistFor can only be constructed from a For: {}", inner_for.kind.name());
    }
}

pub fn distribute(expr: &mut Expr) {
    expr.transform_and_continue_res(&mut |ref mut e| {
        if let For { ref iters, ref builder, ref func } = e.kind {
            /* Got a top-level For. Transform into a distributed For. */
            let result = distfor_expr(&e)?;

            /* Don't continue to distribute once we distribute the outer loop. */
            return Ok((Some(result), false));
        }

        /* Didn't get a For, so continue to traverse until we find one. */
        return Ok((None, true));
    });
}

#[test]
fn distribute_test() {
    let code = "|x:vec[i32]|result(for(x, merger[i32,+], |b2,i2,e2|merge(b2, e2)))";
    let expr = parse_expr(code);
    let transformed_expr = distribute(&mut expr);

    print!("{}\n", transformed_expr.pretty_print());
}