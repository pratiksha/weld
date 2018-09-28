//! Utility functions to sort a list of {k, v} pairs on keys, then strip thekeys.

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

fn gen_zero_keyfunc(element_ty: Type, ctx: &Expr<Type>) -> WeldResult<Expr<Type>> {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);
    
    let param = Parameter {
        name: sym_gen.new_symbol("e"),
        ty: element_ty.clone()
    };
    let element_ident = exprs::ident_from_param(param.clone())?;
    let lookup = exprs::getfield_expr(element_ident.clone(), 0)?;
    let lookup_func = exprs::lambda_expr(vec![param], lookup)?;
    Ok(lookup_func)
}

/// Sort {key, value} pairs by key, and return only the values.
pub fn gen_sorted_values_by_key(vec: &Expr<Type>, ctx: &Expr<Type>) -> WeldResult<Expr<Type>> {
    let vec_ty = if let Vector(ref ty) = vec.ty {
        (**ty).clone()
    } else {
        return compile_err!("Cannot sort a non-Vector: {}\n", print_type(&vec.ty));
    };
    
    let elem_ty = if let Vector(ref ty) = vec.ty {
        if let Struct(ref tys) = **ty {
            if tys.len() < 2 { return compile_err!("Cannot strip keys from length 1 struct\n"); };
            tys[1].clone()
        } else {
            return compile_err!("Cannot strip keys from non-struct\n");
        }
    } else {
        return compile_err!("Cannot sort a non-Vector: {}\n", print_type(&vec.ty));
    };

    /* sort on first element (worker index) */
    let sorted_results = exprs::sort_expr(vec.clone(),
                                          gen_zero_keyfunc(vec_ty.clone(), vec).unwrap())?;

    /* strip first element to leave only values */
    let appender = exprs::newbuilder_expr(BuilderKind::Appender(Box::new(elem_ty.clone())), None)?;
    let lookup_params = code_util::new_loop_params(&appender.ty, &vec_ty, ctx);

    let b_ident = exprs::ident_from_param(lookup_params[0].clone()).unwrap();
    let e_ident = exprs::ident_from_param(lookup_params[2].clone()).unwrap();

    print!("generating merge here\n");
    let merge = exprs::merge_expr(b_ident,
                                  exprs::getfield_expr(e_ident, 1)?)?;
    let merge_func = exprs::lambda_expr(lookup_params.clone(), merge)?;
    let element_iter = Iter { data: Box::new(sorted_results),
                              start: None, end: None, stride: None,
                              kind: IterKind::ScalarIter,
                              strides: None, shape: None };
    Ok(exprs::for_expr(vec![element_iter], appender, merge_func, false).unwrap())
}

