//! Utility functions to sort a list of {k, v} pairs on keys, then strip the keys.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use conf::ParsedConf;
use error::*;
use ast::constructors;
use util::SymbolGenerator;

use optimizer::transforms::distribute::code_util;

/// generate a comparator to sort on the first element
fn gen_zero_cmp_func(element_ty: Type, ctx: &Expr) -> WeldResult<Expr> {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);
    
    let left_param = Parameter {
        name: sym_gen.new_symbol("e1"),
        ty: element_ty.clone()
    };
    let right_param = Parameter {
        name: sym_gen.new_symbol("e2"),
        ty: element_ty.clone()
    };
    
    let left_ident  = constructors::ident_from_param(left_param.clone())?;
    let right_ident = constructors::ident_from_param(right_param.clone())?;

    let left_lookup  = constructors::getfield_expr(left_ident.clone(), 0)?;
    let right_lookup = constructors::getfield_expr(right_ident.clone(), 0)?;

    let comparator = constructors::default_compare_expr(left_lookup, right_lookup)?;

    let lookup_func = constructors::lambda_expr(vec![left_param, right_param], comparator)?;
    Ok(lookup_func)
}

/// Sort {key, value} pairs by key, and return only the values.
pub fn gen_sorted_values_by_key(vec: &Expr, ctx: &Expr) -> WeldResult<Expr> {
    let vec_ty = if let Vector(ref ty) = vec.ty {
        (**ty).clone()
    } else {
        return compile_err!("Cannot sort a non-Vector: {}\n", &vec.ty);
    };
    
    let elem_ty = if let Vector(ref ty) = vec.ty {
        if let Struct(ref tys) = **ty {
            if tys.len() < 2 { return compile_err!("Cannot strip keys from length 1 struct\n"); };
            tys[1].clone()
        } else {
            return compile_err!("Cannot strip keys from non-struct\n");
        }
    } else {
        return compile_err!("Cannot sort a non-Vector: {}\n", &vec.ty);
    };

    /* sort on first element (worker index) */
    let sorted_results = constructors::sort_expr(vec.clone(),
                                                 gen_zero_cmp_func(vec_ty.clone(), vec).unwrap())?;

    /* strip first element to leave only values */
    let appender = constructors::newbuilder_expr(BuilderKind::Appender(Box::new(elem_ty.clone())), None)?;
    let lookup_params = code_util::new_loop_params(&appender.ty, &vec_ty, ctx);

    let b_ident = constructors::ident_from_param(lookup_params[0].clone()).unwrap();
    let e_ident = constructors::ident_from_param(lookup_params[2].clone()).unwrap();

    print!("generating merge here\n");
    let merge = constructors::merge_expr(b_ident,
                                  constructors::getfield_expr(e_ident, 1)?)?;
    let merge_func = constructors::lambda_expr(lookup_params.clone(), merge)?;
    let element_iter = Iter { data: Box::new(sorted_results),
                              start: None, end: None, stride: None,
                              kind: IterKind::ScalarIter,
                              strides: None, shape: None };
    Ok(constructors::for_expr(vec![element_iter], appender, merge_func, false).unwrap())
}

