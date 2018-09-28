//! Utilities for common code generation tasks.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use exprs;
use util::SymbolGenerator;

pub fn new_sym(name: &str, ctx: &Expr<Type>) -> Symbol {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);

    sym_gen.new_symbol(name)
}

pub fn new_sym_and_ident(name: &str, ty: &Type, ctx: &Expr<Type>) -> (Symbol, Expr<Type>) {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);
    let new_name = sym_gen.new_symbol(name);
    let new_ident = exprs::ident_expr(new_name.clone(), ty.clone()).unwrap();

    (new_name, new_ident)
}

pub fn new_param_with_name(name: &str, ty: &Type, ctx: &Expr<Type>) -> Parameter<Type> {
    let mut sym_gen = SymbolGenerator::from_expression(ctx);
    let new_name = sym_gen.new_symbol(name);

    Parameter { name: new_name, ty: ty.clone() }
}

/// Use builder and element Params to create a simple merge(b, e) Expr.
pub fn simple_merge_expr(builder: &Parameter<Type>, element: &Parameter<Type>) -> Expr<Type> {
    let builder_ident = exprs::ident_from_param(builder.clone()).unwrap();
    let element_ident = exprs::ident_from_param(element.clone()).unwrap();

    exprs::merge_expr(builder_ident, element_ident).unwrap()
}

pub fn gen_builder(ty: &Type, ctx: &Expr<Type>) -> Parameter<Type> {
    new_param_with_name("b", ty, ctx)
}

pub fn gen_index(ctx: &Expr<Type>) -> Parameter<Type> {
    new_param_with_name("i", &Scalar(ScalarKind::I64), ctx)
}

pub fn gen_element(ty: &Type, ctx: &Expr<Type>) -> Parameter<Type> {
    new_param_with_name("e", ty, ctx)
}

/// Generate a set of typical For loop function parameters with names "b", "i", "e" for builder, index, element.
pub fn new_loop_params(builder_ty: &Type, element_ty: &Type, ctx: &Expr<Type>) -> Vec<Parameter<Type>> {
    let mut loop_params = vec![];
    loop_params.push(gen_builder(builder_ty, ctx));
    loop_params.push(gen_index(ctx));
    loop_params.push(gen_element(element_ty, ctx));

    loop_params
}

pub fn simple_iter(data: Expr<Type>) -> Iter<Type> {
    Iter { data: Box::new(data),
           start: None, end: None, stride: None,
           kind: IterKind::ScalarIter,
           strides: None, shape: None }
}