use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use conf::ParsedConf;
use error::*;
use ast::constructors;
use util::SymbolGenerator;

use optimizer::transforms::distribute::code_util;
use optimizer::transforms::distribute::distribute::vec_info;
use optimizer::transforms::distribute::shard;

const DISPATCH_SYM: &str = "dispatch_one";

/// non_iter_args and iters should not have any names in common.
/// Mutates shard_idents and params.
/// Return value is a dummy Option.
pub fn append_args_to_shards(shard_idents: &mut Vec<Expr>,
                             params: &mut Vec<Parameter>,
                             non_iter_args: &Vec<Expr>) -> WeldResult<Option<Expr>> {
    for name in non_iter_args.iter() {
        if let Ident(ref sym) = name.kind {
            shard_idents.push((*name).clone());
            params.push( Parameter{ name: sym.clone(), ty: name.ty.clone() } );
        } else {
            return compile_err!("append_args_to_shards: non-Ident args not allowed in distribute");
        }
    }

    Ok(None)
}

/// Build a struct containing slices of iters and any additional arguments that will be inputs to the distributed subprogram.
/// Return a vec with the struct as the only element, representing a void* pointing to the struct,
/// and a vec of Parameters corresponding to the struct elements.
pub fn build_args_struct(iters: &Vec<vec_info>,
                         partition_loop_idx: &Expr,
                         partition_loop_elt: &Expr,
                         non_iter_args: &Vec<Expr>) -> WeldResult<(Expr, Vec<Parameter>)> {
    let (mut arg_idents, mut params) = shard::partitions_to_shards(iters, partition_loop_idx, partition_loop_elt).unwrap();

    // Mutates arg_idents and params.
    append_args_to_shards(&mut arg_idents, &mut params, non_iter_args).unwrap();

    // store args in a vec so that they are heap-allocated,
    // which allows us to allocate in the shared address space
    let ret_struct = constructors::makestruct_expr(arg_idents).unwrap();
    let ret_vec = constructors::makevector_expr(vec![ret_struct]).unwrap();

    Ok((ret_vec, params))
}

/// Generate a loop that will build the input arguments for a call to dispatch given {start, size} partition parameters.
/// Return the loop and a vector of the corresponding input parameters for the subprogram.
pub fn gen_args_loop(iters: &Vec<vec_info>,
                     total_len: &Expr,
                     non_iter_args: &Vec<Expr>,
                     nworkers_conf: &i32,
                     ctx: &Expr) -> WeldResult<(Expr, Vec<Parameter>)> {
    /* Generate the UDF call to create partitions. */
    let partition_udf = shard::gen_partition_call(total_len, nworkers_conf).unwrap();
    let (partition_name, partition_ident) = code_util::new_sym_and_ident("part", &partition_udf.ty, ctx);

    let partitions_iter = code_util::simple_iter(partition_ident.clone());

    let args_idx = code_util::gen_index(ctx);
    let args_elt = if let Vector(ref ty) = partition_udf.ty {
        code_util::gen_element(ty, ctx)
    } else {
        return compile_err!("gen_args_loop: partition list not a Vector");
    };

    /* Iterate over the partitions and build up the struct of input args corresponding to each one (iter slices + any remaining args). */
    let (args_pointer, input_params) = build_args_struct(&iters,
                                                         &constructors::ident_from_param(args_idx.clone()).unwrap(),
                                                         &constructors::ident_from_param(args_elt.clone()).unwrap(), &non_iter_args).unwrap();
    let args_appender =  constructors::newbuilder_expr(BuilderKind::Appender(Box::new(args_pointer.ty.clone())),
                                                None).unwrap();
    let args_builder = code_util::gen_builder(&args_appender.ty, ctx);
    let args_merge = constructors::merge_expr(
        constructors::ident_from_param(args_builder.clone()).unwrap(),
        args_pointer).unwrap();
    let args_func = constructors::lambda_expr(vec![args_builder, args_idx, args_elt], args_merge).unwrap();
    
    let args_loop = constructors::for_expr(vec![partitions_iter],
                                    args_appender,
                                    args_func, false).unwrap();

    /* Wrap the partition UDF in a Let so that it doesn't get called on every iteration. */
    let partition_let = if let Ident(ref sym) = partition_ident.kind {
        constructors::let_expr(sym.clone(), partition_udf, args_loop).unwrap()
    } else {
        return compile_err!("gen_create_dispatch_args: partitions not an Ident\n");
    };
    
    let args_res = constructors::result_expr(partition_let).unwrap(); /* materialize the entire list of nworkers args */

    Ok((args_res, input_params))
}

/// Generate the UDF to call the dispatch function, which takes a vector of arguments as input.
pub fn gen_dispatch(program_body: &Expr,
                    return_type: &Type,
                    index_expr: Expr,
                    args_expr: Expr) -> WeldResult<Expr> {
    let mut print_conf = PrettyPrintConfig::new();
    print_conf.show_types = true;
    print_conf.should_indent = false;
    let code = constructors::literal_expr(LiteralKind::StringLiteral(program_body.pretty_print_config(&print_conf)))?;
    let mut dispatch_expr = constructors::cudf_expr(DISPATCH_SYM.to_string(),
                                         vec![code,
                                              index_expr.clone(), /* param referencing iteration idx, aka worker ID */
                                              args_expr.clone()], /* param referencing arg */
                                         Vector(Box::new(return_type.clone()))).unwrap();
    dispatch_expr = constructors::lookup_expr(dispatch_expr, constructors::zero_i64_literal()?)?;
    let dispatch_with_id = constructors::makestruct_expr(vec![index_expr.clone(), dispatch_expr])?;
    
    Ok(dispatch_with_id)
}

/// Generate a loop to dispatch a subprogram corresponding to each set of args in args_iter.
/// Worker i will receive the ith set of args.
pub fn gen_dispatch_loop(args_iter: Iter, subprog: &Expr, return_type: &Type, ctx: &Expr) -> WeldResult<Expr> {
    /* generated dispatch call appends worker ID to the result, so it returns a struct of {worker ID, pointer to result data}. */
    let res_struct_ty = Struct(vec![Scalar(ScalarKind::I64),
                                    return_type.clone()]);

    let args_ty = if let Vector(ref ty) = args_iter.data.ty {
        (**ty).clone()
    } else {
        return compile_err!("gen_dispatch_loop: args are not a Vector");
    };
    
    let result_builder = constructors::newbuilder_expr(
        BuilderKind::Appender(Box::new(res_struct_ty.clone())), None)?;
    let result_params = code_util::new_loop_params(&result_builder.ty, &args_ty, ctx);

    let builder = result_params[0].clone();
    let idx     = result_params[1].clone();
    let elem    = result_params[2].clone();

    let func = gen_dispatch(subprog,
                            &return_type,
                            constructors::ident_from_param(idx).unwrap(),
                            constructors::ident_from_param(elem).unwrap()).unwrap();
    let merge = constructors::merge_expr(constructors::ident_from_param(builder).unwrap(),
                                  func).unwrap();
    let lambda = constructors::lambda_expr(result_params, merge).unwrap();

    let dispatch_loop = constructors::for_expr(vec![args_iter], result_builder,
                                        lambda, false).unwrap();

    Ok(dispatch_loop)
}