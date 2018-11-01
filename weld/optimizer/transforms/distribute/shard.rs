//! Functions to generate shards from vectors and vector Slices from shards.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use conf::ParsedConf;
use error::*;
use ast::constructors;
use util::SymbolGenerator;

use optimizer::transforms::distribute::distribute::vec_info;

const PARTITION_SYM: &str = "partition_data";

/// @param  vec_ident: the vector to slice.
/// @param  partition_ident: the {start, size} parameters for the shard.
/// @return an Expr representing the slice vec_ident[start:start+size] or Err.
fn partition_to_slice(vec_ident: &Expr,
                      partition_ident: &Expr) -> WeldResult<Expr> {
    constructors::slice_expr(vec_ident.clone(), /* slice each input iter. */
                      constructors::getfield_expr(partition_ident.clone(), 0)?, /* slice start index */
                      constructors::getfield_expr(partition_ident.clone(), 1)?  /* slice size */)
}

/// Call a CUDF to split total_len into nworkers_conf shards.
pub fn gen_partition_call(total_len: &Expr, nworkers_conf: &i32) -> WeldResult<Expr> {
    let nworkers = constructors::literal_expr(LiteralKind::I32Literal(nworkers_conf.clone())).unwrap();
    let increment = constructors::one_i64_literal().unwrap();
    let partition_type = Struct(vec![Scalar(ScalarKind::I64),
                                     Scalar(ScalarKind::I64)]);    
    let partition_expr = constructors::cudf_expr(PARTITION_SYM.to_string(),
                                      vec![total_len.clone(),
                                           nworkers,
                                           increment],
                                          Vector(Box::new(partition_type)))?;

    Ok(partition_expr)
}

/// Generate a function to convert a partition into a vector slice.
/// If the vector is already sharded and on the cluster, ignore the partition and pass in the corresponding shard.
/// Otherwise, return the slice vi.ident[start:start+size].
fn vec_info_to_shard(vi: &vec_info,
                     partition_loop_idx: &Expr,
                     partition_loop_elt: &Expr) -> WeldResult<Expr> {
    if vi.is_sharded {
        constructors::lookup_expr(vi.original_ident.clone(), partition_loop_idx.clone())
    } else {
        partition_to_slice(&vi.original_ident, &partition_loop_elt)
    }
}

/// Convert the input {start, size} partition into Weld Slices.
/// If the iter is already sharded, just add the corresponding element of the data.
/// Note that if any shard lengths are not identical across iters,
/// the distributed program will fail at runtime.
/// If the vector is already sharded on the cluster, the number of shards must be equal to the number of workers.
/// Return the slice expressions and the corresponding list of input parameters for the subprogram.
pub fn partitions_to_shards(iters: &Vec<vec_info>,
                            partition_loop_idx: &Expr,
                            partition_loop_elt: &Expr) -> WeldResult<(Vec<Expr>,
                                                                            Vec<Parameter>)> {
    let mut shards = vec![];
    let mut params = vec![];

    for vi in iters.iter() {
        if let Ident(ref sym) = vi.ident.kind {
            shards.push(vec_info_to_shard(vi, partition_loop_idx, partition_loop_elt).unwrap());
            params.push( Parameter{ name: sym.clone(), ty: vi.ident.ty.clone() } );
        } else {
            return compile_err!("partitions_to_slices: non-Ident not allowed in distribute iter");
        }
    }

    Ok((shards, params))
}