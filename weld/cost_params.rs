//! Memory parameters to use in the cost model.
//! Constants are floats in order to reduce the need for casting when computing costs.

/* Selectivity conversion constant for Annotations */
pub const SELECTIVITY_UNITS: f64 = 10000.0;

/* ************************* System/CPU Parameters  ************************* */

/*
Instructions (Linux):

    Cores: lscpu
    Clock Frequency: lscpu
*/

/// Number of CPU cores.
pub const CORES: f64 = 4.0;

/// 2.9 GHz
pub const CLOCK_FREQUENCY: f64 = 2E9;

/* ************************* Memory Parameters  ************************* */

/*
Instructions (Linux):

    L1, L2, L3 Cache Size: lscpu, divide by line size
    Clock Frequency: lscpu
    Cache Line Size: cat /proc/cpuinfo, read the clflush size field
*/

/// Memory throughput at different levels of the hierarchy (index 0 is L1 cache, etc.).
/// Units are bytes/sec.
pub const L1_THROUGHPUT: f64 = 529E9;
pub const L2_THROUGHPUT: f64 = 350E9;
pub const L3_THROUGHPUT: f64 = 120E9;
pub const MEM_THROUGHPUT: f64 = 55E9;

/// Cache sizes in *blocks/lines*
pub const L1_SIZE: f64 = 500.0;
pub const L2_SIZE: f64 = 4000.0;
pub const L3_SIZE: f64 = 240000.0;

/// Cache line size in bytes.
pub const CACHE_LINE_SIZE: f64 = 64.0;

pub const BITS_PER_BYTE: u32 = 8;
pub fn mem_cost_sequential(elt_size_bits: u32) -> f64 {
    let bytes_per_elt: u32 = elt_size_bits / BITS_PER_BYTE;
    let sec_per_byte: f64 = 1.0 / L1_THROUGHPUT;
    let sec_per_elt: f64 = sec_per_byte * (bytes_per_elt as f64);
    sec_per_elt
}

/// Memory Latencies
pub const L1_LATENCY: f64 = 1.0;
pub const L2_LATENCY: f64 = 7.0;
pub const L3_LATENCY: f64 = 19.0;
pub const MEM_LATENCY: f64 = 36.0;

pub fn mem_cost_random() -> f64 {
    L1_LATENCY
}

/* ************************* Instruction Parameters ************************* */

/// constant to scale vectorized costs by
pub const VEC_CONSTANT: f64 = 0.8;

/// Latency of a standard binary op (+ - / * >= etc.)
pub const BINOP_LATENCY: f64 = 1.0;
/// Latency of a vectorized binary op (+ - / * >= etc.)
pub const BINOP_VEC_LATENCY: f64 = 4.0;

/// Latency of a standard unary op (square / square root)
pub const UNOP_LATENCY: f64 = 4.0;
pub const UNOP_VEC_LATENCY: f64 = 8.0;

/// Latency of an atomic add with no contention.
pub const ATOMICADD_LATENCY: f64 = 20.0;
/// Penalty of contention for an atomic add.
pub const ATOMICADD_PENALTY: f64 = 2000.0;

/// Given a constant condition, The number of branches fall a certain way for the
/// branch predictor to predict it correctly subsequently.
pub const BRANCHPRED_PREDICTABLE_IT_DIST: f64 = 10.0;

/// The max latency of a branch misprediction.
pub const BRANCHPRED_LATENCY: f64 = 20.0;
/// The latency of executing a branching instruction.
pub const BRANCH_LATENCY: f64 = 1.0;

/// The branch misprediction penalty. The penalty is an expected value; when
/// selectivity is closer to 0.5, the penalty is closer to the full
/// BRANCHPRED_LATENCY.
/// TODO compute correctly from i32!
pub fn branch_mispredict_penalty(selectivity: i32) -> f64 {
    let sel_float = (selectivity as f64) / SELECTIVITY_UNITS;

    /* Returns a parabola whose maximum is at at s=0.5 */
    -2.0 * BRANCHPRED_LATENCY * (sel_float - 0.5).abs() + BRANCHPRED_LATENCY + 3.0
}

/* ************************* Hashtable Parameters ************************* */

pub const INSERT_LATENCY_SINGLE: f64 = 4.5e-07 * CLOCK_FREQUENCY;
pub const INSERT_LATENCY_CONTEND: f64 = 1.1e-06 * CLOCK_FREQUENCY; /* 4 threads */
pub const LOOKUP_LATENCY: f64 = 5.5e-07 * CLOCK_FREQUENCY;
pub const LOOKUP_LOCAL_LATENCY: f64 = 5.5e-08 * CLOCK_FREQUENCY;
