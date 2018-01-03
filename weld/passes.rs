use super::ast::*;
use super::error::*;

use super::transforms::loop_fusion;
use super::transforms::loop_fusion_2;
use super::transforms::inliner;
use super::transforms::size_inference;
use super::transforms::short_circuit;
use super::transforms::annotator;
use super::transforms::vectorizer;
use super::transforms::predication;
use super::transforms::measurement;
use super::transforms::unroller;

use super::expr_hash::*;

use std::collections::HashMap;

pub type PassFn = fn(&mut Expr<Type>);

/// A single IR to IR transformation.
pub struct Transformation {
    pub func: PassFn,
    pub experimental: bool,
}

/// Manually implement Clone for Transformation because it cannot be #derived due to the fn type inside it.
impl Clone for Transformation {
    fn clone(&self) -> Transformation {
        Transformation {
            func: self.func,
            experimental: self.experimental,
        }
    }
}

impl Transformation {
    pub fn new(func: PassFn) -> Transformation {
        Transformation {
            func: func,
            experimental: false,
        }
    }

    pub fn new_experimental(func: PassFn) -> Transformation {
        Transformation {
            func: func,
            experimental: true,
        }
    }
}

#[derive(Clone)]
pub struct Pass {
    transforms: Vec<Transformation>,
    pass_name: String,
    execute_once: bool,
}

impl Pass {
    pub fn new(transforms: Vec<Transformation>, pass_name: &'static str,
               execute_once: bool) -> Pass {
        return Pass {
            transforms: transforms,
            pass_name: String::from(pass_name),
            execute_once: execute_once,
        };
    }

    pub fn transform(&self, mut expr: &mut Expr<Type>, use_experimental: bool) -> WeldResult<()> {
        let mut continue_pass = true;
        let mut before = ExprHash::from(expr)?.value();
        while continue_pass {
            for transform in self.transforms.iter() {
                // Skip experimental transformations unless the flag is explicitly set.
                if transform.experimental && !use_experimental {
                    continue;
                }
                (transform.func)(&mut expr);
            }
            let after = ExprHash::from(expr)?.value();
            continue_pass = !(before == after);
            before = after;

            if self.execute_once {
                break;
            }
        }

        Ok(())
    }
    
    pub fn pass_name(&self) -> String {
        self.pass_name.clone()
    }
}

lazy_static! {
    pub static ref OPTIMIZATION_PASSES: HashMap<&'static str, Pass> = {
        let mut m = HashMap::new();
        m.insert("inline-apply",
                 Pass::new(vec![Transformation::new(inliner::inline_apply)], "inline-apply", false));
        m.insert("inline-let",
                 Pass::new(vec![Transformation::new(inliner::inline_let)], "inline-let", false));
        m.insert("inline-zip",
                 Pass::new(vec![Transformation::new(inliner::inline_zips)], "inline-zip", false));
        m.insert("loop-fusion",
                 Pass::new(vec![Transformation::new(loop_fusion::fuse_loops_vertical),
                                Transformation::new(loop_fusion_2::fuse_loops_2),
                                Transformation::new(loop_fusion_2::move_merge_before_let),
                                Transformation::new(inliner::inline_get_field),
                                Transformation::new(inliner::inline_let),
                                Transformation::new_experimental(loop_fusion_2::aggressive_inline_let),
                                Transformation::new_experimental(loop_fusion_2::merge_makestruct_loops)],
                 "loop-fusion", false));
        m.insert("unroll-static-loop",
                 Pass::new(vec![Transformation::new(unroller::unroll_static_loop)],
                 "unroll-static-loop", false));
        m.insert("infer-size",
                 Pass::new(vec![Transformation::new(size_inference::infer_size)],
                 "infer-size", false));
        m.insert("short-circuit-booleans",
                 Pass::new(vec![Transformation::new(short_circuit::short_circuit_booleans)],
                 "short-circuit-booleans", false));
	m.insert("inline-literals",		   
                 Pass::new(vec![Transformation::new(inliner::inline_negate),
			        Transformation::new(inliner::inline_cast)],
			   "inline-literals", false));		
        m.insert("inline-let-getfield",
                 Pass::new(vec![Transformation::new(inliner::inline_let_getfield)], "inline-let-getfield", false));
	m.insert("unroll-structs",	
		 Pass::new(vec![Transformation::new(inliner::unroll_structs)],	
			   "unroll-structs", false));	
	m.insert("measurement",
                 Pass::new(vec![Transformation::new(measurement::generate_measurement_branch)],
                           "measurement", true));
        m.insert("predicate",
                 Pass::new(vec![Transformation::new(predication::predicate_merge_expr),
                                Transformation::new(predication::predicate_simple_expr)],
                 "predicate", false));
        m.insert("vectorize",
                 Pass::new(vec![Transformation::new(vectorizer::vectorize)],
                 "vectorize", false));
        m.insert("fix-iterate",
                 Pass::new(vec![Transformation::new(annotator::force_iterate_parallel_fors)],
                 "fix-iterate", false));
        m
    };
}
