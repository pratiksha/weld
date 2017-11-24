use super::ast::*;
use super::error::*;

use super::transforms::loop_fusion;
use super::transforms::inliner;
use super::transforms::size_inference;
use super::transforms::short_circuit;
use super::transforms::annotator;
use super::transforms::vectorizer;
use super::transforms::predication;
use super::transforms::measurement;

use super::expr_hash::*;

use std::collections::HashMap;

pub struct Pass {
    transforms: Vec<fn(&mut Expr<Type>)>,
    pass_name: String,
    execute_once: bool,
}

/// Manually implement Clone for Pass because it cannot be #derived due to the fn type inside it.
impl Clone for Pass {
    fn clone(&self) -> Pass {
        Pass {
            transforms: self.transforms.iter().map(|p| *p).collect::<Vec<_>>(),
            pass_name: self.pass_name.clone(),
            execute_once: self.execute_once.clone(),
        }
    }
}

impl Pass {
    pub fn new(transforms: Vec<fn(&mut Expr<Type>)>, pass_name: &'static str,
               execute_once: bool) -> Pass {
        return Pass {
            transforms: transforms,
            pass_name: String::from(pass_name),
            execute_once: execute_once,
        };
    }

    pub fn transform(&self, mut expr: &mut Expr<Type>) -> WeldResult<()> {
        let mut continue_pass = true;
        let mut before = ExprHash::from(expr)?.value();
        while continue_pass {
            for transform in &self.transforms {
                transform(&mut expr);
            }
            let after = ExprHash::from(expr)?.value();
            continue_pass = !(before == after);
            before = after;

            if self.execute_once {
                continue_pass = false;
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
                 Pass::new(vec![inliner::inline_apply], "inline-apply", false));
        m.insert("inline-let",
                 Pass::new(vec![inliner::inline_let], "inline-let", false));
        m.insert("inline-zip",
                 Pass::new(vec![inliner::inline_zips], "inline-zip", false));
        m.insert("loop-fusion",
                 Pass::new(vec![loop_fusion::fuse_loops_horizontal,
                                loop_fusion::fuse_loops_vertical,
                                inliner::inline_get_field],
                 "loop-fusion", false));
        m.insert("infer-size",
                 Pass::new(vec![size_inference::infer_size],
                 "infer-size", false));
        m.insert("short-circuit-booleans",
                 Pass::new(vec![short_circuit::short_circuit_booleans],
                 "short-circuit-booleans"));
        m.insert("predicate",
                 Pass::new(vec![predication::predicate],
                           "predicate", false));
        m.insert("measurement",
                 Pass::new(vec![measurement::generate_measurement_branch],
                 "measurement", true));
        m.insert("vectorize",
                 Pass::new(vec![vectorizer::vectorize],
                 "vectorize", false));
        m.insert("fix-iterate",
                 Pass::new(vec![annotator::force_iterate_parallel_fors],
                 "fix-iterate", false));

        m
    };
}
