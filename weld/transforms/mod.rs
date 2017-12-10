//! Common transformations on expressions.

pub mod loop_fusion;
pub mod loop_fusion_2;
pub mod uniquify;
pub mod inliner;
pub mod size_inference;
pub mod annotator;
pub mod predication;
pub mod vectorizer;
pub mod measurement;
pub mod short_circuit;
pub mod unroller;