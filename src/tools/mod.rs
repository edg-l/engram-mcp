//! MCP tool handlers for the Engram memory server.
//!
//! All items in this module are used by the MCP server binary (main.rs).
//! The dead_code warnings appear because the CLI binary doesn't use these.
#![allow(dead_code)]
#![allow(unused_imports)]

mod handler;
mod handoff;
mod schemas;
mod scoring;

#[cfg(test)]
mod test_utils;

pub use handler::*;
pub use handoff::*;
pub use schemas::*;
pub use scoring::{compute_hybrid_score, compute_tag_boost};
