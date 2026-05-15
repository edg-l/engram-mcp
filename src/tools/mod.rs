//! MCP tool handlers for the Engram memory server.

// `pub use` re-exports below are part of the public API consumed by integration tests,
// benches, and external users; the lib crate itself doesn't reference them, which trips
// `unused_imports` under `-D warnings` during lib-test/bin compilation.
#![allow(unused_imports)]

mod handler;
mod handoff;
pub mod schemas;
pub mod scoring;
pub mod store;

#[cfg(test)]
mod test_utils;

pub use handler::{ToolHandler, parse_search_mode};
pub use handoff::{create_handoff, resume_handoff, score_handoff_sections, search_handoffs};
pub use schemas::{
    MemoryUpdateInput, ToolProfile, dedup_threshold, get_tool_definitions, get_tool_definitions_for,
};
pub use scoring::{SearchMode, compute_hybrid_score, compute_tag_boost};
pub use store::{StoreOutcome, store_with_dedup};
