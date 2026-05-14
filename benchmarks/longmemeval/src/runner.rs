//! Query runner: wraps `ToolHandler` calls and extracts session ids from results.

use anyhow::Context;
use regex::Regex;
use serde_json::json;
use std::sync::OnceLock;

/// Compiled regex for the `session:<id>` tag pattern.
fn session_tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^session:(.+)$").expect("static regex must compile"))
}

/// Run a `memory_query` against the handler and return an ordered list of session ids.
///
/// The returned list preserves retrieval order (rank-1 session at index 0). Memories without
/// a `session:*` tag are silently skipped. If a memory has multiple `session:*` tags, the
/// first one wins.
///
/// # Arguments
/// * `handler` - The `ToolHandler` bound to the question's isolated database.
/// * `question` - The natural-language question text used as the search query.
/// * `limit` - Maximum number of memories to retrieve (passed through to `memory_query`).
pub async fn run_query(
    handler: &engram_mcp::tools::ToolHandler,
    question: &str,
    limit: usize,
) -> anyhow::Result<Vec<String>> {
    let result = handler
        .handle_tool(
            "memory_query",
            json!({
                "query": question,
                "limit": limit,
                "branch_mode": "all",
                "min_relevance": 0.0
            }),
        )
        .context("memory_query failed")?;

    let re = session_tag_re();
    let memories = result
        .get("memories")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut session_ids = Vec::with_capacity(memories.len());
    for entry in &memories {
        let tags = entry
            .pointer("/memory/tags")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let session_id = tags.iter().find_map(|t| {
            t.as_str()
                .and_then(|s| re.captures(s))
                .and_then(|caps| caps.get(1))
                .map(|m| m.as_str().to_owned())
        });

        match session_id {
            Some(sid) => session_ids.push(sid),
            None => {
                tracing::debug!(
                    memory_id = entry
                        .pointer("/memory/id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?"),
                    "memory has no session:* tag; skipping"
                );
            }
        }
    }

    Ok(session_ids)
}

/// Run a `memory_context` against the handler and return an ordered list of session ids.
///
/// The returned list preserves retrieval order (rank-1 session at index 0). Memories without
/// a `session:*` tag are silently skipped. If a memory has multiple `session:*` tags, the
/// first one wins.
///
/// `min_score` is set to 0.0 so mode-specific score scales do not filter results.
pub async fn run_context(
    handler: &engram_mcp::tools::ToolHandler,
    context: &str,
    limit: usize,
) -> anyhow::Result<Vec<String>> {
    let result = handler
        .handle_tool(
            "memory_context",
            json!({
                "context": context,
                "limit": limit,
                "branch_mode": "all",
                "min_score": 0.0
            }),
        )
        .context("memory_context failed")?;

    let re = session_tag_re();
    let memories = result
        .get("memories")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut session_ids = Vec::with_capacity(memories.len());
    for entry in &memories {
        // memory_context returns flat objects (no /memory/ wrapper unlike memory_query).
        let tags = entry
            .get("tags")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let session_id = tags.iter().find_map(|t| {
            t.as_str()
                .and_then(|s| re.captures(s))
                .and_then(|caps| caps.get(1))
                .map(|m| m.as_str().to_owned())
        });

        match session_id {
            Some(sid) => session_ids.push(sid),
            None => {
                tracing::debug!(
                    memory_id = entry.get("id").and_then(|v| v.as_str()).unwrap_or("?"),
                    "memory has no session:* tag; skipping"
                );
            }
        }
    }

    Ok(session_ids)
}
