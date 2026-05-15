//! Integration tests for the single-handoff diversity backfill in `handoff_resume`.
//!
//! Verifies that when a branch has only one handoff in its chain, `handoff_resume`
//! supplements `linked_memories` with high-similarity Decision/Pattern/Debug memories.
//! Tests also verify that the backfill is NOT triggered for chains of length >= 2,
//! and that branch filtering is respected.

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::memory::HandoffSections;
use engram_mcp::tools::scoring::SearchMode;
use engram_mcp::tools::{ToolHandler, create_handoff};
use serde_json::json;

/// Open a single on-disk DB (via tempfile) that both the `ToolHandler` and the direct
/// `create_handoff` calls share.  Returns the handler, a second handle to the same DB
/// for direct calls, the embedding service for those direct calls, and the TempDir guard
/// that must be kept alive for the duration of the test.
fn setup(project_id: &str) -> (ToolHandler, Database, EmbeddingService, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let db_path = dir.path().join("test.db");

    // First handle: for direct create_handoff calls.
    let db_direct = Database::open(&db_path).expect("DB must open");
    db_direct
        .get_or_create_project(project_id, project_id)
        .expect("project creation must succeed");

    let embedding_direct = EmbeddingService::new().expect("embedding model must be available");

    // Second handle: for the ToolHandler.
    let db_handler = Database::open(&db_path).expect("DB must open");
    let embedding_handler = EmbeddingService::new().expect("embedding model must be available");
    let handler = ToolHandler::new(
        db_handler,
        embedding_handler,
        project_id.to_string(),
        None, // no current_branch; tests pass branch explicitly
        SearchMode::default(),
    );

    (handler, db_direct, embedding_direct, dir)
}

fn make_sections(summary: &str, continues_from: Option<String>) -> HandoffSections {
    HandoffSections {
        summary: summary.to_string(),
        decisions: vec!["Use token-based auth".to_string()],
        todos: vec!["Write tests".to_string()],
        blockers: vec![],
        mental_model: "Authentication layer".to_string(),
        next_steps: vec!["Deploy to staging".to_string()],
        notes: None,
        continues_from,
    }
}

fn store_memory(handler: &ToolHandler, content: &str, memory_type: &str, branch: &str) {
    handler
        .handle_tool(
            "memory_store",
            json!({
                "content": content,
                "type": memory_type,
                "branch": branch,
            }),
        )
        .unwrap_or_else(|e| panic!("memory_store failed: {e}"));
}

/// A single handoff on feat/x with 3 related Decisions (auth token vocabulary).
/// resume must backfill linked_memories from those Decisions; Fact must be excluded.
#[test]
fn single_handoff_backfills_with_decisions() {
    let project_id = "div-single-test";
    let (handler, db, embedding, _dir) = setup(project_id);

    // Create one handoff on feat/x about auth token refresh (auto_link=false so no
    // derived_from links exist initially).
    let sections = make_sections("auth token refresh logic for session management", None);
    create_handoff(
        &db,
        &embedding,
        project_id,
        Some("feat/x"),
        sections,
        0.85,
        true,
        false,
    )
    .expect("create handoff must succeed");

    // Store 3 Decision memories with overlapping auth token vocabulary.
    store_memory(
        &handler,
        "Decision: use JWT refresh tokens for auth session continuity",
        "decision",
        "feat/x",
    );
    store_memory(
        &handler,
        "Decision: access tokens expire after 15 minutes; refresh tokens last 7 days",
        "decision",
        "feat/x",
    );
    store_memory(
        &handler,
        "Decision: token rotation strategy prevents replay attacks in auth flow",
        "decision",
        "feat/x",
    );

    // Store 1 Fact about an unrelated topic; it must not appear in linked_memories.
    store_memory(
        &handler,
        "Fact: the office coffee machine is on the third floor",
        "fact",
        "feat/x",
    );

    // Resume with a query about auth tokens. max_sections=10 so the single handoff's
    // sections (at most 6) leave budget remaining for the backfill.
    let result = handler
        .handle_tool(
            "handoff_resume",
            json!({
                "branch": "feat/x",
                "query": "auth token",
                "max_sections": 10,
            }),
        )
        .expect("handoff_resume must succeed");

    let chain = result["chain"].as_array().expect("chain must be array");
    assert_eq!(chain.len(), 1, "chain must contain exactly 1 handoff");

    let linked = result["linked_memories"]
        .as_array()
        .expect("linked_memories must be array");
    assert!(
        !linked.is_empty(),
        "linked_memories must be non-empty after backfill; got empty list"
    );

    // All linked entries must be Decision type (Fact must be excluded).
    for entry in linked {
        let mem_type = entry["memory_type"]
            .as_str()
            .expect("memory_type must be a string");
        assert_eq!(
            mem_type, "decision",
            "linked_memories must only contain Decision entries; found: {mem_type}"
        );
    }
}

/// Two handoffs chained via continues_from. The backfill must NOT run.
/// Only original derived_from links (if any) survive in linked_memories.
#[test]
fn chain_of_two_skips_backfill() {
    let project_id = "div-chain-test";
    let (handler, db, embedding, _dir) = setup(project_id);

    // Store 3 Decision memories on feat/chain.
    store_memory(
        &handler,
        "Decision: use JWT refresh tokens for auth session continuity",
        "decision",
        "feat/chain",
    );
    store_memory(
        &handler,
        "Decision: access tokens expire after 15 minutes; refresh tokens last 7 days",
        "decision",
        "feat/chain",
    );
    store_memory(
        &handler,
        "Decision: token rotation strategy prevents replay attacks in auth flow",
        "decision",
        "feat/chain",
    );

    // Create handoff A on feat/chain (auto_link=false so derived_from list is empty).
    let sections_a = make_sections("auth token session management handoff A", None);
    let result_a = create_handoff(
        &db,
        &embedding,
        project_id,
        Some("feat/chain"),
        sections_a,
        0.85,
        true,
        false,
    )
    .expect("create handoff A must succeed");

    // Sleep to ensure B gets a strictly later created_at (Unix seconds precision).
    std::thread::sleep(std::time::Duration::from_millis(1100));

    // Create handoff B on feat/chain, continuing from A.
    let sections_b = make_sections(
        "auth token session management handoff B",
        Some(result_a.id.clone()),
    );
    create_handoff(
        &db,
        &embedding,
        project_id,
        Some("feat/chain"),
        sections_b,
        0.85,
        true,
        false,
    )
    .expect("create handoff B must succeed");

    // Resume: chain must be 2, backfill must NOT run.
    let result = handler
        .handle_tool(
            "handoff_resume",
            json!({
                "branch": "feat/chain",
                "query": "auth token",
                "max_sections": 5,
            }),
        )
        .expect("handoff_resume must succeed");

    let chain = result["chain"].as_array().expect("chain must be array");
    assert_eq!(chain.len(), 2, "chain must contain 2 handoffs");

    // linked_memories must contain only derived_from entries (none because auto_link=false).
    let linked = result["linked_memories"]
        .as_array()
        .expect("linked_memories must be array");
    assert!(
        linked.is_empty(),
        "linked_memories must be empty for chain-of-2 with no derived_from links; got {linked:?}"
    );
}

/// One handoff on feat/x; Decisions exist only on feat/y (different branch).
/// Backfill must not include cross-branch Decisions.
#[test]
fn backfill_respects_branch_filter() {
    let project_id = "div-branch-test";
    let (handler, db, embedding, _dir) = setup(project_id);

    // Create one handoff on feat/x (auto_link=false).
    let sections = make_sections("auth token refresh logic", None);
    create_handoff(
        &db,
        &embedding,
        project_id,
        Some("feat/x"),
        sections,
        0.85,
        true,
        false,
    )
    .expect("create handoff must succeed");

    // Store 2 Decisions on feat/y (different branch; must NOT appear in resume of feat/x).
    store_memory(
        &handler,
        "Decision: use JWT refresh tokens for auth session continuity on feat/y",
        "decision",
        "feat/y",
    );
    store_memory(
        &handler,
        "Decision: access tokens expire after 15 minutes on feat/y",
        "decision",
        "feat/y",
    );

    // Resume on feat/x. max_sections=10 so budget is available; backfill runs but
    // finds no candidates on feat/x (only feat/y decisions exist), yielding empty linked_memories.
    let result = handler
        .handle_tool(
            "handoff_resume",
            json!({
                "branch": "feat/x",
                "query": "auth token",
                "max_sections": 10,
            }),
        )
        .expect("handoff_resume must succeed");

    let chain = result["chain"].as_array().expect("chain must be array");
    assert_eq!(chain.len(), 1, "chain must contain 1 handoff");

    let linked = result["linked_memories"]
        .as_array()
        .expect("linked_memories must be array");
    assert!(
        linked.is_empty(),
        "linked_memories must be empty when cross-branch Decisions are excluded; got {linked:?}"
    );
}
