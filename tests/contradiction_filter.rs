//! Integration tests for the type-aware contradiction filter.
//!
//! Verifies that:
//! - Handoff memories on the existing side are excluded from contradiction scanning.
//! - Cross-type pairs (e.g. Fact vs Decision) are not flagged as contradictions.
//! - Same-type pairs (Decision vs Decision) with near-identical content still trigger a warning.

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::tools::ToolHandler;
use engram_mcp::tools::scoring::SearchMode;
use serde_json::{Value, json};

fn setup() -> (ToolHandler, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let db = Database::open(&db_path).unwrap();
    let embedding = EmbeddingService::new().unwrap();
    let project_id = "contradiction-filter-test".to_string();
    db.get_or_create_project(&project_id, &project_id).unwrap();
    let handler = ToolHandler::new(db, embedding, project_id, None, SearchMode::default());
    (handler, dir)
}

fn store(handler: &ToolHandler, content: &str, memory_type: &str) -> Value {
    handler
        .handle_tool(
            "memory_store",
            json!({
                "content": content,
                "type": memory_type,
            }),
        )
        .unwrap_or_else(|e| panic!("memory_store failed: {e}"))
}

fn contradiction_count(result: &Value) -> usize {
    result["potential_contradictions"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0)
}

/// Store a Handoff, then store a Fact with near-identical wording.
/// The Handoff must not appear in the contradiction list.
#[test]
fn fact_does_not_contradict_handoff() {
    let (handler, _dir) = setup();

    store(
        &handler,
        "Decided on PostgreSQL for the database tier because of concurrent writes.",
        "handoff",
    );

    let result = store(
        &handler,
        "We chose PostgreSQL for the database because of concurrent write requirements.",
        "fact",
    );

    assert_eq!(
        contradiction_count(&result),
        0,
        "a Fact must not be flagged as contradicting a Handoff; got: {result}"
    );
}

/// Store a Fact, then store a Decision with overlapping content.
/// Cross-type pairs must not be flagged.
#[test]
fn decision_does_not_contradict_fact() {
    let (handler, _dir) = setup();

    store(&handler, "The auth service runs on port 8443.", "fact");

    let result = store(
        &handler,
        "Auth service should run on port 8443 with mTLS.",
        "decision",
    );

    assert_eq!(
        contradiction_count(&result),
        0,
        "a Decision must not be flagged as contradicting a Fact (cross-type); got: {result}"
    );
}

/// Store two Decisions that contradict each other on auth strategy.
/// "JWT tokens" vs "session cookies" for stateless session management score ~0.89 cosine
/// with the MRL-256 Decision prefix — above the 0.85 contradiction threshold but below the
/// 0.90 dedup threshold, so the second store triggers a warning rather than a merge.
#[test]
fn decision_still_contradicts_decision() {
    let (handler, _dir) = setup();

    store(
        &handler,
        "Authentication uses JWT tokens for stateless session management.",
        "decision",
    );

    let result = store(
        &handler,
        "Authentication uses session cookies for stateless session management.",
        "decision",
    );

    let count = contradiction_count(&result);
    let similarity = result["potential_contradictions"]
        .as_array()
        .and_then(|a| a.first())
        .and_then(|c| c["similarity"].as_f64());

    assert!(
        count >= 1,
        "two Decisions on the same auth topic with opposing choices must trigger at least \
         one contradiction warning (cosine >= 0.85); got {count} warnings \
         (measured similarity={similarity:?}); full result: {result}"
    );
}
