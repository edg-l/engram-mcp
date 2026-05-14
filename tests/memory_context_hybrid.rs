//! Integration tests for SearchMode plumbing through `memory_context`.
//!
//! Three tests mirror the structure of `tests/search_mode.rs` but exercise the
//! `memory_context` tool path (flat fallback — no clusters in a small corpus).
//! Constructor injection via `ToolHandler::new(..., SearchMode::X)` is used;
//! no environment mutation.

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::tools::ToolHandler;
use engram_mcp::tools::scoring::SearchMode;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn make_handler(mode: SearchMode) -> (ToolHandler, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let db = Database::open(&db_path).unwrap();
    let embedding = EmbeddingService::new().unwrap();
    let project_id = "ctx-mode-test".to_string();
    db.get_or_create_project(&project_id, &project_id).unwrap();
    let handler = ToolHandler::new(db, embedding, project_id, None, mode);
    (handler, dir)
}

fn store(handler: &ToolHandler, content: &str, tags: &[&str]) -> String {
    let tags_json: Vec<Value> = tags.iter().map(|t| json!(t)).collect();
    let result = handler
        .handle_tool(
            "memory_store",
            json!({
                "content": content,
                "type": "fact",
                "tags": tags_json,
                "importance": 0.7,
            }),
        )
        .expect("store failed");
    result["id"].as_str().unwrap().to_string()
}

fn context_ids(handler: &ToolHandler, ctx: &str, limit: usize) -> Vec<String> {
    let result = handler
        .handle_tool(
            "memory_context",
            json!({
                "context": ctx,
                "limit": limit,
                "branch_mode": "all",
                "min_score": 0.0,
                // Force flat path (no clusters in a small corpus, but be explicit).
                "hierarchical": false,
            }),
        )
        .expect("memory_context failed");
    result["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Corpus (same split as search_mode.rs)
//
// lexical_id  - contains rare token "ZXQFROBNICATE_CTX99" and nothing else
//               semantically related to gradient descent. BM25 ranks it first.
//
// semantic_id - describes neural network training / loss minimisation.
//               Cosine similarity is high for gradient-descent paraphrases.
//
// noise_*     - three unrelated memories.
// ---------------------------------------------------------------------------

fn load_corpus(handler: &ToolHandler) -> (String, String) {
    let lexical_id = store(
        handler,
        "ZXQFROBNICATE_CTX99 drives the downstream pipeline configuration",
        &["config", "pipeline"],
    );

    let semantic_id = store(
        handler,
        "Neural network training converges when gradient descent reduces loss over successive epochs",
        &["ml", "training"],
    );

    store(
        handler,
        "Database connection pool capped at 20 concurrent clients",
        &["database", "pool"],
    );
    store(
        handler,
        "Authentication tokens expire after 24 hours and need renewal",
        &["auth", "tokens"],
    );
    store(
        handler,
        "Frontend bundle compiled by webpack with tree shaking enabled",
        &["frontend", "webpack"],
    );

    (lexical_id, semantic_id)
}

// ---------------------------------------------------------------------------
// Test 1: Vector mode — semantic hit surfaces for a semantic paraphrase.
//
// Note: mdbr-leaf-ir embeddings capture literal token presence, so a hard
// assertion that "lexical_id is absent" is not always reliable — the rare
// token may share subword features with the query. What we verify is:
//   1. The call completes without panic.
//   2. semantic_id appears somewhere in the top-5.
//   3. semantic_id is not ranked below lexical_id (if both present).
// ---------------------------------------------------------------------------

#[test]
fn vector_mode_context_uses_cosine() {
    let (handler, _dir) = make_handler(SearchMode::Vector);
    let (lexical_id, semantic_id) = load_corpus(&handler);

    let semantic_query =
        "backpropagation gradient descent minimises the loss function over training epochs";
    let ids = context_ids(&handler, semantic_query, 5);

    assert!(
        ids.contains(&semantic_id),
        "Vector mode context: semantic hit must surface for semantic paraphrase; got: {ids:?}"
    );

    // semantic_id should not rank below lexical_id.
    let sem_pos = ids.iter().position(|id| id == &semantic_id);
    let lex_pos = ids.iter().position(|id| id == &lexical_id);
    if let (Some(sp), Some(lp)) = (sem_pos, lex_pos) {
        assert!(
            sp <= lp,
            "Vector mode: semantic hit (pos {sp}) should not be ranked below lexical hit (pos {lp})"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: BM25 mode — lexical-only query surfaces lexical_id at position 0.
//
// The rare token "ZXQFROBNICATE_CTX99" has no semantic relationship to
// semantic_id. BM25 ranks lexical_id first; semantic_id must not appear.
// ---------------------------------------------------------------------------

#[test]
fn bm25_mode_context_hits_lexical_only() {
    let (handler, _dir) = make_handler(SearchMode::Bm25);
    let (lexical_id, semantic_id) = load_corpus(&handler);

    let token_query = "ZXQFROBNICATE_CTX99";
    let ids = context_ids(&handler, token_query, 5);

    assert!(
        !ids.is_empty(),
        "BM25 mode context: rare-token query must return at least one hit"
    );
    assert_eq!(
        ids[0], lexical_id,
        "BM25 mode context: rare token must rank lexical_id at position 0; got: {ids:?}"
    );
    assert!(
        !ids.contains(&semantic_id),
        "BM25 mode context: semantic-only memory must not surface for a pure lexical query; got: {ids:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Hybrid mode — combined query surfaces both lexical and semantic hits.
// ---------------------------------------------------------------------------

#[test]
fn hybrid_mode_context_surfaces_both() {
    let (handler, _dir) = make_handler(SearchMode::Hybrid);
    let (lexical_id, semantic_id) = load_corpus(&handler);

    let combined_query = "backpropagation optimisation loss minimisation ZXQFROBNICATE_CTX99";
    let ids = context_ids(&handler, combined_query, 5);

    assert!(
        ids.contains(&semantic_id),
        "Hybrid mode context: semantic hit must surface; got: {ids:?}"
    );
    assert!(
        ids.contains(&lexical_id),
        "Hybrid mode context: lexical hit must surface; got: {ids:?}"
    );
}
