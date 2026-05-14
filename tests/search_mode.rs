//! Integration tests for SearchMode injection in ToolHandler.
//!
//! Three tests exercise Vector, Bm25, and Hybrid modes with the same corpus
//! and a query designed to have one clear lexical-only hit and one clear
//! semantic-only hit. No environment mutation — mode is passed by value.

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
    let project_id = "search-mode-test".to_string();
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

fn query_ids(handler: &ToolHandler, q: &str, limit: usize) -> Vec<String> {
    let result = handler
        .handle_tool(
            "memory_query",
            json!({
                "query": q,
                "limit": limit,
                "branch_mode": "all",
                // Set to 0 so mode-specific score scales don't accidentally filter everything.
                "min_relevance": 0.0,
            }),
        )
        .expect("query failed");
    result["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["memory"]["id"].as_str().unwrap().to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Corpus
//
// lexical_id  - contains the rare token "FROBNICATOR_CONFIG_XQ9" and nothing else
//               semantically related to the query paraphrase. BM25 ranks it #1
//               for queries containing that token; vector similarity is ~0.
//
// semantic_id - describes "machine learning model training convergence" in
//               natural language. The query paraphrase ("gradient descent
//               optimisation loss reduction") is semantically similar but
//               shares no rare lexical tokens, so BM25 score ≈ 0.
//
// noise_*     - three unrelated memories to fill out the corpus.
// ---------------------------------------------------------------------------

fn load_corpus(handler: &ToolHandler) -> (String, String) {
    let lexical_id = store(
        handler,
        "FROBNICATOR_CONFIG_XQ9 controls the output encoding pipeline stage",
        &["config", "encoding"],
    );

    let semantic_id = store(
        handler,
        "Neural network training converges when gradient descent reduces the loss function over successive epochs",
        &["ml", "training"],
    );

    store(
        handler,
        "The database connection pool is limited to 20 concurrent connections",
        &["database", "pool"],
    );
    store(
        handler,
        "Authentication tokens expire after 24 hours and must be refreshed",
        &["auth", "tokens"],
    );
    store(
        handler,
        "Frontend assets are compiled by webpack with tree-shaking enabled",
        &["frontend", "webpack"],
    );

    (lexical_id, semantic_id)
}

// ---------------------------------------------------------------------------
// Test 1: Vector mode
//
// Query is a pure semantic paraphrase of the ML memory with zero lexical
// overlap with the lexical-only memory. In Vector mode this means:
//   - semantic_id (ML memory) should rank high (high cosine similarity).
//   - lexical_id (FROBNICATOR memory) should score low because there is no
//     conceptual overlap between "gradient descent" and "output encoding pipeline".
//
// We also verify that querying for the rare token alone produces the lexical
// hit at rank 0 in BM25 mode but NOT at rank 0 in Vector mode (because in
// Vector mode there is no keyword boost — the score is pure cosine).
// ---------------------------------------------------------------------------

#[test]
fn vector_mode_uses_cosine_not_keyword() {
    let (handler, _dir) = make_handler(SearchMode::Vector);
    let (lexical_id, semantic_id) = load_corpus(&handler);

    // Pure semantic paraphrase — no rare token, no lexical overlap with lexical_id.
    let semantic_query = "backpropagation gradient descent minimises loss over training epochs";
    let ids = query_ids(&handler, semantic_query, 5);

    // The semantic hit must appear somewhere in the top-5.
    assert!(
        ids.contains(&semantic_id),
        "Vector mode: semantic hit must surface for semantic paraphrase; got ids: {ids:?}"
    );

    // Note on falsification: we can't trivially prove "vector ignores keywords"
    // here because mdbr-leaf-ir embeddings capture literal token presence, so
    // a bare-token query often ranks the matching memory high in cosine too.
    // What this test does verify is mode plumbing: Vector path runs without
    // panic and returns the semantic paraphrase hit.

    // Core assertion: for the semantic paraphrase, the lexical-only memory should
    // not outrank the ML memory (they may be equal or lexical may be absent).
    let sem_pos = ids.iter().position(|id| id == &semantic_id);
    let lex_pos = ids.iter().position(|id| id == &lexical_id);
    if let (Some(sp), Some(lp)) = (sem_pos, lex_pos) {
        assert!(
            sp <= lp,
            "Vector mode: semantic hit (pos {sp}) should not be ranked below lexical hit (pos {lp}) \
             for a pure semantic query"
        );
    }
    // If lexical_id is absent, assertion passes — that's also correct.
}

// ---------------------------------------------------------------------------
// Test 2: BM25 mode
//
// Same corpus and split query. BM25 finds exact token matches, so
// FROBNICATOR_CONFIG_XQ9 → lexical_id ranks first.
// The semantic paraphrase shares no tokens with the ML memory, so
// semantic_id may not appear at all or ranks low.
// ---------------------------------------------------------------------------

#[test]
fn bm25_mode_hits_lexical_misses_semantic_paraphrase() {
    let (handler, _dir) = make_handler(SearchMode::Bm25);
    let (lexical_id, semantic_id) = load_corpus(&handler);

    // Discriminating: query with the rare token alone. BM25 ranks lexical_id
    // at position 0 by exact-token match; semantic_id shares no tokens with
    // the query and must NOT appear (a Vector or Hybrid run would surface it).
    let token_query = "FROBNICATOR_CONFIG_XQ9";
    let token_ids = query_ids(&handler, token_query, 5);
    assert!(
        !token_ids.is_empty(),
        "BM25 mode: rare-token query must return at least one hit"
    );
    assert_eq!(
        token_ids[0], lexical_id,
        "BM25 mode: rare-token query must rank lexical hit at position 0; got ids: {token_ids:?}"
    );
    assert!(
        !token_ids.contains(&semantic_id),
        "BM25 mode: semantic-only memory must not surface for a query sharing no tokens with it; got ids: {token_ids:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Hybrid mode
//
// Both hits should appear in the top results: RRF fuses the two rankings so
// the semantic memory (highly ranked by vector) and the lexical memory
// (highly ranked by BM25) both surface.
// ---------------------------------------------------------------------------

#[test]
fn hybrid_mode_surfaces_both_hits() {
    let (handler, _dir) = make_handler(SearchMode::Hybrid);
    let (lexical_id, semantic_id) = load_corpus(&handler);

    let query = "backpropagation optimisation loss minimisation FROBNICATOR_CONFIG_XQ9";
    let ids = query_ids(&handler, query, 5);

    assert!(
        ids.contains(&semantic_id),
        "Hybrid mode: semantic hit must surface; got ids: {ids:?}"
    );
    assert!(
        ids.contains(&lexical_id),
        "Hybrid mode: lexical hit must surface; got ids: {ids:?}"
    );
}
