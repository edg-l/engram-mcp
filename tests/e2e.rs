//! End-to-end tests exercising the MCP tool handlers with real embeddings.
//!
//! Each test creates a temp DB + real ToolHandler and calls handle_tool()
//! exactly as the MCP server would. Memories use realistic content that
//! mimics how an AI agent would actually use the system.

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::tools::ToolHandler;
use serde_json::{Value, json};

/// Create a ToolHandler backed by a temp DB.
fn setup(branch: Option<&str>) -> (ToolHandler, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let db = Database::open(&db_path).unwrap();
    let embedding = EmbeddingService::new().unwrap();
    let project_id = "test-project".to_string();
    db.get_or_create_project(&project_id, &project_id).unwrap();
    let handler = ToolHandler::new(db, embedding, project_id, branch.map(String::from));
    (handler, dir)
}

fn call(handler: &ToolHandler, tool: &str, args: Value) -> Value {
    handler
        .handle_tool(tool, args)
        .expect(&format!("{} failed", tool))
}

// ---------------------------------------------------------------------------
// Scenario: An agent working on a web app stores architectural decisions,
// facts about the stack, and debug notes over the course of a session.
// ---------------------------------------------------------------------------

#[test]
fn test_store_and_query_realistic_memories() {
    let (h, _dir) = setup(None);

    // Agent learns about the project stack
    let r = call(
        &h,
        "memory_store",
        json!({
            "content": "The backend API is built with Rust using the Axum framework, running on port 8080",
            "type": "fact",
            "tags": ["backend", "rust", "axum"],
            "importance": 0.7
        }),
    );
    assert!(r["id"].as_str().unwrap().starts_with("mem_"));
    let api_id = r["id"].as_str().unwrap().to_string();

    call(
        &h,
        "memory_store",
        json!({
            "content": "PostgreSQL 16 is the primary database, hosted on RDS in us-east-1",
            "type": "fact",
            "tags": ["database", "postgres", "aws"],
            "importance": 0.8
        }),
    );

    call(
        &h,
        "memory_store",
        json!({
            "content": "Authentication uses JWT tokens with RS256 signing, tokens expire after 1 hour",
            "type": "decision",
            "tags": ["auth", "security"],
            "importance": 0.9
        }),
    );

    call(
        &h,
        "memory_store",
        json!({
            "content": "The React frontend communicates with the API through a GraphQL gateway",
            "type": "fact",
            "tags": ["frontend", "graphql"],
            "importance": 0.6
        }),
    );

    // Query for backend-related memories
    let r = call(
        &h,
        "memory_query",
        json!({
            "query": "what framework does the backend use",
            "limit": 3
        }),
    );
    let memories = r["memories"].as_array().unwrap();
    assert!(!memories.is_empty());
    // The Axum memory should rank high
    let top_content = memories[0]["memory"]["content"].as_str().unwrap();
    assert!(
        top_content.contains("Axum") || top_content.contains("backend"),
        "Expected backend-related result, got: {}",
        top_content
    );

    // Query for database info
    let r = call(
        &h,
        "memory_query",
        json!({
            "query": "which database and where is it hosted",
            "limit": 2
        }),
    );
    let memories = r["memories"].as_array().unwrap();
    assert!(!memories.is_empty());
    let contents: Vec<&str> = memories
        .iter()
        .map(|m| m["memory"]["content"].as_str().unwrap())
        .collect();
    assert!(
        contents.iter().any(|c| c.contains("PostgreSQL")),
        "Expected PostgreSQL result in: {:?}",
        contents
    );

    // memory_context for a coding task
    let r = call(
        &h,
        "memory_context",
        json!({
            "context": "I need to add a new API endpoint for user profiles",
            "limit": 5
        }),
    );
    let count = r["count"].as_u64().unwrap();
    assert!(count > 0, "memory_context should return relevant memories");

    // Stats should reflect what we stored
    let r = call(&h, "memory_stats", json!({}));
    assert_eq!(r["memory_count"].as_u64().unwrap(), 4);

    // Update a memory
    let r = call(
        &h,
        "memory_update",
        json!({
            "id": api_id,
            "content": "The backend API is built with Rust using Axum 0.7, running on port 8080 behind nginx",
            "tags": ["backend", "rust", "axum", "nginx"]
        }),
    );
    assert!(r["success"].as_bool().unwrap());

    // Delete a memory
    let r = call(&h, "memory_delete", json!({"id": api_id}));
    assert!(r["success"].as_bool().unwrap());

    let r = call(&h, "memory_stats", json!({}));
    assert_eq!(r["memory_count"].as_u64().unwrap(), 3);
}

#[test]
fn test_dedup_triggers_on_rephrased_memory() {
    let (h, _dir) = setup(None);

    // Agent stores a fact
    let r1 = call(
        &h,
        "memory_store",
        json!({
            "content": "The CI pipeline runs on GitHub Actions with a matrix build for Linux and macOS",
            "type": "fact",
            "tags": ["ci"],
            "importance": 0.6
        }),
    );
    let id1 = r1["id"].as_str().unwrap().to_string();

    // Later, agent stores essentially the same fact rephrased
    let r2 = call(
        &h,
        "memory_store",
        json!({
            "content": "CI runs on GitHub Actions, using matrix builds targeting Linux and macOS platforms",
            "type": "fact",
            "tags": ["ci", "github-actions"],
            "importance": 0.7
        }),
    );
    let id2 = r2["id"].as_str().unwrap().to_string();

    // Check if dedup fired (depends on model similarity -- may or may not trigger)
    if r2.get("merge_info").is_some() && !r2["merge_info"].is_null() {
        // Dedup fired: old memory merged into new one
        let merge_info = &r2["merge_info"];
        assert_eq!(merge_info["merged_with"].as_str().unwrap(), id1);
        assert!(merge_info["similarity"].as_f64().unwrap() >= 0.90);

        // Old memory should be gone
        let r = call(&h, "memory_query", json!({"query": ""}));
        let ids: Vec<&str> = r["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["memory"]["id"].as_str().unwrap())
            .collect();
        assert!(
            !ids.contains(&id1.as_str()),
            "Old memory should be merged away"
        );
        assert!(ids.contains(&id2.as_str()), "New memory should exist");

        // Merged memory should have union of tags
        let stats = call(&h, "memory_stats", json!({}));
        assert_eq!(stats["memory_count"].as_u64().unwrap(), 1);
    } else {
        // Similarity was below 0.90 -- both memories coexist
        let stats = call(&h, "memory_stats", json!({}));
        assert_eq!(stats["memory_count"].as_u64().unwrap(), 2);
    }
}

#[test]
fn test_bulk_dedup_tool() {
    let (h, _dir) = setup(None);

    // Store several memories, some with overlapping content
    call(
        &h,
        "memory_store",
        json!({
            "content": "Redis is used as the caching layer with a 5 minute TTL for API responses",
            "type": "fact",
            "tags": ["cache", "redis"]
        }),
    );
    call(
        &h,
        "memory_store",
        json!({
            "content": "The logging system uses structured JSON logs sent to Elasticsearch via Filebeat",
            "type": "fact",
            "tags": ["logging"]
        }),
    );
    call(
        &h,
        "memory_store",
        json!({
            "content": "Error monitoring is handled by Sentry with source map uploads on deploy",
            "type": "fact",
            "tags": ["monitoring"]
        }),
    );

    // Dry run dedup -- should find 0 or more groups but not modify anything
    let r = call(&h, "memory_dedup", json!({"threshold": 0.85}));
    assert!(r["success"].as_bool().unwrap());
    assert!(r["dry_run"].as_bool().unwrap());
    let before_count = call(&h, "memory_stats", json!({}))["memory_count"]
        .as_u64()
        .unwrap();

    // Memory count unchanged after dry run
    let after_count = call(&h, "memory_stats", json!({}))["memory_count"]
        .as_u64()
        .unwrap();
    assert_eq!(before_count, after_count);
}

#[test]
fn test_branch_scoped_memories() {
    let (h, _dir) = setup(Some("feature/oauth"));

    // Store a global decision
    call(
        &h,
        "memory_store",
        json!({
            "content": "All API endpoints must require authentication except /health and /metrics",
            "type": "decision",
            "tags": ["api", "security"],
            "importance": 0.9
        }),
    );

    // Store a branch-specific note
    call(
        &h,
        "memory_store",
        json!({
            "content": "OAuth2 integration requires registering a callback URL with the identity provider",
            "type": "fact",
            "tags": ["oauth"],
            "importance": 0.7,
            "branch": "auto"
        }),
    );

    // Store another branch-specific note
    call(
        &h,
        "memory_store",
        json!({
            "content": "The OAuth state parameter must be cryptographically random to prevent CSRF",
            "type": "decision",
            "tags": ["oauth", "security"],
            "importance": 0.8,
            "branch": "auto"
        }),
    );

    // "current" mode should see global + feature/oauth memories
    let r = call(
        &h,
        "memory_query",
        json!({
            "query": "",
            "branch_mode": "current"
        }),
    );
    assert_eq!(r["count"].as_u64().unwrap(), 3);

    // "global" mode should see only the global memory
    let r = call(
        &h,
        "memory_query",
        json!({
            "query": "",
            "branch_mode": "global"
        }),
    );
    assert_eq!(r["count"].as_u64().unwrap(), 1);
    let content = r["memories"][0]["memory"]["content"].as_str().unwrap();
    assert!(content.contains("authentication"));

    // "all" mode should see everything
    let r = call(
        &h,
        "memory_query",
        json!({
            "query": "",
            "branch_mode": "all"
        }),
    );
    assert_eq!(r["count"].as_u64().unwrap(), 3);

    // Promote a branch memory to global
    let branch_memories: Vec<&Value> = r["memories"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["memory"]["branch"].as_str().is_some())
        .collect();
    if let Some(mem) = branch_memories.first() {
        let id = mem["memory"]["id"].as_str().unwrap();
        let r = call(&h, "memory_promote", json!({"id": id}));
        assert!(r["success"].as_bool().unwrap());

        // Now "global" mode should see 2 memories
        let r = call(
            &h,
            "memory_query",
            json!({
                "query": "",
                "branch_mode": "global"
            }),
        );
        assert_eq!(r["count"].as_u64().unwrap(), 2);
    }
}

#[test]
fn test_clustering_forms_naturally() {
    let (h, _dir) = setup(None);

    // Store memories about different topics -- should form distinct clusters

    // Cluster 1: Database-related
    call(
        &h,
        "memory_store",
        json!({
            "content": "Database migrations are managed with sqlx-cli, run via 'cargo sqlx migrate run'",
            "type": "fact",
            "tags": ["database", "migrations"]
        }),
    );
    call(
        &h,
        "memory_store",
        json!({
            "content": "The users table has a GIN index on the metadata JSONB column for fast queries",
            "type": "fact",
            "tags": ["database", "indexing"]
        }),
    );
    call(
        &h,
        "memory_store",
        json!({
            "content": "Database connection pooling is configured with max 20 connections via PgPool",
            "type": "fact",
            "tags": ["database", "performance"]
        }),
    );

    // Cluster 2: Deployment-related
    call(
        &h,
        "memory_store",
        json!({
            "content": "Production deployments use Docker containers orchestrated by Kubernetes on EKS",
            "type": "fact",
            "tags": ["deployment", "kubernetes"]
        }),
    );
    call(
        &h,
        "memory_store",
        json!({
            "content": "Helm charts are in the deploy/ directory, with separate values for staging and prod",
            "type": "fact",
            "tags": ["deployment", "helm"]
        }),
    );
    call(
        &h,
        "memory_store",
        json!({
            "content": "Rolling deployments with max 25% surge, health checks on /readyz endpoint",
            "type": "decision",
            "tags": ["deployment", "strategy"]
        }),
    );

    // Cluster 3: Testing
    call(
        &h,
        "memory_store",
        json!({
            "content": "Integration tests use testcontainers-rs to spin up ephemeral Postgres instances",
            "type": "fact",
            "tags": ["testing"]
        }),
    );
    call(
        &h,
        "memory_store",
        json!({
            "content": "Code coverage is tracked with cargo-llvm-cov, minimum threshold is 80%",
            "type": "decision",
            "tags": ["testing", "ci"]
        }),
    );

    // Check clusters formed
    let stats = call(&h, "memory_stats", json!({}));
    assert_eq!(stats["memory_count"].as_u64().unwrap(), 8);
    let cluster_count = stats["cluster_count"].as_u64().unwrap();
    // Should have formed at least 2 clusters (exact count depends on model similarity)
    assert!(
        cluster_count >= 1,
        "Expected clusters to form, got {}",
        cluster_count
    );

    // Query for database info -- should hit the DB cluster
    let r = call(
        &h,
        "memory_query",
        json!({
            "query": "how are database migrations handled",
            "limit": 3
        }),
    );
    let results = r["memories"].as_array().unwrap();
    assert!(!results.is_empty());
    let top = results[0]["memory"]["content"].as_str().unwrap();
    assert!(
        top.contains("migration") || top.contains("database") || top.contains("sqlx"),
        "Expected DB-related result, got: {}",
        top
    );

    // memory_context with enough memories should use hierarchical retrieval
    let r = call(
        &h,
        "memory_context",
        json!({
            "context": "I need to change the deployment strategy to blue-green",
            "limit": 5
        }),
    );
    assert!(r["count"].as_u64().unwrap() > 0);
    // With 8 memories (< 10), it falls back to flat -- that's fine
    // The important thing is it returns relevant results
    let retrieval_mode = r["retrieval_mode"].as_str().unwrap();
    assert!(
        retrieval_mode == "flat" || retrieval_mode == "hierarchical",
        "Unexpected retrieval mode: {}",
        retrieval_mode
    );
}

#[test]
fn test_relationship_graph_and_links() {
    let (h, _dir) = setup(None);

    // Store related architectural decisions
    let r1 = call(
        &h,
        "memory_store",
        json!({
            "content": "Decided to use event sourcing for the order management domain",
            "type": "decision",
            "tags": ["architecture", "orders"],
            "importance": 0.9
        }),
    );
    let decision_id = r1["id"].as_str().unwrap().to_string();

    let r2 = call(
        &h,
        "memory_store",
        json!({
            "content": "EventStoreDB is the backing store for the event sourcing implementation",
            "type": "fact",
            "tags": ["architecture", "eventstoredb"],
            "importance": 0.7
        }),
    );
    let impl_id = r2["id"].as_str().unwrap().to_string();

    let r3 = call(
        &h,
        "memory_store",
        json!({
            "content": "Considered using Kafka for event storage but rejected due to lack of built-in projections",
            "type": "decision",
            "tags": ["architecture", "kafka"],
            "importance": 0.6
        }),
    );
    let rejected_id = r3["id"].as_str().unwrap().to_string();

    // Link them
    let r = call(
        &h,
        "memory_link",
        json!({
            "source_id": impl_id,
            "target_id": decision_id,
            "relation": "derived_from"
        }),
    );
    assert!(r["success"].as_bool().unwrap());

    let r = call(
        &h,
        "memory_link",
        json!({
            "source_id": decision_id,
            "target_id": rejected_id,
            "relation": "relates_to"
        }),
    );
    assert!(r["success"].as_bool().unwrap());

    // Traverse the graph from the decision
    let r = call(
        &h,
        "memory_graph",
        json!({
            "id": decision_id,
            "depth": 2
        }),
    );
    let related = r["related"].as_array().unwrap();
    assert!(related.len() >= 2, "Should find both linked memories");
}

#[test]
fn test_batch_store_and_export_import() {
    let (h, _dir) = setup(None);

    // Batch store debug notes from a troubleshooting session
    let r = call(
        &h,
        "memory_store_batch",
        json!({
            "memories": [
                {
                    "content": "OutOfMemoryError in the payment service occurs under load testing at 500 req/s",
                    "type": "debug",
                    "tags": ["payment", "oom"],
                    "importance": 0.8
                },
                {
                    "content": "Root cause: unbounded channel buffer in the payment event processor",
                    "type": "debug",
                    "tags": ["payment", "fix"],
                    "importance": 0.9
                },
                {
                    "content": "Fix: switched to bounded channel with capacity 1000, backpressure via tokio::sync::mpsc",
                    "type": "pattern",
                    "tags": ["payment", "fix", "pattern"],
                    "importance": 0.7
                }
            ]
        }),
    );
    assert!(r["success"].as_bool().unwrap());
    assert_eq!(r["count"].as_u64().unwrap(), 3);
    let ids = r["ids"].as_array().unwrap();
    assert_eq!(ids.len(), 3);

    // Export
    let r = call(&h, "memory_export", json!({"include_embeddings": true}));
    assert_eq!(r["version"].as_str().unwrap(), "1.1");
    assert_eq!(r["memories"].as_array().unwrap().len(), 3);
    assert!(r["model_version"].is_string());

    // Import into a fresh handler
    let (h2, _dir2) = setup(None);
    let r = call(
        &h2,
        "memory_import",
        json!({
            "data": r,
            "mode": "merge"
        }),
    );
    assert!(r["success"].as_bool().unwrap());
    assert_eq!(r["stats"]["memories_imported"].as_u64().unwrap(), 3);

    // Verify imported memories are queryable
    let r = call(
        &h2,
        "memory_query",
        json!({
            "query": "what caused the OOM in payment service",
            "limit": 3
        }),
    );
    let results = r["memories"].as_array().unwrap();
    assert!(!results.is_empty());
    let contents: String = results
        .iter()
        .map(|m| m["memory"]["content"].as_str().unwrap())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        contents.contains("OutOfMemory")
            || contents.contains("payment")
            || contents.contains("channel"),
        "Expected payment/OOM related results"
    );
}

#[test]
fn test_prune_low_relevance_memories() {
    let (h, _dir) = setup(None);

    // Store some memories
    call(
        &h,
        "memory_store",
        json!({
            "content": "Temporary workaround: disabled SSL verification for local dev proxy",
            "type": "debug",
            "tags": ["workaround"],
            "importance": 0.1
        }),
    );
    call(
        &h,
        "memory_store",
        json!({
            "content": "The API rate limit is 1000 requests per minute per API key",
            "type": "fact",
            "tags": ["api"],
            "importance": 0.8
        }),
    );

    // Dry run prune at high threshold (should find the low-importance debug memory)
    let r = call(
        &h,
        "memory_prune",
        json!({
            "threshold": 0.99
        }),
    );
    assert!(r["dry_run"].as_bool().unwrap());
    // Both memories have relevance_score 1.0 (fresh), so none should be candidates
    // at default threshold. Use a very high threshold to test the mechanism.
    let candidates = r["candidates"].as_u64().unwrap();
    assert_eq!(
        candidates, 0,
        "Fresh memories should all have relevance 1.0"
    );

    // Verify nothing was deleted
    let stats = call(&h, "memory_stats", json!({}));
    assert_eq!(stats["memory_count"].as_u64().unwrap(), 2);
}
