/// Verifies that the direct db-tier insert path used by the longmemeval
/// harness does NOT auto-dedup or merge near-identical memories.
///
/// The ToolHandler::memory_store path fires find_duplicates (>= 0.90
/// cosine) and then silently merges them. This test bypasses that handler
/// and asserts both rows survive, confirming the harness ingest is safe.
#[tokio::test]
async fn ingest_does_not_auto_dedup_near_identical_turn_pairs() {
    use engram_mcp::db::Database;
    use engram_mcp::embedding::EmbeddingService;
    use engram_mcp::memory::{Memory, MemoryType};

    let dir = tempfile::TempDir::new().expect("tempdir");
    let db_path = dir.path().join("test.db");
    let db = Database::open(&db_path).expect("open db");
    db.get_or_create_project("lme-bypass-test", "lme-bypass-test")
        .expect("create project");

    let embedding = EmbeddingService::new().expect("embedding service");

    let now = chrono::Utc::now().timestamp();

    // Two near-identical memories — only the trailing character differs.
    // Cosine similarity will exceed 0.95.
    let contents = [
        "USER: What is the capital of France?\nASSISTANT: The capital of France is Paris.",
        "USER: What is the capital of France?\nASSISTANT: The capital of France is Paris!",
    ];

    let mut ids = Vec::new();
    for content in &contents {
        let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
        let memory = Memory {
            id: id.clone(),
            project_id: "lme-bypass-test".to_string(),
            memory_type: MemoryType::Fact,
            content: content.to_string(),
            summary: None,
            tags: vec!["session:s1".to_string()],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
            pinned: false,
            global: false,
        };

        let vec = embedding
            .embed_memory(MemoryType::Fact, content)
            .expect("embed");

        // Direct db-tier insert — same path as harness ingest.rs.
        db.store_memory(&memory).expect("store memory");
        db.store_embedding(&id, &vec, embedding.model_version())
            .expect("store embedding");

        ids.push(id);
    }

    // Both memories must survive — no merge happened.
    let all = db
        .get_all_memories_for_project("lme-bypass-test")
        .expect("list memories");

    assert_eq!(
        all.len(),
        2,
        "expected 2 memories, got {}; direct insert path must not auto-dedup",
        all.len()
    );

    // Confirm the IDs we inserted are both present.
    let stored_ids: std::collections::HashSet<String> = all.into_iter().map(|m| m.id).collect();
    for expected_id in &ids {
        assert!(
            stored_ids.contains(expected_id),
            "memory {} was not found after direct insert",
            expected_id
        );
    }
}
