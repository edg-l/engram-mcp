//! Integration test for the handoff create → resume → search flow.
//!
//! Uses a real `EmbeddingService` (model cached locally) and an in-memory SQLite database.
//! Covers Task 3B.7 of the handoff feature plan.

use engram_mcp::db::{Database, encode_section_embeddings};
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::export::{ExportedMemory, HandoffSidecar, create_export};
use engram_mcp::memory::{HandoffSections, Memory, MemoryType};
use engram_mcp::tools::{create_handoff, resume_handoff, search_handoffs};

/// Build a minimal HandoffSections for a session on feat/x.
fn make_sections(
    summary: &str,
    blockers: Vec<String>,
    continues_from: Option<String>,
) -> HandoffSections {
    HandoffSections {
        summary: summary.to_string(),
        decisions: vec!["Use SQLite for storage".to_string()],
        todos: vec!["Add integration tests".to_string()],
        blockers,
        mental_model: "Layered service architecture".to_string(),
        next_steps: vec!["Deploy to staging".to_string()],
        notes: None,
        continues_from,
    }
}

/// In-memory DB, create handoff A on feat/x, create handoff B continuing from A on same branch.
/// Run resume_handoff on feat/x; assert chain length 2 and top sections include B's content.
/// Then run search_handoffs with a query matching A's blockers section; assert A appears.
#[test]
fn handoff_flow_create_resume_search() {
    let db = Database::open_in_memory().expect("in-memory DB must open");
    let project_id = "handoff-flow-proj";
    db.get_or_create_project(project_id, "Handoff Flow Test")
        .expect("project creation must succeed");

    let embedding = EmbeddingService::new().expect("embedding model must be available");

    // Create handoff A on feat/x.
    let sections_a = make_sections(
        "Session A: set up database schema",
        vec!["Migration tool not yet installed".to_string()],
        None,
    );
    let result_a = create_handoff(
        &db,
        &embedding,
        project_id,
        Some("feat/x"),
        sections_a,
        0.85,
        true,
        false,
    )
    .expect("create handoff A must succeed");

    let id_a = result_a.id.clone();

    // Sleep 1.1 s to ensure B gets a strictly later created_at (stored as Unix seconds).
    // Without this, both handoffs share the same second and DB ordering is non-deterministic.
    std::thread::sleep(std::time::Duration::from_millis(1100));

    // Create handoff B on feat/x, continuing from A.
    let sections_b = make_sections(
        "Session B: implemented embedding pipeline",
        vec![],
        Some(id_a.clone()),
    );
    let result_b = create_handoff(
        &db,
        &embedding,
        project_id,
        Some("feat/x"),
        sections_b,
        0.85,
        true,
        false,
    )
    .expect("create handoff B must succeed");

    let id_b = result_b.id.clone();

    // Resume on feat/x: should see a chain of length 2 (A then B, oldest first).
    let resume_result = resume_handoff(
        &db,
        &embedding,
        project_id,
        Some("feat/x"),
        Some("embedding pipeline"),
        5,
        false,
    )
    .expect("resume must succeed");

    assert_eq!(
        resume_result.chain.len(),
        2,
        "chain must contain both handoffs A and B, got: {:?}",
        resume_result.chain
    );
    assert_eq!(
        resume_result.chain[0], id_a,
        "oldest handoff A must be first in chain"
    );
    assert_eq!(
        resume_result.chain[1], id_b,
        "newest handoff B must be last in chain"
    );

    // Top sections should include content from B (highest-relevance to query about embedding pipeline).
    assert!(
        !resume_result.top_sections.is_empty(),
        "top_sections must not be empty"
    );
    let has_b_section = resume_result
        .top_sections
        .iter()
        .any(|s| s.handoff_id == id_b);
    assert!(
        has_b_section,
        "at least one top section should come from handoff B"
    );

    // Search with a query matching A's blockers section.
    let search_result = search_handoffs(
        &db,
        &embedding,
        project_id,
        "migration tool not installed",
        None, // all branches
        10,
        Some(&["blockers".to_string()]),
    )
    .expect("search must succeed");

    assert!(
        !search_result.matches.is_empty(),
        "search must return at least one match"
    );
    let has_a_blocker = search_result
        .matches
        .iter()
        .any(|m| m.handoff_id == id_a && m.section_name == "blockers");
    assert!(
        has_a_blocker,
        "handoff A's blockers section must appear in search results"
    );
}

/// Export a handoff to JSON, import into a fresh DB, verify sections and section_embeddings
/// byte length are preserved.  Covers Task 5.4 of the handoff feature plan.
#[test]
fn export_import_preserves_handoff_sections() {
    let db = Database::open_in_memory().expect("in-memory DB must open");
    let project_id = "export-import-proj";
    db.get_or_create_project(project_id, "Export Import Test")
        .expect("project creation must succeed");

    let embedding = EmbeddingService::new().expect("embedding model must be available");

    let sections = HandoffSections {
        summary: "Implemented the export round-trip".to_string(),
        decisions: vec!["Additive export schema".to_string()],
        todos: vec!["Write more tests".to_string()],
        blockers: vec!["Waiting on CI".to_string()],
        mental_model: "Sidecar table holds per-section embeddings".to_string(),
        next_steps: vec!["Run full QA".to_string()],
        notes: Some("Remember to bump schema_version".to_string()),
        continues_from: None,
    };

    let result = create_handoff(
        &db,
        &embedding,
        project_id,
        Some("feat/export-test"),
        sections.clone(),
        0.85,
        true,
        false,
    )
    .expect("create handoff must succeed");

    let handoff_id = result.id.clone();

    // Fetch the sidecar to obtain the stored section embeddings byte length.
    let (original_sections, original_section_vecs) = db
        .get_handoff_sections(&handoff_id)
        .expect("get_handoff_sections must succeed")
        .expect("sidecar must exist after create_handoff");

    let original_section_count = original_section_vecs.len();
    let expected_bytes_len = original_section_count * 256 * 4;

    // Build export data via create_export (the same path used by memory_export).
    let memories = db
        .get_all_memories_for_project(project_id)
        .expect("get all memories must succeed");
    let relationships = db
        .get_all_relationships_for_project(project_id)
        .expect("get all relationships must succeed");

    let key_strings: Vec<String> = original_section_vecs
        .iter()
        .map(|(k, _)| k.clone())
        .collect();
    let keys: Vec<&str> = key_strings.iter().map(|s| s.as_str()).collect();
    let vecs: Vec<Vec<f32>> = original_section_vecs.into_iter().map(|(_, v)| v).collect();
    let (keys_str, bytes) = encode_section_embeddings(&keys, &vecs);

    let mut sidecars = std::collections::HashMap::new();
    sidecars.insert(
        handoff_id.clone(),
        HandoffSidecar {
            sections: original_sections.clone(),
            keys: keys_str.clone(),
            bytes: bytes.clone(),
        },
    );

    let export_data = create_export(
        project_id,
        memories,
        relationships,
        None,
        sidecars,
        Some("test-model".to_string()),
    );

    // Verify the export JSON contains the sidecar fields for the handoff.
    let exported_memory = export_data
        .memories
        .iter()
        .find(|m| m.memory.id == handoff_id)
        .expect("exported_memory must contain the handoff");
    assert!(
        exported_memory.sections.is_some(),
        "exported handoff must include sections"
    );
    assert!(
        exported_memory.section_embedding_keys.is_some(),
        "exported handoff must include section_embedding_keys"
    );
    assert!(
        exported_memory.section_embeddings.is_some(),
        "exported handoff must include section_embeddings"
    );

    // Import into a fresh DB.
    let db2 = Database::open_in_memory().expect("fresh in-memory DB must open");
    db2.get_or_create_project(project_id, "Export Import Test 2")
        .expect("project creation must succeed");

    for exported in &export_data.memories {
        let mut memory = exported.memory.clone();
        memory.project_id = project_id.to_string();
        db2.store_memory(&memory)
            .expect("store_memory must succeed");

        // Re-generate embedding (no full-content embedding in export for this test).
        let emb = embedding
            .embed_memory(memory.memory_type, &memory.content)
            .expect("embed must succeed");
        db2.store_embedding(&memory.id, &emb, embedding.model_version())
            .expect("store_embedding must succeed");

        // Import handoff sidecar.
        if let (Some(s), Some(k), Some(eb)) = (
            &exported.sections,
            &exported.section_embedding_keys,
            &exported.section_embeddings,
        ) {
            use engram_mcp::export::decode_section_embedding_bytes;
            let raw = decode_section_embedding_bytes(eb).expect("decode must succeed");
            db2.insert_handoff_sections(&memory.id, s, k, &raw)
                .expect("insert_handoff_sections must succeed");
        }
    }

    // Verify the imported sidecar matches.
    let (imported_sections, imported_vecs) = db2
        .get_handoff_sections(&handoff_id)
        .expect("get_handoff_sections on imported DB must succeed")
        .expect("sidecar must exist after import");

    assert_eq!(
        imported_sections.summary, original_sections.summary,
        "summary must be preserved"
    );
    assert_eq!(
        imported_sections.decisions, original_sections.decisions,
        "decisions must be preserved"
    );
    assert_eq!(
        imported_sections.todos, original_sections.todos,
        "todos must be preserved"
    );
    assert_eq!(
        imported_vecs.len(),
        original_section_count,
        "section count must match"
    );

    // Verify byte length of re-encoded embeddings matches the original.
    let imported_key_strings: Vec<String> = imported_vecs.iter().map(|(k, _)| k.clone()).collect();
    let imported_keys: Vec<&str> = imported_key_strings.iter().map(|s| s.as_str()).collect();
    let imported_section_vecs: Vec<Vec<f32>> = imported_vecs.into_iter().map(|(_, v)| v).collect();
    let (_, imported_bytes) = encode_section_embeddings(&imported_keys, &imported_section_vecs);
    assert_eq!(
        imported_bytes.len(),
        expected_bytes_len,
        "re-encoded section_embeddings byte length must match original"
    );
}

/// Import a Handoff memory from an old-format export (no sidecar fields).
/// The memory row must be stored successfully and `get_handoff_sections` must return `None`.
/// Covers the backward-compat path added in the handoff feature (Task 5.x).
#[test]
fn import_old_export_without_sidecar_fields() {
    let db = Database::open_in_memory().expect("in-memory DB must open");
    let project_id = "old-export-compat-proj";
    db.get_or_create_project(project_id, "Old Export Compat Test")
        .expect("project creation must succeed");

    // Build a minimal ExportedMemory representing an old-format Handoff export:
    // sections, section_embedding_keys, section_embeddings are all None.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let memory = Memory {
        id: "old-handoff-id-001".to_string(),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Handoff,
        content: "## Summary\nOld session handoff without sidecar.\n".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.8,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: Some("feat/old".to_string()),
        merged_from: None,
        external_artifacts: None,
        pinned: true,
        global: false,
    };

    let exported = ExportedMemory {
        memory: memory.clone(),
        embedding: None,
        sections: None,
        section_embedding_keys: None,
        section_embeddings: None,
    };

    // Store the memory row (simulating what memory_import does for the old format).
    db.store_memory(&exported.memory)
        .expect("store_memory must succeed for old-format handoff");

    // No sidecar insert — that's the backward-compat path.
    assert!(
        exported.sections.is_none()
            && exported.section_embedding_keys.is_none()
            && exported.section_embeddings.is_none(),
        "old-format export must have no sidecar fields"
    );

    // The memory row must exist.
    let fetched = db
        .get_memory(&memory.id)
        .expect("get_memory must not error")
        .expect("memory row must exist after import");
    assert_eq!(fetched.id, memory.id);
    assert_eq!(fetched.memory_type, MemoryType::Handoff);

    // get_handoff_sections must return None (no sidecar was inserted).
    let sidecar = db
        .get_handoff_sections(&memory.id)
        .expect("get_handoff_sections must not error");
    assert!(
        sidecar.is_none(),
        "old-format import must leave sidecar absent (got Some)"
    );
}
