//! Tests for external-artifact freshness on memories (Phase 4).

use engram_mcp::db::Database;
use engram_mcp::format::render_artifacts;
use engram_mcp::memory::{Memory, MemoryType};

fn setup_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("artifacts_test.db");
    let db = Database::open(&db_path).unwrap();
    db.get_or_create_project("art-proj", "art-proj").unwrap();
    (db, dir)
}

fn base_memory(id: &str) -> Memory {
    let now = chrono::Utc::now().timestamp();
    Memory {
        id: id.to_string(),
        project_id: "art-proj".to_string(),
        memory_type: MemoryType::Fact,
        content: "test content".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: None,
        merged_from: None,
        external_artifacts: None,
        pinned: false,
        global: false,
    }
}

// Test 1: store memory with artifacts → retrieve → artifacts preserved.
#[test]
fn store_with_artifacts_persists() {
    let (db, _dir) = setup_db();

    let artifacts = vec![
        "/tmp/x.png".to_string(),
        "https://example.com/y".to_string(),
    ];
    let memory = Memory {
        id: "art-mem-1".to_string(),
        external_artifacts: Some(artifacts.clone()),
        ..base_memory("art-mem-1")
    };
    db.store_memory(&memory).unwrap();

    let retrieved = db.get_memory("art-mem-1").unwrap().unwrap();
    let got = retrieved
        .external_artifacts
        .expect("artifacts should be present");
    assert_eq!(got, artifacts, "retrieved artifacts must match stored");
}

// Test 2: update semantics — replace, clear, preserve.
#[test]
fn update_replaces_artifacts() {
    let (db, _dir) = setup_db();

    // Start with [a].
    let memory = Memory {
        external_artifacts: Some(vec!["/tmp/a.txt".to_string()]),
        ..base_memory("art-mem-2")
    };
    db.store_memory(&memory).unwrap();

    // Replace with [b, c].
    let mut updated = db.get_memory("art-mem-2").unwrap().unwrap();
    updated.external_artifacts = Some(vec!["/tmp/b.txt".to_string(), "/tmp/c.txt".to_string()]);
    db.update_memory(&updated).unwrap();

    let r = db.get_memory("art-mem-2").unwrap().unwrap();
    assert_eq!(
        r.external_artifacts.as_deref(),
        Some(["/tmp/b.txt".to_string(), "/tmp/c.txt".to_string()].as_slice()),
        "update with [b, c] must replace"
    );

    // Clear with empty vec → stored as None.
    let mut cleared = db.get_memory("art-mem-2").unwrap().unwrap();
    cleared.external_artifacts = Some(vec![]); // empty = clear
    db.update_memory(&cleared).unwrap();

    let r = db.get_memory("art-mem-2").unwrap().unwrap();
    assert!(
        r.external_artifacts.is_none(),
        "update with empty vec must clear artifacts"
    );

    // Preserve: update content only, don't touch external_artifacts.
    // (Here external_artifacts is already None; we verify it stays None.)
    let mut preserved = db.get_memory("art-mem-2").unwrap().unwrap();
    preserved.content = "updated content".to_string();
    // Do NOT change external_artifacts — it should stay None.
    db.update_memory(&preserved).unwrap();

    let r = db.get_memory("art-mem-2").unwrap().unwrap();
    assert!(
        r.external_artifacts.is_none(),
        "omitting external_artifacts change must preserve existing (None) value"
    );
}

// Handler-level update semantics: when external_artifacts is omitted in the MCP
// update payload, the existing value must be preserved (not silently cleared).
#[test]
fn handler_update_preserves_artifacts_when_omitted() {
    use engram_mcp::embedding::EmbeddingService;
    use engram_mcp::tools::ToolHandler;
    use engram_mcp::tools::scoring::SearchMode;
    use serde_json::json;

    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("handler_update.db");
    let db = Database::open(&db_path).unwrap();
    let project_id = "art-handler".to_string();
    db.get_or_create_project(&project_id, &project_id).unwrap();
    let embedding = EmbeddingService::new().unwrap();
    let handler = ToolHandler::new(db, embedding, project_id, None, SearchMode::default());

    // Store a memory with artifacts via the public MCP path.
    let stored = handler
        .handle_tool(
            "memory_store",
            json!({
                "content": "remember to look at /tmp/a and /tmp/b",
                "type": "fact",
                "external_artifacts": ["/tmp/a", "/tmp/b"],
            }),
        )
        .expect("memory_store failed");
    let id = stored["id"].as_str().expect("id").to_string();

    // Update content only — do NOT pass external_artifacts.
    handler
        .handle_tool(
            "memory_update",
            json!({
                "id": id,
                "content": "new content unrelated to artifacts",
            }),
        )
        .expect("memory_update failed");

    // Retrieve and confirm artifacts are still present.
    let after = handler
        .handle_tool("memory_query", json!({"query": "", "limit": 10}))
        .expect("memory_query failed");
    let memories = after["memories"].as_array().expect("memories array");
    let target = memories
        .iter()
        .find(|m| m["memory"]["id"].as_str() == Some(id.as_str()))
        .expect("stored memory must be returned");
    let artifacts = target["memory"]["external_artifacts"]
        .as_array()
        .expect("artifacts must still be present after omit-update");
    let names: Vec<&str> = artifacts.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(
        names,
        vec!["/tmp/a", "/tmp/b"],
        "omitting external_artifacts in memory_update must preserve existing list"
    );
}

// Test 3: render_artifacts marks missing local paths and leaves present ones clean.
#[test]
fn format_marks_missing_local_path() {
    use std::io::Write;

    // Create a real tempfile so one path exists.
    let tmpdir = tempfile::tempdir().unwrap();
    let real_path = tmpdir.path().join("real_file.txt");
    let mut f = std::fs::File::create(&real_path).unwrap();
    writeln!(f, "content").unwrap();
    drop(f);

    let bogus_path = "/definitely/not/here/zzz_engram_test_12345.png".to_string();
    let real_path_str = real_path.to_string_lossy().to_string();

    let artifacts = Some(vec![bogus_path.clone(), real_path_str.clone()]);
    let rendered = render_artifacts(&artifacts);

    assert!(
        rendered.contains("[missing]"),
        "bogus path must be marked [missing]; got:\n{rendered}"
    );
    assert!(
        rendered.contains(&bogus_path),
        "bogus path must appear in output"
    );
    // The real path must NOT have [missing].
    let real_line = rendered
        .lines()
        .find(|l| l.contains(&real_path_str))
        .expect("real path must appear in rendered output");
    assert!(
        !real_line.contains("[missing]"),
        "existing file must not be marked [missing]; line: {real_line}"
    );
}

// Test 4: render_artifacts does NOT call exists() on non-local strings.
#[test]
fn format_skips_check_for_url() {
    let artifacts = Some(vec!["https://example.com/x".to_string()]);
    let rendered = render_artifacts(&artifacts);

    assert!(
        !rendered.contains("[missing]"),
        "URL should not be marked [missing]; got:\n{rendered}"
    );
    assert!(
        rendered.contains("https://example.com/x"),
        "URL must appear in rendered output"
    );
}

// Test 5: export/import round-trips external_artifacts.
#[test]
fn export_import_roundtrips_artifacts() {
    use engram_mcp::export::{create_export, validate_import};

    let (db, _dir) = setup_db();

    let artifacts = vec!["/tmp/export_a".to_string(), "/tmp/export_b".to_string()];
    let memory = Memory {
        id: "art-exp-1".to_string(),
        external_artifacts: Some(artifacts.clone()),
        ..base_memory("art-exp-1")
    };
    db.store_memory(&memory).unwrap();

    // Export.
    let memories = db.get_all_memories_for_project("art-proj").unwrap();
    let export_data = create_export(
        "art-proj",
        memories,
        vec![],
        None,
        std::collections::HashMap::new(),
        None,
    );
    assert!(validate_import(&export_data).is_ok());

    // Round-trip through JSON.
    let json_bytes = serde_json::to_string(&export_data).unwrap();
    let reimported: engram_mcp::export::ExportData = serde_json::from_str(&json_bytes).unwrap();

    let exported_mem = reimported
        .memories
        .iter()
        .find(|em| em.memory.id == "art-exp-1")
        .expect("exported memory must be present");

    assert_eq!(
        exported_mem.memory.external_artifacts.as_deref(),
        Some(artifacts.as_slice()),
        "external_artifacts must survive export/import JSON round-trip"
    );

    // Import into a fresh DB.
    let dir2 = tempfile::tempdir().unwrap();
    let db2 = Database::open(dir2.path().join("import.db")).unwrap();
    db2.get_or_create_project("art-proj", "art-proj").unwrap();

    for em in reimported.memories {
        if db2.get_memory(&em.memory.id).unwrap().is_none() {
            db2.store_memory(&em.memory).unwrap();
        }
    }

    let from_import = db2.get_memory("art-exp-1").unwrap().unwrap();
    assert_eq!(
        from_import.external_artifacts.as_deref(),
        Some(artifacts.as_slice()),
        "external_artifacts must survive full import into fresh DB"
    );
}
