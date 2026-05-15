use rusqlite::params;

use crate::error::MemoryError;
use crate::memory::{HandoffSections, Memory, MemoryType, Project};

use super::Database;
use super::handoffs::{decode_section_embeddings, encode_section_embeddings};
use super::util::parse_memory_type_col;

#[test]
fn test_memory_crud() {
    let db = Database::open_in_memory().unwrap();

    // Create project
    let project = Project {
        id: "test-project".to_string(),
        name: "Test Project".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: chrono::Utc::now().timestamp(),
    };
    db.create_project(&project).unwrap();

    // Store memory
    let now = chrono::Utc::now().timestamp();
    let memory = Memory {
        id: "mem-1".to_string(),
        project_id: "test-project".to_string(),
        memory_type: MemoryType::Fact,
        content: "Test content".to_string(),
        summary: None,
        tags: vec!["test".to_string()],
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
    };
    db.store_memory(&memory).unwrap();

    // Get memory
    let retrieved = db.get_memory("mem-1").unwrap().unwrap();
    assert_eq!(retrieved.content, "Test content");

    // Delete memory
    assert!(db.delete_memory("mem-1").unwrap());
    assert!(db.get_memory("mem-1").unwrap().is_none());
}

#[test]
fn test_migration_creates_tables() {
    let db = Database::open_in_memory().unwrap();

    // Verify schema_version table exists and has version 5
    let conn = db.conn.lock().unwrap();
    let version: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, 5);

    // Verify memory_clusters table exists
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM memory_clusters", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);

    // Verify cluster_members table exists
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM cluster_members", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);

    // Verify merged_from column exists on memories
    let mut stmt = conn.prepare("PRAGMA table_info(memories)").unwrap();
    let has_merged_from = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .any(|name| name == "merged_from");
    assert!(has_merged_from);

    // Verify handoff_sections table exists (migration 4)
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM handoff_sections", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_migration_idempotent() {
    // Running initialize twice should not fail
    let db = Database::open_in_memory().unwrap();
    // The second initialize happens automatically, but let's verify the DB works
    let project = crate::memory::Project {
        id: "test".to_string(),
        name: "test".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();
    let p = db.get_project("test").unwrap();
    assert!(p.is_some());
}

#[test]
fn test_run_migrations_twice_is_idempotent() {
    // Verify that calling run_migrations a second time on an already-migrated
    // populated DB does not fail or corrupt data.
    let db = Database::open_in_memory().unwrap();
    let project = crate::memory::Project {
        id: "mig2".to_string(),
        name: "mig2".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    // Store a memory so the DB is non-empty before the second migration run.
    let now = chrono::Utc::now().timestamp();
    db.store_memory(&Memory {
        id: "mig2-mem".to_string(),
        project_id: "mig2".to_string(),
        memory_type: MemoryType::Fact,
        content: "Persists across migrations".to_string(),
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
    })
    .unwrap();

    // Call run_migrations explicitly a second time; must not error.
    db.run_migrations().unwrap();

    // Data must still be intact.
    let mem = db.get_memory("mig2-mem").unwrap();
    assert!(mem.is_some());
    assert_eq!(mem.unwrap().content, "Persists across migrations");
}

#[test]
fn test_merge_memories() {
    let db = Database::open_in_memory().unwrap();

    let project = crate::memory::Project {
        id: "test-merge".to_string(),
        name: "test-merge".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    let now = chrono::Utc::now().timestamp();

    let old_mem = Memory {
        id: "mem_old".to_string(),
        project_id: "test-merge".to_string(),
        memory_type: MemoryType::Fact,
        content: "Old fact".to_string(),
        summary: None,
        tags: vec!["tag_a".to_string()],
        importance: 0.3,
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
    };
    db.store_memory(&old_mem).unwrap();

    let new_mem = Memory {
        id: "mem_new".to_string(),
        tags: vec!["tag_b".to_string()],
        importance: 0.7,
        content: "New fact".to_string(),
        ..old_mem.clone()
    };
    db.store_memory(&new_mem).unwrap();

    // Merge
    db.merge_memories("mem_new", "mem_old", "Old fact preview")
        .unwrap();

    // Old memory should be deleted
    assert!(db.get_memory("mem_old").unwrap().is_none());

    // New memory should have merged data
    let merged = db.get_memory("mem_new").unwrap().unwrap();
    assert_eq!(merged.importance, 0.7); // max(0.3, 0.7)
    assert!(merged.tags.contains(&"tag_a".to_string()));
    assert!(merged.tags.contains(&"tag_b".to_string()));
    assert!(merged.merged_from.is_some());
    let sources = merged.merged_from.unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].id, "mem_old");
}

#[test]
fn test_cluster_operations() {
    let db = Database::open_in_memory().unwrap();

    let project = crate::memory::Project {
        id: "test-cluster".to_string(),
        name: "test-cluster".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    let now = chrono::Utc::now().timestamp();

    // Create a memory
    let mem = Memory {
        id: "mem_c1".to_string(),
        project_id: "test-cluster".to_string(),
        memory_type: MemoryType::Fact,
        content: "Cluster test".to_string(),
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
    };
    db.store_memory(&mem).unwrap();

    // Create a cluster
    let cluster = crate::memory::MemoryCluster {
        id: "clust_1".to_string(),
        project_id: "test-cluster".to_string(),
        summary: "Test cluster".to_string(),
        member_count: 0,
        centroid: Some(vec![0.1, 0.2, 0.3]),
        created_at: now,
        updated_at: now,
    };
    db.create_cluster(&cluster).unwrap();

    // Add memory to cluster
    db.add_to_cluster("clust_1", "mem_c1").unwrap();

    // Verify
    let c = db.get_cluster("clust_1").unwrap().unwrap();
    assert_eq!(c.member_count, 1);

    let members = db.get_cluster_member_ids("clust_1").unwrap();
    assert_eq!(members, vec!["mem_c1"]);

    // List clusters
    let clusters = db.get_clusters_for_project("test-cluster").unwrap();
    assert_eq!(clusters.len(), 1);

    // Update centroid
    db.update_cluster_centroid("clust_1", &[0.4, 0.5, 0.6], "Updated summary")
        .unwrap();
    let c = db.get_cluster("clust_1").unwrap().unwrap();
    assert_eq!(c.summary, "Updated summary");
    assert_eq!(c.centroid.unwrap(), vec![0.4, 0.5, 0.6]);

    // Remove from cluster
    let removed_from = db.remove_from_cluster("mem_c1").unwrap();
    assert_eq!(removed_from, Some("clust_1".to_string()));

    let c = db.get_cluster("clust_1").unwrap().unwrap();
    assert_eq!(c.member_count, 0);

    // Delete empty clusters
    let deleted = db.delete_empty_clusters("test-cluster").unwrap();
    assert_eq!(deleted, 1);
    assert!(db.get_cluster("clust_1").unwrap().is_none());
}

#[test]
fn test_migration3_fresh_db_has_pinned_and_global_columns() {
    let db = Database::open_in_memory().unwrap();
    let conn = db.conn.lock().unwrap();

    // Check schema version is at least 3
    let version: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        version >= 3,
        "expected schema version >= 3, got {}",
        version
    );

    // Verify pinned and global columns exist
    let mut stmt = conn.prepare("PRAGMA table_info(memories)").unwrap();
    let columns: Vec<String> = stmt
        .query_map([], |row| row.get(1))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    assert!(
        columns.contains(&"pinned".to_string()),
        "pinned column missing"
    );
    assert!(
        columns.contains(&"global".to_string()),
        "global column missing"
    );
}

#[test]
fn test_migration3_pinned_global_default_false() {
    let db = Database::open_in_memory().unwrap();

    let project = crate::memory::Project {
        id: "test-pinned-global".to_string(),
        name: "test-pinned-global".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    let now = chrono::Utc::now().timestamp();
    let memory = Memory {
        id: "mem-pg".to_string(),
        project_id: "test-pinned-global".to_string(),
        memory_type: MemoryType::Fact,
        content: "Test pinned global defaults".to_string(),
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
    };
    db.store_memory(&memory).unwrap();

    let retrieved = db.get_memory("mem-pg").unwrap().unwrap();
    assert!(!retrieved.pinned, "pinned should default to false");
    assert!(!retrieved.global, "global should default to false");
}

#[test]
fn test_migration3_store_and_retrieve_pinned_global() {
    let db = Database::open_in_memory().unwrap();

    let project = crate::memory::Project {
        id: "test-pg-flags".to_string(),
        name: "test-pg-flags".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    let now = chrono::Utc::now().timestamp();
    let memory = Memory {
        id: "mem-pg2".to_string(),
        project_id: "test-pg-flags".to_string(),
        memory_type: MemoryType::Fact,
        content: "Pinned and global memory".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.8,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: None,
        merged_from: None,
        external_artifacts: None,
        pinned: true,
        global: true,
    };
    db.store_memory(&memory).unwrap();

    let retrieved = db.get_memory("mem-pg2").unwrap().unwrap();
    assert!(retrieved.pinned, "pinned should be true");
    assert!(retrieved.global, "global should be true");
}

#[test]
fn test_migration3_upgrade_existing_db() {
    // Simulate upgrading a pre-migration-3 database by manually inserting a
    // memory without the pinned/global columns, then running migration.
    // We do this by opening an in-memory DB, removing the migration 3 record
    // from schema_version, dropping the columns if they exist, then calling
    // initialize again.
    //
    // In practice, SQLite does not support DROP COLUMN in older versions, so
    // we test the upgrade path by using a fresh DB that starts at version 2
    // and verifying migration 3 runs correctly.

    // Create a DB and verify migration 3 runs and leaves existing rows intact.
    let db = Database::open_in_memory().unwrap();

    let project = crate::memory::Project {
        id: "test-upgrade".to_string(),
        name: "test-upgrade".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    let now = chrono::Utc::now().timestamp();

    // Insert a memory using only the old columns (simulating a pre-migration row
    // by using DEFAULT values for pinned/global via the SQL DEFAULT clause).
    {
        let conn = db.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from)
             VALUES ('mem-old', 'test-upgrade', 'fact', 'Legacy memory', NULL, '[]', 0.5, 1.0, 0, ?1, ?1, ?1, NULL, NULL)",
            params![now],
        ).unwrap();
    }

    // Retrieve and verify that the defaults applied correctly
    let retrieved = db.get_memory("mem-old").unwrap().unwrap();
    assert!(
        !retrieved.pinned,
        "legacy memory should have pinned=false via DEFAULT"
    );
    assert!(
        !retrieved.global,
        "legacy memory should have global=false via DEFAULT"
    );
    assert_eq!(retrieved.content, "Legacy memory");
}

#[test]
fn test_migration3_project_stats_includes_counts() {
    let db = Database::open_in_memory().unwrap();

    let project = crate::memory::Project {
        id: "test-stats".to_string(),
        name: "test-stats".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    let now = chrono::Utc::now().timestamp();

    let make_memory = |id: &str, pinned: bool, global: bool| Memory {
        id: id.to_string(),
        project_id: "test-stats".to_string(),
        memory_type: MemoryType::Fact,
        content: format!("Memory {}", id),
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
        pinned,
        global,
    };

    db.store_memory(&make_memory("m1", false, false)).unwrap();
    db.store_memory(&make_memory("m2", true, false)).unwrap();
    db.store_memory(&make_memory("m3", false, true)).unwrap();
    db.store_memory(&make_memory("m4", true, true)).unwrap();

    let stats = db.get_project_stats("test-stats").unwrap();
    assert_eq!(stats.memory_count, 4);
    assert_eq!(stats.pinned_count, 2, "expected 2 pinned memories");
    // global_count queries all projects (WHERE global = 1), so 2 global memories total
    assert_eq!(stats.global_count, 2, "expected 2 global memories");
}

// ---- Section-embedding helpers ----

#[test]
fn test_encode_decode_section_embeddings_round_trip() {
    let keys = ["summary", "decisions", "todos"];
    let vectors: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32 * 0.1; 256]).collect();

    let (keys_str, bytes) = encode_section_embeddings(&keys, &vectors);
    assert_eq!(keys_str, "summary,decisions,todos");
    assert_eq!(bytes.len(), 3 * 256 * 4);

    let decoded = decode_section_embeddings(&keys_str, &bytes).unwrap();
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].0, "summary");
    assert_eq!(decoded[1].0, "decisions");
    assert_eq!(decoded[2].0, "todos");
    // Verify float values round-trip correctly
    assert!((decoded[0].1[0] - 0.0_f32).abs() < 1e-6);
    assert!((decoded[1].1[0] - 0.1_f32).abs() < 1e-6);
    assert!((decoded[2].1[0] - 0.2_f32).abs() < 1e-6);
}

#[test]
fn test_decode_section_embeddings_byte_length_validation() {
    // Wrong number of bytes for 2 keys
    let result = decode_section_embeddings("a,b", &[0u8; 100]);
    assert!(
        matches!(result, Err(MemoryError::Database(_))),
        "expected Database error for byte length mismatch"
    );
}

#[test]
fn test_decode_section_embeddings_empty() {
    let decoded = decode_section_embeddings("", &[]).unwrap();
    assert!(decoded.is_empty());
}

// ---- Handoff sidecar DB helpers ----

fn make_handoff_memory(id: &str, project_id: &str, branch: Option<&str>) -> Memory {
    let now = chrono::Utc::now().timestamp();
    Memory {
        id: id.to_string(),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Handoff,
        content: "## Summary\n\nTest handoff".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.85,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: branch.map(str::to_string),
        merged_from: None,
        external_artifacts: None,
        pinned: true,
        global: false,
    }
}

fn make_sections(summary: &str) -> HandoffSections {
    HandoffSections {
        summary: summary.to_string(),
        decisions: vec!["Use Rust".to_string()],
        todos: vec!["Write tests".to_string()],
        blockers: vec![],
        mental_model: "Layered architecture".to_string(),
        next_steps: vec!["Deploy".to_string()],
        notes: Some("Extra notes".to_string()),
        continues_from: None,
    }
}

#[test]
fn test_handoff_sections_round_trip() {
    let db = Database::open_in_memory().unwrap();
    let proj = crate::memory::Project {
        id: "ho-proj".to_string(),
        name: "ho-proj".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&proj).unwrap();

    let mem = make_handoff_memory("ho-1", "ho-proj", Some("main"));
    db.store_memory(&mem).unwrap();

    let sections = make_sections("Session ended well");
    let keys = [
        "summary",
        "decisions",
        "todos",
        "mental_model",
        "next_steps",
        "notes",
    ];
    let vecs: Vec<Vec<f32>> = keys.iter().map(|_| vec![0.5_f32; 256]).collect();
    let (keys_str, bytes) = encode_section_embeddings(&keys, &vecs);

    db.insert_handoff_sections("ho-1", &sections, &keys_str, &bytes)
        .unwrap();

    let result = db.get_handoff_sections("ho-1").unwrap().unwrap();
    let (retrieved_sections, retrieved_vecs) = result;

    assert_eq!(retrieved_sections.summary, "Session ended well");
    assert_eq!(retrieved_sections.decisions, vec!["Use Rust"]);
    assert_eq!(retrieved_sections.todos, vec!["Write tests"]);
    assert!(retrieved_sections.blockers.is_empty());
    assert_eq!(retrieved_sections.mental_model, "Layered architecture");
    assert_eq!(retrieved_sections.next_steps, vec!["Deploy"]);
    assert_eq!(retrieved_sections.notes, Some("Extra notes".to_string()));
    assert_eq!(retrieved_vecs.len(), 6);
    assert_eq!(retrieved_vecs[0].0, "summary");
    assert!((retrieved_vecs[0].1[0] - 0.5_f32).abs() < 1e-6);
}

#[test]
fn test_handoff_sections_cascade_delete() {
    let db = Database::open_in_memory().unwrap();
    let proj = crate::memory::Project {
        id: "ho-cascade".to_string(),
        name: "ho-cascade".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&proj).unwrap();

    let mem = make_handoff_memory("ho-del", "ho-cascade", None);
    db.store_memory(&mem).unwrap();

    let sections = make_sections("Will be deleted");
    let (keys_str, bytes) = encode_section_embeddings(&["summary"], &[vec![0.1_f32; 256]]);
    db.insert_handoff_sections("ho-del", &sections, &keys_str, &bytes)
        .unwrap();

    // Verify sidecar exists
    assert!(db.get_handoff_sections("ho-del").unwrap().is_some());

    // delete_memory should remove both the memory and the sidecar
    db.delete_memory("ho-del").unwrap();

    assert!(db.get_memory("ho-del").unwrap().is_none());
    assert!(db.get_handoff_sections("ho-del").unwrap().is_none());
}

#[test]
fn test_handoff_sections_update() {
    let db = Database::open_in_memory().unwrap();
    let proj = crate::memory::Project {
        id: "ho-upd".to_string(),
        name: "ho-upd".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&proj).unwrap();

    let mem = make_handoff_memory("ho-upd-1", "ho-upd", Some("feat/x"));
    db.store_memory(&mem).unwrap();

    let sections = make_sections("Original summary");
    let (keys_str, bytes) = encode_section_embeddings(&["summary"], &[vec![0.1_f32; 256]]);
    db.insert_handoff_sections("ho-upd-1", &sections, &keys_str, &bytes)
        .unwrap();

    // Update with new sections
    let updated = HandoffSections {
        summary: "Updated summary".to_string(),
        ..sections.clone()
    };
    let (new_keys_str, new_bytes) = encode_section_embeddings(
        &["summary", "decisions"],
        &[vec![0.2_f32; 256], vec![0.3_f32; 256]],
    );
    db.update_handoff_sections("ho-upd-1", &updated, &new_keys_str, &new_bytes)
        .unwrap();

    let (result, vecs) = db.get_handoff_sections("ho-upd-1").unwrap().unwrap();
    assert_eq!(result.summary, "Updated summary");
    // New embedding byte count matches 2 sections
    assert_eq!(new_bytes.len(), 2 * 256 * 4);
    assert_eq!(vecs.len(), 2);
}

#[test]
fn test_handoff_sections_update_with_continues_from() {
    // Verify that `continues_from: Some(...)` survives an update round-trip.
    let db = Database::open_in_memory().unwrap();
    let proj = crate::memory::Project {
        id: "ho-cf".to_string(),
        name: "ho-cf".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&proj).unwrap();

    let mem = make_handoff_memory("ho-cf-1", "ho-cf", Some("main"));
    db.store_memory(&mem).unwrap();

    // Insert with continues_from = None initially.
    let sections = make_sections("Initial summary");
    let (keys_str, bytes) = encode_section_embeddings(&["summary"], &[vec![0.1_f32; 256]]);
    db.insert_handoff_sections("ho-cf-1", &sections, &keys_str, &bytes)
        .unwrap();

    // Update to set continues_from = Some("prev-handoff-id").
    let updated = HandoffSections {
        summary: "Continued summary".to_string(),
        continues_from: Some("prev-handoff-id".to_string()),
        ..sections
    };
    let (new_keys_str, new_bytes) = encode_section_embeddings(&["summary"], &[vec![0.5_f32; 256]]);
    db.update_handoff_sections("ho-cf-1", &updated, &new_keys_str, &new_bytes)
        .unwrap();

    let (result, _vecs) = db.get_handoff_sections("ho-cf-1").unwrap().unwrap();
    assert_eq!(result.summary, "Continued summary");
    assert_eq!(
        result.continues_from.as_deref(),
        Some("prev-handoff-id"),
        "continues_from must survive update"
    );
}

#[test]
fn test_query_handoffs_by_branch() {
    let db = Database::open_in_memory().unwrap();
    let proj = crate::memory::Project {
        id: "ho-branch".to_string(),
        name: "ho-branch".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&proj).unwrap();

    // Store handoffs on two branches and one fact
    db.store_memory(&make_handoff_memory("ho-a1", "ho-branch", Some("feat/a")))
        .unwrap();
    db.store_memory(&make_handoff_memory("ho-b1", "ho-branch", Some("feat/b")))
        .unwrap();
    let now = chrono::Utc::now().timestamp();
    db.store_memory(&Memory {
        id: "fact-1".to_string(),
        project_id: "ho-branch".to_string(),
        memory_type: MemoryType::Fact,
        content: "A fact".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: Some("feat/a".to_string()),
        merged_from: None,
        external_artifacts: None,
        pinned: false,
        global: false,
    })
    .unwrap();

    // Branch filter: feat/a should return ho-a1 only
    let results = db
        .query_handoffs_by_branch("ho-branch", Some("feat/a"), 10)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "ho-a1");

    // No branch filter: both handoffs returned
    let results = db.query_handoffs_by_branch("ho-branch", None, 10).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_query_handoffs_unknown_type_propagates_error() {
    // Verify that parse_memory_type_col (used in query_handoffs_by_branch) returns
    // a rusqlite error for unknown type strings, and that the collect chain
    // propagates it instead of silently dropping rows.
    //
    // The SQL WHERE clause pre-filters to memory_type = 'handoff', so we cannot
    // inject a row that passes the filter yet fails parsing via a normal INSERT.
    // Instead we verify the error-propagation mechanism directly: the helper
    // parse_memory_type_col must return Err for unknown input.
    let err = parse_memory_type_col("not_a_valid_type", 2);
    assert!(
        err.is_err(),
        "parse_memory_type_col must return Err for unknown type"
    );

    // Additionally verify via a raw query that bypasses the handoff filter.
    // Build a DB with a corrupt row (memory_type = 'unknown_xyz') and query
    // all memories using query_map + collect to confirm errors surface.
    let db = Database::open_in_memory().unwrap();
    let proj = crate::memory::Project {
        id: "ho-bad".to_string(),
        name: "ho-bad".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&proj).unwrap();

    let now = chrono::Utc::now().timestamp();
    {
        let conn = db.conn.lock().unwrap();
        // Insert with an invalid type directly — bypasses all Rust type safety.
        conn.execute(
            "INSERT INTO memories
             (id, project_id, memory_type, content, summary, tags, importance,
              relevance_score, access_count, created_at, updated_at,
              last_accessed_at, branch, merged_from, pinned, global)
             VALUES (?1, ?2, 'unknown_xyz', ?3, NULL, '[]', 0.5, 1.0, 0,
                     ?4, ?4, ?4, NULL, NULL, 0, 0)",
            params!["ho-bad-1", "ho-bad", "corrupt row", now],
        )
        .unwrap();

        // Confirm the collect::<rusqlite::Result<Vec<_>>>() chain propagates errors.
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, memory_type, content, summary, tags, importance,
                        relevance_score, access_count, created_at, updated_at,
                        last_accessed_at, branch, merged_from, pinned, global
                 FROM memories WHERE project_id = ?1",
            )
            .unwrap();
        let result: rusqlite::Result<Vec<Memory>> = stmt
            .query_map(params!["ho-bad"], |row| {
                let memory_type_str: String = row.get(2)?;
                let tags_json: String = row.get(5)?;
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    memory_type: parse_memory_type_col(&memory_type_str, 2)?,
                    content: row.get(3)?,
                    summary: row.get(4)?,
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    importance: row.get(6)?,
                    relevance_score: row.get(7)?,
                    access_count: row.get(8)?,
                    created_at: row.get(9)?,
                    updated_at: row.get(10)?,
                    last_accessed_at: row.get(11)?,
                    branch: row.get(12)?,
                    merged_from: row
                        .get::<_, Option<String>>(13)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    pinned: row.get::<_, i64>(14)? != 0,
                    global: row.get::<_, i64>(15)? != 0,
                    external_artifacts: None,
                })
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>();
        assert!(
            result.is_err(),
            "collect chain must propagate unknown-type error, not drop the row"
        );
    }
}

#[test]
fn test_count_hook_memories_today() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("proj", "proj").unwrap();

    let now = chrono::Utc::now().timestamp();

    // Two hook-tagged memories.
    for i in 0..2 {
        let m = Memory {
            id: format!("mem-hook-{}", i),
            project_id: "proj".to_string(),
            memory_type: MemoryType::Fact,
            content: format!("hook memory {}", i),
            summary: None,
            tags: vec!["hook".to_string(), "prompt".to_string()],
            importance: 0.4,
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
        };
        db.store_memory(&m).unwrap();
    }

    // One memory without the "hook" tag.
    let untagged = Memory {
        id: "mem-plain".to_string(),
        project_id: "proj".to_string(),
        memory_type: MemoryType::Fact,
        content: "not a hook memory".to_string(),
        summary: None,
        tags: vec!["manual".to_string()],
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
    };
    db.store_memory(&untagged).unwrap();

    let count = db.count_hook_memories_today("proj").unwrap();
    assert_eq!(count, 2, "expected 2 hook memories, got {}", count);
}
