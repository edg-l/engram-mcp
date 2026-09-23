use rusqlite::params;

use crate::error::MemoryError;
use crate::memory::{AdrSections, AdrStatus, HandoffSections, Memory, MemoryType, Project};

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

    // Verify schema_version table exists and is at the latest migration
    let conn = db.conn.lock().unwrap();
    let version: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, 12);

    // Verify the curation sidecars exist
    for table in ["memory_status", "memory_trash"] {
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0, "{table} should exist and be empty");
    }

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

    // Verify adr_sections table exists (migration 6)
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM adr_sections", [], |row| row.get(0))
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
    db.merge_memories("mem_new", "mem_old").unwrap();

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
    // Provenance must carry what the consumed memory said, not a truncation of it:
    // the merge deleted the only other copy.
    assert_eq!(sources[0].content.as_deref(), Some("Old fact"));

    // And the consumed memory is recoverable.
    let entry = db.latest_trash_for_memory("mem_old").unwrap().unwrap();
    assert_eq!(entry.op, crate::db::OP_MERGE);
    assert_eq!(entry.memory.content, "Old fact");
}

/// A merge that keeps the *existing* memory must still record the consumed one.
///
/// The survivor is chosen by scope (a global memory beats a local one), so the consumed
/// memory is not always the older of the pair. Provenance that described the survivor
/// would point at content that was never lost.
#[test]
fn merge_records_the_memory_that_was_actually_consumed() {
    let db = Database::open_in_memory().unwrap();
    let project = crate::memory::Project {
        id: "merge-side".to_string(),
        name: "merge-side".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    let survivor = Memory {
        id: "mem_survivor".to_string(),
        project_id: "merge-side".to_string(),
        memory_type: MemoryType::Fact,
        content: "Survivor content".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: 0,
        updated_at: 0,
        last_accessed_at: 0,
        branch: None,
        merged_from: None,
        external_artifacts: None,
        pinned: false,
        global: true,
    };
    let consumed = Memory {
        id: "mem_consumed".to_string(),
        content: "Consumed content that is longer than a preview would keep".to_string(),
        global: false,
        ..survivor.clone()
    };
    db.store_memory(&survivor).unwrap();
    db.store_memory(&consumed).unwrap();

    db.merge_memories("mem_survivor", "mem_consumed").unwrap();

    let merged = db.get_memory("mem_survivor").unwrap().unwrap();
    let sources = merged.merged_from.unwrap();
    assert_eq!(sources[0].id, "mem_consumed");
    assert_eq!(
        sources[0].content.as_deref(),
        Some("Consumed content that is longer than a preview would keep")
    );
}

/// Merging a composite into another memory carries the composite's provenance along, so
/// an id it absorbed earlier still resolves to a live memory afterwards.
#[test]
fn merging_a_composite_carries_its_provenance_forward() {
    let db = Database::open_in_memory().unwrap();
    let project = crate::memory::Project {
        id: "merge-chain".to_string(),
        name: "merge-chain".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    let base = Memory {
        id: "mem_a".to_string(),
        project_id: "merge-chain".to_string(),
        memory_type: MemoryType::Fact,
        content: "A".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: 0,
        updated_at: 0,
        last_accessed_at: 0,
        branch: None,
        merged_from: None,
        external_artifacts: None,
        pinned: false,
        global: false,
    };
    for (id, content) in [("mem_a", "A"), ("mem_b", "B"), ("mem_c", "C")] {
        db.store_memory(&Memory {
            id: id.to_string(),
            content: content.to_string(),
            ..base.clone()
        })
        .unwrap();
    }

    db.merge_memories("mem_b", "mem_a").unwrap();
    db.merge_memories("mem_c", "mem_b").unwrap();

    let sources: Vec<String> = db
        .get_memory("mem_c")
        .unwrap()
        .unwrap()
        .merged_from
        .unwrap()
        .into_iter()
        .map(|s| s.id)
        .collect();
    assert_eq!(sources, vec!["mem_a".to_string(), "mem_b".to_string()]);
    assert_eq!(
        db.find_merge_survivor("merge-chain", "mem_a")
            .unwrap()
            .as_deref(),
        Some("mem_c")
    );
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
        tried: vec![],
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

// ---- ADR DB helpers ----

fn make_adr_sections(title: &str) -> AdrSections {
    AdrSections {
        title: title.to_string(),
        context: format!("Context for {}", title),
        decision: format!("Decision for {}", title),
        consequences: format!("Consequences for {}", title),
    }
}

fn fake_embedding() -> Vec<f32> {
    vec![0.1f32; 256]
}

fn store_test_adr(db: &Database, id: &str, project_id: &str, title: &str) -> (String, u32) {
    let sections = make_adr_sections(title);
    let now = chrono::Utc::now().timestamp();
    db.store_adr_atomic(
        id,
        project_id,
        &sections,
        AdrStatus::Proposed,
        0.7,
        false,
        &fake_embedding(),
        "test",
        now,
        None,
    )
    .unwrap()
}

#[test]
fn adr_numbering_is_sequential_and_gapfree() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("adr-seq", "adr-seq").unwrap();

    let (_, n1) = store_test_adr(&db, "adr-s1", "adr-seq", "First ADR");
    let (_, n2) = store_test_adr(&db, "adr-s2", "adr-seq", "Second ADR");
    let (_, n3) = store_test_adr(&db, "adr-s3", "adr-seq", "Third ADR");

    assert_eq!(n1, 1, "first ADR should be number 1");
    assert_eq!(n2, 2, "second ADR should be number 2");
    assert_eq!(n3, 3, "third ADR should be number 3");
}

#[test]
fn adr_numbering_is_per_project() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("proj-a", "proj-a").unwrap();
    db.get_or_create_project("proj-b", "proj-b").unwrap();

    let (_, na1) = store_test_adr(&db, "adr-a1", "proj-a", "ADR A1");
    let (_, nb1) = store_test_adr(&db, "adr-b1", "proj-b", "ADR B1");
    let (_, na2) = store_test_adr(&db, "adr-a2", "proj-a", "ADR A2");
    let (_, nb2) = store_test_adr(&db, "adr-b2", "proj-b", "ADR B2");

    assert_eq!(na1, 1, "project-a first ADR should be 1");
    assert_eq!(na2, 2, "project-a second ADR should be 2");
    assert_eq!(
        nb1, 1,
        "project-b first ADR should be 1 (independent sequence)"
    );
    assert_eq!(nb2, 2, "project-b second ADR should be 2");
}

#[test]
fn adr_unique_number_constraint() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("adr-uniq", "adr-uniq").unwrap();

    // Store a memory row first so the FK is satisfied.
    let now = chrono::Utc::now().timestamp();
    {
        let conn = db.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memories \
             (id, project_id, memory_type, content, summary, tags, importance, \
              relevance_score, access_count, created_at, updated_at, last_accessed_at, \
              branch, merged_from, pinned, global) \
             VALUES ('adr-dup-a', 'adr-uniq', 'adr', 'content', NULL, '[]', 0.5, 1.0, 0, \
                     ?1, ?1, ?1, NULL, NULL, 0, 0)",
            params![now],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories \
             (id, project_id, memory_type, content, summary, tags, importance, \
              relevance_score, access_count, created_at, updated_at, last_accessed_at, \
              branch, merged_from, pinned, global) \
             VALUES ('adr-dup-b', 'adr-uniq', 'adr', 'content', NULL, '[]', 0.5, 1.0, 0, \
                     ?1, ?1, ?1, NULL, NULL, 0, 0)",
            params![now],
        )
        .unwrap();
    }

    let sections = make_adr_sections("Duplicate");
    db.insert_adr_sidecar(
        "adr-dup-a",
        "adr-uniq",
        42,
        AdrStatus::Proposed,
        &sections,
        now,
        now,
    )
    .unwrap();

    // Inserting with the same (project_id, adr_number) must fail.
    let result = db.insert_adr_sidecar(
        "adr-dup-b",
        "adr-uniq",
        42,
        AdrStatus::Proposed,
        &sections,
        now,
        now,
    );
    assert!(
        matches!(result, Err(MemoryError::Database(_))),
        "duplicate (project_id, adr_number) must produce a Database error"
    );
}

#[test]
fn update_adr_status_roundtrip() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("adr-upd", "adr-upd").unwrap();

    let (id, _) = store_test_adr(&db, "adr-upd-1", "adr-upd", "Status Roundtrip");

    // Start: Proposed
    let (_, status, _) = db.get_adr_sections(&id).unwrap().unwrap();
    assert_eq!(status, AdrStatus::Proposed);

    // Transition: Proposed -> Accepted
    let updated = db.update_adr_status(&id, AdrStatus::Accepted).unwrap();
    assert!(updated, "update should return true when row exists");

    let (_, status, _) = db.get_adr_sections(&id).unwrap().unwrap();
    assert_eq!(status, AdrStatus::Accepted);

    // Absent memory_id returns false.
    let updated_missing = db
        .update_adr_status("nonexistent", AdrStatus::Deprecated)
        .unwrap();
    assert!(!updated_missing, "update on missing id should return false");
}

#[test]
fn list_adrs_filter_by_status() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("adr-list", "adr-list").unwrap();

    store_test_adr(&db, "adr-l1", "adr-list", "ADR One");
    store_test_adr(&db, "adr-l2", "adr-list", "ADR Two");
    store_test_adr(&db, "adr-l3", "adr-list", "ADR Three");

    // Transition the second one to Accepted.
    db.update_adr_status("adr-l2", AdrStatus::Accepted).unwrap();

    // Unfiltered: all three.
    let all = db.list_adrs("adr-list", None).unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].0, 1);
    assert_eq!(all[1].0, 2);
    assert_eq!(all[2].0, 3);

    // Filter proposed: first and third.
    let proposed = db.list_adrs("adr-list", Some(AdrStatus::Proposed)).unwrap();
    assert_eq!(proposed.len(), 2);
    assert!(
        proposed
            .iter()
            .all(|(_, s, _, _)| *s == AdrStatus::Proposed)
    );

    // Filter accepted: only the second.
    let accepted = db.list_adrs("adr-list", Some(AdrStatus::Accepted)).unwrap();
    assert_eq!(accepted.len(), 1);
    assert_eq!(accepted[0].2, "ADR Two");
}

#[test]
fn get_adr_by_number_found_and_missing() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("adr-bynum", "adr-bynum").unwrap();

    let (id, number) = store_test_adr(&db, "adr-bn1", "adr-bynum", "By Number Test");
    assert_eq!(number, 1);

    let found = db.get_adr_by_number("adr-bynum", 1).unwrap();
    assert_eq!(found, Some(id));

    let missing = db.get_adr_by_number("adr-bynum", 999).unwrap();
    assert!(missing.is_none());
}

#[test]
fn insert_adr_sidecar_explicit_number() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("adr-explicit", "adr-explicit")
        .unwrap();

    // Insert a memory row manually so the FK is satisfied.
    let now = chrono::Utc::now().timestamp();
    {
        let conn = db.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memories \
             (id, project_id, memory_type, content, summary, tags, importance, \
              relevance_score, access_count, created_at, updated_at, last_accessed_at, \
              branch, merged_from, pinned, global) \
             VALUES ('adr-exp-1', 'adr-explicit', 'adr', 'content', NULL, '[]', 0.7, 1.0, 0, \
                     ?1, ?1, ?1, NULL, NULL, 0, 0)",
            params![now],
        )
        .unwrap();
    }

    let sections = make_adr_sections("Explicit Number ADR");

    // Insert with explicit number 7.
    db.insert_adr_sidecar(
        "adr-exp-1",
        "adr-explicit",
        7,
        AdrStatus::Accepted,
        &sections,
        now,
        now,
    )
    .unwrap();

    // Verify retrieval.
    let (num, status, retrieved) = db.get_adr_sections("adr-exp-1").unwrap().unwrap();
    assert_eq!(num, 7);
    assert_eq!(status, AdrStatus::Accepted);
    assert_eq!(retrieved.title, "Explicit Number ADR");

    // Verify get_adr_by_number works.
    let mid = db.get_adr_by_number("adr-explicit", 7).unwrap();
    assert_eq!(mid, Some("adr-exp-1".to_string()));
}

// ============================================
// Curation: supersession, dead status, trash
// ============================================

/// Build a project with two memories and return the db.
fn curation_fixture() -> Database {
    let db = Database::open_in_memory().unwrap();
    let project = Project {
        id: "curation".to_string(),
        name: "curation".to_string(),
        root_path: None,
        decay_rate: 0.01,
        created_at: 0,
    };
    db.create_project(&project).unwrap();

    for (id, content) in [
        ("mem_old", "The collapse is X"),
        ("mem_new", "The collapse is Y"),
    ] {
        let memory = Memory {
            id: id.to_string(),
            project_id: "curation".to_string(),
            memory_type: MemoryType::Decision,
            content: content.to_string(),
            summary: None,
            tags: vec!["collapse".to_string()],
            importance: 0.7,
            relevance_score: 1.0,
            access_count: 0,
            created_at: 0,
            updated_at: 0,
            last_accessed_at: 0,
            branch: None,
            merged_from: None,
            external_artifacts: None,
            pinned: false,
            global: false,
        };
        db.store_memory(&memory).unwrap();
    }
    db
}

fn supersedes_edge(db: &Database, newer: &str, older: &str) {
    let rel = crate::memory::Relationship {
        id: format!("rel_{newer}_{older}"),
        source_id: newer.to_string(),
        target_id: older.to_string(),
        relation_type: crate::memory::RelationType::Supersedes,
        strength: 1.0,
        created_at: 0,
    };
    db.create_relationship(&rel).unwrap();
}

#[test]
fn supersession_map_reads_the_edge_direction() {
    let db = curation_fixture();
    supersedes_edge(&db, "mem_new", "mem_old");

    let map = db.get_supersession_map("curation").unwrap();
    assert!(
        map.is_superseded("mem_old"),
        "the target of the edge is the superseded one"
    );
    assert!(!map.is_superseded("mem_new"));
    assert_eq!(map.terminal_successor("mem_old"), Some("mem_new"));
}

#[test]
fn dead_status_round_trips() {
    let db = curation_fixture();
    assert!(!db.is_dead("mem_old").unwrap());

    db.set_dead("mem_old", true, Some("service retired"))
        .unwrap();
    assert!(db.is_dead("mem_old").unwrap());
    assert_eq!(db.count_dead("curation").unwrap(), 1);
    assert!(db.get_dead_ids("curation").unwrap().contains("mem_old"));

    db.set_dead("mem_old", false, None).unwrap();
    assert!(!db.is_dead("mem_old").unwrap());
    assert_eq!(db.count_dead("curation").unwrap(), 0);
}

#[test]
fn delete_puts_the_memory_in_the_trash() {
    let db = curation_fixture();
    db.delete_memory("mem_old").unwrap();

    assert!(db.get_memory("mem_old").unwrap().is_none());
    let entry = db.latest_trash_for_memory("mem_old").unwrap().unwrap();
    assert_eq!(entry.op, crate::db::OP_DELETE);
    assert_eq!(entry.memory.content, "The collapse is X");
    assert_eq!(db.count_trash("curation").unwrap(), 1);
}

#[test]
fn restore_brings_back_content_embedding_and_edges() {
    let db = curation_fixture();
    supersedes_edge(&db, "mem_new", "mem_old");
    db.store_embedding("mem_old", &[0.25_f32; 256], "test-model")
        .unwrap();

    db.delete_memory("mem_old").unwrap();
    assert!(db.get_embedding("mem_old").unwrap().is_none());
    assert!(
        db.get_supersession_map("curation")
            .unwrap()
            .terminal_successor("mem_old")
            .is_none()
    );

    let entry = db.latest_trash_for_memory("mem_old").unwrap().unwrap();
    let outcome = db.restore_trash_entry(entry.trash_id).unwrap();

    assert_eq!(outcome.edges_restored, 1);
    assert_eq!(outcome.edges_dropped, 0);
    assert!(!outcome.overwrote_existing);

    let restored = db.get_memory("mem_old").unwrap().unwrap();
    assert_eq!(restored.content, "The collapse is X");
    assert_eq!(db.get_embedding("mem_old").unwrap().unwrap().len(), 256);
    // The supersedes edge is back, so retrieval will redirect again.
    let map = db.get_supersession_map("curation").unwrap();
    assert_eq!(map.terminal_successor("mem_old"), Some("mem_new"));
    // The entry is consumed.
    assert!(db.latest_trash_for_memory("mem_old").unwrap().is_none());
}

/// An edge whose other end was also deleted cannot be recreated; the restore must say so
/// rather than fail or silently drop it.
#[test]
fn restore_reports_edges_it_could_not_reconnect() {
    let db = curation_fixture();
    supersedes_edge(&db, "mem_new", "mem_old");

    db.delete_memory("mem_old").unwrap();
    db.delete_memory("mem_new").unwrap();

    let entry = db.latest_trash_for_memory("mem_old").unwrap().unwrap();
    let outcome = db.restore_trash_entry(entry.trash_id).unwrap();
    assert_eq!(outcome.edges_restored, 0);
    assert_eq!(outcome.edges_dropped, 1);
}

#[test]
fn restoring_over_a_live_memory_snapshots_it_first() {
    let db = curation_fixture();

    // Simulate a content-replacing update: snapshot, then overwrite.
    db.trash_memory("mem_old", crate::db::OP_UPDATE).unwrap();
    let mut edited = db.get_memory("mem_old").unwrap().unwrap();
    edited.content = "A reconstruction that lost the tail".to_string();
    db.update_memory(&edited).unwrap();

    let entry = db.latest_trash_for_memory("mem_old").unwrap().unwrap();
    let outcome = db.restore_trash_entry(entry.trash_id).unwrap();

    assert!(outcome.overwrote_existing);
    assert_eq!(
        db.get_memory("mem_old").unwrap().unwrap().content,
        "The collapse is X"
    );
    // The reconstruction is itself recoverable, so the restore is undoable.
    let latest = db.latest_trash_for_memory("mem_old").unwrap().unwrap();
    assert_eq!(latest.memory.content, "A reconstruction that lost the tail");
}

#[test]
fn sweep_trash_respects_the_retention_window() {
    let db = curation_fixture();
    db.delete_memory("mem_old").unwrap();
    assert_eq!(db.count_trash("curation").unwrap(), 1);

    // A retention window that has not elapsed keeps the entry.
    assert_eq!(db.sweep_trash(30).unwrap(), 0);
    assert_eq!(db.count_trash("curation").unwrap(), 1);

    // Zero or negative means keep forever, not delete everything.
    assert_eq!(db.sweep_trash(0).unwrap(), 0);
    assert_eq!(db.count_trash("curation").unwrap(), 1);
}

#[test]
fn wipe_is_recoverable() {
    let db = curation_fixture();
    let wiped = db.delete_project_data("curation").unwrap();
    assert_eq!(wiped, 2);
    assert_eq!(db.count_trash("curation").unwrap(), 2);

    let entry = db.latest_trash_for_memory("mem_new").unwrap().unwrap();
    assert_eq!(entry.op, crate::db::OP_WIPE);
    db.restore_trash_entry(entry.trash_id).unwrap();
    assert!(db.get_memory("mem_new").unwrap().is_some());
}

#[test]
fn resolve_known_project_matches_unique_last_segment() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("git:example.com/edgar/antworld", "antworld")
        .unwrap();
    db.get_or_create_project("home-project", "home-project")
        .unwrap();

    let resolved = db.resolve_known_project("antworld", None).unwrap();
    assert_eq!(resolved.as_deref(), Some("git:example.com/edgar/antworld"));
}

#[test]
fn resolve_known_project_matches_case_insensitively() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("git:example.com/edgar/antworld", "antworld")
        .unwrap();

    let resolved = db.resolve_known_project("AntWorld", None).unwrap();
    assert_eq!(resolved.as_deref(), Some("git:example.com/edgar/antworld"));
}

#[test]
fn resolve_known_project_ambiguous_short_name_lists_only_candidates() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("git:example.com/alice/widget", "widget")
        .unwrap();
    db.get_or_create_project("git:example.com/bob/widget", "widget")
        .unwrap();
    db.get_or_create_project("home-project", "home-project")
        .unwrap();

    let err = db.resolve_known_project("widget", None).unwrap_err();
    match err {
        MemoryError::UnknownProject { requested, known } => {
            assert_eq!(requested, "widget");
            assert!(known.contains("git:example.com/alice/widget"));
            assert!(known.contains("git:example.com/bob/widget"));
            assert!(!known.contains("home-project"));
        }
        other => panic!("expected UnknownProject, got {other:?}"),
    }
}

#[test]
fn resolve_known_project_expands_legacy_absolute_path() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("~/dev/antworld", "antworld")
        .unwrap();

    // Not a real directory, so this falls back to folding the string itself
    // against a home prefix rather than walking a git root.
    let resolved = db
        .resolve_known_project("/home/someoneelse/dev/antworld", None)
        .unwrap();
    assert_eq!(resolved.as_deref(), Some("~/dev/antworld"));
}

#[test]
fn resolve_known_project_no_match_returns_none() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("home-project", "home-project")
        .unwrap();

    assert_eq!(
        db.resolve_known_project("does-not-exist", None).unwrap(),
        None
    );
}

#[test]
fn resolve_known_project_follows_alias_to_survivor() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("old", "old").unwrap();
    db.get_or_create_project("new", "new").unwrap();

    db.merge_project("old", "new", true).unwrap();

    // The merged-away id has no `projects` row and no memories left, so without
    // alias resolution this would fall through to `Ok(None)` (or, for a write path
    // that treats `None` as "create it"), recreating the dead project.
    assert!(!db.project_exists("old").unwrap());
    assert_eq!(
        db.resolve_known_project("old", None).unwrap().as_deref(),
        Some("new")
    );
}

#[test]
fn resolve_known_project_alias_resolution_beats_fuzzy_matching() {
    let db = Database::open_in_memory().unwrap();
    // A project whose last path segment happens to equal the merged-away id.
    db.get_or_create_project("git:example.com/someone/old", "old")
        .unwrap();
    db.get_or_create_project("old", "old").unwrap();
    db.get_or_create_project("new", "new").unwrap();

    db.merge_project("old", "new", true).unwrap();

    // Without alias-first resolution this would be ambiguous (or silently pick the
    // fuzzy match) instead of following the exact alias to its real survivor.
    assert_eq!(
        db.resolve_known_project("old", None).unwrap().as_deref(),
        Some("new")
    );
}

#[test]
fn resolve_known_project_follows_alias_chain() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("c", "c").unwrap();
    let now = chrono::Utc::now().timestamp();
    // Insert a two-hop chain directly: `merge_projects_in` always collapses an
    // existing alias to the new target at merge time, so a live chain longer than
    // one hop is not otherwise reachable through the public API.
    let conn = db.conn.lock().unwrap();
    conn.execute(
        "INSERT INTO project_aliases (alias, project_id, merged_at) VALUES ('a', 'b', ?1)",
        params![now],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO project_aliases (alias, project_id, merged_at) VALUES ('b', 'c', ?1)",
        params![now],
    )
    .unwrap();
    drop(conn);

    assert_eq!(
        db.resolve_known_project("a", None).unwrap().as_deref(),
        Some("c")
    );
}

#[test]
fn resolve_known_project_alias_cycle_does_not_hang() {
    let db = Database::open_in_memory().unwrap();
    let now = chrono::Utc::now().timestamp();
    let conn = db.conn.lock().unwrap();
    conn.execute(
        "INSERT INTO project_aliases (alias, project_id, merged_at) VALUES ('a', 'b', ?1)",
        params![now],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO project_aliases (alias, project_id, merged_at) VALUES ('b', 'a', ?1)",
        params![now],
    )
    .unwrap();
    drop(conn);

    // Corrupt data with no sane target; must terminate rather than loop, and must
    // not be mistaken for a valid resolution.
    assert_eq!(db.resolve_known_project("a", None).unwrap(), None);
}

// ---- Project merge and directory reconciliation ----

fn store_fact(db: &Database, id: &str, project_id: &str) {
    let memory = Memory {
        id: id.to_string(),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Fact,
        content: format!("content of {id}"),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: 0,
        updated_at: 0,
        last_accessed_at: 0,
        branch: None,
        merged_from: None,
        external_artifacts: None,
        pinned: false,
        global: false,
    };
    db.store_memory(&memory).unwrap();
}

fn project_ids(db: &Database) -> Vec<String> {
    db.list_projects()
        .unwrap()
        .into_iter()
        .map(|p| p.id)
        .collect()
}

fn memory_count(db: &Database, project_id: &str) -> usize {
    db.get_project_stats(project_id).unwrap().memory_count
}

fn root_path(db: &Database, project_id: &str) -> Option<String> {
    db.get_project(project_id)
        .unwrap()
        .and_then(|p| p.root_path)
}

fn store_adr_at(db: &Database, id: &str, project_id: &str, created_at: i64) -> u32 {
    db.store_adr_atomic(
        id,
        project_id,
        &make_adr_sections(id),
        AdrStatus::Proposed,
        0.7,
        false,
        &fake_embedding(),
        "test",
        created_at,
        None,
    )
    .unwrap()
    .1
}

fn adr_number_of(db: &Database, memory_id: &str) -> i64 {
    let conn = db.conn.lock().unwrap();
    conn.query_row(
        "SELECT adr_number FROM adr_sections WHERE memory_id = ?1",
        params![memory_id],
        |row| row.get(0),
    )
    .unwrap()
}

#[test]
fn reconcile_merges_home_relative_id_after_remote_add() {
    let db = Database::open_in_memory().unwrap();
    // Before the remote existed, the repo's id was its home-relative directory.
    db.get_or_create_project("~/dev/gardener", "~/dev/gardener")
        .unwrap();
    store_fact(&db, "before-remote", "~/dev/gardener");
    let derived = "git:example.com/owner/gardener";
    db.get_or_create_project(derived, derived).unwrap();
    store_fact(&db, "after-remote", derived);

    let report = db
        .reconcile_project_root(derived, "~/dev/gardener")
        .unwrap();

    assert_eq!(report.merges.len(), 1);
    assert_eq!(report.merges[0].from, "~/dev/gardener");
    assert_eq!(report.merges[0].to, derived);
    assert_eq!(report.merges[0].memories, 1);
    assert!(report.merges[0].project_row);
    assert_eq!(memory_count(&db, derived), 2);
    assert_eq!(project_ids(&db), vec![derived.to_string()]);
    assert_eq!(root_path(&db, derived).as_deref(), Some("~/dev/gardener"));
    assert_eq!(
        db.project_aliases()
            .unwrap()
            .get("~/dev/gardener")
            .map(String::as_str),
        Some(derived)
    );
}

#[test]
fn reconcile_merges_by_root_path_after_remote_change() {
    let db = Database::open_in_memory().unwrap();
    let old = "git:example.com/old-owner/repo";
    let new = "git:example.com/new-owner/repo";
    // The old remote's id recorded its directory the last time it was derived there.
    db.reconcile_project_root(old, "~/dev/repo").unwrap();
    store_fact(&db, "under-old-remote", old);

    let report = db.reconcile_project_root(new, "~/dev/repo").unwrap();

    assert_eq!(report.merges.len(), 1);
    assert_eq!(report.merges[0].from, old);
    assert_eq!(memory_count(&db, new), 1);
    assert!(!db.project_exists(old).unwrap());
    assert_eq!(root_path(&db, new).as_deref(), Some("~/dev/repo"));
}

#[test]
fn reconcile_merges_project_known_only_through_memories() {
    let db = Database::open_in_memory().unwrap();
    store_fact(&db, "orphan", "~/dev/norow");
    let derived = "git:example.com/owner/norow";

    let report = db.reconcile_project_root(derived, "~/dev/norow").unwrap();

    assert_eq!(report.merges.len(), 1);
    assert!(!report.merges[0].project_row);
    assert_eq!(memory_count(&db, derived), 1);
    assert!(!db.project_exists("~/dev/norow").unwrap());
}

#[test]
fn merge_renumbers_colliding_adrs_across_the_merged_pair() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("old", "old").unwrap();
    db.get_or_create_project("new", "new").unwrap();
    assert_eq!(store_adr_at(&db, "new-early", "new", 50), 1);
    assert_eq!(store_adr_at(&db, "old-mid", "old", 100), 1);
    assert_eq!(store_adr_at(&db, "new-late", "new", 200), 2);

    let report = db.merge_project("old", "new", true).unwrap();

    // Chronological across both projects; the earliest keeps its number.
    assert_eq!(adr_number_of(&db, "new-early"), 1);
    assert_eq!(adr_number_of(&db, "old-mid"), 2);
    assert_eq!(adr_number_of(&db, "new-late"), 3);
    assert_eq!(report.merges[0].adrs, 1);
    assert_eq!(report.adrs_renumbered, 2);
    assert_eq!(db.next_adr_number("new").unwrap(), 4);
}

#[test]
fn reconcile_is_a_noop_when_nothing_matches() {
    let db = Database::open_in_memory().unwrap();
    db.reconcile_project_root("git:example.com/owner/other", "~/dev/other")
        .unwrap();
    store_fact(&db, "other-fact", "git:example.com/owner/other");
    let mine = "git:example.com/owner/mine";

    let first = db.reconcile_project_root(mine, "~/dev/mine").unwrap();
    assert!(first.is_empty());
    assert_eq!(root_path(&db, mine).as_deref(), Some("~/dev/mine"));
    assert_eq!(memory_count(&db, "git:example.com/owner/other"), 1);

    // Once the root is recorded, a repeat run reads and writes nothing.
    let changes = |db: &Database| -> i64 {
        let conn = db.conn.lock().unwrap();
        conn.query_row("SELECT total_changes()", [], |row| row.get(0))
            .unwrap()
    };
    let before = changes(&db);
    let second = db.reconcile_project_root(mine, "~/dev/mine").unwrap();
    assert!(second.is_empty());
    assert_eq!(changes(&db), before);
}

#[test]
fn explicit_identity_never_reconciles() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("~/dev/pinned", "~/dev/pinned")
        .unwrap();
    store_fact(&db, "pinned-fact", "~/dev/pinned");

    let explicit = crate::project::resolve_project(Some("git:example.com/owner/pinned".into()));
    assert_eq!(explicit.root, None);
    db.reconcile_identity(&explicit);

    assert_eq!(memory_count(&db, "~/dev/pinned"), 1);
    assert!(
        db.get_project("git:example.com/owner/pinned")
            .unwrap()
            .is_none()
    );
    assert!(db.project_aliases().unwrap().is_empty());
}

#[test]
fn merge_dry_run_reports_the_same_counts_and_changes_nothing() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("old", "old").unwrap();
    db.get_or_create_project("new", "new").unwrap();
    store_fact(&db, "a", "old");
    store_fact(&db, "b", "old");
    store_fact(&db, "c", "new");

    let dry = db.merge_project("old", "new", false).unwrap();
    assert_eq!(memory_count(&db, "old"), 2);
    assert!(db.get_project("old").unwrap().is_some());
    assert!(db.project_aliases().unwrap().is_empty());

    let applied = db.merge_project("old", "new", true).unwrap();
    assert_eq!(dry, applied);
    assert_eq!(memory_count(&db, "new"), 3);
}

#[test]
fn merge_refuses_unknown_ids_and_self_merge() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("known", "known").unwrap();

    assert!(matches!(
        db.merge_project("known", "known", true),
        Err(MemoryError::InvalidArguments { .. })
    ));
    assert!(matches!(
        db.merge_project("missing", "known", true),
        Err(MemoryError::UnknownProject { requested, .. }) if requested == "missing"
    ));
    assert!(matches!(
        db.merge_project("known", "missing", true),
        Err(MemoryError::UnknownProject { requested, .. }) if requested == "missing"
    ));
}

#[test]
fn merged_trash_restores_into_the_target_project() {
    let db = Database::open_in_memory().unwrap();
    db.get_or_create_project("old", "old").unwrap();
    db.get_or_create_project("new", "new").unwrap();
    store_fact(&db, "deleted-before-merge", "old");
    db.delete_memory("deleted-before-merge").unwrap();

    let report = db.merge_project("old", "new", true).unwrap();
    assert_eq!(report.merges[0].trash, 1);

    let entry = db
        .latest_trash_for_memory("deleted-before-merge")
        .unwrap()
        .unwrap();
    db.restore_trash_entry(entry.trash_id).unwrap();
    let restored = db.get_memory("deleted-before-merge").unwrap().unwrap();
    assert_eq!(restored.project_id, "new");
    assert!(!db.project_exists("old").unwrap());
}

#[test]
fn merge_repoints_aliases_that_named_the_merged_away_project() {
    let db = Database::open_in_memory().unwrap();
    for id in ["a", "b", "c"] {
        db.get_or_create_project(id, id).unwrap();
    }
    db.merge_project("a", "b", true).unwrap();
    db.merge_project("b", "c", true).unwrap();

    let aliases = db.project_aliases().unwrap();
    assert_eq!(aliases.get("a").map(String::as_str), Some("c"));
    assert_eq!(aliases.get("b").map(String::as_str), Some("c"));
}
