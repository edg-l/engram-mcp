//! Regression test for the production decay path.
//!
//! `Database::open()` (file-backed, used in real installs) and
//! `Database::open_in_memory()` (test-only) both must register `EXP()` and `LN()` as
//! SQLite scalar functions. The bundled SQLite is not compiled with
//! `SQLITE_ENABLE_MATH_FUNCTIONS`, so the decay query would otherwise silently fail
//! in production. This test exercises the production constructor against a real on-disk
//! database to catch any future regression where only the in-memory path registers them.

use engram_mcp::db::Database;
use engram_mcp::memory::{Memory, MemoryType, Project};

#[test]
fn decay_runs_on_file_backed_database() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("decay_regression.db");
    let db = Database::open(&db_path).expect("Database::open (production constructor)");

    let project_id = "decay-regression".to_string();
    let project = Project {
        id: project_id.clone(),
        name: project_id.clone(),
        root_path: None,
        decay_rate: 0.01,
        created_at: chrono::Utc::now().timestamp(),
    };
    db.create_project(&project).expect("create_project");

    // Insert a non-pinned memory whose last_accessed_at is 7 days ago.
    // Expected post-decay relevance:
    //   time_decay        = exp(-0.01 * 7) ~= 0.9324
    //   importance_factor = 0.5 + (0.5 * 0.5) = 0.75
    //   usage_boost       = ln(1) * 0.1 = 0
    //   relevance         ~= 0.9324 * 0.75 ~= 0.6993
    let seven_days_ago = chrono::Utc::now().timestamp() - 7 * 86400;
    let mem = Memory {
        id: "mem_decay_regression".to_string(),
        project_id: project_id.clone(),
        memory_type: MemoryType::Fact,
        content: "Stale fact that should decay below 1.0".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: seven_days_ago,
        updated_at: seven_days_ago,
        last_accessed_at: seven_days_ago,
        branch: None,
        merged_from: None,
        external_artifacts: None,
        pinned: false,
        global: false,
    };
    db.store_memory(&mem).expect("store_memory");

    // Run decay through the public production API.
    let updated = db
        .update_relevance_scores(&project_id, project.decay_rate)
        .expect("update_relevance_scores must not fail with 'no such function: EXP'");
    assert!(
        updated >= 1,
        "expected at least 1 memory updated, got {updated}"
    );

    let after = db
        .get_memory("mem_decay_regression")
        .expect("get_memory")
        .expect("memory must exist after decay");

    assert!(
        after.relevance_score < 1.0,
        "non-pinned memory must decay below 1.0; got relevance_score = {} (decay query likely never ran)",
        after.relevance_score
    );
    assert!(
        (after.relevance_score - 0.6993).abs() < 0.01,
        "expected relevance ~0.6993 after 7-day decay of importance=0.5 memory; got {}",
        after.relevance_score
    );
}

#[test]
fn decay_skips_pinned_memory_on_file_backed_database() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("decay_pinned.db");
    let db = Database::open(&db_path).expect("Database::open");

    let project_id = "decay-pinned".to_string();
    let project = Project {
        id: project_id.clone(),
        name: project_id.clone(),
        root_path: None,
        decay_rate: 0.5,
        created_at: chrono::Utc::now().timestamp(),
    };
    db.create_project(&project).expect("create_project");

    let one_year_ago = chrono::Utc::now().timestamp() - 365 * 86400;
    let pinned = Memory {
        id: "mem_pinned".to_string(),
        project_id: project_id.clone(),
        memory_type: MemoryType::Handoff,
        content: "Pinned content".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.85,
        relevance_score: 1.0,
        access_count: 0,
        created_at: one_year_ago,
        updated_at: one_year_ago,
        last_accessed_at: one_year_ago,
        branch: None,
        merged_from: None,
        external_artifacts: None,
        pinned: true,
        global: false,
    };
    db.store_memory(&pinned).expect("store_memory");

    db.update_relevance_scores(&project_id, project.decay_rate)
        .expect("update_relevance_scores");

    let after = db.get_memory("mem_pinned").unwrap().unwrap();
    assert_eq!(
        after.relevance_score, 1.0,
        "pinned memory must remain at relevance 1.0 across decay"
    );
}
