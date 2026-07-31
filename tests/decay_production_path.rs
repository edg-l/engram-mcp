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

    // A memory last accessed 7 calendar days ago, in a project that has received no
    // stores since. Decay runs on store-days (see `db::activity`), so nothing has
    // displaced it and only the importance factor applies:
    //   time_decay        = exp(-0.01 * 0) = 1.0
    //   importance_factor = 0.5 + (0.5 * 0.5) = 0.75
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

    // Run decay through the public production API. This must not fail with
    // "no such function: EXP" — the bundled SQLite has no math functions, so the
    // production constructor has to register them just as the in-memory one does.
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
        (after.relevance_score - 0.75).abs() < 0.001,
        "a memory in a project with no later stores must not decay on calendar time; \
         got {}",
        after.relevance_score
    );

    // Now displace it: seven days on which the project received a store, all still in
    // the past relative to nothing in particular — the calendar is irrelevant, the
    // store-days are what count.
    //   time_decay        = exp(-0.01 * 7) ~= 0.9324
    //   relevance         ~= 0.9324 * 0.75 ~= 0.6993
    for day in 1..=7 {
        let at = seven_days_ago + day * 86400;
        db.store_memory(&Memory {
            id: format!("mem_displacer_{day}"),
            created_at: at,
            updated_at: at,
            last_accessed_at: at,
            content: format!("Newer knowledge landing on day {day}"),
            ..mem.clone()
        })
        .expect("store displacing memory");
    }

    db.update_relevance_scores(&project_id, project.decay_rate)
        .expect("update_relevance_scores");

    let after = db
        .get_memory("mem_decay_regression")
        .expect("get_memory")
        .expect("memory must exist after decay");
    assert!(
        after.relevance_score < 1.0,
        "displaced memory must decay below 1.0; got {}",
        after.relevance_score
    );
    assert!(
        (after.relevance_score - 0.6993).abs() < 0.01,
        "expected relevance ~0.6993 after 7 store-days of displacement; got {}",
        after.relevance_score
    );
}

/// Hook-captured stores are automatic, so they must not age deliberately curated
/// knowledge: a session where passive capture was the only thing that happened has not
/// made anything staler.
#[test]
fn hook_captures_do_not_advance_the_clock() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("decay_hooks.db");
    let db = Database::open(&db_path).expect("Database::open");

    let project_id = "decay-hooks".to_string();
    let project = Project {
        id: project_id.clone(),
        name: project_id.clone(),
        root_path: None,
        decay_rate: 0.01,
        created_at: chrono::Utc::now().timestamp(),
    };
    db.create_project(&project).expect("create_project");

    let base = chrono::Utc::now().timestamp() - 30 * 86400;
    let curated = Memory {
        id: "mem_curated".to_string(),
        project_id: project_id.clone(),
        memory_type: MemoryType::Decision,
        content: "A decision worth keeping".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: base,
        updated_at: base,
        last_accessed_at: base,
        branch: None,
        merged_from: None,
        external_artifacts: None,
        pinned: false,
        global: false,
    };
    db.store_memory(&curated).expect("store curated");

    // Twenty days of nothing but automatic capture.
    for day in 1..=20 {
        let at = base + day * 86400;
        db.store_memory(&Memory {
            id: format!("mem_hook_{day}"),
            created_at: at,
            updated_at: at,
            last_accessed_at: at,
            memory_type: MemoryType::Fact,
            tags: vec!["hook".to_string(), "session_summary".to_string()],
            content: format!("Auto-captured session summary {day}"),
            ..curated.clone()
        })
        .expect("store hook capture");
    }

    db.update_relevance_scores(&project_id, project.decay_rate)
        .expect("update_relevance_scores");

    let after = db
        .get_memory("mem_curated")
        .expect("get_memory")
        .expect("memory must exist");
    assert!(
        (after.relevance_score - 0.75).abs() < 0.001,
        "hook captures must not advance the decay clock; got {}",
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
