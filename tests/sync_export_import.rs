//! `engram-cli export --all-projects`/`--since` and `import -`/merge-mode last-write-wins
//! convergence, added by Phase 4 (whole-store and incremental export/import).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use engram_mcp::export::{ExportData, ExportScope};

/// The binary built for this test run. `CARGO_BIN_EXE_*` always points at the current
/// profile's build, so a stale binary from another profile cannot be picked up.
fn cli() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_engram-cli"))
}

#[derive(Debug)]
struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn run(db: &std::path::Path, project: &str, args: &[&str]) -> Run {
    let out = Command::new(cli())
        .env("ENGRAM_DB", db)
        .env("ENGRAM_PROJECT", project)
        .args(args)
        .output()
        .expect("failed to spawn engram-cli");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Like `run`, but pipes `input` to the child's stdin — for `import -`.
fn run_with_stdin(db: &std::path::Path, project: &str, args: &[&str], input: &str) -> Run {
    let mut child = Command::new(cli())
        .env("ENGRAM_DB", db)
        .env("ENGRAM_PROJECT", project)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn engram-cli");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(input.as_bytes())
        .expect("write payload to stdin");
    let out = child.wait_with_output().expect("wait for engram-cli");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Store one memory and return its id, read back via `--json list` (the CLI's `store`
/// output only prints the id in prose, not JSON).
fn store(db: &std::path::Path, project: &str, content: &str) -> String {
    let stored = run(db, project, &["store", content]);
    assert_eq!(stored.code, Some(0), "{}", stored.stderr);
    let listed = run(db, project, &["--json", "list", "--status", "all"]);
    let value: serde_json::Value = serde_json::from_str(&listed.stdout).expect("list json");
    value["memories"]
        .as_array()
        .expect("memories array")
        .iter()
        .find(|m| m["content"].as_str() == Some(content))
        .and_then(|m| m["id"].as_str())
        .unwrap_or_else(|| panic!("no memory with content {content:?} in {listed:?}"))
        .to_string()
}

/// Overwrite `memories.updated_at` directly, for deterministic `--since` boundaries
/// without depending on wall-clock second granularity between two `store` calls.
fn set_updated_at(db_path: &std::path::Path, memory_id: &str, ts: i64) {
    let conn = rusqlite::Connection::open(db_path).expect("open db for direct write");
    conn.execute(
        "UPDATE memories SET updated_at = ?1 WHERE id = ?2",
        rusqlite::params![ts, memory_id],
    )
    .expect("set updated_at");
}

/// All-projects export/import round trip: two projects in one store, exported together
/// and imported into a fresh store, preserving memories, a relationship, a closed todo,
/// and a dead flag — each staying under its own project rather than being re-homed.
#[test]
fn all_projects_export_import_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let db_a = dir.path().join("a.db");

    let id_a1 = store(
        &db_a,
        "proj-a",
        "The alpha service caches responses in Redis with a 5 minute TTL",
    );
    let id_a2 = store(
        &db_a,
        "proj-a",
        "Database migrations for the alpha service run via a Rust binary called migrate",
    );
    let id_b = store(
        &db_a,
        "proj-b",
        "The beta service authenticates requests using mTLS certificates",
    );

    let linked = run(&db_a, "proj-a", &["link", &id_a1, &id_a2]);
    assert_eq!(linked.code, Some(0), "{:?}", linked);

    let added = run(&db_a, "proj-a", &["todo", "add", "Ship the alpha rollout"]);
    assert_eq!(added.code, Some(0), "{:?}", added);
    let todos = run(
        &db_a,
        "proj-a",
        &["--json", "todo", "list", "--status", "all"],
    );
    let todos_json: serde_json::Value = serde_json::from_str(&todos.stdout).unwrap();
    let todo_id = todos_json["todos"][0]["id"].as_str().unwrap().to_string();
    let done = run(&db_a, "proj-a", &["todo", "done", &todo_id]);
    assert_eq!(done.code, Some(0), "{:?}", done);

    let dead = run(
        &db_a,
        "proj-b",
        &[
            "update",
            &id_b,
            "--dead",
            "--dead-reason",
            "service retired",
        ],
    );
    assert_eq!(dead.code, Some(0), "{:?}", dead);

    let exported = run(
        &db_a,
        "proj-a",
        &["export", "--all-projects", "--embeddings"],
    );
    assert_eq!(exported.code, Some(0), "{:?}", exported);
    let export_data: ExportData = serde_json::from_str(&exported.stdout).expect("export json");
    assert_eq!(export_data.scope, ExportScope::AllProjects);
    assert_eq!(export_data.version, "1.3");
    assert_eq!(
        export_data.project_id, "",
        "project_id is meaningless for an all-projects export"
    );

    let projects: std::collections::HashSet<&str> = export_data
        .memories
        .iter()
        .map(|m| m.memory.project_id.as_str())
        .collect();
    assert!(projects.contains("proj-a"));
    assert!(projects.contains("proj-b"));

    // Import into a fresh store, scoped to a project neither payload memory belongs to —
    // all-projects import must ignore it entirely and re-home nothing.
    let db_b = dir.path().join("b.db");
    let imported = run_with_stdin(
        &db_b,
        "importer-default",
        &["import", "-"],
        &exported.stdout,
    );
    assert_eq!(imported.code, Some(0), "{:?}", imported);
    assert!(
        imported.stdout.contains("1 relationships"),
        "{:?}",
        imported
    );

    let list_a = run(&db_b, "proj-a", &["--json", "list", "--status", "all"]);
    let list_a_json: serde_json::Value = serde_json::from_str(&list_a.stdout).unwrap();
    // 2 facts + 1 todo memory.
    assert_eq!(list_a_json["count"].as_u64(), Some(3), "{:?}", list_a);

    let list_b = run(&db_b, "proj-b", &["--json", "list", "--status", "all"]);
    let list_b_json: serde_json::Value = serde_json::from_str(&list_b.stdout).unwrap();
    assert_eq!(list_b_json["count"].as_u64(), Some(1), "{:?}", list_b);

    let dead_list = run(&db_b, "proj-b", &["--json", "list", "--status", "dead"]);
    let dead_json: serde_json::Value = serde_json::from_str(&dead_list.stdout).unwrap();
    assert_eq!(
        dead_json["count"].as_u64(),
        Some(1),
        "dead flag must survive the round trip: {:?}",
        dead_list
    );

    let todo_list_b = run(
        &db_b,
        "proj-a",
        &["--json", "todo", "list", "--status", "done"],
    );
    let todo_list_json: serde_json::Value = serde_json::from_str(&todo_list_b.stdout).unwrap();
    assert_eq!(
        todo_list_json["count"].as_u64(),
        Some(1),
        "closed todo status must survive the round trip: {:?}",
        todo_list_b
    );

    // The importer's own project (never referenced by any payload memory) stays empty.
    let list_importer = run(
        &db_b,
        "importer-default",
        &["--json", "list", "--status", "all"],
    );
    let list_importer_json: serde_json::Value =
        serde_json::from_str(&list_importer.stdout).unwrap();
    assert_eq!(
        list_importer_json["count"].as_u64(),
        Some(0),
        "{:?}",
        list_importer
    );
}

/// `--since` returns only rows updated at or after the given timestamp, inclusive of the
/// exact boundary.
#[test]
fn since_filters_to_rows_at_or_after_the_timestamp() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("since.db");

    let old_id = store(&db, "since-proj", "Old fact untouched by the sync window");
    let new_id = store(&db, "since-proj", "New fact inside the sync window");

    let base = 1_700_000_000_i64;
    set_updated_at(&db, &old_id, base);
    set_updated_at(&db, &new_id, base + 1000);

    let cutoff = base + 500;
    let exported = run(
        &db,
        "since-proj",
        &["export", "--since", &cutoff.to_string()],
    );
    assert_eq!(exported.code, Some(0), "{:?}", exported);
    let export_data: ExportData = serde_json::from_str(&exported.stdout).unwrap();
    let ids: Vec<&str> = export_data
        .memories
        .iter()
        .map(|m| m.memory.id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec![new_id.as_str()],
        "only the row at/after the cutoff must be exported"
    );

    // The filter is inclusive: a row exactly at the boundary must still appear.
    let at_boundary = run(
        &db,
        "since-proj",
        &["export", "--since", &(base + 1000).to_string()],
    );
    let boundary_data: ExportData = serde_json::from_str(&at_boundary.stdout).unwrap();
    assert_eq!(
        boundary_data.memories.len(),
        1,
        "the exact-boundary row must be included"
    );
}

/// Re-running an import with the same payload changes nothing the second time: every
/// memory is already present at an equal or newer `updated_at`, so merge mode skips all
/// of them rather than re-storing or re-updating.
#[test]
fn reimporting_the_same_payload_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let db_src = dir.path().join("src.db");
    store(
        &db_src,
        "idem-proj",
        "The gateway rate-limits with a token bucket",
    );
    store(
        &db_src,
        "idem-proj",
        "Nightly backups are pruned after 30 days by a cron job",
    );

    let exported = run(&db_src, "idem-proj", &["export"]);
    assert_eq!(exported.code, Some(0), "{:?}", exported);

    let db_dst = dir.path().join("dst.db");
    let first = run_with_stdin(&db_dst, "idem-proj", &["import", "-"], &exported.stdout);
    assert_eq!(first.code, Some(0), "{:?}", first);
    assert!(first.stdout.contains("Imported 2 memories"), "{:?}", first);

    let second = run_with_stdin(&db_dst, "idem-proj", &["import", "-"], &exported.stdout);
    assert_eq!(second.code, Some(0), "{:?}", second);
    assert!(
        second.stdout.contains("Imported 0 memories, 0 updated")
            && second.stdout.contains("(2 skipped)"),
        "second import must be a no-op on counts: {:?}",
        second
    );
}

/// Merge mode is last-write-wins on `updated_at`: an older incoming copy leaves local
/// content untouched, a newer one replaces it. Payloads are built from a real export's
/// shape (captured once) with the id, content, and timestamp overridden, so the test
/// controls the LWW comparison deterministically without depending on wall-clock gaps
/// between `store` calls.
#[test]
fn merge_import_is_last_write_wins_both_directions() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("lww.db");

    let id = store(&db, "lww-proj", "Baseline content");
    let local_ts = 1_700_000_000_i64;
    set_updated_at(&db, &id, local_ts);

    let seed = run(&db, "lww-proj", &["export"]);
    assert_eq!(seed.code, Some(0), "{:?}", seed);
    let seed_value: serde_json::Value = serde_json::from_str(&seed.stdout).unwrap();
    let template = seed_value["memories"][0].clone();

    let build_payload = |content: &str, updated_at: i64| -> String {
        let mut memory = template.clone();
        memory["content"] = serde_json::json!(content);
        memory["updated_at"] = serde_json::json!(updated_at);
        let payload = serde_json::json!({
            "version": "1.3",
            "project_id": "lww-proj",
            "scope": "project",
            "memories": [memory],
            "relationships": [],
            "exported_at": updated_at,
            "model_version": serde_json::Value::Null,
        });
        serde_json::to_string(&payload).unwrap()
    };

    // Older incoming: ties/older keep the local copy.
    let older = run_with_stdin(
        &db,
        "lww-proj",
        &["import", "-"],
        &build_payload("Older content", local_ts - 100),
    );
    assert_eq!(older.code, Some(0), "{:?}", older);
    assert!(older.stdout.contains("(1 skipped)"), "{:?}", older);
    let after_older = run(&db, "lww-proj", &["--json", "show", &id]);
    let after_older_json: serde_json::Value = serde_json::from_str(&after_older.stdout).unwrap();
    assert_eq!(
        after_older_json["memory"]["content"].as_str(),
        Some("Baseline content"),
        "an older incoming copy must not replace local content"
    );

    // Newer incoming: replaces local content.
    let newer = run_with_stdin(
        &db,
        "lww-proj",
        &["import", "-"],
        &build_payload("Newer content", local_ts + 100),
    );
    assert_eq!(newer.code, Some(0), "{:?}", newer);
    assert!(
        newer.stdout.contains("Imported 0 memories, 1 updated"),
        "{:?}",
        newer
    );
    let after_newer = run(&db, "lww-proj", &["--json", "show", &id]);
    let after_newer_json: serde_json::Value = serde_json::from_str(&after_newer.stdout).unwrap();
    assert_eq!(
        after_newer_json["memory"]["content"].as_str(),
        Some("Newer content"),
        "a newer incoming copy must replace local content"
    );
}
