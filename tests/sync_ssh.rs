//! `engram-cli sync <target>`, added by Phase 5 (bidirectional incremental replication
//! over `ssh`). `ENGRAM_SSH_CMD` lets the whole ssh round trip be replaced with a stub
//! script that execs the real `engram-cli` binary against a second `ENGRAM_DB` — the
//! "remote" — so the two-machine protocol is exercised without a network or a second
//! host.

use std::path::{Path, PathBuf};
use std::process::Command;

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

fn run(db: &Path, project: &str, args: &[&str]) -> Run {
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

/// Store one memory directly against a project's own database and return its id, read
/// back via `--json list` (the CLI's `store` output only prints the id in prose).
fn store(db: &Path, project: &str, content: &str) -> String {
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

fn has_memory(db: &Path, project: &str, content: &str) -> bool {
    let listed = run(db, project, &["--json", "list", "--status", "all"]);
    let value: serde_json::Value = serde_json::from_str(&listed.stdout).expect("list json");
    value["memories"]
        .as_array()
        .expect("memories array")
        .iter()
        .any(|m| m["content"].as_str() == Some(content))
}

/// Write the stub ssh replacement to `path` and mark it executable. It ignores its
/// first argument (the ssh target), execs its second argument (`--remote-bin`) with the
/// rest of the args, and points that child at `ENGRAM_SYNC_STUB_REMOTE_DB` — the
/// "remote" database — instead of whatever `ENGRAM_DB` it inherited from the caller.
/// When `ENGRAM_SYNC_STUB_FAIL_IMPORT` is set and the forwarded subcommand is `import`,
/// it fails without touching the remote at all, simulating a push that never lands.
fn write_stub_ssh(path: &Path) {
    let script = r#"#!/usr/bin/env bash
set -euo pipefail
shift # ssh target — irrelevant to the stub, the "connection" is just this exec
remote_bin="$1"
shift
if [ "${1:-}" = "import" ] && [ -n "${ENGRAM_SYNC_STUB_FAIL_IMPORT:-}" ]; then
    echo "stub: forced import failure" >&2
    exit 7
fi
export ENGRAM_DB="$ENGRAM_SYNC_STUB_REMOTE_DB"
exec "$remote_bin" "$@"
"#;
    std::fs::write(path, script).expect("write stub ssh script");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .expect("chmod stub ssh script");
    }
}

/// Run `engram-cli sync` against the stub, wiring `ENGRAM_DB` (local) and
/// `ENGRAM_SYNC_STUB_REMOTE_DB` (remote) so the stub's `exec` lands on the right store.
fn run_sync(
    local_db: &Path,
    stub: &Path,
    remote_db: &Path,
    fail_import: bool,
    extra_args: &[&str],
) -> Run {
    let mut cmd = Command::new(cli());
    cmd.env("ENGRAM_DB", local_db)
        .env("ENGRAM_SSH_CMD", stub)
        .env("ENGRAM_SYNC_STUB_REMOTE_DB", remote_db)
        .args(["sync", "stub-target", "--remote-bin"])
        .arg(cli())
        .args(extra_args);
    if fail_import {
        cmd.env("ENGRAM_SYNC_STUB_FAIL_IMPORT", "1");
    } else {
        cmd.env_remove("ENGRAM_SYNC_STUB_FAIL_IMPORT");
    }
    let out = cmd.output().expect("failed to spawn engram-cli sync");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn sync_state(db: &Path) -> Option<(i64, i64)> {
    let conn = rusqlite::Connection::open(db).expect("open db for sync_state check");
    conn.query_row(
        "SELECT pull_watermark, push_watermark FROM sync_state WHERE remote = 'stub-target'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .ok()
}

/// A first sync moves each side's memory to the other: the local store gains the
/// remote's memory and the remote store gains the local's, and both watermarks land in
/// `sync_state` for the target.
#[test]
fn first_sync_moves_memories_both_ways() {
    let dir = tempfile::tempdir().unwrap();
    let local_db = dir.path().join("local.db");
    let remote_db = dir.path().join("remote.db");
    let stub = dir.path().join("stub_ssh.sh");
    write_stub_ssh(&stub);

    store(
        &local_db,
        "local-proj",
        "Only known locally before the first sync",
    );
    store(
        &remote_db,
        "remote-proj",
        "Only known on the remote before the first sync",
    );

    let synced = run_sync(&local_db, &stub, &remote_db, false, &[]);
    assert_eq!(synced.code, Some(0), "{:?}", synced);

    assert!(
        has_memory(
            &local_db,
            "remote-proj",
            "Only known on the remote before the first sync"
        ),
        "the remote's memory must arrive locally: {:?}",
        synced
    );
    assert!(
        has_memory(
            &remote_db,
            "local-proj",
            "Only known locally before the first sync"
        ),
        "the local memory must arrive on the remote: {:?}",
        synced
    );

    let (pull_wm, push_wm) = sync_state(&local_db).expect("sync_state row for stub-target");
    assert!(pull_wm > 0, "pull watermark must advance: {pull_wm}");
    assert!(push_wm > 0, "push watermark must advance: {push_wm}");
}

/// Running sync again immediately after a first sync transfers nothing new: both sides
/// are already converged, so counts and watermarks are unchanged.
#[test]
fn second_sync_is_a_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let local_db = dir.path().join("local.db");
    let remote_db = dir.path().join("remote.db");
    let stub = dir.path().join("stub_ssh.sh");
    write_stub_ssh(&stub);

    store(&local_db, "local-proj", "Local content before any sync");
    store(&remote_db, "remote-proj", "Remote content before any sync");

    let first = run_sync(&local_db, &stub, &remote_db, false, &[]);
    assert_eq!(first.code, Some(0), "{:?}", first);
    let watermarks_after_first = sync_state(&local_db).expect("sync_state after first sync");

    let local_count_before = run(
        &local_db,
        "local-proj",
        &["--json", "list", "--status", "all"],
    );
    let remote_count_before = run(
        &remote_db,
        "remote-proj",
        &["--json", "list", "--status", "all"],
    );

    let second = run_sync(&local_db, &stub, &remote_db, false, &[]);
    assert_eq!(second.code, Some(0), "{:?}", second);
    // The `since` filter is inclusive, so the boundary row is re-sent on every sync (see
    // the Watermarks design note) — a no-op means nothing new landed on either side, not
    // that zero bytes moved. "Imported 0 memories, 0 updated" is the remote's own
    // `import` output, forwarded verbatim by the push half.
    assert!(
        second.stdout.contains("pull: 0 memories, 0 updated")
            && second.stdout.contains("Imported 0 memories, 0 updated"),
        "an immediate second sync must import/update nothing new on either side: {:?}",
        second
    );

    let watermarks_after_second = sync_state(&local_db).expect("sync_state after second sync");
    assert_eq!(
        watermarks_after_first, watermarks_after_second,
        "a no-op sync must not move either watermark"
    );

    let local_count_after = run(
        &local_db,
        "local-proj",
        &["--json", "list", "--status", "all"],
    );
    let remote_count_after = run(
        &remote_db,
        "remote-proj",
        &["--json", "list", "--status", "all"],
    );
    assert_eq!(local_count_before.stdout, local_count_after.stdout);
    assert_eq!(remote_count_before.stdout, remote_count_after.stdout);
}

/// A memory created on the "remote" after the first sync is missing from the local
/// store until the next sync, at which point the incremental pull half picks it up.
#[test]
fn remote_only_memory_arrives_on_next_sync() {
    let dir = tempfile::tempdir().unwrap();
    let local_db = dir.path().join("local.db");
    let remote_db = dir.path().join("remote.db");
    let stub = dir.path().join("stub_ssh.sh");
    write_stub_ssh(&stub);

    store(&local_db, "local-proj", "Local seed content");
    store(&remote_db, "remote-proj", "Remote seed content");

    let first = run_sync(&local_db, &stub, &remote_db, false, &[]);
    assert_eq!(first.code, Some(0), "{:?}", first);

    // New knowledge shows up on the remote after the first sync has already run.
    store(
        &remote_db,
        "remote-proj",
        "Remote memory added after the first sync",
    );
    assert!(!has_memory(
        &local_db,
        "remote-proj",
        "Remote memory added after the first sync"
    ));

    let second = run_sync(&local_db, &stub, &remote_db, false, &[]);
    assert_eq!(second.code, Some(0), "{:?}", second);

    assert!(
        has_memory(
            &local_db,
            "remote-proj",
            "Remote memory added after the first sync"
        ),
        "a post-first-sync remote memory must arrive on the next sync: {:?}",
        second
    );
}

/// When the remote's `import` fails, the push half's watermark must not advance —
/// but the pull half, already committed earlier in the same invocation, is untouched
/// by that later failure: its watermark still advances.
#[test]
fn failed_push_leaves_push_watermark_unchanged_but_pull_advances() {
    let dir = tempfile::tempdir().unwrap();
    let local_db = dir.path().join("local.db");
    let remote_db = dir.path().join("remote.db");
    let stub = dir.path().join("stub_ssh.sh");
    write_stub_ssh(&stub);

    store(
        &local_db,
        "local-proj",
        "Local content the push half tries to send",
    );
    store(
        &remote_db,
        "remote-proj",
        "Remote content the pull half fetches",
    );

    let synced = run_sync(&local_db, &stub, &remote_db, true, &[]);
    assert_eq!(
        synced.code,
        Some(1),
        "a failed remote import must fail the whole sync command: {:?}",
        synced
    );
    assert!(
        synced.stderr.contains("remote import failed"),
        "{:?}",
        synced
    );

    let (pull_wm, push_wm) = sync_state(&local_db).expect("sync_state row for stub-target");
    assert!(
        pull_wm > 0,
        "pull already committed earlier in the same invocation must still advance: {pull_wm}"
    );
    assert_eq!(
        push_wm, 0,
        "a failed push must leave the push watermark untouched"
    );

    // The pull half's own data really did land locally, independent of the push failure.
    assert!(has_memory(
        &local_db,
        "remote-proj",
        "Remote content the pull half fetches"
    ));
    // The remote never received the local memory: its own import never ran to
    // completion (the stub fails before exec'ing the real binary).
    assert!(!has_memory(
        &remote_db,
        "local-proj",
        "Local content the push half tries to send"
    ));
}
