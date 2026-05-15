//! Binary-spawn integration tests for `engram-cli hook-event`.
//!
//! Each test spawns the real binary with `ENGRAM_DB` pointing at a unique
//! `TempDir`, reads back the DB directly via the library API, and asserts on
//! what was (or was not) stored.

use assert_cmd::Command;
use engram_mcp::db::Database;
use engram_mcp::memory::{Memory, MemoryType};
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Spawn `engram-cli hook-event <event>` with stdin piped from `stdin_json`.
/// Returns the `Assert` handle so callers can chain `.success()` etc.
fn spawn_hook_event(event: &str, stdin_json: &str, db_path: &Path) -> assert_cmd::assert::Assert {
    let mut cmd = Command::cargo_bin("engram-cli").unwrap();
    cmd.env("ENGRAM_DB", db_path)
        .env("ENGRAM_PROJECT", "test-bin-project")
        // Disable SubagentStop (off by default, but make it explicit in tests)
        .env("ENGRAM_HOOK_SUBAGENTSTOP_ENABLED", "false")
        .arg("hook-event")
        .arg(event)
        .write_stdin(stdin_json.to_string())
        .assert()
}

/// Open the DB at `db_path` and return all memories for project `test-bin-project`.
fn read_memories(db_path: &Path) -> Vec<Memory> {
    let db = Database::open(db_path).expect("failed to open test DB");
    db.get_all_memories_for_project("test-bin-project")
        .expect("failed to list memories")
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Blocker B1: hook-event reads the JSON payload from stdin when --payload is omitted.
///
/// Pipe `UserPromptSubmit` JSON to stdin (no --payload flag).
/// The dispatch handler must store a `Fact` row containing "Postgres".
#[test]
fn hook_event_reads_payload_from_stdin_without_flag() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    // Prompt must match the cue regex (decided/chose/prefer/switch) and be >= 40 chars.
    let stdin_json =
        r#"{"prompt":"we decided to switch to Postgres because of write throughput issues"}"#;

    let mut cmd = Command::cargo_bin("engram-cli").unwrap();
    cmd.env("ENGRAM_DB", &db_path)
        .env("ENGRAM_PROJECT", "test-bin-project")
        .env("ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED", "true")
        .arg("hook-event")
        .arg("UserPromptSubmit")
        .write_stdin(stdin_json.to_string())
        .assert()
        .success();

    let memories = read_memories(&db_path);
    let fact = memories
        .iter()
        .find(|m| m.memory_type == MemoryType::Fact && m.content.contains("Postgres"));
    assert!(
        fact.is_some(),
        "expected a Fact row containing 'Postgres', got:\n{:#?}",
        memories
    );
}

/// --payload flag overrides stdin content.
///
/// Pass `--payload` with a refactor decision while writing unrelated garbage to stdin.
/// Only the --payload content should be stored.
#[test]
fn hook_event_payload_flag_overrides_stdin() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    let payload_json =
        r#"{"prompt":"we decided to refactor the auth module because of the latency issue"}"#;

    let mut cmd = Command::cargo_bin("engram-cli").unwrap();
    cmd.env("ENGRAM_DB", &db_path)
        .env("ENGRAM_PROJECT", "test-bin-project")
        .env("ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED", "true")
        .arg("hook-event")
        .arg("UserPromptSubmit")
        .arg("--payload")
        .arg(payload_json)
        // Unrelated stdin — must be ignored when --payload is given.
        .write_stdin("garbage that should not be stored")
        .assert()
        .success();

    let memories = read_memories(&db_path);

    // The --payload content should produce a Fact.
    let from_payload = memories
        .iter()
        .find(|m| m.memory_type == MemoryType::Fact && m.content.contains("auth module"));
    assert!(
        from_payload.is_some(),
        "expected a Fact row for 'auth module', got:\n{:#?}",
        memories
    );

    // The garbage stdin must NOT be stored.
    let from_stdin = memories
        .iter()
        .find(|m| m.content.contains("garbage that should not be stored"));
    assert!(
        from_stdin.is_none(),
        "stdin content must not be stored when --payload is provided, got:\n{:#?}",
        memories
    );
}

/// Stop is now a no-op — nothing is stored regardless of payload content.
#[test]
fn stop_event_stores_nothing() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    // Build a 300-char summary (previously Stop required >= 200 chars, now irrelevant).
    let long_summary = format!(
        "Session ended after completing the authentication refactor. {}",
        "x".repeat(250)
    );
    let stdin_json = format!(r#"{{"response":"{}"}}"#, long_summary);

    let mut cmd = Command::cargo_bin("engram-cli").unwrap();
    cmd.env("ENGRAM_DB", &db_path)
        .env("ENGRAM_PROJECT", "test-bin-project")
        .current_dir(tmp.path())
        .arg("hook-event")
        .arg("Stop")
        .write_stdin(stdin_json)
        .assert()
        .success();

    let memories = read_memories(&db_path);
    assert!(
        memories.is_empty(),
        "Stop is a no-op: expected no stored memories, got:\n{:#?}",
        memories
    );
}

/// PreCompact is now a no-op — nothing is stored regardless of payload content.
#[test]
fn pre_compact_event_stores_nothing() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    // Test both the previously-ignored custom_instructions path and the empty path
    // in a single test since both are now no-ops.
    for stdin_json in &[
        r#"{"compaction_reason":"manual","custom_instructions":"focus on auth code paths"}"#,
        r#"{"compaction_reason":"auto"}"#,
    ] {
        let tmp2 = TempDir::new().unwrap();
        let db_path2 = tmp2.path().join("test.db");

        let mut cmd = Command::cargo_bin("engram-cli").unwrap();
        cmd.env("ENGRAM_DB", &db_path2)
            .env("ENGRAM_PROJECT", "test-bin-project")
            .current_dir(tmp2.path())
            .arg("hook-event")
            .arg("PreCompact")
            .write_stdin(stdin_json.to_string())
            .assert()
            .success();

        let memories = read_memories(&db_path2);
        assert!(
            memories.is_empty(),
            "PreCompact is a no-op: expected no stored memories for payload {:?}, got:\n{:#?}",
            stdin_json,
            memories
        );
    }

    // Verify the original db_path wasn't written to either
    let memories = read_memories(&db_path);
    assert!(memories.is_empty());
}

/// SessionEnd reads the transcript file and stores an unpinned Fact tagged session_summary.
#[test]
fn session_end_reads_transcript_and_stores_fact() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    // Write a JSONL transcript file whose last line is an assistant entry.
    let transcript_path = tmp.path().join("transcript.jsonl");
    let mut f = std::fs::File::create(&transcript_path).unwrap();
    writeln!(f, r#"{{"type":"user","message":{{"content":"hello"}}}}"#).unwrap();
    writeln!(
        f,
        r#"{{"type":"assistant","message":{{"content":[{{"type":"text","text":"The refactor is complete and all tests pass successfully."}}]}}}}"#
    )
    .unwrap();

    let stdin_json = format!(
        r#"{{"reason":"logout","transcript_path":"{}"}}"#,
        transcript_path.to_str().unwrap()
    );

    spawn_hook_event("SessionEnd", &stdin_json, &db_path).success();

    let memories = read_memories(&db_path);
    let fact = memories
        .iter()
        .find(|m| m.memory_type == MemoryType::Fact && m.content.contains("refactor is complete"));
    assert!(
        fact.is_some(),
        "expected a Fact row containing 'refactor is complete', got:\n{:#?}",
        memories
    );

    let fact = fact.unwrap();
    assert!(
        !fact.pinned,
        "session_summary Fact must not be pinned, got pinned=true"
    );
    assert!(
        fact.tags.contains(&"session_summary".to_string()),
        "expected 'session_summary' tag, got tags: {:?}",
        fact.tags
    );
    assert!(
        fact.tags.contains(&"hook".to_string()),
        "expected 'hook' tag, got tags: {:?}",
        fact.tags
    );
    assert!(
        fact.importance <= 0.5,
        "hook_importance_cap must clamp stored importance to <= 0.5, got {}",
        fact.importance
    );
}

/// SessionEnd with no transcript_path stores nothing and exits successfully.
#[test]
fn session_end_missing_transcript_path_stores_nothing() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    let stdin_json = r#"{"reason":"logout"}"#;

    spawn_hook_event("SessionEnd", stdin_json, &db_path).success();

    let memories = read_memories(&db_path);
    assert!(
        memories.is_empty(),
        "expected nothing stored when transcript_path is absent, got:\n{:#?}",
        memories
    );
}
