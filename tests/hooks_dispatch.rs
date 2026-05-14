//! Binary-spawn integration tests for `engram-cli hook-event`.
//!
//! Each test spawns the real binary with `ENGRAM_DB` pointing at a unique
//! `TempDir`, reads back the DB directly via the library API, and asserts on
//! what was (or was not) stored.

use assert_cmd::Command;
use engram_mcp::db::Database;
use engram_mcp::memory::{Memory, MemoryType};
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
/// The dispatch handler must store a `Decision` row containing "Postgres".
#[test]
fn hook_event_reads_payload_from_stdin_without_flag() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    // Prompt must match the cue regex and be >= 40 chars.
    let stdin_json =
        r#"{"prompt":"we decided to switch to Postgres because of write throughput issues"}"#;

    spawn_hook_event("UserPromptSubmit", stdin_json, &db_path).success();

    let memories = read_memories(&db_path);
    let decision = memories
        .iter()
        .find(|m| m.memory_type == MemoryType::Decision && m.content.contains("Postgres"));
    assert!(
        decision.is_some(),
        "expected a Decision row containing 'Postgres', got:\n{:#?}",
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
        .arg("hook-event")
        .arg("UserPromptSubmit")
        .arg("--payload")
        .arg(payload_json)
        // Unrelated stdin — must be ignored when --payload is given.
        .write_stdin("garbage that should not be stored")
        .assert()
        .success();

    let memories = read_memories(&db_path);

    // The --payload content should produce a Decision.
    let from_payload = memories
        .iter()
        .find(|m| m.memory_type == MemoryType::Decision && m.content.contains("auth module"));
    assert!(
        from_payload.is_some(),
        "expected a Decision row for 'auth module', got:\n{:#?}",
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

/// Blocker B2: `Stop` on a non-git workspace stores a Fact fallback, not an error.
///
/// Spawn from a temp dir that is NOT a git repo. The branch detection returns
/// `None`, so `store_handoff_or_fact` falls back to a plain Fact row.
#[test]
fn stop_on_non_git_workspace_does_not_error() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    // Build a 300-char summary (Stop requires >= 200 chars).
    let long_summary = format!(
        "Session ended after completing the authentication refactor. {}",
        "x".repeat(250)
    );
    let stdin_json = format!(r#"{{"last_assistant_message":"{}"}}"#, long_summary);

    let mut cmd = Command::cargo_bin("engram-cli").unwrap();
    cmd.env("ENGRAM_DB", &db_path)
        .env("ENGRAM_PROJECT", "test-bin-project")
        // Run from the non-git temp dir so branch detection returns None.
        .current_dir(tmp.path())
        .arg("hook-event")
        .arg("Stop")
        .write_stdin(stdin_json)
        .assert()
        .success();

    let memories = read_memories(&db_path);

    // Must have stored a Fact row (not a Handoff) since there's no git branch.
    let fact_row = memories.iter().find(|m| m.memory_type == MemoryType::Fact);
    assert!(
        fact_row.is_some(),
        "expected a Fact row for the Stop fallback, got:\n{:#?}",
        memories
    );

    // The Fact must NOT be a Handoff.
    let handoff_row = memories
        .iter()
        .find(|m| m.memory_type == MemoryType::Handoff);
    assert!(
        handoff_row.is_none(),
        "must not store a Handoff on non-git workspace, got:\n{:#?}",
        memories
    );

    // The row must carry both the "hook" and "stop_fallback" tags.
    let fact = fact_row.unwrap();
    assert!(
        fact.tags.contains(&"hook".to_string()),
        "expected 'hook' tag, got tags: {:?}",
        fact.tags
    );
    assert!(
        fact.tags.contains(&"stop_fallback".to_string()),
        "expected 'stop_fallback' tag, got tags: {:?}",
        fact.tags
    );
}

/// PreCompact: custom_instructions content is captured in the stored Fact.
///
/// When `custom_instructions` is non-empty the dispatch handler stores its
/// value directly rather than generating a timestamp summary.
#[test]
fn pre_compact_uses_custom_instructions_when_present() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    let stdin_json = r#"{"trigger":"manual","custom_instructions":"focus on auth code paths"}"#;

    let mut cmd = Command::cargo_bin("engram-cli").unwrap();
    cmd.env("ENGRAM_DB", &db_path)
        .env("ENGRAM_PROJECT", "test-bin-project")
        // Non-git cwd forces Fact fallback.
        .current_dir(tmp.path())
        .arg("hook-event")
        .arg("PreCompact")
        .write_stdin(stdin_json)
        .assert()
        .success();

    let memories = read_memories(&db_path);

    let row = memories.iter().find(|m| {
        m.memory_type == MemoryType::Fact && m.content.contains("focus on auth code paths")
    });
    assert!(
        row.is_some(),
        "expected a Fact row containing the custom_instructions text, got:\n{:#?}",
        memories
    );
}

/// PreCompact: a timestamp summary is generated when custom_instructions is empty.
///
/// When `custom_instructions` is empty the handler stores a
/// "pre-compaction checkpoint at <rfc3339>" string.
#[test]
fn pre_compact_falls_back_to_timestamp_summary_when_empty() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    let stdin_json = r#"{"trigger":"auto","custom_instructions":""}"#;

    let mut cmd = Command::cargo_bin("engram-cli").unwrap();
    cmd.env("ENGRAM_DB", &db_path)
        .env("ENGRAM_PROJECT", "test-bin-project")
        // Non-git cwd forces Fact fallback.
        .current_dir(tmp.path())
        .arg("hook-event")
        .arg("PreCompact")
        .write_stdin(stdin_json)
        .assert()
        .success();

    let memories = read_memories(&db_path);

    let row = memories.iter().find(|m| {
        m.memory_type == MemoryType::Fact && m.content.contains("pre-compaction checkpoint at")
    });
    assert!(
        row.is_some(),
        "expected a Fact row with 'pre-compaction checkpoint at' timestamp, got:\n{:#?}",
        memories
    );
}
