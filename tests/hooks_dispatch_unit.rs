use engram_mcp::db::Database;
use engram_mcp::hooks::HookEvent;
use engram_mcp::hooks::dispatch::{DispatchOutcome, dispatch};
use std::sync::{Mutex, MutexGuard, OnceLock};
use tempfile::TempDir;

// ---- helpers ----

fn fresh_db() -> (TempDir, Database) {
    let dir = TempDir::new().expect("tempdir");
    let db_path = dir.path().join("test.db");
    let db = Database::open(&db_path).expect("open db");
    db.get_or_create_project("test-project", "Test Project")
        .expect("create project");
    (dir, db)
}

/// Serialise tests that mutate env vars so they cannot race each other.
/// Returns a guard; drop it to release the lock at the end of the test.
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// RAII guard that removes an env var when dropped, even on panic.
/// Must be acquired while holding the `env_lock()` mutex.
struct EnvGuard(&'static str);
impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: env_lock() serialises all env-mutating tests; no concurrent access.
        unsafe { std::env::remove_var(self.0) }
    }
}

// ---- Task 4a.2 (updated: UserPromptSubmit now defaults off, enable via env) ----

#[test]
fn dispatch_user_prompt_submit_below_min_chars_skipped() {
    let _guard = env_lock();
    // SAFETY: protected by env_lock(); no other env-mutating test runs concurrently.
    unsafe {
        std::env::set_var("ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED", "true");
    }
    let _env = EnvGuard("ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED");
    let (_dir, db) = fresh_db();
    // "fix" is 3 chars, well below the default minimum of 40
    let raw = r#"{"prompt":"fix"}"#;
    let outcome = dispatch(
        HookEvent::UserPromptSubmit,
        raw,
        false,
        &db,
        None,
        "test-project",
    )
    .expect("dispatch ok");
    assert!(
        matches!(outcome, DispatchOutcome::Skipped("prompt_too_short")),
        "expected Skipped(\"prompt_too_short\"), got {:?}",
        outcome
    );
}

// ---- Task 4a.3 (updated: UserPromptSubmit now defaults off, enable via env) ----

#[test]
fn dispatch_user_prompt_submit_no_cue_skipped() {
    let _guard = env_lock();
    // SAFETY: protected by env_lock(); no other env-mutating test runs concurrently.
    unsafe {
        std::env::set_var("ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED", "true");
    }
    let _env = EnvGuard("ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED");
    let (_dir, db) = fresh_db();
    // 71 chars, no cue word from the tightened default regex (decided/chose/prefer/switch)
    let prompt = "this is a sufficiently long string that does not contain the magic words";
    assert!(prompt.len() >= 40, "sanity: prompt must be >= min_chars");
    let raw = format!(r#"{{"prompt":"{}"}}"#, prompt);
    let outcome = dispatch(
        HookEvent::UserPromptSubmit,
        &raw,
        false,
        &db,
        None,
        "test-project",
    )
    .expect("dispatch ok");
    assert!(
        matches!(outcome, DispatchOutcome::Skipped("prompt_no_cue_match")),
        "expected Skipped(\"prompt_no_cue_match\"), got {:?}",
        outcome
    );
}

// ---- Task 4a.4 ----

#[test]
fn dispatch_post_tool_use_success_skipped() {
    let (_dir, db) = fresh_db();
    // exit_code 0 and a benign tool_response — not a failure
    let raw = r#"{"tool_name":"Bash","exit_code":0,"tool_response":{"ok":true}}"#;
    let outcome = dispatch(
        HookEvent::PostToolUse,
        raw,
        false,
        &db,
        None,
        "test-project",
    )
    .expect("dispatch ok");
    assert!(
        matches!(outcome, DispatchOutcome::Skipped("tool_succeeded")),
        "expected Skipped(\"tool_succeeded\"), got {:?}",
        outcome
    );
}

// ---- Task 4a.5 ----

#[test]
fn dispatch_post_tool_use_denylisted_tool_skipped() {
    let (_dir, db) = fresh_db();
    // "Read" is in the default denylist; exit_code 1 triggers the failure path,
    // which then hits allow_tool and yields "tool_denied".
    let raw = r#"{"tool_name":"Read","exit_code":1}"#;
    let outcome = dispatch(
        HookEvent::PostToolUse,
        raw,
        false,
        &db,
        None,
        "test-project",
    )
    .expect("dispatch ok");
    assert!(
        matches!(outcome, DispatchOutcome::Skipped("tool_denied")),
        "expected Skipped(\"tool_denied\"), got {:?}",
        outcome
    );
}

// ---- Task 4a.6 ----

#[test]
fn dispatch_subagent_stop_disabled_by_default() {
    let _guard = env_lock();
    // SAFETY: protected by env_lock(); no other env-mutating test runs concurrently.
    unsafe {
        std::env::remove_var("ENGRAM_HOOK_SUBAGENTSTOP_ENABLED");
    }

    let (_dir, db) = fresh_db();
    let message = "x".repeat(300);
    let raw = format!(
        r#"{{"agent_type":"plan-implementer","last_assistant_message":"{}"}}"#,
        message
    );
    // allow_event is checked before payload parsing, so we get "event_disabled"
    let outcome = dispatch(
        HookEvent::SubagentStop,
        &raw,
        false,
        &db,
        None,
        "test-project",
    )
    .expect("dispatch ok");
    assert!(
        matches!(outcome, DispatchOutcome::Skipped("event_disabled")),
        "expected Skipped(\"event_disabled\"), got {:?}",
        outcome
    );
}

// ---- Task 8: dispatch_stop_is_noop ----

#[test]
fn dispatch_stop_is_noop() {
    let (_dir, db) = fresh_db();
    let raw = r#"{"stop_hook_active":false,"response":"The agent completed the task successfully after extensive analysis."}"#;
    let outcome =
        dispatch(HookEvent::Stop, raw, false, &db, None, "test-project").expect("dispatch ok");
    assert!(
        matches!(outcome, DispatchOutcome::Skipped("stop_noop")),
        "expected Skipped(\"stop_noop\"), got {:?}",
        outcome
    );
}

// ---- Task 8: dispatch_pre_compact_is_noop ----

#[test]
fn dispatch_pre_compact_is_noop() {
    let (_dir, db) = fresh_db();
    let raw = r#"{"compaction_reason":"context_full","current_context_usage":90000,"context_limit":100000}"#;
    let outcome = dispatch(HookEvent::PreCompact, raw, false, &db, None, "test-project")
        .expect("dispatch ok");
    assert!(
        matches!(outcome, DispatchOutcome::Skipped("precompact_noop")),
        "expected Skipped(\"precompact_noop\"), got {:?}",
        outcome
    );
}

// ---- Task 8: dispatch_user_prompt_submit_remember_trigger ----

#[test]
fn dispatch_user_prompt_submit_remember_trigger() {
    let _guard = env_lock();
    // SAFETY: protected by env_lock(); no other env-mutating test runs concurrently.
    unsafe {
        std::env::set_var("ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED", "true");
    }
    let _env = EnvGuard("ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED");
    let (_dir, db) = fresh_db();
    // Contains #remember and is >= 40 chars, but has NO cue word (decided/chose/prefer/switch).
    let prompt = "this is a sufficiently long string with #remember but no cue word at all";
    assert!(prompt.len() >= 40, "sanity: prompt must be >= min_chars");
    let raw = format!(r#"{{"prompt":"{}"}}"#, prompt);
    let outcome = dispatch(
        HookEvent::UserPromptSubmit,
        &raw,
        true, // dry_run so we don't need an embedding service
        &db,
        None,
        "test-project",
    )
    .expect("dispatch ok");
    assert!(
        matches!(outcome, DispatchOutcome::DryRun(_)),
        "expected DryRun (not Skipped), got {:?}",
        outcome
    );
}
