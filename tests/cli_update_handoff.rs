//! CLI tests for `engram-cli update` against a handoff memory.
//!
//! Verifies that updating a handoff's content through `cmd_update` rebuilds the
//! `handoff_sections` sidecar and its per-section embeddings (rather than only
//! re-embedding the whole memory), and that malformed handoff content is rejected
//! before anything is written.

use std::path::PathBuf;
use std::process::Command;

fn cli_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_engram-cli"))
}

fn run_cli(
    db_path: &std::path::Path,
    project: &str,
    args: &[&str],
) -> (std::process::ExitStatus, String, String) {
    let bin = cli_bin();
    if !bin.exists() {
        eprintln!(
            "engram-cli binary not found at {}; skipping CLI spawn",
            bin.display()
        );
        let status = Command::new("true").status().unwrap();
        return (status, String::new(), String::new());
    }

    let out = Command::new(&bin)
        .env("ENGRAM_DB", db_path.to_str().unwrap())
        .env("ENGRAM_PROJECT", project)
        .env("ENGRAM_BRANCH", "feat/cli-update-handoff-test")
        .args(args)
        .output()
        .expect("failed to spawn engram-cli");

    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    (out.status, stdout, stderr)
}

fn extract_handoff_id(stdout: &str) -> String {
    let marker = "Handoff created: ";
    let pos = stdout
        .find(marker)
        .expect("could not find 'Handoff created:' in stdout");
    let after = &stdout[pos + marker.len()..];
    after
        .split_whitespace()
        .next()
        .expect("no token after 'Handoff created:'")
        .to_string()
}

/// `update --content` on a handoff must rebuild the sidecar's section embeddings, not
/// just re-embed the whole memory: a new decision only lands in `handoff_search` results
/// if the sidecar was rebuilt, and the old decision's section text must be gone from it.
#[test]
fn cli_update_handoff_content_rebuilds_sections() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let project = "cli-update-handoff-proj";

    let bin = cli_bin();
    if !bin.exists() {
        eprintln!("engram-cli binary not found; skipping test");
        return;
    }

    let (status, stdout, stderr) = run_cli(
        &db_path,
        project,
        &[
            "handoff",
            "create",
            "--summary",
            "Initial handoff summary.",
            "--decisions",
            "Cache widget metadata in Redis with a five minute TTL.",
        ],
    );
    assert!(
        status.success(),
        "handoff create failed (exit {})\nstdout: {}\nstderr: {}",
        status,
        stdout,
        stderr
    );
    let id = extract_handoff_id(&stdout);

    let new_content = "## Summary\n\n\
        Updated handoff summary.\n\n\
        ## Decisions\n\n\
        - Migrated the auth flow to OAuth2 device code grant.\n";

    let (status, stdout, stderr) = run_cli(
        &db_path,
        project,
        &["update", &id, "--content", new_content],
    );
    assert!(
        status.success(),
        "update failed (exit {})\nstdout: {}\nstderr: {}",
        status,
        stdout,
        stderr
    );

    // The new decision must be findable by section search: only true if the sidecar's
    // section embeddings were rebuilt from the new content, not just the whole-memory one.
    let (status, stdout, stderr) = run_cli(
        &db_path,
        project,
        &["handoff", "search", "OAuth2 device code grant"],
    );
    assert!(
        status.success(),
        "handoff search failed (exit {})\nstdout: {}\nstderr: {}",
        status,
        stdout,
        stderr
    );
    assert!(
        stdout.contains("OAuth2 device code grant"),
        "expected updated decision text in handoff search results, got:\n{}",
        stdout
    );

    // The old decision text must no longer live in the sidecar.
    let (status, stdout, stderr) = run_cli(&db_path, project, &["handoff", "show", &id]);
    assert!(
        status.success(),
        "handoff show failed (exit {})\nstdout: {}\nstderr: {}",
        status,
        stdout,
        stderr
    );
    assert!(
        !stdout.contains("Redis"),
        "old decision text should have been replaced, got:\n{}",
        stdout
    );
    assert!(
        stdout.contains("OAuth2 device code grant"),
        "expected new decision text in handoff show output, got:\n{}",
        stdout
    );
}

/// `update --content` on a handoff must reject content missing the required `## Summary`
/// heading before writing anything, leaving the previous content in place.
#[test]
fn cli_update_handoff_rejects_malformed_content() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let project = "cli-update-handoff-malformed-proj";

    let bin = cli_bin();
    if !bin.exists() {
        eprintln!("engram-cli binary not found; skipping test");
        return;
    }

    let (status, stdout, stderr) = run_cli(
        &db_path,
        project,
        &["handoff", "create", "--summary", "Initial handoff summary."],
    );
    assert!(
        status.success(),
        "handoff create failed (exit {})\nstdout: {}\nstderr: {}",
        status,
        stdout,
        stderr
    );
    let id = extract_handoff_id(&stdout);

    let (status, _stdout, stderr) = run_cli(
        &db_path,
        project,
        &[
            "update",
            &id,
            "--content",
            "just some plain text, no headings",
        ],
    );
    assert!(
        !status.success(),
        "update with malformed handoff content should have failed"
    );
    assert!(
        stderr.contains("Summary"),
        "expected error to name the missing heading, got:\n{}",
        stderr
    );

    // The memory must be untouched: still showing the original summary.
    let (status, stdout, stderr) = run_cli(&db_path, project, &["handoff", "show", &id]);
    assert!(
        status.success(),
        "handoff show failed (exit {})\nstdout: {}\nstderr: {}",
        status,
        stdout,
        stderr
    );
    assert!(
        stdout.contains("Initial handoff summary."),
        "expected original content to survive a rejected update, got:\n{}",
        stdout
    );
}
