//! `engram-cli -p <short-name>` resolves against the store's known project ids
//! rather than requiring the full `git:host/path` or `~/path` form.

use std::path::PathBuf;
use std::process::Command;

fn cli() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_engram-cli"))
}

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn run(db: &std::path::Path, args: &[&str]) -> Run {
    let out = Command::new(cli())
        .env("ENGRAM_DB", db)
        .env("ENGRAM_PROJECT", "home-project")
        .env("ENGRAM_BRANCH", "main")
        .args(args)
        .output()
        .expect("failed to spawn engram-cli");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn short_name_resolves_to_full_project_id() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");
    let full_id = "git:example.com/edgar/antworld";

    // The write path creates the project verbatim under its full id, exactly
    // as if the store had received it from the MCP server.
    let stored = run(
        &db,
        &[
            "-p",
            full_id,
            "store",
            "Antworld uses a custom ECS",
            "-t",
            "fact",
        ],
    );
    assert_eq!(stored.code, Some(0), "{}", stored.stderr);

    // A short name uniquely matching that id's last path segment resolves, and
    // the CLI's normal project-reporting output shows the expanded full id.
    let stats = run(&db, &["-p", "antworld", "stats"]);
    assert_eq!(stats.code, Some(0), "{}", stats.stderr);
    assert!(
        stats.stdout.contains(full_id),
        "expected the full id in stats output: {}",
        stats.stdout
    );

    // Case-insensitively too.
    let stats_upper = run(&db, &["-p", "AntWorld", "stats"]);
    assert_eq!(stats_upper.code, Some(0), "{}", stats_upper.stderr);
    assert!(
        stats_upper.stdout.contains(full_id),
        "expected the full id in stats output: {}",
        stats_upper.stdout
    );
}

#[test]
fn ambiguous_short_name_is_rejected_with_only_the_candidates() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");

    for full_id in [
        "git:example.com/alice/widget",
        "git:example.com/bob/widget",
        "git:example.com/edgar/gadget",
    ] {
        let stored = run(
            &db,
            &["-p", full_id, "store", "A fact about widgets", "-t", "fact"],
        );
        assert_eq!(stored.code, Some(0), "{}", stored.stderr);
    }

    let stats = run(&db, &["-p", "widget", "stats"]);
    assert_eq!(stats.code, Some(1));
    assert!(stats.stderr.contains("git:example.com/alice/widget"));
    assert!(stats.stderr.contains("git:example.com/bob/widget"));
    assert!(!stats.stderr.contains("gadget"));
}
