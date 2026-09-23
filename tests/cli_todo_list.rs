//! Tests for `engram-cli todo list`'s compact-by-default rendering and `--full`.

use std::path::PathBuf;
use std::process::Command;

/// The binary built for this test run. `CARGO_BIN_EXE_*` always points at the
/// current profile's build, so a stale binary from another profile cannot be
/// picked up.
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
        .env("ENGRAM_PROJECT", "todo-list-cli-test")
        .args(args)
        .output()
        .expect("failed to spawn engram-cli");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Default output is a compact title; `--full` renders the todo's text in full, as before
/// this default existed.
#[test]
fn todo_list_defaults_to_compact_and_full_shows_the_whole_text() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db.sqlite");

    let long = "Investigate the flaky connection pool test. It reproduces under load and \
                only on the CI runners, never locally.";
    let added = run(&db, &["todo", "add", long]);
    assert_eq!(added.code, Some(0), "{}", added.stderr);
    let id = added
        .stdout
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("add ok: "))
        .expect("add must report an id")
        .to_string();

    let compact = run(&db, &["todo", "list"]);
    assert_eq!(compact.code, Some(0), "{}", compact.stderr);
    assert!(
        compact
            .stdout
            .contains("Investigate the flaky connection pool test. ("),
        "got: {}",
        compact.stdout
    );
    assert!(
        !compact.stdout.contains("only on the CI runners"),
        "compact mode must not print the full text; got: {}",
        compact.stdout
    );
    assert!(compact.stdout.contains(&id), "got: {}", compact.stdout);

    let full = run(&db, &["todo", "list", "--full"]);
    assert_eq!(full.code, Some(0), "{}", full.stderr);
    assert!(full.stdout.contains(long), "got: {}", full.stdout);
}
