//! Tests that `handoff create` and `adr create` never block on stdin when they
//! cannot get an interactive answer, and that the `engram` server binary explains
//! itself instead of failing an MCP handshake against a terminal.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(30);

fn bin(name: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
    let release = manifest_dir.join("release").join(name);
    if release.exists() {
        release
    } else {
        manifest_dir.join("debug").join(name)
    }
}

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Wait for a child that still has an open stdin pipe. Returns `None` if it is
/// still running at the deadline, which is what a prompt-on-stdin hang looks like.
fn wait_with_open_stdin(mut child: Child) -> Option<Run> {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        match child.try_wait().expect("try_wait failed") {
            Some(status) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                child
                    .stdout
                    .take()
                    .unwrap()
                    .read_to_string(&mut stdout)
                    .ok();
                child
                    .stderr
                    .take()
                    .unwrap()
                    .read_to_string(&mut stderr)
                    .ok();
                return Some(Run {
                    code: status.code(),
                    stdout,
                    stderr,
                });
            }
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

/// Spawn engram-cli with stdin held open as an idle pipe: an interactive prompt
/// would block here rather than seeing EOF.
fn run_cli(db: &std::path::Path, args: &[&str]) -> Option<Run> {
    let child = Command::new(bin("engram-cli"))
        .env("ENGRAM_DB", db)
        .env("ENGRAM_PROJECT", "non-interactive-test")
        .env("ENGRAM_BRANCH", "feat/non-interactive")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn engram-cli");
    wait_with_open_stdin(child)
}

#[test]
fn handoff_create_does_not_prompt_when_non_interactive() {
    if !bin("engram-cli").exists() {
        eprintln!("engram-cli not built; skipping");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");

    let run = run_cli(
        &db,
        &[
            "handoff",
            "create",
            "--non-interactive",
            "--summary",
            "Wired up the release pipeline",
            "--todos",
            "Tag the release",
        ],
    )
    .expect("handoff create --non-interactive must not wait on stdin");

    assert_eq!(
        run.code,
        Some(0),
        "stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.stdout.contains("Handoff created:"),
        "stdout: {}",
        run.stdout
    );
    // Sections that were not passed are simply left empty; no prompt is printed.
    assert!(
        !run.stdout.contains("blank line to finish"),
        "no prompt should be printed: {}",
        run.stdout
    );
}

#[test]
fn handoff_create_skips_prompts_when_stdin_is_not_a_terminal() {
    if !bin("engram-cli").exists() {
        eprintln!("engram-cli not built; skipping");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");

    // No --non-interactive flag: a piped stdin cannot answer, so prompting is
    // skipped anyway. This is the CI / agent case that used to hang.
    let run = run_cli(
        &db,
        &[
            "handoff",
            "create",
            "--summary",
            "Session with no tty attached",
        ],
    )
    .expect("handoff create must not wait on a non-terminal stdin");

    assert_eq!(
        run.code,
        Some(0),
        "stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.stdout.contains("Handoff created:"),
        "stdout: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("Project: non-interactive-test"),
        "create should report where the handoff landed: {}",
        run.stdout
    );
}

#[test]
fn handoff_create_without_summary_fails_fast() {
    if !bin("engram-cli").exists() {
        eprintln!("engram-cli not built; skipping");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");

    let run =
        run_cli(&db, &["handoff", "create", "--non-interactive"]).expect("must not wait on stdin");

    assert_ne!(run.code, Some(0));
    assert!(
        run.stderr.contains("--summary") || run.stdout.contains("--summary"),
        "error should name the flag to pass\nstdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
}

#[test]
fn adr_create_is_non_interactive_with_flags() {
    if !bin("engram-cli").exists() {
        eprintln!("engram-cli not built; skipping");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");

    let run = run_cli(
        &db,
        &[
            "adr",
            "create",
            "--non-interactive",
            "--title",
            "Record decisions as ADRs",
            "--decision",
            "Every structural decision gets an ADR.",
        ],
    )
    .expect("adr create --non-interactive must not wait on stdin");

    assert_eq!(
        run.code,
        Some(0),
        "stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.stdout.contains("ADR-0001 created"),
        "stdout: {}",
        run.stdout
    );

    // A required section that was not passed is an error, not a prompt.
    let run = run_cli(
        &db,
        &[
            "adr",
            "create",
            "--non-interactive",
            "--title",
            "No decision",
        ],
    )
    .expect("must not wait on stdin");
    assert_ne!(run.code, Some(0));
    assert!(
        run.stderr.contains("--decision") || run.stdout.contains("--decision"),
        "stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
}

#[test]
fn engram_binary_explains_itself_instead_of_serving() {
    if !bin("engram").exists() {
        eprintln!("engram not built; skipping");
        return;
    }

    for args in [vec!["--help"], vec!["store", "something"]] {
        let child = Command::new(bin("engram"))
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn engram");
        let run =
            wait_with_open_stdin(child).expect("engram must not start a server for arguments");

        assert!(
            run.stdout.contains("engram-cli"),
            "usage notice should point at engram-cli, got: {}",
            run.stdout
        );
        assert!(
            !run.stderr.contains("ConnectionClosed"),
            "should not attempt an MCP handshake: {}",
            run.stderr
        );
    }

    // `--help` is a successful request for help; an unsupported argument is not.
    let help = Command::new(bin("engram"))
        .arg("--help")
        .output()
        .expect("spawn");
    assert_eq!(help.status.code(), Some(0));

    let bad = Command::new(bin("engram"))
        .arg("store")
        .output()
        .expect("spawn");
    assert_eq!(bad.status.code(), Some(2));
}
