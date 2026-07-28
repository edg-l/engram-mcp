//! Tests for `engram-cli --json`.
//!
//! Every supported read command must emit a single parseable JSON document on
//! stdout with nothing else mixed in, and commands that cannot render JSON must
//! reject the flag rather than printing prose a caller expected to parse.

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
        .env("ENGRAM_PROJECT", "json-test")
        .env("ENGRAM_BRANCH", "feat/json")
        .args(args)
        .output()
        .expect("failed to spawn engram-cli");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Seed one memory, one handoff, and one ADR so every read command has content.
fn seed(db: &std::path::Path) -> String {
    let stored = run(
        db,
        &[
            "store",
            "The gateway rate-limits with a token bucket",
            "-t",
            "fact",
            "-g",
            "gateway,limits",
        ],
    );
    assert_eq!(stored.code, Some(0), "{}", stored.stderr);

    let handoff = run(
        db,
        &[
            "handoff",
            "create",
            "--non-interactive",
            "--summary",
            "Tuned the gateway rate limiter",
            "--todos",
            "Measure p99 under load",
        ],
    );
    assert_eq!(handoff.code, Some(0), "{}", handoff.stderr);

    let adr = run(
        db,
        &[
            "adr",
            "create",
            "--non-interactive",
            "--title",
            "Rate-limit at the gateway",
            "--context",
            "Per-service limiters drifted apart.",
            "--decision",
            "One token-bucket limiter at the gateway.",
            "--consequences",
            "Single choke point to monitor.",
        ],
    );
    assert_eq!(adr.code, Some(0), "{}", adr.stderr);

    // The memory ID for `show`, read back through --json list.
    let listed = run(db, &["--json", "list"]);
    let value: serde_json::Value = serde_json::from_str(&listed.stdout).expect("list json");
    value["memories"][0]["id"]
        .as_str()
        .expect("a memory id")
        .to_string()
}

#[test]
fn every_supported_command_emits_parseable_json() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");
    let memory_id = seed(&db);

    let commands: Vec<Vec<&str>> = vec![
        vec!["--json", "stats"],
        vec!["--json", "projects"],
        vec!["--json", "list"],
        vec!["--json", "show", &memory_id],
        vec!["--json", "query", "rate limiting"],
        vec!["--json", "context", "gateway limits"],
        vec!["--json", "handoff", "resume"],
        vec!["--json", "handoff", "search", "load"],
        vec!["--json", "adr", "list"],
        vec!["--json", "adr", "show", "1"],
    ];

    for args in commands {
        let out = run(&db, &args);
        assert_eq!(
            out.code,
            Some(0),
            "{:?} exited {:?}\nstderr: {}",
            args,
            out.code,
            out.stderr
        );
        serde_json::from_str::<serde_json::Value>(&out.stdout).unwrap_or_else(|e| {
            panic!(
                "{args:?} did not emit valid JSON ({e})\nstdout: {}",
                out.stdout
            )
        });
    }
}

#[test]
fn json_documents_carry_the_expected_shape() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");
    seed(&db);

    let stats: serde_json::Value =
        serde_json::from_str(&run(&db, &["--json", "stats"]).stdout).unwrap();
    assert_eq!(stats["project"].as_str().unwrap(), "json-test");
    assert_eq!(stats["memory_count"].as_u64().unwrap(), 3);
    assert_eq!(stats["adr_count"].as_u64().unwrap(), 1);

    let query: serde_json::Value =
        serde_json::from_str(&run(&db, &["--json", "query", "rate limiting"]).stdout).unwrap();
    assert_eq!(query["query"].as_str().unwrap(), "rate limiting");
    assert!(query["count"].as_u64().unwrap() >= 1);
    let first = &query["memories"][0];
    assert!(first["memory"]["id"].as_str().unwrap().starts_with("mem_"));
    assert!(first["score"].is_number());
    assert!(first["semantic_score"].is_number());

    let resume: serde_json::Value =
        serde_json::from_str(&run(&db, &["--json", "handoff", "resume"]).stdout).unwrap();
    assert_eq!(resume["branch"].as_str().unwrap(), "feat/json");
    assert!(
        !resume["top_sections"].as_array().unwrap().is_empty(),
        "resume should carry scored sections: {resume}"
    );

    let adr: serde_json::Value =
        serde_json::from_str(&run(&db, &["--json", "adr", "show", "1"]).stdout).unwrap();
    assert_eq!(adr["number"].as_u64().unwrap(), 1);
    assert_eq!(adr["status"].as_str().unwrap(), "proposed");
    assert_eq!(adr["title"].as_str().unwrap(), "Rate-limit at the gateway");
}

#[test]
fn empty_results_are_still_json() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");

    // Nothing stored: the documents exist with zero counts rather than prose.
    for args in [
        vec!["--json", "list"],
        vec!["--json", "query", "anything"],
        vec!["--json", "handoff", "search", "anything"],
        vec!["--json", "adr", "list"],
    ] {
        let out = run(&db, &args);
        let value: serde_json::Value = serde_json::from_str(&out.stdout)
            .unwrap_or_else(|e| panic!("{args:?}: {e}\nstdout: {}", out.stdout));
        assert_eq!(
            value["count"].as_u64(),
            Some(0),
            "{args:?} should report a zero count: {value}"
        );
    }
}

#[test]
fn json_is_rejected_where_it_is_not_supported() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");

    for args in [
        vec!["--json", "store", "something", "-t", "fact"],
        vec!["--json", "decay"],
        vec!["--json", "insights"],
    ] {
        let out = run(&db, &args);
        assert_eq!(
            out.code,
            Some(2),
            "{args:?} should reject --json\nstdout: {}",
            out.stdout
        );
        assert!(
            out.stderr.contains("--json is not supported"),
            "{args:?} stderr: {}",
            out.stderr
        );
    }
}

#[test]
fn piping_into_an_early_exiting_reader_does_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");

    // Content large enough that the output cannot fit in the pipe buffer, so the
    // writer is still writing when the reader exits.
    let big = "The runbook step repeats for every region. ".repeat(3000);
    let stored = run(&db, &["store", &big, "-t", "fact"]);
    assert_eq!(stored.code, Some(0), "{}", stored.stderr);

    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("'{}' --json list | head -1", cli().display()))
        .env("ENGRAM_DB", &db)
        .env("ENGRAM_PROJECT", "json-test")
        .output()
        .expect("failed to spawn pipeline");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("panicked") && !stderr.contains("Broken pipe"),
        "piped output should die quietly on SIGPIPE, got: {stderr}"
    );
}
