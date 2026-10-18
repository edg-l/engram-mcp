//! The shared embedding process, driven through the real `engram-cli` binary.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_engram-cli"))
}

/// `engram-cli` against a scratch store, with sharing on and its socket in `runtime`.
fn cli(runtime: &Path, db: &Path, shared: bool, args: &[&str]) -> std::process::Output {
    let output = Command::new(bin())
        .args(args)
        .env("XDG_RUNTIME_DIR", runtime)
        .env("ENGRAM_DB", db)
        .env("ENGRAM_PROJECT", "embed-test")
        .env("ENGRAM_EMBED_SHARED", if shared { "1" } else { "0" })
        .env("ENGRAM_EMBED_IDLE_SECS", "2")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn wait_until(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(unix)]
#[test]
fn shared_process_answers_like_the_local_model_and_exits_when_idle() {
    // A short base path: a Unix socket path is limited to about 100 bytes.
    let dir = tempfile::Builder::new()
        .prefix("et")
        .tempdir_in("/tmp")
        .unwrap();
    let runtime = dir.path().join("r");
    std::fs::create_dir(&runtime).unwrap();
    let db = dir.path().join("m.db");
    let socket = runtime
        .join("engram")
        .join(format!("embed-{}.sock", env!("CARGO_PKG_VERSION")));

    cli(
        &runtime,
        &db,
        true,
        &["store", "The deploy window is Tuesday mornings"],
    );
    assert!(
        socket.exists(),
        "the store should have started the shared process"
    );

    let query = ["--json", "query", "when do we deploy"];
    let shared: serde_json::Value =
        serde_json::from_slice(&cli(&runtime, &db, true, &query).stdout).unwrap();
    let local: serde_json::Value =
        serde_json::from_slice(&cli(&runtime, &db, false, &query).stdout).unwrap();
    assert_eq!(
        shared["memories"][0]["score"],
        local["memories"][0]["score"]
    );
    assert!(shared["count"].as_u64().unwrap() > 0);

    // Idle for longer than ENGRAM_EMBED_IDLE_SECS: the process exits and removes its socket.
    wait_until("the idle exit", || !socket.exists());

    // The next call starts it again.
    cli(&runtime, &db, true, &query);
    assert!(socket.exists());
    wait_until("the second idle exit", || !socket.exists());
}
