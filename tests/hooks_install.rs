//! Round-trip integration tests for `engram-cli hooks install/uninstall/status`.
//!
//! Each test uses a temporary directory as `HOME` so the real
//! `~/.claude/settings.json` is never touched.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;

// ── helpers ──────────────────────────────────────────────────────────────────

fn engram_cli(home: &std::path::Path) -> Command {
    let mut cmd = Command::cargo_bin("engram-cli").unwrap();
    cmd.env("HOME", home);
    // Point the DB at the temp dir so we don't touch the real DB.
    cmd.env("ENGRAM_DB", home.join("test.db").to_str().unwrap());
    cmd
}

fn read_settings(home: &std::path::Path) -> Value {
    let path = home.join(".claude").join("settings.json");
    let s = fs::read_to_string(&path).expect("settings.json not found");
    serde_json::from_str(&s).expect("settings.json is not valid JSON")
}

/// Count how many inner hook objects across ALL events carry `_source: "engram-cli"`.
fn count_engram_hooks(settings: &Value) -> usize {
    let hooks = match settings.get("hooks").and_then(|v| v.as_object()) {
        Some(h) => h,
        None => return 0,
    };
    let mut total = 0;
    for (_event, arr_val) in hooks {
        if let Some(arr) = arr_val.as_array() {
            for matcher_block in arr {
                if let Some(inner_hooks) = matcher_block.get("hooks").and_then(|v| v.as_array()) {
                    for h in inner_hooks {
                        if h.get("_source").and_then(|v| v.as_str()) == Some("engram-cli") {
                            total += 1;
                        }
                    }
                }
            }
        }
    }
    total
}

/// Return the set of event names for which an `engram-cli` hook exists.
fn engram_hook_events(settings: &Value) -> Vec<String> {
    let hooks = match settings.get("hooks").and_then(|v| v.as_object()) {
        Some(h) => h,
        None => return Vec::new(),
    };
    let mut events = Vec::new();
    for (event, arr_val) in hooks {
        if let Some(arr) = arr_val.as_array() {
            let has_ours = arr.iter().any(|block| {
                block
                    .get("hooks")
                    .and_then(|v| v.as_array())
                    .map(|inner| {
                        inner.iter().any(|h| {
                            h.get("_source").and_then(|v| v.as_str()) == Some("engram-cli")
                        })
                    })
                    .unwrap_or(false)
            });
            if has_ours {
                events.push(event.clone());
            }
        }
    }
    events
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// `hooks install` creates exactly 6 managed entries, none for SessionStart.
/// A backup file is created if a pre-existing settings.json was present.
#[test]
fn install_creates_six_managed_entries_and_backup() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path();
    let claude_dir = home.join(".claude");
    fs::create_dir_all(&claude_dir).unwrap();

    // Pre-seed a settings.json so the backup logic is exercised.
    let initial = serde_json::json!({ "effortLevel": "high" });
    fs::write(
        claude_dir.join("settings.json"),
        serde_json::to_string_pretty(&initial).unwrap(),
    )
    .unwrap();

    engram_cli(home)
        .args(["hooks", "install"])
        .assert()
        .success();

    let settings = read_settings(home);

    // Exactly 6 engram-cli hook entries.
    assert_eq!(
        count_engram_hooks(&settings),
        6,
        "expected 6 engram-cli hook entries, got:\n{}",
        serde_json::to_string_pretty(&settings).unwrap()
    );

    // The 6 expected events are present.
    let mut events = engram_hook_events(&settings);
    events.sort();
    let mut expected = vec![
        "UserPromptSubmit",
        "PostToolUse",
        "Stop",
        "PreCompact",
        "SessionEnd",
        "SubagentStop",
    ];
    expected.sort();
    assert_eq!(events, expected, "managed event names do not match");

    // SessionStart must NOT be present.
    assert!(
        !events.contains(&"SessionStart".to_string()),
        "SessionStart should not be managed"
    );

    // A backup file must exist (we pre-seeded a settings.json).
    let bak_count = fs::read_dir(&claude_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("settings.json.bak.")
        })
        .count();
    assert_eq!(bak_count, 1, "expected exactly one backup file");
}

/// After `install`, user-authored Stop hook survives `uninstall`.
#[test]
fn uninstall_removes_only_engram_entries() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path();
    let claude_dir = home.join(".claude");
    fs::create_dir_all(&claude_dir).unwrap();

    // Pre-seed a settings.json with a user-authored Stop hook (no _source key).
    let user_hook = serde_json::json!({
        "hooks": {
            "Stop": [
                {
                    "matcher": "my-tool",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "echo user-stop-hook"
                        }
                    ]
                }
            ]
        }
    });
    fs::write(
        claude_dir.join("settings.json"),
        serde_json::to_string_pretty(&user_hook).unwrap(),
    )
    .unwrap();

    // Install engram entries.
    engram_cli(home)
        .args(["hooks", "install"])
        .assert()
        .success();

    // Verify both user and engram entries coexist under Stop.
    let after_install = read_settings(home);
    let stop_arr = after_install["hooks"]["Stop"]
        .as_array()
        .expect("Stop should be an array");
    assert_eq!(
        stop_arr.len(),
        2,
        "Stop should have 2 entries after install"
    );

    // Uninstall engram entries.
    engram_cli(home)
        .args(["hooks", "uninstall"])
        .assert()
        .success();

    let after_uninstall = read_settings(home);

    // No engram entries should remain.
    assert_eq!(
        count_engram_hooks(&after_uninstall),
        0,
        "all engram-cli hooks should be removed"
    );

    // User-authored Stop hook must survive.
    let stop_arr_after = after_uninstall["hooks"]["Stop"]
        .as_array()
        .expect("Stop key should still exist with user entry");
    assert_eq!(
        stop_arr_after.len(),
        1,
        "user-authored Stop hook should survive"
    );
    let user_cmd = stop_arr_after[0]["hooks"][0]["command"]
        .as_str()
        .unwrap_or("");
    assert_eq!(user_cmd, "echo user-stop-hook");
}

/// Running `hooks install` twice does not duplicate entries.
#[test]
fn install_is_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path();

    // First install.
    engram_cli(home)
        .args(["hooks", "install"])
        .assert()
        .success();

    let settings_after_first = read_settings(home);
    assert_eq!(count_engram_hooks(&settings_after_first), 6);

    // Second install.
    let output = engram_cli(home)
        .args(["hooks", "install"])
        .output()
        .expect("failed to run second install");
    assert!(output.status.success());

    // Entry count must not increase.
    let settings_after_second = read_settings(home);
    assert_eq!(
        count_engram_hooks(&settings_after_second),
        6,
        "second install should not duplicate entries"
    );

    // Second run output should mention that entries were skipped (not added again).
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("skipped") || stdout.contains("Already present"),
        "second install should report skipped entries; got: {}",
        stdout
    );
}
