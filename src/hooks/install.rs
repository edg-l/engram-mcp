//! Install/uninstall/status subcommands that manage `~/.claude/settings.json`.
//!
//! # Tagging strategy
//!
//! Each hook entry added by `engram-cli` carries a `"_source": "engram-cli"` key
//! directly on the inner hook object (inside the `"hooks"` array of a matcher block).
//! Claude Code only enforces `type` and `command`; extra keys are passed through
//! silently. This lets `uninstall` find and remove exactly our entries without
//! touching any other user-authored hooks.
//!
//! # Atomic write
//!
//! We serialize to `<path>.tmp`, fsync the file, then `rename` it over the original.
//! Before the first write we copy the existing file to
//! `<path>.bak.<rfc3339-timestamp-no-colons>` so the user can recover.

use std::path::PathBuf;

use serde_json::{Map, Value};

use crate::error::MemoryError;

/// Events managed by `engram-cli hooks install`.
///
/// `SessionStart` is intentionally absent — the canonical loader is
/// `scripts/engram-hook.sh` and must not be duplicated here.
/// `Stop` and `PreCompact` are intentionally absent — both are no-ops in
/// the dispatch layer, so installing wiring for them serves no purpose.
/// `PostToolUse` is intentionally absent — the dispatch handler still
/// exists for users who wire it manually, but auto-installing it produces
/// too much low-signal noise to be useful by default.
const MANAGED_EVENTS: &[(&str, u32)] = &[
    ("UserPromptSubmit", 4),
    ("SessionEnd", 5),
    ("SubagentStop", 4),
];

const SOURCE_KEY: &str = "_source";
const SOURCE_VALUE: &str = "engram-cli";

// ── Public report types ────────────────────────────────────────────────────

/// Report returned by [`install`].
pub struct InstallReport {
    /// Events for which a new entry was written.
    pub added: Vec<String>,
    /// Events whose entry was already present and left unchanged.
    pub skipped: Vec<String>,
    /// Path of the backup file created before writing, if any.
    pub backup_path: Option<PathBuf>,
    /// Path of the settings file that was written.
    pub settings_path: PathBuf,
}

/// Report returned by [`uninstall`].
pub struct UninstallReport {
    /// Events whose `engram-cli` entry was removed.
    pub removed: Vec<String>,
    /// Path of the backup file created before writing, if any.
    pub backup_path: Option<PathBuf>,
    /// Path of the settings file that was written.
    pub settings_path: PathBuf,
}

/// Report returned by [`status`].
pub struct StatusReport {
    /// Path of the settings file that was read.
    pub settings_path: PathBuf,
    /// Event names where an `engram-cli` entry exists.
    pub managed: Vec<String>,
    /// Event names where another (non-engram) entry is also registered.
    pub shadowed: Vec<String>,
}

// ── Path helper ───────────────────────────────────────────────────────────

fn settings_path() -> Result<PathBuf, MemoryError> {
    let home = dirs::home_dir().ok_or_else(|| {
        MemoryError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "could not determine home directory",
        ))
    })?;
    Ok(home.join(".claude").join("settings.json"))
}

// ── JSON helpers ──────────────────────────────────────────────────────────

/// Read settings.json; return an empty object if the file does not exist.
fn read_settings(path: &PathBuf) -> Result<Value, MemoryError> {
    match std::fs::read_to_string(path) {
        Ok(s) => {
            let v: Value = serde_json::from_str(&s)?;
            Ok(v)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Value::Object(Map::new())),
        Err(e) => Err(MemoryError::Io(e)),
    }
}

/// Backup + atomic write: write `<path>.tmp`, fsync, rename.
fn write_settings_atomic(path: &PathBuf, value: &Value) -> Result<Option<PathBuf>, MemoryError> {
    // Create parent directory if it does not exist.
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Backup existing file.
    let backup_path = if path.exists() {
        let ts = chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
            .replace(':', "");
        let bak = path.with_extension(format!("json.bak.{}", ts));
        std::fs::copy(path, &bak)?;
        Some(bak)
    } else {
        None
    };

    // Write to temp file.
    let tmp_path = path.with_extension("json.tmp");
    {
        use std::io::Write;
        let serialized = serde_json::to_string_pretty(value)?;
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(serialized.as_bytes())?;
        f.sync_all()?;
    }

    // Atomic rename.
    std::fs::rename(&tmp_path, path)?;

    Ok(backup_path)
}

/// Build the matcher-block Value for a given event.
fn make_matcher_block(event: &str, timeout: u32) -> Value {
    let hook_obj = serde_json::json!({
        "type": "command",
        "command": format!("engram-cli hook-event {}", event),
        "timeout": timeout,
        SOURCE_KEY: SOURCE_VALUE,
    });
    serde_json::json!({
        "matcher": "",
        "hooks": [hook_obj],
    })
}

/// Return `true` if the given matcher-block Value was written by `engram-cli`.
fn block_is_ours(block: &Value) -> bool {
    if let Some(hooks_arr) = block.get("hooks").and_then(|v| v.as_array()) {
        hooks_arr.iter().any(|h| {
            h.get(SOURCE_KEY)
                .and_then(|v| v.as_str())
                .map(|s| s == SOURCE_VALUE)
                .unwrap_or(false)
        })
    } else {
        false
    }
}

// ── Public functions ──────────────────────────────────────────────────────

/// Install `engram-cli` hook entries into `~/.claude/settings.json`.
///
/// Idempotent: if a matching `_source=engram-cli` block already exists for an
/// event it is reported as `skipped`; no duplicate is appended.
pub fn install() -> Result<InstallReport, MemoryError> {
    let path = settings_path()?;
    let mut root = read_settings(&path)?;

    // Ensure top-level value is an object.
    let root_obj = root.as_object_mut().ok_or_else(|| {
        MemoryError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "settings.json top-level is not a JSON object; refusing to overwrite",
        ))
    })?;

    // Ensure "hooks" is an object (or absent).
    if let Some(existing_hooks) = root_obj.get("hooks")
        && !existing_hooks.is_object()
    {
        return Err(MemoryError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "settings.json: \"hooks\" key is not a JSON object; refusing to overwrite",
        )));
    }

    let hooks_obj = root_obj
        .entry("hooks")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .expect("ensured above");

    let mut added = Vec::new();
    let mut skipped = Vec::new();

    for (event, timeout) in MANAGED_EVENTS {
        let arr = hooks_obj
            .entry(*event)
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| {
                MemoryError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("settings.json: hooks[\"{}\"] is not an array", event),
                ))
            })?;

        if arr.iter().any(block_is_ours) {
            skipped.push(event.to_string());
        } else {
            arr.push(make_matcher_block(event, *timeout));
            added.push(event.to_string());
        }
    }

    let backup_path = if added.is_empty() {
        // Nothing changed; skip backup and write.
        None
    } else {
        write_settings_atomic(&path, &root)?
    };

    Ok(InstallReport {
        added,
        skipped,
        backup_path,
        settings_path: path,
    })
}

/// Remove all `engram-cli` hook entries from `~/.claude/settings.json`.
///
/// User-authored entries are preserved. Events whose matcher list becomes
/// empty after removal are deleted from the hooks object.
pub fn uninstall() -> Result<UninstallReport, MemoryError> {
    let path = settings_path()?;
    let mut root = read_settings(&path)?;

    // If file was absent, nothing to do.
    if root.as_object().map(|o| o.is_empty()).unwrap_or(false) && !path.exists() {
        return Ok(UninstallReport {
            removed: Vec::new(),
            backup_path: None,
            settings_path: path,
        });
    }

    let root_obj = root.as_object_mut().ok_or_else(|| {
        MemoryError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "settings.json top-level is not a JSON object; refusing to modify",
        ))
    })?;

    let mut removed = Vec::new();

    if let Some(hooks_val) = root_obj.get_mut("hooks")
        && let Some(hooks_obj) = hooks_val.as_object_mut()
    {
        // Collect event names to process; can't iterate mutably while also mutating.
        let events: Vec<String> = hooks_obj.keys().cloned().collect();

        for event in &events {
            if let Some(arr_val) = hooks_obj.get_mut(event)
                && let Some(arr) = arr_val.as_array_mut()
            {
                let before = arr.len();
                arr.retain(|block| !block_is_ours(block));
                let after = arr.len();
                if after < before {
                    removed.push(event.clone());
                }
            }
        }

        // Remove events whose matcher list is now empty.
        hooks_obj.retain(|_k, v| v.as_array().map(|a| !a.is_empty()).unwrap_or(true));
    }

    let backup_path = if removed.is_empty() {
        None
    } else {
        write_settings_atomic(&path, &root)?
    };

    Ok(UninstallReport {
        removed,
        backup_path,
        settings_path: path,
    })
}

/// Report which events are currently managed by `engram-cli` and which have
/// additional (non-engram) hooks registered alongside ours.
pub fn status() -> Result<StatusReport, MemoryError> {
    let path = settings_path()?;
    let root = read_settings(&path)?;

    let mut managed = Vec::new();
    let mut shadowed = Vec::new();

    if let Some(hooks_obj) = root.get("hooks").and_then(|v| v.as_object()) {
        for (event, arr_val) in hooks_obj {
            if let Some(arr) = arr_val.as_array() {
                let has_ours = arr.iter().any(block_is_ours);
                let has_others = arr.iter().any(|b| !block_is_ours(b));

                if has_ours {
                    managed.push(event.clone());
                }
                if has_ours && has_others {
                    shadowed.push(event.clone());
                }
            }
        }
    }

    Ok(StatusReport {
        settings_path: path,
        managed,
        shadowed,
    })
}
