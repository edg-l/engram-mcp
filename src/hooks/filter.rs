//! Noise filter for hook events. All configuration is read from environment variables.

use crate::hooks::HookEvent;
use regex::Regex;
use std::borrow::Cow;
use std::sync::OnceLock;

/// Returns `true` if the event is enabled for hook processing.
///
/// Reads `ENGRAM_HOOK_<UPPERCASE_VARIANT>_ENABLED` (e.g. `ENGRAM_HOOK_STOP_ENABLED`).
/// All events default to enabled **except** `SubagentStop`, which defaults to `false`.
pub fn allow_event(event: HookEvent) -> bool {
    let var_name = format!("ENGRAM_HOOK_{}_ENABLED", event_env_name(event));
    match std::env::var(&var_name).as_deref() {
        Ok("0") | Ok("false") | Ok("no") => false,
        Ok("1") | Ok("true") | Ok("yes") => true,
        _ => {
            // Default: SubagentStop is off, everything else is on
            event != HookEvent::SubagentStop
        }
    }
}

/// Returns `true` if the named tool should be processed by hooks.
///
/// Allowlist (checked first): `ENGRAM_HOOK_TOOL_ALLOWLIST` — comma-separated names.
///   If set and non-empty, only those tools are allowed.
/// Denylist (fallback): `ENGRAM_HOOK_TOOL_DENYLIST` — comma-separated names.
///   Default denylist: `"Read,Glob,Grep,WebFetch,WebSearch"`.
pub fn allow_tool(tool: &str) -> bool {
    // Allowlist check — takes precedence over denylist
    if let Ok(allowlist_env) = std::env::var("ENGRAM_HOOK_TOOL_ALLOWLIST") {
        let allowlist = allowlist_env.trim().to_string();
        if !allowlist.is_empty() {
            return allowlist.split(',').any(|t| t.trim() == tool);
        }
    }

    // Denylist check
    let denylist_env =
        std::env::var("ENGRAM_HOOK_TOOL_DENYLIST").unwrap_or_else(|_| default_denylist());
    let denied = denylist_env.split(',').any(|t| t.trim() == tool);
    !denied
}

/// Returns the minimum prompt/message length (in bytes) for content to be stored.
/// Reads `ENGRAM_HOOK_MIN_CHARS`, default `40`.
pub fn min_chars() -> usize {
    std::env::var("ENGRAM_HOOK_MIN_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40)
}

/// Returns the minimum importance score for stored hook memories.
/// Reads `ENGRAM_HOOK_MIN_IMPORTANCE`, default `0.5`.
pub fn min_importance() -> f64 {
    std::env::var("ENGRAM_HOOK_MIN_IMPORTANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5)
}

/// Returns a compiled cue regex. The built-in default is compiled exactly once
/// via `OnceLock` and reused; only an explicit `ENGRAM_HOOK_PROMPT_CUE_REGEX`
/// override triggers per-invocation compilation. Returns `None` if a custom
/// override fails to compile.
pub fn compiled_prompt_cue_regex() -> Option<Cow<'static, Regex>> {
    match std::env::var("ENGRAM_HOOK_PROMPT_CUE_REGEX") {
        Ok(s) if !s.is_empty() => Regex::new(&s).ok().map(Cow::Owned),
        _ => Some(Cow::Borrowed(default_cue_regex())),
    }
}

fn default_cue_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(default_prompt_cue_regex()).expect("default cue regex must compile")
    })
}

// ---- private helpers ----

fn event_env_name(event: HookEvent) -> &'static str {
    match event {
        HookEvent::SessionStart => "SESSIONSTART",
        HookEvent::UserPromptSubmit => "USERPROMPTSUBMIT",
        HookEvent::PostToolUse => "POSTTOOLUSE",
        HookEvent::Stop => "STOP",
        HookEvent::PreCompact => "PRECOMPACT",
        HookEvent::SessionEnd => "SESSIONEND",
        HookEvent::SubagentStop => "SUBAGENTSTOP",
    }
}

fn default_denylist() -> String {
    "Read,Glob,Grep,WebFetch,WebSearch".to_string()
}

fn default_prompt_cue_regex() -> &'static str {
    r"(?i)\b(let'?s|decided?|we'?ll|going to|switch(?:ed)? to|use|prefer|chose|choose|reason|because)\b"
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env-mutation tests: each test uses distinct env var names or serial guards.
    // To avoid a `serial_test` dependency we instead use resolver-injection helpers
    // that accept an explicit lookup closure so no real env is touched.

    fn allow_tool_with<F: Fn(&str) -> Option<String>>(tool: &str, lookup: F) -> bool {
        let allowlist = lookup("ENGRAM_HOOK_TOOL_ALLOWLIST");
        if let Some(allowlist_env) = allowlist {
            let allowlist_env = allowlist_env.trim().to_string();
            if !allowlist_env.is_empty() {
                return allowlist_env.split(',').any(|t| t.trim() == tool);
            }
        }
        let denylist_env = lookup("ENGRAM_HOOK_TOOL_DENYLIST").unwrap_or_else(default_denylist);
        let denied = denylist_env.split(',').any(|t| t.trim() == tool);
        !denied
    }

    fn allow_event_with<F: Fn(&str) -> Option<String>>(event: HookEvent, lookup: F) -> bool {
        let var_name = format!("ENGRAM_HOOK_{}_ENABLED", event_env_name(event));
        match lookup(&var_name).as_deref() {
            Some("0") | Some("false") | Some("no") => false,
            Some("1") | Some("true") | Some("yes") => true,
            _ => event != HookEvent::SubagentStop,
        }
    }

    #[allow(dead_code)]
    fn default_denylist_test() -> String {
        default_denylist()
    }

    fn min_chars_with<F: Fn(&str) -> Option<String>>(lookup: F) -> usize {
        lookup("ENGRAM_HOOK_MIN_CHARS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(40)
    }

    #[test]
    fn denylist_default_blocks_read() {
        let no_env = |_: &str| None::<String>;
        assert!(!allow_tool_with("Read", no_env));
        assert!(!allow_tool_with("Glob", no_env));
        assert!(!allow_tool_with("WebSearch", no_env));
    }

    #[test]
    fn denylist_default_allows_bash() {
        let no_env = |_: &str| None::<String>;
        assert!(allow_tool_with("Bash", no_env));
        assert!(allow_tool_with("Edit", no_env));
    }

    #[test]
    fn allowlist_overrides_denylist() {
        // When allowlist is set, only the listed tools are allowed
        let allowlist_only_bash = |key: &str| -> Option<String> {
            if key == "ENGRAM_HOOK_TOOL_ALLOWLIST" {
                Some("Bash,Edit".to_string())
            } else {
                None
            }
        };
        assert!(allow_tool_with("Bash", allowlist_only_bash));
        assert!(allow_tool_with("Edit", allowlist_only_bash));
        // Read is in default denylist but NOT in allowlist — should be denied
        assert!(!allow_tool_with("Read", allowlist_only_bash));
    }

    #[test]
    fn min_chars_respects_env() {
        assert_eq!(min_chars_with(|_| None), 40);
        assert_eq!(
            min_chars_with(|k| {
                if k == "ENGRAM_HOOK_MIN_CHARS" {
                    Some("100".to_string())
                } else {
                    None
                }
            }),
            100
        );
        // Garbage value falls back to default
        assert_eq!(
            min_chars_with(|k| {
                if k == "ENGRAM_HOOK_MIN_CHARS" {
                    Some("not_a_number".to_string())
                } else {
                    None
                }
            }),
            40
        );
    }

    #[test]
    fn event_enabled_default_on_except_subagent() {
        let no_env = |_: &str| None::<String>;
        assert!(allow_event_with(HookEvent::Stop, no_env));
        assert!(allow_event_with(HookEvent::UserPromptSubmit, no_env));
        assert!(allow_event_with(HookEvent::PostToolUse, no_env));
        assert!(allow_event_with(HookEvent::PreCompact, no_env));
        assert!(allow_event_with(HookEvent::SessionEnd, no_env));
        assert!(allow_event_with(HookEvent::SessionStart, no_env));
        assert!(!allow_event_with(HookEvent::SubagentStop, no_env));
    }

    #[test]
    fn event_enabled_can_be_overridden() {
        let stop_disabled = |key: &str| -> Option<String> {
            if key == "ENGRAM_HOOK_STOP_ENABLED" {
                Some("false".to_string())
            } else {
                None
            }
        };
        assert!(!allow_event_with(HookEvent::Stop, stop_disabled));

        let subagent_enabled = |key: &str| -> Option<String> {
            if key == "ENGRAM_HOOK_SUBAGENTSTOP_ENABLED" {
                Some("true".to_string())
            } else {
                None
            }
        };
        assert!(allow_event_with(HookEvent::SubagentStop, subagent_enabled));
    }

    #[test]
    fn custom_denylist() {
        let custom_deny = |key: &str| -> Option<String> {
            if key == "ENGRAM_HOOK_TOOL_DENYLIST" {
                Some("Bash,Edit".to_string())
            } else {
                None
            }
        };
        assert!(!allow_tool_with("Bash", custom_deny));
        assert!(!allow_tool_with("Edit", custom_deny));
        // Read is NOT in the custom denylist, should be allowed
        assert!(allow_tool_with("Read", custom_deny));
    }
}
