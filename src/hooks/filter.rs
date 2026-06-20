//! Noise filter for hook events. All configuration is read from environment variables.

use crate::hooks::HookEvent;
use regex::Regex;
use std::borrow::Cow;
use std::sync::OnceLock;

/// Returns `true` if the event is enabled for hook processing.
///
/// Reads `ENGRAM_HOOK_<UPPERCASE_VARIANT>_ENABLED` (e.g. `ENGRAM_HOOK_STOP_ENABLED`).
/// All events default to enabled **except** `SubagentStop` and `UserPromptSubmit`,
/// which both default to `false` to avoid high-volume passive captures.
pub fn allow_event(event: HookEvent) -> bool {
    let var_name = format!("ENGRAM_HOOK_{}_ENABLED", event_env_name(event));
    match std::env::var(&var_name).as_deref() {
        Ok("0") | Ok("false") | Ok("no") => false,
        Ok("1") | Ok("true") | Ok("yes") => true,
        _ => !matches!(event, HookEvent::SubagentStop | HookEvent::UserPromptSubmit),
    }
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
/// Note: the returned value is subsequently capped by `hook_importance_cap` (max 0.5) in the
/// dispatch layer, so setting `ENGRAM_HOOK_MIN_IMPORTANCE` above 0.5 has no effect — only
/// values below 0.5 actually change the floor used for hook captures.
pub fn min_importance() -> f64 {
    std::env::var("ENGRAM_HOOK_MIN_IMPORTANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5)
}

/// Returns the similarity threshold above which hook captures are skipped (not merged).
///
/// Reads `ENGRAM_HOOK_DEDUP_SKIP`, default `0.95`, clamped to `[0.5, 1.0]`.
/// When the best matching existing memory has similarity >= this value, the hook memory
/// is silently dropped rather than stored+merged — avoiding near-identical duplicates
/// from repeated hook triggers.
pub fn hook_dedup_skip_threshold() -> f32 {
    std::env::var("ENGRAM_HOOK_DEDUP_SKIP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.5, 1.0))
        .unwrap_or(0.95)
}

/// Returns the maximum number of hook-captured memories allowed per project per day (UTC).
///
/// Reads `ENGRAM_HOOK_DAILY_CAP`, default `50`.
/// A value of `0` means unlimited — no cap is enforced.
/// Non-numeric or negative values fall back to the default of `50`.
pub fn hook_daily_cap() -> usize {
    std::env::var("ENGRAM_HOOK_DAILY_CAP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50)
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

fn default_prompt_cue_regex() -> &'static str {
    r"(?i)\b(decided?|chose|prefer(?:s|red|ence)?|switch(?:ed)? to)\b"
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env-mutation tests: each test uses distinct env var names or serial guards.
    // To avoid a `serial_test` dependency we instead use resolver-injection helpers
    // that accept an explicit lookup closure so no real env is touched.

    fn allow_event_with<F: Fn(&str) -> Option<String>>(event: HookEvent, lookup: F) -> bool {
        let var_name = format!("ENGRAM_HOOK_{}_ENABLED", event_env_name(event));
        match lookup(&var_name).as_deref() {
            Some("0") | Some("false") | Some("no") => false,
            Some("1") | Some("true") | Some("yes") => true,
            _ => !matches!(event, HookEvent::SubagentStop | HookEvent::UserPromptSubmit),
        }
    }

    fn min_chars_with<F: Fn(&str) -> Option<String>>(lookup: F) -> usize {
        lookup("ENGRAM_HOOK_MIN_CHARS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(40)
    }

    fn hook_daily_cap_with<F: Fn(&str) -> Option<String>>(lookup: F) -> usize {
        lookup("ENGRAM_HOOK_DAILY_CAP")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(50)
    }

    fn hook_dedup_skip_threshold_with<F: Fn(&str) -> Option<String>>(lookup: F) -> f32 {
        lookup("ENGRAM_HOOK_DEDUP_SKIP")
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.5, 1.0))
            .unwrap_or(0.95)
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
    fn event_enabled_default_on_except_subagent_and_userpromptsubmit() {
        let no_env = |_: &str| None::<String>;
        assert!(allow_event_with(HookEvent::Stop, no_env));
        assert!(allow_event_with(HookEvent::PostToolUse, no_env));
        assert!(allow_event_with(HookEvent::PreCompact, no_env));
        assert!(allow_event_with(HookEvent::SessionEnd, no_env));
        assert!(allow_event_with(HookEvent::SessionStart, no_env));
        // Both SubagentStop and UserPromptSubmit default to off
        assert!(!allow_event_with(HookEvent::SubagentStop, no_env));
        assert!(!allow_event_with(HookEvent::UserPromptSubmit, no_env));
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
    fn hook_daily_cap_default_and_override() {
        let no_env = |_: &str| None::<String>;
        assert_eq!(hook_daily_cap_with(no_env), 50, "default should be 50");

        let capped = |k: &str| -> Option<String> {
            if k == "ENGRAM_HOOK_DAILY_CAP" {
                Some("100".to_string())
            } else {
                None
            }
        };
        assert_eq!(hook_daily_cap_with(capped), 100);

        // 0 means unlimited
        let unlimited = |k: &str| -> Option<String> {
            if k == "ENGRAM_HOOK_DAILY_CAP" {
                Some("0".to_string())
            } else {
                None
            }
        };
        assert_eq!(hook_daily_cap_with(unlimited), 0, "0 means unlimited");

        // Garbage falls back to default
        let garbage = |k: &str| -> Option<String> {
            if k == "ENGRAM_HOOK_DAILY_CAP" {
                Some("not_a_number".to_string())
            } else {
                None
            }
        };
        assert_eq!(
            hook_daily_cap_with(garbage),
            50,
            "garbage should fall back to 50"
        );
    }

    #[test]
    fn hook_dedup_skip_threshold_default_and_clamp() {
        let no_env = |_: &str| None::<String>;
        assert!(
            (hook_dedup_skip_threshold_with(no_env) - 0.95).abs() < 1e-6,
            "default should be 0.95"
        );

        let custom = |k: &str| -> Option<String> {
            if k == "ENGRAM_HOOK_DEDUP_SKIP" {
                Some("0.80".to_string())
            } else {
                None
            }
        };
        assert!((hook_dedup_skip_threshold_with(custom) - 0.80).abs() < 1e-6);

        // Below clamp floor
        let too_low = |k: &str| -> Option<String> {
            if k == "ENGRAM_HOOK_DEDUP_SKIP" {
                Some("0.1".to_string())
            } else {
                None
            }
        };
        assert!(
            (hook_dedup_skip_threshold_with(too_low) - 0.5).abs() < 1e-6,
            "values below 0.5 should be clamped to 0.5"
        );

        // Above clamp ceiling
        let too_high = |k: &str| -> Option<String> {
            if k == "ENGRAM_HOOK_DEDUP_SKIP" {
                Some("1.5".to_string())
            } else {
                None
            }
        };
        assert!(
            (hook_dedup_skip_threshold_with(too_high) - 1.0).abs() < 1e-6,
            "values above 1.0 should be clamped to 1.0"
        );
    }
}
