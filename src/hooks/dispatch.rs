use crate::db::Database;
use crate::embedding::EmbeddingService;
use crate::error::MemoryError;
use crate::hooks::HookEvent;
use crate::hooks::filter;
use crate::hooks::payload::{
    PostToolUsePayload, PreCompactPayload, SessionEndPayload, SessionStartPayload, StopPayload,
    SubagentStopPayload, UserPromptSubmitPayload,
};
use crate::hooks::redact;
use crate::hooks::transcript::extract_last_assistant_text;
use crate::memory::{Memory, MemoryType};
use crate::summarize::{generate_summary, should_auto_summarize};
use crate::tools::{StoreOutcome, dedup_threshold, store_with_dedup};

#[derive(Debug)]
#[allow(dead_code)] // Fields read by callers via Debug formatting and pattern matching
pub enum DispatchOutcome {
    Stored(uuid::Uuid),
    Skipped(&'static str),
    DryRun(String),
}

pub fn dispatch(
    event: HookEvent,
    raw_json: &str,
    dry_run: bool,
    db: &Database,
    embedding_service: Option<&EmbeddingService>,
    project_id: &str,
) -> Result<DispatchOutcome, MemoryError> {
    if raw_json.is_empty() {
        return Ok(DispatchOutcome::Skipped("empty_payload"));
    }

    if !filter::allow_event(event) {
        return Ok(DispatchOutcome::Skipped("event_disabled"));
    }

    // Daily capture cap: skip (without storing) if the project has already hit its quota today.
    // Only enforced when not in dry_run mode, and only when the cap is > 0 (0 = unlimited).
    if !dry_run {
        let cap = filter::hook_daily_cap();
        if cap > 0 && db.count_hook_memories_today(project_id)? >= cap {
            return Ok(DispatchOutcome::Skipped("daily_cap_reached"));
        }
    }

    match event {
        HookEvent::SessionStart => {
            if serde_json::from_str::<SessionStartPayload>(raw_json).is_err() {
                return Ok(DispatchOutcome::Skipped("parse_error"));
            }
            Ok(DispatchOutcome::Skipped("session_start_handled_by_script"))
        }

        HookEvent::UserPromptSubmit => {
            let payload = match serde_json::from_str::<UserPromptSubmitPayload>(raw_json) {
                Ok(p) => p,
                Err(_) => return Ok(DispatchOutcome::Skipped("parse_error")),
            };

            let prompt = match payload.prompt.as_deref() {
                Some(p) if !p.is_empty() => p,
                _ => return Ok(DispatchOutcome::Skipped("empty_prompt")),
            };

            if prompt.len() < filter::min_chars() {
                return Ok(DispatchOutcome::Skipped("prompt_too_short"));
            }

            // Explicit opt-in via #remember bypasses the cue-regex gate.
            if !prompt.contains("#remember") {
                let cue_re = match filter::compiled_prompt_cue_regex() {
                    Some(r) => r,
                    None => return Ok(DispatchOutcome::Skipped("invalid_cue_regex")),
                };
                if !cue_re.is_match(prompt) {
                    return Ok(DispatchOutcome::Skipped("prompt_no_cue_match"));
                }
            }

            let content = redact::redact(prompt);
            let branch = current_branch();

            if dry_run {
                return Ok(DispatchOutcome::DryRun(format!(
                    "UserPromptSubmit: would store Fact (branch={:?}): {}",
                    branch,
                    &content[..content.floor_char_boundary(80)]
                )));
            }

            let outcome = store_memory(
                db,
                embedding_service,
                project_id,
                &content,
                MemoryType::Fact,
                &["hook", "prompt"],
                hook_importance_cap(filter::min_importance()),
                branch.as_deref(),
                false,
                false,
            )?;
            Ok(outcome_to_dispatch(outcome))
        }

        HookEvent::PostToolUse => {
            let payload = match serde_json::from_str::<PostToolUsePayload>(raw_json) {
                Ok(p) => p,
                Err(_) => return Ok(DispatchOutcome::Skipped("parse_error")),
            };

            let tool_name = payload.tool_name.as_deref().unwrap_or("unknown");

            if !tool_response_indicates_failure(payload.tool_response.as_ref(), payload.exit_code) {
                return Ok(DispatchOutcome::Skipped("tool_succeeded"));
            }

            if !filter::allow_tool(tool_name) {
                return Ok(DispatchOutcome::Skipped("tool_denied"));
            }

            let error_text = extract_error_text(payload.tool_response.as_ref(), payload.exit_code);

            let input_str = payload
                .tool_input
                .as_ref()
                .map(|i| {
                    let s = i.to_string();
                    if s.len() > 500 {
                        format!("{}…", &s[..s.floor_char_boundary(500)])
                    } else {
                        s
                    }
                })
                .unwrap_or_default();

            let content_raw = format!("{} failed: {}\nInput: {}", tool_name, error_text, input_str);
            let content = redact::redact(&content_raw);
            let branch = current_branch();
            let tags = vec!["hook", "failure", tool_name];

            if dry_run {
                return Ok(DispatchOutcome::DryRun(format!(
                    "PostToolUse: would store Debug for tool '{}' failure",
                    tool_name
                )));
            }

            let outcome = store_memory(
                db,
                embedding_service,
                project_id,
                &content,
                MemoryType::Debug,
                &tags,
                hook_importance_cap(0.6),
                branch.as_deref(),
                false,
                false,
            )?;
            Ok(outcome_to_dispatch(outcome))
        }

        HookEvent::Stop => {
            // Stop fires on every agent turn — capturing per-turn is pure volume noise.
            match serde_json::from_str::<StopPayload>(raw_json) {
                Ok(_) => {}
                Err(_) => return Ok(DispatchOutcome::Skipped("parse_error")),
            }
            Ok(DispatchOutcome::Skipped("stop_noop"))
        }

        HookEvent::PreCompact => {
            // SessionEnd already captures transcript-based summaries; a mid-session checkpoint would just duplicate it.
            match serde_json::from_str::<PreCompactPayload>(raw_json) {
                Ok(_) => {}
                Err(_) => return Ok(DispatchOutcome::Skipped("parse_error")),
            }
            Ok(DispatchOutcome::Skipped("precompact_noop"))
        }

        HookEvent::SessionEnd => {
            let payload = match serde_json::from_str::<SessionEndPayload>(raw_json) {
                Ok(p) => p,
                Err(_) => return Ok(DispatchOutcome::Skipped("parse_error")),
            };

            match payload.reason.as_deref() {
                Some("clear") | Some("resume") => {
                    return Ok(DispatchOutcome::Skipped("reason_filtered"));
                }
                _ => {}
            }

            let transcript_path = match payload.transcript_path.as_deref() {
                Some(p) => p,
                None => return Ok(DispatchOutcome::Skipped("no_transcript_path")),
            };

            let raw_text = match extract_last_assistant_text(transcript_path, 1) {
                Some(t) => t,
                None => return Ok(DispatchOutcome::Skipped("no_assistant_message")),
            };

            let redacted = redact::redact(&raw_text);

            // Truncate to SESSION_SUMMARY_MAX bytes on a safe char boundary.
            let truncated = if redacted.len() > SESSION_SUMMARY_MAX {
                let cut = redacted.floor_char_boundary(SESSION_SUMMARY_MAX);
                format!("{}…", &redacted[..cut])
            } else {
                redacted
            };

            let branch = current_branch();

            if dry_run {
                return Ok(DispatchOutcome::DryRun(format!(
                    "SessionEnd: would store session-summary Fact from transcript (branch={:?}): {}",
                    branch,
                    &truncated[..truncated.floor_char_boundary(80)]
                )));
            }

            let outcome = store_memory(
                db,
                embedding_service,
                project_id,
                &truncated,
                MemoryType::Fact,
                &["hook", "session_summary"],
                hook_importance_cap(0.4),
                branch.as_deref(),
                false,
                false,
            )?;
            Ok(outcome_to_dispatch(outcome))
        }

        HookEvent::SubagentStop => {
            let payload = match serde_json::from_str::<SubagentStopPayload>(raw_json) {
                Ok(p) => p,
                Err(_) => return Ok(DispatchOutcome::Skipped("parse_error")),
            };

            let message = match payload.last_assistant_message.as_deref() {
                Some(m) if m.len() >= 200 => m,
                _ => return Ok(DispatchOutcome::Skipped("message_too_short")),
            };

            let agent_type = payload.agent_type.as_deref().unwrap_or("unknown");
            let content = redact::redact(message);
            let branch = current_branch();
            let tags = vec!["hook", "subagent", agent_type];

            if dry_run {
                return Ok(DispatchOutcome::DryRun(format!(
                    "SubagentStop: would store Fact for agent_type='{}'",
                    agent_type
                )));
            }

            let outcome = store_memory(
                db,
                embedding_service,
                project_id,
                &content,
                MemoryType::Fact,
                &tags,
                hook_importance_cap(0.55),
                branch.as_deref(),
                false,
                false,
            )?;
            Ok(outcome_to_dispatch(outcome))
        }
    }
}

// ---- private helpers ----

/// Maximum bytes of transcript text to store in a session-summary memory.
const SESSION_SUMMARY_MAX: usize = 4000;

/// Hard-clamp importance to a maximum of 0.5 for all hook-stored memories.
/// Manual `memory_store` calls must always be able to outrank passive hook captures.
fn hook_importance_cap(raw: f64) -> f64 {
    raw.clamp(0.0, 0.5)
}

/// Convert a `StoreOutcome` to a `DispatchOutcome`.
///
/// For `Stored` and `Merged`, the uuid is extracted from the `"mem_<uuid>"` id by
/// stripping the `"mem_"` prefix and parsing. Failures fall back to a new random UUID.
fn outcome_to_dispatch(outcome: StoreOutcome) -> DispatchOutcome {
    match outcome {
        StoreOutcome::Stored(full_id) | StoreOutcome::Merged { id: full_id, .. } => {
            let uuid = full_id
                .strip_prefix("mem_")
                .and_then(|s| s.parse::<uuid::Uuid>().ok())
                .unwrap_or_else(|| {
                    tracing::warn!(
                        "hook store returned non-standard memory id {:?}; emitting random uuid",
                        full_id
                    );
                    uuid::Uuid::new_v4()
                });
            DispatchOutcome::Stored(uuid)
        }
        StoreOutcome::SkippedSimilar { .. } => DispatchOutcome::Skipped("dedup_similar"),
    }
}

/// Build and store a `Memory`, routing through `store_with_dedup`.
///
/// When `embedding_service` is `Some`, the embedding is computed and dedup is applied:
/// - the hook-specific skip threshold suppresses storage of near-identical memories.
/// - the MCP dedup threshold controls when two memories are merged.
///
/// When `embedding_service` is `None`, the memory is stored without dedup or embedding.
#[allow(clippy::too_many_arguments)]
fn store_memory(
    db: &Database,
    embedding_service: Option<&EmbeddingService>,
    project_id: &str,
    content: &str,
    memory_type: MemoryType,
    tags: &[&str],
    importance: f64,
    branch: Option<&str>,
    pinned: bool,
    global: bool,
) -> Result<StoreOutcome, MemoryError> {
    let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
    let now = chrono::Utc::now().timestamp();

    let summary = if should_auto_summarize(content, None) {
        Some(generate_summary(content))
    } else {
        None
    };

    let memory = Memory {
        id: id.clone(),
        project_id: project_id.to_string(),
        memory_type,
        content: content.to_string(),
        summary,
        tags: tags.iter().map(|t| t.to_string()).collect(),
        importance: importance.clamp(0.0, 1.0),
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: branch.map(String::from),
        merged_from: None,
        pinned,
        global,
    };

    if let Some(es) = embedding_service {
        let embedding = es.embed_memory(memory_type, content)?;
        store_with_dedup(
            db,
            Some(es),
            project_id,
            memory,
            Some(&embedding),
            dedup_threshold(),
            Some(filter::hook_dedup_skip_threshold()),
        )
    } else {
        store_with_dedup(db, None, project_id, memory, None, 0.0, None)
    }
}

/// Detect the current git branch, identical logic to `cli.rs::get_current_branch`.
fn current_branch() -> Option<String> {
    if let Ok(branch) = std::env::var("ENGRAM_BRANCH")
        && !branch.is_empty()
    {
        return Some(branch);
    }

    let git_root = find_git_root()?;
    let git_dir = git_root.join(".git");

    if let Ok(head_content) = std::fs::read_to_string(git_dir.join("HEAD")) {
        let head = head_content.trim();
        if let Some(branch_ref) = head.strip_prefix("ref: refs/heads/") {
            return Some(branch_ref.to_string());
        }
        if head.len() >= 7 {
            return Some(format!("detached-{}", &head[..7]));
        }
    }

    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(&git_root)
        .output()
        && output.status.success()
    {
        let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if branch == "HEAD" {
            if let Ok(sha_output) = std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .current_dir(&git_root)
                .output()
                && sha_output.status.success()
            {
                let sha = String::from_utf8_lossy(&sha_output.stdout)
                    .trim()
                    .to_string();
                return Some(format!("detached-{}", sha));
            }
        } else {
            return Some(branch);
        }
    }

    None
}

fn find_git_root() -> Option<std::path::PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        if current.join(".git").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Returns `true` if the tool response or exit code signals a failure.
///
/// Checks (in order of reliability):
/// 1. `resp` is a JSON object with a non-empty `error` or `stderr` string field.
/// 2. `resp` is a JSON object with a nested numeric `exit_code`, `returncode`, or `code` field != 0.
/// 3. Top-level `exit_code` is `Some(c)` with `c != 0` (UNVERIFIED: not a guaranteed field).
/// 4. Last resort: compact JSON serialization lowercased contains "error", "failed", or "panic".
fn tool_response_indicates_failure(
    resp: Option<&serde_json::Value>,
    exit_code: Option<i32>,
) -> bool {
    if let Some(serde_json::Value::Object(map)) = resp {
        // Check for non-empty error/stderr string fields
        for key in &["error", "stderr"] {
            if let Some(serde_json::Value::String(s)) = map.get(*key)
                && !s.is_empty()
            {
                return true;
            }
        }

        // Check for nested numeric exit code fields
        for key in &["exit_code", "returncode", "code"] {
            if let Some(serde_json::Value::Number(n)) = map.get(*key)
                && n.as_i64().is_some_and(|v| v != 0)
            {
                return true;
            }
        }
    }

    // Top-level exit_code fallback (UNVERIFIED weak signal)
    if exit_code.is_some_and(|c| c != 0) {
        return true;
    }

    // Last resort: compact JSON string scan
    if let Some(v) = resp
        && let Ok(compact) = serde_json::to_string(v)
    {
        let lower = compact.to_lowercase();
        if lower.contains("error") || lower.contains("failed") || lower.contains("panic") {
            return true;
        }
    }

    false
}

/// Extract a human-readable error description from the tool response.
fn extract_error_text(resp: Option<&serde_json::Value>, exit_code: Option<i32>) -> String {
    if let Some(serde_json::Value::Object(map)) = resp {
        for key in &["error", "stderr", "message"] {
            if let Some(serde_json::Value::String(s)) = map.get(*key)
                && !s.is_empty()
            {
                return s.clone();
            }
        }
    }

    match resp {
        Some(v) => serde_json::to_string(v).unwrap_or_else(|_| "<unserializable>".to_string()),
        None => format!("exit_code={:?}", exit_code),
    }
}
