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
use crate::memory::{HandoffSections, Memory, MemoryType};
use crate::summarize::{generate_summary, should_auto_summarize};
use crate::tools::create_handoff;

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

            let cue_re = match filter::compiled_prompt_cue_regex() {
                Some(r) => r,
                None => return Ok(DispatchOutcome::Skipped("invalid_cue_regex")),
            };
            if !cue_re.is_match(prompt) {
                return Ok(DispatchOutcome::Skipped("prompt_no_cue_match"));
            }

            let content = redact::redact(prompt);
            let branch = current_branch();

            if dry_run {
                return Ok(DispatchOutcome::DryRun(format!(
                    "UserPromptSubmit: would store Decision (branch={:?}): {}",
                    branch,
                    &content[..content.len().min(80)]
                )));
            }

            let id = store_memory(
                db,
                embedding_service,
                project_id,
                &content,
                MemoryType::Decision,
                &["hook", "prompt"],
                filter::min_importance(),
                branch.as_deref(),
                false,
                false,
            )?;
            Ok(DispatchOutcome::Stored(id))
        }

        HookEvent::PostToolUse => {
            let payload = match serde_json::from_str::<PostToolUsePayload>(raw_json) {
                Ok(p) => p,
                Err(_) => return Ok(DispatchOutcome::Skipped("parse_error")),
            };

            let tool_name = payload.tool_name.as_deref().unwrap_or("unknown");

            // Check for failure signal
            let is_failure = payload.exit_code.is_some_and(|c| c != 0)
                || payload
                    .tool_response
                    .as_ref()
                    .map(|r| {
                        let s = format!("{:?}", r).to_lowercase();
                        s.contains("error") || s.contains("failed") || s.contains("panic")
                    })
                    .unwrap_or(false);

            if !is_failure {
                return Ok(DispatchOutcome::Skipped("tool_succeeded"));
            }

            if !filter::allow_tool(tool_name) {
                return Ok(DispatchOutcome::Skipped("tool_denied"));
            }

            let error_or_response = payload
                .tool_response
                .as_ref()
                .map(|r| format!("{:?}", r))
                .unwrap_or_else(|| format!("exit_code={:?}", payload.exit_code));

            let input_str = payload
                .tool_input
                .as_ref()
                .map(|i| {
                    let s = format!("{}", i);
                    if s.len() > 500 {
                        format!("{}…", &s[..500])
                    } else {
                        s
                    }
                })
                .unwrap_or_default();

            let content_raw = format!(
                "{} failed: {}\nInput: {}",
                tool_name, error_or_response, input_str
            );
            let content = redact::redact(&content_raw);
            let branch = current_branch();
            let tags = vec!["hook", "failure", tool_name];

            if dry_run {
                return Ok(DispatchOutcome::DryRun(format!(
                    "PostToolUse: would store Debug for tool '{}' failure",
                    tool_name
                )));
            }

            let id = store_memory(
                db,
                embedding_service,
                project_id,
                &content,
                MemoryType::Debug,
                &tags,
                0.6,
                branch.as_deref(),
                false,
                false,
            )?;
            Ok(DispatchOutcome::Stored(id))
        }

        HookEvent::Stop => {
            let payload = match serde_json::from_str::<StopPayload>(raw_json) {
                Ok(p) => p,
                Err(_) => return Ok(DispatchOutcome::Skipped("parse_error")),
            };

            if payload.stop_hook_active == Some(true) {
                return Ok(DispatchOutcome::Skipped("stop_reentrant"));
            }

            let message = match payload.last_assistant_message.as_deref() {
                Some(m) if m.len() >= 200 => m,
                Some(_) => return Ok(DispatchOutcome::Skipped("too_short")),
                None => return Ok(DispatchOutcome::Skipped("too_short")),
            };

            let summary = redact::redact(message);
            let branch = current_branch();

            if dry_run {
                return Ok(DispatchOutcome::DryRun(format!(
                    "Stop: would store handoff (branch={:?}): {}",
                    branch,
                    &summary[..summary.len().min(80)]
                )));
            }

            let id = store_handoff_or_fact(
                db,
                embedding_service,
                project_id,
                &summary,
                branch.as_deref(),
                0.7,
                true,
                &["hook", "stop_fallback"],
            )?;
            Ok(DispatchOutcome::Stored(id))
        }

        HookEvent::PreCompact => {
            let payload = match serde_json::from_str::<PreCompactPayload>(raw_json) {
                Ok(p) => p,
                Err(_) => return Ok(DispatchOutcome::Skipped("parse_error")),
            };

            let summary_text = if let Some(s) = payload.custom_instructions.as_deref()
                && !s.trim().is_empty()
            {
                s.to_string()
            } else {
                format!(
                    "pre-compaction checkpoint at {}",
                    chrono::Utc::now().to_rfc3339()
                )
            };
            let summary = redact::redact(&summary_text);
            let branch = current_branch();

            if dry_run {
                return Ok(DispatchOutcome::DryRun(format!(
                    "PreCompact: would store handoff (branch={:?}): {}",
                    branch,
                    &summary[..summary.len().min(80)]
                )));
            }

            let id = store_handoff_or_fact(
                db,
                embedding_service,
                project_id,
                &summary,
                branch.as_deref(),
                0.8,
                true,
                &["hook", "precompact_fallback"],
            )?;
            Ok(DispatchOutcome::Stored(id))
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

            let reason_str = payload.reason.as_deref().unwrap_or("unknown").to_string();
            let content = redact::redact(&format!(
                "Session ended ({}). Project: {}",
                reason_str, project_id
            ));
            let branch = current_branch();
            let tags = vec!["hook", "session_end", reason_str.as_str()];

            if dry_run {
                return Ok(DispatchOutcome::DryRun(format!(
                    "SessionEnd: would store Fact (reason={})",
                    reason_str
                )));
            }

            let id = store_memory(
                db,
                embedding_service,
                project_id,
                &content,
                MemoryType::Fact,
                &tags,
                0.3,
                branch.as_deref(),
                false,
                false,
            )?;
            Ok(DispatchOutcome::Stored(id))
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

            let id = store_memory(
                db,
                embedding_service,
                project_id,
                &content,
                MemoryType::Fact,
                &tags,
                0.55,
                branch.as_deref(),
                false,
                false,
            )?;
            Ok(DispatchOutcome::Stored(id))
        }
    }
}

// ---- private helpers ----

/// Store a handoff if branch is `Some`, falling back to a plain Fact on `InvalidType` or `None` branch.
#[allow(clippy::too_many_arguments)]
fn store_handoff_or_fact(
    db: &Database,
    embedding_service: Option<&EmbeddingService>,
    project_id: &str,
    summary: &str,
    branch: Option<&str>,
    importance: f64,
    pinned: bool,
    fallback_tags: &[&str],
) -> Result<uuid::Uuid, MemoryError> {
    if let Some(b) = branch
        && let Some(es) = embedding_service
    {
        let sections = HandoffSections {
            summary: summary.to_string(),
            decisions: vec![],
            todos: vec![],
            blockers: vec![],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };
        match create_handoff(
            db,
            es,
            project_id,
            Some(b),
            sections,
            importance,
            pinned,
            true,
        ) {
            Ok(result) => {
                let id: uuid::Uuid = result
                    .id
                    .trim_start_matches("mem_")
                    .parse()
                    .unwrap_or_else(|_| uuid::Uuid::new_v4());
                return Ok(id);
            }
            Err(MemoryError::InvalidType(_)) => {
                // Fall through to plain Fact store
            }
            Err(e) => return Err(e),
        }
    }

    // Plain Fact fallback (no branch, or InvalidType from create_handoff)
    store_memory(
        db,
        embedding_service,
        project_id,
        summary,
        MemoryType::Fact,
        fallback_tags,
        importance,
        branch,
        pinned,
        false,
    )
}

/// Build and store a `Memory`, then store its embedding if `embedding_service` is available.
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
) -> Result<uuid::Uuid, MemoryError> {
    let mem_uuid = uuid::Uuid::new_v4();
    let id = format!("mem_{}", mem_uuid.simple());
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

    db.store_memory(&memory)?;

    if let Some(es) = embedding_service {
        let embedding = es.embed_memory(memory_type, content)?;
        db.store_embedding(&id, &embedding, es.model_version())?;
    }

    Ok(mem_uuid)
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
