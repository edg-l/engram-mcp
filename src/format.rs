//! Compact text formatting for MCP tool results.
//!
//! Converts JSON tool results into compact text for token-efficient LLM consumption.

use std::path::Path;

use crate::db::Database;
use crate::memory::{AdrSections, AdrStatus, HandoffSections, Memory, TodoItem, TodoStatus};
use serde_json::Value;

/// Return true if `path` looks like a local filesystem path that should be
/// existence-checked.  Heuristics:
/// - Starts with `/` (Unix absolute)
/// - Starts with `./` or `../` (relative)
/// - Starts with a Windows drive letter followed by `:\` or `:/`
fn is_local_looking(path: &str) -> bool {
    if path.starts_with('/') || path.starts_with("./") || path.starts_with("../") {
        return true;
    }
    // Windows drive letter: C:\ or C:/
    let bytes = path.as_bytes();
    if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
    {
        return true;
    }
    false
}

/// Render the external_artifacts list for a `Memory` struct.
///
/// Each artifact is printed on its own line under `**Artifacts:**`.
/// Local-looking paths are checked for existence; missing ones are suffixed with ` [missing]`.
/// Non-local paths (URLs, ticket IDs, etc.) are printed as-is.
///
/// Returns an empty string if `artifacts` is `None` or empty.
pub fn render_artifacts(artifacts: &Option<Vec<String>>) -> String {
    let Some(list) = artifacts else {
        return String::new();
    };
    if list.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n**Artifacts:**\n");
    for path in list {
        if is_local_looking(path) {
            if Path::new(path).exists() {
                out.push_str(&format!("- {}\n", path));
            } else {
                out.push_str(&format!("- {}  [missing]\n", path));
            }
        } else {
            out.push_str(&format!("- {}\n", path));
        }
    }
    out
}

/// Render a `Handoff` memory as a human-readable section-aware view.
///
/// Produces section headers, `- [ ]` todo checkboxes, and bullet points for blockers.
/// Called from `format_memory_content` when the memory type is `handoff`.
pub fn format_handoff(memory: &Memory, sections: &HandoffSections) -> String {
    let mut out = String::new();

    // Header line with ID and importance
    out.push_str(&format!(
        "[handoff] {} (importance: {:.2})\n",
        memory.id, memory.importance
    ));

    // Summary
    out.push_str("\n## Summary\n");
    out.push_str(&sections.summary);
    out.push('\n');

    // Decisions
    if !sections.decisions.is_empty() {
        out.push_str("\n## Decisions\n");
        for d in &sections.decisions {
            out.push_str(&format!("- {}\n", d));
        }
    }

    // Todos with checkboxes
    if !sections.todos.is_empty() {
        out.push_str("\n## Todos\n");
        for t in &sections.todos {
            out.push_str(&format!("- [ ] {}\n", t));
        }
    }

    // Blockers with bullet points
    if !sections.blockers.is_empty() {
        out.push_str("\n## Blockers\n");
        for b in &sections.blockers {
            out.push_str(&format!("- {}\n", b));
        }
    }

    // Dead ends, so they are not retried
    if !sections.tried.is_empty() {
        out.push_str("\n## Tried\n");
        for t in &sections.tried {
            out.push_str(&format!("- {}\n", t));
        }
    }

    // Mental model
    if !sections.mental_model.is_empty() {
        out.push_str("\n## Mental Model\n");
        out.push_str(&sections.mental_model);
        out.push('\n');
    }

    // Next steps
    if !sections.next_steps.is_empty() {
        out.push_str("\n## Next Steps\n");
        for s in &sections.next_steps {
            out.push_str(&format!("- {}\n", s));
        }
    }

    // Notes
    if let Some(notes) = &sections.notes
        && !notes.is_empty()
    {
        out.push_str("\n## Notes\n");
        out.push_str(notes);
        out.push('\n');
    }

    out.trim_end().to_string()
}

/// Render an ADR memory as a human-readable section-aware view.
///
/// Emits a header line `[adr-{:04} | {status}]` followed by the four Nygard sections.
pub fn format_adr(number: u32, status: AdrStatus, sections: &AdrSections) -> String {
    let mut out = String::new();

    out.push_str(&format!("[adr-{:04} | {}]\n", number, status));

    out.push_str("\n## Context\n");
    out.push_str(&sections.context);
    out.push('\n');

    out.push_str("\n## Decision\n");
    out.push_str(&sections.decision);
    out.push('\n');

    out.push_str("\n## Consequences\n");
    out.push_str(&sections.consequences);
    out.push('\n');

    out.trim_end().to_string()
}

/// Parse the ADR number from the `# NNNN. Title` heading in rendered ADR markdown.
fn parse_adr_number_from_content(content: &str) -> Option<u32> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("# ") {
            let raw = rest.trim();
            if let Some(dot_pos) = raw.find(". ") {
                let prefix = &raw[..dot_pos];
                if let Ok(n) = prefix.parse::<u32>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Parse the ADR status from the `## Status\n\n{status} — {date}` section in rendered markdown.
fn parse_adr_status_from_content(content: &str) -> Option<AdrStatus> {
    let mut in_status = false;
    for line in content.lines() {
        if line.trim() == "## Status" {
            in_status = true;
            continue;
        }
        if in_status {
            // A new section heading ends the Status block.
            if line.starts_with("## ") {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Status line: "{status} — {date}" or just "{status}"
            let status_word = trimmed
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches('\u{2014}')
                .trim();
            return status_word.parse::<AdrStatus>().ok();
        }
    }
    None
}

/// Format memory content for display, with section-aware rendering for handoffs and ADRs.
///
/// When `mem_type` is `"handoff"`, attempts to parse structured sections from
/// `content` via `HandoffSections::parse_markdown` and delegates to `format_handoff`.
/// When `mem_type` is `"adr"`, parses number/status from markdown headings and
/// delegates to `format_adr`. Falls back to plain `content` on parse failure or for
/// other memory types.
pub fn format_memory_content(memory: &Memory, max_len: usize) -> String {
    use crate::memory::MemoryType;
    match memory.memory_type {
        MemoryType::Handoff => match HandoffSections::parse_markdown(&memory.content) {
            Ok(sections) => {
                let rendered = format_handoff(memory, &sections);
                truncate_str(&rendered, max_len)
            }
            Err(_) => truncate_str(&memory.content, max_len),
        },
        MemoryType::Adr => match AdrSections::parse_markdown(&memory.content) {
            Ok(sections) => {
                let number = parse_adr_number_from_content(&memory.content).unwrap_or(0);
                let status =
                    parse_adr_status_from_content(&memory.content).unwrap_or(AdrStatus::Proposed);
                let rendered = format_adr(number, status, &sections);
                truncate_str(&rendered, max_len)
            }
            Err(_) => truncate_str(&memory.content, max_len),
        },
        _ => truncate_str(&memory.content, max_len),
    }
}

/// Render one todo as a checkbox line: `- [ ] text` open, `- [x]` done, `- [~]` dropped
/// with its reason, since a dropped todo without the reason is the thing the reason exists
/// to prevent.
pub fn format_todo(todo: &TodoItem) -> String {
    let (box_mark, suffix) = match todo.status {
        TodoStatus::Open => (" ", String::new()),
        TodoStatus::Done => ("x", String::new()),
        TodoStatus::Dropped => (
            "~",
            match todo.reason.as_deref() {
                Some(r) => format!(" (dropped: {r})"),
                None => " (dropped)".to_string(),
            },
        ),
    };
    let scope = match todo.branch.as_deref() {
        Some(b) => format!(" [{b}]"),
        None => String::new(),
    };
    format!(
        "- [{}] {}{}{}  {}",
        box_mark, todo.text, scope, suffix, todo.id
    )
}

/// Render a Handoff or ADR memory using the DB sidecar row.
///
/// For Handoff: calls `db.get_handoff_sections`. For ADR: calls `db.get_adr_sections`.
/// Falls back to `format_memory_content` on DB miss or error.
fn format_memory_content_with_db(memory: &Memory, db: &Database, max_len: usize) -> String {
    use crate::memory::MemoryType;
    match memory.memory_type {
        MemoryType::Handoff => match db.get_handoff_sections(&memory.id) {
            Ok(Some((sections, _))) => {
                let rendered = format_handoff(memory, &sections);
                truncate_str(&rendered, max_len)
            }
            _ => format_memory_content(memory, max_len),
        },
        MemoryType::Adr => match db.get_adr_sections(&memory.id) {
            Ok(Some((number, status, sections))) => {
                let rendered = format_adr(number, status, &sections);
                truncate_str(&rendered, max_len)
            }
            _ => format_memory_content(memory, max_len),
        },
        MemoryType::Todo => match db.get_todo(&memory.id) {
            Ok(Some(todo)) => truncate_str(&format_todo(&todo), max_len),
            _ => format_memory_content(memory, max_len),
        },
        _ => format_memory_content(memory, max_len),
    }
}

/// Format handoff memory content from raw JSON fields, without a full `Memory` struct.
///
/// Used by JSON-based formatters when a `Database` is available to load the sidecar.
/// Builds a minimal `Memory` and delegates to `format_memory_content_with_db`.
fn format_memory_content_from_json_with_db(
    id: &str,
    content: &str,
    importance: f64,
    max_len: usize,
    db: &Database,
) -> String {
    let memory = Memory {
        id: id.to_string(),
        project_id: String::new(),
        memory_type: crate::memory::MemoryType::Handoff,
        content: content.to_string(),
        summary: None,
        tags: vec![],
        importance,
        relevance_score: 1.0,
        access_count: 0,
        created_at: 0,
        updated_at: 0,
        last_accessed_at: 0,
        branch: None,
        merged_from: None,
        external_artifacts: None,
        pinned: false,
        global: false,
    };
    format_memory_content_with_db(&memory, db, max_len)
}

/// Truncate a string to a maximum length, adding "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max_len).collect::<String>())
    }
}

/// Format a tool result as compact text for LLM consumption.
/// Optimized for readability and token efficiency.
///
/// For live tool-handler paths that have access to a `Database`, prefer
/// `compact_tool_result_with_db` so Handoff memories are rendered via the sidecar.
#[allow(dead_code)] // Used by lib unit tests; not reached by the engram-cli binary.
pub fn compact_tool_result(tool_name: &str, result: &Value, content_length: usize) -> String {
    match tool_name {
        "memory_store" => compact_store(result),
        "memory_query" => compact_query(result, content_length, None),
        "memory_context" => compact_context(result, content_length, None),
        "memory_graph" => compact_graph(result),
        "memory_store_batch" => compact_batch_store(result),
        "memory_prune" => compact_prune(result),
        "memory_promote" => compact_promote(result),
        "memory_dedup" => compact_dedup(result),
        "memory_stats" => compact_stats(result),
        "memory_projects" => compact_projects(result),
        "memory_update" => compact_update(result, content_length),
        "memory_delete" => compact_delete(result, content_length),
        "memory_delete_batch" => compact_delete_batch(result, content_length),
        "memory_list" => compact_list(result),
        "memory_trash" => compact_trash(result),
        "handoff_resume" => compact_handoff_resume(result),
        "todo_write" => compact_todo_write(result),
        "todo_list" => compact_todo_list(result),
        "memory_restore" => compact_restore(result),
        _ => compact_fallback(result),
    }
}

/// Like `compact_tool_result` but uses `db` to load handoff sidecar sections for
/// `memory_query` and `memory_context` results, giving section-aware rendering.
#[allow(dead_code)] // Used by the engram MCP server binary; not reached by engram-cli.
pub fn compact_tool_result_with_db(
    tool_name: &str,
    result: &Value,
    content_length: usize,
    db: &Database,
) -> String {
    match tool_name {
        "memory_store" => compact_store(result),
        "memory_query" => compact_query(result, content_length, Some(db)),
        "memory_context" => compact_context(result, content_length, Some(db)),
        "memory_graph" => compact_graph(result),
        "memory_store_batch" => compact_batch_store(result),
        "memory_prune" => compact_prune(result),
        "memory_promote" => compact_promote(result),
        "memory_dedup" => compact_dedup(result),
        "memory_stats" => compact_stats(result),
        "memory_projects" => compact_projects(result),
        "memory_update" => compact_update(result, content_length),
        "memory_delete" => compact_delete(result, content_length),
        "memory_delete_batch" => compact_delete_batch(result, content_length),
        "memory_list" => compact_list(result),
        "memory_trash" => compact_trash(result),
        "handoff_resume" => compact_handoff_resume(result),
        "todo_write" => compact_todo_write(result),
        "todo_list" => compact_todo_list(result),
        "memory_restore" => compact_restore(result),
        _ => compact_fallback(result),
    }
}

fn compact_store(result: &Value) -> String {
    let id = result.get("id").and_then(|v| v.as_str()).unwrap_or("?");
    let mut out = format!("Stored {}", id);

    if let Some(project) = result.get("project").and_then(|v| v.as_str()) {
        out.push_str(&format!(" in {}", project));
    }

    if let Some(branch) = result.get("branch").and_then(|v| v.as_str()) {
        out.push_str(&format!(" (branch: {})", branch));
    }

    if let Some(merge) = result.get("merge_info")
        && !merge.is_null()
    {
        let merged_with = merge
            .get("merged_with")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let sim = merge
            .get("similarity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        out.push_str(&format!(
            "\nMerged with duplicate {} (similarity: {:.2})",
            merged_with, sim
        ));
    }

    if let Some(superseded) = result.get("superseded").and_then(|v| v.as_array())
        && !superseded.is_empty()
    {
        let ids: Vec<&str> = superseded.iter().filter_map(|v| v.as_str()).collect();
        out.push_str(&format!(
            "\nSupersedes {}. Those memories are no longer returned by search; queries that \
             matched them now return this one.",
            ids.join(", ")
        ));
    }

    // The point of reporting candidates is that the caller is asked, so they have to be
    // visible in the tool result an agent actually reads.
    if let Some(candidates) = result.get("possible_supersedes").and_then(|v| v.as_array())
        && !candidates.is_empty()
    {
        out.push_str("\n\nExisting memories on what looks like the same subject:");
        for candidate in candidates {
            let id = candidate.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let similarity = candidate
                .get("similarity")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let preview = candidate
                .get("preview")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            out.push_str(&format!("\n  {} ({:.2}) {}", id, similarity, preview));
        }
        out.push_str(
            "\nIf this replaces one of them, store again with `supersedes` rather than keeping \
             both. If it merely elaborates, ignore this.",
        );
    }

    out
}

fn compact_query(result: &Value, content_length: usize, db: Option<&Database>) -> String {
    let memories = result.get("memories").and_then(|v| v.as_array());
    let Some(arr) = memories else {
        return "No results.".to_string();
    };
    if arr.is_empty() {
        return "No results.".to_string();
    }

    let mut out = format!("{} result(s):\n", arr.len());

    for mem in arr {
        let memory = mem.get("memory").unwrap_or(mem);
        let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let mem_type = memory
            .get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let content = memory.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let score = mem.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let tags = memory.get("tags").and_then(|v| v.as_array());
        let importance = memory
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        out.push_str(&format!("\n[{}] {} ({}", id, mem_type, format_score(score)));
        if importance >= 0.7 {
            out.push_str(&format!(", importance: {:.1}", importance));
        }
        out.push(')');
        if let Some(tags) = tags
            && !tags.is_empty()
        {
            let tag_strs: Vec<&str> = tags.iter().filter_map(|t| t.as_str()).collect();
            if !tag_strs.is_empty() {
                out.push_str(&format!(" [{}]", tag_strs.join(", ")));
            }
        }
        out.push('\n');
        // A redirect: the query matched a memory this one superseded, and this is what
        // replaced it. Saying so is the difference between a current answer and an
        // unexplained one.
        if let Some(via) = mem.get("matched_via")
            && !via.is_null()
        {
            let superseded_id = via
                .get("superseded_id")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let preview = via
                .get("superseded_preview")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            out.push_str(&format!("replaces {}: {}\n", superseded_id, preview));
        }
        // Section-aware rendering for handoffs via DB sidecar; plain content for other types.
        let formatted_content = if mem_type == "handoff" {
            if let Some(db) = db {
                format_memory_content_from_json_with_db(id, content, importance, content_length, db)
            } else {
                truncate_str(content, content_length)
            }
        } else {
            truncate_str(content, content_length)
        };
        out.push_str(&formatted_content);
        out.push('\n');
        // Render external artifacts with existence check for local-looking paths.
        let artifacts: Option<Vec<String>> = memory
            .get("external_artifacts")
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        out.push_str(&render_artifacts(&artifacts));
    }

    out
}

fn compact_context(result: &Value, content_length: usize, db: Option<&Database>) -> String {
    let memories = result.get("memories").and_then(|v| v.as_array());
    let Some(arr) = memories else {
        return "No relevant memories.".to_string();
    };
    if arr.is_empty() {
        return "No relevant memories.".to_string();
    }

    let mode = result
        .get("retrieval_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("flat");
    let mut out = format!(
        "{} relevant memory/memories ({} retrieval):\n",
        arr.len(),
        mode
    );

    for mem in arr {
        let id = mem.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let mem_type = mem.get("type").and_then(|v| v.as_str()).unwrap_or("?");
        let content = mem.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let sim = mem
            .get("similarity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let importance = mem
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let tags = mem.get("tags").and_then(|v| v.as_array());

        out.push_str(&format!("\n[{}] {} ({})", id, mem_type, format_score(sim)));
        if let Some(tags) = tags
            && !tags.is_empty()
        {
            let tag_strs: Vec<&str> = tags.iter().filter_map(|t| t.as_str()).collect();
            if !tag_strs.is_empty() {
                out.push_str(&format!(" [{}]", tag_strs.join(", ")));
            }
        }
        out.push('\n');
        // A redirect: the context matched a memory this one superseded.
        if let Some(via) = mem.get("matched_via")
            && !via.is_null()
        {
            let superseded_id = via
                .get("superseded_id")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            out.push_str(&format!("replaces {}\n", superseded_id));
        }
        // Section-aware rendering for handoffs via DB sidecar; plain content for other types.
        let formatted_content = if mem_type == "handoff" {
            if let Some(db) = db {
                format_memory_content_from_json_with_db(id, content, importance, content_length, db)
            } else {
                truncate_str(content, content_length)
            }
        } else {
            truncate_str(content, content_length)
        };
        out.push_str(&formatted_content);
        out.push('\n');
        // Render external artifacts with existence check for local-looking paths.
        let artifacts: Option<Vec<String>> = mem
            .get("external_artifacts")
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        out.push_str(&render_artifacts(&artifacts));
    }

    out
}

fn compact_graph(result: &Value) -> String {
    let root = result.get("root");
    let root_id = root
        .and_then(|r| r.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let root_type = root
        .and_then(|r| r.get("memory_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let root_content = root
        .and_then(|r| r.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut out = format!(
        "Root: [{}] {} - {}\n",
        root_id,
        root_type,
        truncate_str(root_content, 100)
    );

    if let Some(related) = result.get("related").and_then(|v| v.as_array()) {
        if related.is_empty() {
            out.push_str("No related memories.");
        } else {
            out.push_str(&format!("\n{} related:", related.len()));
            for rel in related {
                let memory = rel.get("memory").unwrap_or(rel);
                let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let mem_type = memory
                    .get("memory_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let content = memory.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let relation = rel.get("relation").and_then(|v| v.as_str()).unwrap_or("?");
                let direction = rel.get("direction").and_then(|v| v.as_str()).unwrap_or("?");
                let depth = rel.get("depth").and_then(|v| v.as_u64()).unwrap_or(0);

                let indent = "  ".repeat(depth as usize);
                let arrow = if direction == "outgoing" { "->" } else { "<-" };
                out.push_str(&format!(
                    "\n{}{} {} [{}] {} - {}",
                    indent,
                    arrow,
                    relation,
                    id,
                    mem_type,
                    truncate_str(content, 80)
                ));
            }
        }
    }

    out
}

fn compact_batch_store(result: &Value) -> String {
    let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
    let ids = result.get("ids").and_then(|v| v.as_array());
    let mut out = format!("Stored {} memories", count);
    if let Some(project) = result.get("project").and_then(|v| v.as_str()) {
        out.push_str(&format!(" in {}", project));
    }
    if let Some(ids) = ids {
        let id_strs: Vec<&str> = ids.iter().filter_map(|v| v.as_str()).collect();
        if !id_strs.is_empty() {
            out.push_str(&format!(": {}", id_strs.join(", ")));
        }
    }
    out
}

fn compact_prune(result: &Value) -> String {
    let dry_run = result
        .get("dry_run")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let candidates = result
        .get("candidates")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let deleted = result.get("deleted").and_then(|v| v.as_u64()).unwrap_or(0);
    let threshold = result
        .get("threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.2);

    if dry_run {
        format!(
            "Prune dry run: {} memories below {:.2} threshold. Set confirm=true to delete.",
            candidates, threshold
        )
    } else {
        format!(
            "Pruned {} memories below {:.2} threshold.",
            deleted, threshold
        )
    }
}

fn compact_promote(result: &Value) -> String {
    let success = result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let message = result.get("message").and_then(|v| v.as_str()).unwrap_or("");
    if success {
        message.to_string()
    } else {
        format!("Failed: {}", message)
    }
}

fn compact_dedup(result: &Value) -> String {
    let dry_run = result
        .get("dry_run")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let groups = result
        .get("duplicate_groups")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total = result
        .get("total_duplicates")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let merged = result.get("merged").and_then(|v| v.as_u64()).unwrap_or(0);

    if dry_run {
        format!(
            "Dedup dry run: {} duplicate groups ({} total). Set confirm=true to merge.",
            groups, total
        )
    } else {
        format!("Dedup complete: merged {} duplicate memories.", merged)
    }
}

/// Render the memory a destructive tool destroyed.
///
/// A memory often carries more than the claim it appears to be about, and only some of
/// it is recoverable from the repository. Printing it is what lets the caller notice
/// what they were about to lose while the tool result is still in front of them.
fn render_destroyed(memory: &Value, content_length: usize, out: &mut String) {
    let content = memory.get("content").and_then(|v| v.as_str()).unwrap_or("");
    out.push_str(&format!(
        "\n\n{}",
        truncate_str(content, content_length.max(500))
    ));

    if let Some(sources) = memory.get("merged_from").and_then(|v| v.as_array())
        && !sources.is_empty()
    {
        out.push_str(&format!(
            "\n\nThis memory had absorbed {} other memor{} by dedup:",
            sources.len(),
            if sources.len() == 1 { "y" } else { "ies" }
        ));
        for source in sources {
            let id = source.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let text = source
                .get("content")
                .and_then(|v| v.as_str())
                .or_else(|| source.get("content_preview").and_then(|v| v.as_str()))
                .unwrap_or("");
            out.push_str(&format!(
                "\n\n  [{}] {}",
                id,
                truncate_str(text, content_length.max(500))
            ));
        }
    }
}

fn compact_update(result: &Value, content_length: usize) -> String {
    if !result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return format!(
            "Failed: {}",
            result.get("message").and_then(|v| v.as_str()).unwrap_or("")
        );
    }

    let mut out = "Memory updated".to_string();

    match result.get("dead").and_then(|v| v.as_bool()) {
        Some(true) => out.push_str(" and marked dead: excluded from all retrieval."),
        _ => out.push('.'),
    }

    // Content is replaced wholesale, not patched, so the previous version is gone unless
    // the caller is handed it here.
    if result
        .get("content_replaced")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        && let Some(previous) = result.get("previous")
    {
        out.push_str("\n\nReplaced content:");
        render_destroyed(previous, content_length, &mut out);
        let id = result
            .get("previous")
            .and_then(|p| p.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        out.push_str(&format!(
            "\n\nRecoverable with memory_restore id={} until the trash is swept.",
            id
        ));
    }

    out
}

fn compact_delete(result: &Value, content_length: usize) -> String {
    if !result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return format!(
            "Failed: {}",
            result.get("message").and_then(|v| v.as_str()).unwrap_or("")
        );
    }

    let mut out = "Memory deleted.".to_string();
    if let Some(deleted) = result.get("deleted")
        && !deleted.is_null()
    {
        out.push_str("\n\nDeleted content:");
        render_destroyed(deleted, content_length, &mut out);
        let id = deleted.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        out.push_str(&format!(
            "\n\nRecoverable with memory_restore id={} until the trash is swept.",
            id
        ));
    }
    out
}

fn compact_delete_batch(result: &Value, content_length: usize) -> String {
    let deleted = result.get("deleted").and_then(|v| v.as_u64()).unwrap_or(0);
    let mut out = format!(
        "{} memor{} deleted.",
        deleted,
        if deleted == 1 { "y" } else { "ies" }
    );

    if let Some(memories) = result.get("memories").and_then(|v| v.as_array()) {
        for memory in memories {
            let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            out.push_str(&format!("\n\n[{}]", id));
            render_destroyed(memory, content_length, &mut out);
        }
        if !memories.is_empty() {
            out.push_str("\n\nAll recoverable with memory_restore until the trash is swept.");
        }
    }
    out
}

fn compact_list(result: &Value) -> String {
    let Some(rows) = result.get("memories").and_then(|v| v.as_array()) else {
        return "No memories.".to_string();
    };
    if rows.is_empty() {
        return "No memories.".to_string();
    }

    let status = result
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("live");
    let total = result.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
    let mut out = format!("{} of {} ({}):\n", rows.len(), total, status);

    for row in rows {
        let id = row.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let mem_type = row.get("type").and_then(|v| v.as_str()).unwrap_or("?");
        let content = row.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let relevance = row
            .get("relevance_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let mut marks = String::new();
        if row.get("pinned").and_then(|v| v.as_bool()).unwrap_or(false) {
            marks.push_str(" [pinned]");
        }
        if row.get("dead").and_then(|v| v.as_bool()).unwrap_or(false) {
            marks.push_str(" [dead]");
        }
        if let Some(successor) = row.get("superseded_by").and_then(|v| v.as_str()) {
            marks.push_str(&format!(" [superseded by {}]", successor));
        }

        out.push_str(&format!(
            "\n[{}] {} ({:.2}){}\n{}\n",
            id, mem_type, relevance, marks, content
        ));
    }
    out
}

fn compact_trash(result: &Value) -> String {
    let Some(entries) = result.get("entries").and_then(|v| v.as_array()) else {
        return "Trash is empty.".to_string();
    };
    if entries.is_empty() {
        return "Trash is empty.".to_string();
    }

    let total = result.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
    let mut out = format!("{} of {} recoverable:\n", entries.len(), total);

    for entry in entries {
        let trash_id = entry.get("trash_id").and_then(|v| v.as_i64()).unwrap_or(0);
        let memory_id = entry
            .get("memory_id")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let op = entry.get("op").and_then(|v| v.as_str()).unwrap_or("?");
        let mem_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("?");
        let preview = entry.get("preview").and_then(|v| v.as_str()).unwrap_or("");
        let chars = entry
            .get("content_chars")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        out.push_str(&format!(
            "\ntrash_id {} — {} by {} ({}, {} chars)\n{}\n",
            trash_id, memory_id, op, mem_type, chars, preview
        ));
    }
    out.push_str("\nRestore with memory_restore id=<memory-id> or trash_id=<n>.");
    out
}

fn compact_restore(result: &Value) -> String {
    if !result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return format!(
            "Failed: {}",
            result.get("message").and_then(|v| v.as_str()).unwrap_or("")
        );
    }

    let id = result.get("id").and_then(|v| v.as_str()).unwrap_or("?");
    let op = result
        .get("trashed_by")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let mut out = format!("Restored {} (removed by {}).", id, op);

    if result
        .get("overwrote_existing")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        out.push_str("\nA live memory with that ID was replaced; it is now in the trash itself.");
    }
    let restored = result
        .get("edges_restored")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if restored > 0 {
        out.push_str(&format!("\nReconnected {} relationship(s).", restored));
    }
    let dropped = result
        .get("edges_dropped")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if dropped > 0 {
        out.push_str(&format!(
            "\n{} relationship(s) could not be restored: the memory at the other end is gone.",
            dropped
        ));
    }
    out
}

fn compact_stats(result: &Value) -> String {
    let count = result
        .get("memory_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let rels = result
        .get("relationship_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let rel = result
        .get("avg_relevance")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let clusters = result
        .get("cluster_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let project = result
        .get("project_id")
        .and_then(|v| v.as_str())
        .unwrap_or("?");

    let dead = result
        .get("dead_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let trash = result
        .get("trash_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut out = format!(
        "Project: {}\nMemories: {}, Relationships: {}, Clusters: {}, Avg relevance: {:.2}",
        project, count, rels, clusters, rel
    );
    if dead > 0 {
        out.push_str(&format!("\nDead (excluded from retrieval): {}", dead));
    }
    if trash > 0 {
        out.push_str(&format!("\nRecoverable in trash: {}", trash));
    }
    out
}

fn compact_projects(result: &Value) -> String {
    let current = result
        .get("current_project")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let projects = result
        .get("projects")
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[]);

    if projects.is_empty() {
        return format!(
            "Current project: {}\nNo projects in the memory store.",
            current
        );
    }

    let mut out = format!("Current project: {}\n", current);
    for project in projects {
        let id = project
            .get("project_id")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let memories = project
            .get("memory_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let handoffs = project
            .get("handoff_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let adrs = project
            .get("adr_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let marker = if project.get("current").and_then(|v| v.as_bool()) == Some(true) {
            " (current)"
        } else {
            ""
        };
        out.push_str(&format!(
            "- {}{}: {} memories, {} handoffs, {} ADRs\n",
            id, marker, memories, handoffs, adrs
        ));
    }
    out.trim_end().to_string()
}

/// Lead with open work. The raw serialization buries `open_todos` behind whatever
/// `linked_memories` happens to contain, and a nudge the caller has to scroll past is not
/// a nudge — outstanding work is the first thing a resuming agent needs.
fn compact_handoff_resume(result: &Value) -> String {
    let mut out = String::new();

    let branch = result
        .get("branch")
        .and_then(|v| v.as_str())
        .unwrap_or("(no branch)");
    let latest = result
        .get("latest_handoff_id")
        .and_then(|v| v.as_str())
        .unwrap_or("none");
    let chain_len = result
        .get("chain")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    out.push_str(&format!(
        "Resuming {branch}: {chain_len} handoff(s) in chain, latest {latest}\n"
    ));

    if let Some(msg) = result.get("message").and_then(|v| v.as_str()) {
        out.push_str(&format!("Note: {msg}\n"));
    }

    // Always stated, including the empty case: silence reads as "no list exists" and the
    // caller stops looking.
    let todos = result
        .get("open_todos")
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or_default();
    if todos.is_empty() {
        out.push_str(
            "\nOpen todos: none. Add one with todo_write when work should outlive this session.\n",
        );
    } else {
        out.push_str("\nOpen todos (durable list — reconcile with todo_write as you work):\n");
        for t in todos {
            if let Some(text) = t.as_str() {
                out.push_str(&format!("- [ ] {text}\n"));
            }
        }
    }

    if let Some(blockers) = result.get("open_blockers").and_then(|v| v.as_array())
        && !blockers.is_empty()
    {
        out.push_str("\nOpen blockers:\n");
        for b in blockers {
            if let Some(text) = b.as_str() {
                out.push_str(&format!("- {text}\n"));
            }
        }
    }

    if let Some(sections) = result.get("top_sections").and_then(|v| v.as_array())
        && !sections.is_empty()
    {
        out.push_str("\nTop sections:\n");
        for sec in sections {
            let name = sec
                .get("section_name")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let hid = sec
                .get("handoff_id")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let score = sec.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let text = sec
                .get("section_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            out.push_str(&format!(
                "\n[{name}] {hid} ({})\n{text}\n",
                format_score(score)
            ));
        }
    }

    if let Some(linked) = result.get("linked_memories").and_then(|v| v.as_array())
        && !linked.is_empty()
    {
        out.push_str("\nLinked memories:\n");
        for m in linked {
            let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let mtype = m
                .get("memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("memory");
            let preview: String = m
                .get("summary")
                .and_then(|v| v.as_str())
                .or_else(|| m.get("content").and_then(|v| v.as_str()))
                .unwrap_or("")
                .chars()
                .take(160)
                .collect();
            out.push_str(&format!("- [{mtype}] {preview}  {id}\n"));
        }
    }

    out.trim_end().to_string()
}

/// Report what each op did, and surface `possible_duplicates` prominently — the whole
/// point of reporting them is that the caller reconsiders before leaving two copies.
fn compact_todo_write(result: &Value) -> String {
    let empty = vec![];
    let results = result
        .get("results")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    let open_count = result
        .get("open_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut out = String::new();
    for r in results {
        let op = r.get("op").and_then(|v| v.as_str()).unwrap_or("?");
        let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("");
        match r.get("error").and_then(|v| v.as_str()) {
            Some(err) => out.push_str(&format!("{op} failed: {err}\n")),
            None => out.push_str(&format!("{op} ok: {id}\n")),
        }
        if let Some(dups) = r.get("possible_duplicates").and_then(|v| v.as_array())
            && !dups.is_empty()
        {
            out.push_str("  possible duplicates of an existing open todo:\n");
            for d in dups {
                let did = d.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let text = d.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let sim = d.get("similarity").and_then(|v| v.as_f64()).unwrap_or(0.0);
                out.push_str(&format!("  - [{}] {} ({})\n", did, text, format_score(sim)));
            }
        }
    }
    out.push_str(&format!("{open_count} open todo(s)."));
    out
}

fn compact_todo_list(result: &Value) -> String {
    let empty = vec![];
    let todos = result
        .get("todos")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    let open = result
        .get("open_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let done = result
        .get("done_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let dropped = result
        .get("dropped_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let tally = format!("{open} open, {done} done, {dropped} dropped");

    if todos.is_empty() {
        return format!("No todos matched. Project totals: {tally}.");
    }

    let mut out = String::new();
    for t in todos {
        let text = t.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let id = t.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let status = t.get("status").and_then(|v| v.as_str()).unwrap_or("open");
        let mark = match status {
            "done" => "x",
            "dropped" => "~",
            _ => " ",
        };
        let scope = match t.get("branch").and_then(|v| v.as_str()) {
            Some(b) => format!(" [{b}]"),
            None => String::new(),
        };
        let reason = match (status, t.get("reason").and_then(|v| v.as_str())) {
            ("dropped", Some(r)) => format!(" (dropped: {r})"),
            _ => String::new(),
        };
        out.push_str(&format!("- [{mark}] {text}{scope}{reason}  {id}\n"));
    }
    out.push_str(&format!("Project totals: {tally}."));
    out
}

fn compact_fallback(result: &Value) -> String {
    // For unhandled tools, use minimal JSON
    serde_json::to_string(result).unwrap_or_else(|_| result.to_string())
}

fn format_score(score: f64) -> String {
    format!("{:.0}%", score * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_compact_projects() {
        let result = json!({
            "current_project": "home",
            "count": 2,
            "projects": [
                {
                    "project_id": "home",
                    "memory_count": 12,
                    "handoff_count": 2,
                    "adr_count": 1,
                    "latest_activity_at": 1_700_000_000,
                    "current": true,
                },
                {
                    "project_id": "other",
                    "memory_count": 3,
                    "handoff_count": 0,
                    "adr_count": 0,
                    "latest_activity_at": null,
                    "current": false,
                },
            ],
        });

        let out = compact_projects(&result);
        assert!(out.contains("Current project: home"));
        assert!(out.contains("- home (current): 12 memories, 2 handoffs, 1 ADRs"));
        assert!(out.contains("- other: 3 memories, 0 handoffs, 0 ADRs"));
    }

    #[test]
    fn test_compact_projects_empty() {
        let out = compact_projects(&json!({"current_project": "home", "count": 0, "projects": []}));
        assert!(out.contains("No projects in the memory store."));
    }

    #[test]
    fn test_compact_query_result() {
        let result = json!({
            "count": 2,
            "memories": [
                {
                    "memory": {
                        "id": "mem_123",
                        "memory_type": "fact",
                        "content": "This is a test memory",
                        "summary": "Test memory",
                        "tags": ["test"],
                        "importance": 0.8
                    },
                    "score": 0.95,
                    "semantic_score": 0.9,
                    "keyword_score": 0.5
                },
                {
                    "memory": {
                        "id": "mem_456",
                        "memory_type": "decision",
                        "content": "Another memory",
                        "tags": [],
                        "importance": 0.5
                    },
                    "score": 0.75,
                    "semantic_score": 0.7,
                    "keyword_score": 0.3
                }
            ]
        });

        let compact = compact_tool_result("memory_query", &result, 300);

        // Plain text format should contain memory IDs, types, content
        assert!(
            compact.contains("mem_123"),
            "Should contain first memory ID"
        );
        assert!(
            compact.contains("mem_456"),
            "Should contain second memory ID"
        );
        assert!(compact.contains("fact"), "Should contain memory type");
        assert!(compact.contains("decision"), "Should contain memory type");
        assert!(
            compact.contains("This is a test memory"),
            "Should contain content"
        );
        assert!(compact.contains("test"), "Should contain tags");
        assert!(compact.contains("2 result"), "Should show result count");
    }

    #[test]
    fn test_compact_store_result() {
        let result = json!({
            "id": "mem_789",
            "message": "Memory stored successfully"
        });

        let compact = compact_tool_result("memory_store", &result, 300);

        assert!(compact.contains("mem_789"), "Should contain memory ID");
        assert!(compact.starts_with("Stored"), "Should start with 'Stored'");
    }

    #[test]
    fn test_compact_vs_full_size() {
        let full_result = json!({
            "count": 3,
            "memories": [
                {
                    "memory": {
                        "id": "mem_001",
                        "project_id": "test-project",
                        "memory_type": "fact",
                        "content": "The project uses Rust for the backend implementation",
                        "summary": "Backend is Rust",
                        "tags": ["architecture", "rust"],
                        "importance": 0.8,
                        "relevance_score": 0.95,
                        "access_count": 5,
                        "created_at": 1700000000,
                        "updated_at": 1700000000,
                        "last_accessed_at": 1700001000
                    },
                    "score": 0.92,
                    "semantic_score": 0.88,
                    "keyword_score": 0.45
                }
            ]
        });

        let full_json = serde_json::to_string(&full_result).unwrap();
        let compact = compact_tool_result("memory_query", &full_result, 300);

        // Plain text should be significantly smaller than full JSON
        assert!(
            compact.len() < full_json.len(),
            "Compact ({}) should be smaller than full ({})",
            compact.len(),
            full_json.len()
        );
    }

    #[test]
    fn test_compact_query_content_length() {
        let long_content = "a".repeat(500);
        let result = json!({
            "memories": [
                {
                    "memory": {
                        "id": "mem_001",
                        "memory_type": "fact",
                        "content": long_content,
                        "tags": [],
                        "importance": 0.5
                    },
                    "score": 0.8
                }
            ]
        });

        let compact_100 = compact_tool_result("memory_query", &result, 100);
        let compact_300 = compact_tool_result("memory_query", &result, 300);

        assert!(compact_100.len() < compact_300.len());
        assert!(compact_100.contains("..."));
    }

    /// The whole point of `possible_supersedes` is that the caller is asked. Over MCP the
    /// tool result is the compact text and the JSON is discarded, so a field the renderer
    /// drops does not exist as far as an agent is concerned.
    #[test]
    fn store_output_surfaces_supersession_candidates() {
        let result = json!({
            "id": "mem_new",
            "message": "Memory stored successfully",
            "project": "proj",
            "possible_supersedes": [
                {"id": "mem_old", "similarity": 0.83, "type": "fact",
                 "preview": "Release gating runs on Jenkins.", "updated_at": 0}
            ]
        });
        let out = compact_tool_result("memory_store", &result, 300);
        assert!(out.contains("mem_old"), "candidate id missing from: {out}");
        assert!(out.contains("Release gating runs on Jenkins."));
        assert!(
            out.contains("supersedes"),
            "no instruction on what to do: {out}"
        );
    }

    #[test]
    fn store_output_reports_what_it_superseded() {
        let result = json!({
            "id": "mem_new",
            "message": "Memory stored successfully",
            "project": "proj",
            "superseded": ["mem_old"]
        });
        let out = compact_tool_result("memory_store", &result, 300);
        assert!(out.contains("Supersedes mem_old"), "got: {out}");
    }

    /// Update replaces content wholesale, so the previous version has to reach the caller.
    #[test]
    fn update_output_includes_the_replaced_content() {
        let result = json!({
            "success": true,
            "message": "Memory updated successfully",
            "content_replaced": true,
            "dead": false,
            "previous": {
                "id": "mem_1",
                "content": "Original wording, including a tail nobody re-read.",
                "memory_type": "fact"
            }
        });
        let out = compact_tool_result("memory_update", &result, 300);
        assert!(out.contains("Original wording, including a tail nobody re-read."));
        assert!(
            out.contains("memory_restore id=mem_1"),
            "no way back offered: {out}"
        );
    }

    #[test]
    fn update_output_reports_dead() {
        let result = json!({
            "success": true, "message": "Memory updated successfully",
            "content_replaced": false, "dead": true
        });
        let out = compact_tool_result("memory_update", &result, 300);
        assert!(out.contains("dead"), "got: {out}");
    }

    /// A delete must show the claims it destroyed, including any that dedup folded in:
    /// one id can stand for several memories.
    #[test]
    fn delete_output_includes_content_and_merged_predecessors() {
        let result = json!({
            "success": true,
            "message": "Memory deleted successfully",
            "recoverable": true,
            "deleted": {
                "id": "mem_1",
                "memory_type": "debug",
                "content": "A finding about current code.",
                "merged_from": [
                    {"id": "mem_0", "content_preview": "trunc",
                     "content": "A method lesson that is in no commit.", "merged_at": 0}
                ]
            }
        });
        let out = compact_tool_result("memory_delete", &result, 300);
        assert!(out.contains("A finding about current code."));
        assert!(
            out.contains("A method lesson that is in no commit."),
            "merged-in claim invisible, which is exactly how one gets lost: {out}"
        );
        assert!(out.contains("mem_0"));
    }

    #[test]
    fn delete_batch_output_includes_every_memory() {
        let result = json!({
            "success": true, "deleted": 2, "message": "2 memories deleted",
            "memories": [
                {"id": "mem_1", "content": "first claim", "memory_type": "fact"},
                {"id": "mem_2", "content": "second claim", "memory_type": "fact"}
            ]
        });
        let out = compact_tool_result("memory_delete_batch", &result, 300);
        assert!(
            out.contains("first claim") && out.contains("second claim"),
            "got: {out}"
        );
    }

    #[test]
    fn list_output_marks_superseded_and_dead() {
        let result = json!({
            "project": "proj", "status": "all", "order": "relevance",
            "total": 2, "count": 2, "offset": 0,
            "memories": [
                {"id": "mem_1", "type": "fact", "content": "old claim", "tags": [],
                 "importance": 0.5, "relevance_score": 1.0, "access_count": 0,
                 "created_at": 0, "updated_at": 0, "pinned": false,
                 "dead": false, "superseded_by": "mem_2"},
                {"id": "mem_3", "type": "fact", "content": "gone", "tags": [],
                 "importance": 0.5, "relevance_score": 1.0, "access_count": 0,
                 "created_at": 0, "updated_at": 0, "pinned": false,
                 "dead": true, "superseded_by": null}
            ]
        });
        let out = compact_tool_result("memory_list", &result, 300);
        assert!(out.contains("[superseded by mem_2]"), "got: {out}");
        assert!(out.contains("[dead]"), "got: {out}");
    }

    #[test]
    fn trash_and_restore_render_their_own_shape() {
        let trash = json!({
            "project": "proj", "total": 1, "count": 1,
            "entries": [{"trash_id": 3, "memory_id": "mem_1", "op": "merge",
                         "trashed_at": 0, "type": "fact", "preview": "consumed by dedup",
                         "content_chars": 759, "relationships": 0}]
        });
        let out = compact_tool_result("memory_trash", &trash, 300);
        assert!(
            out.contains("trash_id 3") && out.contains("by merge"),
            "got: {out}"
        );
        assert!(out.contains("759 chars"));

        let restore = json!({
            "success": true, "id": "mem_1", "trashed_by": "delete",
            "overwrote_existing": false, "edges_restored": 1, "edges_dropped": 2,
            "message": "Memory restored"
        });
        let out = compact_tool_result("memory_restore", &restore, 300);
        assert!(out.contains("Restored mem_1"), "got: {out}");
        assert!(out.contains("Reconnected 1"), "got: {out}");
        assert!(
            out.contains("2 relationship(s) could not be restored"),
            "got: {out}"
        );
    }

    /// A redirected result must say it is a stand-in; otherwise it reads as a direct hit
    /// on a query whose wording it does not even contain.
    #[test]
    fn query_and_context_output_mark_redirects() {
        let query = json!({
            "count": 1,
            "memories": [{
                "memory": {"id": "mem_new", "memory_type": "fact",
                           "content": "Gating is on GitHub Actions.", "tags": [], "importance": 0.5},
                "score": 0.7,
                "matched_via": {"superseded_id": "mem_old",
                                "superseded_preview": "Gating runs on Jenkins."}
            }]
        });
        let out = compact_tool_result("memory_query", &query, 300);
        assert!(out.contains("replaces mem_old"), "got: {out}");
        assert!(out.contains("Gating runs on Jenkins."), "got: {out}");

        let context = json!({
            "context": "gating", "count": 1, "retrieval_mode": "flat",
            "memories": [{"id": "mem_new", "type": "fact",
                          "content": "Gating is on GitHub Actions.", "tags": [],
                          "importance": 0.5, "similarity": 0.7,
                          "matched_via": {"superseded_id": "mem_old",
                                          "superseded_preview": "Gating runs on Jenkins."}}]
        });
        let out = compact_tool_result("memory_context", &context, 300);
        assert!(out.contains("replaces mem_old"), "got: {out}");
    }

    #[test]
    fn stats_output_includes_curation_counts() {
        let result = json!({
            "project_id": "proj", "memory_count": 10, "relationship_count": 2,
            "avg_relevance": 0.5, "cluster_count": 1, "dead_count": 3, "trash_count": 4
        });
        let out = compact_tool_result("memory_stats", &result, 300);
        assert!(
            out.contains("Dead (excluded from retrieval): 3"),
            "got: {out}"
        );
        assert!(out.contains("Recoverable in trash: 4"), "got: {out}");
    }
}
