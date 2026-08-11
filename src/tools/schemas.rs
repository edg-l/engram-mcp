#![allow(dead_code)]

use rmcp::model::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::Arc;

use crate::memory::HandoffSections;

// ============================================
// ToolProfile
// ============================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolProfile {
    #[default]
    Full,
    Core,
    Minimal,
}

impl std::str::FromStr for ToolProfile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "full" => Ok(Self::Full),
            "core" => Ok(Self::Core),
            "minimal" => Ok(Self::Minimal),
            other => Err(format!(
                "unknown ToolProfile {other:?}; expected full|core|minimal"
            )),
        }
    }
}

const MINIMAL_TOOLS: &[&str] = &["memory_context", "memory_store", "handoff_resume"];

const CORE_TOOLS: &[&str] = &[
    "memory_context",
    "memory_store",
    "handoff_resume",
    "memory_query",
    "memory_update",
    "memory_delete",
    "memory_link",
    "memory_graph",
    "handoff_create",
    "memory_store_batch",
    "memory_delete_batch",
    "memory_projects",
    "memory_list",
    "memory_restore",
    "adr_create",
    "adr_show",
    "adr_list",
];

fn filter_by_name(all: &[Tool], names: &[&str]) -> Vec<Tool> {
    all.iter()
        .filter(|t| names.contains(&&*t.name))
        .cloned()
        .collect()
}

pub fn get_tool_definitions_for(profile: ToolProfile) -> Vec<Tool> {
    let all = get_tool_definitions();
    match profile {
        ToolProfile::Full => all,
        ToolProfile::Core => filter_by_name(&all, CORE_TOOLS),
        ToolProfile::Minimal => filter_by_name(&all, MINIMAL_TOOLS),
    }
}

// ============================================
// Default helpers for serde
// ============================================

fn default_importance() -> f64 {
    0.5
}

fn default_branch_mode() -> String {
    "current".to_string()
}

fn default_semantic_weight() -> f64 {
    0.7
}

fn default_limit() -> usize {
    10
}

fn default_min_relevance() -> f64 {
    0.1
}

fn default_strength() -> f64 {
    1.0
}

fn default_depth() -> usize {
    2
}

fn default_import_mode() -> String {
    "merge".to_string()
}

fn default_context_limit() -> usize {
    5
}

fn default_context_min_score() -> f64 {
    0.3
}

fn default_hierarchical() -> bool {
    true
}

fn default_prune_threshold() -> f64 {
    0.2
}

fn default_dedup_threshold() -> f32 {
    0.90
}

fn default_handoff_importance() -> f64 {
    0.85
}

fn default_handoff_pinned() -> bool {
    true
}

fn default_auto_link() -> bool {
    true
}

fn default_max_sections() -> usize {
    5
}

fn default_include_off_branch() -> bool {
    false
}

fn default_true() -> bool {
    true
}

fn default_adr_importance() -> f64 {
    0.85
}

fn default_proposed_status() -> String {
    "proposed".into()
}

// ============================================
// Input arg structs
// ============================================

#[derive(Debug, Deserialize)]
pub struct MemoryStoreInput {
    pub content: String,
    /// Memory type. `memory_type` is accepted as an alias: callers reach for it
    /// often enough that rejecting it reads as the server losing the field.
    #[serde(rename = "type", alias = "memory_type")]
    pub memory_type: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_importance")]
    pub importance: f64,
    pub summary: Option<String>,
    #[serde(default)]
    pub related_to: Vec<String>,
    /// Memories this one replaces. Each gets a `supersedes` edge from the new memory and
    /// stops being returned by retrieval, which redirects to this one instead.
    #[serde(default)]
    pub supersedes: Vec<String>,
    /// Branch for this memory: null/omitted = global, "auto" = current branch, "branch-name" = explicit
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub pinned: bool,
    /// Make this memory visible across all projects. Global memories always have branch=null.
    #[serde(default)]
    pub global: bool,
    /// Optional list of external artifact references (file paths, URLs, ticket IDs).
    #[serde(default)]
    pub external_artifacts: Option<Vec<String>>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryQueryInput {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_min_relevance")]
    pub min_relevance: f64,
    #[serde(default)]
    pub types: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Deprecated. Ignored in all search modes — Hybrid uses RRF fusion, Vector
    /// uses cosine only, and Bm25 uses lexical scoring only. Retained for
    /// backwards compatibility with older callers; new clients should not set it.
    #[serde(default = "default_semantic_weight")]
    pub semantic_weight: f64,
    /// Branch mode: "current" (default) = global + current branch,
    /// "all" = all branches, "global" = global only, or "branch-name" = specific branch
    #[serde(default = "default_branch_mode")]
    pub branch_mode: String,
    /// Return superseded and dead memories as they are, with no redirect or suppression.
    /// For curation; ordinary retrieval should leave it off.
    #[serde(default)]
    pub include_superseded: bool,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryUpdateInput {
    pub id: String,
    pub content: Option<String>,
    pub importance: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub summary: Option<String>,
    pub pinned: Option<bool>,
    /// Mark the memory dead (its subject no longer exists) or bring it back. A dead
    /// memory is excluded from retrieval with no redirect, because there is nothing
    /// current to redirect to. Use `supersedes` on a new memory instead whenever a
    /// replacement exists.
    pub dead: Option<bool>,
    /// Why it was marked dead. Recorded alongside the flag.
    pub dead_reason: Option<String>,
    /// Replace external_artifacts list. Pass empty array to clear; omit to preserve existing.
    pub external_artifacts: Option<Vec<String>>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryListInput {
    /// Sort order: "created", "updated", "accessed", or "relevance" (default).
    #[serde(default = "default_list_order")]
    pub order: String,
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub types: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Status filter: "live" (default), "superseded", "dead", or "all".
    #[serde(default = "default_list_status")]
    pub status: String,
    /// Characters of content to show per memory.
    #[serde(default = "default_list_content_length")]
    pub content_length: usize,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

fn default_list_order() -> String {
    "relevance".to_string()
}

fn default_list_limit() -> usize {
    50
}

fn default_list_status() -> String {
    "live".to_string()
}

fn default_list_content_length() -> usize {
    160
}

#[derive(Debug, Deserialize)]
pub struct MemoryTrashInput {
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryRestoreInput {
    /// Memory id to restore. The most recent snapshot of it is used unless `trash_id`
    /// names a specific one.
    #[serde(default)]
    pub id: Option<String>,
    /// Exact snapshot to restore, from `memory_trash`.
    #[serde(default)]
    pub trash_id: Option<i64>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryDeleteInput {
    pub id: String,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryLinkInput {
    pub source_id: String,
    pub target_id: String,
    pub relation: String,
    #[serde(default = "default_strength")]
    pub strength: f64,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryGraphInput {
    pub id: String,
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default)]
    pub relation_types: Vec<String>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryStoreBatchInput {
    pub memories: Vec<MemoryStoreInput>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryDeleteBatchInput {
    pub ids: Vec<String>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryExportInput {
    #[serde(default)]
    pub include_embeddings: bool,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryImportInput {
    pub data: Value,
    #[serde(default = "default_import_mode")]
    pub mode: String,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryStatsInput {
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryContextInput {
    /// The context or conversation to find relevant memories for
    pub context: String,
    /// Maximum number of memories to return (default: 5)
    #[serde(default = "default_context_limit")]
    pub limit: usize,
    /// Minimum similarity score (default: 0.3)
    #[serde(default = "default_context_min_score")]
    pub min_score: f64,
    /// Filter by memory types
    #[serde(default)]
    pub types: Vec<String>,
    /// Enable hierarchical retrieval via clusters (default: true)
    #[serde(default = "default_hierarchical")]
    pub hierarchical: bool,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryPruneInput {
    /// Minimum relevance score to keep (memories below this are candidates for deletion)
    #[serde(default = "default_prune_threshold")]
    pub threshold: f64,
    /// If true, actually delete. If false (default), just show what would be deleted.
    #[serde(default)]
    pub confirm: bool,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryPromoteInput {
    /// ID of the memory to promote from branch-local to global
    pub id: String,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryDedupInput {
    /// Similarity threshold for duplicate detection (default: 0.90)
    #[serde(default = "default_dedup_threshold")]
    pub threshold: f32,
    /// If true, execute merges. If false (default), dry run.
    #[serde(default)]
    pub confirm: bool,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

/// Input for the `handoff_create` MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffCreateInput {
    /// Git branch to scope this handoff to. Defaults to the current branch.
    pub branch: Option<String>,
    /// Structured session sections.
    pub sections: HandoffSections,
    /// Importance score in [0, 1]. Default 0.85.
    #[serde(default = "default_handoff_importance")]
    pub importance: f64,
    /// Pin this handoff so it is exempt from decay and auto-prune. Default true.
    #[serde(default = "default_handoff_pinned")]
    pub pinned: bool,
    /// Auto-link this handoff to related decisions/patterns/debug memories. Default true.
    #[serde(default = "default_auto_link")]
    pub auto_link: bool,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

/// Input for the `handoff_resume` MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffResumeInput {
    /// Branch to fetch handoffs for. Defaults to the current branch.
    pub branch: Option<String>,
    /// Query string for scoring sections. Defaults to the latest handoff summary.
    pub query: Option<String>,
    /// Maximum number of top sections to return. Default 5.
    #[serde(default = "default_max_sections")]
    pub max_sections: usize,
    /// When true, include handoffs from all branches even if a branch was resolved. Default false.
    #[serde(default = "default_include_off_branch")]
    pub include_off_branch: bool,
    /// Cap on characters per returned `section_text`. Omitted = server default
    /// (1500); explicit `0` disables truncation. Oversized sections are truncated at
    /// the nearest paragraph/sentence boundary and annotated with a marker so the
    /// caller can recognise the elision and fetch the full text via the handoff ID.
    pub max_chars_per_section: Option<usize>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

/// Input for the `handoff_search` MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffSearchInput {
    /// Query text to score sections against.
    pub query: String,
    /// Branch to filter handoffs by. `None` means all branches.
    pub branch: Option<String>,
    /// Maximum number of matches to return. Default 10.
    pub limit: Option<usize>,
    /// Filter results to these section names only (e.g. `["blockers", "todos"]`).
    /// Case-insensitive. `None` means all sections.
    pub section_filter: Option<Vec<String>>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

/// Input for the `adr_create` MCP tool.
#[derive(Debug, Deserialize)]
pub struct AdrCreateInput {
    pub title: String,
    pub context: String,
    pub decision: String,
    pub consequences: String,
    #[serde(default = "default_proposed_status")]
    pub status: String,
    #[serde(default = "default_adr_importance")]
    pub importance: f64,
    #[serde(default = "default_true")]
    pub pinned: bool,
    #[serde(default)]
    pub supersedes: Option<u32>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

/// Input for the `adr_update_status` MCP tool.
#[derive(Debug, Deserialize)]
pub struct AdrUpdateStatusInput {
    pub number: u32,
    pub status: String,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

/// Input for the `adr_list` MCP tool.
#[derive(Debug, Deserialize)]
pub struct AdrListInput {
    #[serde(default)]
    pub status: Option<String>,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

/// Input for the `adr_show` MCP tool.
#[derive(Debug, Deserialize)]
pub struct AdrShowInput {
    pub number: u32,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

/// Input for the `adr_export` MCP tool.
#[derive(Debug, Deserialize)]
pub struct AdrExportInput {
    #[serde(default)]
    pub number: Option<u32>,
    #[serde(default)]
    pub dir: Option<String>,
    #[serde(default = "default_true")]
    pub dry_run: bool,
    /// Project to operate on. `None` = the server's own project.
    #[serde(default)]
    pub project: Option<String>,
}

// ============================================
// Utility fns used by schemas and tool definitions
// ============================================

/// Read dedup threshold from ENGRAM_DEDUP_THRESHOLD env var, clamped to [0.5, 1.0].
pub fn dedup_threshold() -> f32 {
    std::env::var("ENGRAM_DEDUP_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .map(|v| v.clamp(0.5, 1.0))
        .unwrap_or(0.90)
}

pub fn make_input_schema(schema: Value) -> Arc<Map<String, Value>> {
    match schema {
        Value::Object(map) => Arc::new(map),
        _ => Arc::new(Map::new()),
    }
}

/// Description of the shared optional `project` argument.
const PROJECT_ARG_DESCRIPTION: &str = "Project to operate on. Omit to use the server's own project (derived from its working directory). Pass a project ID from memory_projects to read or write another project's memories. Unknown project IDs are rejected.";

/// Build a tool input schema with the shared optional `project` argument added.
fn project_scoped_schema(schema: Value) -> Arc<Map<String, Value>> {
    let mut schema = schema;
    if let Some(properties) = schema.get_mut("properties").and_then(|p| p.as_object_mut()) {
        properties.insert(
            "project".to_string(),
            json!({"type": "string", "description": PROJECT_ARG_DESCRIPTION}),
        );
    }
    make_input_schema(schema)
}

pub fn get_tool_definitions() -> Vec<Tool> {
    vec![
        // === Core tools (used frequently by agents) ===
        Tool::new(
            "memory_store",
            "Save a piece of knowledge for later recall. Use this whenever you learn something worth remembering: project facts, architectural decisions, user preferences, recurring patterns, or debug findings. Duplicates are auto-detected and merged. Use `pinned: true` for permanent knowledge that must never decay, and `global: true` for knowledge that applies across all projects.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember. Be specific and self-contained -- this will be retrieved by semantic search later."},
                    // "handoff" is intentionally excluded; use handoff_create.
                    "type": {"type": "string", "enum": ["fact", "decision", "preference", "pattern", "debug", "entity"], "description": "fact=objective info, decision=choices made and why, preference=how the user likes things, pattern=recurring approaches/solutions, debug=troubleshooting findings, entity=people/systems/services"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "2-5 short lowercase tags for the topic. Tags improve search ranking -- use domain terms like 'database', 'auth', 'deployment'."},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "How critical this is. 0.3=minor detail, 0.5=normal (default), 0.7=important, 0.9=critical decision or constraint."},
                    "summary": {"type": "string", "description": "Optional short summary. Auto-generated for long content if omitted."},
                    "related_to": {"type": "array", "items": {"type": "string"}, "description": "Memory IDs this relates to. Creates 'relates_to' links."},
                    "supersedes": {"type": "array", "items": {"type": "string"}, "description": "Memory IDs this one replaces, when you have learned that an existing memory is no longer true. Those memories stop being returned by search; queries that would have matched them return this memory instead, marked with what it replaced. Use this rather than storing a second, contradicting memory. The store result reports `possible_supersedes` when it finds existing memories on the same subject."},
                    "branch": {"type": "string", "description": "Git branch scope. Omit for global (visible everywhere), 'auto' for current branch only, or an explicit branch name."},
                    "pinned": {"type": "boolean", "description": "Pin this memory so it never decays or gets pruned. Use for critical, permanent knowledge."},
                    "global": {"type": "boolean", "description": "Make this memory visible across all projects. Global memories always have branch=null."},
                    "external_artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of external artifact references (file paths, URLs, ticket IDs). Strings are surfaced at retrieval. Local-looking paths (absolute, ./ or ../, or drive-letter) are checked for existence and marked `[missing]` if absent on the server's filesystem."
                    }
                },
                "required": ["content", "type"]
            })),
        ),
        Tool::new(
            "memory_query",
            "Search for specific memories using a question or keywords. Use this when you need to find something you previously stored -- a specific fact, decision, or detail. Returns scored results with semantic + keyword matching. For broad context gathering, prefer memory_context instead.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language question or keywords. E.g. 'what database do we use' or 'authentication decision'."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Max results to return (default 10)."},
                    "offset": {"type": "integer", "minimum": 0, "description": "Skip first N results for pagination."},
                    "include_superseded": {"type": "boolean", "description": "Return superseded and dead memories as they are, with no redirect and no suppression. For auditing the store; leave off for normal recall."},
                    "min_relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Minimum stored relevance (decay) threshold (default 0.1). Memories with decay below this are excluded regardless of retrieval score. Note: memory_context uses min_score for cosine cutoff — this field gates on stored decay only."},
                    "types": {"type": "array", "items": {"type": "string"}, "description": "Filter by memory type(s): fact, decision, preference, pattern, debug, entity."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter to memories with any of these tags."},
                    "semantic_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Deprecated and ignored. Retained for backwards compatibility; the active retrieval mode is controlled by the ENGRAM_SEARCH_MODE env var."},
                    "branch_mode": {"type": "string", "description": "'current' (default) = global + current branch, 'all' = all branches, 'global' = global only, or a specific branch name."},
                    "content_length": {"type": "integer", "minimum": 1, "description": "Max characters to show per memory content (default 300)."}
                },
                "required": ["query"]
            })),
        ),
        Tool::new(
            "memory_context",
            "Retrieve memories relevant to your current task or conversation. Use this at the start of a task to load background knowledge, or when you need context about what the project does, how it works, or what decisions were made. Unlike memory_query, this is optimized for broad relevance rather than specific lookups.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "Describe what you're working on or thinking about. E.g. 'adding a new API endpoint for user profiles' or 'debugging the payment service timeout'."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Max memories to return (default 5)."},
                    "min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Minimum similarity threshold (default 0.3)."},
                    "types": {"type": "array", "items": {"type": "string"}, "description": "Filter by memory type(s)."},
                    "hierarchical": {"type": "boolean", "description": "Use cluster-based retrieval for diverse results (default true). Set false for flat similarity ranking."},
                    "content_length": {"type": "integer", "minimum": 1, "description": "Max characters to show per memory content (default 300)."}
                },
                "required": ["context"]
            })),
        ),
        Tool::new(
            "memory_update",
            "Correct or update an existing memory. Use when information has changed (e.g. a version was upgraded, a decision was revised). Only provide fields you want to change. Supports `pinned` to protect a memory from decay/pruning.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID to update (from a previous query result)."},
                    "content": {"type": "string", "description": "New content (replaces old, re-indexes for search)."},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "New importance level."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags (replaces old)."},
                    "summary": {"type": "string", "description": "New summary (replaces old)."},
                    "pinned": {"type": "boolean", "description": "Pin this memory so it never decays or gets pruned. Use for critical, permanent knowledge."},
                    "dead": {"type": "boolean", "description": "Mark this memory dead: its subject no longer exists (the service was retired, the file deleted) and there is no replacement. Dead memories are excluded from search entirely. If a replacement exists, store it with `supersedes` instead so searches get redirected rather than nothing."},
                    "dead_reason": {"type": "string", "description": "Why it is dead. Recorded alongside the flag."},
                    "external_artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Replace external_artifacts list. Pass empty array to clear; omit to preserve existing."
                    }
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_delete",
            "Remove a memory that is no longer relevant or was stored in error.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID to delete."}
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_stats",
            "Get a summary of stored memories: total count, relationship count, average relevance, and cluster count. Use to understand the current state of the memory store.",
            project_scoped_schema(json!({"type": "object", "properties": {}})),
        ),
        Tool::new(
            "memory_projects",
            "List every project in the memory store with its memory, handoff, and ADR counts. Use this to discover the project ID to pass as the `project` argument on other tools when you need another project's memories.",
            make_input_schema(json!({"type": "object", "properties": {}})),
        ),
        // === Relationship tools (use when tracking how knowledge connects) ===
        Tool::new(
            "memory_link",
            "Create a typed relationship between two memories. Use when one memory supersedes another (newer decision replaces older), or when you want to track that two memories are related or one is derived from another.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "source_id": {"type": "string", "description": "ID of the source memory."},
                    "target_id": {"type": "string", "description": "ID of the target memory."},
                    "relation": {"type": "string", "enum": ["relates_to", "supersedes", "derived_from"], "description": "relates_to=general connection, supersedes=source replaces target, derived_from=source was based on target."},
                    "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Relationship strength (default 1.0)."}
                },
                "required": ["source_id", "target_id", "relation"]
            })),
        ),
        Tool::new(
            "memory_graph",
            "Explore how a memory connects to others. Traverses the relationship graph outward from a memory, showing linked memories up to a configurable depth. Use when you need to understand the context around a specific decision or fact.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID to start traversal from."},
                    "depth": {"type": "integer", "minimum": 1, "maximum": 5, "description": "How many hops to traverse (default 2)."},
                    "relation_types": {"type": "array", "items": {"type": "string"}, "description": "Filter to specific relation types: relates_to, supersedes, derived_from."}
                },
                "required": ["id"]
            })),
        ),
        // === Batch and maintenance tools ===
        Tool::new(
            "memory_store_batch",
            "Store multiple memories at once (up to 100). More efficient than individual stores for bulk operations like ingesting documentation or session notes.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "type": {"type": "string", "enum": ["fact", "decision", "preference", "pattern", "debug", "entity"]},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "summary": {"type": "string"},
                                "external_artifacts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional list of external artifact references (file paths, URLs, ticket IDs). Strings are surfaced at retrieval. Local-looking paths (absolute, ./ or ../, or drive-letter) are checked for existence and marked `[missing]` if absent on the server's filesystem."
                                }
                            },
                            "required": ["content", "type"]
                        },
                        "maxItems": 100
                    }
                },
                "required": ["memories"]
            })),
        ),
        Tool::new(
            "memory_delete_batch",
            "Delete multiple memories by ID in one operation.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {"ids": {"type": "array", "items": {"type": "string"}, "description": "Memory IDs to delete."}},
                "required": ["ids"]
            })),
        ),
        Tool::new(
            "memory_export",
            "Export all project memories to JSON for backup or transfer to another project.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "include_embeddings": {"type": "boolean", "description": "Include embedding vectors in export (larger file, but avoids re-embedding on import)."}
                }
            })),
        ),
        Tool::new(
            "memory_import",
            "Import memories from a JSON export. Use 'merge' mode to add without overwriting, 'replace' to wipe and reload.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "data": {"type": "object", "description": "The JSON export data (from memory_export)."},
                    "mode": {"type": "string", "enum": ["merge", "replace"], "description": "'merge' (default) = skip existing IDs, 'replace' = delete all then import."}
                },
                "required": ["data"]
            })),
        ),
        Tool::new(
            "memory_list",
            "Enumerate stored memories directly, without a search query. Use this to audit or curate the store: search only shows what a query matches, so the memories most in need of attention are the ones it never surfaces. Unlike memory_query this can show superseded and dead memories.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "order": {"type": "string", "enum": ["relevance", "created", "updated", "accessed"], "description": "Sort order, newest/highest first (default 'relevance')."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Max memories to return (default 50)."},
                    "offset": {"type": "integer", "minimum": 0, "description": "Skip first N for pagination."},
                    "types": {"type": "array", "items": {"type": "string"}, "description": "Filter by memory type(s)."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter to memories carrying any of these tags."},
                    "status": {"type": "string", "enum": ["live", "superseded", "dead", "all"], "description": "'live' (default) = what retrieval would return; 'superseded' = replaced by a newer memory; 'dead' = subject is gone; 'all' = everything."},
                    "content_length": {"type": "integer", "minimum": 1, "description": "Characters of content to show per memory (default 160)."}
                }
            })),
        ),
        Tool::new(
            "memory_trash",
            "List memories that were deleted, pruned, merged away by dedup, or overwritten by an update, and are still recoverable. Every destructive operation snapshots the memory here first. Use memory_restore to bring one back.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Max entries to return (default 50)."}
                }
            })),
        ),
        Tool::new(
            "memory_restore",
            "Restore a memory from the trash, along with its embedding and any relationships whose other end still exists. Pass `id` for the most recent snapshot of that memory, or `trash_id` from memory_trash for one exact snapshot.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID to restore. Uses its most recent snapshot."},
                    "trash_id": {"type": "integer", "description": "Exact snapshot to restore, from memory_trash."}
                },
                // Neither field alone is required, but one of them is. Expressed here so
                // the published contract matches what the handler enforces.
                "anyOf": [{"required": ["id"]}, {"required": ["trash_id"]}]
            })),
        ),
        Tool::new(
            "memory_prune",
            "Clean up memories that have decayed below a relevance threshold. Memories decay over time if not accessed. Dry run by default -- shows what would be removed without deleting. Pruned memories go to the trash and can be restored.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Remove memories with relevance below this (default 0.2)."},
                    "confirm": {"type": "boolean", "description": "Set true to actually delete. Default false (dry run)."}
                }
            })),
        ),
        Tool::new(
            "memory_promote",
            "Make a branch-scoped memory visible globally. Use when a branch-specific finding should be preserved across all branches.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID to promote from branch-local to global."}
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_dedup",
            "Scan for and merge duplicate memories. Finds memory pairs with high semantic similarity (same type, similarity above threshold) and merges them, preserving tags and importance from both. Dry run by default.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 0.5, "maximum": 1.0, "description": "Similarity threshold for duplicates (default 0.90). Lower = more aggressive dedup."},
                    "confirm": {"type": "boolean", "description": "Set true to execute merges. Default false (dry run, shows what would be merged)."}
                }
            })),
        ),
        // === Handoff tools ===
        Tool::new(
            "handoff_create",
            "Create a session handoff capturing decisions, todos, blockers, dead ends tried, mental model, and next steps. Pinned by default; bypasses dedup.\n\nIMPORTANT — section shape: each section is a SHORT SUMMARY, not a transcript. Hard guidance: keep each section under ~1500 chars; individual list items under ~300 chars. Do NOT paste verbatim tool output, full agent reports, file dumps, or chat logs. If long context matters, store it as a separate memory (memory_store with type=debug/pattern/decision) and rely on auto-linking — those memories surface in handoff_resume's linked_memories. Oversized sections trigger a warning in the response.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Git branch to scope this handoff to. Defaults to the current branch."
                    },
                    "sections": {
                        "type": "object",
                        "description": "Structured session sections. Each section is a short summary — NOT a transcript or full report. Store long content as separate memories and let auto-linking surface them.",
                        "properties": {
                            "summary": {"type": "string", "description": "1–3 sentence summary of the session. Keep under ~500 chars."},
                            "decisions": {"type": "array", "items": {"type": "string"}, "description": "Key decisions, one short line each (what + why, ≤300 chars per item). No transcripts."},
                            "todos": {"type": "array", "items": {"type": "string"}, "description": "Within-session work the next agent should pick up immediately. Concrete, ready-to-execute items, one short line each. Restate any todo from the previous handoff that is still open — resume returns these verbatim, so an omitted todo reads as done."},
                            "blockers": {"type": "array", "items": {"type": "string"}, "description": "Things preventing forward motion right now (missing access, failing dependency, unanswered question). One short line each. Restate any blocker from the previous handoff that is still unresolved — resume returns these verbatim, so an omitted blocker reads as resolved."},
                            "tried": {"type": "array", "items": {"type": "string"}, "description": "Approaches attempted and abandoned, each with the concrete reason it failed, so the next session does not retry them. One short line each ('X, because Y'). Dead ends are permanent — do not restate ones already recorded in an earlier handoff."},
                            "mental_model": {"type": "string", "description": "Architecture or context the next session needs. 1–5 sentences or a short bulleted list. Not a deep dive — link related decision/pattern memories instead."},
                            "next_steps": {"type": "array", "items": {"type": "string"}, "description": "Post-session follow-ups beyond the current thread. Future-facing, not for immediate pickup. One short line each."},
                            "notes": {"type": "string", "description": "Freeform short notes (optional). Do not paste reports or logs here."},
                            "continues_from": {"type": "string", "description": "ID of the handoff this continues from (optional)."}
                        },
                        "required": ["summary"]
                    },
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Importance score (default 0.85)."},
                    "pinned": {"type": "boolean", "description": "Pin this handoff (exempt from decay/prune). Default true."},
                    "auto_link": {"type": "boolean", "description": "Auto-link to related decisions/patterns/debug memories. Default true."}
                },
                "required": ["sections"]
            })),
        ),
        Tool::new(
            "handoff_resume",
            "Resume a session by retrieving the most relevant sections from recent handoffs on the current (or specified) branch, plus linked decisions/patterns/debug notes.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch to fetch handoffs for. Defaults to the current branch."
                    },
                    "query": {
                        "type": "string",
                        "description": "Query string for scoring sections. Defaults to the latest handoff summary."
                    },
                    "max_sections": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum number of top sections to return (default 5)."
                    },
                    "include_off_branch": {
                        "type": "boolean",
                        "description": "Include handoffs from all branches (default false)."
                    },
                    "max_chars_per_section": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Truncate each returned section_text to this many characters. Omitted = server default (1500); explicit 0 disables truncation. Truncated sections cut at the nearest paragraph/sentence boundary and are marked with '… [truncated, N chars total]' so you know to fetch the full text via handoff_search or the memory:// resource."
                    }
                }
            })),
        ),
        Tool::new(
            "handoff_search",
            "Search session handoffs by section content. Filter by branch and/or section name.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for across handoff sections."
                    },
                    "branch": {
                        "type": "string",
                        "description": "Limit results to this branch. Omit to search all branches."
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of matches to return (default 10)."
                    },
                    "section_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only return matches from these section names (e.g. [\"blockers\", \"todos\"]). Case-insensitive."
                    }
                },
                "required": ["query"]
            })),
        ),
        // === ADR tools ===
        Tool::new(
            "adr_create",
            "Create an Architecture Decision Record (ADR) in Nygard style. ADRs are project-global (no branch scope), pinned by default, exempt from decay, and bypass dedup. Use `supersedes` to mark an existing ADR as superseded when this decision replaces it.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short, imperative-mood title (e.g. 'Use SQLite for local storage')."
                    },
                    "context": {
                        "type": "string",
                        "description": "Forces that drove this decision: constraints, requirements, and alternatives considered."
                    },
                    "decision": {
                        "type": "string",
                        "description": "The decision made, stated in full."
                    },
                    "consequences": {
                        "type": "string",
                        "description": "Positive and negative consequences of this decision."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["proposed", "accepted", "superseded", "deprecated", "rejected"],
                        "description": "Lifecycle status (default: proposed)."
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Importance score (default 0.85)."
                    },
                    "pinned": {
                        "type": "boolean",
                        "description": "Pin this ADR so it is exempt from decay and auto-prune (default true)."
                    },
                    "supersedes": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "ADR number this decision supersedes. Marks the old ADR as 'superseded' and creates a Supersedes relationship."
                    }
                },
                "required": ["title", "context", "decision", "consequences"]
            })),
        ),
        Tool::new(
            "adr_update_status",
            "Advance an ADR through its lifecycle (proposed → accepted → deprecated/superseded, etc.). Use adr_create with supersedes to mark an ADR superseded via a new decision. Direct 'superseded' transitions via this tool are rejected — use adr_create with the supersedes field instead.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "ADR number to update."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["proposed", "accepted", "deprecated", "rejected"],
                        "description": "New lifecycle status. 'superseded' is not allowed here; use adr_create with supersedes instead."
                    }
                },
                "required": ["number", "status"]
            })),
        ),
        Tool::new(
            "adr_list",
            "List ADRs for the current project, ordered by ADR number. ADRs are project-global.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["proposed", "accepted", "superseded", "deprecated", "rejected"],
                        "description": "Filter to ADRs with this status. Omit to return all."
                    }
                }
            })),
        ),
        Tool::new(
            "adr_show",
            "Retrieve full details of a single ADR by number.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "ADR number to retrieve."
                    }
                },
                "required": ["number"]
            })),
        ),
        Tool::new(
            "adr_export",
            "Export one or all ADRs to Markdown files on disk. `dry_run` (default true) lists what would be written without creating files. Set `dry_run: false` to write — existing files are overwritten silently. `dir` sets the output directory (default: docs/adr relative to the server's cwd). `number` exports a single ADR; omit to export all.",
            project_scoped_schema(json!({
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "ADR number to export. Omit to export all ADRs."
                    },
                    "dir": {
                        "type": "string",
                        "description": "Output directory for Markdown files (default: docs/adr)."
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true (default), list files that would be written without creating them. Set false to write."
                    }
                }
            })),
        ),
    ]
}
