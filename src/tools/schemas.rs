#![allow(dead_code)]

use rmcp::model::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::Arc;

use crate::memory::HandoffSections;

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

// ============================================
// Input arg structs
// ============================================

#[derive(Debug, Deserialize)]
pub struct MemoryStoreInput {
    pub content: String,
    #[serde(rename = "type")]
    pub memory_type: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_importance")]
    pub importance: f64,
    pub summary: Option<String>,
    #[serde(default)]
    pub related_to: Vec<String>,
    /// Branch for this memory: null/omitted = global, "auto" = current branch, "branch-name" = explicit
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub pinned: bool,
    /// Make this memory visible across all projects. Global memories always have branch=null.
    #[serde(default)]
    pub global: bool,
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
}

#[derive(Debug, Deserialize)]
pub struct MemoryUpdateInput {
    pub id: String,
    pub content: Option<String>,
    pub importance: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub summary: Option<String>,
    pub pinned: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryDeleteInput {
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct MemoryLinkInput {
    pub source_id: String,
    pub target_id: String,
    pub relation: String,
    #[serde(default = "default_strength")]
    pub strength: f64,
}

#[derive(Debug, Deserialize)]
pub struct MemoryGraphInput {
    pub id: String,
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default)]
    pub relation_types: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryStoreBatchInput {
    pub memories: Vec<MemoryStoreInput>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryDeleteBatchInput {
    pub ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryExportInput {
    #[serde(default)]
    pub include_embeddings: bool,
}

#[derive(Debug, Deserialize)]
pub struct MemoryImportInput {
    pub data: Value,
    #[serde(default = "default_import_mode")]
    pub mode: String,
}

#[derive(Debug, Deserialize)]
pub struct MemoryStatsInput {}

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
}

#[derive(Debug, Deserialize)]
pub struct MemoryPruneInput {
    /// Minimum relevance score to keep (memories below this are candidates for deletion)
    #[serde(default = "default_prune_threshold")]
    pub threshold: f64,
    /// If true, actually delete. If false (default), just show what would be deleted.
    #[serde(default)]
    pub confirm: bool,
}

#[derive(Debug, Deserialize)]
pub struct MemoryPromoteInput {
    /// ID of the memory to promote from branch-local to global
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct MemoryDedupInput {
    /// Similarity threshold for duplicate detection (default: 0.90)
    #[serde(default = "default_dedup_threshold")]
    pub threshold: f32,
    /// If true, execute merges. If false (default), dry run.
    #[serde(default)]
    pub confirm: bool,
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

pub fn get_tool_definitions() -> Vec<Tool> {
    vec![
        // === Core tools (used frequently by agents) ===
        Tool::new(
            "memory_store",
            "Save a piece of knowledge for later recall. Use this whenever you learn something worth remembering: project facts, architectural decisions, user preferences, recurring patterns, or debug findings. Duplicates are auto-detected and merged. Use `pinned: true` for permanent knowledge that must never decay, and `global: true` for knowledge that applies across all projects.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember. Be specific and self-contained -- this will be retrieved by semantic search later."},
                    // "handoff" is intentionally excluded; use handoff_create.
                    "type": {"type": "string", "enum": ["fact", "decision", "preference", "pattern", "debug", "entity"], "description": "fact=objective info, decision=choices made and why, preference=how the user likes things, pattern=recurring approaches/solutions, debug=troubleshooting findings, entity=people/systems/services"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "2-5 short lowercase tags for the topic. Tags improve search ranking -- use domain terms like 'database', 'auth', 'deployment'."},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "How critical this is. 0.3=minor detail, 0.5=normal (default), 0.7=important, 0.9=critical decision or constraint."},
                    "summary": {"type": "string", "description": "Optional short summary. Auto-generated for long content if omitted."},
                    "related_to": {"type": "array", "items": {"type": "string"}, "description": "Memory IDs this relates to. Creates 'relates_to' links."},
                    "branch": {"type": "string", "description": "Git branch scope. Omit for global (visible everywhere), 'auto' for current branch only, or an explicit branch name."},
                    "pinned": {"type": "boolean", "description": "Pin this memory so it never decays or gets pruned. Use for critical, permanent knowledge."},
                    "global": {"type": "boolean", "description": "Make this memory visible across all projects. Global memories always have branch=null."}
                },
                "required": ["content", "type"]
            })),
        ),
        Tool::new(
            "memory_query",
            "Search for specific memories using a question or keywords. Use this when you need to find something you previously stored -- a specific fact, decision, or detail. Returns scored results with semantic + keyword matching. For broad context gathering, prefer memory_context instead.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language question or keywords. E.g. 'what database do we use' or 'authentication decision'."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Max results to return (default 10)."},
                    "offset": {"type": "integer", "minimum": 0, "description": "Skip first N results for pagination."},
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
            make_input_schema(json!({
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
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID to update (from a previous query result)."},
                    "content": {"type": "string", "description": "New content (replaces old, re-indexes for search)."},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "New importance level."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags (replaces old)."},
                    "summary": {"type": "string", "description": "New summary (replaces old)."},
                    "pinned": {"type": "boolean", "description": "Pin this memory so it never decays or gets pruned. Use for critical, permanent knowledge."}
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_delete",
            "Remove a memory that is no longer relevant or was stored in error.",
            make_input_schema(json!({
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
            make_input_schema(json!({"type": "object", "properties": {}})),
        ),
        // === Relationship tools (use when tracking how knowledge connects) ===
        Tool::new(
            "memory_link",
            "Create a typed relationship between two memories. Use when one memory supersedes another (newer decision replaces older), when memories contradict each other, or when you want to track that two memories are related or one is derived from another.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "source_id": {"type": "string", "description": "ID of the source memory."},
                    "target_id": {"type": "string", "description": "ID of the target memory."},
                    "relation": {"type": "string", "enum": ["relates_to", "supersedes", "derived_from", "contradicts"], "description": "relates_to=general connection, supersedes=source replaces target, derived_from=source was based on target, contradicts=source conflicts with target."},
                    "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Relationship strength (default 1.0)."}
                },
                "required": ["source_id", "target_id", "relation"]
            })),
        ),
        Tool::new(
            "memory_graph",
            "Explore how a memory connects to others. Traverses the relationship graph outward from a memory, showing linked memories up to a configurable depth. Use when you need to understand the context around a specific decision or fact.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID to start traversal from."},
                    "depth": {"type": "integer", "minimum": 1, "maximum": 5, "description": "How many hops to traverse (default 2)."},
                    "relation_types": {"type": "array", "items": {"type": "string"}, "description": "Filter to specific relation types: relates_to, supersedes, derived_from, contradicts."}
                },
                "required": ["id"]
            })),
        ),
        // === Batch and maintenance tools ===
        Tool::new(
            "memory_store_batch",
            "Store multiple memories at once (up to 100). More efficient than individual stores for bulk operations like ingesting documentation or session notes.",
            make_input_schema(json!({
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
                                "summary": {"type": "string"}
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
            make_input_schema(json!({
                "type": "object",
                "properties": {"ids": {"type": "array", "items": {"type": "string"}, "description": "Memory IDs to delete."}},
                "required": ["ids"]
            })),
        ),
        Tool::new(
            "memory_export",
            "Export all project memories to JSON for backup or transfer to another project.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "include_embeddings": {"type": "boolean", "description": "Include embedding vectors in export (larger file, but avoids re-embedding on import)."}
                }
            })),
        ),
        Tool::new(
            "memory_import",
            "Import memories from a JSON export. Use 'merge' mode to add without overwriting, 'replace' to wipe and reload.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "data": {"type": "object", "description": "The JSON export data (from memory_export)."},
                    "mode": {"type": "string", "enum": ["merge", "replace"], "description": "'merge' (default) = skip existing IDs, 'replace' = delete all then import."}
                },
                "required": ["data"]
            })),
        ),
        Tool::new(
            "memory_prune",
            "Clean up memories that have decayed below a relevance threshold. Memories decay over time if not accessed. Dry run by default -- shows what would be removed without deleting.",
            make_input_schema(json!({
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
            make_input_schema(json!({
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
            make_input_schema(json!({
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
            "Create a session handoff capturing decisions, todos, blockers, mental model, and next steps. Pinned by default; bypasses dedup and contradiction detection.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Git branch to scope this handoff to. Defaults to the current branch."
                    },
                    "sections": {
                        "type": "object",
                        "description": "Structured session sections.",
                        "properties": {
                            "summary": {"type": "string", "description": "High-level summary of the session."},
                            "decisions": {"type": "array", "items": {"type": "string"}, "description": "Key decisions made."},
                            "todos": {"type": "array", "items": {"type": "string"}, "description": "Outstanding action items."},
                            "blockers": {"type": "array", "items": {"type": "string"}, "description": "Known blockers."},
                            "mental_model": {"type": "string", "description": "Architecture or context the next session needs."},
                            "next_steps": {"type": "array", "items": {"type": "string"}, "description": "Concrete next steps."},
                            "notes": {"type": "string", "description": "Freeform notes (optional)."},
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
            make_input_schema(json!({
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
                    }
                }
            })),
        ),
        Tool::new(
            "handoff_search",
            "Search session handoffs by section content. Filter by branch and/or section name.",
            make_input_schema(json!({
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
    ]
}
