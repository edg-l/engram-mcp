//! MCP tool handlers for the Engram memory server.
//!
//! All items in this module are used by the MCP server binary (main.rs).
//! The dead_code warnings appear because the CLI binary doesn't use these.
#![allow(dead_code)]

use rmcp::model::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::sync::Arc;

use crate::cache::{QueryEmbeddingCache, SearchResultCache};
use crate::db::Database;
use crate::embedding::{EmbeddingService, cosine_similarity};
use crate::error::MemoryError;
use crate::export::{self, ExportData, ExportedMemory, ImportMode, ImportStats};
use crate::memory::{
    Memory, MemoryType, MemoryWithScore, ProjectStats, RelationType, Relationship,
};
use crate::summarize::{generate_summary, should_auto_summarize};

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
}

fn default_importance() -> f64 {
    0.5
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
    /// Weight for semantic search (0.0-1.0). Keyword weight = 1 - semantic_weight.
    /// Default 0.7 means 70% semantic, 30% keyword.
    #[serde(default = "default_semantic_weight")]
    pub semantic_weight: f64,
    /// Branch mode: "current" (default) = global + current branch,
    /// "all" = all branches, "global" = global only, or "branch-name" = specific branch
    #[serde(default = "default_branch_mode")]
    pub branch_mode: String,
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

#[derive(Debug, Deserialize)]
pub struct MemoryUpdateInput {
    pub id: String,
    pub content: Option<String>,
    pub importance: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub summary: Option<String>,
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

fn default_strength() -> f64 {
    1.0
}

#[derive(Debug, Deserialize)]
pub struct MemoryGraphInput {
    pub id: String,
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default)]
    pub relation_types: Vec<String>,
}

fn default_depth() -> usize {
    2
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

fn default_import_mode() -> String {
    "merge".to_string()
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

fn default_context_limit() -> usize {
    5
}

fn default_context_min_score() -> f64 {
    0.3
}

fn default_hierarchical() -> bool {
    true
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

fn default_prune_threshold() -> f64 {
    0.2
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

fn default_dedup_threshold() -> f32 {
    0.90
}

/// Read dedup threshold from ENGRAM_DEDUP_THRESHOLD env var, clamped to [0.5, 1.0].
fn dedup_threshold() -> f32 {
    std::env::var("ENGRAM_DEDUP_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .map(|v| v.clamp(0.5, 1.0))
        .unwrap_or(0.90)
}

#[derive(Debug, Serialize)]
pub struct MergeInfo {
    pub merged_with: String,
    pub similarity: f64,
    pub old_content_preview: String,
}

#[derive(Debug, Serialize)]
pub struct MemoryStoreResult {
    pub id: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub potential_contradictions: Vec<PotentialContradiction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge_info: Option<MergeInfo>,
}

#[derive(Debug, Serialize)]
pub struct PotentialContradiction {
    pub memory_id: String,
    pub summary: String,
    pub similarity: f64,
}

#[derive(Debug, Serialize)]
pub struct ContradictionWarning {
    pub memory_id: String,
    pub contradicts_id: String,
    pub content_preview: String,
}

#[derive(Debug, Serialize)]
pub struct MemoryQueryResult {
    pub memories: Vec<MemoryWithScore>,
    pub count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub contradiction_warnings: Vec<ContradictionWarning>,
}

#[derive(Debug, Serialize)]
pub struct MemoryGraphResult {
    pub root: Memory,
    pub related: Vec<RelatedMemory>,
}

#[derive(Debug, Serialize)]
pub struct RelatedMemory {
    pub memory: Memory,
    pub relation: String,
    pub direction: String,
    pub depth: usize,
}

fn make_input_schema(schema: Value) -> Arc<Map<String, Value>> {
    match schema {
        Value::Object(map) => Arc::new(map),
        _ => Arc::new(Map::new()),
    }
}

pub fn get_tool_definitions() -> Vec<Tool> {
    vec![
        Tool::new(
            "memory_store",
            "Store a memory (fact/decision/preference/pattern/debug/entity).",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {"type": "string", "enum": ["fact", "decision", "preference", "pattern", "debug", "entity"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "summary": {"type": "string"},
                    "related_to": {"type": "array", "items": {"type": "string"}, "description": "IDs to link"},
                    "branch": {"type": "string", "description": "Branch for this memory: null/omitted = global, 'auto' = current branch, or explicit branch name"}
                },
                "required": ["content", "type"]
            })),
        ),
        Tool::new(
            "memory_query",
            "Search memories (hybrid semantic + keyword).",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "offset": {"type": "integer", "minimum": 0},
                    "min_relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "semantic_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "0-1, default 0.7"},
                    "branch_mode": {"type": "string", "description": "'current' (default) = global + current branch, 'all' = all branches, 'global' = global only, or specific branch name"}
                },
                "required": ["query"]
            })),
        ),
        Tool::new(
            "memory_update",
            "Update a memory.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"}
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_delete",
            "Delete a memory.",
            make_input_schema(json!({
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_link",
            "Link two memories.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "relation": {"type": "string", "enum": ["relates_to", "supersedes", "derived_from", "contradicts"]},
                    "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["source_id", "target_id", "relation"]
            })),
        ),
        Tool::new(
            "memory_graph",
            "Get memory with related memories.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "depth": {"type": "integer", "minimum": 1, "maximum": 5},
                    "relation_types": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_store_batch",
            "Store multiple memories (max 100).",
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
            "Delete multiple memories.",
            make_input_schema(json!({
                "type": "object",
                "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["ids"]
            })),
        ),
        Tool::new(
            "memory_export",
            "Export memories to JSON.",
            make_input_schema(json!({
                "type": "object",
                "properties": {"include_embeddings": {"type": "boolean"}}
            })),
        ),
        Tool::new(
            "memory_import",
            "Import memories from JSON.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                    "mode": {"type": "string", "enum": ["merge", "replace"]}
                },
                "required": ["data"]
            })),
        ),
        Tool::new(
            "memory_stats",
            "Get memory statistics.",
            make_input_schema(json!({"type": "object", "properties": {}})),
        ),
        Tool::new(
            "memory_context",
            "Get memories relevant to context.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "context": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "hierarchical": {"type": "boolean", "description": "Use cluster-based retrieval (default true)"}
                },
                "required": ["context"]
            })),
        ),
        Tool::new(
            "memory_prune",
            "Remove decayed memories (dry run by default).",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "confirm": {"type": "boolean"}
                }
            })),
        ),
        Tool::new(
            "memory_promote",
            "Promote a branch-local memory to global visibility.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID to promote"}
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_dedup",
            "Find and merge duplicate memories (dry run by default).",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 0.5, "maximum": 1.0, "description": "Similarity threshold (default 0.90)"},
                    "confirm": {"type": "boolean", "description": "Set true to execute merges"}
                }
            })),
        ),
    ]
}

pub struct ToolHandler {
    db: Database,
    embedding: EmbeddingService,
    project_id: String,
    /// Current git branch (None if not in git repo)
    current_branch: Option<String>,
    /// Cache for query embeddings to avoid recomputation
    query_cache: QueryEmbeddingCache,
    /// Cache for search results to avoid repeated similarity computations
    search_cache: SearchResultCache,
}

impl ToolHandler {
    pub fn new(
        db: Database,
        embedding: EmbeddingService,
        project_id: String,
        current_branch: Option<String>,
    ) -> Self {
        Self {
            db,
            embedding,
            project_id,
            current_branch,
            query_cache: QueryEmbeddingCache::new(),
            search_cache: SearchResultCache::new(),
        }
    }

    /// Get the current branch.
    pub fn current_branch(&self) -> Option<&str> {
        self.current_branch.as_deref()
    }

    /// Get a reference to the embedding service for reuse.
    pub fn embedding_service(&self) -> &EmbeddingService {
        &self.embedding
    }

    /// Get a reference to the database for reuse.
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Get the project ID.
    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    /// Convert branch_mode string to the Option<Option<&str>> format for DB queries.
    /// - "current" -> global + current branch (falls back to "all" if no branch detected)
    /// - "global" -> global only (branch IS NULL)
    /// - "all" -> no filter
    /// - other -> global + that specific branch
    fn branch_mode_to_filter<'a>(&'a self, branch_mode: &'a str) -> Option<Option<&'a str>> {
        match branch_mode {
            "all" => None,
            "global" => Some(None),
            "current" => {
                match self.current_branch.as_deref() {
                    Some(branch) => Some(Some(branch)),
                    None => None, // Fall back to "all" if no branch detected
                }
            }
            specific => Some(Some(specific)),
        }
    }

    /// Invalidate search result cache (call after memory modifications).
    fn invalidate_search_cache(&self) {
        self.search_cache.invalidate_project(&self.project_id);
    }

    /// Find memories that are potential duplicates of the given embedding.
    /// Returns (memory_id, similarity) pairs where similarity >= threshold and same type.
    fn find_duplicates(
        &self,
        embedding: &[f32],
        memory_type: MemoryType,
        threshold: f32,
        existing_embeddings: &[(String, Vec<f32>)],
    ) -> Result<Vec<(String, f32)>, MemoryError> {
        let mut duplicates = Vec::new();

        for (existing_id, existing_vec) in existing_embeddings {
            let similarity = cosine_similarity(embedding, existing_vec);
            if similarity >= threshold {
                // Check same memory type
                if let Ok(Some(mem)) = self.db.get_memory(existing_id)
                    && mem.memory_type == memory_type
                {
                    duplicates.push((existing_id.clone(), similarity));
                }
            }
        }

        // Sort by similarity descending
        duplicates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(duplicates)
    }

    pub fn handle_tool(&self, name: &str, arguments: Value) -> Result<Value, MemoryError> {
        match name {
            "memory_store" => self.memory_store(arguments),
            "memory_query" => self.memory_query(arguments),
            "memory_update" => self.memory_update(arguments),
            "memory_delete" => self.memory_delete(arguments),
            "memory_link" => self.memory_link(arguments),
            "memory_graph" => self.memory_graph(arguments),
            "memory_store_batch" => self.memory_store_batch(arguments),
            "memory_delete_batch" => self.memory_delete_batch(arguments),
            "memory_export" => self.memory_export(arguments),
            "memory_import" => self.memory_import(arguments),
            "memory_stats" => self.memory_stats(arguments),
            "memory_context" => self.memory_context(arguments),
            "memory_prune" => self.memory_prune(arguments),
            "memory_promote" => self.memory_promote(arguments),
            "memory_dedup" => self.memory_dedup(arguments),
            _ => Ok(json!({"error": format!("Unknown tool: {}", name)})),
        }
    }

    fn memory_store(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryStoreInput = serde_json::from_value(arguments)?;

        let memory_type: MemoryType = input
            .memory_type
            .parse()
            .map_err(|_| MemoryError::InvalidType(input.memory_type.clone()))?;

        let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
        let now = chrono::Utc::now().timestamp();

        // Auto-generate summary if needed
        let summary = if should_auto_summarize(&input.content, input.summary.as_deref()) {
            Some(generate_summary(&input.content))
        } else {
            input.summary
        };

        // Resolve branch: null/omitted = global (None), "auto" = current branch, else explicit
        let branch = match input.branch.as_deref() {
            None | Some("") => None, // Global
            Some("auto") => self.current_branch.clone(),
            Some(explicit) => Some(explicit.to_string()),
        };

        let memory = Memory {
            id: id.clone(),
            project_id: self.project_id.clone(),
            memory_type,
            content: input.content.clone(),
            summary,
            tags: input.tags,
            importance: input.importance.clamp(0.0, 1.0),
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: branch.clone(),
            merged_from: None,
        };

        // Generate embedding first to check for duplicates
        let embedding = self.embedding.embed_memory(memory_type, &input.content)?;

        // Fetch all embeddings once (reused for both dedup and contradiction checks)
        let pre_store_embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;

        // Check for duplicates (configurable via ENGRAM_DEDUP_THRESHOLD, default 0.90)
        let dedup_threshold = dedup_threshold();
        let duplicates =
            self.find_duplicates(&embedding, memory_type, dedup_threshold, &pre_store_embeddings)?;

        // Store the new memory
        self.db.store_memory(&memory)?;
        self.db
            .store_embedding(&id, &embedding, self.embedding.model_version())?;

        // If duplicate found, merge (keep new, absorb old)
        let mut merge_info: Option<MergeInfo> = None;
        if let Some((dup_id, dup_similarity)) = duplicates.first() {
            let old_preview: String = self
                .db
                .get_memory(dup_id)?
                .map(|m| m.content.chars().take(100).collect())
                .unwrap_or_default();

            self.db.merge_memories(&id, dup_id, &old_preview)?;

            merge_info = Some(MergeInfo {
                merged_with: dup_id.clone(),
                similarity: *dup_similarity as f64,
                old_content_preview: old_preview,
            });
        }

        // Check for contradictions using pre-fetched embeddings, excluding self and any merged duplicate
        let mut potential_contradictions: Vec<PotentialContradiction> = Vec::new();
        const CONTRADICTION_THRESHOLD: f32 = 0.85;
        let merged_id: Option<&str> = merge_info.as_ref().map(|mi| mi.merged_with.as_str());

        for (existing_id, existing_vec) in &pre_store_embeddings {
            if existing_id.as_str() == id.as_str() {
                continue; // Skip self
            }
            if merged_id == Some(existing_id.as_str()) {
                continue; // Skip the merged-away duplicate
            }
            let similarity = cosine_similarity(&embedding, existing_vec);
            if similarity >= CONTRADICTION_THRESHOLD
                && let Ok(Some(existing_memory)) = self.db.get_memory(existing_id)
            {
                potential_contradictions.push(PotentialContradiction {
                    memory_id: existing_id.clone(),
                    summary: existing_memory
                        .summary
                        .unwrap_or_else(|| existing_memory.content.chars().take(100).collect()),
                    similarity: similarity as f64,
                });
            }
        }

        // Create relationships to related memories
        for related_id in input.related_to {
            let rel = Relationship {
                id: format!("rel_{}", uuid::Uuid::new_v4().simple()),
                source_id: id.clone(),
                target_id: related_id,
                relation_type: RelationType::RelatesTo,
                strength: 1.0,
                created_at: now,
            };
            self.db.create_relationship(&rel)?;
        }

        // Assign to cluster
        let _cluster_id = self.assign_to_cluster(&id, &embedding, &input.content, memory.importance)?;

        // Invalidate search cache since we added new data
        self.invalidate_search_cache();

        let message = if merge_info.is_some() {
            "Memory stored and merged with duplicate".to_string()
        } else if potential_contradictions.is_empty() {
            "Memory stored successfully".to_string()
        } else {
            format!(
                "Memory stored. Warning: {} potential contradiction(s) detected - consider using memory_link with 'supersedes' or 'contradicts' relation.",
                potential_contradictions.len()
            )
        };

        Ok(json!(MemoryStoreResult {
            id,
            message,
            branch,
            potential_contradictions,
            merge_info,
        }))
    }

    fn memory_query(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryQueryInput = serde_json::from_value(arguments)?;

        // Parse type filters
        let type_filters: Vec<MemoryType> =
            input.types.iter().filter_map(|t| t.parse().ok()).collect();

        // Clamp semantic_weight to valid range
        let semantic_weight = input.semantic_weight.clamp(0.0, 1.0);
        let keyword_weight = 1.0 - semantic_weight;

        // Optimization: if query is empty, skip search and use filter-only path
        if input.query.trim().is_empty() {
            let branch_filter = self.branch_mode_to_filter(&input.branch_mode);
            let memories = self.db.query_memories_with_branch(
                &self.project_id,
                if type_filters.is_empty() {
                    None
                } else {
                    Some(&type_filters)
                },
                if input.tags.is_empty() {
                    None
                } else {
                    Some(&input.tags)
                },
                Some(input.min_relevance),
                input.limit + input.offset,
                branch_filter,
            )?;

            let results: Vec<MemoryWithScore> = memories
                .into_iter()
                .skip(input.offset)
                .take(input.limit)
                .map(|m| {
                    let score = m.relevance_score;
                    MemoryWithScore {
                        memory: m,
                        score,
                        semantic_score: 0.0,
                        keyword_score: 0.0,
                    }
                })
                .collect();

            return Ok(json!(MemoryQueryResult {
                count: results.len(),
                memories: results,
                contradiction_warnings: vec![],
            }));
        }

        // Run semantic search
        let query_embedding = if let Some(cached) = self.query_cache.get(&input.query) {
            cached
        } else {
            let embedding = self.embedding.embed(&input.query)?;
            self.query_cache
                .insert(input.query.clone(), embedding.clone());
            embedding
        };

        // Get semantic scores
        let semantic_scores: std::collections::HashMap<String, f32> = if let Some(cached_results) =
            self.search_cache.get(&self.project_id, &query_embedding)
        {
            cached_results.into_iter().collect()
        } else {
            let embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;

            let scored: Vec<(String, f32)> = embeddings
                .iter()
                .map(|(id, vec)| {
                    let similarity = cosine_similarity(&query_embedding, vec);
                    (id.clone(), similarity)
                })
                .collect();

            // Cache the results
            self.search_cache
                .insert(&self.project_id, &query_embedding, scored.clone());
            scored.into_iter().collect()
        };

        let branch_filter = self.branch_mode_to_filter(&input.branch_mode);

        // Run keyword search (FTS5) with branch filtering applied at the DB level
        let keyword_results = self.db.keyword_search_with_branch(
            &self.project_id,
            &input.query,
            input.limit * 5, // Get more to ensure we have enough after filtering
            branch_filter,
        )?;

        // Normalize keyword scores (BM25 scores can vary widely)
        // Find max keyword score for normalization
        let max_keyword_score = keyword_results
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0_f64, f64::max);

        let keyword_scores: std::collections::HashMap<String, f64> = if max_keyword_score > 0.0 {
            keyword_results
                .into_iter()
                .map(|(id, score)| (id, score / max_keyword_score))
                .collect()
        } else {
            std::collections::HashMap::new()
        };

        // Collect all candidate IDs from both searches
        let mut candidate_ids: HashSet<String> = semantic_scores.keys().cloned().collect();
        candidate_ids.extend(keyword_scores.keys().cloned());

        // Batch fetch all candidate memories
        let candidate_ids_vec: Vec<String> = candidate_ids.into_iter().collect();
        let memories_map = self.db.get_memories_batch(&candidate_ids_vec)?;

        // Calculate hybrid scores and build results
        let mut scored_results: Vec<(String, f64, f64, f64)> = Vec::new(); // (id, combined, semantic, keyword)

        for id in candidate_ids_vec.iter() {
            let Some(memory) = memories_map.get(id) else {
                continue;
            };

            // Filter by branch
            match branch_filter {
                None => {} // "all" - no filtering
                Some(None) => {
                    // "global" - only global memories
                    if memory.branch.is_some() {
                        continue;
                    }
                }
                Some(Some(branch)) => {
                    // specific branch - global + that branch
                    if let Some(ref mem_branch) = memory.branch
                        && mem_branch != branch
                    {
                        continue;
                    }
                    // branch is None (global) -> include
                }
            }

            // Filter by types
            if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
                continue;
            }

            // Filter by tags
            if !input.tags.is_empty() && !input.tags.iter().any(|t| memory.tags.contains(t)) {
                continue;
            }

            let semantic_score = *semantic_scores.get(id).unwrap_or(&0.0) as f64;
            let keyword_score = *keyword_scores.get(id).unwrap_or(&0.0);

            // Hybrid score: weighted combination of semantic and keyword scores
            let hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score;

            // Apply relevance decay
            let final_score = hybrid_score * memory.relevance_score;

            if final_score >= input.min_relevance {
                scored_results.push((id.clone(), final_score, semantic_score, keyword_score));
            }
        }

        // Sort by combined score descending
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply pagination and build final results
        let mut results: Vec<MemoryWithScore> = Vec::new();
        let mut result_ids: Vec<String> = Vec::new();

        for (id, score, semantic_score, keyword_score) in scored_results
            .into_iter()
            .skip(input.offset)
            .take(input.limit)
        {
            if let Some(memory) = memories_map.get(&id) {
                result_ids.push(id);
                let mut memory_clone = memory.clone();
                memory_clone.access_count += 1;
                results.push(MemoryWithScore {
                    memory: memory_clone,
                    score,
                    semantic_score,
                    keyword_score,
                });
            }
        }

        // Batch record access for all result memories
        if !result_ids.is_empty() {
            let _ = self.db.record_access_batch(&result_ids);
        }

        // Batch check for contradiction relationships among returned memories
        let contradiction_warnings = self.check_contradictions_batch(&result_ids)?;

        Ok(json!(MemoryQueryResult {
            count: results.len(),
            memories: results,
            contradiction_warnings,
        }))
    }

    /// Check for contradiction relationships among a set of memory IDs using batch operations.
    fn check_contradictions_batch(
        &self,
        result_ids: &[String],
    ) -> Result<Vec<ContradictionWarning>, MemoryError> {
        if result_ids.is_empty() {
            return Ok(Vec::new());
        }

        let result_id_set: HashSet<&String> = result_ids.iter().collect();

        // Batch fetch all outgoing relationships (1 query instead of N)
        let relationships_map = self.db.get_relationships_from_batch(result_ids)?;

        // Collect IDs of targets that are contradicted AND in our result set
        let mut target_ids_to_fetch: Vec<String> = Vec::new();
        let mut contradiction_pairs: Vec<(String, String)> = Vec::new();

        for (source_id, rels) in &relationships_map {
            for rel in rels {
                if rel.relation_type == RelationType::Contradicts
                    && result_id_set.contains(&rel.target_id)
                {
                    target_ids_to_fetch.push(rel.target_id.clone());
                    contradiction_pairs.push((source_id.clone(), rel.target_id.clone()));
                }
            }
        }

        if contradiction_pairs.is_empty() {
            return Ok(Vec::new());
        }

        // Batch fetch target memories for content preview (1 query instead of M)
        let targets_map = self.db.get_memories_batch(&target_ids_to_fetch)?;

        // Build warnings
        let mut warnings: Vec<ContradictionWarning> = Vec::new();
        for (source_id, target_id) in contradiction_pairs {
            if let Some(target) = targets_map.get(&target_id) {
                warnings.push(ContradictionWarning {
                    memory_id: source_id,
                    contradicts_id: target_id,
                    content_preview: target.content.chars().take(100).collect(),
                });
            }
        }

        Ok(warnings)
    }

    fn memory_update(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryUpdateInput = serde_json::from_value(arguments)?;

        let mut memory = self
            .db
            .get_memory(&input.id)?
            .ok_or_else(|| MemoryError::NotFound(input.id.clone()))?;

        let now = chrono::Utc::now().timestamp();
        memory.updated_at = now;

        if let Some(content) = input.content {
            memory.content = content.clone();
            // Re-generate embedding
            let embedding = self.embedding.embed_memory(memory.memory_type, &content)?;
            self.db
                .store_embedding(&memory.id, &embedding, self.embedding.model_version())?;

            // Regenerate summary if content changed and no explicit summary provided
            if input.summary.is_none() && should_auto_summarize(&content, memory.summary.as_deref())
            {
                memory.summary = Some(generate_summary(&content));
            }
        }

        if let Some(importance) = input.importance {
            memory.importance = importance.clamp(0.0, 1.0);
        }

        if let Some(tags) = input.tags {
            memory.tags = tags;
        }

        if let Some(summary) = input.summary {
            memory.summary = Some(summary);
        }

        self.db.update_memory(&memory)?;

        // Invalidate search cache since we updated data
        self.invalidate_search_cache();

        Ok(json!({"success": true, "message": "Memory updated successfully"}))
    }

    fn memory_delete(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDeleteInput = serde_json::from_value(arguments)?;

        // Remove from cluster before deleting
        if let Ok(Some(cluster_id)) = self.db.remove_from_cluster(&input.id) {
            // Recalculate centroid if cluster still has members
            let member_ids = self.db.get_cluster_member_ids(&cluster_id)?;
            if member_ids.is_empty() {
                let _ = self.db.delete_empty_clusters(&self.project_id);
            } else {
                let new_centroid = self.compute_cluster_centroid(&member_ids)?;
                let summary = self.generate_cluster_summary(&member_ids)?;
                if let Some(centroid) = new_centroid {
                    let _ = self.db.update_cluster_centroid(&cluster_id, &centroid, &summary);
                }
            }
        }

        let deleted = self.db.delete_memory(&input.id)?;

        if deleted {
            // Invalidate search cache since we deleted data
            self.invalidate_search_cache();
            Ok(json!({"success": true, "message": "Memory deleted successfully"}))
        } else {
            Ok(json!({"success": false, "message": "Memory not found"}))
        }
    }

    fn memory_link(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryLinkInput = serde_json::from_value(arguments)?;

        let relation_type: RelationType = input
            .relation
            .parse()
            .map_err(|_| MemoryError::InvalidRelation(input.relation.clone()))?;

        // Verify both memories exist
        self.db
            .get_memory(&input.source_id)?
            .ok_or_else(|| MemoryError::NotFound(input.source_id.clone()))?;
        self.db
            .get_memory(&input.target_id)?
            .ok_or_else(|| MemoryError::NotFound(input.target_id.clone()))?;

        let rel = Relationship {
            id: format!("rel_{}", uuid::Uuid::new_v4().simple()),
            source_id: input.source_id,
            target_id: input.target_id,
            relation_type,
            strength: input.strength.clamp(0.0, 1.0),
            created_at: chrono::Utc::now().timestamp(),
        };

        self.db.create_relationship(&rel)?;

        Ok(json!({"success": true, "id": rel.id, "message": "Relationship created successfully"}))
    }

    fn memory_graph(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryGraphInput = serde_json::from_value(arguments)?;

        let root = self
            .db
            .get_memory(&input.id)?
            .ok_or_else(|| MemoryError::NotFound(input.id.clone()))?;

        let relation_filters: Vec<RelationType> = input
            .relation_types
            .iter()
            .filter_map(|r| r.parse().ok())
            .collect();

        // BFS traversal with batch operations
        // O(depth * 3) queries instead of O(nodes * 3)
        let related = self.traverse_graph_bfs(&input.id, input.depth, &relation_filters)?;

        // Record access to root memory
        self.db.record_access(&input.id)?;

        Ok(json!(MemoryGraphResult { root, related }))
    }

    /// BFS-based graph traversal using batch operations for efficiency.
    /// Processes nodes level by level, batching relationship and memory fetches.
    fn traverse_graph_bfs(
        &self,
        start_id: &str,
        max_depth: usize,
        relation_filters: &[RelationType],
    ) -> Result<Vec<RelatedMemory>, MemoryError> {
        let mut results: Vec<RelatedMemory> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start_id.to_string());

        // Queue holds (memory_id, depth)
        let mut current_level: Vec<String> = vec![start_id.to_string()];

        for current_depth in 1..=max_depth {
            if current_level.is_empty() {
                break;
            }

            // Batch fetch outgoing relationships for entire level
            let outgoing_map = self.db.get_relationships_from_batch(&current_level)?;
            // Batch fetch incoming relationships for entire level
            let incoming_map = self.db.get_relationships_to_batch(&current_level)?;

            // Collect all new neighbor IDs and their relationship info
            // (neighbor_id, relation_type_str, direction, from_id)
            let mut neighbors_info: Vec<(String, String, String)> = Vec::new();
            let mut neighbor_ids: Vec<String> = Vec::new();

            // Process outgoing
            for rels in outgoing_map.values() {
                for rel in rels {
                    if visited.contains(&rel.target_id) {
                        continue;
                    }
                    if !relation_filters.is_empty()
                        && !relation_filters.contains(&rel.relation_type)
                    {
                        continue;
                    }
                    if !visited.contains(&rel.target_id) {
                        visited.insert(rel.target_id.clone());
                        neighbor_ids.push(rel.target_id.clone());
                        neighbors_info.push((
                            rel.target_id.clone(),
                            rel.relation_type.as_str().to_string(),
                            "outgoing".to_string(),
                        ));
                    }
                }
            }

            // Process incoming
            for rels in incoming_map.values() {
                for rel in rels {
                    if visited.contains(&rel.source_id) {
                        continue;
                    }
                    if !relation_filters.is_empty()
                        && !relation_filters.contains(&rel.relation_type)
                    {
                        continue;
                    }
                    if !visited.contains(&rel.source_id) {
                        visited.insert(rel.source_id.clone());
                        neighbor_ids.push(rel.source_id.clone());
                        neighbors_info.push((
                            rel.source_id.clone(),
                            rel.relation_type.as_str().to_string(),
                            "incoming".to_string(),
                        ));
                    }
                }
            }

            if neighbor_ids.is_empty() {
                break;
            }

            // Batch fetch all neighbor memories
            let memories_map = self.db.get_memories_batch(&neighbor_ids)?;

            // Build results for this level
            for (neighbor_id, relation, direction) in neighbors_info {
                if let Some(memory) = memories_map.get(&neighbor_id) {
                    results.push(RelatedMemory {
                        memory: memory.clone(),
                        relation,
                        direction,
                        depth: current_depth,
                    });
                }
            }

            // Next level: all neighbors found at this level
            current_level = neighbor_ids;
        }

        Ok(results)
    }

    fn memory_store_batch(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryStoreBatchInput = serde_json::from_value(arguments)?;

        if input.memories.len() > 100 {
            return Ok(json!({"success": false, "error": "Maximum 100 memories per batch"}));
        }

        let now = chrono::Utc::now().timestamp();
        let mut memories: Vec<Memory> = Vec::new();
        let mut embeddings: Vec<(String, Vec<f32>, String)> = Vec::new();
        let mut ids: Vec<String> = Vec::new();

        // Prepare all memories and embeddings
        let mut contents: Vec<String> = Vec::new();
        for mem_input in &input.memories {
            let memory_type: MemoryType = mem_input
                .memory_type
                .parse()
                .map_err(|_| MemoryError::InvalidType(mem_input.memory_type.clone()))?;
            contents.push(format!("{}: {}", memory_type.as_str(), &mem_input.content));
        }

        // Batch embed all content
        let all_embeddings = self.embedding.embed_batch(contents)?;

        for (i, mem_input) in input.memories.into_iter().enumerate() {
            let memory_type: MemoryType = mem_input
                .memory_type
                .parse()
                .map_err(|_| MemoryError::InvalidType(mem_input.memory_type.clone()))?;

            let id = format!("mem_{}", uuid::Uuid::new_v4().simple());

            // Auto-generate summary if needed
            let summary = if should_auto_summarize(&mem_input.content, mem_input.summary.as_deref())
            {
                Some(generate_summary(&mem_input.content))
            } else {
                mem_input.summary
            };

            // Resolve branch: null/omitted = global (None), "auto" = current branch, else explicit
            let branch = match mem_input.branch.as_deref() {
                None | Some("") => None, // Global
                Some("auto") => self.current_branch.clone(),
                Some(explicit) => Some(explicit.to_string()),
            };

            let memory = Memory {
                id: id.clone(),
                project_id: self.project_id.clone(),
                memory_type,
                content: mem_input.content,
                summary,
                tags: mem_input.tags,
                importance: mem_input.importance.clamp(0.0, 1.0),
                relevance_score: 1.0,
                access_count: 0,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                branch,
                merged_from: None,
            };

            embeddings.push((
                id.clone(),
                all_embeddings[i].clone(),
                self.embedding.model_version().to_string(),
            ));
            ids.push(id);
            memories.push(memory);
        }

        // Store memories in batch
        let stored = self.db.store_memories_batch(&memories)?;
        self.db.store_embeddings_batch(&embeddings)?;

        // Assign each new memory to a cluster
        for (i, mem) in memories.iter().enumerate() {
            let _ = self.assign_to_cluster(&mem.id, &all_embeddings[i], &mem.content, mem.importance);
        }

        // Invalidate search cache since we added new data
        if stored > 0 {
            self.invalidate_search_cache();
        }

        Ok(json!({
            "success": true,
            "count": stored,
            "ids": ids,
            "message": format!("{} memories stored successfully", stored)
        }))
    }

    fn memory_delete_batch(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDeleteBatchInput = serde_json::from_value(arguments)?;

        // Remove from clusters before deleting
        let mut affected_clusters: HashSet<String> = HashSet::new();
        for id in &input.ids {
            if let Ok(Some(cluster_id)) = self.db.remove_from_cluster(id) {
                affected_clusters.insert(cluster_id);
            }
        }

        let deleted = self.db.delete_memories_batch(&input.ids)?;

        // Cleanup affected clusters
        for cluster_id in &affected_clusters {
            let member_ids = self.db.get_cluster_member_ids(cluster_id)?;
            if member_ids.is_empty() {
                let _ = self.db.delete_empty_clusters(&self.project_id);
            } else {
                let new_centroid = self.compute_cluster_centroid(&member_ids)?;
                let summary = self.generate_cluster_summary(&member_ids)?;
                if let Some(centroid) = new_centroid {
                    let _ = self.db.update_cluster_centroid(cluster_id, &centroid, &summary);
                }
            }
        }

        if deleted > 0 {
            // Invalidate search cache since we deleted data
            self.invalidate_search_cache();
        }

        Ok(json!({
            "success": true,
            "deleted": deleted,
            "message": format!("{} memories deleted", deleted)
        }))
    }

    fn memory_export(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryExportInput = serde_json::from_value(arguments)?;

        let memories = self.db.get_all_memories_for_project(&self.project_id)?;
        let relationships = self
            .db
            .get_all_relationships_for_project(&self.project_id)?;

        let embeddings = if input.include_embeddings {
            Some(self.db.get_all_embeddings_for_project(&self.project_id)?)
        } else {
            None
        };

        let export_data = export::create_export(
            &self.project_id,
            memories,
            relationships,
            embeddings,
            Some(self.embedding.model_version().to_string()),
        );

        Ok(json!(export_data))
    }

    fn memory_import(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryImportInput = serde_json::from_value(arguments)?;

        let export_data: ExportData = serde_json::from_value(input.data)?;

        // Validate version
        export::validate_import(&export_data).map_err(MemoryError::Embedding)?;

        // Warn about embedding model version mismatch
        let model_warning: Option<String> =
            export_data.model_version.as_ref().and_then(|imported_model| {
                if imported_model != self.embedding.model_version() {
                    Some(format!(
                        "Warning: embeddings were generated with '{}' but current model is '{}'. Re-embedding recommended.",
                        imported_model,
                        self.embedding.model_version()
                    ))
                } else {
                    None
                }
            });

        let mode: ImportMode = input.mode.parse().unwrap_or(ImportMode::Merge);

        let mut stats = ImportStats::default();

        // In replace mode, clear existing data first
        if mode == ImportMode::Replace {
            self.db.delete_project_data(&self.project_id)?;
        }

        let now = chrono::Utc::now().timestamp();

        // Import memories
        for exported in export_data.memories {
            let ExportedMemory {
                mut memory,
                embedding: encoded_embedding,
            } = exported;

            // Update project_id to current project
            memory.project_id = self.project_id.clone();
            memory.updated_at = now;

            // Check if memory already exists (in merge mode)
            if mode == ImportMode::Merge && self.db.get_memory(&memory.id)?.is_some() {
                stats.memories_skipped += 1;
                continue;
            }

            self.db.store_memory(&memory)?;
            stats.memories_imported += 1;

            // Import embedding if provided
            if let Some(encoded) = encoded_embedding {
                if let Ok(vector) = export::decode_embedding(&encoded) {
                    self.db
                        .store_embedding(&memory.id, &vector, self.embedding.model_version())?;
                    stats.embeddings_imported += 1;
                }
            } else {
                // Generate new embedding
                let embedding = self
                    .embedding
                    .embed_memory(memory.memory_type, &memory.content)?;
                self.db
                    .store_embedding(&memory.id, &embedding, self.embedding.model_version())?;
                stats.embeddings_imported += 1;
            }
        }

        // Import relationships
        for rel in export_data.relationships {
            // Verify both memories exist
            let source_exists = self.db.get_memory(&rel.source_id)?.is_some();
            let target_exists = self.db.get_memory(&rel.target_id)?.is_some();

            if source_exists && target_exists {
                self.db.create_relationship(&rel)?;
                stats.relationships_imported += 1;
            } else {
                stats.relationships_skipped += 1;
            }
        }

        // Invalidate search cache since we imported data
        if stats.memories_imported > 0 {
            self.invalidate_search_cache();
        }

        Ok(json!({
            "success": true,
            "stats": stats,
            "message": format!(
                "Imported {} memories, {} relationships ({} skipped)",
                stats.memories_imported,
                stats.relationships_imported,
                stats.memories_skipped + stats.relationships_skipped
            ),
            "model_warning": model_warning,
        }))
    }

    fn memory_stats(&self, _arguments: Value) -> Result<Value, MemoryError> {
        let stats: ProjectStats = self.db.get_project_stats(&self.project_id)?;
        let clusters = self.db.get_clusters_for_project(&self.project_id)?;

        Ok(json!({
            "project_id": self.project_id,
            "memory_count": stats.memory_count,
            "relationship_count": stats.relationship_count,
            "avg_relevance": stats.avg_relevance,
            "cluster_count": clusters.len(),
        }))
    }

    fn memory_context(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryContextInput = serde_json::from_value(arguments)?;

        // Parse type filters
        let type_filters: Vec<MemoryType> =
            input.types.iter().filter_map(|t| t.parse().ok()).collect();

        // Generate embedding for the context
        let context_embedding = if let Some(cached) = self.query_cache.get(&input.context) {
            cached
        } else {
            let embedding = self.embedding.embed(&input.context)?;
            self.query_cache
                .insert(input.context.clone(), embedding.clone());
            embedding
        };

        // Check if hierarchical retrieval is viable (avoid DB queries when not requested)
        let should_use_hierarchical = if input.hierarchical {
            let clusters_result = self.db.get_clusters_for_project(&self.project_id)?;
            if !clusters_result.is_empty() {
                let stats = self.db.get_project_stats(&self.project_id)?;
                if stats.memory_count >= 10 {
                    Some(clusters_result)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(clusters) = should_use_hierarchical {
            // Hierarchical: query cluster centroids, then fetch members from top clusters
            let mut cluster_scores: Vec<(String, f32)> = Vec::new();
            for cluster in &clusters {
                if let Some(ref centroid) = cluster.centroid {
                    let similarity = cosine_similarity(&context_embedding, centroid);
                    if similarity >= input.min_score as f32 {
                        cluster_scores.push((cluster.id.clone(), similarity));
                    }
                }
            }
            cluster_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Get members from top clusters
            let mut memories: Vec<Value> = Vec::new();
            let mut memory_ids: Vec<String> = Vec::new();

            let all_embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;
            let embedding_map: std::collections::HashMap<String, Vec<f32>> =
                all_embeddings.into_iter().collect();

            for (cluster_id, _cluster_sim) in cluster_scores.iter().take(input.limit) {
                let member_ids = self.db.get_cluster_member_ids(cluster_id)?;
                let member_set: std::collections::HashSet<&String> = member_ids.iter().collect();

                // Score individual members using pre-fetched embeddings
                let mut member_scores: Vec<(String, f32)> = embedding_map
                    .iter()
                    .filter(|(id, _)| member_set.contains(id))
                    .map(|(id, vec)| (id.clone(), cosine_similarity(&context_embedding, vec)))
                    .collect();
                member_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Batch fetch all scored member memories
                let scored_ids: Vec<String> = member_scores.iter().map(|(id, _)| id.clone()).collect();
                let members_map = self.db.get_memories_batch(&scored_ids)?;

                for (id, similarity) in member_scores {
                    if memories.len() >= input.limit {
                        break;
                    }
                    if let Some(memory) = members_map.get(&id) {
                        // Apply branch filter (default: current branch mode)
                        let branch_filter = self.branch_mode_to_filter("current");
                        match branch_filter {
                            None => {}
                            Some(None) => {
                                if memory.branch.is_some() {
                                    continue;
                                }
                            }
                            Some(Some(branch)) => {
                                if let Some(ref mem_branch) = memory.branch
                                    && mem_branch != branch
                                {
                                    continue;
                                }
                            }
                        }

                        if !type_filters.is_empty()
                            && !type_filters.contains(&memory.memory_type)
                        {
                            continue;
                        }

                        memory_ids.push(id);
                        memories.push(json!({
                            "id": memory.id,
                            "type": memory.memory_type.as_str(),
                            "content": memory.content,
                            "summary": memory.summary,
                            "tags": memory.tags,
                            "importance": memory.importance,
                            "relevance_score": memory.relevance_score,
                            "similarity": similarity,
                            "cluster_id": cluster_id,
                        }));
                    }
                }
            }

            // Record access
            if !memory_ids.is_empty() {
                let _ = self.db.record_access_batch(&memory_ids);
            }

            // Build cluster stats for clusters that contributed results
            let mut clusters_hit: Vec<Value> = Vec::new();
            for (cluster_id, cluster_sim) in cluster_scores.iter().take(input.limit) {
                if let Ok(Some(cluster)) = self.db.get_cluster(cluster_id) {
                    clusters_hit.push(json!({
                        "cluster_id": cluster_id,
                        "summary": cluster.summary,
                        "similarity": cluster_sim,
                        "member_count": cluster.member_count,
                    }));
                }
            }

            Ok(json!({
                "context": input.context,
                "count": memories.len(),
                "memories": memories,
                "retrieval_mode": "hierarchical",
                "clusters_hit": clusters_hit,
            }))
        } else {
            // Flat retrieval (original behavior)
            let embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;

            let mut scored: Vec<(String, f32)> = embeddings
                .iter()
                .map(|(id, vec)| (id.clone(), cosine_similarity(&context_embedding, vec)))
                .filter(|(_, score)| *score >= input.min_score as f32)
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut memories: Vec<Value> = Vec::new();
            let mut memory_ids: Vec<String> = Vec::new();

            let top_scored: Vec<(String, f32)> = scored.into_iter().take(input.limit * 2).collect();

            // Batch fetch all candidate memories
            let flat_ids: Vec<String> = top_scored.iter().map(|(id, _)| id.clone()).collect();
            let flat_map = self.db.get_memories_batch(&flat_ids)?;

            for (id, similarity) in top_scored {
                if let Some(memory) = flat_map.get(&id) {
                    // Apply branch filter
                    let branch_filter = self.branch_mode_to_filter("current");
                    match branch_filter {
                        None => {}
                        Some(None) => {
                            if memory.branch.is_some() {
                                continue;
                            }
                        }
                        Some(Some(branch)) => {
                            if let Some(ref mem_branch) = memory.branch
                                && mem_branch != branch
                            {
                                continue;
                            }
                        }
                    }

                    if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
                        continue;
                    }

                    if memories.len() >= input.limit {
                        break;
                    }

                    memory_ids.push(id);
                    memories.push(json!({
                        "id": memory.id,
                        "type": memory.memory_type.as_str(),
                        "content": memory.content,
                        "summary": memory.summary,
                        "tags": memory.tags,
                        "importance": memory.importance,
                        "relevance_score": memory.relevance_score,
                        "similarity": similarity,
                    }));
                }
            }

            if !memory_ids.is_empty() {
                let _ = self.db.record_access_batch(&memory_ids);
            }

            Ok(json!({
                "context": input.context,
                "count": memories.len(),
                "memories": memories,
                "retrieval_mode": "flat",
            }))
        }
    }

    fn memory_prune(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryPruneInput = serde_json::from_value(arguments)?;

        // Get all memories and filter by relevance threshold
        let all_memories = self.db.get_all_memories_for_project(&self.project_id)?;
        let candidates: Vec<&Memory> = all_memories
            .iter()
            .filter(|m| m.relevance_score < input.threshold)
            .collect();

        if candidates.is_empty() {
            return Ok(json!({
                "success": true,
                "dry_run": !input.confirm,
                "threshold": input.threshold,
                "candidates": 0,
                "deleted": 0,
                "message": format!("No memories below threshold {:.2}", input.threshold),
                "memories": []
            }));
        }

        // Build list of candidates for display
        let candidate_info: Vec<Value> = candidates
            .iter()
            .map(|m| {
                json!({
                    "id": m.id,
                    "type": m.memory_type.as_str(),
                    "relevance_score": m.relevance_score,
                    "importance": m.importance,
                    "summary": m.summary.clone().unwrap_or_else(|| {
                        m.content.chars().take(80).collect::<String>()
                    }),
                    "created_at": m.created_at,
                    "last_accessed_at": m.last_accessed_at,
                })
            })
            .collect();

        let candidate_count = candidates.len();

        if input.confirm {
            // Actually delete
            let ids: Vec<String> = candidates.iter().map(|m| m.id.clone()).collect();
            let deleted = self.db.delete_memories_batch(&ids)?;

            // Invalidate cache since we deleted data
            self.invalidate_search_cache();

            Ok(json!({
                "success": true,
                "dry_run": false,
                "threshold": input.threshold,
                "candidates": candidate_count,
                "deleted": deleted,
                "message": format!("Deleted {} memories below threshold {:.2}", deleted, input.threshold),
                "memories": candidate_info
            }))
        } else {
            // Dry run - just show what would be deleted
            Ok(json!({
                "success": true,
                "dry_run": true,
                "threshold": input.threshold,
                "candidates": candidate_count,
                "deleted": 0,
                "message": format!(
                    "Found {} memories below threshold {:.2}. Set confirm=true to delete.",
                    candidate_count, input.threshold
                ),
                "memories": candidate_info
            }))
        }
    }

    fn memory_promote(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryPromoteInput = serde_json::from_value(arguments)?;

        // Get the memory first to verify it exists and get its current state
        let memory = self
            .db
            .get_memory(&input.id)?
            .ok_or_else(|| MemoryError::NotFound(input.id.clone()))?;

        // Check if already global
        if memory.branch.is_none() {
            return Ok(json!({
                "success": true,
                "id": input.id,
                "message": "Memory is already global",
                "was_branch": null
            }));
        }

        let was_branch = memory.branch.clone();

        // Promote to global
        let promoted = self.db.promote_memory(&input.id)?;

        if promoted {
            // Invalidate search cache since we changed data
            self.invalidate_search_cache();

            Ok(json!({
                "success": true,
                "id": input.id,
                "message": format!("Memory promoted from branch '{}' to global", was_branch.as_deref().unwrap_or("?")),
                "was_branch": was_branch
            }))
        } else {
            Ok(json!({
                "success": false,
                "id": input.id,
                "message": "Failed to promote memory"
            }))
        }
    }

    fn memory_dedup(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDedupInput = serde_json::from_value(arguments)?;
        let threshold = input.threshold.clamp(0.5, 1.0);

        // Get all embeddings and memories for the project
        let all_embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;

        // Pre-fetch all memories upfront to avoid O(n) individual get_memory calls
        let all_memories_list = self.db.get_all_memories_for_project(&self.project_id)?;
        let all_memories: std::collections::HashMap<String, Memory> = all_memories_list
            .into_iter()
            .map(|m| (m.id.clone(), m))
            .collect();

        // Build duplicate groups: for each pair with similarity >= threshold and same type
        let mut processed: HashSet<String> = HashSet::new();
        let mut groups: Vec<Vec<(String, f32)>> = Vec::new(); // groups of (id, similarity_to_first)

        for i in 0..all_embeddings.len() {
            let (ref id_i, ref vec_i) = all_embeddings[i];
            if processed.contains(id_i) {
                continue;
            }

            let mem_i = match all_memories.get(id_i) {
                Some(m) => m,
                None => continue,
            };

            let mut group = vec![(id_i.clone(), 1.0_f32)];

            for (id_j, vec_j) in all_embeddings.iter().skip(i + 1) {
                if processed.contains(id_j) {
                    continue;
                }

                let similarity = cosine_similarity(vec_i, vec_j);
                if similarity >= threshold
                    && let Some(mem_j) = all_memories.get(id_j)
                    && mem_j.memory_type == mem_i.memory_type
                {
                    group.push((id_j.clone(), similarity));
                }
            }

            if group.len() > 1 {
                for (id, _) in &group {
                    processed.insert(id.clone());
                }
                groups.push(group);
            }
        }

        if groups.is_empty() {
            return Ok(json!({
                "success": true,
                "dry_run": !input.confirm,
                "threshold": threshold,
                "duplicate_groups": 0,
                "total_duplicates": 0,
                "merged": 0,
                "message": format!("No duplicates found at threshold {:.2}", threshold),
                "groups": []
            }));
        }

        // Build group info for display
        let mut group_info: Vec<Value> = Vec::new();
        let mut total_duplicates = 0usize;

        for group in &groups {
            let mut members: Vec<Value> = Vec::new();
            for (id, sim) in group {
                if let Some(mem) = all_memories.get(id) {
                    members.push(json!({
                        "id": id,
                        "type": mem.memory_type.as_str(),
                        "similarity": sim,
                        "content_preview": mem.content.chars().take(100).collect::<String>(),
                        "updated_at": mem.updated_at,
                    }));
                }
            }
            total_duplicates += members.len() - 1; // -1 because one is kept
            group_info.push(json!({"members": members}));
        }

        if input.confirm {
            let mut merged_count = 0usize;

            for group in &groups {
                // Keep the most recently updated memory, merge others into it
                let with_time: Vec<(String, f32, i64)> = group
                    .iter()
                    .filter_map(|(id, sim)| {
                        all_memories.get(id).map(|m| (id.clone(), *sim, m.updated_at))
                    })
                    .collect();

                let mut sorted = with_time;
                sorted.sort_by(|a, b| b.2.cmp(&a.2)); // newest first

                if sorted.len() < 2 {
                    continue;
                }

                let keeper_id = sorted[0].0.clone();
                for (old_id, _, _) in &sorted[1..] {
                    let old_preview: String = all_memories.get(old_id)
                        .map(|m| m.content.chars().take(100).collect())
                        .unwrap_or_default();
                    self.db.merge_memories(&keeper_id, old_id, &old_preview)?;
                    merged_count += 1;
                }
            }

            self.invalidate_search_cache();

            Ok(json!({
                "success": true,
                "dry_run": false,
                "threshold": threshold,
                "duplicate_groups": groups.len(),
                "total_duplicates": total_duplicates,
                "merged": merged_count,
                "message": format!("Merged {} duplicate memories from {} groups", merged_count, groups.len()),
                "groups": group_info
            }))
        } else {
            Ok(json!({
                "success": true,
                "dry_run": true,
                "threshold": threshold,
                "duplicate_groups": groups.len(),
                "total_duplicates": total_duplicates,
                "merged": 0,
                "message": format!("Found {} duplicate groups ({} duplicates). Set confirm=true to merge.", groups.len(), total_duplicates),
                "groups": group_info
            }))
        }
    }

    /// Generate a cluster summary from member memories.
    /// Uses the first sentence of the highest-importance member + top keywords across all members.
    fn generate_cluster_summary(&self, member_ids: &[String]) -> Result<String, MemoryError> {
        if member_ids.is_empty() {
            return Ok("Empty cluster".to_string());
        }

        let members = self.db.get_memories_batch(member_ids)?;
        if members.is_empty() {
            return Ok("Empty cluster".to_string());
        }

        // Find highest-importance member
        let best_member = members
            .values()
            .max_by(|a, b| {
                a.importance
                    .partial_cmp(&b.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Get first sentence from best member
        let first_sentence = crate::summarize::extract_first_sentence(&best_member.content);

        // Collect keywords from all members
        let all_content: String = members
            .values()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let keywords = crate::summarize::extract_keywords(&all_content, 3);

        if keywords.is_empty() {
            Ok(first_sentence)
        } else {
            Ok(format!("{} [{}]", first_sentence, keywords.join(", ")))
        }
    }

    /// Assign a memory to the best matching cluster, or create a new one.
    fn assign_to_cluster(
        &self,
        memory_id: &str,
        embedding: &[f32],
        content: &str,
        _importance: f64,
    ) -> Result<Option<String>, MemoryError> {
        use crate::memory::MemoryCluster;

        let clusters = self.db.get_clusters_for_project(&self.project_id)?;

        // Find best matching cluster by centroid similarity
        const CLUSTER_THRESHOLD: f32 = 0.75;
        let mut best_match: Option<(String, f32)> = None;

        for cluster in &clusters {
            if let Some(ref centroid) = cluster.centroid {
                let similarity = cosine_similarity(embedding, centroid);
                if similarity >= CLUSTER_THRESHOLD
                    && (best_match.is_none() || similarity > best_match.as_ref().unwrap().1)
                {
                    best_match = Some((cluster.id.clone(), similarity));
                }
            }
        }

        let now = chrono::Utc::now().timestamp();

        if let Some((cluster_id, _)) = best_match {
            // Add to existing cluster
            self.db.add_to_cluster(&cluster_id, memory_id)?;

            // Update centroid (running average)
            let member_ids = self.db.get_cluster_member_ids(&cluster_id)?;
            let new_centroid = self.compute_cluster_centroid(&member_ids)?;
            let summary = self.generate_cluster_summary(&member_ids)?;

            if let Some(centroid) = new_centroid {
                self.db
                    .update_cluster_centroid(&cluster_id, &centroid, &summary)?;
            }

            Ok(Some(cluster_id))
        } else {
            // Create new cluster
            let cluster_id = format!("clust_{}", uuid::Uuid::new_v4().simple());
            let summary = crate::summarize::extract_first_sentence(content);

            let cluster = MemoryCluster {
                id: cluster_id.clone(),
                project_id: self.project_id.clone(),
                summary,
                member_count: 1,
                centroid: Some(embedding.to_vec()),
                created_at: now,
                updated_at: now,
            };

            self.db.create_cluster(&cluster)?;
            self.db.add_to_cluster(&cluster_id, memory_id)?;

            Ok(Some(cluster_id))
        }
    }

    /// Compute the centroid (average embedding) for a set of memory IDs.
    fn compute_cluster_centroid(
        &self,
        member_ids: &[String],
    ) -> Result<Option<Vec<f32>>, MemoryError> {
        if member_ids.is_empty() {
            return Ok(None);
        }

        let all_embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;
        let member_set: HashSet<&String> = member_ids.iter().collect();

        let mut sum: Option<Vec<f32>> = None;
        let mut count = 0usize;

        for (id, vec) in &all_embeddings {
            if member_set.contains(id) {
                count += 1;
                match &mut sum {
                    None => sum = Some(vec.clone()),
                    Some(s) => {
                        for (i, v) in vec.iter().enumerate() {
                            if i < s.len() {
                                s[i] += v;
                            }
                        }
                    }
                }
            }
        }

        Ok(sum.map(|mut s| {
            let c = count as f32;
            for v in &mut s {
                *v /= c;
            }
            s
        }))
    }
}
