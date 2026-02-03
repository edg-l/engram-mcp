//! MCP tool handlers for the Engram memory server.
//!
//! All items in this module are used by the MCP server binary (main.rs).
//! The dead_code warnings appear because the CLI binary doesn't use these.
#![allow(dead_code)]

use rmcp::model::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::Arc;

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
}

fn default_limit() -> usize {
    10
}

fn default_min_relevance() -> f64 {
    0.3
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

#[derive(Debug, Serialize)]
pub struct MemoryStoreResult {
    pub id: String,
    pub message: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub potential_contradictions: Vec<PotentialContradiction>,
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
            "Store a new memory. Use this to save facts, decisions, preferences, patterns, debug info, or entities about the project.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content of the memory to store"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["fact", "decision", "preference", "pattern", "debug", "entity"],
                        "description": "The type of memory"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorizing the memory"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Importance score (0.0-1.0), affects decay rate"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short summary for listings"
                    },
                    "related_to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of related memories to link to"
                    }
                },
                "required": ["content", "type"]
            })),
        ),
        Tool::new(
            "memory_query",
            "Search for relevant memories using semantic search. Returns memories ranked by similarity and relevance.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of results to return"
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of results to skip (for pagination)"
                    },
                    "min_relevance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Minimum relevance score threshold"
                    },
                    "types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by memory types"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags"
                    }
                },
                "required": ["query"]
            })),
        ),
        Tool::new(
            "memory_update",
            "Update an existing memory's content, importance, tags, or summary.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The memory ID to update"
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for the memory"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "New importance score"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags (replaces existing)"
                    },
                    "summary": {
                        "type": "string",
                        "description": "New summary"
                    }
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_delete",
            "Delete a memory and its relationships.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The memory ID to delete"
                    }
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_link",
            "Create a relationship between two memories.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Source memory ID"
                    },
                    "target_id": {
                        "type": "string",
                        "description": "Target memory ID"
                    },
                    "relation": {
                        "type": "string",
                        "enum": ["relates_to", "supersedes", "derived_from", "contradicts"],
                        "description": "Type of relationship"
                    },
                    "strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Relationship strength"
                    }
                },
                "required": ["source_id", "target_id", "relation"]
            })),
        ),
        Tool::new(
            "memory_graph",
            "Retrieve a memory with its related memories (graph traversal).",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The root memory ID"
                    },
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Maximum traversal depth"
                    },
                    "relation_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by relationship types"
                    }
                },
                "required": ["id"]
            })),
        ),
        Tool::new(
            "memory_store_batch",
            "Store multiple memories atomically in a single transaction. Maximum 100 memories per batch.",
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
                        "maxItems": 100,
                        "description": "Array of memories to store"
                    }
                },
                "required": ["memories"]
            })),
        ),
        Tool::new(
            "memory_delete_batch",
            "Delete multiple memories by ID in a single transaction.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of memory IDs to delete"
                    }
                },
                "required": ["ids"]
            })),
        ),
        Tool::new(
            "memory_export",
            "Export all project memories to JSON format.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "include_embeddings": {
                        "type": "boolean",
                        "description": "Include embeddings in export (increases size)"
                    }
                }
            })),
        ),
        Tool::new(
            "memory_import",
            "Import memories from JSON export format.",
            make_input_schema(json!({
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "The export data object to import"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["merge", "replace"],
                        "description": "Import mode: merge (skip duplicates) or replace (clear existing)"
                    }
                },
                "required": ["data"]
            })),
        ),
        Tool::new(
            "memory_stats",
            "Get statistics about the project's memory store.",
            make_input_schema(json!({
                "type": "object",
                "properties": {}
            })),
        ),
    ]
}

pub struct ToolHandler {
    db: Database,
    embedding: EmbeddingService,
    project_id: String,
}

impl ToolHandler {
    pub fn new(db: Database, embedding: EmbeddingService, project_id: String) -> Self {
        Self {
            db,
            embedding,
            project_id,
        }
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
        };

        // Generate embedding first to check for potential contradictions
        let embedding = self.embedding.embed_memory(memory_type, &input.content)?;

        // Check for potential contradictions (high similarity with existing memories)
        let existing_embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;
        let mut potential_contradictions: Vec<PotentialContradiction> = Vec::new();

        // Threshold for flagging potential contradictions (very high similarity)
        const CONTRADICTION_THRESHOLD: f32 = 0.85;

        for (existing_id, existing_vec) in &existing_embeddings {
            let similarity = cosine_similarity(&embedding, existing_vec);
            if similarity >= CONTRADICTION_THRESHOLD {
                // High similarity - might be contradicting or duplicating
                if let Ok(Some(existing_memory)) = self.db.get_memory(existing_id) {
                    potential_contradictions.push(PotentialContradiction {
                        memory_id: existing_id.clone(),
                        summary: existing_memory
                            .summary
                            .unwrap_or_else(|| existing_memory.content.chars().take(100).collect()),
                        similarity: similarity as f64,
                    });
                }
            }
        }

        self.db.store_memory(&memory)?;

        self.db
            .store_embedding(&id, &embedding, self.embedding.model_version())?;

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

        let message = if potential_contradictions.is_empty() {
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
            potential_contradictions,
        }))
    }

    fn memory_query(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryQueryInput = serde_json::from_value(arguments)?;

        // Parse type filters
        let type_filters: Vec<MemoryType> =
            input.types.iter().filter_map(|t| t.parse().ok()).collect();

        // Optimization: if query is empty, skip embedding search and use filter-only path
        let scored_ids: Vec<(String, f32)> = if input.query.trim().is_empty() {
            // Filter-only path: get memories directly from DB
            let memories = self.db.query_memories(
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
            )?;
            memories
                .into_iter()
                .map(|m| (m.id, m.relevance_score as f32))
                .collect()
        } else {
            // Semantic search path
            let query_embedding = self.embedding.embed(&input.query)?;
            let embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;

            let mut scored: Vec<(String, f32)> = embeddings
                .iter()
                .map(|(id, vec)| {
                    let similarity = cosine_similarity(&query_embedding, vec);
                    (id.clone(), similarity)
                })
                .collect();

            // Sort by similarity descending
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored
        };

        // Fetch memories and calculate final scores
        let mut results: Vec<MemoryWithScore> = Vec::new();
        let mut result_ids: Vec<String> = Vec::new();
        let mut skipped = 0;

        for (id, similarity) in scored_ids {
            if results.len() >= input.limit {
                break;
            }

            if let Some(mut memory) = self.db.get_memory(&id)? {
                // Filter by types (only for semantic path, DB path already filtered)
                if !input.query.trim().is_empty() {
                    if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
                        continue;
                    }

                    // Filter by tags
                    if !input.tags.is_empty() && !input.tags.iter().any(|t| memory.tags.contains(t))
                    {
                        continue;
                    }
                }

                // Calculate final score: similarity * relevance
                let score = (similarity as f64) * memory.relevance_score;

                if score < input.min_relevance {
                    continue;
                }

                // Apply offset (pagination)
                if skipped < input.offset {
                    skipped += 1;
                    continue;
                }

                // Record access (reinforcement)
                self.db.record_access(&id)?;
                memory.access_count += 1;

                result_ids.push(id);
                results.push(MemoryWithScore { memory, score });
            }
        }

        // Check for contradiction relationships among returned memories
        let mut contradiction_warnings: Vec<ContradictionWarning> = Vec::new();
        for id in &result_ids {
            // Check outgoing contradicts relationships
            let Ok(rels) = self.db.get_relationships_from(id) else {
                continue;
            };
            for rel in rels {
                if rel.relation_type != RelationType::Contradicts
                    || !result_ids.contains(&rel.target_id)
                {
                    continue;
                }
                if let Ok(Some(target)) = self.db.get_memory(&rel.target_id) {
                    contradiction_warnings.push(ContradictionWarning {
                        memory_id: id.clone(),
                        contradicts_id: rel.target_id.clone(),
                        content_preview: target.content.chars().take(100).collect(),
                    });
                }
            }
        }

        Ok(json!(MemoryQueryResult {
            count: results.len(),
            memories: results,
            contradiction_warnings,
        }))
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

        Ok(json!({"success": true, "message": "Memory updated successfully"}))
    }

    fn memory_delete(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDeleteInput = serde_json::from_value(arguments)?;

        let deleted = self.db.delete_memory(&input.id)?;

        if deleted {
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

        let mut related: Vec<RelatedMemory> = Vec::new();
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(input.id.clone());

        self.traverse_graph(
            &input.id,
            1,
            input.depth,
            &relation_filters,
            &mut visited,
            &mut related,
        )?;

        // Record access to root memory
        self.db.record_access(&input.id)?;

        Ok(json!(MemoryGraphResult { root, related }))
    }

    fn traverse_graph(
        &self,
        memory_id: &str,
        current_depth: usize,
        max_depth: usize,
        relation_filters: &[RelationType],
        visited: &mut std::collections::HashSet<String>,
        results: &mut Vec<RelatedMemory>,
    ) -> Result<(), MemoryError> {
        if current_depth > max_depth {
            return Ok(());
        }

        // Get outgoing relationships
        let outgoing = self.db.get_relationships_from(memory_id)?;
        for rel in outgoing {
            if visited.contains(&rel.target_id) {
                continue;
            }
            if !relation_filters.is_empty() && !relation_filters.contains(&rel.relation_type) {
                continue;
            }

            if let Some(memory) = self.db.get_memory(&rel.target_id)? {
                visited.insert(rel.target_id.clone());
                results.push(RelatedMemory {
                    memory: memory.clone(),
                    relation: rel.relation_type.as_str().to_string(),
                    direction: "outgoing".to_string(),
                    depth: current_depth,
                });

                self.traverse_graph(
                    &rel.target_id,
                    current_depth + 1,
                    max_depth,
                    relation_filters,
                    visited,
                    results,
                )?;
            }
        }

        // Get incoming relationships
        let incoming = self.db.get_relationships_to(memory_id)?;
        for rel in incoming {
            if visited.contains(&rel.source_id) {
                continue;
            }
            if !relation_filters.is_empty() && !relation_filters.contains(&rel.relation_type) {
                continue;
            }

            if let Some(memory) = self.db.get_memory(&rel.source_id)? {
                visited.insert(rel.source_id.clone());
                results.push(RelatedMemory {
                    memory: memory.clone(),
                    relation: rel.relation_type.as_str().to_string(),
                    direction: "incoming".to_string(),
                    depth: current_depth,
                });

                self.traverse_graph(
                    &rel.source_id,
                    current_depth + 1,
                    max_depth,
                    relation_filters,
                    visited,
                    results,
                )?;
            }
        }

        Ok(())
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

        Ok(json!({
            "success": true,
            "count": stored,
            "ids": ids,
            "message": format!("{} memories stored successfully", stored)
        }))
    }

    fn memory_delete_batch(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDeleteBatchInput = serde_json::from_value(arguments)?;

        let deleted = self.db.delete_memories_batch(&input.ids)?;

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

        let export_data =
            export::create_export(&self.project_id, memories, relationships, embeddings);

        Ok(json!(export_data))
    }

    fn memory_import(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryImportInput = serde_json::from_value(arguments)?;

        let export_data: ExportData = serde_json::from_value(input.data)?;

        // Validate version
        export::validate_import(&export_data).map_err(MemoryError::Embedding)?;

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

        Ok(json!({
            "success": true,
            "stats": stats,
            "message": format!(
                "Imported {} memories, {} relationships ({} skipped)",
                stats.memories_imported,
                stats.relationships_imported,
                stats.memories_skipped + stats.relationships_skipped
            )
        }))
    }

    fn memory_stats(&self, _arguments: Value) -> Result<Value, MemoryError> {
        let stats: ProjectStats = self.db.get_project_stats(&self.project_id)?;

        Ok(json!({
            "project_id": self.project_id,
            "memory_count": stats.memory_count,
            "relationship_count": stats.relationship_count,
            "avg_relevance": stats.avg_relevance
        }))
    }
}
