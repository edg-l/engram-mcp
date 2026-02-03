use rmcp::model::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::sync::Arc;

use crate::db::Database;
use crate::embedding::{cosine_similarity, EmbeddingService};
use crate::error::MemoryError;
use crate::memory::{Memory, MemoryType, MemoryWithScore, Relationship, RelationType};

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

#[derive(Debug, Serialize)]
pub struct MemoryStoreResult {
    pub id: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct MemoryQueryResult {
    pub memories: Vec<MemoryWithScore>,
    pub count: usize,
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
            _ => Ok(json!({"error": format!("Unknown tool: {}", name)})),
        }
    }

    fn memory_store(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryStoreInput = serde_json::from_value(arguments)?;

        let memory_type = MemoryType::from_str(&input.memory_type)
            .ok_or_else(|| MemoryError::InvalidType(input.memory_type.clone()))?;

        let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
        let now = chrono::Utc::now().timestamp();

        let memory = Memory {
            id: id.clone(),
            project_id: self.project_id.clone(),
            memory_type,
            content: input.content.clone(),
            summary: input.summary,
            tags: input.tags,
            importance: input.importance.clamp(0.0, 1.0),
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        self.db.store_memory(&memory)?;

        // Generate and store embedding
        let embedding = self.embedding.embed_memory(memory_type, &input.content)?;
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

        Ok(json!(MemoryStoreResult {
            id,
            message: "Memory stored successfully".to_string(),
        }))
    }

    fn memory_query(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryQueryInput = serde_json::from_value(arguments)?;

        // Generate query embedding
        let query_embedding = self.embedding.embed(&input.query)?;

        // Get all embeddings for the project
        let embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;

        // Calculate similarities and get memory IDs sorted by score
        let mut scored_ids: Vec<(String, f32)> = embeddings
            .iter()
            .map(|(id, vec)| {
                let similarity = cosine_similarity(&query_embedding, vec);
                (id.clone(), similarity)
            })
            .collect();

        // Sort by similarity descending
        scored_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Parse type filters
        let type_filters: Vec<MemoryType> = input
            .types
            .iter()
            .filter_map(|t| MemoryType::from_str(t))
            .collect();

        // Fetch memories and calculate final scores
        let mut results: Vec<MemoryWithScore> = Vec::new();
        for (id, similarity) in scored_ids {
            if results.len() >= input.limit {
                break;
            }

            if let Some(mut memory) = self.db.get_memory(&id)? {
                // Filter by types
                if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
                    continue;
                }

                // Filter by tags
                if !input.tags.is_empty() && !input.tags.iter().any(|t| memory.tags.contains(t)) {
                    continue;
                }

                // Calculate final score: similarity * relevance
                let score = (similarity as f64) * memory.relevance_score;

                if score < input.min_relevance {
                    continue;
                }

                // Record access (reinforcement)
                self.db.record_access(&id)?;
                memory.access_count += 1;

                results.push(MemoryWithScore { memory, score });
            }
        }

        Ok(json!(MemoryQueryResult {
            count: results.len(),
            memories: results,
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

        let relation_type = RelationType::from_str(&input.relation)
            .ok_or_else(|| MemoryError::InvalidRelation(input.relation.clone()))?;

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
            .filter_map(|r| RelationType::from_str(r))
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
}
