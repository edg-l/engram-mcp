use rusqlite::{Connection, params};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::error::MemoryError;
use crate::memory::{Memory, MemoryType, Project, ProjectStats, RelationType, Relationship};

const SCHEMA: &str = r#"
-- Core memory storage
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    tags TEXT,
    importance REAL DEFAULT 0.5,
    relevance_score REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_accessed_at INTEGER NOT NULL
);

-- Vector embeddings (stored separately for efficiency)
CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    model_version TEXT NOT NULL
);

-- Relationship graph
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    created_at INTEGER NOT NULL,
    UNIQUE(source_id, target_id, relation_type)
);

-- Project configuration
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    root_path TEXT,
    decay_rate REAL DEFAULT 0.01,
    created_at INTEGER NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_relevance ON memories(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_project_type ON memories(project_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);

-- Compound indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_memories_project_relevance
    ON memories(project_id, relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_relationships_source_type
    ON relationships(source_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_relationships_target_type
    ON relationships(target_id, relation_type);

-- FTS5 full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    summary,
    tags,
    content=memories,
    content_rowid=rowid
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary, tags)
    VALUES (NEW.rowid, NEW.content, NEW.summary, NEW.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags)
    VALUES ('delete', OLD.rowid, OLD.content, OLD.summary, OLD.tags);
    INSERT INTO memories_fts(rowid, content, summary, tags)
    VALUES (NEW.rowid, NEW.content, NEW.summary, NEW.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags)
    VALUES ('delete', OLD.rowid, OLD.content, OLD.summary, OLD.tags);
END;
"#;

#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, MemoryError> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.initialize()?;
        Ok(db)
    }

    #[allow(dead_code)] // Used in tests
    pub fn open_in_memory() -> Result<Self, MemoryError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.initialize()?;
        Ok(db)
    }

    fn initialize(&self) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(SCHEMA)?;
        drop(conn);
        self.migrate_fts()?;
        Ok(())
    }

    /// Migrate existing memories to FTS index if empty.
    fn migrate_fts(&self) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();

        // Check if FTS table is empty
        let fts_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))?;

        if fts_count == 0 {
            // Check if there are memories to migrate
            let memory_count: i64 =
                conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

            if memory_count > 0 {
                // Populate FTS from existing memories
                conn.execute(
                    "INSERT INTO memories_fts(rowid, content, summary, tags)
                     SELECT rowid, content, summary, tags FROM memories",
                    [],
                )?;
            }
        }

        Ok(())
    }

    // Project operations
    pub fn create_project(&self, project: &Project) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO projects (id, name, root_path, decay_rate, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![project.id, project.name, project.root_path, project.decay_rate, project.created_at],
        )?;
        Ok(())
    }

    pub fn get_project(&self, id: &str) -> Result<Option<Project>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, root_path, decay_rate, created_at FROM projects WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            Ok(Some(Project {
                id: row.get(0)?,
                name: row.get(1)?,
                root_path: row.get(2)?,
                decay_rate: row.get(3)?,
                created_at: row.get(4)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn get_or_create_project(&self, id: &str, name: &str) -> Result<Project, MemoryError> {
        if let Some(project) = self.get_project(id)? {
            return Ok(project);
        }

        let project = Project {
            id: id.to_string(),
            name: name.to_string(),
            root_path: Some(id.to_string()),
            decay_rate: 0.01,
            created_at: chrono::Utc::now().timestamp(),
        };
        self.create_project(&project)?;
        Ok(project)
    }

    // Memory operations
    pub fn store_memory(&self, memory: &Memory) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let tags_json = serde_json::to_string(&memory.tags)?;
        conn.execute(
            "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                memory.id,
                memory.project_id,
                memory.memory_type.as_str(),
                memory.content,
                memory.summary,
                tags_json,
                memory.importance,
                memory.relevance_score,
                memory.access_count,
                memory.created_at,
                memory.updated_at,
                memory.last_accessed_at,
            ],
        )?;
        Ok(())
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<Memory>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at
             FROM memories WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            let memory_type_str: String = row.get(2)?;
            let tags_json: String = row.get(5)?;
            Ok(Some(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                memory_type: memory_type_str.parse().unwrap_or(MemoryType::Fact),
                content: row.get(3)?,
                summary: row.get(4)?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                importance: row.get(6)?,
                relevance_score: row.get(7)?,
                access_count: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
                last_accessed_at: row.get(11)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_memory(&self, memory: &Memory) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let tags_json = serde_json::to_string(&memory.tags)?;
        conn.execute(
            "UPDATE memories SET content = ?1, summary = ?2, tags = ?3, importance = ?4, relevance_score = ?5, access_count = ?6, updated_at = ?7, last_accessed_at = ?8
             WHERE id = ?9",
            params![
                memory.content,
                memory.summary,
                tags_json,
                memory.importance,
                memory.relevance_score,
                memory.access_count,
                memory.updated_at,
                memory.last_accessed_at,
                memory.id,
            ],
        )?;
        Ok(())
    }

    pub fn delete_memory(&self, id: &str) -> Result<bool, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let rows_affected = conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(rows_affected > 0)
    }

    pub fn query_memories(
        &self,
        project_id: &str,
        types: Option<&[MemoryType]>,
        tags: Option<&[String]>,
        min_relevance: Option<f64>,
        limit: usize,
    ) -> Result<Vec<Memory>, MemoryError> {
        let conn = self.conn.lock().unwrap();

        let mut sql = String::from(
            "SELECT id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at
             FROM memories WHERE project_id = ?1"
        );

        if min_relevance.is_some() {
            sql.push_str(" AND relevance_score >= ?2");
        }

        sql.push_str(" ORDER BY relevance_score DESC LIMIT ?3");

        let mut stmt = conn.prepare(&sql)?;

        let min_rel = min_relevance.unwrap_or(0.0);
        let rows = stmt.query_map(params![project_id, min_rel, limit as i64], |row| {
            let memory_type_str: String = row.get(2)?;
            let tags_json: String = row.get(5)?;
            Ok(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                memory_type: memory_type_str.parse().unwrap_or(MemoryType::Fact),
                content: row.get(3)?,
                summary: row.get(4)?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                importance: row.get(6)?,
                relevance_score: row.get(7)?,
                access_count: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
                last_accessed_at: row.get(11)?,
            })
        })?;

        let mut memories: Vec<Memory> = rows.filter_map(|r| r.ok()).collect();

        // Filter by types if specified
        if let Some(types) = types {
            memories.retain(|m| types.contains(&m.memory_type));
        }

        // Filter by tags if specified
        if let Some(filter_tags) = tags {
            memories.retain(|m| filter_tags.iter().any(|t| m.tags.contains(t)));
        }

        Ok(memories)
    }

    pub fn get_all_memories_for_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<Memory>, MemoryError> {
        self.query_memories(project_id, None, None, None, 10000)
    }

    /// Store multiple memories in a single transaction
    #[allow(dead_code)] // Used by MCP server tools
    pub fn store_memories_batch(&self, memories: &[Memory]) -> Result<usize, MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        let mut count = 0;
        for memory in memories {
            let tags_json = serde_json::to_string(&memory.tags)?;
            tx.execute(
                "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    memory.id,
                    memory.project_id,
                    memory.memory_type.as_str(),
                    memory.content,
                    memory.summary,
                    tags_json,
                    memory.importance,
                    memory.relevance_score,
                    memory.access_count,
                    memory.created_at,
                    memory.updated_at,
                    memory.last_accessed_at,
                ],
            )?;
            count += 1;
        }

        tx.commit()?;
        Ok(count)
    }

    /// Store multiple embeddings in a single transaction
    #[allow(dead_code)] // Used by MCP server tools
    pub fn store_embeddings_batch(
        &self,
        embeddings: &[(String, Vec<f32>, String)],
    ) -> Result<usize, MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        let mut count = 0;
        for (memory_id, vector, model_version) in embeddings {
            let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
            tx.execute(
                "INSERT OR REPLACE INTO embeddings (memory_id, vector, model_version) VALUES (?1, ?2, ?3)",
                params![memory_id, vector_bytes, model_version],
            )?;
            count += 1;
        }

        tx.commit()?;
        Ok(count)
    }

    /// Delete multiple memories by ID in a single transaction
    pub fn delete_memories_batch(&self, ids: &[String]) -> Result<usize, MemoryError> {
        if ids.is_empty() {
            return Ok(0);
        }

        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        let mut total_deleted = 0;
        for id in ids {
            let rows = tx.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
            total_deleted += rows;
        }

        tx.commit()?;
        Ok(total_deleted)
    }

    /// Delete all memories and relationships for a project
    pub fn delete_project_data(&self, project_id: &str) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let rows = conn.execute(
            "DELETE FROM memories WHERE project_id = ?1",
            params![project_id],
        )?;
        Ok(rows)
    }

    /// Get all relationships for a project
    pub fn get_all_relationships_for_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<Relationship>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT r.id, r.source_id, r.target_id, r.relation_type, r.strength, r.created_at
             FROM relationships r
             JOIN memories m ON r.source_id = m.id
             WHERE m.project_id = ?1",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Get memory count and stats for a project
    pub fn get_project_stats(&self, project_id: &str) -> Result<ProjectStats, MemoryError> {
        let conn = self.conn.lock().unwrap();

        let memory_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?1",
            params![project_id],
            |row| row.get(0),
        )?;

        let relationship_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM relationships r JOIN memories m ON r.source_id = m.id WHERE m.project_id = ?1",
            params![project_id],
            |row| row.get(0),
        )?;

        let avg_relevance: f64 = conn.query_row(
            "SELECT COALESCE(AVG(relevance_score), 0.0) FROM memories WHERE project_id = ?1",
            params![project_id],
            |row| row.get(0),
        )?;

        Ok(ProjectStats {
            memory_count: memory_count as usize,
            relationship_count: relationship_count as usize,
            avg_relevance,
        })
    }

    pub fn record_access(&self, id: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ?1, relevance_score = MIN(1.0, relevance_score + 0.1) WHERE id = ?2",
            params![now, id],
        )?;
        Ok(())
    }

    // Embedding operations
    pub fn store_embedding(
        &self,
        memory_id: &str,
        vector: &[f32],
        model_version: &str,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (memory_id, vector, model_version) VALUES (?1, ?2, ?3)",
            params![memory_id, vector_bytes, model_version],
        )?;
        Ok(())
    }

    #[allow(dead_code)] // Available for export functionality
    pub fn get_embedding(&self, memory_id: &str) -> Result<Option<Vec<f32>>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT vector FROM embeddings WHERE memory_id = ?1")?;
        let mut rows = stmt.query(params![memory_id])?;

        if let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            let vector: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(Some(vector))
        } else {
            Ok(None)
        }
    }

    pub fn get_all_embeddings_for_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<(String, Vec<f32>)>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT e.memory_id, e.vector FROM embeddings e
             JOIN memories m ON e.memory_id = m.id
             WHERE m.project_id = ?1",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            let memory_id: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            let vector: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok((memory_id, vector))
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Full-text search using FTS5 with BM25 scoring.
    /// Returns (memory_id, bm25_score) pairs sorted by relevance.
    /// The BM25 score is negated (SQLite returns negative values, we flip to positive).
    pub fn keyword_search(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, f64)>, MemoryError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().unwrap();

        // Escape special FTS5 characters and build match expression
        let escaped_query = Self::escape_fts_query(query);
        if escaped_query.is_empty() {
            return Ok(Vec::new());
        }

        // FTS5 bm25() returns negative scores (more negative = more relevant)
        // We negate to get positive scores where higher = more relevant
        let mut stmt = conn.prepare(
            "SELECT m.id, -bm25(memories_fts) as score
             FROM memories_fts
             JOIN memories m ON memories_fts.rowid = m.rowid
             WHERE memories_fts MATCH ?1 AND m.project_id = ?2
             ORDER BY score DESC
             LIMIT ?3",
        )?;

        let rows = stmt.query_map(params![escaped_query, project_id, limit as i64], |row| {
            let id: String = row.get(0)?;
            let score: f64 = row.get(1)?;
            Ok((id, score))
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Escape special FTS5 query characters for safe searching.
    fn escape_fts_query(query: &str) -> String {
        // Split into words and wrap each in quotes for exact matching
        // This prevents FTS5 syntax errors from special characters
        query
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .map(|word| {
                // Remove any existing quotes and special chars that could break FTS
                let cleaned: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                    .collect();
                if cleaned.is_empty() {
                    String::new()
                } else {
                    format!("\"{}\"", cleaned)
                }
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" OR ")
    }

    // Relationship operations
    pub fn create_relationship(&self, rel: &Relationship) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO relationships (id, source_id, target_id, relation_type, strength, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                rel.id,
                rel.source_id,
                rel.target_id,
                rel.relation_type.as_str(),
                rel.strength,
                rel.created_at,
            ],
        )?;
        Ok(())
    }

    #[allow(dead_code)] // Used by CLI binary
    pub fn get_relationships_from(
        &self,
        source_id: &str,
    ) -> Result<Vec<Relationship>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, relation_type, strength, created_at FROM relationships WHERE source_id = ?1"
        )?;
        let rows = stmt.query_map(params![source_id], |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    #[allow(dead_code)] // Used by CLI binary
    pub fn get_relationships_to(&self, target_id: &str) -> Result<Vec<Relationship>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, relation_type, strength, created_at FROM relationships WHERE target_id = ?1"
        )?;
        let rows = stmt.query_map(params![target_id], |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    // Decay operations
    pub fn update_relevance_scores(
        &self,
        project_id: &str,
        decay_rate: f64,
    ) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().timestamp();

        // Update relevance scores based on decay algorithm
        // relevance = max(0.1, base_decay * importance_factor + usage_boost)
        let rows_affected = conn.execute(
            r#"
            UPDATE memories
            SET relevance_score = MAX(0.1,
                EXP(-?1 * ((?2 - last_accessed_at) / 86400.0)) * (0.5 + importance * 0.5)
                + LN(1 + access_count) * 0.1
            )
            WHERE project_id = ?3
            "#,
            params![decay_rate, now, project_id],
        )?;

        Ok(rows_affected)
    }

    // ============================================
    // Batch operations for performance optimization
    // ============================================

    /// Fetch multiple memories by ID in a single query.
    /// Returns a HashMap for O(1) lookups.
    pub fn get_memories_batch(
        &self,
        ids: &[String],
    ) -> Result<HashMap<String, Memory>, MemoryError> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let conn = self.conn.lock().unwrap();

        // Build query with placeholders
        let placeholders: Vec<&str> = ids.iter().map(|_| "?").collect();
        let sql = format!(
            "SELECT id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at
             FROM memories WHERE id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = conn.prepare(&sql)?;

        // Convert ids to params
        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

        let rows = stmt.query_map(params.as_slice(), |row| {
            let memory_type_str: String = row.get(2)?;
            let tags_json: String = row.get(5)?;
            Ok(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                memory_type: memory_type_str.parse().unwrap_or(MemoryType::Fact),
                content: row.get(3)?,
                summary: row.get(4)?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                importance: row.get(6)?,
                relevance_score: row.get(7)?,
                access_count: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
                last_accessed_at: row.get(11)?,
            })
        })?;

        let mut result = HashMap::new();
        for memory in rows.flatten() {
            result.insert(memory.id.clone(), memory);
        }

        Ok(result)
    }

    /// Record access for multiple memories in a single transaction.
    pub fn record_access_batch(&self, ids: &[String]) -> Result<usize, MemoryError> {
        if ids.is_empty() {
            return Ok(0);
        }

        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().timestamp();

        let mut count = 0;
        for id in ids {
            let rows = tx.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ?1, relevance_score = MIN(1.0, relevance_score + 0.1) WHERE id = ?2",
                params![now, id],
            )?;
            count += rows;
        }

        tx.commit()?;
        Ok(count)
    }

    /// Check which memory IDs exist in the database.
    /// Returns a HashSet of existing IDs.
    #[allow(dead_code)] // Available for import optimization
    pub fn memories_exist_batch(&self, ids: &[String]) -> Result<HashSet<String>, MemoryError> {
        if ids.is_empty() {
            return Ok(HashSet::new());
        }

        let conn = self.conn.lock().unwrap();

        let placeholders: Vec<&str> = ids.iter().map(|_| "?").collect();
        let sql = format!(
            "SELECT id FROM memories WHERE id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

        let rows = stmt.query_map(params.as_slice(), |row| {
            let id: String = row.get(0)?;
            Ok(id)
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Get relationships from multiple source IDs in a single query.
    /// Returns a HashMap from source_id to Vec<Relationship>.
    pub fn get_relationships_from_batch(
        &self,
        source_ids: &[String],
    ) -> Result<HashMap<String, Vec<Relationship>>, MemoryError> {
        if source_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let conn = self.conn.lock().unwrap();

        let placeholders: Vec<&str> = source_ids.iter().map(|_| "?").collect();
        let sql = format!(
            "SELECT id, source_id, target_id, relation_type, strength, created_at FROM relationships WHERE source_id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> = source_ids
            .iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();

        let rows = stmt.query_map(params.as_slice(), |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        let mut result: HashMap<String, Vec<Relationship>> = HashMap::new();
        for rel in rows.flatten() {
            result.entry(rel.source_id.clone()).or_default().push(rel);
        }

        Ok(result)
    }

    /// Get relationships to multiple target IDs in a single query.
    /// Returns a HashMap from target_id to Vec<Relationship>.
    pub fn get_relationships_to_batch(
        &self,
        target_ids: &[String],
    ) -> Result<HashMap<String, Vec<Relationship>>, MemoryError> {
        if target_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let conn = self.conn.lock().unwrap();

        let placeholders: Vec<&str> = target_ids.iter().map(|_| "?").collect();
        let sql = format!(
            "SELECT id, source_id, target_id, relation_type, strength, created_at FROM relationships WHERE target_id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> = target_ids
            .iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();

        let rows = stmt.query_map(params.as_slice(), |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        let mut result: HashMap<String, Vec<Relationship>> = HashMap::new();
        for rel in rows.flatten() {
            result.entry(rel.target_id.clone()).or_default().push(rel);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_crud() {
        let db = Database::open_in_memory().unwrap();

        // Create project
        let project = Project {
            id: "test-project".to_string(),
            name: "Test Project".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: chrono::Utc::now().timestamp(),
        };
        db.create_project(&project).unwrap();

        // Store memory
        let now = chrono::Utc::now().timestamp();
        let memory = Memory {
            id: "mem-1".to_string(),
            project_id: "test-project".to_string(),
            memory_type: MemoryType::Fact,
            content: "Test content".to_string(),
            summary: None,
            tags: vec!["test".to_string()],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };
        db.store_memory(&memory).unwrap();

        // Get memory
        let retrieved = db.get_memory("mem-1").unwrap().unwrap();
        assert_eq!(retrieved.content, "Test content");

        // Delete memory
        assert!(db.delete_memory("mem-1").unwrap());
        assert!(db.get_memory("mem-1").unwrap().is_none());
    }
}
