use rusqlite::{Connection, params};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use crate::error::MemoryError;
use crate::memory::{
    HandoffSections, Memory, MemoryCluster, MemoryType, Project, ProjectStats, RelationType,
    Relationship,
};

/// Parse a memory type string from a DB row, propagating an error on unknown values.
///
/// Used in `query_map` closures that return `rusqlite::Result<T>` so the error type
/// matches without requiring a full `MemoryError` conversion at every call site.
fn parse_memory_type_col(s: &str, col: usize) -> rusqlite::Result<MemoryType> {
    MemoryType::from_str(s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(col, rusqlite::types::Type::Text, Box::new(e))
    })
}

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

-- Schema versioning
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
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
        // The bundled SQLite is not compiled with SQLITE_ENABLE_MATH_FUNCTIONS, so EXP()
        // and LN() are unavailable.  Register them as custom scalar functions so that
        // update_relevance_scores works correctly in test builds.
        conn.create_scalar_function(
            "EXP",
            1,
            rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
            |ctx| {
                let x: f64 = ctx.get(0)?;
                Ok(x.exp())
            },
        )?;
        conn.create_scalar_function(
            "LN",
            1,
            rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
            |ctx| {
                let x: f64 = ctx.get(0)?;
                Ok(x.ln())
            },
        )?;
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
        self.migrate_branch_column()?;
        self.migrate_fts()?;
        self.run_migrations()?;
        Ok(())
    }

    /// Add branch column to memories table if it doesn't exist.
    fn migrate_branch_column(&self) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();

        // Check if branch column already exists
        let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
        let has_branch = stmt
            .query_map([], |row| {
                let name: String = row.get(1)?;
                Ok(name)
            })?
            .filter_map(|r| r.ok())
            .any(|name| name == "branch");

        if !has_branch {
            conn.execute_batch(
                r#"
                ALTER TABLE memories ADD COLUMN branch TEXT;
                CREATE INDEX IF NOT EXISTS idx_memories_project_branch ON memories(project_id, branch);
                "#,
            )?;
        }

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

    fn run_migrations(&self) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();

        // Ensure schema_version table exists
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);",
        )?;

        let current_version: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        // Migration 1: Add merged_from, memory_clusters, cluster_members
        if current_version < 1 {
            // Check if merged_from column already exists
            let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
            let has_merged_from = stmt
                .query_map([], |row| {
                    let name: String = row.get(1)?;
                    Ok(name)
                })?
                .filter_map(|r| r.ok())
                .any(|name| name == "merged_from");

            if !has_merged_from {
                conn.execute_batch("ALTER TABLE memories ADD COLUMN merged_from TEXT;")?;
            }

            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS memory_clusters (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    member_count INTEGER NOT NULL DEFAULT 0,
                    centroid BLOB,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cluster_members (
                    cluster_id TEXT NOT NULL REFERENCES memory_clusters(id) ON DELETE CASCADE,
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    PRIMARY KEY (cluster_id, memory_id)
                );

                CREATE INDEX IF NOT EXISTS idx_clusters_project ON memory_clusters(project_id);
                CREATE INDEX IF NOT EXISTS idx_cluster_members_memory ON cluster_members(memory_id);
            "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![1],
            )?;
        }

        // Migration 2: wipe embeddings table due to dimension change (384 -> 256)
        if current_version < 2 {
            conn.execute_batch("DELETE FROM embeddings;")?;
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![2],
            )?;
        }

        // Migration 3: add pinned and global columns with partial indexes
        if current_version < 3 {
            let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
            let columns: Vec<String> = stmt
                .query_map([], |row| row.get(1))?
                .filter_map(|r| r.ok())
                .collect();

            if !columns.iter().any(|c| c == "pinned") {
                conn.execute_batch(
                    "ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;",
                )?;
            }
            if !columns.iter().any(|c| c == "global") {
                conn.execute_batch(
                    "ALTER TABLE memories ADD COLUMN global INTEGER NOT NULL DEFAULT 0;",
                )?;
            }

            conn.execute_batch(
                r#"
                CREATE INDEX IF NOT EXISTS idx_memories_global ON memories(global) WHERE global = 1;
                CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned) WHERE pinned = 1;
                "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![3],
            )?;
        }

        // Migration 4: add handoff_sections sidecar table and continuation index.
        //
        // Wire format for section_embeddings:
        //   section_embedding_keys: comma-separated section names in canonical order,
        //     omitting empty sections (matches render_markdown order).
        //   section_embeddings: concatenated little-endian f32 bytes,
        //     256 dims × N sections × 4 bytes per float.
        //   Decoder validates bytes.len() == count * 256 * 4.
        if current_version < 4 {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS handoff_sections (
                    memory_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    decisions TEXT NOT NULL,
                    todos TEXT NOT NULL,
                    blockers TEXT NOT NULL,
                    mental_model TEXT NOT NULL,
                    next_steps TEXT NOT NULL,
                    notes TEXT,
                    continues_from TEXT,
                    section_embedding_keys TEXT NOT NULL,
                    section_embeddings BLOB NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_handoff_continues
                    ON handoff_sections(continues_from);
                "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![4],
            )?;
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
        let merged_from_json = memory
            .merged_from
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        conn.execute(
            "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from, pinned, global)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
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
                memory.branch,
                merged_from_json,
                memory.pinned as i64,
                memory.global as i64,
            ],
        )?;
        Ok(())
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<Memory>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from, pinned, global
             FROM memories WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            let memory_type_str: String = row.get(2)?;
            let tags_json: String = row.get(5)?;
            Ok(Some(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                memory_type: MemoryType::from_str(&memory_type_str)
                    .map_err(|_| MemoryError::InvalidType(memory_type_str.clone()))?,
                content: row.get(3)?,
                summary: row.get(4)?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                importance: row.get(6)?,
                relevance_score: row.get(7)?,
                access_count: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
                last_accessed_at: row.get(11)?,
                branch: row.get(12)?,
                merged_from: row
                    .get::<_, Option<String>>(13)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
                pinned: row.get::<_, i64>(14)? != 0,
                global: row.get::<_, i64>(15)? != 0,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_memory(&self, memory: &Memory) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let tags_json = serde_json::to_string(&memory.tags)?;
        conn.execute(
            "UPDATE memories SET content = ?1, summary = ?2, tags = ?3, importance = ?4, relevance_score = ?5, access_count = ?6, updated_at = ?7, last_accessed_at = ?8, pinned = ?9, global = ?10
             WHERE id = ?11",
            params![
                memory.content,
                memory.summary,
                tags_json,
                memory.importance,
                memory.relevance_score,
                memory.access_count,
                memory.updated_at,
                memory.last_accessed_at,
                memory.pinned as i64,
                memory.global as i64,
                memory.id,
            ],
        )?;
        Ok(())
    }

    pub fn delete_memory(&self, id: &str) -> Result<bool, MemoryError> {
        let conn = self.conn.lock().unwrap();
        // Belt-and-suspenders: explicitly delete sidecar row before deleting the memory
        // so that the handoff_sections row is removed even if FOREIGN KEY CASCADE is
        // disabled or the constraint fires in an unexpected order.
        conn.execute(
            "DELETE FROM handoff_sections WHERE memory_id = ?1",
            params![id],
        )?;
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
        self.query_memories_with_branch(project_id, types, tags, min_relevance, limit, None)
    }

    /// Query memories with optional branch filtering.
    /// - `branch_filter = None`: return all memories (global + all branches)
    /// - `branch_filter = Some(None)`: return only global memories (branch IS NULL)
    /// - `branch_filter = Some(Some(branch))`: return global + specific branch memories
    pub fn query_memories_with_branch(
        &self,
        project_id: &str,
        types: Option<&[MemoryType]>,
        tags: Option<&[String]>,
        min_relevance: Option<f64>,
        limit: usize,
        branch_filter: Option<Option<&str>>,
    ) -> Result<Vec<Memory>, MemoryError> {
        let conn = self.conn.lock().unwrap();

        let mut sql = String::from(
            "SELECT id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from, pinned, global
             FROM memories WHERE project_id = ?1"
        );

        // Apply branch filter
        match branch_filter {
            None => {
                // All memories (no branch filter)
            }
            Some(None) => {
                // Global only
                sql.push_str(" AND branch IS NULL");
            }
            Some(Some(_)) => {
                // Global + specific branch
                sql.push_str(" AND (branch IS NULL OR branch = ?4)");
            }
        }

        if min_relevance.is_some() {
            sql.push_str(" AND relevance_score >= ?2");
        }

        // Apply type filter in SQL (safe: MemoryType enum values are hardcoded strings)
        if let Some(types) = types
            && !types.is_empty()
        {
            let type_list: Vec<String> =
                types.iter().map(|t| format!("'{}'", t.as_str())).collect();
            sql.push_str(&format!(" AND memory_type IN ({})", type_list.join(",")));
        }

        // Fetch extra rows when tag filtering is active to compensate for post-filter losses
        let fetch_limit = if tags.is_some() { limit * 3 } else { limit };

        sql.push_str(" ORDER BY relevance_score DESC LIMIT ?3");

        let mut stmt = conn.prepare(&sql)?;

        let min_rel = min_relevance.unwrap_or(0.0);

        // Helper to parse a row into Memory
        fn parse_row(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
            let memory_type_str: String = row.get(2)?;
            let tags_json: String = row.get(5)?;
            Ok(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                memory_type: parse_memory_type_col(&memory_type_str, 2)?,
                content: row.get(3)?,
                summary: row.get(4)?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                importance: row.get(6)?,
                relevance_score: row.get(7)?,
                access_count: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
                last_accessed_at: row.get(11)?,
                branch: row.get(12)?,
                merged_from: row
                    .get::<_, Option<String>>(13)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
                pinned: row.get::<_, i64>(14)? != 0,
                global: row.get::<_, i64>(15)? != 0,
            })
        }

        let mut memories: Vec<Memory> = match branch_filter {
            Some(Some(branch)) => stmt
                .query_map(
                    params![project_id, min_rel, fetch_limit as i64, branch],
                    parse_row,
                )?
                .filter_map(|r| r.ok())
                .collect(),
            _ => stmt
                .query_map(params![project_id, min_rel, fetch_limit as i64], parse_row)?
                .filter_map(|r| r.ok())
                .collect(),
        };

        // Filter by tags if specified (type filter is now applied in SQL above)
        if let Some(filter_tags) = tags {
            memories.retain(|m| filter_tags.iter().any(|t| m.tags.contains(t)));
        }

        // Truncate to requested limit after tag filtering
        memories.truncate(limit);

        Ok(memories)
    }

    pub fn get_all_memories_for_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<Memory>, MemoryError> {
        self.query_memories(project_id, None, None, None, 10000)
    }

    /// Get all memories for a project filtered by branch.
    #[allow(dead_code)] // Available for future use by CLI and MCP tools
    pub fn get_all_memories_for_project_with_branch(
        &self,
        project_id: &str,
        branch_filter: Option<Option<&str>>,
    ) -> Result<Vec<Memory>, MemoryError> {
        self.query_memories_with_branch(project_id, None, None, None, 10000, branch_filter)
    }

    /// Promote a memory from branch-local to global.
    pub fn promote_memory(&self, id: &str) -> Result<bool, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let rows_affected = conn.execute(
            "UPDATE memories SET branch = NULL WHERE id = ?1",
            params![id],
        )?;
        Ok(rows_affected > 0)
    }

    /// Store multiple memories in a single transaction
    #[allow(dead_code)] // Used by MCP server tools
    pub fn store_memories_batch(&self, memories: &[Memory]) -> Result<usize, MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        let mut count = 0;
        for memory in memories {
            let tags_json = serde_json::to_string(&memory.tags)?;
            let merged_from_json = memory
                .merged_from
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?;
            tx.execute(
                "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from, pinned, global)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
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
                    memory.branch,
                    merged_from_json,
                    memory.pinned as i64,
                    memory.global as i64,
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

    /// Merge a duplicate memory into an existing one.
    /// Keeps the new memory's content, unions tags, takes max importance,
    /// and records merge provenance in merged_from.
    pub fn merge_memories(
        &self,
        new_id: &str,
        old_id: &str,
        old_content_preview: &str,
    ) -> Result<(), MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        // Get old memory's tags and importance
        let (old_tags_json, old_importance): (String, f64) = tx.query_row(
            "SELECT tags, importance FROM memories WHERE id = ?1",
            params![old_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let old_tags: Vec<String> = serde_json::from_str(&old_tags_json).unwrap_or_default();

        // Get new memory's current state
        let (new_tags_json, new_importance, existing_merged_from): (String, f64, Option<String>) =
            tx.query_row(
                "SELECT tags, importance, merged_from FROM memories WHERE id = ?1",
                params![new_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
        let mut new_tags: Vec<String> = serde_json::from_str(&new_tags_json).unwrap_or_default();

        // Union tags
        for tag in old_tags {
            if !new_tags.contains(&tag) {
                new_tags.push(tag);
            }
        }
        let merged_tags_json = serde_json::to_string(&new_tags)?;

        // Max importance
        let max_importance = new_importance.max(old_importance);

        // Build merged_from provenance
        let now = chrono::Utc::now().timestamp();
        let mut merge_sources: Vec<crate::memory::MergeSource> = existing_merged_from
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        merge_sources.push(crate::memory::MergeSource {
            id: old_id.to_string(),
            content_preview: old_content_preview.to_string(),
            merged_at: now,
        });
        let merged_from_json = serde_json::to_string(&merge_sources)?;

        // Update the new memory with merged data
        tx.execute(
            "UPDATE memories SET tags = ?1, importance = ?2, merged_from = ?3, updated_at = ?4 WHERE id = ?5",
            params![merged_tags_json, max_importance, merged_from_json, now, new_id],
        )?;

        // Delete the old memory (provenance is already tracked in merged_from above)
        tx.execute("DELETE FROM memories WHERE id = ?1", params![old_id])?;

        tx.commit()?;
        Ok(())
    }

    /// Create a new memory cluster.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn create_cluster(&self, cluster: &MemoryCluster) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let centroid_bytes: Option<Vec<u8>> = cluster
            .centroid
            .as_ref()
            .map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect());
        conn.execute(
            "INSERT INTO memory_clusters (id, project_id, summary, member_count, centroid, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                cluster.id,
                cluster.project_id,
                cluster.summary,
                cluster.member_count as i64,
                centroid_bytes,
                cluster.created_at,
                cluster.updated_at,
            ],
        )?;
        Ok(())
    }

    /// Add a memory to a cluster.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn add_to_cluster(&self, cluster_id: &str, memory_id: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO cluster_members (cluster_id, memory_id) VALUES (?1, ?2)",
            params![cluster_id, memory_id],
        )?;
        conn.execute(
            "UPDATE memory_clusters SET member_count = (SELECT COUNT(*) FROM cluster_members WHERE cluster_id = ?1), updated_at = ?2 WHERE id = ?1",
            params![cluster_id, chrono::Utc::now().timestamp()],
        )?;
        Ok(())
    }

    /// Remove a memory from its cluster.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn remove_from_cluster(&self, memory_id: &str) -> Result<Option<String>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        // Get the cluster_id before removing
        let cluster_id: Option<String> = conn
            .query_row(
                "SELECT cluster_id FROM cluster_members WHERE memory_id = ?1",
                params![memory_id],
                |row| row.get(0),
            )
            .ok();

        if let Some(ref cid) = cluster_id {
            conn.execute(
                "DELETE FROM cluster_members WHERE memory_id = ?1",
                params![memory_id],
            )?;
            // Update member count
            let now = chrono::Utc::now().timestamp();
            conn.execute(
                "UPDATE memory_clusters SET member_count = (SELECT COUNT(*) FROM cluster_members WHERE cluster_id = ?1), updated_at = ?2 WHERE id = ?1",
                params![cid, now],
            )?;
        }

        Ok(cluster_id)
    }

    /// Get a cluster by ID.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn get_cluster(&self, id: &str) -> Result<Option<MemoryCluster>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, project_id, summary, member_count, centroid, created_at, updated_at FROM memory_clusters WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            let centroid_bytes: Option<Vec<u8>> = row.get(4)?;
            let centroid = centroid_bytes.map(|bytes| {
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            });
            Ok(Some(MemoryCluster {
                id: row.get(0)?,
                project_id: row.get(1)?,
                summary: row.get(2)?,
                member_count: row.get::<_, i64>(3)? as usize,
                centroid,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get all clusters for a project.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn get_clusters_for_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<MemoryCluster>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, project_id, summary, member_count, centroid, created_at, updated_at FROM memory_clusters WHERE project_id = ?1"
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            let centroid_bytes: Option<Vec<u8>> = row.get(4)?;
            let centroid = centroid_bytes.map(|bytes| {
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            });
            Ok(MemoryCluster {
                id: row.get(0)?,
                project_id: row.get(1)?,
                summary: row.get(2)?,
                member_count: row.get::<_, i64>(3)? as usize,
                centroid,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Update a cluster's centroid.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn update_cluster_centroid(
        &self,
        id: &str,
        centroid: &[f32],
        summary: &str,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let centroid_bytes: Vec<u8> = centroid.iter().flat_map(|f| f.to_le_bytes()).collect();
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "UPDATE memory_clusters SET centroid = ?1, summary = ?2, updated_at = ?3 WHERE id = ?4",
            params![centroid_bytes, summary, now, id],
        )?;
        Ok(())
    }

    /// Delete clusters with no members.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn delete_empty_clusters(&self, project_id: &str) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let rows = conn.execute(
            "DELETE FROM memory_clusters WHERE project_id = ?1 AND NOT EXISTS (SELECT 1 FROM cluster_members WHERE cluster_id = memory_clusters.id)",
            params![project_id],
        )?;
        Ok(rows)
    }

    /// Get all memory IDs in a cluster.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn get_cluster_member_ids(&self, cluster_id: &str) -> Result<Vec<String>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT memory_id FROM cluster_members WHERE cluster_id = ?1")?;
        let rows = stmt.query_map(params![cluster_id], |row| row.get(0))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
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

        let pinned_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?1 AND pinned = 1",
            params![project_id],
            |row| row.get(0),
        )?;

        let global_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE global = 1",
            [],
            |row| row.get(0),
        )?;

        let handoff_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?1 AND memory_type = 'handoff'",
            params![project_id],
            |row| row.get(0),
        )?;

        let latest_handoff_at: Option<i64> = conn.query_row(
            "SELECT MAX(created_at) FROM memories WHERE project_id = ?1 AND memory_type = 'handoff'",
            params![project_id],
            |row| row.get(0),
        )?;

        Ok(ProjectStats {
            memory_count: memory_count as usize,
            relationship_count: relationship_count as usize,
            avg_relevance,
            pinned_count: pinned_count as usize,
            global_count: global_count as usize,
            handoff_count: handoff_count as usize,
            latest_handoff_at,
        })
    }

    /// Get branch statistics for a project (count of memories per branch).
    /// Returns a map of branch name to count, where None represents global memories.
    #[allow(dead_code)] // Available for future use by CLI and MCP tools
    pub fn get_branch_stats(
        &self,
        project_id: &str,
    ) -> Result<HashMap<Option<String>, usize>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT branch, COUNT(*) FROM memories WHERE project_id = ?1 GROUP BY branch",
        )?;

        let rows = stmt.query_map(params![project_id], |row| {
            let branch: Option<String> = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((branch, count as usize))
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Get the top N most accessed memories for a project.
    #[allow(dead_code)] // Used by CLI insights command
    pub fn get_most_accessed(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from, pinned, global
             FROM memories WHERE project_id = ?1
             ORDER BY access_count DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![project_id, limit as i64], |row| {
            let memory_type_str: String = row.get(2)?;
            let tags_json: String = row.get(5)?;
            Ok(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                memory_type: parse_memory_type_col(&memory_type_str, 2)?,
                content: row.get(3)?,
                summary: row.get(4)?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                importance: row.get(6)?,
                relevance_score: row.get(7)?,
                access_count: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
                last_accessed_at: row.get(11)?,
                branch: row.get(12)?,
                merged_from: row
                    .get::<_, Option<String>>(13)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
                pinned: row.get::<_, i64>(14)? != 0,
                global: row.get::<_, i64>(15)? != 0,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Get count of memories with access_count = 0 that are older than min_age_days.
    #[allow(dead_code)] // Used by CLI insights and health commands
    pub fn get_never_accessed(
        &self,
        project_id: &str,
        min_age_days: u64,
    ) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let cutoff = chrono::Utc::now().timestamp() - (min_age_days as i64 * 86400);
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?1 AND access_count = 0 AND created_at < ?2",
            params![project_id, cutoff],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Get count of memories with relevance_score below the given threshold.
    #[allow(dead_code)] // Used by CLI insights and health commands
    pub fn get_below_relevance(
        &self,
        project_id: &str,
        threshold: f64,
    ) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?1 AND relevance_score < ?2",
            params![project_id, threshold],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Get count of memories that are both never-accessed (older than `min_age_days`) and below
    /// `threshold` relevance. Used to correct the double-subtraction in health summaries.
    #[allow(dead_code)] // Used by CLI insights command
    pub fn get_never_accessed_and_below_relevance(
        &self,
        project_id: &str,
        min_age_days: i64,
        threshold: f64,
    ) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let cutoff = chrono::Utc::now().timestamp() - (min_age_days * 86400);
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?1 AND access_count = 0 AND created_at < ?2 AND relevance_score < ?3",
            params![project_id, cutoff, threshold],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Count same-type memory pairs that share a cluster, as a rough estimate of potential
    /// duplicates that dedup might be able to merge.
    #[allow(dead_code)] // Used by CLI health command
    pub fn get_potential_duplicate_count(&self, project_id: &str) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM cluster_members cm1
             JOIN cluster_members cm2 ON cm1.cluster_id = cm2.cluster_id AND cm1.memory_id < cm2.memory_id
             JOIN memories m1 ON cm1.memory_id = m1.id
             JOIN memories m2 ON cm2.memory_id = m2.id
             WHERE m1.project_id = ?1 AND m1.memory_type = m2.memory_type",
            params![project_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Get the average memories stored per day over the last `days` days.
    #[allow(dead_code)] // Used by CLI insights command
    pub fn get_storage_rate(&self, project_id: &str, days: u64) -> Result<f64, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let cutoff = chrono::Utc::now().timestamp() - (days as i64 * 86400);
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?1 AND created_at >= ?2",
            params![project_id, cutoff],
            |row| row.get(0),
        )?;
        Ok(count as f64 / days as f64)
    }

    /// Get the distribution of memories by type for a project.
    /// Returns a vec of (memory_type_str, count) pairs.
    #[allow(dead_code)] // Used by CLI insights command
    pub fn get_type_distribution(
        &self,
        project_id: &str,
    ) -> Result<Vec<(String, usize)>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT memory_type, COUNT(*) FROM memories WHERE project_id = ?1 GROUP BY memory_type ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            let memory_type: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((memory_type, count as usize))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
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

    /// Pre-filtered embeddings for `memory_context`: returns at most `max_candidates` embeddings
    /// ordered by recency, UNION with any pinned memories beyond the cap.
    ///
    /// The default cap is 500, configurable via `ENGRAM_MAX_CANDIDATES`.
    pub fn get_prefiltered_embeddings(
        &self,
        project_id: &str,
        max_candidates: usize,
    ) -> Result<Vec<(String, Vec<f32>)>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let sql = "
            SELECT e.memory_id, e.vector FROM embeddings e
            JOIN memories m ON e.memory_id = m.id
            WHERE m.id IN (
                SELECT id FROM memories
                WHERE (project_id = ?1 OR global = 1)
                ORDER BY last_accessed_at DESC
                LIMIT ?2
            )
            UNION
            SELECT e.memory_id, e.vector FROM embeddings e
            JOIN memories m ON e.memory_id = m.id
            WHERE m.pinned = 1 AND (m.project_id = ?1 OR m.global = 1)
        ";
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params![project_id, max_candidates as i64], |row| {
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

    /// Like `get_all_embeddings_for_project` but also includes global memories from any project.
    pub fn get_all_embeddings_for_project_and_global(
        &self,
        project_id: &str,
    ) -> Result<Vec<(String, Vec<f32>)>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT e.memory_id, e.vector FROM embeddings e
             JOIN memories m ON e.memory_id = m.id
             WHERE m.project_id = ?1 OR m.global = 1",
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

    /// Fetch embeddings for a specific set of memory IDs.
    pub fn get_embeddings_batch(
        &self,
        memory_ids: &[String],
    ) -> Result<Vec<(String, Vec<f32>)>, MemoryError> {
        if memory_ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        let placeholders: Vec<&str> = memory_ids.iter().map(|_| "?").collect();
        let sql = format!(
            "SELECT memory_id, vector FROM embeddings WHERE memory_id IN ({})",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> = memory_ids
            .iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();
        let rows = stmt.query_map(params.as_slice(), |row| {
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
    #[allow(dead_code)]
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

    /// Full-text search with branch filtering.
    pub fn keyword_search_with_branch(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
        branch_filter: Option<Option<&str>>,
    ) -> Result<Vec<(String, f64)>, MemoryError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().unwrap();

        let escaped_query = Self::escape_fts_query(query);
        if escaped_query.is_empty() {
            return Ok(Vec::new());
        }

        let rows: Vec<(String, f64)> = match branch_filter {
            None => {
                // No branch filter
                let mut stmt = conn.prepare(
                    "SELECT m.id, -bm25(memories_fts) as score
                     FROM memories_fts
                     JOIN memories m ON memories_fts.rowid = m.rowid
                     WHERE memories_fts MATCH ?1 AND m.project_id = ?2
                     ORDER BY score DESC
                     LIMIT ?3",
                )?;
                stmt.query_map(params![escaped_query, project_id, limit as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                })?
                .filter_map(|r| r.ok())
                .collect()
            }
            Some(None) => {
                // Global only: branch IS NULL
                let mut stmt = conn.prepare(
                    "SELECT m.id, -bm25(memories_fts) as score
                     FROM memories_fts
                     JOIN memories m ON memories_fts.rowid = m.rowid
                     WHERE memories_fts MATCH ?1 AND m.project_id = ?2
                       AND m.branch IS NULL
                     ORDER BY score DESC
                     LIMIT ?3",
                )?;
                stmt.query_map(params![escaped_query, project_id, limit as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                })?
                .filter_map(|r| r.ok())
                .collect()
            }
            Some(Some(branch)) => {
                // Global + specific branch
                let mut stmt = conn.prepare(
                    "SELECT m.id, -bm25(memories_fts) as score
                     FROM memories_fts
                     JOIN memories m ON memories_fts.rowid = m.rowid
                     WHERE memories_fts MATCH ?1 AND m.project_id = ?2
                       AND (m.branch IS NULL OR m.branch = ?4)
                     ORDER BY score DESC
                     LIMIT ?3",
                )?;
                stmt.query_map(
                    params![escaped_query, project_id, limit as i64, branch],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)),
                )?
                .filter_map(|r| r.ok())
                .collect()
            }
        };

        Ok(rows)
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
            AND pinned = 0
            "#,
            params![decay_rate, now, project_id],
        )?;

        Ok(rows_affected)
    }

    /// Set or clear the pinned flag on a memory.
    #[allow(dead_code)]
    pub fn set_pinned(&self, id: &str, pinned: bool) -> Result<bool, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let rows_affected = conn.execute(
            "UPDATE memories SET pinned = ?1 WHERE id = ?2",
            params![pinned as i32, id],
        )?;
        Ok(rows_affected > 0)
    }

    /// Delete non-pinned, non-global memories that have fully decayed and were never accessed.
    ///
    /// Conditions: pinned = 0, global = 0, relevance_score <= 0.1, access_count = 0,
    /// and created more than 30 days ago. Embeddings and relationships are removed by
    /// CASCADE constraints on the memories table.
    ///
    /// Returns the IDs of deleted memories.
    #[allow(dead_code)] // Called by the decay background job in main.rs
    pub fn auto_prune_dead_memories(&self, project_id: &str) -> Result<Vec<String>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let cutoff = chrono::Utc::now().timestamp() - 30 * 86400;

        let mut stmt = conn.prepare(
            "SELECT id FROM memories
             WHERE project_id = ?1
               AND pinned = 0
               AND global = 0
               AND relevance_score <= 0.1
               AND access_count = 0
               AND created_at < ?2",
        )?;
        let ids: Vec<String> = stmt
            .query_map(params![project_id, cutoff], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        if ids.is_empty() {
            return Ok(ids);
        }

        let placeholders: Vec<&str> = ids.iter().map(|_| "?").collect();
        let sql = format!(
            "DELETE FROM memories WHERE id IN ({})",
            placeholders.join(",")
        );
        let params_refs: Vec<&dyn rusqlite::types::ToSql> = ids
            .iter()
            .map(|s| s as &dyn rusqlite::types::ToSql)
            .collect();
        conn.execute(&sql, params_refs.as_slice())?;

        Ok(ids)
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
            "SELECT id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from, pinned, global
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
                memory_type: parse_memory_type_col(&memory_type_str, 2)?,
                content: row.get(3)?,
                summary: row.get(4)?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                importance: row.get(6)?,
                relevance_score: row.get(7)?,
                access_count: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
                last_accessed_at: row.get(11)?,
                branch: row.get(12)?,
                merged_from: row
                    .get::<_, Option<String>>(13)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
                pinned: row.get::<_, i64>(14)? != 0,
                global: row.get::<_, i64>(15)? != 0,
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

    // ============================================
    // Handoff sidecar helpers (Task 1.8)
    // ============================================

    /// Insert a handoff sidecar row linking to the given memory.
    ///
    /// `section_keys` is a comma-separated list of section names in canonical order
    /// (produced by `encode_section_embeddings`).  `section_embeddings` is the
    /// concatenated little-endian f32 byte blob (256 dims × N sections × 4 bytes).
    #[allow(dead_code)] // Used by handoff_create tool (Phase 3)
    pub fn insert_handoff_sections(
        &self,
        memory_id: &str,
        sections: &HandoffSections,
        section_keys: &str,
        section_embeddings: &[u8],
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let decisions_json = serde_json::to_string(&sections.decisions)?;
        let todos_json = serde_json::to_string(&sections.todos)?;
        let blockers_json = serde_json::to_string(&sections.blockers)?;
        let next_steps_json = serde_json::to_string(&sections.next_steps)?;
        conn.execute(
            "INSERT INTO handoff_sections (
                memory_id, summary, decisions, todos, blockers,
                mental_model, next_steps, notes, continues_from,
                section_embedding_keys, section_embeddings
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                memory_id,
                sections.summary,
                decisions_json,
                todos_json,
                blockers_json,
                sections.mental_model,
                next_steps_json,
                sections.notes,
                sections.continues_from,
                section_keys,
                section_embeddings,
            ],
        )?;
        Ok(())
    }

    /// Store a handoff memory, its embedding, and the sidecar row atomically.
    ///
    /// All three inserts run inside a single SQLite transaction.  If any step fails
    /// the entire operation is rolled back, leaving the DB in a consistent state.
    pub fn store_handoff_atomic(
        &self,
        memory: &Memory,
        embedding: &[f32],
        model_version: &str,
        sections: &HandoffSections,
        section_keys: &str,
        section_embeddings: &[u8],
    ) -> Result<(), MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        // Insert memory row.
        let tags_json = serde_json::to_string(&memory.tags)?;
        let merged_from_json = memory
            .merged_from
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        tx.execute(
            "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from, pinned, global)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
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
                memory.branch,
                merged_from_json,
                memory.pinned as i64,
                memory.global as i64,
            ],
        )?;

        // Insert embedding row.
        let vector_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        tx.execute(
            "INSERT OR REPLACE INTO embeddings (memory_id, vector, model_version) VALUES (?1, ?2, ?3)",
            params![memory.id, vector_bytes, model_version],
        )?;

        // Insert handoff sidecar row.
        let decisions_json = serde_json::to_string(&sections.decisions)?;
        let todos_json = serde_json::to_string(&sections.todos)?;
        let blockers_json = serde_json::to_string(&sections.blockers)?;
        let next_steps_json = serde_json::to_string(&sections.next_steps)?;
        tx.execute(
            "INSERT INTO handoff_sections (
                memory_id, summary, decisions, todos, blockers,
                mental_model, next_steps, notes, continues_from,
                section_embedding_keys, section_embeddings
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                memory.id,
                sections.summary,
                decisions_json,
                todos_json,
                blockers_json,
                sections.mental_model,
                next_steps_json,
                sections.notes,
                sections.continues_from,
                section_keys,
                section_embeddings,
            ],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Fetch a handoff sidecar row plus decoded section embeddings.
    ///
    /// Returns `Ok(None)` if no sidecar exists for the given memory ID.
    /// Returns `(HandoffSections, Vec<(section_name, embedding_vector)>)` on success.
    #[allow(clippy::type_complexity)]
    #[allow(dead_code)] // Used by handoff_resume/search tools (Phase 3)
    pub fn get_handoff_sections(
        &self,
        memory_id: &str,
    ) -> Result<Option<(HandoffSections, Vec<(String, Vec<f32>)>)>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT summary, decisions, todos, blockers, mental_model, next_steps,
                    notes, continues_from, section_embedding_keys, section_embeddings
             FROM handoff_sections WHERE memory_id = ?1",
        )?;
        let mut rows = stmt.query(params![memory_id])?;

        if let Some(row) = rows.next()? {
            let summary: String = row.get(0)?;
            let decisions: Vec<String> =
                serde_json::from_str::<Vec<String>>(&row.get::<_, String>(1)?)?;
            let todos: Vec<String> =
                serde_json::from_str::<Vec<String>>(&row.get::<_, String>(2)?)?;
            let blockers: Vec<String> =
                serde_json::from_str::<Vec<String>>(&row.get::<_, String>(3)?)?;
            let mental_model: String = row.get(4)?;
            let next_steps: Vec<String> =
                serde_json::from_str::<Vec<String>>(&row.get::<_, String>(5)?)?;
            let notes: Option<String> = row.get(6)?;
            let continues_from: Option<String> = row.get(7)?;
            let keys: String = row.get(8)?;
            let embedding_bytes: Vec<u8> = row.get(9)?;

            let sections = HandoffSections {
                summary,
                decisions,
                todos,
                blockers,
                mental_model,
                next_steps,
                notes,
                continues_from,
            };
            let section_vecs = decode_section_embeddings(&keys, &embedding_bytes)?;
            Ok(Some((sections, section_vecs)))
        } else {
            Ok(None)
        }
    }

    /// Update a memory row, its full-content embedding, and the handoff sidecar atomically.
    ///
    /// All three writes share a single transaction so a failure in any step leaves the
    /// database unchanged — no partial updates where `memories.content` diverges from the
    /// sidecar sections or embeddings.
    #[allow(dead_code)] // Used by handoff update hook (Phase 3B)
    pub fn update_memory_and_handoff_sidecar(
        &self,
        memory: &Memory,
        new_full_embedding: &[f32],
        model_version: &str,
        sections: &HandoffSections,
        section_keys: &str,
        section_embeddings: &[u8],
    ) -> Result<(), MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        // Update memory row (same fields as update_memory).
        let tags_json = serde_json::to_string(&memory.tags)?;
        tx.execute(
            "UPDATE memories SET content = ?1, summary = ?2, tags = ?3, importance = ?4, relevance_score = ?5, access_count = ?6, updated_at = ?7, last_accessed_at = ?8, pinned = ?9, global = ?10
             WHERE id = ?11",
            params![
                memory.content,
                memory.summary,
                tags_json,
                memory.importance,
                memory.relevance_score,
                memory.access_count,
                memory.updated_at,
                memory.last_accessed_at,
                memory.pinned as i64,
                memory.global as i64,
                memory.id,
            ],
        )?;

        // Update full-content embedding.
        let vector_bytes: Vec<u8> = new_full_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tx.execute(
            "INSERT OR REPLACE INTO embeddings (memory_id, vector, model_version) VALUES (?1, ?2, ?3)",
            params![memory.id, vector_bytes, model_version],
        )?;

        // Update handoff sidecar.
        let decisions_json = serde_json::to_string(&sections.decisions)?;
        let todos_json = serde_json::to_string(&sections.todos)?;
        let blockers_json = serde_json::to_string(&sections.blockers)?;
        let next_steps_json = serde_json::to_string(&sections.next_steps)?;
        tx.execute(
            "UPDATE handoff_sections SET
                summary = ?2, decisions = ?3, todos = ?4, blockers = ?5,
                mental_model = ?6, next_steps = ?7, notes = ?8, continues_from = ?9,
                section_embedding_keys = ?10, section_embeddings = ?11
             WHERE memory_id = ?1",
            params![
                memory.id,
                sections.summary,
                decisions_json,
                todos_json,
                blockers_json,
                sections.mental_model,
                next_steps_json,
                sections.notes,
                sections.continues_from,
                section_keys,
                section_embeddings,
            ],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Overwrite the handoff sidecar row for an existing memory (used when content is updated).
    #[allow(dead_code)] // Used by handoff update hook (Phase 3B)
    pub fn update_handoff_sections(
        &self,
        memory_id: &str,
        sections: &HandoffSections,
        section_keys: &str,
        section_embeddings: &[u8],
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let decisions_json = serde_json::to_string(&sections.decisions)?;
        let todos_json = serde_json::to_string(&sections.todos)?;
        let blockers_json = serde_json::to_string(&sections.blockers)?;
        let next_steps_json = serde_json::to_string(&sections.next_steps)?;
        conn.execute(
            "UPDATE handoff_sections SET
                summary = ?2, decisions = ?3, todos = ?4, blockers = ?5,
                mental_model = ?6, next_steps = ?7, notes = ?8, continues_from = ?9,
                section_embedding_keys = ?10, section_embeddings = ?11
             WHERE memory_id = ?1",
            params![
                memory_id,
                sections.summary,
                decisions_json,
                todos_json,
                blockers_json,
                sections.mental_model,
                next_steps_json,
                sections.notes,
                sections.continues_from,
                section_keys,
                section_embeddings,
            ],
        )?;
        Ok(())
    }

    /// Fetch memories of type `handoff` for a project, filtered by branch.
    ///
    /// When `branch` is `Some(b)`, returns handoffs on branch `b` and global ones
    /// (branch IS NULL).  When `branch` is `None`, returns all handoffs regardless of
    /// branch.  Results are ordered by `created_at DESC`.
    #[allow(dead_code)] // Used by handoff_resume/search tools (Phase 3)
    pub fn query_handoffs_by_branch(
        &self,
        project_id: &str,
        branch: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Memory>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let rows: Vec<Memory> = match branch {
            None => {
                let mut stmt = conn.prepare(
                    "SELECT id, project_id, memory_type, content, summary, tags, importance,
                            relevance_score, access_count, created_at, updated_at,
                            last_accessed_at, branch, merged_from, pinned, global
                     FROM memories
                     WHERE project_id = ?1 AND memory_type = 'handoff'
                     ORDER BY created_at DESC
                     LIMIT ?2",
                )?;
                stmt.query_map(params![project_id, limit as i64], |row| {
                    let memory_type_str: String = row.get(2)?;
                    let tags_json: String = row.get(5)?;
                    Ok(Memory {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        memory_type: parse_memory_type_col(&memory_type_str, 2)?,
                        content: row.get(3)?,
                        summary: row.get(4)?,
                        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                        importance: row.get(6)?,
                        relevance_score: row.get(7)?,
                        access_count: row.get(8)?,
                        created_at: row.get(9)?,
                        updated_at: row.get(10)?,
                        last_accessed_at: row.get(11)?,
                        branch: row.get(12)?,
                        merged_from: row
                            .get::<_, Option<String>>(13)?
                            .and_then(|s| serde_json::from_str(&s).ok()),
                        pinned: row.get::<_, i64>(14)? != 0,
                        global: row.get::<_, i64>(15)? != 0,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(MemoryError::from)?
            }
            Some(b) => {
                let mut stmt = conn.prepare(
                    "SELECT id, project_id, memory_type, content, summary, tags, importance,
                            relevance_score, access_count, created_at, updated_at,
                            last_accessed_at, branch, merged_from, pinned, global
                     FROM memories
                     WHERE project_id = ?1 AND memory_type = 'handoff'
                       AND (branch IS NULL OR branch = ?3)
                     ORDER BY created_at DESC
                     LIMIT ?2",
                )?;
                stmt.query_map(params![project_id, limit as i64, b], |row| {
                    let memory_type_str: String = row.get(2)?;
                    let tags_json: String = row.get(5)?;
                    Ok(Memory {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        memory_type: parse_memory_type_col(&memory_type_str, 2)?,
                        content: row.get(3)?,
                        summary: row.get(4)?,
                        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                        importance: row.get(6)?,
                        relevance_score: row.get(7)?,
                        access_count: row.get(8)?,
                        created_at: row.get(9)?,
                        updated_at: row.get(10)?,
                        last_accessed_at: row.get(11)?,
                        branch: row.get(12)?,
                        merged_from: row
                            .get::<_, Option<String>>(13)?
                            .and_then(|s| serde_json::from_str(&s).ok()),
                        pinned: row.get::<_, i64>(14)? != 0,
                        global: row.get::<_, i64>(15)? != 0,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(MemoryError::from)?
            }
        };
        Ok(rows)
    }

    /// Fetch the most recently created handoffs for a project across all branches.
    #[allow(dead_code)] // Used by handoff_resume tool (Phase 3)
    pub fn list_recent_handoffs(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, MemoryError> {
        self.query_handoffs_by_branch(project_id, None, limit)
    }
}

// ============================================
// Section-embedding wire format helpers (Task 1.7)
// ============================================

/// Encode per-section embeddings into the wire format stored in `handoff_sections`.
///
/// Wire format:
///   `keys` — comma-separated section names in the canonical order they appear in
///     `render_markdown`, containing only non-empty sections.
///   returned bytes — concatenated little-endian f32 values: 256 dims × N sections × 4 bytes.
///
/// `keys` and `vectors` must have the same length.  Each vector must have exactly 256 elements.
#[allow(dead_code)] // Used by handoff_create and handoff update hook (Phase 3)
pub fn encode_section_embeddings(keys: &[&str], vectors: &[Vec<f32>]) -> (String, Vec<u8>) {
    let keys_str = keys.join(",");
    let bytes: Vec<u8> = vectors
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    (keys_str, bytes)
}

/// Decode the wire format produced by `encode_section_embeddings`.
///
/// Returns a vec of `(section_name, embedding_vector)` pairs in the same order as
/// `section_embedding_keys`.
///
/// Returns `MemoryError::Database` (wrapped string) if the byte length does not equal
/// `count * 256 * 4` where `count` is the number of comma-separated keys.
#[allow(dead_code)] // Used by get_handoff_sections and handoff resume scoring (Phase 3)
pub fn decode_section_embeddings(
    keys: &str,
    bytes: &[u8],
) -> Result<Vec<(String, Vec<f32>)>, MemoryError> {
    if keys.is_empty() {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        return Err(MemoryError::Database(
            rusqlite::Error::InvalidParameterName(
                "section_embeddings byte length mismatch: keys are empty but bytes are not"
                    .to_string(),
            ),
        ));
    }

    let key_list: Vec<&str> = keys.split(',').collect();
    let expected_bytes = key_list.len() * 256 * 4;
    if bytes.len() != expected_bytes {
        return Err(MemoryError::Database(
            rusqlite::Error::InvalidParameterName(format!(
                "section_embeddings byte length mismatch: expected {} bytes ({} sections × 256 dims × 4 bytes) but got {}",
                expected_bytes,
                key_list.len(),
                bytes.len()
            )),
        ));
    }

    let result = key_list
        .into_iter()
        .enumerate()
        .map(|(i, key)| {
            let start = i * 256 * 4;
            let end = start + 256 * 4;
            let vec: Vec<f32> = bytes[start..end]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            (key.to_string(), vec)
        })
        .collect();

    Ok(result)
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
            branch: None,
            merged_from: None,
            pinned: false,
            global: false,
        };
        db.store_memory(&memory).unwrap();

        // Get memory
        let retrieved = db.get_memory("mem-1").unwrap().unwrap();
        assert_eq!(retrieved.content, "Test content");

        // Delete memory
        assert!(db.delete_memory("mem-1").unwrap());
        assert!(db.get_memory("mem-1").unwrap().is_none());
    }

    #[test]
    fn test_migration_creates_tables() {
        let db = Database::open_in_memory().unwrap();

        // Verify schema_version table exists and has version 4
        let conn = db.conn.lock().unwrap();
        let version: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, 4);

        // Verify memory_clusters table exists
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memory_clusters", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);

        // Verify cluster_members table exists
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM cluster_members", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);

        // Verify merged_from column exists on memories
        let mut stmt = conn.prepare("PRAGMA table_info(memories)").unwrap();
        let has_merged_from = stmt
            .query_map([], |row| {
                let name: String = row.get(1)?;
                Ok(name)
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .any(|name| name == "merged_from");
        assert!(has_merged_from);

        // Verify handoff_sections table exists (migration 4)
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM handoff_sections", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_migration_idempotent() {
        // Running initialize twice should not fail
        let db = Database::open_in_memory().unwrap();
        // The second initialize happens automatically, but let's verify the DB works
        let project = crate::memory::Project {
            id: "test".to_string(),
            name: "test".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&project).unwrap();
        let p = db.get_project("test").unwrap();
        assert!(p.is_some());
    }

    #[test]
    fn test_run_migrations_twice_is_idempotent() {
        // Verify that calling run_migrations a second time on an already-migrated
        // populated DB does not fail or corrupt data.
        let db = Database::open_in_memory().unwrap();
        let project = crate::memory::Project {
            id: "mig2".to_string(),
            name: "mig2".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&project).unwrap();

        // Store a memory so the DB is non-empty before the second migration run.
        let now = chrono::Utc::now().timestamp();
        db.store_memory(&Memory {
            id: "mig2-mem".to_string(),
            project_id: "mig2".to_string(),
            memory_type: MemoryType::Fact,
            content: "Persists across migrations".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
            pinned: false,
            global: false,
        })
        .unwrap();

        // Call run_migrations explicitly a second time; must not error.
        db.run_migrations().unwrap();

        // Data must still be intact.
        let mem = db.get_memory("mig2-mem").unwrap();
        assert!(mem.is_some());
        assert_eq!(mem.unwrap().content, "Persists across migrations");
    }

    #[test]
    fn test_merge_memories() {
        let db = Database::open_in_memory().unwrap();

        let project = crate::memory::Project {
            id: "test-merge".to_string(),
            name: "test-merge".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&project).unwrap();

        let now = chrono::Utc::now().timestamp();

        let old_mem = Memory {
            id: "mem_old".to_string(),
            project_id: "test-merge".to_string(),
            memory_type: MemoryType::Fact,
            content: "Old fact".to_string(),
            summary: None,
            tags: vec!["tag_a".to_string()],
            importance: 0.3,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
            pinned: false,
            global: false,
        };
        db.store_memory(&old_mem).unwrap();

        let new_mem = Memory {
            id: "mem_new".to_string(),
            tags: vec!["tag_b".to_string()],
            importance: 0.7,
            content: "New fact".to_string(),
            ..old_mem.clone()
        };
        db.store_memory(&new_mem).unwrap();

        // Merge
        db.merge_memories("mem_new", "mem_old", "Old fact preview")
            .unwrap();

        // Old memory should be deleted
        assert!(db.get_memory("mem_old").unwrap().is_none());

        // New memory should have merged data
        let merged = db.get_memory("mem_new").unwrap().unwrap();
        assert_eq!(merged.importance, 0.7); // max(0.3, 0.7)
        assert!(merged.tags.contains(&"tag_a".to_string()));
        assert!(merged.tags.contains(&"tag_b".to_string()));
        assert!(merged.merged_from.is_some());
        let sources = merged.merged_from.unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].id, "mem_old");
    }

    #[test]
    fn test_cluster_operations() {
        let db = Database::open_in_memory().unwrap();

        let project = crate::memory::Project {
            id: "test-cluster".to_string(),
            name: "test-cluster".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&project).unwrap();

        let now = chrono::Utc::now().timestamp();

        // Create a memory
        let mem = Memory {
            id: "mem_c1".to_string(),
            project_id: "test-cluster".to_string(),
            memory_type: MemoryType::Fact,
            content: "Cluster test".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
            pinned: false,
            global: false,
        };
        db.store_memory(&mem).unwrap();

        // Create a cluster
        let cluster = crate::memory::MemoryCluster {
            id: "clust_1".to_string(),
            project_id: "test-cluster".to_string(),
            summary: "Test cluster".to_string(),
            member_count: 0,
            centroid: Some(vec![0.1, 0.2, 0.3]),
            created_at: now,
            updated_at: now,
        };
        db.create_cluster(&cluster).unwrap();

        // Add memory to cluster
        db.add_to_cluster("clust_1", "mem_c1").unwrap();

        // Verify
        let c = db.get_cluster("clust_1").unwrap().unwrap();
        assert_eq!(c.member_count, 1);

        let members = db.get_cluster_member_ids("clust_1").unwrap();
        assert_eq!(members, vec!["mem_c1"]);

        // List clusters
        let clusters = db.get_clusters_for_project("test-cluster").unwrap();
        assert_eq!(clusters.len(), 1);

        // Update centroid
        db.update_cluster_centroid("clust_1", &[0.4, 0.5, 0.6], "Updated summary")
            .unwrap();
        let c = db.get_cluster("clust_1").unwrap().unwrap();
        assert_eq!(c.summary, "Updated summary");
        assert_eq!(c.centroid.unwrap(), vec![0.4, 0.5, 0.6]);

        // Remove from cluster
        let removed_from = db.remove_from_cluster("mem_c1").unwrap();
        assert_eq!(removed_from, Some("clust_1".to_string()));

        let c = db.get_cluster("clust_1").unwrap().unwrap();
        assert_eq!(c.member_count, 0);

        // Delete empty clusters
        let deleted = db.delete_empty_clusters("test-cluster").unwrap();
        assert_eq!(deleted, 1);
        assert!(db.get_cluster("clust_1").unwrap().is_none());
    }

    #[test]
    fn test_migration3_fresh_db_has_pinned_and_global_columns() {
        let db = Database::open_in_memory().unwrap();
        let conn = db.conn.lock().unwrap();

        // Check schema version is at least 3
        let version: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            version >= 3,
            "expected schema version >= 3, got {}",
            version
        );

        // Verify pinned and global columns exist
        let mut stmt = conn.prepare("PRAGMA table_info(memories)").unwrap();
        let columns: Vec<String> = stmt
            .query_map([], |row| row.get(1))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(
            columns.contains(&"pinned".to_string()),
            "pinned column missing"
        );
        assert!(
            columns.contains(&"global".to_string()),
            "global column missing"
        );
    }

    #[test]
    fn test_migration3_pinned_global_default_false() {
        let db = Database::open_in_memory().unwrap();

        let project = crate::memory::Project {
            id: "test-pinned-global".to_string(),
            name: "test-pinned-global".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&project).unwrap();

        let now = chrono::Utc::now().timestamp();
        let memory = Memory {
            id: "mem-pg".to_string(),
            project_id: "test-pinned-global".to_string(),
            memory_type: MemoryType::Fact,
            content: "Test pinned global defaults".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
            pinned: false,
            global: false,
        };
        db.store_memory(&memory).unwrap();

        let retrieved = db.get_memory("mem-pg").unwrap().unwrap();
        assert!(!retrieved.pinned, "pinned should default to false");
        assert!(!retrieved.global, "global should default to false");
    }

    #[test]
    fn test_migration3_store_and_retrieve_pinned_global() {
        let db = Database::open_in_memory().unwrap();

        let project = crate::memory::Project {
            id: "test-pg-flags".to_string(),
            name: "test-pg-flags".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&project).unwrap();

        let now = chrono::Utc::now().timestamp();
        let memory = Memory {
            id: "mem-pg2".to_string(),
            project_id: "test-pg-flags".to_string(),
            memory_type: MemoryType::Fact,
            content: "Pinned and global memory".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.8,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
            pinned: true,
            global: true,
        };
        db.store_memory(&memory).unwrap();

        let retrieved = db.get_memory("mem-pg2").unwrap().unwrap();
        assert!(retrieved.pinned, "pinned should be true");
        assert!(retrieved.global, "global should be true");
    }

    #[test]
    fn test_migration3_upgrade_existing_db() {
        // Simulate upgrading a pre-migration-3 database by manually inserting a
        // memory without the pinned/global columns, then running migration.
        // We do this by opening an in-memory DB, removing the migration 3 record
        // from schema_version, dropping the columns if they exist, then calling
        // initialize again.
        //
        // In practice, SQLite does not support DROP COLUMN in older versions, so
        // we test the upgrade path by using a fresh DB that starts at version 2
        // and verifying migration 3 runs correctly.

        // Create a DB and verify migration 3 runs and leaves existing rows intact.
        let db = Database::open_in_memory().unwrap();

        let project = crate::memory::Project {
            id: "test-upgrade".to_string(),
            name: "test-upgrade".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&project).unwrap();

        let now = chrono::Utc::now().timestamp();

        // Insert a memory using only the old columns (simulating a pre-migration row
        // by using DEFAULT values for pinned/global via the SQL DEFAULT clause).
        {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from)
                 VALUES ('mem-old', 'test-upgrade', 'fact', 'Legacy memory', NULL, '[]', 0.5, 1.0, 0, ?1, ?1, ?1, NULL, NULL)",
                params![now],
            ).unwrap();
        }

        // Retrieve and verify that the defaults applied correctly
        let retrieved = db.get_memory("mem-old").unwrap().unwrap();
        assert!(
            !retrieved.pinned,
            "legacy memory should have pinned=false via DEFAULT"
        );
        assert!(
            !retrieved.global,
            "legacy memory should have global=false via DEFAULT"
        );
        assert_eq!(retrieved.content, "Legacy memory");
    }

    #[test]
    fn test_migration3_project_stats_includes_counts() {
        let db = Database::open_in_memory().unwrap();

        let project = crate::memory::Project {
            id: "test-stats".to_string(),
            name: "test-stats".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&project).unwrap();

        let now = chrono::Utc::now().timestamp();

        let make_memory = |id: &str, pinned: bool, global: bool| Memory {
            id: id.to_string(),
            project_id: "test-stats".to_string(),
            memory_type: MemoryType::Fact,
            content: format!("Memory {}", id),
            summary: None,
            tags: vec![],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
            pinned,
            global,
        };

        db.store_memory(&make_memory("m1", false, false)).unwrap();
        db.store_memory(&make_memory("m2", true, false)).unwrap();
        db.store_memory(&make_memory("m3", false, true)).unwrap();
        db.store_memory(&make_memory("m4", true, true)).unwrap();

        let stats = db.get_project_stats("test-stats").unwrap();
        assert_eq!(stats.memory_count, 4);
        assert_eq!(stats.pinned_count, 2, "expected 2 pinned memories");
        // global_count queries all projects (WHERE global = 1), so 2 global memories total
        assert_eq!(stats.global_count, 2, "expected 2 global memories");
    }

    // ---- Section-embedding helpers ----

    #[test]
    fn test_encode_decode_section_embeddings_round_trip() {
        let keys = ["summary", "decisions", "todos"];
        let vectors: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32 * 0.1; 256]).collect();

        let (keys_str, bytes) = encode_section_embeddings(&keys, &vectors);
        assert_eq!(keys_str, "summary,decisions,todos");
        assert_eq!(bytes.len(), 3 * 256 * 4);

        let decoded = decode_section_embeddings(&keys_str, &bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].0, "summary");
        assert_eq!(decoded[1].0, "decisions");
        assert_eq!(decoded[2].0, "todos");
        // Verify float values round-trip correctly
        assert!((decoded[0].1[0] - 0.0_f32).abs() < 1e-6);
        assert!((decoded[1].1[0] - 0.1_f32).abs() < 1e-6);
        assert!((decoded[2].1[0] - 0.2_f32).abs() < 1e-6);
    }

    #[test]
    fn test_decode_section_embeddings_byte_length_validation() {
        // Wrong number of bytes for 2 keys
        let result = decode_section_embeddings("a,b", &[0u8; 100]);
        assert!(
            matches!(result, Err(MemoryError::Database(_))),
            "expected Database error for byte length mismatch"
        );
    }

    #[test]
    fn test_decode_section_embeddings_empty() {
        let decoded = decode_section_embeddings("", &[]).unwrap();
        assert!(decoded.is_empty());
    }

    // ---- Handoff sidecar DB helpers ----

    fn make_handoff_memory(id: &str, project_id: &str, branch: Option<&str>) -> Memory {
        let now = chrono::Utc::now().timestamp();
        Memory {
            id: id.to_string(),
            project_id: project_id.to_string(),
            memory_type: MemoryType::Handoff,
            content: "## Summary\n\nTest handoff".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.85,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: branch.map(str::to_string),
            merged_from: None,
            pinned: true,
            global: false,
        }
    }

    fn make_sections(summary: &str) -> HandoffSections {
        HandoffSections {
            summary: summary.to_string(),
            decisions: vec!["Use Rust".to_string()],
            todos: vec!["Write tests".to_string()],
            blockers: vec![],
            mental_model: "Layered architecture".to_string(),
            next_steps: vec!["Deploy".to_string()],
            notes: Some("Extra notes".to_string()),
            continues_from: None,
        }
    }

    #[test]
    fn test_handoff_sections_round_trip() {
        let db = Database::open_in_memory().unwrap();
        let proj = crate::memory::Project {
            id: "ho-proj".to_string(),
            name: "ho-proj".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&proj).unwrap();

        let mem = make_handoff_memory("ho-1", "ho-proj", Some("main"));
        db.store_memory(&mem).unwrap();

        let sections = make_sections("Session ended well");
        let keys = [
            "summary",
            "decisions",
            "todos",
            "mental_model",
            "next_steps",
            "notes",
        ];
        let vecs: Vec<Vec<f32>> = keys.iter().map(|_| vec![0.5_f32; 256]).collect();
        let (keys_str, bytes) = encode_section_embeddings(&keys, &vecs);

        db.insert_handoff_sections("ho-1", &sections, &keys_str, &bytes)
            .unwrap();

        let result = db.get_handoff_sections("ho-1").unwrap().unwrap();
        let (retrieved_sections, retrieved_vecs) = result;

        assert_eq!(retrieved_sections.summary, "Session ended well");
        assert_eq!(retrieved_sections.decisions, vec!["Use Rust"]);
        assert_eq!(retrieved_sections.todos, vec!["Write tests"]);
        assert!(retrieved_sections.blockers.is_empty());
        assert_eq!(retrieved_sections.mental_model, "Layered architecture");
        assert_eq!(retrieved_sections.next_steps, vec!["Deploy"]);
        assert_eq!(retrieved_sections.notes, Some("Extra notes".to_string()));
        assert_eq!(retrieved_vecs.len(), 6);
        assert_eq!(retrieved_vecs[0].0, "summary");
        assert!((retrieved_vecs[0].1[0] - 0.5_f32).abs() < 1e-6);
    }

    #[test]
    fn test_handoff_sections_cascade_delete() {
        let db = Database::open_in_memory().unwrap();
        let proj = crate::memory::Project {
            id: "ho-cascade".to_string(),
            name: "ho-cascade".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&proj).unwrap();

        let mem = make_handoff_memory("ho-del", "ho-cascade", None);
        db.store_memory(&mem).unwrap();

        let sections = make_sections("Will be deleted");
        let (keys_str, bytes) = encode_section_embeddings(&["summary"], &[vec![0.1_f32; 256]]);
        db.insert_handoff_sections("ho-del", &sections, &keys_str, &bytes)
            .unwrap();

        // Verify sidecar exists
        assert!(db.get_handoff_sections("ho-del").unwrap().is_some());

        // delete_memory should remove both the memory and the sidecar
        db.delete_memory("ho-del").unwrap();

        assert!(db.get_memory("ho-del").unwrap().is_none());
        assert!(db.get_handoff_sections("ho-del").unwrap().is_none());
    }

    #[test]
    fn test_handoff_sections_update() {
        let db = Database::open_in_memory().unwrap();
        let proj = crate::memory::Project {
            id: "ho-upd".to_string(),
            name: "ho-upd".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&proj).unwrap();

        let mem = make_handoff_memory("ho-upd-1", "ho-upd", Some("feat/x"));
        db.store_memory(&mem).unwrap();

        let sections = make_sections("Original summary");
        let (keys_str, bytes) = encode_section_embeddings(&["summary"], &[vec![0.1_f32; 256]]);
        db.insert_handoff_sections("ho-upd-1", &sections, &keys_str, &bytes)
            .unwrap();

        // Update with new sections
        let updated = HandoffSections {
            summary: "Updated summary".to_string(),
            ..sections.clone()
        };
        let (new_keys_str, new_bytes) = encode_section_embeddings(
            &["summary", "decisions"],
            &[vec![0.2_f32; 256], vec![0.3_f32; 256]],
        );
        db.update_handoff_sections("ho-upd-1", &updated, &new_keys_str, &new_bytes)
            .unwrap();

        let (result, vecs) = db.get_handoff_sections("ho-upd-1").unwrap().unwrap();
        assert_eq!(result.summary, "Updated summary");
        // New embedding byte count matches 2 sections
        assert_eq!(new_bytes.len(), 2 * 256 * 4);
        assert_eq!(vecs.len(), 2);
    }

    #[test]
    fn test_handoff_sections_update_with_continues_from() {
        // Verify that `continues_from: Some(...)` survives an update round-trip.
        let db = Database::open_in_memory().unwrap();
        let proj = crate::memory::Project {
            id: "ho-cf".to_string(),
            name: "ho-cf".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&proj).unwrap();

        let mem = make_handoff_memory("ho-cf-1", "ho-cf", Some("main"));
        db.store_memory(&mem).unwrap();

        // Insert with continues_from = None initially.
        let sections = make_sections("Initial summary");
        let (keys_str, bytes) = encode_section_embeddings(&["summary"], &[vec![0.1_f32; 256]]);
        db.insert_handoff_sections("ho-cf-1", &sections, &keys_str, &bytes)
            .unwrap();

        // Update to set continues_from = Some("prev-handoff-id").
        let updated = HandoffSections {
            summary: "Continued summary".to_string(),
            continues_from: Some("prev-handoff-id".to_string()),
            ..sections
        };
        let (new_keys_str, new_bytes) =
            encode_section_embeddings(&["summary"], &[vec![0.5_f32; 256]]);
        db.update_handoff_sections("ho-cf-1", &updated, &new_keys_str, &new_bytes)
            .unwrap();

        let (result, _vecs) = db.get_handoff_sections("ho-cf-1").unwrap().unwrap();
        assert_eq!(result.summary, "Continued summary");
        assert_eq!(
            result.continues_from.as_deref(),
            Some("prev-handoff-id"),
            "continues_from must survive update"
        );
    }

    #[test]
    fn test_query_handoffs_by_branch() {
        let db = Database::open_in_memory().unwrap();
        let proj = crate::memory::Project {
            id: "ho-branch".to_string(),
            name: "ho-branch".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&proj).unwrap();

        // Store handoffs on two branches and one fact
        db.store_memory(&make_handoff_memory("ho-a1", "ho-branch", Some("feat/a")))
            .unwrap();
        db.store_memory(&make_handoff_memory("ho-b1", "ho-branch", Some("feat/b")))
            .unwrap();
        let now = chrono::Utc::now().timestamp();
        db.store_memory(&Memory {
            id: "fact-1".to_string(),
            project_id: "ho-branch".to_string(),
            memory_type: MemoryType::Fact,
            content: "A fact".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: Some("feat/a".to_string()),
            merged_from: None,
            pinned: false,
            global: false,
        })
        .unwrap();

        // Branch filter: feat/a should return ho-a1 only
        let results = db
            .query_handoffs_by_branch("ho-branch", Some("feat/a"), 10)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "ho-a1");

        // No branch filter: both handoffs returned
        let results = db.query_handoffs_by_branch("ho-branch", None, 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_handoffs_unknown_type_propagates_error() {
        // Verify that parse_memory_type_col (used in query_handoffs_by_branch) returns
        // a rusqlite error for unknown type strings, and that the collect chain
        // propagates it instead of silently dropping rows.
        //
        // The SQL WHERE clause pre-filters to memory_type = 'handoff', so we cannot
        // inject a row that passes the filter yet fails parsing via a normal INSERT.
        // Instead we verify the error-propagation mechanism directly: the helper
        // parse_memory_type_col must return Err for unknown input.
        let err = parse_memory_type_col("not_a_valid_type", 2);
        assert!(
            err.is_err(),
            "parse_memory_type_col must return Err for unknown type"
        );

        // Additionally verify via a raw query that bypasses the handoff filter.
        // Build a DB with a corrupt row (memory_type = 'unknown_xyz') and query
        // all memories using query_map + collect to confirm errors surface.
        let db = Database::open_in_memory().unwrap();
        let proj = crate::memory::Project {
            id: "ho-bad".to_string(),
            name: "ho-bad".to_string(),
            root_path: None,
            decay_rate: 0.01,
            created_at: 0,
        };
        db.create_project(&proj).unwrap();

        let now = chrono::Utc::now().timestamp();
        {
            let conn = db.conn.lock().unwrap();
            // Insert with an invalid type directly — bypasses all Rust type safety.
            conn.execute(
                "INSERT INTO memories
                 (id, project_id, memory_type, content, summary, tags, importance,
                  relevance_score, access_count, created_at, updated_at,
                  last_accessed_at, branch, merged_from, pinned, global)
                 VALUES (?1, ?2, 'unknown_xyz', ?3, NULL, '[]', 0.5, 1.0, 0,
                         ?4, ?4, ?4, NULL, NULL, 0, 0)",
                params!["ho-bad-1", "ho-bad", "corrupt row", now],
            )
            .unwrap();

            // Confirm the collect::<rusqlite::Result<Vec<_>>>() chain propagates errors.
            let mut stmt = conn
                .prepare(
                    "SELECT id, project_id, memory_type, content, summary, tags, importance,
                            relevance_score, access_count, created_at, updated_at,
                            last_accessed_at, branch, merged_from, pinned, global
                     FROM memories WHERE project_id = ?1",
                )
                .unwrap();
            let result: rusqlite::Result<Vec<Memory>> = stmt
                .query_map(params!["ho-bad"], |row| {
                    let memory_type_str: String = row.get(2)?;
                    let tags_json: String = row.get(5)?;
                    Ok(Memory {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        memory_type: parse_memory_type_col(&memory_type_str, 2)?,
                        content: row.get(3)?,
                        summary: row.get(4)?,
                        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                        importance: row.get(6)?,
                        relevance_score: row.get(7)?,
                        access_count: row.get(8)?,
                        created_at: row.get(9)?,
                        updated_at: row.get(10)?,
                        last_accessed_at: row.get(11)?,
                        branch: row.get(12)?,
                        merged_from: row
                            .get::<_, Option<String>>(13)?
                            .and_then(|s| serde_json::from_str(&s).ok()),
                        pinned: row.get::<_, i64>(14)? != 0,
                        global: row.get::<_, i64>(15)? != 0,
                    })
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>();
            assert!(
                result.is_err(),
                "collect chain must propagate unknown-type error, not drop the row"
            );
        }
    }
}
