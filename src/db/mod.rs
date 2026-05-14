use rusqlite::Connection;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::error::MemoryError;

mod util;

mod batch;
mod clusters;
mod embeddings;
mod handoffs;
mod memories;
mod migrations;
mod relationships;

pub use handoffs::encode_section_embeddings;

#[cfg(test)]
mod tests;

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
    pub(super) conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, MemoryError> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 3000;",
        )?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.initialize()?;
        Ok(db)
    }

    #[allow(dead_code)] // Used in tests
    pub fn open_in_memory() -> Result<Self, MemoryError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 3000;",
        )?;
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
}
