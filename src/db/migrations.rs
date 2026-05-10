use rusqlite::params;

use crate::error::MemoryError;

use super::Database;

impl Database {
    /// Add branch column to memories table if it doesn't exist.
    pub(super) fn migrate_branch_column(&self) -> Result<(), MemoryError> {
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
    pub(super) fn migrate_fts(&self) -> Result<(), MemoryError> {
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

    pub(super) fn run_migrations(&self) -> Result<(), MemoryError> {
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
}
