//! Whole-store and incremental export getters.
//!
//! `query_memories` (and the `get_all_*_for_project` helpers built on it) cap results at
//! 10000 rows, which is the right default for interactive retrieval but silently
//! truncates a full-store export. These getters are unbounded and take an optional
//! project filter (`None` = every project) and an optional `since` watermark, so a
//! single set of queries serves both `engram-cli export` (project-scoped, `since = None`)
//! and `export --all-projects [--since T]`.
//!
//! `since` filtering is inclusive (`>= since`) and keyed, for every getter, on the owning
//! memory's `updated_at` rather than a sidecar's own timestamp: a relationship's
//! `created_at` never changes after insert but a newly-arrived endpoint still needs it
//! re-sent, and a status row's `marked_at` is only `<=` the memory's `updated_at` (not
//! `==` — a later unrelated content edit advances one without the other), so filtering a
//! sidecar on its own timestamp can silently drop it from an incremental export while the
//! memory it belongs to is still included. Keying every getter on `updated_at` (relationships:
//! `OR` either endpoint's) avoids that class of bug uniformly instead of case by case.

use std::collections::HashMap;

use rusqlite::params;

use crate::error::MemoryError;
use crate::memory::{Memory, Relationship, TodoItem};

use super::Database;
use super::status::StatusRowMap;
use super::todos::{TODO_COLUMNS, map_todo_row};
use super::util::{MEMORY_COLUMNS, map_memory_row};

impl Database {
    /// Every memory, optionally restricted to one project and to rows updated at or
    /// after `since`. Unlike `get_all_memories_for_project`, there is no row cap.
    #[allow(dead_code)] // Used by engram-cli export/sync, not the engram MCP-server binary
    pub fn export_memories(
        &self,
        project: Option<&str>,
        since: Option<i64>,
    ) -> Result<Vec<Memory>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut sql = format!("SELECT {MEMORY_COLUMNS} FROM memories WHERE 1=1");
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(p) = project {
            args.push(Box::new(p.to_string()));
            sql.push_str(&format!(" AND project_id = ?{}", args.len()));
        }
        if let Some(s) = since {
            args.push(Box::new(s));
            sql.push_str(&format!(" AND updated_at >= ?{}", args.len()));
        }
        sql.push_str(" ORDER BY id");
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), map_memory_row)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Every relationship whose source memory is in `project` (when given), included when
    /// its own `created_at >= since` or either endpoint's `updated_at >= since`.
    #[allow(dead_code)] // Used by engram-cli export/sync, not the engram MCP-server binary
    pub fn export_relationships(
        &self,
        project: Option<&str>,
        since: Option<i64>,
    ) -> Result<Vec<Relationship>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut sql = String::from(
            "SELECT r.id, r.source_id, r.target_id, r.relation_type, r.strength, r.created_at
             FROM relationships r
             JOIN memories src ON r.source_id = src.id
             JOIN memories tgt ON r.target_id = tgt.id
             WHERE 1=1",
        );
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(p) = project {
            args.push(Box::new(p.to_string()));
            sql.push_str(&format!(" AND src.project_id = ?{}", args.len()));
        }
        if let Some(s) = since {
            args.push(Box::new(s));
            let idx = args.len();
            sql.push_str(&format!(
                " AND (r.created_at >= ?{idx} OR src.updated_at >= ?{idx} OR tgt.updated_at >= ?{idx})"
            ));
        }
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str
                    .parse()
                    .unwrap_or(crate::memory::RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Every stored embedding, optionally restricted to one project and to memories
    /// updated at or after `since`.
    #[allow(dead_code)] // Used by engram-cli export/sync, not the engram MCP-server binary
    pub fn export_embeddings(
        &self,
        project: Option<&str>,
        since: Option<i64>,
    ) -> Result<Vec<(String, Vec<f32>)>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut sql = String::from(
            "SELECT e.memory_id, e.vector FROM embeddings e
             JOIN memories m ON e.memory_id = m.id
             WHERE 1=1",
        );
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(p) = project {
            args.push(Box::new(p.to_string()));
            sql.push_str(&format!(" AND m.project_id = ?{}", args.len()));
        }
        if let Some(s) = since {
            args.push(Box::new(s));
            sql.push_str(&format!(" AND m.updated_at >= ?{}", args.len()));
        }
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            let memory_id: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            let vector: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();
            Ok((memory_id, vector))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Every `memory_status` row (dead or since-revived), optionally restricted to one
    /// project and to rows whose owning memory was updated at or after `since`.
    ///
    /// Filtered on the memory's own `updated_at`, not the status row's `marked_at`: a
    /// later, unrelated content edit advances `updated_at` without touching `marked_at`
    /// (`set_dead_at` only enforces `marked_at <= updated_at`, not equality), so filtering
    /// on `marked_at` alone can let `since` land strictly between the two and silently
    /// drop the status row from an incremental export while the memory itself is still
    /// included — reviving a dead memory (or losing a revival) on the importer's side.
    #[allow(dead_code)] // Used by engram-cli export/sync, not the engram MCP-server binary
    pub fn export_status_rows(
        &self,
        project: Option<&str>,
        since: Option<i64>,
    ) -> Result<StatusRowMap, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut sql = String::from(
            "SELECT s.memory_id, s.dead, s.reason, s.marked_at
             FROM memory_status s
             JOIN memories m ON s.memory_id = m.id
             WHERE 1=1",
        );
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(p) = project {
            args.push(Box::new(p.to_string()));
            sql.push_str(&format!(" AND m.project_id = ?{}", args.len()));
        }
        if let Some(s) = since {
            args.push(Box::new(s));
            sql.push_str(&format!(" AND m.updated_at >= ?{}", args.len()));
        }
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                (
                    row.get::<_, i64>(1)? != 0,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, i64>(3)?,
                ),
            ))
        })?;
        Ok(rows.flatten().collect())
    }

    /// Every `todo_items` row, optionally restricted to one project and to todos whose
    /// owning memory was updated at or after `since`.
    #[allow(dead_code)] // Used by engram-cli export/sync, not the engram MCP-server binary
    pub fn export_todo_rows(
        &self,
        project: Option<&str>,
        since: Option<i64>,
    ) -> Result<Vec<TodoItem>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut sql = format!(
            "SELECT {TODO_COLUMNS} FROM todo_items t
             JOIN memories m ON m.id = t.memory_id
             WHERE 1=1"
        );
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(p) = project {
            args.push(Box::new(p.to_string()));
            sql.push_str(&format!(" AND m.project_id = ?{}", args.len()));
        }
        if let Some(s) = since {
            args.push(Box::new(s));
            sql.push_str(&format!(" AND m.updated_at >= ?{}", args.len()));
        }
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), map_todo_row)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// `(pull_watermark, push_watermark)` for a remote, `(0, 0)` for one never synced.
    #[allow(dead_code)] // Used by engram-cli sync, not the engram MCP-server binary
    pub fn get_sync_state(&self, remote: &str) -> Result<(i64, i64), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let result = conn.query_row(
            "SELECT pull_watermark, push_watermark FROM sync_state WHERE remote = ?1",
            params![remote],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        );
        match result {
            Ok(watermarks) => Ok(watermarks),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok((0, 0)),
            Err(e) => Err(e.into()),
        }
    }

    /// Advance the pull watermark for a remote. Independent of `set_push_watermark`: a
    /// failed push must never retroactively touch an already-committed pull watermark,
    /// and vice versa.
    #[allow(dead_code)] // Used by engram-cli sync, not the engram MCP-server binary
    pub fn set_pull_watermark(
        &self,
        remote: &str,
        watermark: i64,
        now: i64,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO sync_state (remote, pull_watermark, push_watermark, last_pull_at)
             VALUES (?1, ?2, 0, ?3)
             ON CONFLICT(remote) DO UPDATE SET pull_watermark = ?2, last_pull_at = ?3",
            params![remote, watermark, now],
        )?;
        Ok(())
    }

    /// Advance the push watermark for a remote. Independent of `set_pull_watermark`.
    #[allow(dead_code)] // Used by engram-cli sync, not the engram MCP-server binary
    pub fn set_push_watermark(
        &self,
        remote: &str,
        watermark: i64,
        now: i64,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO sync_state (remote, pull_watermark, push_watermark, last_push_at)
             VALUES (?1, 0, ?2, ?3)
             ON CONFLICT(remote) DO UPDATE SET push_watermark = ?2, last_push_at = ?3",
            params![remote, watermark, now],
        )?;
        Ok(())
    }

    /// Record that a memory's current state (as of `origin_updated_at`) arrived unchanged
    /// from `remote`, so the push half of a later `sync` doesn't echo it straight back.
    /// Called only by `sync`'s in-process pull import, never by a plain `engram-cli import`.
    #[allow(dead_code)] // Used by engram-cli sync, not the engram MCP-server binary
    pub fn set_memory_origin(
        &self,
        memory_id: &str,
        remote: &str,
        origin_updated_at: i64,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memory_origin (memory_id, remote, origin_updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(memory_id) DO UPDATE SET remote = ?2, origin_updated_at = ?3",
            params![memory_id, remote, origin_updated_at],
        )?;
        Ok(())
    }

    /// Among `candidates` (memory id, its current `updated_at`), the ids whose
    /// `memory_origin` row says they arrived from `remote` and still hold exactly that
    /// state — i.e. nothing has edited them locally since. `sync`'s push half excludes
    /// these from what it sends, since sending them back would just echo `remote`'s own
    /// content to itself. A later local edit always advances `updated_at` past the pinned
    /// `origin_updated_at`, so it drops out of this set on its own — no explicit clearing.
    #[allow(dead_code)] // Used by engram-cli sync, not the engram MCP-server binary
    pub fn origin_unchanged_since_pull(
        &self,
        remote: &str,
        candidates: &[(String, i64)],
    ) -> Result<std::collections::HashSet<String>, MemoryError> {
        if candidates.is_empty() {
            return Ok(std::collections::HashSet::new());
        }
        let conn = self.conn.lock().unwrap();
        // One query for the whole remote, not one placeholder per candidate: memory_origin
        // is bounded by what has ever been pulled from `remote`, which is unrelated to how
        // many candidates the push query found, so a per-candidate IN(...) list needlessly
        // risks SQLite's ~32k bound-variable limit on a store with enough new content.
        let mut stmt = conn
            .prepare("SELECT memory_id, origin_updated_at FROM memory_origin WHERE remote = ?1")?;
        let origins: HashMap<String, i64> = stmt
            .query_map(params![remote], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<rusqlite::Result<HashMap<_, _>>>()?;
        Ok(candidates
            .iter()
            .filter(|(id, updated_at)| origins.get(id) == Some(updated_at))
            .map(|(id, _)| id.clone())
            .collect())
    }
}
