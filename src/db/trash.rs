//! Snapshots of memories removed or overwritten by destructive operations.
//!
//! Every path that destroys memory content writes a snapshot here first, so that a
//! delete, a dedup merge, or a content-replacing update can be undone. Entries are
//! swept after a retention window; see [`Database::sweep_trash`].

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::error::MemoryError;
use crate::memory::{Memory, RelationType, Relationship};

use super::Database;
use super::util::{MEMORY_COLUMNS, map_memory_row};

/// Operation that trashed a memory. Recorded verbatim so a restore can explain itself.
pub const OP_DELETE: &str = "delete";
pub const OP_MERGE: &str = "merge";
pub const OP_UPDATE: &str = "update";
pub const OP_PRUNE: &str = "prune";
pub const OP_WIPE: &str = "wipe";

/// JSON stored in `memory_trash.payload`.
#[derive(Serialize, Deserialize)]
struct TrashPayload {
    memory: Memory,
    /// Edges touching this memory at snapshot time. `ON DELETE CASCADE` removes these
    /// when the memory goes, so they have to be captured to make a restore complete.
    #[serde(default)]
    relationships: Vec<Relationship>,
}

/// One recoverable snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrashEntry {
    pub trash_id: i64,
    /// One of the `OP_*` constants.
    pub op: String,
    pub trashed_at: i64,
    pub memory: Memory,
    pub relationships: Vec<Relationship>,
}

/// Read every edge with this memory at either end.
fn snapshot_relationships(
    conn: &Connection,
    memory_id: &str,
) -> Result<Vec<Relationship>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, source_id, target_id, relation_type, strength, created_at
         FROM relationships WHERE source_id = ?1 OR target_id = ?1",
    )?;
    let rows = stmt.query_map(params![memory_id], |row| {
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

/// Write a snapshot of `memory_id` to the trash.
///
/// Takes a `&Connection` so it can run inside an open transaction (`Transaction`
/// dereferences to `Connection`), keeping the snapshot and the destructive statement
/// in one atomic unit. Returns `false` if there was no such memory to snapshot.
pub(super) fn trash_memory_in(
    conn: &Connection,
    memory_id: &str,
    op: &str,
    now: i64,
) -> Result<bool, MemoryError> {
    let sql = format!("SELECT {MEMORY_COLUMNS} FROM memories WHERE id = ?1");
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(params![memory_id])?;
    let Some(row) = rows.next()? else {
        return Ok(false);
    };
    let memory = map_memory_row(row)?;
    drop(rows);
    drop(stmt);

    let relationships = snapshot_relationships(conn, memory_id)?;

    let (embedding, model_version): (Option<Vec<u8>>, Option<String>) = conn
        .query_row(
            "SELECT vector, model_version FROM embeddings WHERE memory_id = ?1",
            params![memory_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap_or((None, None));

    let project_id = memory.project_id.clone();
    let payload = serde_json::to_string(&TrashPayload {
        memory,
        relationships,
    })?;

    conn.execute(
        "INSERT INTO memory_trash
            (memory_id, project_id, op, payload, embedding, model_version, trashed_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            memory_id,
            project_id,
            op,
            payload,
            embedding,
            model_version,
            now
        ],
    )?;

    Ok(true)
}

impl Database {
    /// Snapshot a memory to the trash outside of any caller-held transaction.
    pub fn trash_memory(&self, memory_id: &str, op: &str) -> Result<bool, MemoryError> {
        let conn = self.conn.lock().unwrap();
        trash_memory_in(&conn, memory_id, op, chrono::Utc::now().timestamp())
    }

    /// Most recent snapshots for a project, newest first.
    pub fn list_trash(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<TrashEntry>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT trash_id, op, payload, trashed_at FROM memory_trash
             WHERE project_id = ?1
             ORDER BY trashed_at DESC, trash_id DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![project_id, limit as i64], |row| {
            let payload: String = row.get(2)?;
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                payload,
                row.get::<_, i64>(3)?,
            ))
        })?;

        let mut entries = Vec::new();
        for (trash_id, op, payload, trashed_at) in rows.flatten() {
            let Ok(parsed) = serde_json::from_str::<TrashPayload>(&payload) else {
                continue;
            };
            entries.push(TrashEntry {
                trash_id,
                op,
                trashed_at,
                memory: parsed.memory,
                relationships: parsed.relationships,
            });
        }
        Ok(entries)
    }

    /// Look up a single snapshot by its trash id.
    pub fn get_trash_entry(&self, trash_id: i64) -> Result<Option<TrashEntry>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT trash_id, op, payload, trashed_at FROM memory_trash WHERE trash_id = ?1",
        )?;
        let mut rows = stmt.query(params![trash_id])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };
        let payload: String = row.get(2)?;
        let parsed: TrashPayload = serde_json::from_str(&payload)?;
        Ok(Some(TrashEntry {
            trash_id: row.get(0)?,
            op: row.get(1)?,
            trashed_at: row.get(3)?,
            memory: parsed.memory,
            relationships: parsed.relationships,
        }))
    }

    /// Newest snapshot for a given memory id, whatever operation produced it.
    pub fn latest_trash_for_memory(
        &self,
        memory_id: &str,
    ) -> Result<Option<TrashEntry>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let trash_id: Option<i64> = conn
            .query_row(
                "SELECT trash_id FROM memory_trash WHERE memory_id = ?1
                 ORDER BY trashed_at DESC, trash_id DESC LIMIT 1",
                params![memory_id],
                |row| row.get(0),
            )
            .ok();
        drop(conn);
        match trash_id {
            Some(id) => self.get_trash_entry(id),
            None => Ok(None),
        }
    }

    /// Put a trashed memory back.
    ///
    /// Restores the memory row, its embedding, and every snapshotted edge whose other
    /// endpoint still exists; edges to memories that are themselves gone are skipped and
    /// counted. If the memory id is currently in use (the snapshot came from an update),
    /// the live row is snapshotted first and then replaced, so a restore is itself
    /// undoable. Consumes the trash entry on success.
    pub fn restore_trash_entry(&self, trash_id: i64) -> Result<RestoreOutcome, MemoryError> {
        let entry = self
            .get_trash_entry(trash_id)?
            .ok_or_else(|| MemoryError::NotFound(format!("trash entry {trash_id}")))?;

        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().timestamp();

        // If a live row occupies this id, snapshot it before overwriting.
        let overwrote_existing = trash_memory_in(&tx, &entry.memory.id, OP_UPDATE, now)?;

        let m = &entry.memory;
        let tags_json = serde_json::to_string(&m.tags)?;
        let merged_from_json: Option<String> = m
            .merged_from
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let artifacts_json: Option<String> = m
            .external_artifacts
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(serde_json::to_string)
            .transpose()?;

        tx.execute(
            "INSERT OR REPLACE INTO memories
                (id, project_id, memory_type, content, summary, tags, importance,
                 relevance_score, access_count, created_at, updated_at, last_accessed_at,
                 branch, merged_from, pinned, global, external_artifacts)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                m.id,
                m.project_id,
                m.memory_type.as_str(),
                m.content,
                m.summary,
                tags_json,
                m.importance,
                m.relevance_score,
                m.access_count,
                m.created_at,
                m.updated_at,
                m.last_accessed_at,
                m.branch,
                merged_from_json,
                m.pinned as i64,
                m.global as i64,
                artifacts_json,
            ],
        )?;

        // Restore the embedding so the memory is searchable again without re-embedding.
        let embedding: Option<Vec<u8>> = tx
            .query_row(
                "SELECT embedding FROM memory_trash WHERE trash_id = ?1",
                params![trash_id],
                |row| row.get(0),
            )
            .unwrap_or(None);
        let model_version: Option<String> = tx
            .query_row(
                "SELECT model_version FROM memory_trash WHERE trash_id = ?1",
                params![trash_id],
                |row| row.get(0),
            )
            .unwrap_or(None);
        if let (Some(bytes), Some(version)) = (embedding, model_version) {
            tx.execute(
                "INSERT OR REPLACE INTO embeddings (memory_id, vector, model_version)
                 VALUES (?1, ?2, ?3)",
                params![m.id, bytes, version],
            )?;
        }

        // Restore edges whose other endpoint still exists.
        let mut edges_restored = 0usize;
        let mut edges_dropped = 0usize;
        for rel in &entry.relationships {
            let other = if rel.source_id == m.id {
                &rel.target_id
            } else {
                &rel.source_id
            };
            let other_exists: i64 = tx
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE id = ?1",
                    params![other],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            if other_exists == 0 {
                edges_dropped += 1;
                continue;
            }
            tx.execute(
                "INSERT OR IGNORE INTO relationships
                    (id, source_id, target_id, relation_type, strength, created_at)
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
            edges_restored += 1;
        }

        tx.execute(
            "DELETE FROM memory_trash WHERE trash_id = ?1",
            params![trash_id],
        )?;

        tx.commit()?;

        Ok(RestoreOutcome {
            memory: entry.memory,
            op: entry.op,
            overwrote_existing,
            edges_restored,
            edges_dropped,
        })
    }

    /// Drop trash entries older than the retention window. `retention_days <= 0` keeps
    /// everything.
    #[allow(dead_code)] // Called by the decay background job in main.rs
    pub fn sweep_trash(&self, retention_days: i64) -> Result<usize, MemoryError> {
        if retention_days <= 0 {
            return Ok(0);
        }
        let conn = self.conn.lock().unwrap();
        let cutoff = chrono::Utc::now().timestamp() - retention_days * 86400;
        let removed = conn.execute(
            "DELETE FROM memory_trash WHERE trashed_at < ?1",
            params![cutoff],
        )?;
        Ok(removed)
    }

    /// Number of entries currently in the trash for a project.
    pub fn count_trash(&self, project_id: &str) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_trash WHERE project_id = ?1",
            params![project_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}

/// What a restore actually put back.
#[derive(Debug, Clone)]
pub struct RestoreOutcome {
    pub memory: Memory,
    /// The operation that had trashed it.
    pub op: String,
    /// Whether a live memory with the same id was replaced (and itself snapshotted).
    pub overwrote_existing: bool,
    pub edges_restored: usize,
    /// Edges skipped because the memory at the other end no longer exists.
    pub edges_dropped: usize,
}
