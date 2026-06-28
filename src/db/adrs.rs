use std::str::FromStr;

use chrono::Utc;
use rusqlite::params;

use crate::error::MemoryError;
use crate::memory::{AdrSections, AdrStatus};

use super::Database;

#[allow(dead_code)] // Used by ADR MCP tools (Phase 3)
impl Database {
    /// Store an ADR memory, its embedding, and the `adr_sections` sidecar row atomically.
    ///
    /// Content is rendered internally from `sections`; the caller does not supply it.
    ///
    /// When `supersede_old_id` is `Some(old_id)`, the old ADR's status is flipped to
    /// `superseded` and a `Supersedes` relationship edge (new → old) is inserted, all
    /// within the same transaction as the new ADR insertion.  This prevents orphan ADRs:
    /// either the entire create+supersede succeeds or nothing is committed.
    ///
    /// Number assignment: an `IN-transaction MAX+1` query is used. This is safe within a
    /// single process because the `Database` connection is behind `Arc<Mutex>`. For
    /// multi-process writers the `UNIQUE(project_id, adr_number)` constraint on
    /// `adr_sections` will reject a colliding commit — the losing writer must retry.
    #[allow(clippy::too_many_arguments)]
    pub fn store_adr_atomic(
        &self,
        id: &str,
        project_id: &str,
        sections: &AdrSections,
        status: AdrStatus,
        importance: f64,
        pinned: bool,
        embedding: &[f32],
        model_version: &str,
        created_at: i64,
        supersede_old_id: Option<&str>,
    ) -> Result<(String, u32), MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        // Allocate the next ADR number within this project.
        let next: i64 = tx.query_row(
            "SELECT COALESCE(MAX(adr_number),0)+1 FROM adr_sections WHERE project_id = ?1",
            params![project_id],
            |r| r.get(0),
        )?;
        debug_assert!(next > 0 && next <= u32::MAX as i64);

        // Render the canonical markdown content.
        let content = sections.render_markdown(next as u32, status, created_at);

        // Insert the memory row.
        tx.execute(
            "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, \
             relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, \
             merged_from, pinned, global)
             VALUES (?1, ?2, 'adr', ?3, NULL, '[]', ?4, 1.0, 0, ?5, ?5, ?5, NULL, NULL, ?6, 0)",
            params![
                id,
                project_id,
                content,
                importance,
                created_at,
                pinned as i64,
            ],
        )?;

        // Insert the embedding row.
        let vector_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        tx.execute(
            "INSERT OR REPLACE INTO embeddings (memory_id, vector, model_version) VALUES (?1, ?2, ?3)",
            params![id, vector_bytes, model_version],
        )?;

        // Insert the adr_sections sidecar row.
        tx.execute(
            "INSERT INTO adr_sections \
             (memory_id, project_id, adr_number, status, title, context, decision, consequences, \
              created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)",
            params![
                id,
                project_id,
                next,
                status.as_str(),
                sections.title,
                sections.context,
                sections.decision,
                sections.consequences,
                created_at,
            ],
        )?;

        // If superseding an old ADR, flip its status and create the relationship edge
        // inside this same transaction so the whole operation is atomic.
        if let Some(old_id) = supersede_old_id {
            tx.execute(
                "UPDATE adr_sections SET status = 'superseded', updated_at = ?1 WHERE memory_id = ?2",
                params![created_at, old_id],
            )?;
            tx.execute(
                "UPDATE memories SET updated_at = ?1 WHERE id = ?2",
                params![created_at, old_id],
            )?;
            let rel_id = format!("rel_{}", uuid::Uuid::new_v4().simple());
            tx.execute(
                "INSERT OR REPLACE INTO relationships \
                 (id, source_id, target_id, relation_type, strength, created_at) \
                 VALUES (?1, ?2, ?3, 'supersedes', 1.0, ?4)",
                params![rel_id, id, old_id, created_at],
            )?;
        }

        tx.commit()?;
        Ok((id.to_string(), next as u32))
    }

    /// Fetch the sidecar data for an ADR memory.
    ///
    /// Returns `Ok(None)` when no sidecar row exists for the given `memory_id`.
    pub fn get_adr_sections(
        &self,
        memory_id: &str,
    ) -> Result<Option<(u32, AdrStatus, AdrSections)>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT adr_number, status, title, context, decision, consequences \
             FROM adr_sections WHERE memory_id = ?1",
        )?;
        let mut rows = stmt.query(params![memory_id])?;

        if let Some(row) = rows.next()? {
            let number: i64 = row.get(0)?;
            let status_str: String = row.get(1)?;
            let status = AdrStatus::from_str(&status_str)
                .map_err(|e| MemoryError::InvalidType(e.to_string()))?;
            let sections = AdrSections {
                title: row.get(2)?,
                context: row.get(3)?,
                decision: row.get(4)?,
                consequences: row.get(5)?,
            };
            Ok(Some((number as u32, status, sections)))
        } else {
            Ok(None)
        }
    }

    /// Return the `memory_id` for the ADR with the given number in a project.
    ///
    /// Returns `Ok(None)` when no ADR with that number exists.
    pub fn get_adr_by_number(
        &self,
        project_id: &str,
        number: u32,
    ) -> Result<Option<String>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT memory_id FROM adr_sections WHERE project_id = ?1 AND adr_number = ?2",
        )?;
        let mut rows = stmt.query(params![project_id, number as i64])?;

        if let Some(row) = rows.next()? {
            let memory_id: String = row.get(0)?;
            Ok(Some(memory_id))
        } else {
            Ok(None)
        }
    }

    /// Update the status of an ADR in both the sidecar table and the `memories` row.
    ///
    /// Touching `memories.updated_at` re-fires the FTS update trigger.
    /// Returns `true` when a row was updated, `false` when `memory_id` was not found.
    ///
    /// Transition validation is NOT performed here; it lives in the tool layer.
    pub fn update_adr_status(
        &self,
        memory_id: &str,
        new_status: AdrStatus,
    ) -> Result<bool, MemoryError> {
        let now = Utc::now().timestamp();
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        let rows_changed = tx.execute(
            "UPDATE adr_sections SET status = ?1, updated_at = ?2 WHERE memory_id = ?3",
            params![new_status.as_str(), now, memory_id],
        )?;

        tx.execute(
            "UPDATE memories SET updated_at = ?1 WHERE id = ?2",
            params![now, memory_id],
        )?;

        tx.commit()?;
        Ok(rows_changed > 0)
    }

    /// List ADRs for a project, optionally filtered by status.
    ///
    /// Returns `(adr_number, status, title, memory_id)` tuples ordered by `adr_number ASC`.
    /// Uses two prepared statements (with/without the status predicate) to avoid
    /// string-building the SQL at runtime.
    pub fn list_adrs(
        &self,
        project_id: &str,
        status_filter: Option<AdrStatus>,
    ) -> Result<Vec<(u32, AdrStatus, String, String)>, MemoryError> {
        let conn = self.conn.lock().unwrap();

        let rows: Vec<(u32, AdrStatus, String, String)> = match status_filter {
            None => {
                let mut stmt = conn.prepare(
                    "SELECT adr_number, status, title, memory_id \
                     FROM adr_sections WHERE project_id = ?1 \
                     ORDER BY adr_number ASC",
                )?;
                stmt.query_map(params![project_id], |row| {
                    let number: i64 = row.get(0)?;
                    let status_str: String = row.get(1)?;
                    let title: String = row.get(2)?;
                    let memory_id: String = row.get(3)?;
                    Ok((number, status_str, title, memory_id))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(MemoryError::from)?
                .into_iter()
                .map(|(number, status_str, title, memory_id)| {
                    AdrStatus::from_str(&status_str)
                        .map_err(|e| MemoryError::InvalidType(e.to_string()))
                        .map(|status| (number as u32, status, title, memory_id))
                })
                .collect::<Result<Vec<_>, _>>()?
            }
            Some(sf) => {
                let mut stmt = conn.prepare(
                    "SELECT adr_number, status, title, memory_id \
                     FROM adr_sections WHERE project_id = ?1 AND status = ?2 \
                     ORDER BY adr_number ASC",
                )?;
                stmt.query_map(params![project_id, sf.as_str()], |row| {
                    let number: i64 = row.get(0)?;
                    let status_str: String = row.get(1)?;
                    let title: String = row.get(2)?;
                    let memory_id: String = row.get(3)?;
                    Ok((number, status_str, title, memory_id))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(MemoryError::from)?
                .into_iter()
                .map(|(number, status_str, title, memory_id)| {
                    AdrStatus::from_str(&status_str)
                        .map_err(|e| MemoryError::InvalidType(e.to_string()))
                        .map(|status| (number as u32, status, title, memory_id))
                })
                .collect::<Result<Vec<_>, _>>()?
            }
        };

        Ok(rows)
    }

    /// Insert an `adr_sections` row with an explicit number (used by JSON import).
    ///
    /// Does NOT auto-allocate a number. Any `UNIQUE(project_id, adr_number)` violation
    /// propagates as `MemoryError::Database` so the importer can detect collisions.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_adr_sidecar(
        &self,
        memory_id: &str,
        project_id: &str,
        number: u32,
        status: AdrStatus,
        sections: &AdrSections,
        created_at: i64,
        updated_at: i64,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO adr_sections \
             (memory_id, project_id, adr_number, status, title, context, decision, consequences, \
              created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                memory_id,
                project_id,
                number as i64,
                status.as_str(),
                sections.title,
                sections.context,
                sections.decision,
                sections.consequences,
                created_at,
                updated_at,
            ],
        )?;
        Ok(())
    }
}
