use rusqlite::{Connection, params};

use crate::error::MemoryError;
use crate::memory::{HandoffSections, Memory};

use super::Database;
use super::util::{MEMORY_COLUMNS, map_memory_row};

/// Bind the sidecar columns for `sections` in the order used by both the INSERT and the
/// UPDATE below. Keeping one builder means a new section touches one place, not four.
fn section_params(
    sections: &HandoffSections,
) -> Result<(String, String, String, String, String), MemoryError> {
    Ok((
        serde_json::to_string(&sections.decisions)?,
        serde_json::to_string(&sections.todos)?,
        serde_json::to_string(&sections.blockers)?,
        serde_json::to_string(&sections.tried)?,
        serde_json::to_string(&sections.next_steps)?,
    ))
}

/// Insert the sidecar row for a handoff. Shared by the standalone and transactional paths.
fn insert_sections_row(
    conn: &Connection,
    memory_id: &str,
    sections: &HandoffSections,
    section_keys: &str,
    section_embeddings: &[u8],
) -> Result<(), MemoryError> {
    let (decisions, todos, blockers, tried, next_steps) = section_params(sections)?;
    conn.execute(
        "INSERT INTO handoff_sections (
            memory_id, summary, decisions, todos, blockers, tried,
            mental_model, next_steps, notes, continues_from,
            section_embedding_keys, section_embeddings
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            memory_id,
            sections.summary,
            decisions,
            todos,
            blockers,
            tried,
            sections.mental_model,
            next_steps,
            sections.notes,
            sections.continues_from,
            section_keys,
            section_embeddings,
        ],
    )?;
    Ok(())
}

/// Overwrite the sidecar row for a handoff. Shared by the standalone and transactional paths.
fn update_sections_row(
    conn: &Connection,
    memory_id: &str,
    sections: &HandoffSections,
    section_keys: &str,
    section_embeddings: &[u8],
) -> Result<(), MemoryError> {
    let (decisions, todos, blockers, tried, next_steps) = section_params(sections)?;
    conn.execute(
        "UPDATE handoff_sections SET
            summary = ?2, decisions = ?3, todos = ?4, blockers = ?5, tried = ?6,
            mental_model = ?7, next_steps = ?8, notes = ?9, continues_from = ?10,
            section_embedding_keys = ?11, section_embeddings = ?12
         WHERE memory_id = ?1",
        params![
            memory_id,
            sections.summary,
            decisions,
            todos,
            blockers,
            tried,
            sections.mental_model,
            next_steps,
            sections.notes,
            sections.continues_from,
            section_keys,
            section_embeddings,
        ],
    )?;
    Ok(())
}

impl Database {
    // ============================================
    // Handoff sidecar helpers
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
        insert_sections_row(&conn, memory_id, sections, section_keys, section_embeddings)
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
        insert_sections_row(&tx, &memory.id, sections, section_keys, section_embeddings)?;

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
            "SELECT summary, decisions, todos, blockers, tried, mental_model, next_steps,
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
            let tried: Vec<String> =
                serde_json::from_str::<Vec<String>>(&row.get::<_, String>(4)?)?;
            let mental_model: String = row.get(5)?;
            let next_steps: Vec<String> =
                serde_json::from_str::<Vec<String>>(&row.get::<_, String>(6)?)?;
            let notes: Option<String> = row.get(7)?;
            let continues_from: Option<String> = row.get(8)?;
            let keys: String = row.get(9)?;
            let embedding_bytes: Vec<u8> = row.get(10)?;

            let sections = HandoffSections {
                summary,
                decisions,
                todos,
                blockers,
                tried,
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
        update_sections_row(&tx, &memory.id, sections, section_keys, section_embeddings)?;

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
        update_sections_row(&conn, memory_id, sections, section_keys, section_embeddings)
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
                let mut stmt = conn.prepare(&format!(
                    "SELECT {MEMORY_COLUMNS}
                     FROM memories
                     WHERE project_id = ?1 AND memory_type = 'handoff'
                     ORDER BY created_at DESC
                     LIMIT ?2"
                ))?;
                stmt.query_map(params![project_id, limit as i64], map_memory_row)?
                    .collect::<rusqlite::Result<Vec<_>>>()
                    .map_err(MemoryError::from)?
            }
            Some(b) => {
                let mut stmt = conn.prepare(&format!(
                    "SELECT {MEMORY_COLUMNS}
                     FROM memories
                     WHERE project_id = ?1 AND memory_type = 'handoff'
                       AND (branch IS NULL OR branch = ?3)
                     ORDER BY created_at DESC
                     LIMIT ?2"
                ))?;
                stmt.query_map(params![project_id, limit as i64, b], |row| {
                    map_memory_row(row)
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
// Section-embedding wire format helpers
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
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();
            (key.to_string(), vec)
        })
        .collect();

    Ok(result)
}
