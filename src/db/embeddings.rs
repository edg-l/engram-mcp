use rusqlite::params;

use crate::error::MemoryError;

use super::Database;

// memories.id is the PRIMARY KEY of the memories table (see src/db/migrations.rs),
// so no additional index migration is needed for the IN-clause filtering below.

impl Database {
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

    /// Full-text BM25 search restricted to a specific set of memory IDs.
    ///
    /// Mirrors `keyword_search_with_branch` but constrains results to `ids`.
    /// Returns `(memory_id, bm25_score)` pairs (score positive, higher = better),
    /// ordered descending, capped at `limit`.
    ///
    /// Empty `ids` slice returns an empty vec without executing SQL.
    ///
    /// Used by `memory_context` to score cluster members in BM25 and Hybrid modes.
    /// N calls per `memory_context` invocation (one per selected cluster) — at the
    /// default 5 clusters this is acceptable.
    pub fn keyword_search_within_ids(
        &self,
        project_id: &str,
        query: &str,
        ids: &[String],
        limit: usize,
    ) -> Result<Vec<(String, f32)>, MemoryError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let escaped_query = Self::escape_fts_query(query);
        if escaped_query.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().unwrap();

        // Build dynamic IN clause; memories.id is the PRIMARY KEY so this is index-efficient.
        let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT m.id, -bm25(memories_fts) as score
             FROM memories_fts
             JOIN memories m ON memories_fts.rowid = m.rowid
             WHERE memories_fts MATCH ? AND m.project_id = ?
               AND m.id IN ({})
             ORDER BY score DESC
             LIMIT ?",
            placeholders
        );

        let mut stmt = conn.prepare(&sql)?;

        // Build the full parameter list: [escaped_query, project_id, ...ids, limit]
        let mut all_params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        all_params.push(Box::new(escaped_query));
        all_params.push(Box::new(project_id.to_owned()));
        for id in ids {
            all_params.push(Box::new(id.clone()));
        }
        all_params.push(Box::new(limit as i64));

        let param_refs: Vec<&dyn rusqlite::ToSql> = all_params.iter().map(|p| p.as_ref()).collect();

        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            let id: String = row.get(0)?;
            let score: f64 = row.get(1)?;
            Ok((id, score as f32))
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Memory, MemoryType};

    fn make_memory(id: &str, content: &str, project_id: &str) -> Memory {
        let now = chrono::Utc::now().timestamp();
        Memory {
            id: id.to_owned(),
            project_id: project_id.to_owned(),
            memory_type: MemoryType::Fact,
            content: content.to_owned(),
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
        }
    }

    #[test]
    fn keyword_search_within_ids_excludes_out_of_set() {
        let db = Database::open_in_memory().unwrap();

        let project_id = "test-kwsearch";
        db.get_or_create_project(project_id, project_id).unwrap();

        // Three memories all containing the word "xtokenalpha"
        let m1 = make_memory("id-1", "xtokenalpha first document context", project_id);
        let m2 = make_memory("id-2", "xtokenalpha second document context", project_id);
        let m3 = make_memory("id-3", "xtokenalpha third document context", project_id);
        db.store_memory(&m1).unwrap();
        db.store_memory(&m2).unwrap();
        db.store_memory(&m3).unwrap();

        // Restrict to only id-1 and id-2; id-3 must not appear even though it matches.
        let allowed = vec!["id-1".to_owned(), "id-2".to_owned()];
        let results = db
            .keyword_search_within_ids(project_id, "xtokenalpha", &allowed, 10)
            .unwrap();

        let returned_ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            returned_ids.contains(&"id-1") || returned_ids.contains(&"id-2"),
            "at least one allowed id should match; got: {returned_ids:?}"
        );
        assert!(
            !returned_ids.contains(&"id-3"),
            "id-3 must be excluded (not in allowed set); got: {returned_ids:?}"
        );
    }

    #[test]
    fn keyword_search_within_ids_empty_ids_returns_empty() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "test-empty";
        db.get_or_create_project(project_id, project_id).unwrap();

        let results = db
            .keyword_search_within_ids(project_id, "anything", &[], 10)
            .unwrap();
        assert!(results.is_empty(), "empty ids slice must return empty vec");
    }
}
