use std::collections::{HashMap, HashSet};

use rusqlite::params;

use crate::decay::{ACCESS_REINFORCEMENT, RELEVANCE_CEILING};
use crate::error::MemoryError;
use crate::memory::{Memory, MemoryType, Project, ProjectStats, ProjectSummary};

use super::Database;
use super::util::{MEMORY_COLUMNS, map_memory_row};

/// Reinforcement applied when retrieval returns a memory.
///
/// Params: `?1` = now, `?2` = memory id, `?3` = bump, `?4` = ceiling.
const REINFORCE_SQL: &str = "UPDATE memories \
     SET access_count = access_count + 1, \
         last_accessed_at = ?1, \
         relevance_score = MIN(?4, relevance_score + ?3) \
     WHERE id = ?2";

impl Database {
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

    /// Whether a project is known to the store, either as a `projects` row or as
    /// the owner of at least one memory.
    pub fn project_exists(&self, id: &str) -> Result<bool, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM projects WHERE id = ?1)
                 OR EXISTS(SELECT 1 FROM memories WHERE project_id = ?1)",
            params![id],
            |row| row.get(0),
        )?;
        Ok(exists)
    }

    /// List every project in the store with its memory, handoff, and ADR counts.
    /// Projects that only have memories and no `projects` row are included.
    pub fn list_projects(&self) -> Result<Vec<ProjectSummary>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT p.id,
                    COALESCE(m.total, 0),
                    COALESCE(m.handoffs, 0),
                    COALESCE(m.adrs, 0),
                    m.latest
             FROM (SELECT id FROM projects UNION SELECT project_id FROM memories) p
             LEFT JOIN (
                 SELECT project_id,
                        COUNT(*) AS total,
                        SUM(memory_type = 'handoff') AS handoffs,
                        SUM(memory_type = 'adr') AS adrs,
                        MAX(updated_at) AS latest
                 FROM memories
                 GROUP BY project_id
             ) m ON m.project_id = p.id
             ORDER BY m.latest DESC NULLS LAST, p.id",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(ProjectSummary {
                id: row.get(0)?,
                memory_count: row.get::<_, i64>(1)? as usize,
                handoff_count: row.get::<_, i64>(2)? as usize,
                adr_count: row.get::<_, i64>(3)? as usize,
                latest_activity_at: row.get(4)?,
            })
        })?;

        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
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
        let artifacts_json: Option<String> = memory
            .external_artifacts
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(serde_json::to_string)
            .transpose()?;
        conn.execute(
            "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, merged_from, pinned, global, external_artifacts)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
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
                artifacts_json,
            ],
        )?;
        Ok(())
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<Memory>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {MEMORY_COLUMNS} FROM memories WHERE id = ?1"
        ))?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            Ok(Some(map_memory_row(row)?))
        } else {
            Ok(None)
        }
    }

    pub fn update_memory(&self, memory: &Memory) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let tags_json = serde_json::to_string(&memory.tags)?;
        // Store NULL for None or empty vec (treat empty as clear); non-empty vec as JSON.
        let artifacts_json: Option<String> = memory
            .external_artifacts
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(serde_json::to_string)
            .transpose()?;
        let merged_from_json = memory
            .merged_from
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        conn.execute(
            "UPDATE memories SET content = ?1, summary = ?2, tags = ?3, importance = ?4, relevance_score = ?5, access_count = ?6, updated_at = ?7, last_accessed_at = ?8, pinned = ?9, global = ?10, external_artifacts = ?11, branch = ?12, merged_from = ?13
             WHERE id = ?14",
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
                artifacts_json,
                memory.branch,
                merged_from_json,
                memory.id,
            ],
        )?;
        Ok(())
    }

    pub fn delete_memory(&self, id: &str) -> Result<bool, MemoryError> {
        self.delete_memory_with_op(id, crate::db::OP_DELETE)
    }

    /// Delete a memory, recording which operation destroyed it.
    ///
    /// The snapshot and the delete share one transaction, so the memory is never
    /// removed without a recoverable copy.
    pub fn delete_memory_with_op(&self, id: &str, op: &str) -> Result<bool, MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        super::trash::trash_memory_in(&tx, id, op, chrono::Utc::now().timestamp())?;
        // Belt-and-suspenders: explicitly delete sidecar row before deleting the memory
        // so that the handoff_sections row is removed even if FOREIGN KEY CASCADE is
        // disabled or the constraint fires in an unexpected order.
        tx.execute(
            "DELETE FROM handoff_sections WHERE memory_id = ?1",
            params![id],
        )?;
        let rows_affected = tx.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        tx.commit()?;
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

        let mut sql = format!("SELECT {MEMORY_COLUMNS} FROM memories WHERE project_id = ?1");

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

        let mut memories: Vec<Memory> = match branch_filter {
            Some(Some(branch)) => stmt
                .query_map(
                    params![project_id, min_rel, fetch_limit as i64, branch],
                    map_memory_row,
                )?
                .filter_map(|r| r.ok())
                .collect(),
            _ => stmt
                .query_map(
                    params![project_id, min_rel, fetch_limit as i64],
                    map_memory_row,
                )?
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

    pub fn record_access(&self, id: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            REINFORCE_SQL,
            params![now, id, ACCESS_REINFORCEMENT, RELEVANCE_CEILING],
        )?;
        Ok(())
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
                REINFORCE_SQL,
                params![now, id, ACCESS_REINFORCEMENT, RELEVANCE_CEILING],
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
            "SELECT {MEMORY_COLUMNS}
             FROM memories WHERE id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = conn.prepare(&sql)?;

        // Convert ids to params
        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

        let rows = stmt.query_map(params.as_slice(), map_memory_row)?;

        let mut result = HashMap::new();
        for memory in rows.flatten() {
            result.insert(memory.id.clone(), memory);
        }

        Ok(result)
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

    /// Get the top N most accessed memories for a project.
    #[allow(dead_code)] // Used by CLI insights command
    pub fn get_most_accessed(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {MEMORY_COLUMNS}
             FROM memories WHERE project_id = ?1
             ORDER BY access_count DESC
             LIMIT ?2"
        ))?;
        let rows = stmt.query_map(params![project_id, limit as i64], map_memory_row)?;
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

    /// Delete non-pinned, non-global memories that have fully decayed and were never accessed.
    ///
    /// Conditions: pinned = 0, global = 0, relevance_score <= 0.1, access_count = 0,
    /// and created more than 30 days ago. Each one is snapshotted to the trash first;
    /// this runs unattended on the decay job, so it is the destructive path least likely
    /// to be noticed. Embeddings and relationships are removed by CASCADE constraints.
    ///
    /// Returns the IDs of deleted memories.
    #[allow(dead_code)] // Called by the decay background job in main.rs
    pub fn auto_prune_stale_memories(&self, project_id: &str) -> Result<Vec<String>, MemoryError> {
        let cutoff = chrono::Utc::now().timestamp() - 30 * 86400;

        let ids: Vec<String> = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT id FROM memories
                 WHERE project_id = ?1
                   AND pinned = 0
                   AND global = 0
                   AND relevance_score <= 0.1
                   AND access_count = 0
                   AND created_at < ?2",
            )?;
            stmt.query_map(params![project_id, cutoff], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect()
        };

        if ids.is_empty() {
            return Ok(ids);
        }

        self.delete_memories_batch_with_op(&ids, super::OP_PRUNE)?;
        Ok(ids)
    }

    /// Delete all memories and relationships for a project
    /// Delete every memory in a project, snapshotting each one to the trash first.
    pub fn delete_project_data(&self, project_id: &str) -> Result<usize, MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().timestamp();

        let ids: Vec<String> = {
            let mut stmt = tx.prepare("SELECT id FROM memories WHERE project_id = ?1")?;
            let rows = stmt.query_map(params![project_id], |row| row.get::<_, String>(0))?;
            rows.flatten().collect()
        };
        for id in &ids {
            super::trash::trash_memory_in(&tx, id, super::OP_WIPE, now)?;
        }

        let rows = tx.execute(
            "DELETE FROM memories WHERE project_id = ?1",
            params![project_id],
        )?;
        tx.commit()?;
        Ok(rows)
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

        let adr_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?1 AND memory_type = 'adr'",
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
            adr_count: adr_count as usize,
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

    // Decay operations
    pub fn update_relevance_scores(
        &self,
        project_id: &str,
        decay_rate: f64,
    ) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();

        // The formula, including both clamps, lives in `crate::decay::relevance_from_parts`;
        // RELEVANCE() is that function registered as a SQLite scalar.
        //
        // The elapsed term is store-days, not wall-clock days: the count of days on which
        // this project received a store after the memory was last accessed. See
        // `db::activity`. The subquery is correlated but bounded by the project's own
        // memory count, and `idx_memories_project` covers the lookup.
        let sql = format!(
            r#"
            UPDATE memories
            SET relevance_score = RELEVANCE(
                (SELECT COUNT(DISTINCT m2.created_at / {day})
                 FROM memories m2
                 WHERE m2.project_id = memories.project_id
                   AND m2.created_at / {day} > memories.last_accessed_at / {day}
                   AND {clock}),
                importance,
                access_count,
                ?1
            )
            WHERE project_id = ?2
            AND pinned = 0
            "#,
            day = super::SECONDS_PER_DAY,
            clock = super::CLOCK_ADVANCING_STORE.replace("tags", "m2.tags"),
        );
        let rows_affected = conn.execute(&sql, params![decay_rate, project_id])?;

        Ok(rows_affected)
    }

    /// Merge a duplicate memory into an existing one.
    ///
    /// The survivor keeps its own content, gains the union of both tag sets and the
    /// higher importance, and records the consumed memory in `merged_from`. The consumed
    /// row is deleted, so before that happens it is snapshotted to the trash and its full
    /// content is copied into the provenance entry: this is the only path that destroys
    /// content without anyone asking, and a 100-character preview is not a record of what
    /// was lost.
    pub fn merge_memories(&self, survivor_id: &str, consumed_id: &str) -> Result<(), MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().timestamp();

        // Read the consumed memory's own content, tags and importance here rather than
        // accepting them from the caller, so the provenance entry always describes the
        // memory that actually went away regardless of which side of the pair survives.
        let (consumed_content, old_tags_json, old_importance): (String, String, f64) = tx
            .query_row(
                "SELECT content, tags, importance FROM memories WHERE id = ?1",
                params![consumed_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
        let old_tags: Vec<String> = serde_json::from_str(&old_tags_json).unwrap_or_default();

        // Get the survivor's current state
        let (new_tags_json, new_importance, existing_merged_from): (String, f64, Option<String>) =
            tx.query_row(
                "SELECT tags, importance, merged_from FROM memories WHERE id = ?1",
                params![survivor_id],
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
        let mut merge_sources: Vec<crate::memory::MergeSource> = existing_merged_from
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        merge_sources.push(crate::memory::MergeSource::new(
            consumed_id.to_string(),
            consumed_content,
            now,
        ));
        let merged_from_json = serde_json::to_string(&merge_sources)?;

        // Update the survivor with merged data
        tx.execute(
            "UPDATE memories SET tags = ?1, importance = ?2, merged_from = ?3, updated_at = ?4 WHERE id = ?5",
            params![merged_tags_json, max_importance, merged_from_json, now, survivor_id],
        )?;

        // Snapshot the consumed memory before removing it. `merged_from` records what it
        // said; the trash entry is what makes the merge reversible.
        super::trash::trash_memory_in(&tx, consumed_id, super::OP_MERGE, now)?;
        tx.execute(
            "DELETE FROM handoff_sections WHERE memory_id = ?1",
            params![consumed_id],
        )?;
        tx.execute("DELETE FROM memories WHERE id = ?1", params![consumed_id])?;

        tx.commit()?;
        Ok(())
    }

    /// Count hook-captured memories stored today (UTC) for the given project.
    ///
    /// A memory is considered hook-captured if its `tags` JSON array contains the
    /// string `"hook"`. Tags are stored as a JSON array, e.g. `["hook","prompt"]`,
    /// so the LIKE pattern `%"hook"%` matches exactly.
    #[allow(dead_code)] // Called by hooks::dispatch and CLI; not referenced from lib root
    pub fn count_hook_memories_today(&self, project_id: &str) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        // Start-of-today UTC: truncate current timestamp to midnight.
        // Note: `tags LIKE '%"hook"%'` is a full-table scan today. Acceptable at small
        // project sizes; if hook traffic grows, promote `source` (hook vs manual) to a
        // dedicated indexed column.
        let now = chrono::Utc::now();
        let start_of_today = now
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .expect("midnight is always a valid time")
            .and_utc()
            .timestamp();

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?1 AND created_at >= ?2 AND tags LIKE '%\"hook\"%'",
            params![project_id, start_of_today],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}
