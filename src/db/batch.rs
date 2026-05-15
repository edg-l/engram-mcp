use rusqlite::params;

use crate::error::MemoryError;
use crate::memory::Memory;

use super::Database;

impl Database {
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
            let artifacts_json: Option<String> = memory
                .external_artifacts
                .as_ref()
                .filter(|v| !v.is_empty())
                .map(serde_json::to_string)
                .transpose()?;
            tx.execute(
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
}
