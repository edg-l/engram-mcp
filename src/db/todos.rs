use rusqlite::{Row, params};

use crate::error::MemoryError;
use crate::memory::{Memory, TodoItem, TodoStatus};

use super::Database;

/// Columns of the join used to build a [`TodoItem`], in the order [`map_todo_row`] reads them.
const TODO_COLUMNS: &str = "m.id, m.project_id, m.content, m.tags, m.importance, m.created_at, \
     m.updated_at, m.branch, t.status, t.reason, t.closed_at";

fn map_todo_row(row: &Row) -> rusqlite::Result<TodoItem> {
    let tags_json: String = row.get(3)?;
    let status_str: String = row.get(8)?;
    Ok(TodoItem {
        id: row.get(0)?,
        project_id: row.get(1)?,
        text: row.get(2)?,
        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
        importance: row.get(4)?,
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
        branch: row.get(7)?,
        // The sidecar is written only through `store_todo_atomic` / `set_todo_status`, both of
        // which write `TodoStatus::as_str`, so an unparseable value means the row was edited
        // outside this code path.
        status: status_str.parse().unwrap_or(TodoStatus::Open),
        reason: row.get(9)?,
        closed_at: row.get(10)?,
    })
}

impl Database {
    /// Store a todo memory, its embedding, and the lifecycle sidecar in one transaction.
    ///
    /// Deliberately bypasses the dedup and cluster-assignment paths used by `memory_store`:
    /// merging two todos would destroy a work item, and clustering a task list adds nothing.
    pub fn store_todo_atomic(
        &self,
        memory: &Memory,
        embedding: &[f32],
        model_version: &str,
    ) -> Result<(), MemoryError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        let tags_json = serde_json::to_string(&memory.tags)?;
        tx.execute(
            "INSERT INTO memories (id, project_id, memory_type, content, summary, tags, importance, \
             relevance_score, access_count, created_at, updated_at, last_accessed_at, branch, \
             merged_from, pinned, global) \
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
                Option::<String>::None,
                memory.pinned as i64,
                memory.global as i64,
            ],
        )?;

        let vector_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        tx.execute(
            "INSERT OR REPLACE INTO embeddings (memory_id, vector, model_version) VALUES (?1, ?2, ?3)",
            params![memory.id, vector_bytes, model_version],
        )?;

        tx.execute(
            "INSERT INTO todo_items (memory_id, status, reason, closed_at)
             VALUES (?1, ?2, NULL, NULL)",
            params![memory.id, TodoStatus::Open.as_str()],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Fetch a single todo, or `None` when the id is not a todo.
    pub fn get_todo(&self, memory_id: &str) -> Result<Option<TodoItem>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let sql = format!(
            "SELECT {TODO_COLUMNS} FROM todo_items t
             JOIN memories m ON m.id = t.memory_id
             WHERE t.memory_id = ?1"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![memory_id], map_todo_row)?;
        Ok(rows.next().transpose()?)
    }

    /// Todos for a project, newest first.
    ///
    /// `status` filters by lifecycle state; `None` returns every state. `branch` is a
    /// three-way filter: `None` means every branch, `Some(None)` means only project-wide
    /// todos, and `Some(Some(b))` means branch `b` plus the project-wide ones — the same
    /// "branch or global" shape the memory queries use.
    pub fn list_todos(
        &self,
        project_id: &str,
        status: Option<TodoStatus>,
        branch: Option<Option<&str>>,
        limit: usize,
    ) -> Result<Vec<TodoItem>, MemoryError> {
        let conn = self.conn.lock().unwrap();

        let mut sql = format!(
            "SELECT {TODO_COLUMNS} FROM todo_items t
             JOIN memories m ON m.id = t.memory_id
             WHERE m.project_id = ?1"
        );
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(project_id.to_string())];

        if let Some(s) = status {
            args.push(Box::new(s.as_str().to_string()));
            sql.push_str(&format!(" AND t.status = ?{}", args.len()));
        }
        match branch {
            None => {}
            Some(None) => sql.push_str(" AND m.branch IS NULL"),
            Some(Some(b)) => {
                args.push(Box::new(b.to_string()));
                sql.push_str(&format!(
                    " AND (m.branch = ?{} OR m.branch IS NULL)",
                    args.len()
                ));
            }
        }
        args.push(Box::new(limit as i64));
        sql.push_str(&format!(
            " ORDER BY m.created_at DESC LIMIT ?{}",
            args.len()
        ));

        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), map_todo_row)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Move a todo to a new lifecycle state.
    ///
    /// Closing also marks the memory dead, and reopening clears that, so the existing
    /// curation layer keeps finished work out of `memory_query` and `memory_context`
    /// without either of them needing to know what a todo is.
    pub fn set_todo_status(
        &self,
        memory_id: &str,
        status: TodoStatus,
        reason: Option<&str>,
    ) -> Result<(), MemoryError> {
        let now = chrono::Utc::now().timestamp();
        {
            let conn = self.conn.lock().unwrap();
            let closed_at = (!status.is_open()).then_some(now);
            let changed = conn.execute(
                "UPDATE todo_items SET status = ?2, reason = ?3, closed_at = ?4
                 WHERE memory_id = ?1",
                params![memory_id, status.as_str(), reason, closed_at],
            )?;
            if changed == 0 {
                return Err(MemoryError::NotFound(memory_id.to_string()));
            }
            conn.execute(
                "UPDATE memories SET updated_at = ?2 WHERE id = ?1",
                params![memory_id, now],
            )?;
        }

        if status.is_open() {
            self.set_dead(memory_id, false, None)?;
        } else {
            let dead_reason = reason
                .map(str::to_string)
                .unwrap_or_else(|| format!("todo {status}"));
            self.set_dead(memory_id, true, Some(&dead_reason))?;
        }
        Ok(())
    }

    /// Replace a todo's text. The embedding is refreshed by the caller.
    pub fn update_todo_text(&self, memory_id: &str, text: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().timestamp();
        let changed = conn.execute(
            "UPDATE memories SET content = ?2, updated_at = ?3 WHERE id = ?1",
            params![memory_id, text, now],
        )?;
        if changed == 0 {
            return Err(MemoryError::NotFound(memory_id.to_string()));
        }
        Ok(())
    }

    /// Count of todos per lifecycle state for a project.
    pub fn todo_counts(&self, project_id: &str) -> Result<(usize, usize, usize), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT t.status, COUNT(*) FROM todo_items t
             JOIN memories m ON m.id = t.memory_id
             WHERE m.project_id = ?1 GROUP BY t.status",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        let (mut open, mut done, mut dropped) = (0usize, 0usize, 0usize);
        for (status, n) in rows.flatten() {
            match status.parse() {
                Ok(TodoStatus::Open) => open = n as usize,
                Ok(TodoStatus::Done) => done = n as usize,
                Ok(TodoStatus::Dropped) => dropped = n as usize,
                Err(_) => {}
            }
        }
        Ok((open, done, dropped))
    }
}
