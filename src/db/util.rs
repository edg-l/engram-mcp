use std::str::FromStr;

use crate::memory::{Memory, MemoryType};

/// Column list for every `SELECT` whose rows are mapped by [`map_memory_row`].
///
/// Keep the order in sync with that function; the mapper reads by index.
pub(super) const MEMORY_COLUMNS: &str = "id, project_id, memory_type, content, summary, tags, \
     importance, relevance_score, access_count, created_at, updated_at, last_accessed_at, \
     branch, merged_from, pinned, global, external_artifacts";

/// Map a row selected with [`MEMORY_COLUMNS`] into a [`Memory`].
pub(super) fn map_memory_row(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    let memory_type_str: String = row.get(2)?;
    let tags_json: String = row.get(5)?;
    Ok(Memory {
        id: row.get(0)?,
        project_id: row.get(1)?,
        memory_type: parse_memory_type_col(&memory_type_str, 2)?,
        content: row.get(3)?,
        summary: row.get(4)?,
        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
        importance: row.get(6)?,
        relevance_score: row.get(7)?,
        access_count: row.get(8)?,
        created_at: row.get(9)?,
        updated_at: row.get(10)?,
        last_accessed_at: row.get(11)?,
        branch: row.get(12)?,
        merged_from: row
            .get::<_, Option<String>>(13)?
            .and_then(|s| serde_json::from_str(&s).ok()),
        pinned: row.get::<_, i64>(14)? != 0,
        global: row.get::<_, i64>(15)? != 0,
        external_artifacts: row
            .get::<_, Option<String>>(16)?
            .and_then(|s| serde_json::from_str(&s).ok()),
    })
}

/// Parse a memory type string from a DB row, propagating an error on unknown values.
///
/// Used in `query_map` closures that return `rusqlite::Result<T>` so the error type
/// matches without requiring a full `MemoryError` conversion at every call site.
pub(super) fn parse_memory_type_col(s: &str, col: usize) -> rusqlite::Result<MemoryType> {
    MemoryType::from_str(s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(col, rusqlite::types::Type::Text, Box::new(e))
    })
}
