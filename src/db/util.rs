use std::str::FromStr;

use crate::memory::MemoryType;

/// Parse a memory type string from a DB row, propagating an error on unknown values.
///
/// Used in `query_map` closures that return `rusqlite::Result<T>` so the error type
/// matches without requiring a full `MemoryError` conversion at every call site.
pub(super) fn parse_memory_type_col(s: &str, col: usize) -> rusqlite::Result<MemoryType> {
    MemoryType::from_str(s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(col, rusqlite::types::Type::Text, Box::new(e))
    })
}
