use engram_mcp::db::Database;

/// Open a fresh Database inside `dir`, running all migrations via the
/// standard `Database::open` path. The returned `Database` is valid for
/// the lifetime of the caller's `TempDir`.
pub fn setup_db(dir: &tempfile::TempDir) -> anyhow::Result<Database> {
    let db_path = dir.path().join("bench.db");
    let db = Database::open(&db_path)?;
    Ok(db)
}
