use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Memory not found: {0}")]
    NotFound(String),

    #[error("Invalid memory type: {0}")]
    InvalidType(String),

    #[error("Invalid relation type: {0}")]
    InvalidRelation(String),

    #[error("Unknown tool: {0}")]
    UnknownTool(String),

    #[error("Unknown project '{requested}'. Known projects: {known}")]
    UnknownProject { requested: String, known: String },

    /// Tool arguments that failed to deserialize. Lists the field names that were
    /// actually received, so a misnamed or misplaced field is visible from the
    /// error alone rather than looking like the server lost a field that was sent.
    #[error("Invalid arguments for {tool}: {message}. Fields received: {received}")]
    InvalidArguments {
        tool: String,
        message: String,
        received: String,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
