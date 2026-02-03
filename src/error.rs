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

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
