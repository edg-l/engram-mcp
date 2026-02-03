//! Import/export functionality for memories.
//!
//! Supports JSON export format with:
//! - Version header for compatibility
//! - Optional embedding export (base64 encoded)
//! - Import with merge/replace modes

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};

use crate::memory::{Memory, Relationship};

/// Current export format version
pub const EXPORT_VERSION: &str = "1.1";

/// Previous export version (for backward compatibility)
pub const EXPORT_VERSION_1_0: &str = "1.0";

/// Export format for memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportData {
    /// Format version for compatibility checks
    pub version: String,
    /// Project ID these memories belong to
    pub project_id: String,
    /// Exported memories
    pub memories: Vec<ExportedMemory>,
    /// Exported relationships
    pub relationships: Vec<Relationship>,
    /// Export timestamp
    pub exported_at: i64,
}

/// A memory with optional embedding data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedMemory {
    #[serde(flatten)]
    pub memory: Memory,
    /// Base64-encoded embedding vector (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<String>,
}

/// Import mode for handling existing memories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportMode {
    /// Merge with existing data, skip duplicates
    #[default]
    Merge,
    /// Replace all data in the project
    Replace,
}

impl std::str::FromStr for ImportMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "merge" => Ok(Self::Merge),
            "replace" => Ok(Self::Replace),
            _ => Err(format!("invalid import mode: {}", s)),
        }
    }
}

/// Statistics from an import operation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[allow(dead_code)] // Used by MCP server tools
pub struct ImportStats {
    /// Number of memories imported
    pub memories_imported: usize,
    /// Number of memories skipped (duplicates in merge mode)
    pub memories_skipped: usize,
    /// Number of relationships imported
    pub relationships_imported: usize,
    /// Number of relationships skipped
    pub relationships_skipped: usize,
    /// Number of embeddings imported
    pub embeddings_imported: usize,
}

/// Encode an embedding vector to base64
pub fn encode_embedding(vector: &[f32]) -> String {
    let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
    BASE64.encode(&bytes)
}

/// Decode a base64 embedding to a vector
pub fn decode_embedding(encoded: &str) -> Result<Vec<f32>, base64::DecodeError> {
    let bytes = BASE64.decode(encoded)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Create export data from memories and relationships
pub fn create_export(
    project_id: &str,
    memories: Vec<Memory>,
    relationships: Vec<Relationship>,
    embeddings: Option<Vec<(String, Vec<f32>)>>,
) -> ExportData {
    let embedding_map: std::collections::HashMap<String, Vec<f32>> =
        embeddings.unwrap_or_default().into_iter().collect();

    let exported_memories: Vec<ExportedMemory> = memories
        .into_iter()
        .map(|memory| {
            let embedding = embedding_map.get(&memory.id).map(|v| encode_embedding(v));
            ExportedMemory { memory, embedding }
        })
        .collect();

    ExportData {
        version: EXPORT_VERSION.to_string(),
        project_id: project_id.to_string(),
        memories: exported_memories,
        relationships,
        exported_at: chrono::Utc::now().timestamp(),
    }
}

/// Validate import data version compatibility
pub fn validate_import(data: &ExportData) -> Result<(), String> {
    // Support both 1.0 (pre-branch) and 1.1 (with branch)
    if data.version != EXPORT_VERSION && data.version != EXPORT_VERSION_1_0 {
        return Err(format!(
            "Unsupported export version: {}. Expected: {} or {}",
            data.version, EXPORT_VERSION, EXPORT_VERSION_1_0
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_embedding() {
        let original = vec![1.0f32, 2.0, 3.0, -1.5, 0.0];
        let encoded = encode_embedding(&original);
        let decoded = decode_embedding(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_import_mode() {
        assert_eq!("merge".parse::<ImportMode>(), Ok(ImportMode::Merge));
        assert_eq!("REPLACE".parse::<ImportMode>(), Ok(ImportMode::Replace));
        assert!("invalid".parse::<ImportMode>().is_err());
    }

    #[test]
    fn test_validate_import() {
        let valid = ExportData {
            version: EXPORT_VERSION.to_string(),
            project_id: "test".to_string(),
            memories: vec![],
            relationships: vec![],
            exported_at: 0,
        };
        assert!(validate_import(&valid).is_ok());

        // Also accept version 1.0 (backward compat)
        let valid_v1 = ExportData {
            version: EXPORT_VERSION_1_0.to_string(),
            project_id: "test".to_string(),
            memories: vec![],
            relationships: vec![],
            exported_at: 0,
        };
        assert!(validate_import(&valid_v1).is_ok());

        let invalid = ExportData {
            version: "0.1".to_string(),
            project_id: "test".to_string(),
            memories: vec![],
            relationships: vec![],
            exported_at: 0,
        };
        assert!(validate_import(&invalid).is_err());
    }
}
