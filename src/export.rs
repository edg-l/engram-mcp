//! Import/export functionality for memories.
//!
//! Supports JSON export format with:
//! - Version header for compatibility
//! - Optional embedding export (base64 encoded)
//! - Import with merge/replace modes
//! - Handoff sidecar round-trip (sections + section_embeddings, base64 encoded)

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};

use crate::memory::{AdrSections, AdrStatus, HandoffSections, Memory, Relationship};

/// Current export format version
pub const EXPORT_VERSION: &str = "1.2";

/// Previous export version (for backward compatibility)
pub const EXPORT_VERSION_1_1: &str = "1.1";

/// Oldest supported export version (for backward compatibility)
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
    /// Embedding model that produced the stored vectors
    #[serde(default)]
    pub model_version: Option<String>,
}

/// A memory with optional embedding data.
///
/// For `MemoryType::Handoff` memories, the three handoff sidecar fields are populated
/// when the export was created from a DB that has the sidecar row.  Older exports that
/// pre-date the handoff feature will have these fields absent; importers must tolerate
/// their absence and skip the sidecar insert (memory row is still imported normally).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedMemory {
    #[serde(flatten)]
    pub memory: Memory,
    /// Base64-encoded full-content embedding vector (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<String>,
    /// Structured handoff sections (Handoff memories only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sections: Option<HandoffSections>,
    /// Comma-separated section names matching `section_embeddings` (Handoff memories only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_embedding_keys: Option<String>,
    /// Base64-encoded concatenated per-section embedding bytes (Handoff memories only).
    /// Format: 256 dims × N sections × 4 bytes (little-endian f32).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_embeddings: Option<String>,
    /// ADR number (ADR memories only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adr_number: Option<u32>,
    /// ADR status string, e.g. "proposed" (ADR memories only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adr_status: Option<String>,
    /// Structured ADR sections (ADR memories only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adr_sections: Option<AdrSections>,
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
pub fn decode_embedding(encoded: &str) -> Result<Vec<f32>, String> {
    let bytes = BASE64
        .decode(encoded)
        .map_err(|e| format!("Failed to decode embedding: {}", e))?;

    if bytes.len() % 4 != 0 {
        return Err(format!(
            "Corrupted embedding: byte length {} is not a multiple of 4",
            bytes.len()
        ));
    }

    let vector: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(vector)
}

/// Per-handoff sidecar data for export.
///
/// `keys` is the comma-separated section names, `bytes` the raw section embedding blob.
pub struct HandoffSidecar {
    pub sections: HandoffSections,
    pub keys: String,
    pub bytes: Vec<u8>,
}

/// Sidecar data for a `MemoryType::Adr` memory in an export.
///
/// Maps `memory_id → (adr_number, status, sections)`.
pub type AdrSidecarMap = std::collections::HashMap<String, (u32, AdrStatus, AdrSections)>;

/// Create export data from memories and relationships.
///
/// `handoff_sidecars` maps handoff memory IDs to their sidecar data.  Pass an empty map
/// or omit sidecars for non-handoff exports; this parameter is additive and does not
/// affect non-Handoff memories.
///
/// `adr_sidecars` maps ADR memory IDs to `(number, status, sections)` tuples.
/// Pass an empty map when no ADR export is needed.
pub fn create_export(
    project_id: &str,
    memories: Vec<Memory>,
    relationships: Vec<Relationship>,
    embeddings: Option<Vec<(String, Vec<f32>)>>,
    handoff_sidecars: std::collections::HashMap<String, HandoffSidecar>,
    adr_sidecars: &AdrSidecarMap,
    model_version: Option<String>,
) -> ExportData {
    let embedding_map: std::collections::HashMap<String, Vec<f32>> =
        embeddings.unwrap_or_default().into_iter().collect();

    let exported_memories: Vec<ExportedMemory> = memories
        .into_iter()
        .map(|memory| {
            let embedding = embedding_map.get(&memory.id).map(|v| encode_embedding(v));

            // For Handoff memories, include sidecar data if available.
            let (sections, section_embedding_keys, section_embeddings) =
                if let Some(sidecar) = handoff_sidecars.get(&memory.id) {
                    (
                        Some(sidecar.sections.clone()),
                        Some(sidecar.keys.clone()),
                        Some(BASE64.encode(&sidecar.bytes)),
                    )
                } else {
                    (None, None, None)
                };

            // For ADR memories, include sidecar data if available.
            let (adr_number, adr_status, adr_sections) =
                if let Some((num, status, adr_sec)) = adr_sidecars.get(&memory.id) {
                    (
                        Some(*num),
                        Some(status.as_str().to_string()),
                        Some(adr_sec.clone()),
                    )
                } else {
                    (None, None, None)
                };

            ExportedMemory {
                memory,
                embedding,
                sections,
                section_embedding_keys,
                section_embeddings,
                adr_number,
                adr_status,
                adr_sections,
            }
        })
        .collect();

    ExportData {
        version: EXPORT_VERSION.to_string(),
        project_id: project_id.to_string(),
        memories: exported_memories,
        relationships,
        exported_at: chrono::Utc::now().timestamp(),
        model_version,
    }
}

/// Decode a base64-encoded section embeddings blob back to raw bytes.
pub fn decode_section_embedding_bytes(encoded: &str) -> Result<Vec<u8>, String> {
    BASE64
        .decode(encoded)
        .map_err(|e| format!("Failed to decode section_embeddings: {}", e))
}

/// Validate import data version compatibility
pub fn validate_import(data: &ExportData) -> Result<(), String> {
    // Support 1.0 (pre-branch), 1.1 (with branch), and 1.2 (ADR round-trip)
    if data.version != EXPORT_VERSION
        && data.version != EXPORT_VERSION_1_1
        && data.version != EXPORT_VERSION_1_0
    {
        return Err(format!(
            "Unsupported export version: {}. Expected: {}, {}, or {}",
            data.version, EXPORT_VERSION, EXPORT_VERSION_1_1, EXPORT_VERSION_1_0
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
            model_version: None,
        };
        assert!(validate_import(&valid).is_ok());

        // Also accept version 1.0 (backward compat)
        let valid_v1 = ExportData {
            version: EXPORT_VERSION_1_0.to_string(),
            project_id: "test".to_string(),
            memories: vec![],
            relationships: vec![],
            exported_at: 0,
            model_version: None,
        };
        assert!(validate_import(&valid_v1).is_ok());

        let invalid = ExportData {
            version: "0.1".to_string(),
            project_id: "test".to_string(),
            memories: vec![],
            relationships: vec![],
            exported_at: 0,
            model_version: None,
        };
        assert!(validate_import(&invalid).is_err());
    }

    /// 5.3: validate_import accepts a payload with version "1.1" (backward compat).
    #[test]
    fn validate_import_accepts_1_1() {
        let data = ExportData {
            version: EXPORT_VERSION_1_1.to_string(),
            project_id: "test".to_string(),
            memories: vec![],
            relationships: vec![],
            exported_at: 0,
            model_version: None,
        };
        assert!(
            validate_import(&data).is_ok(),
            "version 1.1 must be accepted for backward compatibility"
        );
    }

    /// 5.3: ADR JSON round-trip — create_export then re-import into fresh DB, verify sections.
    #[test]
    fn adr_json_round_trip() {
        use crate::db::Database;
        use crate::memory::{AdrSections, AdrStatus, MemoryType};

        let db_src = Database::open_in_memory().unwrap();
        db_src
            .get_or_create_project("rt-proj", "Round Trip")
            .unwrap();

        let now = chrono::Utc::now().timestamp();
        let fake_emb = vec![0.1_f32; 256];

        let sections = AdrSections {
            title: "Use SQLite".to_string(),
            context: "Need a local DB".to_string(),
            decision: "Use SQLite".to_string(),
            consequences: "Simple deployment".to_string(),
        };

        let (adr_id, adr_num) = db_src
            .store_adr_atomic(
                "adr-rt-1",
                "rt-proj",
                &sections,
                AdrStatus::Accepted,
                0.8,
                true,
                &fake_emb,
                "test",
                now,
                None,
            )
            .unwrap();

        // Build the export with sidecar data.
        let memories = db_src.get_all_memories_for_project("rt-proj").unwrap();
        let relationships = db_src.get_all_relationships_for_project("rt-proj").unwrap();

        let mut adr_sidecars = std::collections::HashMap::new();
        let (num, status, adr_sec) = db_src.get_adr_sections(&adr_id).unwrap().unwrap();
        adr_sidecars.insert(adr_id.clone(), (num, status, adr_sec));

        let export_data = create_export(
            "rt-proj",
            memories,
            relationships,
            None,
            std::collections::HashMap::new(),
            &adr_sidecars,
            None,
        );

        // Version must be "1.2".
        assert_eq!(
            export_data.version, EXPORT_VERSION,
            "exported version must be 1.2"
        );
        assert_eq!(EXPORT_VERSION, "1.2");

        // Serialize → deserialize to simulate round-trip through JSON.
        let json_str = serde_json::to_string(&export_data).unwrap();
        let reimported: ExportData = serde_json::from_str(&json_str).unwrap();

        // validate_import must accept.
        validate_import(&reimported).expect("re-imported data must pass validation");

        // Import into a fresh DB.
        let db_dst = Database::open_in_memory().unwrap();
        db_dst
            .get_or_create_project("rt-proj", "Round Trip Dst")
            .unwrap();

        for exported in reimported.memories {
            let ExportedMemory {
                mut memory,
                adr_number,
                adr_status: adr_status_str,
                adr_sections: adr_sections_data,
                ..
            } = exported;

            memory.project_id = "rt-proj".to_string();
            db_dst.store_memory(&memory).unwrap();

            // Re-import ADR sidecar.
            if memory.memory_type == MemoryType::Adr
                && let (Some(n), Some(s_str), Some(sec)) =
                    (adr_number, adr_status_str, adr_sections_data)
            {
                let st: AdrStatus = s_str.parse().unwrap();
                db_dst
                    .insert_adr_sidecar(&memory.id, "rt-proj", n, st, &sec, now, now)
                    .unwrap();
            }
        }

        // Assert get_adr_sections returns the same number/status/sections.
        let (rt_num, rt_status, rt_sections) =
            db_dst.get_adr_sections("adr-rt-1").unwrap().unwrap();
        assert_eq!(rt_num, adr_num, "adr_number must round-trip");
        assert_eq!(rt_status, AdrStatus::Accepted, "status must round-trip");
        assert_eq!(rt_sections.title, "Use SQLite", "title must round-trip");
        assert_eq!(
            rt_sections.context, "Need a local DB",
            "context must round-trip"
        );
        assert_eq!(
            rt_sections.decision, "Use SQLite",
            "decision must round-trip"
        );
        assert_eq!(
            rt_sections.consequences, "Simple deployment",
            "consequences must round-trip"
        );
    }
}
