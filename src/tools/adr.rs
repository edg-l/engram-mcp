use serde::{Deserialize, Serialize};

use crate::db::Database;
use crate::embedding::EmbeddingService;
use crate::error::MemoryError;
use crate::memory::{AdrSections, AdrStatus, MemoryType};

// ============================================
// ADR result structs
// ============================================

/// Result returned by `create_adr`.
#[derive(Debug, Serialize, Deserialize)]
pub struct AdrCreateResult {
    /// ID of the newly created ADR memory.
    pub id: String,
    /// Sequential ADR number within this project.
    pub adr_number: u32,
    /// Status at creation time.
    pub status: AdrStatus,
    /// ID of the ADR that was superseded, if `supersedes` was provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_id: Option<String>,
}

// ============================================
// ADR free functions
// ============================================

/// Create an ADR memory with embedding, stored atomically.
///
/// ADRs are project-global (no branch).  When `supersedes` is `Some(old_number)`:
///
/// 1. Validation is done read-only BEFORE any write (not-found and invalid-transition
///    failures are returned without creating the new ADR).
/// 2. The new ADR insertion and the old ADR's status flip + Supersedes relationship
///    edge are committed in a single transaction via `store_adr_atomic`, so there is
///    no window where the new ADR exists but the old one has not been updated.
#[allow(clippy::too_many_arguments)]
pub fn create_adr(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
    sections: AdrSections,
    status: AdrStatus,
    importance: f64,
    pinned: bool,
    supersedes: Option<u32>,
) -> Result<AdrCreateResult, MemoryError> {
    // ---- Pre-write validation for supersession ----
    // Resolve and validate the old ADR before generating the embedding or touching
    // the DB with any write.  Common failure modes (not-found, invalid transition)
    // are caught here so the new ADR is never created unnecessarily.
    let supersede_info: Option<(String, bool)> = if let Some(old_number) = supersedes {
        let old_id = db
            .get_adr_by_number(project_id, old_number)?
            .ok_or_else(|| MemoryError::NotFound(format!("ADR-{:04} not found", old_number)))?;

        let (_, old_status, _) = db
            .get_adr_sections(&old_id)?
            .ok_or_else(|| MemoryError::NotFound(format!("ADR sidecar missing for {}", old_id)))?;

        if old_status == AdrStatus::Superseded {
            // Already superseded — idempotent: create the new ADR but skip the flip/edge.
            Some((old_id, false))
        } else {
            if !old_status.can_transition_to(AdrStatus::Superseded) {
                return Err(MemoryError::InvalidType(format!(
                    "invalid ADR status transition: {} -> superseded",
                    old_status
                )));
            }
            Some((old_id, true))
        }
    } else {
        None
    };

    let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
    let now = chrono::Utc::now().timestamp();

    // Combine key fields for a single embedding vector.
    let combined = format!(
        "{}\n{}\n{}\n{}",
        sections.title, sections.context, sections.decision, sections.consequences
    );
    let emb = embedding.embed_memory(MemoryType::Adr, &combined)?;

    // Pass the pre-resolved old_id (if a flip should happen) into store_adr_atomic so
    // the entire create+supersede is a single atomic transaction.
    let supersede_old_id: Option<&str> =
        supersede_info.as_ref().and_then(|(old_id, should_flip)| {
            if *should_flip {
                Some(old_id.as_str())
            } else {
                None
            }
        });

    let (id, number) = db.store_adr_atomic(
        &id,
        project_id,
        &sections,
        status,
        importance.clamp(0.0, 1.0),
        pinned,
        &emb,
        embedding.model_version(),
        now,
        supersede_old_id,
    )?;

    // superseded_id is Some whenever supersedes resolved to an existing ADR (regardless
    // of whether the flip was needed or was already done).
    let superseded_id = supersede_info.map(|(old_id, _)| old_id);

    Ok(AdrCreateResult {
        id,
        adr_number: number,
        status,
        superseded_id,
    })
}
