//! Shared memory-update logic for `memory_update` and `engram-cli update`.
//!
//! A handoff's `content` and its `handoff_sections` sidecar (plus per-section
//! embeddings) must never drift apart, since `handoff_search`/`handoff_resume` read the
//! sidecar. This module is the one place that enforces that: any caller that wants to
//! change a memory's content, tags, importance, pinned/dead status, or a handoff's
//! structured sections goes through [`update_memory`].

use crate::db::{Database, OP_UPDATE, encode_section_embeddings};
use crate::embedding::EmbeddingService;
use crate::error::MemoryError;
use crate::memory::{HandoffSections, HandoffSectionsPatch, Memory, MemoryType};
use crate::summarize::{generate_summary, should_auto_summarize};

use super::handoff::handoff_section_key_texts;

/// Fields a caller may change on a memory. `content` and `sections` are mutually
/// exclusive: both end up rebuilding `content` and the section embeddings, so sending
/// both would leave it ambiguous which one wins.
#[derive(Debug, Default)]
pub struct MemoryUpdateRequest {
    pub id: String,
    pub content: Option<String>,
    /// Handoff-only partial patch to the structured sections: fields present replace,
    /// fields omitted keep the stored value.
    pub sections: Option<HandoffSectionsPatch>,
    pub importance: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub summary: Option<String>,
    pub pinned: Option<bool>,
    /// Mark the memory dead (its subject no longer exists) or bring it back.
    pub dead: Option<bool>,
    /// Why it was marked dead. Recorded alongside the flag.
    pub dead_reason: Option<String>,
    /// Replace external_artifacts list. `Some([])` clears; `None` preserves existing.
    pub external_artifacts: Option<Vec<String>>,
}

/// Result of applying a [`MemoryUpdateRequest`].
pub struct MemoryUpdateOutcome {
    /// The memory as it stood before this update, since content is replaced wholesale
    /// and the previous version is gone from `memories` the moment the row is written.
    pub previous: Memory,
    /// Whether `content` changed and was therefore snapshotted to the trash.
    pub content_replaced: bool,
    /// Dead status after applying `dead`/`dead_reason`.
    pub dead: bool,
}

pub fn update_memory(
    db: &Database,
    embedding: &EmbeddingService,
    request: MemoryUpdateRequest,
) -> Result<MemoryUpdateOutcome, MemoryError> {
    if request.sections.is_some() && request.content.is_some() {
        return Err(MemoryError::InvalidArguments {
            tool: "memory_update".to_string(),
            message: "`sections` and `content` are mutually exclusive; send one or the \
                      other, not both"
                .to_string(),
            received: "content, sections".to_string(),
        });
    }

    let mut memory = db
        .get_memory(&request.id)?
        .ok_or_else(|| MemoryError::NotFound(request.id.clone()))?;

    if request.sections.is_some() && memory.memory_type != MemoryType::Handoff {
        return Err(MemoryError::InvalidArguments {
            tool: "memory_update".to_string(),
            message: format!(
                "`sections` is only valid on handoff memories; {} is a {}",
                request.id, memory.memory_type
            ),
            received: "sections".to_string(),
        });
    }

    // `sections` patches the stored sidecar sections (not the possibly-unparseable
    // `content` string) and renders the result to markdown, then joins the regular
    // content-update path below so the sidecar and section embeddings are rebuilt the
    // same way a direct `content` edit would rebuild them.
    let sections_markdown = match request.sections {
        Some(patch) => {
            let (existing, _) = db.get_handoff_sections(&request.id)?.ok_or_else(|| {
                MemoryError::NotFound(format!("handoff sections for {}", request.id))
            })?;
            Some(existing.merge_patch(patch).render_markdown())
        }
        None => None,
    };
    let new_content = request.content.or(sections_markdown);

    // Content is replaced wholesale, not patched, so the previous version is gone the
    // moment the row is written. Snapshot it and hand it back to the caller.
    let previous = memory.clone();
    let content_replaced = new_content
        .as_ref()
        .is_some_and(|new| *new != memory.content);
    if content_replaced {
        db.trash_memory(&request.id, OP_UPDATE)?;
    }

    memory.updated_at = chrono::Utc::now().timestamp();

    // Handoff update invalidates and rebuilds section embeddings; sidecar must stay in
    // sync with content. Validate and rebuild BEFORE any DB write so a parse failure is
    // a clean abort — the memory row and sidecar are left untouched on error.
    //
    // The tuple carries: (new_sections, full_content_embedding, section_vecs). All
    // three are needed for the atomic update so they are computed together here.
    let handoff_sidecar_update: Option<(HandoffSections, Vec<f32>, Vec<Vec<f32>>)> =
        if memory.memory_type == MemoryType::Handoff {
            if let Some(ref content) = new_content {
                let new_sections = HandoffSections::parse_markdown(content)?;
                let full_embedding = embedding.embed_memory(MemoryType::Handoff, content)?;
                let section_texts = handoff_section_key_texts(&new_sections);
                let mut section_vecs: Vec<Vec<f32>> = Vec::new();
                for (_, text) in &section_texts {
                    section_vecs.push(embedding.embed(text)?);
                }
                Some((new_sections, full_embedding, section_vecs))
            } else {
                None
            }
        } else {
            None
        };

    if let Some(ref content) = new_content {
        memory.content = content.clone();
        // For non-Handoff types, store the embedding now (Handoff uses the atomic path below).
        if memory.memory_type != MemoryType::Handoff {
            let content_embedding = embedding.embed_memory(memory.memory_type, content)?;
            db.store_embedding(&memory.id, &content_embedding, embedding.model_version())?;
        }
        if request.summary.is_none() && should_auto_summarize(content, memory.summary.as_deref()) {
            memory.summary = Some(generate_summary(content));
        }
    }

    if let Some(importance) = request.importance {
        memory.importance = importance.clamp(0.0, 1.0);
    }

    if let Some(tags) = request.tags {
        memory.tags = tags;
    }

    if let Some(summary) = request.summary {
        memory.summary = Some(summary);
    }

    if let Some(pinned) = request.pinned {
        memory.pinned = pinned;
    }

    if let Some(dead) = request.dead {
        db.set_dead(&request.id, dead, request.dead_reason.as_deref())?;
    }

    // external_artifacts update semantics:
    //   - None       -> preserve existing (omit = keep)
    //   - Some([])   -> clear (empty array = delete)
    //   - Some([..]) -> replace with new list
    if let Some(artifacts) = request.external_artifacts {
        memory.external_artifacts = if artifacts.is_empty() {
            None
        } else {
            Some(artifacts)
        };
    }

    // For Handoff memories with new content: write memory row + full-content embedding +
    // sidecar in one transaction so a partial failure cannot leave them out of sync. For
    // all other cases fall back to the regular single-table update.
    if let Some((new_sections, full_embedding, section_vecs)) = handoff_sidecar_update {
        let section_texts = handoff_section_key_texts(&new_sections);
        let keys: Vec<&str> = section_texts.iter().map(|(k, _)| *k).collect();
        let (section_keys_str, section_bytes) = encode_section_embeddings(&keys, &section_vecs);
        db.update_memory_and_handoff_sidecar(
            &memory,
            &full_embedding,
            embedding.model_version(),
            &new_sections,
            &section_keys_str,
            &section_bytes,
        )?;
    } else {
        db.update_memory(&memory)?;
    }

    Ok(MemoryUpdateOutcome {
        previous,
        content_replaced,
        dead: db.is_dead(&request.id)?,
    })
}
