//! Shared dedup core used by both the MCP `memory_store` path and the hook store path.

use std::collections::HashMap;

use crate::db::Database;
use crate::embedding::{EmbeddingService, cosine_similarity};
use crate::error::MemoryError;
use crate::memory::{Memory, MemoryType};

/// Outcome of a `store_with_dedup` call.
pub enum StoreOutcome {
    /// Memory was stored normally. Contains the full `"mem_<uuid>"` id.
    Stored(String),
    /// Memory was stored and auto-merged with a near-duplicate.
    /// Both ids are full `"mem_<uuid>"` strings.
    Merged {
        id: String,
        merged_with: String,
        similarity: f64,
    },
    /// Memory was NOT stored because a similar-enough one already exists.
    #[allow(dead_code)]
    SkippedSimilar { similar_to: String, similarity: f64 },
}

/// Find memories that are potential duplicates of the given embedding.
///
/// Returns `(memory_id, similarity)` pairs where similarity >= threshold and type matches.
pub(crate) fn find_duplicates(
    embedding: &[f32],
    memory_type: MemoryType,
    threshold: f32,
    existing_embeddings: &[(String, Vec<f32>)],
    existing_memories: &HashMap<String, Memory>,
) -> Vec<(String, f32)> {
    let mut duplicates = Vec::new();

    for (existing_id, existing_vec) in existing_embeddings {
        let similarity = cosine_similarity(embedding, existing_vec);
        if similarity >= threshold
            && let Some(mem) = existing_memories.get(existing_id)
            && mem.memory_type == memory_type
        {
            duplicates.push((existing_id.clone(), similarity));
        }
    }

    duplicates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    duplicates
}

/// Store a memory with dedup logic.
///
/// # Pairing rule
/// `embedding_service.is_some()` MUST equal `embedding_vec.is_some()`. A mismatch
/// returns `Err(MemoryError::Embedding(...))`.
///
/// # When both are `None`
/// Dedup is skipped entirely. The memory is stored and `StoreOutcome::Stored` is returned.
///
/// # When both are `Some`
/// Existing project embeddings are fetched and `find_duplicates` is run:
/// - If `skip_above` is `Some(thr)` and the best duplicate similarity >= thr, returns
///   `StoreOutcome::SkippedSimilar` without storing anything.
/// - Otherwise the memory + embedding are stored. If a duplicate >= `dedup_threshold`
///   exists, `merge_memories` is called and `StoreOutcome::Merged` is returned.
/// - If no duplicate meets the threshold, `StoreOutcome::Stored` is returned.
#[allow(clippy::too_many_arguments)]
pub fn store_with_dedup(
    db: &Database,
    embedding_service: Option<&EmbeddingService>,
    project_id: &str,
    memory: Memory,
    embedding_vec: Option<&[f32]>,
    dedup_threshold: f32,
    skip_above: Option<f32>,
) -> Result<StoreOutcome, MemoryError> {
    // Pairing rule
    match (embedding_service.is_some(), embedding_vec.is_some()) {
        (true, false) | (false, true) => {
            return Err(MemoryError::Embedding(
                "store_with_dedup: embedding_service and embedding_vec must both be Some or both be None".to_string(),
            ));
        }
        _ => {}
    }

    let id = memory.id.clone();

    if embedding_service.is_none() {
        // No embedding — store directly, skip dedup.
        db.store_memory(&memory)?;
        return Ok(StoreOutcome::Stored(id));
    }

    let embedding = embedding_vec.unwrap();
    let es = embedding_service.unwrap();

    // Fetch all embeddings once for dedup check.
    let existing_embeddings = db.get_all_embeddings_for_project_and_global(project_id)?;
    let existing_memories_list = db.get_all_memories_for_project(project_id)?;
    let existing_memories: HashMap<String, Memory> = existing_memories_list
        .into_iter()
        .map(|m| (m.id.clone(), m))
        .collect();

    let memory_type = memory.memory_type;
    let duplicates = find_duplicates(
        embedding,
        memory_type,
        dedup_threshold,
        &existing_embeddings,
        &existing_memories,
    );

    // skip_above gate: if the best duplicate is above the skip threshold, don't store.
    if let Some(skip_thr) = skip_above
        && let Some((dup_id, dup_sim)) = duplicates.first()
        && *dup_sim >= skip_thr
    {
        return Ok(StoreOutcome::SkippedSimilar {
            similar_to: dup_id.clone(),
            similarity: *dup_sim as f64,
        });
    }

    // Store memory + embedding.
    db.store_memory(&memory)?;
    db.store_embedding(&id, embedding, es.model_version())?;

    // Merge if a duplicate meets the dedup threshold.
    if let Some((dup_id, dup_similarity)) = duplicates.first() {
        let dup_memory = existing_memories.get(dup_id);
        let old_preview: String = dup_memory
            .map(|m| m.content.chars().take(100).collect())
            .unwrap_or_default();

        let existing_is_global = dup_memory.map(|m| m.global).unwrap_or(false);
        let new_is_global = memory.global;

        let (survivor_id, consumed_id) = if existing_is_global && !new_is_global {
            // Existing global wins: keep existing, consume new local
            (dup_id.as_str(), id.as_str())
        } else {
            // New wins (default): keep new, consume old
            (id.as_str(), dup_id.as_str())
        };

        db.merge_memories(survivor_id, consumed_id, &old_preview)?;

        return Ok(StoreOutcome::Merged {
            id: survivor_id.to_string(),
            merged_with: consumed_id.to_string(),
            similarity: *dup_similarity as f64,
        });
    }

    Ok(StoreOutcome::Stored(id))
}
