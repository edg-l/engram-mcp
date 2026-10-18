//! Applying an `ExportData` payload to the store. The single implementation behind
//! `engram-cli import`, the pull half of `engram-cli sync`, and the `memory_import` MCP
//! tool, so the convergence rules (last-write-wins on `updated_at`, max of access
//! counters, origin pinning, model-mismatch re-embedding, alias redirection) cannot
//! differ between the paths a payload can arrive by.

use std::collections::{BTreeSet, HashSet};

use crate::db::{self, Database};
use crate::embedding::EmbeddingService;
use crate::error::MemoryError;
use crate::export::{self, ExportData, ExportScope, ExportedMemory, ImportMode, ImportStats};
use crate::memory::{AdrStatus, MemoryType};

use super::cluster::{assign_to_cluster, compute_cluster_centroid, generate_cluster_summary};

/// Bytes per section embedding in a handoff sidecar: 256 f32 dimensions.
const SECTION_EMBEDDING_BYTES: usize = 256 * 4;

#[derive(Debug, Clone, Copy)]
pub struct ImportOptions<'a> {
    pub mode: ImportMode,
    /// Project a `Project`-scoped payload is re-homed to. An `AllProjects` payload keeps
    /// each memory's own `project_id` (redirected through local project aliases) and
    /// ignores this.
    pub target_project: &'a str,
    /// The remote a `sync` pull came from. Pins each imported memory's origin so the push
    /// half does not echo it straight back; `None` for any other import, since there is
    /// no remote to attribute the rows to.
    pub origin_remote: Option<&'a str>,
}

#[derive(Debug, Default)]
pub struct ImportOutcome {
    pub stats: ImportStats,
    /// Memories removed from the target project by `ImportMode::Replace` before loading.
    pub cleared: usize,
    /// Every project written to, including a replace-mode target that was only cleared.
    pub touched_projects: BTreeSet<String>,
    /// The payload's embedding model when it differs from the local one; its vectors
    /// were discarded and every imported memory re-embedded locally.
    pub foreign_model: Option<String>,
    /// Sidecars that could not be applied. The memory rows themselves were imported.
    pub warnings: Vec<String>,
}

/// Merge (or, for a single-project payload, replace) `data` into the store.
///
/// Merge is last-write-wins on `memories.updated_at`, ties keeping the local copy, and the
/// source's `updated_at` is preserved rather than stamped with the import time: stamping
/// "now" would make every importer's copy look newest and ping-pong a row between two
/// machines forever. `access_count`/`last_accessed_at` take the max of both sides since
/// they are usage counters, not one machine's claim.
pub fn import_payload(
    db: &Database,
    embedding: &EmbeddingService,
    data: ExportData,
    options: &ImportOptions<'_>,
) -> Result<ImportOutcome, MemoryError> {
    export::validate_import(&data).map_err(MemoryError::Embedding)?;

    let all_projects = data.scope == ExportScope::AllProjects;
    if all_projects && options.mode == ImportMode::Replace {
        return Err(MemoryError::Embedding(
            "replace mode cannot be used with an all-projects payload (scope: all_projects): \
             replace would wipe every project in this store, not just the one the payload came \
             from. Re-export a single project's scope, or import this payload in merge mode."
                .to_string(),
        ));
    }

    let mut outcome = ImportOutcome::default();

    if options.mode == ImportMode::Replace {
        outcome.cleared = db.delete_project_data(options.target_project)?;
        outcome
            .touched_projects
            .insert(options.target_project.to_string());
    }

    // Vectors from a different embedding model are not comparable to local ones in cosine
    // search, so they are dropped and the content re-embedded under the local model.
    outcome.foreign_model = data
        .model_version
        .filter(|v| v != embedding.model_version());

    // Projects merged away on this machine. The sending machine may not have merged them
    // (merges are local), so its payload can still carry the old id; landing those rows
    // under it would recreate the project the merge folded away.
    let project_aliases = if all_projects {
        db.project_aliases()?
    } else {
        Default::default()
    };
    let mut ensured_projects: HashSet<String> = HashSet::new();

    for exported in data.memories {
        let target_project = if all_projects {
            let own = &exported.memory.project_id;
            project_aliases.get(own).unwrap_or(own).clone()
        } else {
            options.target_project.to_string()
        };
        if all_projects && ensured_projects.insert(target_project.clone()) {
            db.get_or_create_project(&target_project, &target_project)?;
        }
        import_memory(
            db,
            embedding,
            exported,
            &target_project,
            options,
            &mut outcome,
        )?;
    }

    for rel in data.relationships {
        let source_exists = db.get_memory(&rel.source_id)?.is_some();
        let target_exists = db.get_memory(&rel.target_id)?.is_some();
        if source_exists && target_exists {
            db.create_relationship(&rel)?;
            outcome.stats.relationships_imported += 1;
        } else {
            outcome.stats.relationships_skipped += 1;
        }
    }

    // New or changed knowledge advances each written project's store-day clock, so
    // relevance is recomputed there, as a store would.
    for pid in &outcome.touched_projects {
        if let Some(project) = db.get_project(pid)? {
            db.update_relevance_scores(pid, project.decay_rate)?;
        }
    }

    Ok(outcome)
}

fn import_memory(
    db: &Database,
    embedding: &EmbeddingService,
    exported: ExportedMemory,
    target_project: &str,
    options: &ImportOptions<'_>,
    outcome: &mut ImportOutcome,
) -> Result<(), MemoryError> {
    let ExportedMemory {
        mut memory,
        embedding: encoded_embedding,
        sections,
        section_embedding_keys,
        section_embeddings: encoded_section_embeddings,
        adr_number,
        adr_status,
        adr_sections,
        status,
        todo,
    } = exported;
    memory.project_id = target_project.to_string();

    // Replace mode has already cleared the project, so only merge looks the id up.
    let mut content_differs = false;
    let is_update = if options.mode == ImportMode::Merge {
        match db.get_memory(&memory.id)? {
            Some(existing) if memory.updated_at <= existing.updated_at => {
                outcome.stats.memories_skipped += 1;
                return Ok(());
            }
            Some(existing) => {
                content_differs = existing.content != memory.content;
                if content_differs {
                    db.trash_memory(&memory.id, db::OP_UPDATE)?;
                }
                memory.access_count = memory.access_count.max(existing.access_count);
                memory.last_accessed_at = memory.last_accessed_at.max(existing.last_accessed_at);
                true
            }
            None => false,
        }
    } else {
        false
    };

    // Only a brand-new ADR can collide: an update already owns its number. A collision is
    // always with a different memory, reached two ways: a row redirected into a merge
    // survivor was numbered in the merged-away project's sequence, or two machines created
    // an ADR with the same number in a project neither merged. The incoming number means
    // nothing in the target project either way, so the row takes the next free one rather
    // than being dropped; ADR numbers are a per-machine display convenience, not a synced
    // identity.
    let mut adr_number = adr_number;
    if !is_update
        && memory.memory_type == MemoryType::Adr
        && let Some(num) = adr_number
        && db.get_adr_by_number(target_project, num)?.is_some()
    {
        adr_number = Some(db.next_adr_number(target_project)?);
    }

    if is_update {
        db.update_memory(&memory)?;
        outcome.stats.memories_updated += 1;
    } else {
        db.store_memory(&memory)?;
        outcome.stats.memories_imported += 1;
    }
    outcome.touched_projects.insert(target_project.to_string());
    if let Some(remote) = options.origin_remote {
        db.set_memory_origin(&memory.id, remote, memory.updated_at)?;
    }

    let carried_vector = if outcome.foreign_model.is_some() {
        None
    } else {
        encoded_embedding
            .as_deref()
            .and_then(|e| export::decode_embedding(e).ok())
    };
    let vector = match carried_vector {
        Some(vector) => vector,
        None => embedding.embed_memory(memory.memory_type, &memory.content)?,
    };
    db.store_embedding(&memory.id, &vector, embedding.model_version())?;
    outcome.stats.embeddings_imported += 1;

    // ADRs and todos never go through clustering on store (`store_adr_atomic`,
    // `store_todo_atomic`), so import leaves them out too.
    if (!is_update || content_differs)
        && memory.memory_type != MemoryType::Adr
        && memory.memory_type != MemoryType::Todo
    {
        // A content-changed update may sit in a cluster whose centroid no longer represents
        // it; vacate that membership first so it cannot end up in two clusters with a stale
        // centroid left behind in the old one.
        if is_update && let Ok(Some(old_cluster_id)) = db.remove_from_cluster(&memory.id) {
            let member_ids = db.get_cluster_member_ids(&old_cluster_id)?;
            if member_ids.is_empty() {
                let _ = db.delete_empty_clusters(target_project);
            } else {
                let summary = generate_cluster_summary(db, &member_ids)?;
                if let Some(centroid) = compute_cluster_centroid(db, &member_ids)? {
                    let _ = db.update_cluster_centroid(&old_cluster_id, &centroid, &summary);
                }
            }
        }
        assign_to_cluster(
            db,
            target_project,
            &memory.id,
            &vector,
            &memory.content,
            memory.importance,
        )?;
    }

    // Handoff and ADR sidecars are not LWW-governed fields, so an update leaves the local
    // sidecar alone and only brand-new rows get one.
    if !is_update && memory.memory_type == MemoryType::Handoff {
        match (sections, section_embedding_keys, encoded_section_embeddings) {
            (Some(sections), Some(keys), Some(encoded)) => {
                match export::decode_section_embedding_bytes(&encoded) {
                    Ok(bytes) => {
                        let key_count = if keys.is_empty() {
                            0
                        } else {
                            keys.split(',').count()
                        };
                        let expected = key_count * SECTION_EMBEDDING_BYTES;
                        if bytes.len() != expected {
                            outcome.warnings.push(format!(
                                "skipping handoff sidecar for {}: section_embeddings is {} bytes, expected {}",
                                memory.id,
                                bytes.len(),
                                expected
                            ));
                        } else if let Err(e) =
                            db.insert_handoff_sections(&memory.id, &sections, &keys, &bytes)
                        {
                            outcome.warnings.push(format!(
                                "failed to import handoff sidecar for {}: {e}",
                                memory.id
                            ));
                        }
                    }
                    Err(e) => outcome.warnings.push(format!(
                        "skipping handoff sidecar for {}: could not decode section_embeddings: {e}",
                        memory.id
                    )),
                }
            }
            _ => outcome.warnings.push(format!(
                "handoff {} imported without sidecar (old export format; sections not available)",
                memory.id
            )),
        }
    }

    if !is_update
        && memory.memory_type == MemoryType::Adr
        && let (Some(num), Some(status_str), Some(adr_sections)) =
            (adr_number, adr_status, adr_sections)
    {
        match status_str.parse::<AdrStatus>() {
            Ok(adr_status) => {
                if let Err(e) = db.insert_adr_sidecar(
                    &memory.id,
                    target_project,
                    num,
                    adr_status,
                    &adr_sections,
                    memory.created_at,
                    memory.updated_at,
                ) {
                    outcome.warnings.push(format!(
                        "failed to insert ADR sidecar for {} (number {num}): {e}",
                        memory.id
                    ));
                }
            }
            Err(e) => outcome.warnings.push(format!(
                "skipping ADR sidecar for {}: invalid status '{status_str}': {e}",
                memory.id
            )),
        }
    }

    // Retrieval status and todo lifecycle are LWW-governed, and both writes are upserts,
    // so they apply to new and updated memories alike.
    if let Some(status) = status {
        db.set_dead_at(
            &memory.id,
            status.dead,
            status.reason.as_deref(),
            status.marked_at,
        )?;
    }
    if let Some(todo) = todo {
        db.upsert_todo_item(
            &memory.id,
            todo.status,
            todo.reason.as_deref(),
            todo.closed_at,
        )?;
    }

    Ok(())
}
