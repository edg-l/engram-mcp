use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::db::{Database, encode_section_embeddings};
use crate::embedding::{EmbeddingService, cosine_similarity};
use crate::error::MemoryError;
use crate::memory::{HandoffSections, Memory, MemoryType, RelationType, Relationship};
use crate::summarize::{generate_summary, should_auto_summarize};

// ============================================
// Handoff result structs
// ============================================

/// Result returned by `handoff_create`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffCreateResult {
    /// ID of the newly created handoff memory.
    pub id: String,
    /// IDs of memories that were auto-linked to this handoff.
    pub linked_memory_ids: Vec<String>,
    /// The `continues_from` field from the sections (sidecar-only, not a graph edge).
    pub continues_from: Option<String>,
}

/// A scored section match returned by `handoff_resume`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffSectionMatch {
    /// ID of the handoff memory that contains this section.
    pub handoff_id: String,
    /// Section name, e.g. "summary", "blockers".
    pub section_name: String,
    /// Full text of the section.
    pub section_text: String,
    /// Cosine similarity score against the query embedding.
    pub score: f32,
}

/// Result returned by `handoff_resume`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffResumeResult {
    /// Resolved branch, or None if no branch could be determined.
    pub branch: Option<String>,
    /// ID of the most recent handoff on the branch.
    pub latest_handoff_id: Option<String>,
    /// Ordered chain of handoff IDs, oldest to newest, via `continues_from`.
    pub chain: Vec<String>,
    /// Top-scoring section excerpts across all handoffs in the chain.
    pub top_sections: Vec<HandoffSectionMatch>,
    /// Memories auto-linked (derived_from) from the latest handoff.
    pub linked_memories: Vec<Memory>,
    /// Explanatory message, e.g. when no branch was detected.
    pub message: Option<String>,
}

/// Result returned by `handoff_search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffSearchResult {
    /// Scored section matches, sorted by similarity descending.
    pub matches: Vec<HandoffSectionMatch>,
}

// ============================================
// Handoff free functions
// ============================================

/// Create a session handoff memory with per-section embeddings and optional auto-linking.
///
/// The `branch` parameter should be the resolved current branch (or an explicit override).
/// If `None`, the function returns `MemoryError::InvalidType` — handoffs require a branch.
#[allow(clippy::too_many_arguments)]
pub fn create_handoff(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
    branch: Option<&str>,
    sections: HandoffSections,
    importance: f64,
    pinned: bool,
    auto_link: bool,
) -> Result<HandoffCreateResult, MemoryError> {
    // Step 1: resolve branch.
    let resolved_branch = match branch {
        Some(b) => b.to_string(),
        None => {
            return Err(MemoryError::InvalidType(
                "handoff requires a branch; detached HEAD or non-git workspace not supported"
                    .to_string(),
            ));
        }
    };

    // Step 2: render markdown for the main memory content.
    let content = sections.render_markdown();

    // Step 3: build the Memory struct.
    let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
    let now = chrono::Utc::now().timestamp();
    let summary = if should_auto_summarize(&content, None) {
        Some(generate_summary(&content))
    } else {
        None
    };

    let memory = Memory {
        id: id.clone(),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Handoff,
        content: content.clone(),
        summary,
        tags: vec![],
        importance: importance.clamp(0.0, 1.0),
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: Some(resolved_branch),
        merged_from: None,
        external_artifacts: None,
        pinned,
        global: false,
    };

    // Step 4: generate full-content embedding.
    let full_embedding = embedding.embed_memory(MemoryType::Handoff, &content)?;

    // Step 5: generate per-section embeddings (prefix-free query path).
    // Section embeddings use the prefix-free query path so resume scoring stays in one vector space.
    let section_texts = handoff_section_key_texts(&sections);

    let mut section_keys: Vec<&str> = Vec::new();
    let mut section_vecs: Vec<Vec<f32>> = Vec::new();
    for (key, text) in &section_texts {
        let vec = embedding.embed(text)?;
        section_keys.push(key);
        section_vecs.push(vec);
    }

    // Step 6: encode section embeddings into wire format.
    let (section_keys_str, section_bytes) = encode_section_embeddings(&section_keys, &section_vecs);

    // Step 7: single transaction — insert memory, embedding, handoff_sections sidecar atomically.
    // continues_from is sidecar-only; single source of truth in handoff_sections.
    db.store_handoff_atomic(
        &memory,
        &full_embedding,
        embedding.model_version(),
        &sections,
        &section_keys_str,
        &section_bytes,
    )?;

    // Step 8: continues_from is stored only in the sidecar (above). No derived_from relationship.

    // Step 9: auto-link to related memories.
    let linked_memory_ids = if auto_link {
        auto_link_handoff_sections(db, embedding, &id, &sections, project_id)?
    } else {
        Vec::new()
    };

    // Step 10: return result.
    let continues_from = sections.continues_from.clone();
    Ok(HandoffCreateResult {
        id,
        linked_memory_ids,
        continues_from,
    })
}

/// Build the ordered list of (section_key, section_text) pairs for a `HandoffSections`.
///
/// Order matches `HandoffSections::render_markdown` exactly. Only non-empty sections are
/// included. This is the single canonical source for the key/text pairing used when
/// generating and encoding per-section embeddings.
pub fn handoff_section_key_texts(sections: &HandoffSections) -> Vec<(&'static str, String)> {
    let mut v = Vec::new();
    if !sections.summary.is_empty() {
        v.push(("summary", sections.summary.clone()));
    }
    if !sections.decisions.is_empty() {
        v.push(("decisions", sections.decisions.join("\n")));
    }
    if !sections.todos.is_empty() {
        v.push(("todos", sections.todos.join("\n")));
    }
    if !sections.blockers.is_empty() {
        v.push(("blockers", sections.blockers.join("\n")));
    }
    if !sections.mental_model.is_empty() {
        v.push(("mental_model", sections.mental_model.clone()));
    }
    if !sections.next_steps.is_empty() {
        v.push(("next_steps", sections.next_steps.join("\n")));
    }
    if let Some(ref notes) = sections.notes
        && !notes.is_empty()
    {
        v.push(("notes", notes.clone()));
    }
    v
}

/// Auto-link a handoff to related decisions, patterns, and debug memories via `derived_from`.
///
/// For each non-empty section text, candidates of types Decision, Pattern, and Debug are
/// scored using `embed_memory_text` (same document prefix used at storage time) so the
/// comparison lives in the same vector space as the stored embeddings.  The prefix-free
/// `embed` vectors stored in the sidecar are NOT used here — those are for resume scoring.
///
/// At most 10 links are created across all sections, preferring highest similarity (>= 0.75).
pub fn auto_link_handoff_sections(
    db: &Database,
    embedding: &EmbeddingService,
    handoff_id: &str,
    sections: &HandoffSections,
    project_id: &str,
) -> Result<Vec<String>, MemoryError> {
    const AUTO_LINK_THRESHOLD: f32 = 0.75;
    const AUTO_LINK_CAP: usize = 10;
    let target_types = [MemoryType::Decision, MemoryType::Pattern, MemoryType::Debug];

    // Collect candidate memories of the target types.
    let candidates: Vec<Memory> =
        db.query_memories(project_id, Some(&target_types), None, None, 1000)?;

    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Fetch stored embeddings for candidates (batch).
    let candidate_ids: Vec<String> = candidates.iter().map(|m| m.id.clone()).collect();
    let stored_embeddings = db.get_embeddings_batch(&candidate_ids)?;
    let embedding_map: std::collections::HashMap<String, Vec<f32>> =
        stored_embeddings.into_iter().collect();
    let candidate_map: std::collections::HashMap<String, &Memory> =
        candidates.iter().map(|m| (m.id.clone(), m)).collect();

    // Collect non-empty section texts.
    let section_texts = handoff_section_key_texts(sections);

    // For each section × each target type, embed using the document prefix and score candidates.
    // Collect (similarity, memory_id) across all combinations.
    let mut scored: Vec<(f32, String)> = Vec::new();
    let mut already_linked: HashSet<String> = HashSet::new();

    for (_section_name, section_text) in &section_texts {
        for &target_type in &target_types {
            let section_vec = embedding.embed_memory_text(target_type, section_text)?;
            for m in &candidates {
                if m.memory_type != target_type {
                    continue;
                }
                if let Some(stored_vec) = embedding_map.get(&m.id) {
                    let sim = cosine_similarity(&section_vec, stored_vec);
                    if sim >= AUTO_LINK_THRESHOLD {
                        scored.push((sim, m.id.clone()));
                    }
                }
            }
        }
    }

    // Sort by similarity descending, deduplicate by memory ID, cap at 10.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut linked_ids: Vec<String> = Vec::new();
    let now = chrono::Utc::now().timestamp();

    for (_, candidate_id) in scored {
        if linked_ids.len() >= AUTO_LINK_CAP {
            break;
        }
        if already_linked.contains(&candidate_id) {
            continue;
        }
        // Verify the candidate still exists in our map.
        if !candidate_map.contains_key(&candidate_id) {
            continue;
        }
        let rel = Relationship {
            id: format!("rel_{}", uuid::Uuid::new_v4().simple()),
            source_id: handoff_id.to_string(),
            target_id: candidate_id.clone(),
            relation_type: RelationType::DerivedFrom,
            strength: 1.0,
            created_at: now,
        };
        db.create_relationship(&rel)?;
        already_linked.insert(candidate_id.clone());
        linked_ids.push(candidate_id);
    }

    Ok(linked_ids)
}

/// Core of `resume_handoff` that accepts a pre-computed query embedding.
///
/// Separated from the embedding step so the scoring logic can be exercised in tests
/// without a real `EmbeddingService`.  Pass `query_vec=None` to skip scoring (returns
/// no section matches); pass `Some(vec)` to score all sections.
pub fn resume_handoff_with_vec(
    db: &Database,
    project_id: &str,
    branch: Option<&str>,
    query_vec: Option<Vec<f32>>,
    max_sections: usize,
    include_off_branch: bool,
) -> Result<HandoffResumeResult, MemoryError> {
    // Step 1: resolve branch; if unresolved, serve off-branch handoffs with explanatory message.
    let (resolved_branch, message, fetch_all) = match branch {
        Some(b) if !b.is_empty() => (Some(b.to_string()), None, include_off_branch),
        _ => (
            None,
            Some("No current branch detected; returning off-branch handoffs only.".to_string()),
            true,
        ),
    };

    // Step 2: fetch latest handoffs.
    let latest_list = if fetch_all {
        db.list_recent_handoffs(project_id, 10)?
    } else {
        db.query_handoffs_by_branch(project_id, resolved_branch.as_deref(), 10)?
    };

    let Some(latest) = latest_list.into_iter().next() else {
        return Ok(HandoffResumeResult {
            branch: resolved_branch,
            latest_handoff_id: None,
            chain: Vec::new(),
            top_sections: Vec::new(),
            linked_memories: Vec::new(),
            message,
        });
    };

    let latest_id = latest.id.clone();

    // Step 3: walk continues_from chain backwards from latest (up to depth 5).
    let mut chain_ids: Vec<String> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut current_id = latest_id.clone();

    loop {
        if visited.contains(&current_id) {
            break; // Cycle detected.
        }
        if chain_ids.len() >= 5 {
            break; // Depth cap.
        }
        visited.insert(current_id.clone());
        chain_ids.push(current_id.clone());

        // Follow continues_from link.
        match db.get_handoff_sections(&current_id)? {
            Some((sections, _)) => match sections.continues_from {
                Some(prev_id) => {
                    current_id = prev_id;
                }
                None => break,
            },
            None => break,
        }
    }

    // Reverse to oldest-first order.
    chain_ids.reverse();

    // Step 4: score every section across all chain handoffs (only when a query vec is available).
    // Load Memory structs for the chain so score_handoff_sections can iterate them.
    let chain_memories: Vec<Memory> = chain_ids
        .iter()
        .filter_map(|hid| db.get_memory(hid).ok().flatten())
        .collect();

    let top_sections = if let Some(ref qvec) = query_vec {
        let mut all_matches = score_handoff_sections(qvec, &chain_memories, db)?;
        all_matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_matches.into_iter().take(max_sections).collect()
    } else {
        Vec::new()
    };

    // Step 5: fetch derived_from linked memories for the latest handoff.
    let derived_rels = db.get_relationships_from(&latest_id)?;
    let derived_target_ids: Vec<String> = derived_rels
        .into_iter()
        .filter(|r| r.relation_type == RelationType::DerivedFrom)
        .map(|r| r.target_id)
        .collect();

    let linked_memories_map = db.get_memories_batch(&derived_target_ids)?;
    let mut linked_memories: Vec<Memory> = derived_target_ids
        .iter()
        .filter_map(|id| linked_memories_map.get(id).cloned())
        .collect();

    // Step 5b: single-handoff diversity backfill.
    // When the chain contains only one handoff, there is no cross-handoff diversity in
    // top_sections. If a query vector is available and the section budget has remaining
    // slots, fill them with the highest-similarity Decision/Pattern/Debug memories.
    let single_handoff = chain_ids.len() == 1;
    let mut backfill_message: Option<String> = None;
    if single_handoff
        && let Some(qvec) = query_vec.as_ref()
        && top_sections.len() < max_sections
    {
        let remaining_slots = max_sections - top_sections.len();

        // IDs already represented — deduplicate against the chain and existing linked_memories.
        let mut seen_ids: HashSet<String> = chain_ids.iter().cloned().collect();
        for lm in &linked_memories {
            seen_ids.insert(lm.id.clone());
        }

        let embeddings = db.get_all_embeddings_for_project_and_global(project_id)?;
        let mems = db.get_all_memories_for_project(project_id)?;
        let mem_map: HashMap<String, Memory> =
            mems.into_iter().map(|m| (m.id.clone(), m)).collect();

        const BACKFILL_TYPES: [MemoryType; 3] =
            [MemoryType::Decision, MemoryType::Pattern, MemoryType::Debug];

        let mut scored: Vec<(f32, String)> = embeddings
            .into_iter()
            .filter_map(|(id, vec)| {
                if seen_ids.contains(&id) {
                    return None;
                }
                let mem = mem_map.get(&id)?;
                if !BACKFILL_TYPES.contains(&mem.memory_type) {
                    return None;
                }
                // Branch filter: candidate must match the handoff branch or be global.
                if let Some(ref hbranch) = resolved_branch {
                    match &mem.branch {
                        Some(mbranch) if mbranch == hbranch => {}
                        None => {}        // global memory — always included
                        _ => return None, // different branch
                    }
                }
                let sim = cosine_similarity(qvec, &vec);
                Some((sim, id))
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut added = 0usize;
        for (_, id) in scored.into_iter().take(remaining_slots) {
            if let Some(mem) = mem_map.get(&id) {
                linked_memories.push(mem.clone());
                added += 1;
            }
        }

        if added >= 1 {
            backfill_message = Some(
                "Single handoff on branch; supplemented with related decisions/patterns/debug memories."
                    .to_string(),
            );
        }
    }

    // Step 6: record access on latest handoff.
    let _ = db.record_access(&latest_id);

    let final_message = backfill_message.or(message);

    Ok(HandoffResumeResult {
        branch: resolved_branch,
        latest_handoff_id: Some(latest_id),
        chain: chain_ids,
        top_sections,
        linked_memories,
        message: final_message,
    })
}

/// Resume a session by retrieving top sections from recent handoffs and linked memories.
///
/// Resolves the branch, fetches the handoff chain (up to depth 5), scores sections
/// against the query embedding, and returns the top `max_sections` matches along with
/// auto-linked memories from the latest handoff.
pub fn resume_handoff(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
    branch: Option<&str>,
    query: Option<&str>,
    max_sections: usize,
    include_off_branch: bool,
) -> Result<HandoffResumeResult, MemoryError> {
    // Resolve the query text, embed it, then delegate to the core implementation.
    // We need the latest handoff to derive a fallback query text when query=None.
    // Peek at the DB to get the fallback before calling the inner function.
    let (resolved_branch_peek, _, fetch_all_peek) = match branch {
        Some(b) if !b.is_empty() => (Some(b.to_string()), None::<String>, include_off_branch),
        _ => (
            None,
            Some("No current branch detected; returning off-branch handoffs only.".to_string()),
            true,
        ),
    };

    // Determine query text for embedding.
    let query_vec = if let Some(q) = query
        && !q.is_empty()
    {
        Some(embedding.embed(q)?)
    } else {
        // Fallback: use latest handoff summary.
        let latest_list = if fetch_all_peek {
            db.list_recent_handoffs(project_id, 1)?
        } else {
            db.query_handoffs_by_branch(project_id, resolved_branch_peek.as_deref(), 1)?
        };
        if let Some(latest) = latest_list.into_iter().next() {
            let fallback_text = match db.get_handoff_sections(&latest.id)? {
                Some((s, _)) => s.summary,
                None => latest
                    .summary
                    .clone()
                    .unwrap_or_else(|| latest.content.chars().take(200).collect()),
            };
            Some(embedding.embed(&fallback_text)?)
        } else {
            // No handoffs exist; inner function will return an empty result.
            None
        }
    };

    resume_handoff_with_vec(
        db,
        project_id,
        branch,
        query_vec,
        max_sections,
        include_off_branch,
    )
}

/// Extract the text content for a named section from `HandoffSections`.
pub fn get_section_text(sections: &HandoffSections, section_name: &str) -> String {
    match section_name {
        "summary" => sections.summary.clone(),
        "decisions" => sections.decisions.join("\n"),
        "todos" => sections.todos.join("\n"),
        "blockers" => sections.blockers.join("\n"),
        "mental_model" => sections.mental_model.clone(),
        "next_steps" => sections.next_steps.join("\n"),
        "notes" => sections.notes.clone().unwrap_or_default(),
        _ => String::new(),
    }
}

/// Score every section of the given handoff memories against a query vector.
///
/// Returns all matches (unsorted, unfiltered). The caller decides limit, ordering,
/// and any section name filtering. Uses the same loop structure as the inner scoring
/// in `resume_handoff_with_vec` to keep behaviour identical post-refactor.
pub fn score_handoff_sections(
    query_vec: &[f32],
    handoffs: &[Memory],
    db: &Database,
) -> Result<Vec<HandoffSectionMatch>, MemoryError> {
    let mut all_matches: Vec<HandoffSectionMatch> = Vec::new();

    for handoff in handoffs {
        if let Some((sections_struct, section_vecs)) = db.get_handoff_sections(&handoff.id)? {
            // section_vecs from get_handoff_sections are the prefix-free stored embeddings.
            for (section_name, section_vec) in section_vecs {
                let score = cosine_similarity(query_vec, &section_vec);
                let section_text = get_section_text(&sections_struct, &section_name);
                all_matches.push(HandoffSectionMatch {
                    handoff_id: handoff.id.clone(),
                    section_name,
                    section_text,
                    score,
                });
            }
        }
    }

    Ok(all_matches)
}

/// Core of `handoff_search` that accepts a pre-computed query vector.
///
/// Separated from the embedding step so the scoring logic can be exercised in tests
/// without a real `EmbeddingService`.
pub fn search_handoffs_with_vec(
    db: &Database,
    project_id: &str,
    query_vec: Vec<f32>,
    branch: Option<&str>,
    limit: usize,
    section_filter: Option<&[String]>,
) -> Result<HandoffSearchResult, MemoryError> {
    // Fetch handoffs matching the branch filter. branch=None means all branches.
    let handoffs = if let Some(b) = branch {
        db.query_handoffs_by_branch(project_id, Some(b), 200)?
    } else {
        db.list_recent_handoffs(project_id, 200)?
    };

    // Score all sections against the query vector.
    let mut all_matches = score_handoff_sections(&query_vec, &handoffs, db)?;

    // Apply section_filter if present (case-insensitive match against section names).
    if let Some(filter) = section_filter {
        let filter_lower: Vec<String> = filter.iter().map(|s| s.to_lowercase()).collect();
        all_matches.retain(|m| filter_lower.contains(&m.section_name.to_lowercase()));
    }

    // Sort by score descending and take top limit.
    all_matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let matches = all_matches.into_iter().take(limit).collect();

    Ok(HandoffSearchResult { matches })
}

/// Search session handoffs by section content, embedding the query via prefix-free `embed`.
pub fn search_handoffs(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
    query: &str,
    branch: Option<&str>,
    limit: usize,
    section_filter: Option<&[String]>,
) -> Result<HandoffSearchResult, MemoryError> {
    let query_vec = embedding.embed(query)?;
    search_handoffs_with_vec(db, project_id, query_vec, branch, limit, section_filter)
}
