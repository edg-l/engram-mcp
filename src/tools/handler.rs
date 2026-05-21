#![allow(dead_code)]

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::OnceLock;

use serde::Serialize;
use serde_json::{Value, json};

use crate::cache::{QueryEmbeddingCache, SearchResultCache};
use crate::db::{Database, encode_section_embeddings};
use crate::embedding::{EmbeddingService, cosine_similarity};
use crate::error::MemoryError;
use crate::export::{self, ExportData, ExportedMemory, HandoffSidecar, ImportMode, ImportStats};
use crate::memory::{
    HandoffSections, Memory, MemoryType, MemoryWithScore, ProjectStats, RelationType, Relationship,
};
use crate::summarize::{generate_summary, should_auto_summarize};

use super::handoff::{create_handoff, handoff_section_key_texts, resume_handoff, search_handoffs};
use super::schemas::{
    HandoffCreateInput, HandoffResumeInput, HandoffSearchInput, MemoryContextInput,
    MemoryDedupInput, MemoryDeleteBatchInput, MemoryDeleteInput, MemoryExportInput,
    MemoryGraphInput, MemoryImportInput, MemoryLinkInput, MemoryPromoteInput, MemoryPruneInput,
    MemoryQueryInput, MemoryStoreBatchInput, MemoryStoreInput, MemoryUpdateInput, ToolProfile,
    dedup_threshold, get_tool_definitions_for,
};
use super::scoring::{
    SearchMode, apply_tag_and_relevance, compute_context_score, compute_hybrid_score,
    compute_tag_boost, rrf_fuse,
};

// ============================================
// Per-process profile + once-warning
// ============================================

/// Active tool profile for this process. Initialized on first dispatch from
/// the ENGRAM_MCP_TOOL_PROFILE env var (mirrors the read in MemoryServer::new).
static ACTIVE_PROFILE: OnceLock<ToolProfile> = OnceLock::new();

fn active_profile() -> ToolProfile {
    *ACTIVE_PROFILE.get_or_init(|| {
        std::env::var("ENGRAM_MCP_TOOL_PROFILE")
            .ok()
            .and_then(|raw| raw.parse::<ToolProfile>().ok())
            .unwrap_or_default()
    })
}

static WARNED_TOOLS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

/// Tool names advertised under the active profile, cached on first dispatch to
/// avoid rebuilding the full `Vec<Tool>` (with all JSON schema payloads) on
/// every call.
static ADVERTISED_NAMES: OnceLock<HashSet<String>> = OnceLock::new();

fn advertised_names() -> &'static HashSet<String> {
    ADVERTISED_NAMES.get_or_init(|| {
        get_tool_definitions_for(active_profile())
            .into_iter()
            .map(|t| t.name.to_string())
            .collect()
    })
}

fn warn_unavailable_once(tool: &str, profile: ToolProfile) {
    let set = WARNED_TOOLS.get_or_init(|| Mutex::new(HashSet::new()));
    let is_new = set.lock().unwrap().insert(tool.to_string());
    if is_new {
        eprintln!(
            "[engram] tool '{tool}' is not advertised under profile {profile:?} but was dispatched; this name may be hidden from future MCP `tools/list` responses."
        );
    }
}

// ============================================
// Result structs
// ============================================

#[derive(Debug, Serialize)]
pub struct MergeInfo {
    pub merged_with: String,
    pub similarity: f64,
    pub old_content_preview: String,
}

#[derive(Debug, Serialize)]
pub struct MemoryStoreResult {
    pub id: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub potential_contradictions: Vec<PotentialContradiction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge_info: Option<MergeInfo>,
}

#[derive(Debug, Serialize)]
pub struct PotentialContradiction {
    pub memory_id: String,
    pub summary: String,
    pub similarity: f64,
}

#[derive(Debug, Serialize)]
pub struct ContradictionWarning {
    pub memory_id: String,
    pub contradicts_id: String,
    pub content_preview: String,
}

#[derive(Debug, Serialize)]
pub struct MemoryQueryResult {
    pub memories: Vec<MemoryWithScore>,
    pub count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub contradiction_warnings: Vec<ContradictionWarning>,
}

#[derive(Debug, Serialize)]
pub struct MemoryGraphResult {
    pub root: Memory,
    pub related: Vec<RelatedMemory>,
}

#[derive(Debug, Serialize)]
pub struct RelatedMemory {
    pub memory: Memory,
    pub relation: String,
    pub direction: String,
    pub depth: usize,
}

// ============================================
// ToolHandler
// ============================================

/// Parse the `ENGRAM_SEARCH_MODE` env value into a `SearchMode`.
///
/// - `None` or empty string → `SearchMode::Hybrid` (default).
/// - Recognized values: "vector", "bm25", "hybrid" (case-insensitive).
/// - Unrecognized value: logs a warning and returns `SearchMode::Hybrid`.
pub fn parse_search_mode(env_val: Option<&str>) -> SearchMode {
    match env_val {
        None | Some("") => SearchMode::Hybrid,
        Some(s) => s.parse().unwrap_or_else(|e| {
            tracing::warn!("ENGRAM_SEARCH_MODE: {e}; falling back to hybrid");
            SearchMode::Hybrid
        }),
    }
}

pub struct ToolHandler {
    db: Database,
    embedding: EmbeddingService,
    project_id: String,
    /// Current git branch (None if not in git repo)
    current_branch: Option<String>,
    /// Retrieval strategy: Vector, Bm25, or Hybrid (default).
    search_mode: SearchMode,
    /// Cache for query embeddings to avoid recomputation
    query_cache: QueryEmbeddingCache,
    /// Cache for search results to avoid repeated similarity computations
    search_cache: SearchResultCache,
}

impl ToolHandler {
    pub fn new(
        db: Database,
        embedding: EmbeddingService,
        project_id: String,
        current_branch: Option<String>,
        search_mode: SearchMode,
    ) -> Self {
        Self {
            db,
            embedding,
            project_id,
            current_branch,
            search_mode,
            query_cache: QueryEmbeddingCache::new(),
            search_cache: SearchResultCache::new(),
        }
    }

    /// Get the current branch.
    pub fn current_branch(&self) -> Option<&str> {
        self.current_branch.as_deref()
    }

    /// Get a reference to the embedding service for reuse.
    pub fn embedding_service(&self) -> &EmbeddingService {
        &self.embedding
    }

    /// Get a reference to the database for reuse.
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Get the project ID.
    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    /// Convert branch_mode string to the Option<Option<&str>> format for DB queries.
    /// - "current" -> global + current branch (falls back to "all" if no branch detected)
    /// - "global" -> global only (branch IS NULL)
    /// - "all" -> no filter
    /// - other -> global + that specific branch
    fn branch_mode_to_filter<'a>(&'a self, branch_mode: &'a str) -> Option<Option<&'a str>> {
        match branch_mode {
            "all" => None,
            "global" => Some(None),
            "current" => {
                match self.current_branch.as_deref() {
                    Some(branch) => Some(Some(branch)),
                    None => None, // Fall back to "all" if no branch detected
                }
            }
            specific => Some(Some(specific)),
        }
    }

    /// Invalidate search result cache (call after memory modifications).
    fn invalidate_search_cache(&self) {
        self.search_cache.invalidate_project(&self.project_id);
    }

    pub fn handle_tool(&self, name: &str, arguments: Value) -> Result<Value, MemoryError> {
        // Warn once per process if a tool is called that isn't advertised under the active profile.
        if !advertised_names().contains(name) {
            warn_unavailable_once(name, active_profile());
        }

        match name {
            "memory_store" => self.memory_store(arguments),
            "memory_query" => self.memory_query(arguments),
            "memory_update" => self.memory_update(arguments),
            "memory_delete" => self.memory_delete(arguments),
            "memory_link" => self.memory_link(arguments),
            "memory_graph" => self.memory_graph(arguments),
            "memory_store_batch" => self.memory_store_batch(arguments),
            "memory_delete_batch" => self.memory_delete_batch(arguments),
            "memory_export" => self.memory_export(arguments),
            "memory_import" => self.memory_import(arguments),
            "memory_stats" => self.memory_stats(arguments),
            "memory_context" => self.memory_context(arguments),
            "memory_prune" => self.memory_prune(arguments),
            "memory_promote" => self.memory_promote(arguments),
            "memory_dedup" => self.memory_dedup(arguments),
            "handoff_create" => self.handoff_create(arguments),
            "handoff_resume" => self.handoff_resume(arguments),
            "handoff_search" => self.handoff_search(arguments),
            _ => Ok(json!({"error": format!("Unknown tool: {}", name)})),
        }
    }

    fn memory_store(&self, arguments: Value) -> Result<Value, MemoryError> {
        use super::store::{StoreOutcome, store_with_dedup};

        let input: MemoryStoreInput = serde_json::from_value(arguments)?;

        let memory_type: MemoryType = input
            .memory_type
            .parse()
            .map_err(|_| MemoryError::InvalidType(input.memory_type.clone()))?;

        let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
        let now = chrono::Utc::now().timestamp();

        // Auto-generate summary if needed
        let summary = if should_auto_summarize(&input.content, input.summary.as_deref()) {
            Some(generate_summary(&input.content))
        } else {
            input.summary
        };

        // Resolve branch: null/omitted = global (None), "auto" = current branch, else explicit
        // If global=true, force branch to None regardless of what was passed
        let branch = if input.global {
            None
        } else {
            match input.branch.as_deref() {
                None | Some("") => None, // Global
                Some("auto") => self.current_branch.clone(),
                Some(explicit) => Some(explicit.to_string()),
            }
        };

        let memory = Memory {
            id: id.clone(),
            project_id: self.project_id.clone(),
            memory_type,
            content: input.content.clone(),
            summary,
            tags: input.tags,
            importance: input.importance.clamp(0.0, 1.0),
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: branch.clone(),
            merged_from: None,
            external_artifacts: input.external_artifacts,
            pinned: input.pinned,
            global: input.global,
        };

        // Generate embedding locally — needed for both dedup and contradiction scan.
        let embedding = self.embedding.embed_memory(memory_type, &input.content)?;

        // Fetch all embeddings and memories now for the contradiction scan after store.
        // (store_with_dedup will re-fetch internally for dedup; that's acceptable here.)
        let pre_store_embeddings = self
            .db
            .get_all_embeddings_for_project_and_global(&self.project_id)?;
        let pre_store_memories_list = self.db.get_all_memories_for_project(&self.project_id)?;
        let pre_store_memories: std::collections::HashMap<String, Memory> = pre_store_memories_list
            .into_iter()
            .map(|m| (m.id.clone(), m))
            .collect();

        // Handoffs are session snapshots; bypass dedup.
        // Pass None for embedding_service to skip dedup for handoffs.
        let dedup_thr = dedup_threshold();
        let outcome = if memory_type != MemoryType::Handoff {
            store_with_dedup(
                &self.db,
                Some(&self.embedding),
                &self.project_id,
                memory,
                Some(&embedding),
                dedup_thr,
                None, // MCP path never skips — always merges duplicates
            )?
        } else {
            // Handoff: store directly, bypassing dedup.
            self.db.store_memory(&memory)?;
            self.db
                .store_embedding(&id, &embedding, self.embedding.model_version())?;
            StoreOutcome::Stored(id.clone())
        };

        // Map store outcome to MergeInfo.
        let (final_id, merge_info) = match outcome {
            StoreOutcome::Stored(stored_id) => (stored_id, None),
            StoreOutcome::Merged {
                id: stored_id,
                merged_with,
                similarity,
            } => {
                let old_preview: String = pre_store_memories
                    .get(&merged_with)
                    .map(|m| m.content.chars().take(100).collect())
                    .unwrap_or_default();
                (
                    stored_id,
                    Some(MergeInfo {
                        merged_with,
                        similarity,
                        old_content_preview: old_preview,
                    }),
                )
            }
            StoreOutcome::SkippedSimilar { .. } => {
                unreachable!("MCP path passes skip_above=None; SkippedSimilar cannot occur")
            }
        };

        // Contradictions are only meaningful within the same non-handoff type; handoffs are session snapshots and warrant no warning either way.
        let mut potential_contradictions: Vec<PotentialContradiction> = Vec::new();
        const CONTRADICTION_THRESHOLD: f32 = 0.85;
        let merged_id: Option<&str> = merge_info.as_ref().map(|mi| mi.merged_with.as_str());

        if memory_type != MemoryType::Handoff {
            for (existing_id, existing_vec) in &pre_store_embeddings {
                if existing_id.as_str() == final_id.as_str() {
                    continue; // Skip self
                }
                if merged_id == Some(existing_id.as_str()) {
                    continue; // Skip the merged-away duplicate
                }
                let Some(existing_memory) = pre_store_memories.get(existing_id) else {
                    continue; // Defensive: embedding exists but memory row missing
                };
                if existing_memory.memory_type == MemoryType::Handoff {
                    continue; // Handoff on the existing side: skip
                }
                if existing_memory.memory_type != memory_type {
                    continue; // Cross-type: not a meaningful contradiction
                }
                let similarity = cosine_similarity(&embedding, existing_vec);
                if similarity >= CONTRADICTION_THRESHOLD {
                    potential_contradictions.push(PotentialContradiction {
                        memory_id: existing_id.clone(),
                        summary: existing_memory
                            .summary
                            .clone()
                            .unwrap_or_else(|| existing_memory.content.chars().take(100).collect()),
                        similarity: similarity as f64,
                    });
                }
            }
        } // end if memory_type != MemoryType::Handoff (contradiction check)

        // Create relationships to related memories
        for related_id in input.related_to {
            let rel = Relationship {
                id: format!("rel_{}", uuid::Uuid::new_v4().simple()),
                source_id: final_id.clone(),
                target_id: related_id,
                relation_type: RelationType::RelatesTo,
                strength: 1.0,
                created_at: now,
            };
            self.db.create_relationship(&rel)?;
        }

        // Assign to cluster
        let _cluster_id = self.assign_to_cluster(
            &final_id,
            &embedding,
            &input.content,
            input.importance.clamp(0.0, 1.0),
        )?;

        // Invalidate search cache since we added new data
        self.invalidate_search_cache();

        let message = if merge_info.is_some() {
            "Memory stored and merged with duplicate".to_string()
        } else if potential_contradictions.is_empty() {
            "Memory stored successfully".to_string()
        } else {
            format!(
                "Memory stored. Warning: {} potential contradiction(s) detected - consider using memory_link with 'supersedes' or 'contradicts' relation.",
                potential_contradictions.len()
            )
        };

        Ok(json!(MemoryStoreResult {
            id: final_id,
            message,
            branch,
            potential_contradictions,
            merge_info,
        }))
    }

    fn memory_query(&self, arguments: Value) -> Result<Value, MemoryError> {
        let mut input: MemoryQueryInput = serde_json::from_value(arguments)?;
        input.limit = input.limit.min(100); // Server-side cap to prevent overflow

        // Parse type filters
        let type_filters: Vec<MemoryType> =
            input.types.iter().filter_map(|t| t.parse().ok()).collect();

        // Optimization: if query is empty, skip search and use filter-only path
        if input.query.trim().is_empty() {
            let branch_filter = self.branch_mode_to_filter(&input.branch_mode);
            let memories = self.db.query_memories_with_branch(
                &self.project_id,
                if type_filters.is_empty() {
                    None
                } else {
                    Some(&type_filters)
                },
                if input.tags.is_empty() {
                    None
                } else {
                    Some(&input.tags)
                },
                Some(input.min_relevance),
                input.limit + input.offset,
                branch_filter,
            )?;

            let results: Vec<MemoryWithScore> = memories
                .into_iter()
                .skip(input.offset)
                .take(input.limit)
                .map(|m| {
                    let score = m.relevance_score;
                    MemoryWithScore {
                        memory: m,
                        score,
                        semantic_score: 0.0,
                        keyword_score: 0.0,
                        rrf_score: 0.0,
                    }
                })
                .collect();

            return Ok(json!(MemoryQueryResult {
                count: results.len(),
                memories: results,
                contradiction_warnings: vec![],
            }));
        }

        let branch_filter = self.branch_mode_to_filter(&input.branch_mode);

        // --- Embedding path (skipped for pure BM25 mode) ---
        let semantic_scores: std::collections::HashMap<String, f32> = if self.search_mode
            != SearchMode::Bm25
        {
            let query_embedding = if let Some(cached) = self.query_cache.get(&input.query) {
                cached
            } else {
                let embedding = self.embedding.embed(&input.query)?;
                self.query_cache
                    .insert(input.query.clone(), embedding.clone());
                embedding
            };

            if let Some(cached_results) = self.search_cache.get(&self.project_id, &query_embedding)
            {
                cached_results.into_iter().collect()
            } else {
                let embeddings = self
                    .db
                    .get_all_embeddings_for_project_and_global(&self.project_id)?;

                let scored: Vec<(String, f32)> = embeddings
                    .iter()
                    .map(|(id, vec)| {
                        let similarity = cosine_similarity(&query_embedding, vec);
                        (id.clone(), similarity)
                    })
                    .collect();

                self.search_cache
                    .insert(&self.project_id, &query_embedding, scored.clone());
                scored.into_iter().collect()
            }
        } else {
            std::collections::HashMap::new()
        };

        // --- BM25 path (skipped for pure Vector mode) ---
        // Returns ordered Vec<(id, raw_score)> from FTS5; we keep the ordering for RRF.
        let bm25_results: Vec<(String, f64)> = if self.search_mode != SearchMode::Vector {
            self.db.keyword_search_with_branch(
                &self.project_id,
                &input.query,
                input.limit * 5, // over-fetch so we have enough after per-memory filters
                branch_filter,
            )?
        } else {
            Vec::new()
        };

        // Collect candidate IDs from whichever rankers ran.
        let mut candidate_ids: HashSet<String> = semantic_scores.keys().cloned().collect();
        for (id, _) in &bm25_results {
            candidate_ids.insert(id.clone());
        }

        // Batch fetch all candidate memories once.
        let candidate_ids_vec: Vec<String> = candidate_ids.into_iter().collect();
        let memories_map = self.db.get_memories_batch(&candidate_ids_vec)?;

        // Extract normalized query words for tag boosting.
        let query_words: Vec<String> = input
            .query
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() >= 3)
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        // Build RRF lookup for Hybrid mode.
        // For Vector and Bm25 modes this map is empty; scoring uses a direct formula instead.
        let rrf_map: std::collections::HashMap<String, f64> = if self.search_mode
            == SearchMode::Hybrid
        {
            // Vector ranking: sort semantic scores descending by similarity.
            let mut vector_ranked: Vec<(&String, f32)> =
                semantic_scores.iter().map(|(id, &s)| (id, s)).collect();
            vector_ranked
                .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let vector_ranks: Vec<String> = vector_ranked
                .into_iter()
                .map(|(id, _)| id.clone())
                .collect();

            // BM25 ranking: already ordered by FTS5 score descending.
            let bm25_ranks: Vec<String> = bm25_results.iter().map(|(id, _)| id.clone()).collect();

            rrf_fuse(&[vector_ranks.as_slice(), bm25_ranks.as_slice()], 60.0)
                .into_iter()
                .collect()
        } else {
            std::collections::HashMap::new()
        };

        // Score, filter, and collect results.
        // Tuple layout: (id, final_score, raw_semantic, raw_keyword, rrf_score)
        let mut scored_results: Vec<(String, f64, f64, f64, f64)> = Vec::new();

        // Compute BM25 max once outside the loop; used to normalize the
        // diagnostic `keyword_score` field per result.
        let max_bm25 = bm25_results.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);

        for id in candidate_ids_vec.iter() {
            let Some(memory) = memories_map.get(id) else {
                continue;
            };

            // Filter by branch
            match branch_filter {
                None => {}                                         // "all" - no filtering
                Some(None) if memory.branch.is_some() => continue, // "global" - skip non-global
                Some(None) => {}
                Some(Some(branch)) => {
                    // specific branch - global + that branch
                    if let Some(ref mem_branch) = memory.branch
                        && mem_branch != branch
                    {
                        continue;
                    }
                    // branch is None (global) -> include
                }
            }

            // Filter by types
            if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
                continue;
            }

            // Filter by tags
            if !input.tags.is_empty() && !input.tags.iter().any(|t| memory.tags.contains(t)) {
                continue;
            }

            // Raw diagnostic scores (always populated regardless of mode).
            let raw_semantic = *semantic_scores.get(id).unwrap_or(&0.0) as f64;
            // Normalize BM25 across the returned set for the diagnostic field.
            let raw_keyword = if max_bm25 > 0.0 {
                bm25_results
                    .iter()
                    .find(|(bid, _)| bid == id)
                    .map(|(_, s)| s / max_bm25)
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            // Tag boost applied uniformly across all modes.
            let tag_boost = compute_tag_boost(&query_words, &memory.tags);

            let (base_score, rrf_score) = match self.search_mode {
                SearchMode::Vector => {
                    // Pure vector: base is cosine similarity.
                    (raw_semantic, 0.0)
                }
                SearchMode::Bm25 => {
                    // Pure BM25: find 0-based rank in bm25_results for RRF-style pseudo-score.
                    // Using the RRF pseudo-form keeps this scale consistent with Hybrid mode.
                    let rank = bm25_results
                        .iter()
                        .position(|(bid, _)| bid == id)
                        .unwrap_or(usize::MAX);
                    let pseudo = if rank == usize::MAX {
                        0.0
                    } else {
                        1.0 / (60.0 + rank as f64 + 1.0)
                    };
                    (pseudo, 0.0)
                }
                SearchMode::Hybrid => {
                    // RRF fused score; 0.0 for IDs absent from both rankers.
                    let fused = *rrf_map.get(id.as_str()).unwrap_or(&0.0);
                    (fused, fused)
                }
            };

            // Filter by decay relevance first; this is mode-agnostic.
            // RRF and BM25-pseudo scores are not on the 0-1 scale that
            // min_relevance was designed for, so we gate on the stored
            // relevance_score (the decay value) instead of final_score.
            if memory.relevance_score < input.min_relevance {
                continue;
            }

            let final_score =
                apply_tag_and_relevance(base_score, tag_boost, memory.relevance_score);

            scored_results.push((
                id.clone(),
                final_score,
                raw_semantic,
                raw_keyword,
                rrf_score,
            ));
        }

        // Sort by final score descending.
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply pagination and build final results.
        let mut results: Vec<MemoryWithScore> = Vec::new();
        let mut result_ids: Vec<String> = Vec::new();

        for (id, score, semantic_score, keyword_score, rrf_score) in scored_results
            .into_iter()
            .skip(input.offset)
            .take(input.limit)
        {
            if let Some(memory) = memories_map.get(&id) {
                result_ids.push(id);
                let mut memory_clone = memory.clone();
                memory_clone.access_count += 1;
                results.push(MemoryWithScore {
                    memory: memory_clone,
                    score,
                    semantic_score,
                    keyword_score,
                    rrf_score,
                });
            }
        }

        // Batch record access for all result memories
        if !result_ids.is_empty() {
            let _ = self.db.record_access_batch(&result_ids);
        }

        // Batch check for contradiction relationships among returned memories
        let contradiction_warnings = self.check_contradictions_batch(&result_ids)?;

        Ok(json!(MemoryQueryResult {
            count: results.len(),
            memories: results,
            contradiction_warnings,
        }))
    }

    /// Check for contradiction relationships among a set of memory IDs using batch operations.
    fn check_contradictions_batch(
        &self,
        result_ids: &[String],
    ) -> Result<Vec<ContradictionWarning>, MemoryError> {
        if result_ids.is_empty() {
            return Ok(Vec::new());
        }

        let result_id_set: HashSet<&String> = result_ids.iter().collect();

        // Batch fetch all outgoing relationships (1 query instead of N)
        let relationships_map = self.db.get_relationships_from_batch(result_ids)?;

        // Collect IDs of targets that are contradicted AND in our result set
        let mut target_ids_to_fetch: Vec<String> = Vec::new();
        let mut contradiction_pairs: Vec<(String, String)> = Vec::new();

        for (source_id, rels) in &relationships_map {
            for rel in rels {
                if rel.relation_type == RelationType::Contradicts
                    && result_id_set.contains(&rel.target_id)
                {
                    target_ids_to_fetch.push(rel.target_id.clone());
                    contradiction_pairs.push((source_id.clone(), rel.target_id.clone()));
                }
            }
        }

        if contradiction_pairs.is_empty() {
            return Ok(Vec::new());
        }

        // Batch fetch target memories for content preview (1 query instead of M)
        let targets_map = self.db.get_memories_batch(&target_ids_to_fetch)?;

        // Build warnings
        let mut warnings: Vec<ContradictionWarning> = Vec::new();
        for (source_id, target_id) in contradiction_pairs {
            if let Some(target) = targets_map.get(&target_id) {
                warnings.push(ContradictionWarning {
                    memory_id: source_id,
                    contradicts_id: target_id,
                    content_preview: target.content.chars().take(100).collect(),
                });
            }
        }

        Ok(warnings)
    }

    fn memory_update(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryUpdateInput = serde_json::from_value(arguments)?;

        let mut memory = self
            .db
            .get_memory(&input.id)?
            .ok_or_else(|| MemoryError::NotFound(input.id.clone()))?;

        let now = chrono::Utc::now().timestamp();
        memory.updated_at = now;

        // Handoff update invalidates and rebuilds section embeddings; sidecar must stay in sync
        // with content. Validate and rebuild BEFORE any DB write so a parse failure is a clean
        // abort — the memory row and sidecar are left untouched on error.
        //
        // The tuple carries: (new_sections, full_content_embedding, section_vecs).
        // All three are needed for the atomic update so they are computed together here.
        let handoff_sidecar_update: Option<(HandoffSections, Vec<f32>, Vec<Vec<f32>>)> =
            if memory.memory_type == MemoryType::Handoff {
                if let Some(ref new_content) = input.content {
                    // (a) Re-parse to validate; reject malformed content before touching the DB.
                    let new_sections = HandoffSections::parse_markdown(new_content)?;

                    // (b) Regenerate full-content embedding.
                    let full_embedding = self
                        .embedding
                        .embed_memory(MemoryType::Handoff, new_content)?;

                    // (c) Regenerate per-section embeddings via prefix-free embed.
                    let section_texts = handoff_section_key_texts(&new_sections);
                    let mut section_vecs: Vec<Vec<f32>> = Vec::new();
                    for (_, text) in &section_texts {
                        section_vecs.push(self.embedding.embed(text)?);
                    }

                    Some((new_sections, full_embedding, section_vecs))
                } else {
                    None
                }
            } else {
                None
            };

        if let Some(ref content) = input.content {
            memory.content = content.clone();
            // For non-Handoff types, store the embedding now (Handoff uses the atomic path below).
            if memory.memory_type != MemoryType::Handoff {
                let embedding = self.embedding.embed_memory(memory.memory_type, content)?;
                self.db
                    .store_embedding(&memory.id, &embedding, self.embedding.model_version())?;
            }
            // Regenerate summary if content changed and no explicit summary provided
            if input.summary.is_none() && should_auto_summarize(content, memory.summary.as_deref())
            {
                memory.summary = Some(generate_summary(content));
            }
        }

        if let Some(importance) = input.importance {
            memory.importance = importance.clamp(0.0, 1.0);
        }

        if let Some(tags) = input.tags {
            memory.tags = tags;
        }

        if let Some(summary) = input.summary {
            memory.summary = Some(summary);
        }

        if let Some(pinned) = input.pinned {
            memory.pinned = pinned;
        }

        // external_artifacts update semantics:
        //   - input.external_artifacts is None  -> preserve existing (omit = keep)
        //   - input.external_artifacts is Some([]) -> clear (empty array = delete)
        //   - input.external_artifacts is Some([a, b, ...]) -> replace with new list
        if let Some(artifacts) = input.external_artifacts {
            if artifacts.is_empty() {
                memory.external_artifacts = None;
            } else {
                memory.external_artifacts = Some(artifacts);
            }
        }
        // If None: leave memory.external_artifacts unchanged (preserves whatever was loaded from DB).

        // (d) For Handoff memories with new content: write memory row + full-content embedding +
        // sidecar in one transaction so a partial failure cannot leave them out of sync.
        // For all other cases fall back to the regular single-table update.
        if let Some((new_sections, full_embedding, section_vecs)) = handoff_sidecar_update {
            let section_texts = handoff_section_key_texts(&new_sections);
            let keys: Vec<&str> = section_texts.iter().map(|(k, _)| *k).collect();
            let (section_keys_str, section_bytes) = encode_section_embeddings(&keys, &section_vecs);
            self.db.update_memory_and_handoff_sidecar(
                &memory,
                &full_embedding,
                self.embedding.model_version(),
                &new_sections,
                &section_keys_str,
                &section_bytes,
            )?;
        } else {
            self.db.update_memory(&memory)?;
        }

        // Invalidate search cache since we updated data
        self.invalidate_search_cache();

        Ok(json!({"success": true, "message": "Memory updated successfully"}))
    }

    fn memory_delete(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDeleteInput = serde_json::from_value(arguments)?;

        // Remove from cluster before deleting
        if let Ok(Some(cluster_id)) = self.db.remove_from_cluster(&input.id) {
            // Recalculate centroid if cluster still has members
            let member_ids = self.db.get_cluster_member_ids(&cluster_id)?;
            if member_ids.is_empty() {
                let _ = self.db.delete_empty_clusters(&self.project_id);
            } else {
                let new_centroid = self.compute_cluster_centroid(&member_ids)?;
                let summary = self.generate_cluster_summary(&member_ids)?;
                if let Some(centroid) = new_centroid {
                    let _ = self
                        .db
                        .update_cluster_centroid(&cluster_id, &centroid, &summary);
                }
            }
        }

        let deleted = self.db.delete_memory(&input.id)?;

        if deleted {
            // Invalidate search cache since we deleted data
            self.invalidate_search_cache();
            Ok(json!({"success": true, "message": "Memory deleted successfully"}))
        } else {
            Ok(json!({"success": false, "message": "Memory not found"}))
        }
    }

    fn memory_link(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryLinkInput = serde_json::from_value(arguments)?;

        let relation_type: RelationType = input
            .relation
            .parse()
            .map_err(|_| MemoryError::InvalidRelation(input.relation.clone()))?;

        // Verify both memories exist
        self.db
            .get_memory(&input.source_id)?
            .ok_or_else(|| MemoryError::NotFound(input.source_id.clone()))?;
        self.db
            .get_memory(&input.target_id)?
            .ok_or_else(|| MemoryError::NotFound(input.target_id.clone()))?;

        let rel = Relationship {
            id: format!("rel_{}", uuid::Uuid::new_v4().simple()),
            source_id: input.source_id,
            target_id: input.target_id,
            relation_type,
            strength: input.strength.clamp(0.0, 1.0),
            created_at: chrono::Utc::now().timestamp(),
        };

        self.db.create_relationship(&rel)?;

        Ok(json!({"success": true, "id": rel.id, "message": "Relationship created successfully"}))
    }

    fn memory_graph(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryGraphInput = serde_json::from_value(arguments)?;

        let root = self
            .db
            .get_memory(&input.id)?
            .ok_or_else(|| MemoryError::NotFound(input.id.clone()))?;

        let relation_filters: Vec<RelationType> = input
            .relation_types
            .iter()
            .filter_map(|r| r.parse().ok())
            .collect();

        // BFS traversal with batch operations
        // O(depth * 3) queries instead of O(nodes * 3)
        let related = self.traverse_graph_bfs(&input.id, input.depth, &relation_filters)?;

        // Record access to root memory
        self.db.record_access(&input.id)?;

        Ok(json!(MemoryGraphResult { root, related }))
    }

    /// BFS-based graph traversal using batch operations for efficiency.
    /// Processes nodes level by level, batching relationship and memory fetches.
    fn traverse_graph_bfs(
        &self,
        start_id: &str,
        max_depth: usize,
        relation_filters: &[RelationType],
    ) -> Result<Vec<RelatedMemory>, MemoryError> {
        let mut results: Vec<RelatedMemory> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start_id.to_string());

        // Queue holds (memory_id, depth)
        let mut current_level: Vec<String> = vec![start_id.to_string()];

        for current_depth in 1..=max_depth {
            if current_level.is_empty() {
                break;
            }

            // Batch fetch outgoing relationships for entire level
            let outgoing_map = self.db.get_relationships_from_batch(&current_level)?;
            // Batch fetch incoming relationships for entire level
            let incoming_map = self.db.get_relationships_to_batch(&current_level)?;

            // Collect all new neighbor IDs and their relationship info
            // (neighbor_id, relation_type_str, direction, from_id)
            let mut neighbors_info: Vec<(String, String, String)> = Vec::new();
            let mut neighbor_ids: Vec<String> = Vec::new();

            // Process outgoing
            for rels in outgoing_map.values() {
                for rel in rels {
                    if visited.contains(&rel.target_id) {
                        continue;
                    }
                    if !relation_filters.is_empty()
                        && !relation_filters.contains(&rel.relation_type)
                    {
                        continue;
                    }
                    visited.insert(rel.target_id.clone());
                    neighbor_ids.push(rel.target_id.clone());
                    neighbors_info.push((
                        rel.target_id.clone(),
                        rel.relation_type.as_str().to_string(),
                        "outgoing".to_string(),
                    ));
                }
            }

            // Process incoming
            for rels in incoming_map.values() {
                for rel in rels {
                    if visited.contains(&rel.source_id) {
                        continue;
                    }
                    if !relation_filters.is_empty()
                        && !relation_filters.contains(&rel.relation_type)
                    {
                        continue;
                    }
                    visited.insert(rel.source_id.clone());
                    neighbor_ids.push(rel.source_id.clone());
                    neighbors_info.push((
                        rel.source_id.clone(),
                        rel.relation_type.as_str().to_string(),
                        "incoming".to_string(),
                    ));
                }
            }

            if neighbor_ids.is_empty() {
                break;
            }

            // Batch fetch all neighbor memories
            let memories_map = self.db.get_memories_batch(&neighbor_ids)?;

            // Build results for this level
            for (neighbor_id, relation, direction) in neighbors_info {
                if let Some(memory) = memories_map.get(&neighbor_id) {
                    results.push(RelatedMemory {
                        memory: memory.clone(),
                        relation,
                        direction,
                        depth: current_depth,
                    });
                }
            }

            // Next level: all neighbors found at this level
            current_level = neighbor_ids;
        }

        Ok(results)
    }

    fn memory_store_batch(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryStoreBatchInput = serde_json::from_value(arguments)?;

        if input.memories.len() > 100 {
            return Ok(json!({"success": false, "error": "Maximum 100 memories per batch"}));
        }

        let now = chrono::Utc::now().timestamp();
        let mut memories: Vec<Memory> = Vec::new();
        let mut embeddings: Vec<(String, Vec<f32>, String)> = Vec::new();
        let mut ids: Vec<String> = Vec::new();

        // Prepare all memories and embeddings
        let mut contents: Vec<String> = Vec::new();
        for mem_input in &input.memories {
            let memory_type: MemoryType = mem_input
                .memory_type
                .parse()
                .map_err(|_| MemoryError::InvalidType(mem_input.memory_type.clone()))?;
            contents.push(format!("{}: {}", memory_type.as_str(), &mem_input.content));
        }

        // Batch embed all content
        let all_embeddings = self.embedding.embed_batch(contents)?;

        for (i, mem_input) in input.memories.into_iter().enumerate() {
            let memory_type: MemoryType = mem_input
                .memory_type
                .parse()
                .map_err(|_| MemoryError::InvalidType(mem_input.memory_type.clone()))?;

            let id = format!("mem_{}", uuid::Uuid::new_v4().simple());

            // Auto-generate summary if needed
            let summary = if should_auto_summarize(&mem_input.content, mem_input.summary.as_deref())
            {
                Some(generate_summary(&mem_input.content))
            } else {
                mem_input.summary
            };

            // Resolve branch: null/omitted = global (None), "auto" = current branch, else explicit
            // If global=true, force branch to None regardless of what was passed
            let branch = if mem_input.global {
                None
            } else {
                match mem_input.branch.as_deref() {
                    None | Some("") => None, // Global
                    Some("auto") => self.current_branch.clone(),
                    Some(explicit) => Some(explicit.to_string()),
                }
            };

            let memory = Memory {
                id: id.clone(),
                project_id: self.project_id.clone(),
                memory_type,
                content: mem_input.content,
                summary,
                tags: mem_input.tags,
                importance: mem_input.importance.clamp(0.0, 1.0),
                relevance_score: 1.0,
                access_count: 0,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                branch,
                merged_from: None,
                external_artifacts: mem_input.external_artifacts,
                pinned: mem_input.pinned,
                global: mem_input.global,
            };

            embeddings.push((
                id.clone(),
                all_embeddings[i].clone(),
                self.embedding.model_version().to_string(),
            ));
            ids.push(id);
            memories.push(memory);
        }

        // Store memories in batch
        let stored = self.db.store_memories_batch(&memories)?;
        self.db.store_embeddings_batch(&embeddings)?;

        // Assign each new memory to a cluster
        for (i, mem) in memories.iter().enumerate() {
            let _ =
                self.assign_to_cluster(&mem.id, &all_embeddings[i], &mem.content, mem.importance);
        }

        // Invalidate search cache since we added new data
        if stored > 0 {
            self.invalidate_search_cache();
        }

        Ok(json!({
            "success": true,
            "count": stored,
            "ids": ids,
            "message": format!("{} memories stored successfully", stored)
        }))
    }

    fn memory_delete_batch(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDeleteBatchInput = serde_json::from_value(arguments)?;

        // Remove from clusters before deleting
        let mut affected_clusters: HashSet<String> = HashSet::new();
        for id in &input.ids {
            if let Ok(Some(cluster_id)) = self.db.remove_from_cluster(id) {
                affected_clusters.insert(cluster_id);
            }
        }

        let deleted = self.db.delete_memories_batch(&input.ids)?;

        // Cleanup affected clusters
        for cluster_id in &affected_clusters {
            let member_ids = self.db.get_cluster_member_ids(cluster_id)?;
            if member_ids.is_empty() {
                let _ = self.db.delete_empty_clusters(&self.project_id);
            } else {
                let new_centroid = self.compute_cluster_centroid(&member_ids)?;
                let summary = self.generate_cluster_summary(&member_ids)?;
                if let Some(centroid) = new_centroid {
                    let _ = self
                        .db
                        .update_cluster_centroid(cluster_id, &centroid, &summary);
                }
            }
        }

        if deleted > 0 {
            // Invalidate search cache since we deleted data
            self.invalidate_search_cache();
        }

        Ok(json!({
            "success": true,
            "deleted": deleted,
            "message": format!("{} memories deleted", deleted)
        }))
    }

    fn memory_export(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryExportInput = serde_json::from_value(arguments)?;

        let memories = self.db.get_all_memories_for_project(&self.project_id)?;
        let relationships = self
            .db
            .get_all_relationships_for_project(&self.project_id)?;

        let embeddings = if input.include_embeddings {
            Some(self.db.get_all_embeddings_for_project(&self.project_id)?)
        } else {
            None
        };

        // Collect handoff sidecar data for all Handoff-type memories.
        let mut handoff_sidecars: std::collections::HashMap<String, HandoffSidecar> =
            std::collections::HashMap::new();
        for memory in &memories {
            if memory.memory_type == MemoryType::Handoff
                && let Some((sections, section_vecs)) = self.db.get_handoff_sections(&memory.id)?
            {
                // Re-encode the sections embeddings to raw bytes for export.
                // Collect key strings first so we can borrow them as &str slices.
                let key_strings: Vec<String> =
                    section_vecs.iter().map(|(k, _)| k.clone()).collect();
                let keys: Vec<&str> = key_strings.iter().map(|s| s.as_str()).collect();
                let vecs: Vec<Vec<f32>> = section_vecs.into_iter().map(|(_, v)| v).collect();
                let (keys_str, bytes) = encode_section_embeddings(&keys, &vecs);
                handoff_sidecars.insert(
                    memory.id.clone(),
                    HandoffSidecar {
                        sections,
                        keys: keys_str,
                        bytes,
                    },
                );
            }
        }

        let export_data = export::create_export(
            &self.project_id,
            memories,
            relationships,
            embeddings,
            handoff_sidecars,
            Some(self.embedding.model_version().to_string()),
        );

        Ok(json!(export_data))
    }

    fn memory_import(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryImportInput = serde_json::from_value(arguments)?;

        let export_data: ExportData = serde_json::from_value(input.data)?;

        // Validate version
        export::validate_import(&export_data).map_err(MemoryError::Embedding)?;

        // Warn about embedding model version mismatch
        let model_warning: Option<String> =
            export_data.model_version.as_ref().and_then(|imported_model| {
                if imported_model != self.embedding.model_version() {
                    Some(format!(
                        "Warning: embeddings were generated with '{}' but current model is '{}'. Re-embedding recommended.",
                        imported_model,
                        self.embedding.model_version()
                    ))
                } else {
                    None
                }
            });

        let mode: ImportMode = input.mode.parse().unwrap_or(ImportMode::Merge);

        let mut stats = ImportStats::default();

        // In replace mode, clear existing data first
        if mode == ImportMode::Replace {
            self.db.delete_project_data(&self.project_id)?;
        }

        let now = chrono::Utc::now().timestamp();

        // Import memories
        for exported in export_data.memories {
            let ExportedMemory {
                mut memory,
                embedding: encoded_embedding,
                sections,
                section_embedding_keys,
                section_embeddings: encoded_section_embeddings,
            } = exported;

            // Update project_id to current project
            memory.project_id = self.project_id.clone();
            memory.updated_at = now;

            // Check if memory already exists (in merge mode)
            if mode == ImportMode::Merge && self.db.get_memory(&memory.id)?.is_some() {
                stats.memories_skipped += 1;
                continue;
            }

            self.db.store_memory(&memory)?;
            stats.memories_imported += 1;

            // Import embedding if provided
            if let Some(encoded) = encoded_embedding {
                if let Ok(vector) = export::decode_embedding(&encoded) {
                    self.db
                        .store_embedding(&memory.id, &vector, self.embedding.model_version())?;
                    stats.embeddings_imported += 1;
                }
            } else {
                // Generate new embedding
                let embedding = self
                    .embedding
                    .embed_memory(memory.memory_type, &memory.content)?;
                self.db
                    .store_embedding(&memory.id, &embedding, self.embedding.model_version())?;
                stats.embeddings_imported += 1;
            }

            // Import handoff sidecar if present (Handoff memories only).
            // Old exports that lack sidecar fields are still imported as memory rows;
            // the sidecar is simply skipped (a subsequent handoff_resume will notice
            // no sections are available).
            if memory.memory_type == MemoryType::Handoff {
                match (sections, section_embedding_keys, encoded_section_embeddings) {
                    (Some(sections_data), Some(keys), Some(encoded_bytes)) => {
                        match export::decode_section_embedding_bytes(&encoded_bytes) {
                            Ok(bytes) => {
                                // Validate byte length before inserting.
                                let key_count = if keys.is_empty() {
                                    0
                                } else {
                                    keys.split(',').count()
                                };
                                if bytes.len() == key_count * 256 * 4 {
                                    if let Err(e) = self.db.insert_handoff_sections(
                                        &memory.id,
                                        &sections_data,
                                        &keys,
                                        &bytes,
                                    ) {
                                        // Log but don't fail the import.
                                        tracing::warn!(
                                            "failed to import handoff sidecar for {}: {}",
                                            memory.id,
                                            e
                                        );
                                    }
                                } else {
                                    tracing::warn!(
                                        "skipping handoff sidecar for {} — section_embeddings byte length mismatch ({} bytes, expected {})",
                                        memory.id,
                                        bytes.len(),
                                        key_count * 256 * 4
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "skipping handoff sidecar for {} — could not decode section_embeddings: {}",
                                    memory.id,
                                    e
                                );
                            }
                        }
                    }
                    _ => {
                        // Old export without sidecar fields — skip sidecar, import memory row only.
                        tracing::info!(
                            "handoff {} imported without sidecar (old export format; sections not available)",
                            memory.id
                        );
                    }
                }
            }
        }

        // Import relationships
        for rel in export_data.relationships {
            // Verify both memories exist
            let source_exists = self.db.get_memory(&rel.source_id)?.is_some();
            let target_exists = self.db.get_memory(&rel.target_id)?.is_some();

            if source_exists && target_exists {
                self.db.create_relationship(&rel)?;
                stats.relationships_imported += 1;
            } else {
                stats.relationships_skipped += 1;
            }
        }

        // Invalidate search cache since we imported data
        if stats.memories_imported > 0 {
            self.invalidate_search_cache();
        }

        Ok(json!({
            "success": true,
            "stats": stats,
            "message": format!(
                "Imported {} memories, {} relationships ({} skipped)",
                stats.memories_imported,
                stats.relationships_imported,
                stats.memories_skipped + stats.relationships_skipped
            ),
            "model_warning": model_warning,
        }))
    }

    fn memory_stats(&self, _arguments: Value) -> Result<Value, MemoryError> {
        let stats: ProjectStats = self.db.get_project_stats(&self.project_id)?;
        let clusters = self.db.get_clusters_for_project(&self.project_id)?;

        Ok(json!({
            "project_id": self.project_id,
            "memory_count": stats.memory_count,
            "relationship_count": stats.relationship_count,
            "avg_relevance": stats.avg_relevance,
            "cluster_count": clusters.len(),
            "pinned_count": stats.pinned_count,
            "global_count": stats.global_count,
            "handoff_count": stats.handoff_count,
            "latest_handoff_at": stats.latest_handoff_at,
        }))
    }

    fn memory_context(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryContextInput = serde_json::from_value(arguments)?;

        // Parse type filters
        let type_filters: Vec<MemoryType> =
            input.types.iter().filter_map(|t| t.parse().ok()).collect();

        // Pre-filter candidate cap (configurable via ENGRAM_MAX_CANDIDATES, default 500)
        let max_candidates: usize = std::env::var("ENGRAM_MAX_CANDIDATES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(500);

        // Generate embedding for the context
        let context_embedding = if let Some(cached) = self.query_cache.get(&input.context) {
            cached
        } else {
            let embedding = self.embedding.embed(&input.context)?;
            self.query_cache
                .insert(input.context.clone(), embedding.clone());
            embedding
        };

        // Check if hierarchical retrieval is viable (avoid DB queries when not requested)
        let should_use_hierarchical = if input.hierarchical {
            let clusters_result = self.db.get_clusters_for_project(&self.project_id)?;
            if !clusters_result.is_empty() {
                let stats = self.db.get_project_stats(&self.project_id)?;
                if stats.memory_count >= 10 {
                    Some(clusters_result)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(clusters) = should_use_hierarchical {
            // Hierarchical: query cluster centroids, then fetch members from top clusters.
            //
            // SearchMode asymmetry for min_score:
            //   Vector  — gate on cosine similarity (existing behavior).
            //   Bm25    — gate on memory.relevance_score (decay value), same as memory_query.
            //   Hybrid  — gate on memory.relevance_score (decay value).
            // This matches the memory_query precedent from Phase 2.
            let mut cluster_scores: Vec<(String, f32)> = Vec::new();
            for cluster in &clusters {
                if let Some(ref centroid) = cluster.centroid {
                    let similarity = cosine_similarity(&context_embedding, centroid);
                    if similarity >= input.min_score as f32 {
                        cluster_scores.push((cluster.id.clone(), similarity));
                    }
                }
            }
            cluster_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Get members from top clusters
            let mut memories: Vec<Value> = Vec::new();
            let mut memory_ids: Vec<String> = Vec::new();

            let all_embeddings = self
                .db
                .get_prefiltered_embeddings(&self.project_id, max_candidates)?;
            let embedding_map: std::collections::HashMap<String, Vec<f32>> =
                all_embeddings.into_iter().collect();

            let num_top_clusters = cluster_scores.len().min(input.limit).max(1);
            let per_cluster_cap = (input.limit / num_top_clusters).max(1);

            // One FTS query per selected cluster (N clusters, default max 5).
            // Each call is a single SQLite FTS5 MATCH with an IN-clause restriction.
            for (cluster_id, _cluster_sim) in cluster_scores.iter().take(num_top_clusters) {
                let member_ids = self.db.get_cluster_member_ids(cluster_id)?;
                let member_set: std::collections::HashSet<&String> = member_ids.iter().collect();

                // Compute raw cosine similarity for all cluster members in embedding map.
                let member_raw: Vec<(String, f32)> = embedding_map
                    .iter()
                    .filter(|(id, _)| member_set.contains(id))
                    .map(|(id, vec)| (id.clone(), cosine_similarity(&context_embedding, vec)))
                    .collect();

                // Build sorted vector ranking for the vector ranker.
                let mut v_sorted = member_raw.clone();
                v_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let v_ranks: Vec<String> = v_sorted.iter().map(|(id, _)| id.clone()).collect();

                // Fetch BM25 scores for cluster members when needed.
                let bm25_for_cluster: Vec<(String, f32)> = if matches!(
                    self.search_mode,
                    SearchMode::Bm25 | SearchMode::Hybrid
                ) {
                    let bm25_res = self.db.keyword_search_within_ids(
                        &self.project_id,
                        &input.context,
                        &member_ids,
                        member_ids.len().max(1),
                    )?;
                    if bm25_res.is_empty()
                        && matches!(self.search_mode, SearchMode::Bm25 | SearchMode::Hybrid)
                    {
                        // FTS returned nothing (short or stop-word query) — fall back to
                        // vector scoring within this cluster.
                        tracing::debug!(
                            cluster_id = %cluster_id,
                            "keyword_search_within_ids returned empty; falling back to vector for this cluster"
                        );
                    }
                    bm25_res
                } else {
                    Vec::new()
                };
                let b_ranks: Vec<String> =
                    bm25_for_cluster.iter().map(|(id, _)| id.clone()).collect();

                // Narrow the per-cluster candidate set by mode.
                // In BM25 mode: restrict to BM25 result IDs only (mirrors the flat-path logic).
                let bm25_id_set: std::collections::HashSet<&str> =
                    b_ranks.iter().map(|s| s.as_str()).collect();
                let filtered_member_raw: Vec<(String, f32)> = match self.search_mode {
                    SearchMode::Vector => member_raw.clone(),
                    SearchMode::Bm25 => member_raw
                        .iter()
                        .filter(|(id, _)| bm25_id_set.contains(id.as_str()))
                        .cloned()
                        .collect(),
                    SearchMode::Hybrid => member_raw.clone(),
                };

                // Batch fetch members to get metadata.
                let member_ids_for_batch: Vec<String> = filtered_member_raw
                    .iter()
                    .map(|(id, _)| id.clone())
                    .collect();
                let members_map = self.db.get_memories_batch(&member_ids_for_batch)?;
                let sim_map: std::collections::HashMap<String, f32> =
                    member_raw.into_iter().collect();

                // Determine per-member final scores according to search mode.
                let mut member_scores: Vec<(String, f32, f32)> = {
                    // Build candidate id set from members that passed narrowing.
                    let all_ids: Vec<&String> = members_map.keys().collect();

                    // Pre-compute RRF fused map for Hybrid mode.
                    let rrf_map: std::collections::HashMap<String, f64> =
                        if matches!(self.search_mode, SearchMode::Hybrid) {
                            let b_empty = b_ranks.is_empty();
                            if b_empty {
                                // BM25 returned nothing; fuse with vector only.
                                rrf_fuse(&[v_ranks.as_slice()], 60.0)
                            } else {
                                rrf_fuse(&[v_ranks.as_slice(), b_ranks.as_slice()], 60.0)
                            }
                            .into_iter()
                            .collect()
                        } else {
                            std::collections::HashMap::new()
                        };

                    all_ids
                        .into_iter()
                        .filter_map(|id| {
                            members_map.get(id).map(|m| {
                                let similarity = *sim_map.get(id).unwrap_or(&0.0);
                                let base = match self.search_mode {
                                    SearchMode::Vector => similarity,
                                    SearchMode::Bm25 => {
                                        if b_ranks.is_empty() {
                                            // FTS fallback: use vector score.
                                            similarity
                                        } else {
                                            let rank = b_ranks
                                                .iter()
                                                .position(|bid| bid == id)
                                                .unwrap_or(usize::MAX);
                                            if rank == usize::MAX {
                                                0.0_f32
                                            } else {
                                                (1.0 / (60.0 + rank as f64 + 1.0)) as f32
                                            }
                                        }
                                    }
                                    SearchMode::Hybrid => {
                                        *rrf_map.get(id.as_str()).unwrap_or(&0.0) as f32
                                    }
                                };
                                // Vector: additive form (recency/importance contribute even at low sim).
                                // Bm25/Hybrid: multiplicative form so base=0 memories score exactly 0.
                                let final_score = match self.search_mode {
                                    SearchMode::Vector => {
                                        compute_hybrid_score(base, m.last_accessed_at, m.importance)
                                    }
                                    SearchMode::Bm25 | SearchMode::Hybrid => compute_context_score(
                                        base,
                                        m.last_accessed_at,
                                        m.importance,
                                    ),
                                };
                                (id.clone(), similarity, final_score)
                            })
                        })
                        .collect()
                };
                member_scores
                    .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

                let mut cluster_count = 0usize;
                for (id, similarity, _hybrid) in member_scores {
                    if memories.len() >= input.limit {
                        break;
                    }
                    if cluster_count > per_cluster_cap {
                        break; // Allow one extra per cluster for flexibility
                    }
                    if let Some(memory) = members_map.get(&id) {
                        // Apply branch filter (default: current branch mode)
                        let branch_filter = self.branch_mode_to_filter("current");
                        match branch_filter {
                            None => {}
                            Some(None) if memory.branch.is_some() => continue,
                            Some(None) => {}
                            Some(Some(branch)) => {
                                if let Some(ref mem_branch) = memory.branch
                                    && mem_branch != branch
                                {
                                    continue;
                                }
                            }
                        }

                        if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
                            continue;
                        }

                        // min_score gate:
                        //   Vector  — gate on cosine similarity (same as before).
                        //   Bm25    — gate on decay relevance_score.
                        //   Hybrid  — gate on decay relevance_score.
                        let passes_min_score = match self.search_mode {
                            SearchMode::Vector => similarity >= input.min_score as f32,
                            SearchMode::Bm25 | SearchMode::Hybrid => {
                                memory.relevance_score >= input.min_score
                            }
                        };
                        if !passes_min_score {
                            continue;
                        }

                        memory_ids.push(id);
                        memories.push(json!({
                            "id": memory.id,
                            "type": memory.memory_type.as_str(),
                            "content": memory.content,
                            "summary": memory.summary,
                            "tags": memory.tags,
                            "importance": memory.importance,
                            "relevance_score": memory.relevance_score,
                            "similarity": similarity,
                            "cluster_id": cluster_id,
                        }));
                        cluster_count += 1;
                    }
                }
            }

            // Record access
            if !memory_ids.is_empty() {
                let _ = self.db.record_access_batch(&memory_ids);
            }

            // Build cluster stats for clusters that contributed results
            let mut clusters_hit: Vec<Value> = Vec::new();
            for (cluster_id, cluster_sim) in cluster_scores.iter().take(input.limit) {
                if let Ok(Some(cluster)) = self.db.get_cluster(cluster_id) {
                    clusters_hit.push(json!({
                        "cluster_id": cluster_id,
                        "summary": cluster.summary,
                        "similarity": cluster_sim,
                        "member_count": cluster.member_count,
                    }));
                }
            }

            Ok(json!({
                "context": input.context,
                "count": memories.len(),
                "memories": memories,
                "retrieval_mode": "hierarchical",
                "clusters_hit": clusters_hit,
            }))
        } else {
            // Flat retrieval with pre-filtering and mode-aware scoring.
            //
            // SearchMode asymmetry for min_score:
            //   Vector  — gate on cosine similarity (existing behavior).
            //   Bm25    — gate on memory.relevance_score (decay value), matching memory_query Phase 2.
            //   Hybrid  — gate on memory.relevance_score (decay value).
            let embeddings = self
                .db
                .get_prefiltered_embeddings(&self.project_id, max_candidates)?;

            // Compute raw cosine similarities for all pre-filtered candidates.
            let all_raw: Vec<(String, f32)> = embeddings
                .iter()
                .map(|(id, vec)| (id.clone(), cosine_similarity(&context_embedding, vec)))
                .collect();

            let all_candidate_ids: Vec<String> = all_raw.iter().map(|(id, _)| id.clone()).collect();

            // Fetch BM25 scores before narrowing the candidate set (needed for BM25/Hybrid modes).
            let bm25_results: Vec<(String, f32)> =
                if matches!(self.search_mode, SearchMode::Bm25 | SearchMode::Hybrid) {
                    self.db.keyword_search_within_ids(
                        &self.project_id,
                        &input.context,
                        &all_candidate_ids,
                        all_candidate_ids.len().max(1),
                    )?
                } else {
                    Vec::new()
                };
            let b_ranks: Vec<String> = bm25_results.iter().map(|(id, _)| id.clone()).collect();

            // Narrow the candidate set:
            //   Vector  — filter by min_score on cosine (existing behavior).
            //   Bm25    — restrict to BM25 result IDs only (mirrors memory_query: non-matching
            //             memories are not candidates, keeping scoring semantics consistent).
            //   Hybrid  — union of all embeddings (vector covers semantic; RRF fuses both).
            let raw_scored: Vec<(String, f32)> = match self.search_mode {
                SearchMode::Vector => all_raw
                    .into_iter()
                    .filter(|(_, score)| *score >= input.min_score as f32)
                    .collect(),
                SearchMode::Bm25 => {
                    let bm25_id_set: std::collections::HashSet<&str> =
                        b_ranks.iter().map(|s| s.as_str()).collect();
                    all_raw
                        .into_iter()
                        .filter(|(id, _)| bm25_id_set.contains(id.as_str()))
                        .collect()
                }
                SearchMode::Hybrid => all_raw,
            };

            let candidate_ids: Vec<String> = raw_scored.iter().map(|(id, _)| id.clone()).collect();

            // Build vector ranking (sorted by cosine desc) for RRF.
            let mut v_sorted = raw_scored.clone();
            v_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let v_ranks: Vec<String> = v_sorted.iter().map(|(id, _)| id.clone()).collect();

            // Pre-compute RRF fused map for Hybrid mode.
            let rrf_map: std::collections::HashMap<String, f64> =
                if matches!(self.search_mode, SearchMode::Hybrid) {
                    if b_ranks.is_empty() {
                        rrf_fuse(&[v_ranks.as_slice()], 60.0)
                    } else {
                        rrf_fuse(&[v_ranks.as_slice(), b_ranks.as_slice()], 60.0)
                    }
                    .into_iter()
                    .collect()
                } else {
                    std::collections::HashMap::new()
                };

            // Batch fetch candidate memories to get metadata.
            let candidate_map = self.db.get_memories_batch(&candidate_ids)?;

            let sim_map: std::collections::HashMap<String, f32> = raw_scored.into_iter().collect();

            // Score each candidate per mode and sort.
            // Vector: additive form (recency/importance contribute even at low similarity).
            // Bm25/Hybrid: multiplicative form so base=0 memories score exactly 0.
            let mut scored: Vec<(String, f32, f32)> = candidate_map
                .iter()
                .map(|(id, m)| {
                    let similarity = *sim_map.get(id).unwrap_or(&0.0);
                    let base = match self.search_mode {
                        SearchMode::Vector => similarity,
                        SearchMode::Bm25 => {
                            let rank = b_ranks
                                .iter()
                                .position(|bid| bid == id)
                                .unwrap_or(usize::MAX);
                            if rank == usize::MAX {
                                0.0_f32
                            } else {
                                (1.0 / (60.0 + rank as f64 + 1.0)) as f32
                            }
                        }
                        SearchMode::Hybrid => *rrf_map.get(id.as_str()).unwrap_or(&0.0) as f32,
                    };
                    let final_score = match self.search_mode {
                        SearchMode::Vector => {
                            compute_hybrid_score(base, m.last_accessed_at, m.importance)
                        }
                        SearchMode::Bm25 | SearchMode::Hybrid => {
                            compute_context_score(base, m.last_accessed_at, m.importance)
                        }
                    };
                    (id.clone(), similarity, final_score)
                })
                .collect();
            scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

            let mut memories: Vec<Value> = Vec::new();
            let mut memory_ids: Vec<String> = Vec::new();

            for (id, similarity, _hybrid) in scored.into_iter().take(input.limit * 2) {
                if let Some(memory) = candidate_map.get(&id) {
                    // Apply branch filter
                    let branch_filter = self.branch_mode_to_filter("current");
                    match branch_filter {
                        None => {}
                        Some(None) if memory.branch.is_some() => continue,
                        Some(None) => {}
                        Some(Some(branch)) => {
                            if let Some(ref mem_branch) = memory.branch
                                && mem_branch != branch
                            {
                                continue;
                            }
                        }
                    }

                    if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
                        continue;
                    }

                    // min_score gate:
                    //   Vector  — cosine similarity (guards applied above already, redundant check).
                    //   Bm25    — decay relevance_score, matching memory_query Phase 2 behavior.
                    //   Hybrid  — decay relevance_score.
                    let passes_min_score = match self.search_mode {
                        SearchMode::Vector => similarity >= input.min_score as f32,
                        SearchMode::Bm25 | SearchMode::Hybrid => {
                            memory.relevance_score >= input.min_score
                        }
                    };
                    if !passes_min_score {
                        continue;
                    }

                    if memories.len() >= input.limit {
                        break;
                    }

                    memory_ids.push(id);
                    memories.push(json!({
                        "id": memory.id,
                        "type": memory.memory_type.as_str(),
                        "content": memory.content,
                        "summary": memory.summary,
                        "tags": memory.tags,
                        "importance": memory.importance,
                        "relevance_score": memory.relevance_score,
                        "similarity": similarity,
                    }));
                }
            }

            if !memory_ids.is_empty() {
                let _ = self.db.record_access_batch(&memory_ids);
            }

            Ok(json!({
                "context": input.context,
                "count": memories.len(),
                "memories": memories,
                "retrieval_mode": "flat",
            }))
        }
    }

    fn memory_prune(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryPruneInput = serde_json::from_value(arguments)?;

        // Get all memories and filter by relevance threshold, excluding pinned memories
        let all_memories = self.db.get_all_memories_for_project(&self.project_id)?;
        let candidates: Vec<&Memory> = all_memories
            .iter()
            .filter(|m| m.relevance_score < input.threshold && !m.pinned)
            .collect();

        if candidates.is_empty() {
            return Ok(json!({
                "success": true,
                "dry_run": !input.confirm,
                "threshold": input.threshold,
                "candidates": 0,
                "deleted": 0,
                "message": format!("No memories below threshold {:.2}", input.threshold),
                "memories": []
            }));
        }

        // Build list of candidates for display
        let candidate_info: Vec<Value> = candidates
            .iter()
            .map(|m| {
                json!({
                    "id": m.id,
                    "type": m.memory_type.as_str(),
                    "relevance_score": m.relevance_score,
                    "importance": m.importance,
                    "summary": m.summary.clone().unwrap_or_else(|| {
                        m.content.chars().take(80).collect::<String>()
                    }),
                    "created_at": m.created_at,
                    "last_accessed_at": m.last_accessed_at,
                })
            })
            .collect();

        let candidate_count = candidates.len();

        if input.confirm {
            // Actually delete
            let ids: Vec<String> = candidates.iter().map(|m| m.id.clone()).collect();
            let deleted = self.db.delete_memories_batch(&ids)?;

            // Invalidate cache since we deleted data
            self.invalidate_search_cache();

            Ok(json!({
                "success": true,
                "dry_run": false,
                "threshold": input.threshold,
                "candidates": candidate_count,
                "deleted": deleted,
                "message": format!("Deleted {} memories below threshold {:.2}", deleted, input.threshold),
                "memories": candidate_info
            }))
        } else {
            // Dry run - just show what would be deleted
            Ok(json!({
                "success": true,
                "dry_run": true,
                "threshold": input.threshold,
                "candidates": candidate_count,
                "deleted": 0,
                "message": format!(
                    "Found {} memories below threshold {:.2}. Set confirm=true to delete.",
                    candidate_count, input.threshold
                ),
                "memories": candidate_info
            }))
        }
    }

    fn memory_promote(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryPromoteInput = serde_json::from_value(arguments)?;

        // Get the memory first to verify it exists and get its current state
        let memory = self
            .db
            .get_memory(&input.id)?
            .ok_or_else(|| MemoryError::NotFound(input.id.clone()))?;

        // Check if already global
        if memory.branch.is_none() {
            return Ok(json!({
                "success": true,
                "id": input.id,
                "message": "Memory is already global",
                "was_branch": null
            }));
        }

        let was_branch = memory.branch.clone();

        // Promote to global
        let promoted = self.db.promote_memory(&input.id)?;

        if promoted {
            // Invalidate search cache since we changed data
            self.invalidate_search_cache();

            Ok(json!({
                "success": true,
                "id": input.id,
                "message": format!("Memory promoted from branch '{}' to global", was_branch.as_deref().unwrap_or("?")),
                "was_branch": was_branch
            }))
        } else {
            Ok(json!({
                "success": false,
                "id": input.id,
                "message": "Failed to promote memory"
            }))
        }
    }

    fn memory_dedup(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDedupInput = serde_json::from_value(arguments)?;
        let threshold = input.threshold.clamp(0.5, 1.0);

        // Get all embeddings and memories for the project
        let all_embeddings = self.db.get_all_embeddings_for_project(&self.project_id)?;

        // Pre-fetch all memories upfront to avoid O(n) individual get_memory calls
        let all_memories_list = self.db.get_all_memories_for_project(&self.project_id)?;
        let all_memories: std::collections::HashMap<String, Memory> = all_memories_list
            .into_iter()
            .map(|m| (m.id.clone(), m))
            .collect();

        // Build duplicate groups: for each pair with similarity >= threshold and same type
        let mut processed: HashSet<String> = HashSet::new();
        let mut groups: Vec<Vec<(String, f32)>> = Vec::new(); // groups of (id, similarity_to_first)

        for i in 0..all_embeddings.len() {
            let (ref id_i, ref vec_i) = all_embeddings[i];
            if processed.contains(id_i) {
                continue;
            }

            let mem_i = match all_memories.get(id_i) {
                Some(m) => m,
                None => continue,
            };

            // Handoffs are session snapshots; never auto-merge.
            if mem_i.memory_type == MemoryType::Handoff {
                continue;
            }

            let mut group = vec![(id_i.clone(), 1.0_f32)];

            for (id_j, vec_j) in all_embeddings.iter().skip(i + 1) {
                if processed.contains(id_j) {
                    continue;
                }

                let similarity = cosine_similarity(vec_i, vec_j);
                if similarity >= threshold
                    && let Some(mem_j) = all_memories.get(id_j)
                    && mem_j.memory_type == mem_i.memory_type
                    && mem_j.memory_type != MemoryType::Handoff
                {
                    group.push((id_j.clone(), similarity));
                }
            }

            if group.len() > 1 {
                for (id, _) in &group {
                    processed.insert(id.clone());
                }
                groups.push(group);
            }
        }

        if groups.is_empty() {
            return Ok(json!({
                "success": true,
                "dry_run": !input.confirm,
                "threshold": threshold,
                "duplicate_groups": 0,
                "total_duplicates": 0,
                "merged": 0,
                "message": format!("No duplicates found at threshold {:.2}", threshold),
                "groups": []
            }));
        }

        // Build group info for display
        let mut group_info: Vec<Value> = Vec::new();
        let mut total_duplicates = 0usize;

        for group in &groups {
            let mut members: Vec<Value> = Vec::new();
            for (id, sim) in group {
                if let Some(mem) = all_memories.get(id) {
                    members.push(json!({
                        "id": id,
                        "type": mem.memory_type.as_str(),
                        "similarity": sim,
                        "content_preview": mem.content.chars().take(100).collect::<String>(),
                        "updated_at": mem.updated_at,
                    }));
                }
            }
            total_duplicates += members.len() - 1; // -1 because one is kept
            group_info.push(json!({"members": members}));
        }

        if input.confirm {
            let mut merged_count = 0usize;

            for group in &groups {
                // Keep the most recently updated memory, merge others into it
                let with_time: Vec<(String, f32, i64)> = group
                    .iter()
                    .filter_map(|(id, sim)| {
                        all_memories
                            .get(id)
                            .map(|m| (id.clone(), *sim, m.updated_at))
                    })
                    .collect();

                let mut sorted = with_time;
                sorted.sort_by_key(|(_, _, updated_at)| std::cmp::Reverse(*updated_at)); // newest first

                if sorted.len() < 2 {
                    continue;
                }

                let keeper_id = sorted[0].0.clone();
                for (old_id, _, _) in &sorted[1..] {
                    let old_preview: String = all_memories
                        .get(old_id)
                        .map(|m| m.content.chars().take(100).collect())
                        .unwrap_or_default();
                    self.db.merge_memories(&keeper_id, old_id, &old_preview)?;
                    merged_count += 1;
                }
            }

            self.invalidate_search_cache();

            Ok(json!({
                "success": true,
                "dry_run": false,
                "threshold": threshold,
                "duplicate_groups": groups.len(),
                "total_duplicates": total_duplicates,
                "merged": merged_count,
                "message": format!("Merged {} duplicate memories from {} groups", merged_count, groups.len()),
                "groups": group_info
            }))
        } else {
            Ok(json!({
                "success": true,
                "dry_run": true,
                "threshold": threshold,
                "duplicate_groups": groups.len(),
                "total_duplicates": total_duplicates,
                "merged": 0,
                "message": format!("Found {} duplicate groups ({} duplicates). Set confirm=true to merge.", groups.len(), total_duplicates),
                "groups": group_info
            }))
        }
    }

    fn handoff_create(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: HandoffCreateInput = serde_json::from_value(arguments)?;

        // Resolve branch: explicit input branch, then current branch from ToolHandler.
        let resolved_branch = input.branch.as_deref().or(self.current_branch.as_deref());

        let result = create_handoff(
            &self.db,
            &self.embedding,
            &self.project_id,
            resolved_branch,
            input.sections,
            input.importance,
            input.pinned,
            input.auto_link,
        )?;

        // Invalidate search cache since we added new data.
        self.invalidate_search_cache();

        Ok(json!(result))
    }

    fn handoff_resume(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: HandoffResumeInput = serde_json::from_value(arguments)?;

        // Resolve branch: explicit input branch, then current branch from ToolHandler.
        let resolved_branch = input.branch.as_deref().or(self.current_branch.as_deref());

        let result = resume_handoff(
            &self.db,
            &self.embedding,
            &self.project_id,
            resolved_branch,
            input.query.as_deref(),
            input.max_sections,
            input.include_off_branch,
            input.max_chars_per_section,
        )?;

        Ok(json!(result))
    }

    fn handoff_search(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: HandoffSearchInput = serde_json::from_value(arguments)?;

        let limit = input.limit.unwrap_or(10);
        let branch = input.branch.as_deref();
        let section_filter = input.section_filter.as_deref();

        let result = search_handoffs(
            &self.db,
            &self.embedding,
            &self.project_id,
            &input.query,
            branch,
            limit,
            section_filter,
        )?;

        Ok(json!(result))
    }

    /// Generate a cluster summary from member memories.
    /// Uses the first sentence of the highest-importance member + top keywords across all members.
    fn generate_cluster_summary(&self, member_ids: &[String]) -> Result<String, MemoryError> {
        if member_ids.is_empty() {
            return Ok("Empty cluster".to_string());
        }

        let members = self.db.get_memories_batch(member_ids)?;
        if members.is_empty() {
            return Ok("Empty cluster".to_string());
        }

        // Find highest-importance member
        let best_member = members
            .values()
            .max_by(|a, b| {
                a.importance
                    .partial_cmp(&b.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Get first sentence from best member
        let first_sentence = crate::summarize::extract_first_sentence(&best_member.content);

        // Collect keywords from all members
        let all_content: String = members
            .values()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let keywords = crate::summarize::extract_keywords(&all_content, 3);

        if keywords.is_empty() {
            Ok(first_sentence)
        } else {
            Ok(format!("{} [{}]", first_sentence, keywords.join(", ")))
        }
    }

    /// Assign a memory to the best matching cluster, or create a new one.
    fn assign_to_cluster(
        &self,
        memory_id: &str,
        embedding: &[f32],
        content: &str,
        _importance: f64,
    ) -> Result<Option<String>, MemoryError> {
        use crate::memory::MemoryCluster;

        let clusters = self.db.get_clusters_for_project(&self.project_id)?;

        // Find best matching cluster by centroid similarity
        const CLUSTER_THRESHOLD: f32 = 0.75;
        let mut best_match: Option<(String, f32)> = None;

        for cluster in &clusters {
            if let Some(ref centroid) = cluster.centroid {
                let similarity = cosine_similarity(embedding, centroid);
                if similarity >= CLUSTER_THRESHOLD
                    && (best_match.is_none() || similarity > best_match.as_ref().unwrap().1)
                {
                    best_match = Some((cluster.id.clone(), similarity));
                }
            }
        }

        let now = chrono::Utc::now().timestamp();

        if let Some((cluster_id, _)) = best_match {
            // Add to existing cluster
            self.db.add_to_cluster(&cluster_id, memory_id)?;

            // Update centroid (running average)
            let member_ids = self.db.get_cluster_member_ids(&cluster_id)?;
            let new_centroid = self.compute_cluster_centroid(&member_ids)?;
            let summary = self.generate_cluster_summary(&member_ids)?;

            if let Some(centroid) = new_centroid {
                self.db
                    .update_cluster_centroid(&cluster_id, &centroid, &summary)?;
            }

            Ok(Some(cluster_id))
        } else {
            // Create new cluster
            let cluster_id = format!("clust_{}", uuid::Uuid::new_v4().simple());
            let summary = crate::summarize::extract_first_sentence(content);

            let cluster = MemoryCluster {
                id: cluster_id.clone(),
                project_id: self.project_id.clone(),
                summary,
                member_count: 1,
                centroid: Some(embedding.to_vec()),
                created_at: now,
                updated_at: now,
            };

            self.db.create_cluster(&cluster)?;
            self.db.add_to_cluster(&cluster_id, memory_id)?;

            Ok(Some(cluster_id))
        }
    }

    /// Compute the centroid (average embedding) for a set of memory IDs.
    fn compute_cluster_centroid(
        &self,
        member_ids: &[String],
    ) -> Result<Option<Vec<f32>>, MemoryError> {
        if member_ids.is_empty() {
            return Ok(None);
        }

        let member_embeddings = self.db.get_embeddings_batch(member_ids)?;

        let mut sum: Option<Vec<f32>> = None;
        let mut count = 0usize;

        for (_id, vec) in &member_embeddings {
            count += 1;
            match &mut sum {
                None => sum = Some(vec.clone()),
                Some(s) => {
                    for (i, v) in vec.iter().enumerate() {
                        if i < s.len() {
                            s[i] += v;
                        }
                    }
                }
            }
        }

        Ok(sum.map(|mut s| {
            let c = count as f32;
            for v in &mut s {
                *v /= c;
            }
            s
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::handoff::{
        create_handoff, resume_handoff_with_vec, search_handoffs_with_vec,
    };
    use super::{Database, EmbeddingService, MemoryError, RelationType, SearchMode, ToolHandler};
    use crate::memory::{HandoffSections, MemoryType};
    use crate::tools::test_utils::{dummy_vec, insert_test_handoff};

    fn test_sections(summary: &str, continues_from: Option<String>) -> HandoffSections {
        HandoffSections {
            summary: summary.to_string(),
            decisions: vec!["Use SQLite".to_string()],
            todos: vec!["Write tests".to_string()],
            blockers: vec!["Awaiting review".to_string()],
            mental_model: "Layered architecture".to_string(),
            next_steps: vec!["Deploy".to_string()],
            notes: Some("Extra note".to_string()),
            continues_from,
        }
    }

    /// 3A.8 test 1: handoff_create_basic
    /// Verifies memory row, sidecar row, and section embedding round-trip using DB helpers.
    #[test]
    fn handoff_create_basic() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "test-proj";
        db.get_or_create_project(project_id, "Test").unwrap();

        let sections = test_sections("Worked on the DB layer", None);
        let section_vecs = vec![
            ("summary", dummy_vec(0.1)),
            ("decisions", dummy_vec(0.2)),
            ("todos", dummy_vec(0.3)),
            ("blockers", dummy_vec(0.4)),
            ("mental_model", dummy_vec(0.5)),
            ("next_steps", dummy_vec(0.6)),
            ("notes", dummy_vec(0.7)),
        ];
        insert_test_handoff(
            &db,
            project_id,
            "ho-basic",
            "main",
            &sections,
            &section_vecs,
        );

        // Verify memory row exists with correct type and branch.
        let memory = db.get_memory("ho-basic").unwrap().unwrap();
        assert_eq!(memory.memory_type, MemoryType::Handoff);
        assert_eq!(memory.branch.as_deref(), Some("main"));
        assert!(memory.pinned);

        // Verify sidecar row round-trips correctly.
        let (retrieved_sections, retrieved_vecs) =
            db.get_handoff_sections("ho-basic").unwrap().unwrap();
        assert_eq!(retrieved_sections.summary, "Worked on the DB layer");
        assert_eq!(retrieved_sections.decisions, vec!["Use SQLite"]);
        assert_eq!(retrieved_vecs.len(), 7);

        // Verify section embedding round-trip (spot-check first vector).
        let (first_key, first_vec) = &retrieved_vecs[0];
        assert_eq!(first_key, "summary");
        assert!((first_vec[0] - dummy_vec(0.1)[0]).abs() < 1e-5);
    }

    /// 3A.8 test 2: handoff_create_rejects_detached_head
    /// When branch=None is passed to create_handoff, it must return InvalidType.
    #[test]
    fn handoff_create_rejects_detached_head() {
        let db = Database::open_in_memory().unwrap();
        db.get_or_create_project("proj", "Test").unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");

        let sections = HandoffSections {
            summary: "test".to_string(),
            decisions: vec![],
            todos: vec![],
            blockers: vec![],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };

        // branch=None simulates a detached HEAD or non-git workspace.
        // create_handoff rejects this before making any embedding call.
        let result = create_handoff(&db, &embedding, "proj", None, sections, 0.85, true, false);

        assert!(
            matches!(result, Err(MemoryError::InvalidType(ref msg)) if msg.contains("handoff requires a branch")),
            "Expected InvalidType error for detached HEAD, got: {:?}",
            result
        );
    }

    /// 3A.8 test 3: handoff_create_chain
    /// Create handoff A, then B with continues_from=A.id.
    /// Verify sidecar field is set and NO derived_from relationship was created for the link.
    #[test]
    fn handoff_create_chain() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "chain-proj";
        db.get_or_create_project(project_id, "Chain Test").unwrap();

        let sections_a = test_sections("Session A summary", None);
        insert_test_handoff(
            &db,
            project_id,
            "ho-a",
            "feat/x",
            &sections_a,
            &[("summary", dummy_vec(0.1))],
        );

        let sections_b = test_sections("Session B summary", Some("ho-a".to_string()));
        insert_test_handoff(
            &db,
            project_id,
            "ho-b",
            "feat/x",
            &sections_b,
            &[("summary", dummy_vec(0.2))],
        );

        // Verify sidecar continues_from on B points to A.
        let (sidecar_b, _) = db.get_handoff_sections("ho-b").unwrap().unwrap();
        assert_eq!(sidecar_b.continues_from.as_deref(), Some("ho-a"));

        // Verify NO derived_from relationship was created for the chain link.
        let rels = db.get_relationships_from("ho-b").unwrap();
        let has_derived_from_to_a = rels
            .iter()
            .any(|r| r.relation_type == RelationType::DerivedFrom && r.target_id == "ho-a");
        assert!(
            !has_derived_from_to_a,
            "continues_from must not create a derived_from relationship"
        );
    }

    /// 3A.8 test 4: handoff_resume_returns_top_sections
    /// Create a handoff with 4 sections, resume with a query, assert top sections sorted by score.
    #[test]
    fn handoff_resume_returns_top_sections() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "resume-proj";
        db.get_or_create_project(project_id, "Resume Test").unwrap();

        // Seed section embeddings with different directions relative to the query vector.
        // query = dummy_vec(0.9), so "summary" (also dummy_vec(0.9)) will score ~1.0.
        let section_vecs = vec![
            ("summary", dummy_vec(0.9)), // highest similarity to query
            ("decisions", dummy_vec(0.5)),
            ("blockers", dummy_vec(0.1)),
            ("next_steps", dummy_vec(0.3)),
        ];
        let sections = HandoffSections {
            summary: "Very relevant summary".to_string(),
            decisions: vec!["A key decision".to_string()],
            todos: vec![],
            blockers: vec!["A blocker".to_string()],
            mental_model: String::new(),
            next_steps: vec!["A next step".to_string()],
            notes: None,
            continues_from: None,
        };
        insert_test_handoff(
            &db,
            project_id,
            "ho-resume",
            "main",
            &sections,
            &section_vecs,
        );

        // Call the inner function with a pre-computed query vector so no EmbeddingService
        // is required.  dummy_vec(0.9) matches the "summary" section exactly.
        let query_vec = dummy_vec(0.9);
        let result = resume_handoff_with_vec(
            &db,
            project_id,
            Some("main"),
            Some(query_vec),
            5,
            false,
            None,
        )
        .expect("resume_handoff_with_vec must succeed");

        assert!(
            !result.top_sections.is_empty(),
            "must return scored sections"
        );

        // First result should be "summary" with score ~1.0 (identical vector).
        assert_eq!(
            result.top_sections[0].section_name, "summary",
            "summary section must rank first"
        );
        assert!(
            (result.top_sections[0].score - 1.0).abs() < 1e-4,
            "summary should score ~1.0, got {}",
            result.top_sections[0].score
        );

        // Scores must be in descending order.
        for i in 0..result.top_sections.len() - 1 {
            assert!(
                result.top_sections[i].score >= result.top_sections[i + 1].score,
                "sections must be sorted by score descending"
            );
        }
    }

    /// 3A.8 test 5: handoff_resume_detached_head_message
    /// When no branch is resolvable, resume_handoff must set message and branch=None.
    #[test]
    fn handoff_resume_detached_head_message() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "detach-proj";
        db.get_or_create_project(project_id, "Detach Test").unwrap();

        // Insert a handoff on an explicit branch so we know what would be returned.
        let sections = test_sections("Some session", None);
        insert_test_handoff(
            &db,
            project_id,
            "ho-detach",
            "main",
            &sections,
            &[("summary", dummy_vec(0.5))],
        );

        // Call resume_handoff_with_vec with branch=None, simulating a detached-HEAD workspace.
        // No EmbeddingService is required because the query vec is pre-computed.
        let result = resume_handoff_with_vec(
            &db,
            project_id,
            None, // branch=None → detached HEAD
            Some(dummy_vec(0.5)),
            5,
            false,
            None,
        )
        .expect("resume_handoff_with_vec must succeed");

        assert!(
            result.branch.is_none(),
            "branch must be None for detached HEAD"
        );
        assert!(
            result.message.is_some(),
            "message must be set for detached HEAD"
        );
        assert!(
            result
                .message
                .as_deref()
                .unwrap()
                .contains("No current branch"),
            "message must explain the situation"
        );
    }

    // ============================================
    // 3B.6 unit tests
    // ============================================

    /// 3B.6 test 1: handoff_search_filters_by_section
    /// Store multiple handoffs with content in different sections.
    /// Search with section_filter=["blockers"]; assert only blocker matches are returned.
    #[test]
    fn handoff_search_filters_by_section() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "search-filter-proj";
        db.get_or_create_project(project_id, "Search Filter Test")
            .unwrap();

        // Two handoffs: one with a blockers section, one with only a todos section.
        let sections_with_blocker = HandoffSections {
            summary: "session with blocker".to_string(),
            decisions: vec![],
            todos: vec![],
            blockers: vec!["DB migration blocking deploy".to_string()],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };
        let sections_todos_only = HandoffSections {
            summary: "session with todos".to_string(),
            decisions: vec![],
            todos: vec!["Write more tests".to_string()],
            blockers: vec![],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };

        // Use distinct vectors so blockers section clearly outscores everything else.
        let blocker_vec = dummy_vec(0.9);
        insert_test_handoff(
            &db,
            project_id,
            "ho-blocker",
            "main",
            &sections_with_blocker,
            &[
                ("summary", dummy_vec(0.2)),
                ("blockers", blocker_vec.clone()),
            ],
        );
        insert_test_handoff(
            &db,
            project_id,
            "ho-todos",
            "main",
            &sections_todos_only,
            &[("summary", dummy_vec(0.3)), ("todos", dummy_vec(0.4))],
        );

        // Query with the blockers vector so blockers section scores highest.
        let filter = vec!["blockers".to_string()];
        let result =
            search_handoffs_with_vec(&db, project_id, blocker_vec, None, 10, Some(&filter))
                .expect("search must succeed");

        // Only the blockers section should appear.
        assert!(!result.matches.is_empty(), "must return at least one match");
        for m in &result.matches {
            assert_eq!(
                m.section_name, "blockers",
                "only blockers sections should be in results, got {}",
                m.section_name
            );
        }
    }

    /// 3B.6 test 2: handoff_search_cross_branch
    /// Handoffs on feat/a and feat/b; search with branch=None; assert both appear.
    #[test]
    fn handoff_search_cross_branch() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "cross-branch-proj";
        db.get_or_create_project(project_id, "Cross Branch Test")
            .unwrap();

        let sections_a = HandoffSections {
            summary: "feat/a session".to_string(),
            decisions: vec![],
            todos: vec![],
            blockers: vec![],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };
        let sections_b = HandoffSections {
            summary: "feat/b session".to_string(),
            decisions: vec![],
            todos: vec![],
            blockers: vec![],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };

        let query_vec = dummy_vec(0.5);
        insert_test_handoff(
            &db,
            project_id,
            "ho-feat-a",
            "feat/a",
            &sections_a,
            &[("summary", dummy_vec(0.5))],
        );
        insert_test_handoff(
            &db,
            project_id,
            "ho-feat-b",
            "feat/b",
            &sections_b,
            &[("summary", dummy_vec(0.5))],
        );

        // branch=None means all branches.
        let result = search_handoffs_with_vec(&db, project_id, query_vec, None, 10, None)
            .expect("search must succeed");

        let handoff_ids: Vec<&str> = result
            .matches
            .iter()
            .map(|m| m.handoff_id.as_str())
            .collect();
        assert!(
            handoff_ids.contains(&"ho-feat-a"),
            "feat/a handoff must appear"
        );
        assert!(
            handoff_ids.contains(&"ho-feat-b"),
            "feat/b handoff must appear"
        );
    }

    /// 3B.6 test 3: handoff_update_rebuilds_sections
    /// Create a handoff, call memory_update with new section content, assert sidecar is rebuilt
    /// and section_embeddings byte length matches new section count * 256 * 4.
    #[test]
    fn handoff_update_rebuilds_sections() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "update-rebuild-proj";
        db.get_or_create_project(project_id, "Update Rebuild Test")
            .unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");

        // Create a handoff with two non-empty sections (summary + decisions).
        let sections = HandoffSections {
            summary: "Original summary".to_string(),
            decisions: vec!["Original decision".to_string()],
            todos: vec![],
            blockers: vec![],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };
        let _ = create_handoff(
            &db,
            &embedding,
            project_id,
            Some("main"),
            sections,
            0.85,
            true,
            false,
        )
        .expect("create must succeed");

        // Retrieve the ID we just created.
        let handoffs = db.list_recent_handoffs(project_id, 1).unwrap();
        let handoff_id = handoffs[0].id.clone();

        // Build new content with three non-empty sections (summary + decisions + blockers).
        let new_sections = HandoffSections {
            summary: "Updated summary".to_string(),
            decisions: vec!["Updated decision".to_string()],
            todos: vec![],
            blockers: vec!["A new blocker".to_string()],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };
        let new_content = new_sections.render_markdown();

        // Build a minimal ToolHandler to call memory_update.
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project_id.to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );

        handler
            .memory_update(json!({"id": handoff_id, "content": new_content}))
            .expect("update must succeed");

        // Verify sidecar was rebuilt: 3 sections now (summary, decisions, blockers).
        let (updated_sections, _) = db
            .get_handoff_sections(&handoff_id)
            .unwrap()
            .expect("sidecar must exist");
        assert_eq!(updated_sections.summary, "Updated summary");
        assert_eq!(updated_sections.blockers, vec!["A new blocker"]);

        // Verify raw byte length: 3 sections * 256 dims * 4 bytes.
        // We check via decode: the returned vecs length should be 3.
        let (_, section_vecs) = db
            .get_handoff_sections(&handoff_id)
            .unwrap()
            .expect("sidecar must exist");
        assert_eq!(
            section_vecs.len(),
            3,
            "should have 3 section embeddings after update"
        );
        for (_, vec) in &section_vecs {
            assert_eq!(vec.len(), 256, "each section embedding must be 256-dim");
        }
    }

    /// 3B.6 test 4: handoff_update_malformed_rejects
    /// Call memory_update on a Handoff with non-parseable content.
    /// Assert MemoryError::InvalidType, original sidecar unchanged, original content unchanged.
    #[test]
    fn handoff_update_malformed_rejects() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "update-malformed-proj";
        db.get_or_create_project(project_id, "Update Malformed Test")
            .unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");

        let sections = HandoffSections {
            summary: "Original summary content".to_string(),
            decisions: vec!["Original decision".to_string()],
            todos: vec![],
            blockers: vec![],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };
        let original_content = sections.render_markdown();

        let _ = create_handoff(
            &db,
            &embedding,
            project_id,
            Some("main"),
            sections.clone(),
            0.85,
            true,
            false,
        )
        .expect("create must succeed");

        let handoffs = db.list_recent_handoffs(project_id, 1).unwrap();
        let handoff_id = handoffs[0].id.clone();

        // Capture original sidecar state.
        let (orig_sections, orig_vecs) = db
            .get_handoff_sections(&handoff_id)
            .unwrap()
            .expect("sidecar must exist");

        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project_id.to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );

        // A string that parse_markdown cannot map to a valid HandoffSections with a summary.
        let malformed = "this is not valid handoff markdown at all !!!";
        let result = handler.memory_update(json!({"id": handoff_id, "content": malformed}));

        assert!(
            matches!(result, Err(MemoryError::InvalidType(_))),
            "must return InvalidType for malformed content, got {:?}",
            result
        );

        // Original memory content must be unchanged.
        let stored = db.get_memory(&handoff_id).unwrap().unwrap();
        assert_eq!(
            stored.content, original_content,
            "content must not be modified on parse failure"
        );

        // Original sidecar must be unchanged.
        let (post_sections, post_vecs) = db
            .get_handoff_sections(&handoff_id)
            .unwrap()
            .expect("sidecar must still exist");
        assert_eq!(
            post_sections.summary, orig_sections.summary,
            "sidecar summary must be unchanged"
        );
        assert_eq!(
            post_vecs.len(),
            orig_vecs.len(),
            "sidecar section count must be unchanged"
        );
    }
}
