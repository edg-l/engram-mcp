#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
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
    AdrSections, AdrStatus, HandoffSections, Memory, MemoryType, MemoryWithScore, ProjectStats,
    RelationType, Relationship,
};
use crate::summarize::{generate_summary, should_auto_summarize};

use super::adr::create_adr;
use super::curation::{
    CurationView, MatchedVia, Resolution, SupersessionCandidate, supersession_candidates,
};
use super::handoff::{create_handoff, handoff_section_key_texts, resume_handoff, search_handoffs};
use super::schemas::{
    AdrCreateInput, AdrExportInput, AdrListInput, AdrShowInput, AdrUpdateStatusInput,
    HandoffCreateInput, HandoffResumeInput, HandoffSearchInput, MemoryContextInput,
    MemoryDedupInput, MemoryDeleteBatchInput, MemoryDeleteInput, MemoryExportInput,
    MemoryGraphInput, MemoryImportInput, MemoryLinkInput, MemoryListInput, MemoryPromoteInput,
    MemoryPruneInput, MemoryQueryInput, MemoryRestoreInput, MemoryStatsInput,
    MemoryStoreBatchInput, MemoryStoreInput, MemoryTrashInput, MemoryUpdateInput, ToolProfile,
    dedup_threshold, get_tool_definitions_for,
};
use super::scoring::{
    SearchMode, apply_tag_and_relevance, compute_context_score, compute_hybrid_score,
    compute_tag_boost, rrf_fuse,
};
use crate::adr_export::{adr_export_target_dir, export_adr_to_disk};

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
    /// Project the memory was stored under. Always reported so a caller can see
    /// which project a store actually landed in.
    pub project: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge_info: Option<MergeInfo>,
    /// Memories this store superseded, as requested via `supersedes`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub superseded: Vec<String>,
    /// Existing memories close enough to be about the same subject but not close enough
    /// to merge. Reported so the caller can supersede one deliberately; never automatic.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub possible_supersedes: Vec<SupersessionCandidate>,
}

#[derive(Debug, Serialize)]
pub struct MemoryQueryResult {
    pub memories: Vec<MemoryWithScore>,
    pub count: usize,
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

/// Deserialize tool arguments, reporting the field names that were actually
/// received on failure. Unknown fields are ignored by the input structs, so a
/// misnamed field otherwise surfaces as a bare "missing field" that looks like
/// the server dropped a field the caller did send.
fn parse_args<T: serde::de::DeserializeOwned>(
    tool: &str,
    arguments: Value,
) -> Result<T, MemoryError> {
    let received = match arguments.as_object() {
        Some(map) if !map.is_empty() => map.keys().cloned().collect::<Vec<_>>().join(", "),
        Some(_) => "(none)".to_string(),
        None => format!("(not an object: {})", type_name_of(&arguments)),
    };

    serde_json::from_value(arguments).map_err(|e| MemoryError::InvalidArguments {
        tool: tool.to_string(),
        message: e.to_string(),
        received,
    })
}

/// JSON type name, for argument payloads that are not objects at all.
fn type_name_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
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

    /// Resolve the project a tool call operates on: the call's `project` argument
    /// when present, otherwise the server's own project. An unknown project is
    /// rejected so a mistyped ID cannot silently read or write the wrong store.
    fn resolve_project(&self, requested: Option<&str>) -> Result<String, MemoryError> {
        let Some(requested) = requested.map(str::trim).filter(|p| !p.is_empty()) else {
            return Ok(self.project_id.clone());
        };
        if requested == self.project_id || self.db.project_exists(requested)? {
            return Ok(requested.to_string());
        }
        let known: Vec<String> = self.db.list_projects()?.into_iter().map(|p| p.id).collect();
        Err(MemoryError::UnknownProject {
            requested: requested.to_string(),
            known: if known.is_empty() {
                "(none)".to_string()
            } else {
                known.join(", ")
            },
        })
    }

    /// Branch filter for a project that may not be the server's own. The current
    /// git branch describes the server's checkout only, so `"current"` widens to
    /// all branches when reading another project.
    fn branch_filter_for<'a>(
        &'a self,
        project: &str,
        branch_mode: &'a str,
    ) -> Option<Option<&'a str>> {
        if branch_mode == "current" && project != self.project_id {
            return None;
        }
        self.branch_mode_to_filter(branch_mode)
    }

    /// The current git branch, but only for the server's own project — it is
    /// meaningless as a default when writing to or reading another project.
    fn current_branch_for(&self, project: &str) -> Option<&str> {
        if project == self.project_id {
            self.current_branch.as_deref()
        } else {
            None
        }
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

    /// Invalidate search result cache for a project (call after memory modifications).
    fn invalidate_search_cache(&self, project: &str) {
        self.search_cache.invalidate_project(project);
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
            "memory_projects" => self.memory_projects(),
            "memory_context" => self.memory_context(arguments),
            "memory_list" => self.memory_list(arguments),
            "memory_trash" => self.memory_trash(arguments),
            "memory_restore" => self.memory_restore(arguments),
            "memory_prune" => self.memory_prune(arguments),
            "memory_promote" => self.memory_promote(arguments),
            "memory_dedup" => self.memory_dedup(arguments),
            "handoff_create" => self.handoff_create(arguments),
            "handoff_resume" => self.handoff_resume(arguments),
            "handoff_search" => self.handoff_search(arguments),
            "adr_create" => self.adr_create(arguments),
            "adr_update_status" => self.adr_update_status(arguments),
            "adr_list" => self.adr_list(arguments),
            "adr_show" => self.adr_show(arguments),
            "adr_export" => self.adr_export(arguments),
            _ => Err(MemoryError::UnknownTool(name.to_string())),
        }
    }

    fn memory_store(&self, arguments: Value) -> Result<Value, MemoryError> {
        use super::store::{StoreOutcome, store_with_dedup_exempting};

        let input: MemoryStoreInput = parse_args("memory_store", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

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
                Some("auto") => self.current_branch_for(&project).map(str::to_string),
                Some(explicit) => Some(explicit.to_string()),
            }
        };

        let memory = Memory {
            id: id.clone(),
            project_id: project.clone(),
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

        // Generate embedding locally — needed for dedup.
        let embedding = self.embedding.embed_memory(memory_type, &input.content)?;

        // Fetch memories now so we can build a content preview if a dedup merge occurs.
        // (store_with_dedup will re-fetch internally for dedup; that's acceptable here.)
        let pre_store_memories_list = self.db.get_all_memories_for_project(&project)?;
        let pre_store_memories: std::collections::HashMap<String, Memory> = pre_store_memories_list
            .into_iter()
            .map(|m| (m.id.clone(), m))
            .collect();

        // Handoffs are session snapshots; bypass dedup.
        // Pass None for embedding_service to skip dedup for handoffs.
        let dedup_thr = dedup_threshold();
        // Memories the caller has explicitly tied to this one are distinct from it by
        // assertion, so dedup must not collapse them together.
        let dedup_exempt: HashSet<String> = input
            .related_to
            .iter()
            .chain(input.supersedes.iter())
            .cloned()
            .collect();
        let outcome = if memory_type != MemoryType::Handoff {
            store_with_dedup_exempting(
                &self.db,
                Some(&self.embedding),
                &project,
                memory,
                Some(&embedding),
                dedup_thr,
                None, // MCP path never skips — always merges duplicates
                &dedup_exempt,
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

        // Record what this memory replaces. The edge is the only record of supersession;
        // retrieval reads it to redirect the superseded memory's matches here.
        let mut superseded: Vec<String> = Vec::new();
        for old_id in &input.supersedes {
            if self.db.get_memory(old_id)?.is_none() {
                return Err(MemoryError::NotFound(old_id.clone()));
            }
            let rel = Relationship {
                id: format!("rel_{}", uuid::Uuid::new_v4().simple()),
                source_id: final_id.clone(),
                target_id: old_id.clone(),
                relation_type: RelationType::Supersedes,
                strength: 1.0,
                created_at: now,
            };
            self.db.create_relationship(&rel)?;
            superseded.push(old_id.clone());
        }

        // Surface near-duplicates that were too far apart to merge but close enough to be
        // about the same thing. The similarities are already computed for dedup and
        // otherwise thrown away; a caller storing "X is now Y" has no other way to learn
        // that a months-old memory says "X is Z", because that memory never surfaced.
        // A memory the store just merged with is reported as a merge, not as something
        // that might need superseding.
        let mut candidate_exclusions: Vec<&str> = vec![final_id.as_str()];
        if let Some(info) = &merge_info {
            candidate_exclusions.push(info.merged_with.as_str());
        }
        let possible_supersedes = if superseded.is_empty() {
            supersession_candidates(
                &self.db,
                &project,
                &embedding,
                memory_type,
                &candidate_exclusions,
            )?
        } else {
            Vec::new()
        };

        // Assign to cluster
        let _cluster_id = self.assign_to_cluster(
            &project,
            &final_id,
            &embedding,
            &input.content,
            input.importance.clamp(0.0, 1.0),
        )?;

        // Invalidate search cache since we added new data
        self.invalidate_search_cache(&project);

        let message = if merge_info.is_some() {
            "Memory stored and merged with duplicate".to_string()
        } else {
            "Memory stored successfully".to_string()
        };

        Ok(json!(MemoryStoreResult {
            id: final_id,
            message,
            project,
            branch,
            merge_info,
            superseded,
            possible_supersedes,
        }))
    }

    fn memory_query(&self, arguments: Value) -> Result<Value, MemoryError> {
        let mut input: MemoryQueryInput = parse_args("memory_query", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;
        input.limit = input.limit.min(100); // Server-side cap to prevent overflow

        // Parse type filters
        let type_filters: Vec<MemoryType> =
            input.types.iter().filter_map(|t| t.parse().ok()).collect();

        // Optimization: if query is empty, skip search and use filter-only path
        if input.query.trim().is_empty() {
            let branch_filter = self.branch_filter_for(&project, &input.branch_mode);
            let memories = self.db.query_memories_with_branch(
                &project,
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

            let candidates: HashMap<String, Memory> =
                memories.iter().map(|m| (m.id.clone(), m.clone())).collect();
            let ranked: Vec<(String, f64, f64, f64, f64)> = memories
                .iter()
                .map(|m| (m.id.clone(), m.relevance_score, 0.0, 0.0, 0.0))
                .collect();
            let curation = self.curation_view(&project, &candidates, input.include_superseded)?;

            let results: Vec<MemoryWithScore> = self
                .apply_curation(&curation, &candidates, ranked)?
                .into_iter()
                .skip(input.offset)
                .take(input.limit)
                .collect();

            return Ok(json!(MemoryQueryResult {
                count: results.len(),
                memories: results,
            }));
        }

        let branch_filter = self.branch_filter_for(&project, &input.branch_mode);

        // --- Embedding path (skipped for pure BM25 mode) ---
        let semantic_scores: std::collections::HashMap<String, f32> =
            if self.search_mode != SearchMode::Bm25 {
                let query_embedding = if let Some(cached) = self.query_cache.get(&input.query) {
                    cached
                } else {
                    let embedding = self.embedding.embed(&input.query)?;
                    self.query_cache
                        .insert(input.query.clone(), embedding.clone());
                    embedding
                };

                if let Some(cached_results) = self.search_cache.get(&project, &query_embedding) {
                    cached_results.into_iter().collect()
                } else {
                    let embeddings = self
                        .db
                        .get_all_embeddings_for_project_and_global(&project)?;

                    let scored: Vec<(String, f32)> = embeddings
                        .iter()
                        .map(|(id, vec)| {
                            let similarity = cosine_similarity(&query_embedding, vec);
                            (id.clone(), similarity)
                        })
                        .collect();

                    self.search_cache
                        .insert(&project, &query_embedding, scored.clone());
                    scored.into_iter().collect()
                }
            } else {
                std::collections::HashMap::new()
            };

        // --- BM25 path (skipped for pure Vector mode) ---
        // Returns ordered Vec<(id, raw_score)> from FTS5; we keep the ordering for RRF.
        let bm25_results: Vec<(String, f64)> = if self.search_mode != SearchMode::Vector {
            self.db.keyword_search_with_branch(
                &project,
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

        // Replace superseded matches with what superseded them, and drop dead ones.
        // Runs before pagination so a page is never short because entries were suppressed.
        let curation = self.curation_view(&project, &memories_map, input.include_superseded)?;
        let results: Vec<MemoryWithScore> = self
            .apply_curation(&curation, &memories_map, scored_results)?
            .into_iter()
            .skip(input.offset)
            .take(input.limit)
            .collect();
        let result_ids: Vec<String> = results.iter().map(|r| r.memory.id.clone()).collect();

        // Batch record access for all result memories
        if !result_ids.is_empty() {
            let _ = self.db.record_access_batch(&result_ids);
        }

        Ok(json!(MemoryQueryResult {
            count: results.len(),
            memories: results,
        }))
    }

    /// Materialize a ranked id list into results, applying curation status.
    ///
    /// Dead matches are dropped, superseded matches are replaced by their successor at
    /// the superseded memory's rank, and a successor reached both ways appears once.
    /// `ranked` is `(memory_id, score, semantic_score, keyword_score, rrf_score)` in
    /// descending score order; `candidates` supplies already-loaded memories, and
    /// anything else is read from the database.
    fn apply_curation(
        &self,
        curation: &CurationView,
        candidates: &HashMap<String, Memory>,
        ranked: Vec<(String, f64, f64, f64, f64)>,
    ) -> Result<Vec<MemoryWithScore>, MemoryError> {
        let mut results: Vec<MemoryWithScore> = Vec::new();
        // Effective id -> index in `results`, so a successor reached both on its own merit
        // and by redirect appears once, at its better rank.
        let mut emitted: HashMap<String, usize> = HashMap::new();

        for (id, score, semantic_score, keyword_score, rrf_score) in ranked {
            let (effective_id, matched_via) = match curation.resolve(&id) {
                Resolution::Keep => (id, None),
                Resolution::Drop => continue,
                Resolution::Redirect { successor_id, via } => (successor_id, Some(via)),
            };

            if let Some(&index) = emitted.get(&effective_id) {
                // Already present. Keep the annotation if this is the first redirect into
                // it, so the caller still learns that the superseded memory matched.
                if let Some(via) = matched_via
                    && results[index].matched_via.is_none()
                {
                    results[index].matched_via = Some(via);
                }
                continue;
            }

            // A redirect target need not have been a candidate itself.
            let memory = match candidates.get(&effective_id) {
                Some(m) => m.clone(),
                None => match self.db.get_memory(&effective_id)? {
                    Some(m) => m,
                    None => continue,
                },
            };

            emitted.insert(effective_id, results.len());
            let mut memory_clone = memory;
            memory_clone.access_count += 1;
            results.push(MemoryWithScore {
                memory: memory_clone,
                score,
                semantic_score,
                keyword_score,
                rrf_score,
                matched_via,
            });
        }

        Ok(results)
    }

    /// Map a `memory_context` candidate to the memory that should actually be shown.
    ///
    /// Returns `None` when the hit should be dropped: the memory is dead, its successor
    /// is dead or missing, the successor fails the caller's type filter, or the successor
    /// was already emitted for an earlier hit. `emitted` tracks the ids already shown.
    fn resolve_context_hit(
        &self,
        curation: &CurationView,
        candidate: &Memory,
        type_filters: &[MemoryType],
        emitted: &mut HashSet<String>,
    ) -> Result<Option<(Memory, Option<MatchedVia>)>, MemoryError> {
        let (memory, via) = match curation.resolve(&candidate.id) {
            Resolution::Drop => return Ok(None),
            Resolution::Keep => (candidate.clone(), None),
            Resolution::Redirect { successor_id, via } => {
                let Some(successor) = self.db.get_memory(&successor_id)? else {
                    return Ok(None);
                };
                // The filter was checked against the superseded memory; the successor is a
                // different memory and has to satisfy it on its own terms.
                if !type_filters.is_empty() && !type_filters.contains(&successor.memory_type) {
                    return Ok(None);
                }
                (successor, Some(via))
            }
        };

        if !emitted.insert(memory.id.clone()) {
            return Ok(None);
        }
        Ok(Some((memory, via)))
    }

    /// Curation status for a retrieval call.
    ///
    /// `raw` short-circuits to an empty view: superseded and dead memories then come back
    /// untouched, which is what curation needs and what ordinary retrieval must not do.
    /// Previews are seeded from the candidate set so a redirect can say what it replaced
    /// without a second round of lookups.
    fn curation_view(
        &self,
        project: &str,
        candidates: &HashMap<String, Memory>,
        raw: bool,
    ) -> Result<CurationView, MemoryError> {
        if raw {
            return Ok(CurationView::empty());
        }
        let mut view = CurationView::load(&self.db, project)?;
        for (id, memory) in candidates {
            view.note_preview(id, &memory.content);
        }
        Ok(view)
    }

    fn memory_update(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryUpdateInput = parse_args("memory_update", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let mut memory = self
            .db
            .get_memory(&input.id)?
            .ok_or_else(|| MemoryError::NotFound(input.id.clone()))?;

        // Content is replaced wholesale, not patched, so the previous version is gone the
        // moment the row is written. Snapshot it and hand it back to the caller.
        let previous = memory.clone();
        let content_replaced = input
            .content
            .as_ref()
            .is_some_and(|new| *new != memory.content);
        if content_replaced {
            self.db.trash_memory(&input.id, crate::db::OP_UPDATE)?;
        }

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

        if let Some(dead) = input.dead {
            self.db
                .set_dead(&input.id, dead, input.dead_reason.as_deref())?;
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
        self.invalidate_search_cache(&project);

        Ok(json!({
            "success": true,
            "message": "Memory updated successfully",
            "content_replaced": content_replaced,
            "recoverable": content_replaced,
            "dead": self.db.is_dead(&input.id)?,
            "previous": previous,
        }))
    }

    fn memory_delete(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDeleteInput = parse_args("memory_delete", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        // Read the memory before it goes so the result can carry what was destroyed.
        // A memory can hold several claims of different lifetimes, and only some of them
        // are recoverable from the repository; the caller cannot check what it cannot see.
        let doomed = self.db.get_memory(&input.id)?;

        // Remove from cluster before deleting
        if let Ok(Some(cluster_id)) = self.db.remove_from_cluster(&input.id) {
            // Recalculate centroid if cluster still has members
            let member_ids = self.db.get_cluster_member_ids(&cluster_id)?;
            if member_ids.is_empty() {
                let _ = self.db.delete_empty_clusters(&project);
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
            self.invalidate_search_cache(&project);
            Ok(json!({
                "success": true,
                "message": "Memory deleted successfully",
                "recoverable": true,
                "deleted": doomed,
            }))
        } else {
            Ok(json!({"success": false, "message": "Memory not found"}))
        }
    }

    fn memory_link(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryLinkInput = parse_args("memory_link", arguments)?;
        self.resolve_project(input.project.as_deref())?;

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
        let input: MemoryGraphInput = parse_args("memory_graph", arguments)?;
        self.resolve_project(input.project.as_deref())?;

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
        let input: MemoryStoreBatchInput = parse_args("memory_store_batch", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

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
            contents.push(format!("{}: {}", memory_type.as_str(), mem_input.content));
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
                    Some("auto") => self.current_branch_for(&project).map(str::to_string),
                    Some(explicit) => Some(explicit.to_string()),
                }
            };

            let memory = Memory {
                id: id.clone(),
                project_id: project.clone(),
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
            let _ = self.assign_to_cluster(
                &project,
                &mem.id,
                &all_embeddings[i],
                &mem.content,
                mem.importance,
            );
        }

        // Invalidate search cache since we added new data
        if stored > 0 {
            self.invalidate_search_cache(&project);
        }

        Ok(json!({
            "success": true,
            "count": stored,
            "project": project,
            "ids": ids,
            "message": format!("{} memories stored successfully", stored)
        }))
    }

    fn memory_delete_batch(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryDeleteBatchInput = parse_args("memory_delete_batch", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        // Same as `memory_delete`: read the memories out before destroying them so the
        // result says what was lost. Batch especially, since one id can stand for several
        // claims that dedup collapsed into a single memory.
        let doomed: Vec<Memory> = input
            .ids
            .iter()
            .filter_map(|id| self.db.get_memory(id).ok().flatten())
            .collect();

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
                let _ = self.db.delete_empty_clusters(&project);
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
            self.invalidate_search_cache(&project);
        }

        Ok(json!({
            "success": true,
            "deleted": deleted,
            "recoverable": true,
            "memories": doomed,
            "message": format!("{} memories deleted", deleted)
        }))
    }

    fn memory_export(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryExportInput = parse_args("memory_export", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let memories = self.db.get_all_memories_for_project(&project)?;
        let relationships = self.db.get_all_relationships_for_project(&project)?;

        let embeddings = if input.include_embeddings {
            Some(self.db.get_all_embeddings_for_project(&project)?)
        } else {
            None
        };

        // Collect handoff sidecar data for all Handoff-type memories.
        let mut handoff_sidecars: std::collections::HashMap<String, HandoffSidecar> =
            std::collections::HashMap::new();
        // Collect ADR sidecar data for all ADR-type memories.
        let mut adr_sidecars: export::AdrSidecarMap = std::collections::HashMap::new();
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
            if memory.memory_type == MemoryType::Adr
                && let Some((num, status, sections)) = self.db.get_adr_sections(&memory.id)?
            {
                adr_sidecars.insert(memory.id.clone(), (num, status, sections));
            }
        }

        let export_data = export::create_export(
            &project,
            memories,
            relationships,
            embeddings,
            handoff_sidecars,
            &adr_sidecars,
            Some(self.embedding.model_version().to_string()),
        );

        Ok(json!(export_data))
    }

    fn memory_import(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryImportInput = parse_args("memory_import", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

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
            self.db.delete_project_data(&project)?;
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
                adr_number,
                adr_status: adr_status_str,
                adr_sections: adr_sections_data,
            } = exported;

            let mem_created_at = memory.created_at;
            let mem_updated_at = memory.updated_at;

            // Update project_id to the target project
            memory.project_id = project.clone();
            memory.updated_at = now;

            // Check if memory already exists (in merge mode)
            if mode == ImportMode::Merge && self.db.get_memory(&memory.id)?.is_some() {
                stats.memories_skipped += 1;
                continue;
            }

            // For ADR memories with a known number, pre-check the number BEFORE storing
            // the memory row.  If the number is already taken, skip the entire memory
            // (memory row + embedding + sidecar) to keep them consistent.
            if memory.memory_type == MemoryType::Adr
                && let Some(num) = adr_number
                && self.db.get_adr_by_number(&project, num)?.is_some()
            {
                stats.memories_skipped += 1;
                tracing::warn!(
                    "skipping imported ADR {} — number {} already exists in project",
                    memory.id,
                    num
                );
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

            // Import ADR sidecar if present (ADR memories only).
            // Number-conflict check above guarantees the number is free at this point.
            if memory.memory_type == MemoryType::Adr
                && let (Some(num), Some(status_str), Some(adr_sec)) =
                    (adr_number, adr_status_str, adr_sections_data)
            {
                match status_str.parse::<AdrStatus>() {
                    Ok(status) => {
                        if let Err(e) = self.db.insert_adr_sidecar(
                            &memory.id,
                            &project,
                            num,
                            status,
                            &adr_sec,
                            mem_created_at,
                            mem_updated_at,
                        ) {
                            tracing::warn!(
                                "failed to insert ADR sidecar for {} (number {}): {}",
                                memory.id,
                                num,
                                e
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "skipping ADR sidecar for {} — invalid status '{}': {}",
                            memory.id,
                            status_str,
                            e
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
            self.invalidate_search_cache(&project);
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

    fn memory_stats(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryStatsInput = parse_args("memory_stats", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let stats: ProjectStats = self.db.get_project_stats(&project)?;
        let clusters = self.db.get_clusters_for_project(&project)?;

        Ok(json!({
            "project_id": project,
            "memory_count": stats.memory_count,
            "relationship_count": stats.relationship_count,
            "avg_relevance": stats.avg_relevance,
            "cluster_count": clusters.len(),
            "pinned_count": stats.pinned_count,
            "global_count": stats.global_count,
            "handoff_count": stats.handoff_count,
            "latest_handoff_at": stats.latest_handoff_at,
            "dead_count": self.db.count_dead(&project)?,
            "trash_count": self.db.count_trash(&project)?,
        }))
    }

    fn memory_projects(&self) -> Result<Value, MemoryError> {
        let projects = self.db.list_projects()?;

        let items: Vec<Value> = projects
            .iter()
            .map(|p| {
                json!({
                    "project_id": p.id,
                    "memory_count": p.memory_count,
                    "handoff_count": p.handoff_count,
                    "adr_count": p.adr_count,
                    "latest_activity_at": p.latest_activity_at,
                    "current": p.id == self.project_id,
                })
            })
            .collect();

        Ok(json!({
            "current_project": self.project_id,
            "count": items.len(),
            "projects": items,
        }))
    }

    fn memory_context(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryContextInput = parse_args("memory_context", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        // Parse type filters
        let type_filters: Vec<MemoryType> =
            input.types.iter().filter_map(|t| t.parse().ok()).collect();

        // Curation status for this project, shared by both retrieval branches. Context is
        // the tool that feeds an agent its background, so a superseded conclusion reaching
        // it unqualified is the worst case for stale memory; `emitted` keeps a successor
        // from appearing twice when several of its predecessors match.
        let curation = self.curation_view(&project, &HashMap::new(), false)?;
        let mut emitted: HashSet<String> = HashSet::new();

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
            let clusters_result = self.db.get_clusters_for_project(&project)?;
            if !clusters_result.is_empty() {
                let stats = self.db.get_project_stats(&project)?;
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
                .get_prefiltered_embeddings(&project, max_candidates)?;
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
                        &project,
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
                        let branch_filter = self.branch_filter_for(&project, "current");
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

                        let Some((memory, matched_via)) = self.resolve_context_hit(
                            &curation,
                            memory,
                            &type_filters,
                            &mut emitted,
                        )?
                        else {
                            continue;
                        };

                        memory_ids.push(memory.id.clone());
                        memories.push(json!({
                            "id": memory.id,
                            "type": memory.memory_type.as_str(),
                            "content": memory.content,
                            "summary": memory.summary,
                            "tags": memory.tags,
                            "importance": memory.importance,
                            "relevance_score": memory.relevance_score,
                            "similarity": similarity,
                            "matched_via": matched_via,
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
                .get_prefiltered_embeddings(&project, max_candidates)?;

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
                        &project,
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
                    let branch_filter = self.branch_filter_for(&project, "current");
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

                    let Some((memory, matched_via)) =
                        self.resolve_context_hit(&curation, memory, &type_filters, &mut emitted)?
                    else {
                        continue;
                    };

                    memory_ids.push(memory.id.clone());
                    memories.push(json!({
                        "id": memory.id,
                        "type": memory.memory_type.as_str(),
                        "content": memory.content,
                        "summary": memory.summary,
                        "tags": memory.tags,
                        "importance": memory.importance,
                        "relevance_score": memory.relevance_score,
                        "similarity": similarity,
                        "matched_via": matched_via,
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

    /// Enumerate memories without going through search.
    ///
    /// Retrieval is the wrong instrument for curation: it only shows what a query
    /// surfaces, so the memories most in need of attention — the ones nothing ever
    /// matches — are exactly the ones it hides. This lists them directly, and unlike
    /// query it can show superseded and dead memories.
    fn memory_list(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryListInput = parse_args("memory_list", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let type_filters: Vec<MemoryType> =
            input.types.iter().filter_map(|t| t.parse().ok()).collect();

        let dead = self.db.get_dead_ids(&project)?;
        let supersession = self.db.get_supersession_map(&project)?;

        let mut memories: Vec<Memory> = self
            .db
            .get_all_memories_for_project(&project)?
            .into_iter()
            .filter(|m| type_filters.is_empty() || type_filters.contains(&m.memory_type))
            .filter(|m| input.tags.is_empty() || input.tags.iter().any(|t| m.tags.contains(t)))
            .filter(|m| {
                let is_dead = dead.contains(&m.id);
                let is_superseded = supersession.is_superseded(&m.id);
                match input.status.as_str() {
                    "dead" => is_dead,
                    "superseded" => is_superseded,
                    "all" => true,
                    // "live" and anything unrecognized: only what retrieval would return.
                    _ => !is_dead && !is_superseded,
                }
            })
            .collect();

        match input.order.as_str() {
            "created" => memories.sort_by_key(|m| std::cmp::Reverse(m.created_at)),
            "updated" => memories.sort_by_key(|m| std::cmp::Reverse(m.updated_at)),
            "accessed" => memories.sort_by_key(|m| std::cmp::Reverse(m.last_accessed_at)),
            _ => memories.sort_by(|a, b| {
                b.relevance_score
                    .partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
        }

        let total = memories.len();
        let rows: Vec<Value> = memories
            .into_iter()
            .skip(input.offset)
            .take(input.limit)
            .map(|m| {
                json!({
                    "id": m.id,
                    "type": m.memory_type.as_str(),
                    "content": m.content.chars().take(input.content_length).collect::<String>(),
                    "tags": m.tags,
                    "importance": m.importance,
                    "relevance_score": m.relevance_score,
                    "access_count": m.access_count,
                    "created_at": m.created_at,
                    "updated_at": m.updated_at,
                    "pinned": m.pinned,
                    "branch": m.branch,
                    "dead": dead.contains(&m.id),
                    "superseded_by": supersession.terminal_successor(&m.id),
                })
            })
            .collect();

        Ok(json!({
            "project": project,
            "status": input.status,
            "order": input.order,
            "total": total,
            "count": rows.len(),
            "offset": input.offset,
            "memories": rows,
        }))
    }

    /// List recoverable snapshots of destroyed memories.
    fn memory_trash(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryTrashInput = parse_args("memory_trash", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let entries = self.db.list_trash(&project, input.limit)?;
        let rows: Vec<Value> = entries
            .iter()
            .map(|e| {
                json!({
                    "trash_id": e.trash_id,
                    "memory_id": e.memory.id,
                    "op": e.op,
                    "trashed_at": e.trashed_at,
                    "type": e.memory.memory_type.as_str(),
                    "preview": e.memory.content.chars().take(200).collect::<String>(),
                    "content_chars": e.memory.content.chars().count(),
                    "relationships": e.relationships.len(),
                })
            })
            .collect();

        Ok(json!({
            "project": project,
            "count": rows.len(),
            "total": self.db.count_trash(&project)?,
            "entries": rows,
        }))
    }

    /// Put a trashed memory back.
    fn memory_restore(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryRestoreInput = parse_args("memory_restore", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let entry = match (input.trash_id, input.id.as_deref()) {
            (Some(trash_id), _) => self.db.get_trash_entry(trash_id)?,
            (None, Some(id)) => self.db.latest_trash_for_memory(id)?,
            (None, None) => {
                return Err(MemoryError::InvalidArguments {
                    tool: "memory_restore".to_string(),
                    message: "needs either `id` (restores its most recent snapshot) or \
                              `trash_id` (restores one exact snapshot)"
                        .to_string(),
                    received: "neither".to_string(),
                });
            }
        };

        let entry = entry.ok_or_else(|| {
            MemoryError::NotFound(
                input
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("trash entry {:?}", input.trash_id)),
            )
        })?;

        let outcome = self.db.restore_trash_entry(entry.trash_id)?;
        self.invalidate_search_cache(&project);

        Ok(json!({
            "success": true,
            "id": outcome.memory.id,
            "trashed_by": outcome.op,
            "overwrote_existing": outcome.overwrote_existing,
            "edges_restored": outcome.edges_restored,
            "edges_dropped": outcome.edges_dropped,
            "memory": outcome.memory,
            "message": if outcome.edges_dropped > 0 {
                format!(
                    "Memory restored. {} relationship(s) could not be restored because the memory at the other end is gone.",
                    outcome.edges_dropped
                )
            } else {
                "Memory restored".to_string()
            },
        }))
    }

    fn memory_prune(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: MemoryPruneInput = parse_args("memory_prune", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        // Get all memories and filter by relevance threshold, excluding pinned memories
        let all_memories = self.db.get_all_memories_for_project(&project)?;
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
            let deleted = self
                .db
                .delete_memories_batch_with_op(&ids, crate::db::OP_PRUNE)?;

            // Invalidate cache since we deleted data
            self.invalidate_search_cache(&project);

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
        let input: MemoryPromoteInput = parse_args("memory_promote", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

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
            self.invalidate_search_cache(&project);

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
        let input: MemoryDedupInput = parse_args("memory_dedup", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;
        let threshold = input.threshold.clamp(0.5, 1.0);

        // Get all embeddings and memories for the project
        let all_embeddings = self.db.get_all_embeddings_for_project(&project)?;

        // Pre-fetch all memories upfront to avoid O(n) individual get_memory calls
        let all_memories_list = self.db.get_all_memories_for_project(&project)?;
        let all_memories: std::collections::HashMap<String, Memory> = all_memories_list
            .into_iter()
            .map(|m| (m.id.clone(), m))
            .collect();

        // Build duplicate groups: for each pair with similarity >= threshold and same type
        // Pairs the graph says are deliberately distinct. A `derived_from` edge is the
        // shape of "this lesson came out of that finding": two memories on one subject
        // with different lifetimes, which is exactly what must not be collapsed back into
        // one record. `supersedes` pairs are the old and new answer to the same question,
        // and merging them would erase the redirect.
        let deliberate_pairs: HashSet<(String, String)> = self
            .db
            .get_all_relationships_for_project(&project)?
            .into_iter()
            .filter(|r| {
                matches!(
                    r.relation_type,
                    RelationType::DerivedFrom | RelationType::Supersedes
                )
            })
            .flat_map(|r| {
                [
                    (r.source_id.clone(), r.target_id.clone()),
                    (r.target_id, r.source_id),
                ]
            })
            .collect();

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
                if deliberate_pairs.contains(&(id_i.clone(), id_j.clone())) {
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
                    self.db.merge_memories(&keeper_id, old_id)?;
                    merged_count += 1;
                }
            }

            self.invalidate_search_cache(&project);

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
        let input: HandoffCreateInput = parse_args("handoff_create", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        // Resolve branch: explicit input branch, then current branch from ToolHandler.
        // The server's branch describes its own checkout, so writing a handoff to
        // another project requires the caller to name that project's branch.
        let resolved_branch = input
            .branch
            .as_deref()
            .or_else(|| self.current_branch_for(&project));
        if resolved_branch.is_none() && project != self.project_id {
            return Err(MemoryError::InvalidType(format!(
                "handoff for project '{project}' requires an explicit branch"
            )));
        }

        let result = create_handoff(
            &self.db,
            &self.embedding,
            &project,
            resolved_branch,
            input.sections,
            input.importance,
            input.pinned,
            input.auto_link,
        )?;

        // Invalidate search cache since we added new data.
        self.invalidate_search_cache(&project);

        Ok(json!(result))
    }

    fn handoff_resume(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: HandoffResumeInput = parse_args("handoff_resume", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        // Resolve branch: explicit input branch, then current branch from ToolHandler.
        let resolved_branch = input
            .branch
            .as_deref()
            .or_else(|| self.current_branch_for(&project));

        let result = resume_handoff(
            &self.db,
            &self.embedding,
            &project,
            resolved_branch,
            input.query.as_deref(),
            input.max_sections,
            input.include_off_branch,
            input.max_chars_per_section,
        )?;

        Ok(json!(result))
    }

    fn handoff_search(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: HandoffSearchInput = parse_args("handoff_search", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let limit = input.limit.unwrap_or(10);
        let branch = input.branch.as_deref();
        let section_filter = input.section_filter.as_deref();

        let result = search_handoffs(
            &self.db,
            &self.embedding,
            &project,
            &input.query,
            branch,
            limit,
            section_filter,
        )?;

        Ok(json!(result))
    }

    fn adr_create(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: AdrCreateInput = parse_args("adr_create", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let status = input
            .status
            .parse::<AdrStatus>()
            .map_err(|e| MemoryError::InvalidType(e.to_string()))?;

        let sections = AdrSections {
            title: input.title,
            context: input.context,
            decision: input.decision,
            consequences: input.consequences,
        };

        let result = create_adr(
            &self.db,
            &self.embedding,
            &project,
            sections,
            status,
            input.importance,
            input.pinned,
            input.supersedes,
        )?;

        self.invalidate_search_cache(&project);

        Ok(json!(result))
    }

    fn adr_update_status(&self, arguments: Value) -> Result<Value, MemoryError> {
        use std::str::FromStr;

        let input: AdrUpdateStatusInput = parse_args("adr_update_status", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let old_id = self
            .db
            .get_adr_by_number(&project, input.number)?
            .ok_or_else(|| MemoryError::NotFound(format!("ADR-{:04} not found", input.number)))?;

        let target_status = AdrStatus::from_str(&input.status)
            .map_err(|e| MemoryError::InvalidType(e.to_string()))?;

        if target_status == AdrStatus::Superseded {
            return Err(MemoryError::InvalidType(
                "use adr_create with the 'supersedes' field to mark an ADR superseded".to_string(),
            ));
        }

        let (_, current_status, _) = self
            .db
            .get_adr_sections(&old_id)?
            .ok_or_else(|| MemoryError::NotFound(format!("ADR sidecar missing for {}", old_id)))?;

        if !current_status.can_transition_to(target_status) {
            return Err(MemoryError::InvalidType(format!(
                "invalid ADR status transition: {} -> {}",
                current_status, target_status
            )));
        }

        self.db.update_adr_status(&old_id, target_status)?;

        Ok(json!({
            "id": old_id,
            "number": input.number,
            "status": target_status.as_str(),
        }))
    }

    fn adr_list(&self, arguments: Value) -> Result<Value, MemoryError> {
        use std::str::FromStr;

        let input: AdrListInput = parse_args("adr_list", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let status_filter = input
            .status
            .as_deref()
            .map(AdrStatus::from_str)
            .transpose()
            .map_err(|e| MemoryError::InvalidType(e.to_string()))?;

        let rows = self.db.list_adrs(&project, status_filter)?;

        let items: Vec<Value> = rows
            .into_iter()
            .map(|(number, status, title, id)| {
                json!({
                    "number": number,
                    "status": status.as_str(),
                    "title": title,
                    "id": id,
                })
            })
            .collect();

        Ok(json!(items))
    }

    fn adr_show(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: AdrShowInput = parse_args("adr_show", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let id = self
            .db
            .get_adr_by_number(&project, input.number)?
            .ok_or_else(|| MemoryError::NotFound(format!("ADR-{:04} not found", input.number)))?;

        let (number, status, sections) = self
            .db
            .get_adr_sections(&id)?
            .ok_or_else(|| MemoryError::NotFound(format!("ADR sidecar missing for {}", id)))?;

        let _ = self.db.record_access(&id);

        Ok(json!({
            "id": id,
            "number": number,
            "status": status.as_str(),
            "title": sections.title,
            "context": sections.context,
            "decision": sections.decision,
            "consequences": sections.consequences,
        }))
    }

    fn adr_export(&self, arguments: Value) -> Result<Value, MemoryError> {
        let input: AdrExportInput = parse_args("adr_export", arguments)?;
        let project = self.resolve_project(input.project.as_deref())?;

        let dir = adr_export_target_dir(input.dir.as_deref());
        let paths = export_adr_to_disk(&self.db, &project, &dir, input.number, input.dry_run)?;

        let files: Vec<String> = paths
            .into_iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        Ok(json!({
            "dry_run": input.dry_run,
            "dir": dir.to_string_lossy(),
            "files": files,
        }))
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
        project: &str,
        memory_id: &str,
        embedding: &[f32],
        content: &str,
        _importance: f64,
    ) -> Result<Option<String>, MemoryError> {
        use crate::memory::MemoryCluster;

        let clusters = self.db.get_clusters_for_project(project)?;

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
                project_id: project.to_string(),
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
    use crate::memory::{HandoffSections, Memory, MemoryType};
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

    /// A superseded memory must not come back from search, and its match must not vanish
    /// either: the successor takes its place, marked with what it replaced.
    ///
    /// This is the whole point of supersession being a retrieval concept rather than a
    /// note in the content. Marking a memory superseded in its own text leaves it ranking
    /// exactly where it did, which is what pushes people to delete it instead.
    #[test]
    fn superseded_memory_is_replaced_by_its_successor_in_query() {
        let db = Database::open_in_memory().unwrap();
        let project = "supersede-query";
        db.get_or_create_project(project, "Test").unwrap();
        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project.to_string(),
            None,
            SearchMode::default(),
        );

        let old = handler
            .memory_store(json!({
                "content": "The day-26 collapse is caused by connection pool exhaustion.",
                "type": "decision",
                "tags": ["collapse"],
            }))
            .unwrap();
        let old_id = old["id"].as_str().unwrap().to_string();

        let new = handler
            .memory_store(json!({
                "content": "The day-26 collapse is caused by a clock skew in the scheduler, not the pool.",
                "type": "decision",
                "tags": ["collapse"],
                "supersedes": [old_id.clone()],
            }))
            .unwrap();
        let new_id = new["id"].as_str().unwrap().to_string();
        assert_eq!(new["superseded"][0].as_str(), Some(old_id.as_str()));

        let results = handler
            .memory_query(json!({"query": "what causes the day-26 collapse", "limit": 10}))
            .unwrap();
        let memories = results["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "the query must not return silence");

        let ids: Vec<&str> = memories
            .iter()
            .map(|m| m["memory"]["id"].as_str().unwrap())
            .collect();
        assert!(
            !ids.contains(&old_id.as_str()),
            "the superseded memory must not be returned"
        );
        assert!(
            ids.contains(&new_id.as_str()),
            "the successor must be returned in its place"
        );

        // The successor also matched on its own merit here, so it appears exactly once.
        assert_eq!(
            ids.iter().filter(|id| **id == new_id).count(),
            1,
            "a successor reached both directly and by redirect must not be duplicated"
        );

        // Auditing still reaches the old memory.
        let raw = handler
            .memory_query(json!({
                "query": "what causes the day-26 collapse",
                "limit": 10,
                "include_superseded": true,
            }))
            .unwrap();
        let raw_ids: Vec<&str> = raw["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["memory"]["id"].as_str().unwrap())
            .collect();
        assert!(
            raw_ids.contains(&old_id.as_str()),
            "include_superseded must show what retrieval hides"
        );
    }

    /// A memory whose only match is a superseded predecessor still yields the successor,
    /// carrying `matched_via` to say where the match came from.
    #[test]
    fn redirect_annotates_where_the_match_came_from() {
        let db = Database::open_in_memory().unwrap();
        let project = "supersede-annotate";
        db.get_or_create_project(project, "Test").unwrap();
        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project.to_string(),
            None,
            SearchMode::default(),
        );

        let old = handler
            .memory_store(json!({
                "content": "Deployments are gated on the nightly canary in Jenkins.",
                "type": "fact",
            }))
            .unwrap();
        let old_id = old["id"].as_str().unwrap().to_string();

        handler
            .memory_store(json!({
                "content": "Release gating moved to GitHub Actions required checks.",
                "type": "fact",
                "supersedes": [old_id.clone()],
            }))
            .unwrap();

        // Query the *old* wording, which the successor does not use.
        let results = handler
            .memory_query(json!({"query": "nightly canary Jenkins gate", "limit": 5}))
            .unwrap();
        let memories = results["memories"].as_array().unwrap();
        assert!(
            !memories.is_empty(),
            "must redirect rather than return nothing"
        );

        let redirected = memories
            .iter()
            .find(|m| !m["matched_via"].is_null())
            .expect("at least one result should be a redirect");
        assert_eq!(
            redirected["matched_via"]["superseded_id"].as_str(),
            Some(old_id.as_str())
        );
    }

    /// A dead memory is excluded outright: there is nothing current to point at.
    #[test]
    fn dead_memory_is_excluded_from_query() {
        let db = Database::open_in_memory().unwrap();
        let project = "dead-query";
        db.get_or_create_project(project, "Test").unwrap();
        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project.to_string(),
            None,
            SearchMode::default(),
        );

        let stored = handler
            .memory_store(json!({
                "content": "The legacy billing cron runs at 03:00 on cron-box-2.",
                "type": "fact",
            }))
            .unwrap();
        let id = stored["id"].as_str().unwrap().to_string();

        let found = handler
            .memory_query(json!({"query": "legacy billing cron schedule", "limit": 5}))
            .unwrap();
        assert!(!found["memories"].as_array().unwrap().is_empty());

        handler
            .memory_update(json!({
                "id": id,
                "dead": true,
                "dead_reason": "cron-box-2 was decommissioned",
            }))
            .unwrap();

        let after = handler
            .memory_query(json!({"query": "legacy billing cron schedule", "limit": 5}))
            .unwrap();
        let ids: Vec<&str> = after["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["memory"]["id"].as_str().unwrap())
            .collect();
        assert!(
            !ids.contains(&id.as_str()),
            "dead memories must not surface"
        );
    }

    /// `memory_update` replaces content wholesale, so it must hand back what it replaced
    /// and leave a recoverable snapshot.
    #[test]
    fn update_returns_the_content_it_replaced() {
        let db = Database::open_in_memory().unwrap();
        let project = "update-echo";
        db.get_or_create_project(project, "Test").unwrap();
        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project.to_string(),
            None,
            SearchMode::default(),
        );

        let stored = handler
            .memory_store(json!({
                "content": "Original wording, including a tail nobody re-read.",
                "type": "fact",
            }))
            .unwrap();
        let id = stored["id"].as_str().unwrap().to_string();

        let updated = handler
            .memory_update(json!({"id": id, "content": "A reconstruction."}))
            .unwrap();

        assert_eq!(updated["content_replaced"], json!(true));
        assert_eq!(
            updated["previous"]["content"].as_str(),
            Some("Original wording, including a tail nobody re-read.")
        );

        let restored = handler.memory_restore(json!({"id": id})).unwrap();
        assert_eq!(
            restored["memory"]["content"].as_str(),
            Some("Original wording, including a tail nobody re-read.")
        );
    }

    /// Deleting returns the memory it destroyed, and the delete stays undoable.
    #[test]
    fn delete_returns_what_it_destroyed_and_is_recoverable() {
        let db = Database::open_in_memory().unwrap();
        let project = "delete-echo";
        db.get_or_create_project(project, "Test").unwrap();
        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project.to_string(),
            None,
            SearchMode::default(),
        );

        let stored = handler
            .memory_store(json!({
                "content": "A finding, plus a method lesson that is in no commit.",
                "type": "debug",
            }))
            .unwrap();
        let id = stored["id"].as_str().unwrap().to_string();

        let deleted = handler.memory_delete(json!({"id": id.clone()})).unwrap();
        assert_eq!(
            deleted["deleted"]["content"].as_str(),
            Some("A finding, plus a method lesson that is in no commit.")
        );
        assert_eq!(deleted["recoverable"], json!(true));

        let trash = handler.memory_trash(json!({})).unwrap();
        assert_eq!(trash["count"], json!(1));

        handler.memory_restore(json!({"id": id.clone()})).unwrap();
        assert!(db.get_memory(&id).unwrap().is_some());
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

    /// Regression: hook-captured Debug memories must not appear in `linked_memories`
    /// on resume, even when an older auto-link created the relationship.  Also
    /// confirms that `Memory.content` in legitimate linked memories is truncated to
    /// a preview so a single oversized linked memory cannot blow up the response.
    #[test]
    fn handoff_resume_filters_hook_captures_and_trims_content() {
        use crate::memory::{RelationType, Relationship};

        let db = Database::open_in_memory().unwrap();
        let project_id = "hook-filter-proj";
        db.get_or_create_project(project_id, "Hook Filter").unwrap();

        let sections = test_sections("Session summary", None);
        insert_test_handoff(
            &db,
            project_id,
            "ho-main",
            "main",
            &sections,
            &[("summary", dummy_vec(0.5))],
        );

        // Insert a hook-captured Debug memory and a curated Debug memory, then link
        // both to the handoff as if a prior auto-link had run.
        let now = chrono::Utc::now().timestamp();
        let hook_mem = Memory {
            id: "mem_hook".to_string(),
            project_id: project_id.to_string(),
            memory_type: MemoryType::Debug,
            content: "Edit failed: ".to_string() + &"x".repeat(50_000),
            summary: None,
            tags: vec!["hook".into(), "failure".into(), "Edit".into()],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: Some("main".into()),
            merged_from: None,
            external_artifacts: None,
            pinned: false,
            global: false,
        };
        let curated_mem = Memory {
            id: "mem_curated".to_string(),
            project_id: project_id.to_string(),
            memory_type: MemoryType::Debug,
            content: "Genuine debug note: ".to_string() + &"y".repeat(5_000),
            summary: None,
            tags: vec!["manual".into()],
            ..hook_mem.clone()
        };
        db.store_memory(&hook_mem).unwrap();
        db.store_memory(&curated_mem).unwrap();

        for target in ["mem_hook", "mem_curated"] {
            db.create_relationship(&Relationship {
                id: format!("rel_{}", target),
                source_id: "ho-main".into(),
                target_id: target.into(),
                relation_type: RelationType::DerivedFrom,
                strength: 1.0,
                created_at: now,
            })
            .unwrap();
        }

        let result = resume_handoff_with_vec(
            &db,
            project_id,
            Some("main"),
            Some(dummy_vec(0.5)),
            5,
            false,
            None,
        )
        .expect("resume_handoff_with_vec must succeed");

        let ids: Vec<&str> = result
            .linked_memories
            .iter()
            .map(|m| m.id.as_str())
            .collect();
        assert!(
            !ids.contains(&"mem_hook"),
            "hook-captured memories must be excluded from linked_memories, got {:?}",
            ids
        );
        assert!(
            ids.contains(&"mem_curated"),
            "curated memories must still surface, got {:?}",
            ids
        );

        // Curated memory content was 5000+ chars; resume must trim to the preview cap.
        let curated = result
            .linked_memories
            .iter()
            .find(|m| m.id == "mem_curated")
            .expect("curated memory present");
        assert!(
            curated.content.chars().count() <= 400,
            "linked_memories content must be truncated, got {} chars",
            curated.content.chars().count()
        );
        assert!(
            curated.content.contains("[truncated"),
            "truncation marker must be present: {}",
            curated.content
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

    // ============================================
    // 5.1: ADR handler tests
    // ============================================

    /// 5.1 test 1: adr_create_assigns_number_and_pins
    /// Call adr_create handler; assert returned adr_number == 1, stored memory is pinned,
    /// memory_type == Adr, and branch == None (project-global).
    #[test]
    fn adr_create_assigns_number_and_pins() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "adr-handler-proj";
        db.get_or_create_project(project_id, "ADR Handler Test")
            .unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project_id.to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );

        let result = handler
            .handle_tool(
                "adr_create",
                json!({
                    "title": "Use SQLite for local storage",
                    "context": "We need a local, zero-dep database.",
                    "decision": "We will use SQLite via rusqlite.",
                    "consequences": "Simple deployment; no concurrent writes."
                }),
            )
            .expect("adr_create must succeed");

        assert_eq!(result["adr_number"], 1, "first ADR should be number 1");

        // Verify stored memory row.
        let adr_id = result["id"].as_str().expect("id must be a string");
        let memory = db.get_memory(adr_id).unwrap().unwrap();
        assert_eq!(
            memory.memory_type,
            MemoryType::Adr,
            "memory_type must be Adr"
        );
        assert!(memory.pinned, "ADR must be pinned by default");
        assert!(
            memory.branch.is_none(),
            "ADR must be project-global (branch == None)"
        );
    }

    /// 5.1 test 2: adr_create_bypasses_dedup
    /// Create two near-identical ADRs; assert numbers 1 and 2 and both persist
    /// (dedup did not merge them).
    #[test]
    fn adr_create_bypasses_dedup() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "adr-dedup-proj";
        db.get_or_create_project(project_id, "ADR Dedup Test")
            .unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project_id.to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );

        let first = handler
            .handle_tool(
                "adr_create",
                json!({
                    "title": "Use SQLite for local storage",
                    "context": "We need a local, zero-dep database.",
                    "decision": "We will use SQLite via rusqlite.",
                    "consequences": "Simple deployment; no concurrent writes."
                }),
            )
            .expect("first adr_create must succeed");

        // Near-identical second ADR — dedup would merge this if it were memory_store.
        let second = handler
            .handle_tool(
                "adr_create",
                json!({
                    "title": "Use SQLite for local storage",
                    "context": "We need a local, zero-dep database.",
                    "decision": "We will use SQLite via rusqlite.",
                    "consequences": "Simple deployment; no concurrent writes."
                }),
            )
            .expect("second adr_create must succeed");

        assert_eq!(first["adr_number"], 1, "first ADR should be number 1");
        assert_eq!(second["adr_number"], 2, "second ADR should be number 2");

        // Both memory rows must still exist.
        let id1 = first["id"].as_str().unwrap();
        let id2 = second["id"].as_str().unwrap();
        assert!(db.get_memory(id1).unwrap().is_some(), "ADR-1 must persist");
        assert!(db.get_memory(id2).unwrap().is_some(), "ADR-2 must persist");
    }

    /// 5.1 test 3: adr_update_status_valid_and_invalid
    /// proposed->accepted succeeds; accepted->proposed is rejected (InvalidType);
    /// a direct update to superseded is rejected with the "use ... supersede" message.
    #[test]
    fn adr_update_status_valid_and_invalid() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "adr-status-proj";
        db.get_or_create_project(project_id, "ADR Status Test")
            .unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project_id.to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );

        // Create an ADR in the proposed state.
        handler
            .handle_tool(
                "adr_create",
                json!({
                    "title": "Use SQLite",
                    "context": "Need a DB.",
                    "decision": "Use SQLite.",
                    "consequences": "Easy to embed."
                }),
            )
            .unwrap();

        // Valid transition: proposed -> accepted.
        let ok = handler.handle_tool(
            "adr_update_status",
            json!({"number": 1, "status": "accepted"}),
        );
        assert!(ok.is_ok(), "proposed->accepted must succeed, got {:?}", ok);

        // Invalid transition: accepted -> proposed.
        let err = handler.handle_tool(
            "adr_update_status",
            json!({"number": 1, "status": "proposed"}),
        );
        assert!(
            matches!(err, Err(MemoryError::InvalidType(_))),
            "accepted->proposed must be InvalidType, got {:?}",
            err
        );

        // Superseded must always be rejected via adr_update_status.
        let err_super = handler.handle_tool(
            "adr_update_status",
            json!({"number": 1, "status": "superseded"}),
        );
        assert!(
            matches!(err_super, Err(MemoryError::InvalidType(ref msg)) if msg.contains("supersede")),
            "direct superseded transition must be rejected with 'supersede' message, got {:?}",
            err_super
        );
    }

    /// 5.1 test 4: adr_supersede_flips_status_and_creates_edge
    /// Create ADR-1, then create ADR-2 with supersedes=1; assert ADR-1 becomes
    /// Superseded AND a Supersedes relationship from ADR-2 -> ADR-1 exists.
    #[test]
    fn adr_supersede_flips_status_and_creates_edge() {
        use crate::memory::RelationType;

        let db = Database::open_in_memory().unwrap();
        let project_id = "adr-supersede-proj";
        db.get_or_create_project(project_id, "ADR Supersede Test")
            .unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project_id.to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );

        // Create ADR-1 (accepted — supersession requires accepted or proposed->accepted first).
        let adr1_result = handler
            .handle_tool(
                "adr_create",
                json!({
                    "title": "Use SQLite",
                    "context": "Need a DB.",
                    "decision": "Use SQLite.",
                    "consequences": "Easy to embed.",
                    "status": "accepted"
                }),
            )
            .unwrap();
        let adr1_id = adr1_result["id"].as_str().unwrap().to_string();

        // Create ADR-2 that supersedes ADR-1.
        let adr2_result = handler
            .handle_tool(
                "adr_create",
                json!({
                    "title": "Use PostgreSQL",
                    "context": "Need a DB with concurrent writes.",
                    "decision": "Use PostgreSQL.",
                    "consequences": "Better concurrency.",
                    "supersedes": 1
                }),
            )
            .unwrap();
        let adr2_id = adr2_result["id"].as_str().unwrap().to_string();

        // ADR-1 must now be Superseded.
        let (_, adr1_status, _) = db.get_adr_sections(&adr1_id).unwrap().unwrap();
        assert_eq!(
            adr1_status,
            crate::memory::AdrStatus::Superseded,
            "ADR-1 must be Superseded after supersession"
        );

        // A Supersedes relationship from ADR-2 -> ADR-1 must exist.
        let rels = db.get_relationships_from(&adr2_id).unwrap();
        let supersedes_rel = rels
            .iter()
            .find(|r| r.target_id == adr1_id && r.relation_type == RelationType::Supersedes);
        assert!(
            supersedes_rel.is_some(),
            "Supersedes relationship from ADR-2 to ADR-1 must exist, got: {:?}",
            rels
        );
    }

    /// 5.1 test 5: create_with_invalid_supersede_does_not_create_adr
    ///
    /// Attempting to create an ADR that supersedes a non-existent number must return Err
    /// AND must not leave any new ADR in the database (no orphan).
    ///
    /// Also verifies that superseding a `proposed` ADR (which cannot transition to
    /// `superseded` directly) returns Err without creating a new ADR.
    #[test]
    fn create_with_invalid_supersede_does_not_create_adr() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "adr-no-orphan-proj";
        db.get_or_create_project(project_id, "No Orphan Test")
            .unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");
        let handler = ToolHandler::new(
            db.clone(),
            embedding,
            project_id.to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );

        // Case 1: supersede a non-existent ADR number.
        let before = db.list_adrs(project_id, None).unwrap().len();
        let err = handler.handle_tool(
            "adr_create",
            json!({
                "title": "Should Not Exist",
                "context": "ctx",
                "decision": "dec",
                "consequences": "cons",
                "supersedes": 999
            }),
        );
        assert!(
            matches!(err, Err(MemoryError::NotFound(_))),
            "superseding non-existent ADR must return NotFound, got {:?}",
            err
        );
        let after = db.list_adrs(project_id, None).unwrap().len();
        assert_eq!(
            before, after,
            "no ADR must be created when supersession target does not exist (orphan check)"
        );

        // Case 2: create a proposed ADR, then attempt to supersede it via adr_create
        // (proposed -> superseded is not a valid transition — proposed can only go to
        // accepted or rejected first).
        handler
            .handle_tool(
                "adr_create",
                json!({
                    "title": "Proposed ADR",
                    "context": "ctx",
                    "decision": "dec",
                    "consequences": "cons",
                    "status": "proposed"
                }),
            )
            .expect("creating the proposed ADR must succeed");

        let count_after_first = db.list_adrs(project_id, None).unwrap().len();
        assert_eq!(count_after_first, 1, "exactly one ADR must exist now");

        let err2 = handler.handle_tool(
            "adr_create",
            json!({
                "title": "Superseder",
                "context": "ctx",
                "decision": "dec",
                "consequences": "cons",
                "supersedes": 1
            }),
        );
        assert!(
            matches!(err2, Err(MemoryError::InvalidType(_))),
            "superseding a proposed ADR must return InvalidType, got {:?}",
            err2
        );
        let count_after_failed = db.list_adrs(project_id, None).unwrap().len();
        assert_eq!(
            count_after_first, count_after_failed,
            "failed supersede must not leave an orphan ADR"
        );
    }
}
