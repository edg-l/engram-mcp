#![allow(clippy::manual_async_fn)]

mod cache;
mod db;
mod decay;
mod embedding;
mod error;
mod export;
mod format;
mod memory;
mod summarize;
mod tools;

use rmcp::ErrorData as McpError;
use rmcp::model::{
    Annotated, CallToolRequestParams, CallToolResult, Content, GetPromptRequestParams,
    GetPromptResult, Implementation, ListPromptsResult, ListResourceTemplatesResult,
    ListResourcesResult, ListToolsResult, PaginatedRequestParams, Prompt, PromptArgument,
    PromptMessage, PromptMessageRole, RawResource, RawResourceTemplate, ReadResourceRequestParams,
    ReadResourceResult, ResourceContents, Role, ServerCapabilities, ServerInfo,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::transport::stdio;
use rmcp::{ServerHandler, ServiceExt};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing_subscriber::{self, EnvFilter};

use crate::db::Database;
use crate::embedding::EmbeddingService;
use crate::error::MemoryError;
use crate::tools::{ToolHandler, get_tool_definitions};

/// Find the git repository root by walking up from current directory.
/// Returns None if not in a git repository.
fn find_git_root() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        if current.join(".git").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Detect the current git branch.
/// Returns None if not in a git repository or on error.
/// Priority: ENGRAM_BRANCH env var > git detection
fn get_current_branch() -> Option<String> {
    // Check environment variable override first
    if let Ok(branch) = std::env::var("ENGRAM_BRANCH")
        && !branch.is_empty()
    {
        return Some(branch);
    }

    // Find git root
    let git_root = find_git_root()?;
    let git_dir = git_root.join(".git");

    // Try reading .git/HEAD directly (faster than spawning git process)
    if let Ok(head_content) = std::fs::read_to_string(git_dir.join("HEAD")) {
        let head = head_content.trim();
        if let Some(branch_ref) = head.strip_prefix("ref: refs/heads/") {
            return Some(branch_ref.to_string());
        }
        // Detached HEAD - use short SHA
        if head.len() >= 7 {
            return Some(format!("detached-{}", &head[..7]));
        }
    }

    // Fallback: try git command
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(&git_root)
        .output()
        && output.status.success()
    {
        let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if branch == "HEAD" {
            // Detached HEAD - get short SHA
            if let Ok(sha_output) = std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .current_dir(&git_root)
                .output()
                && sha_output.status.success()
            {
                let sha = String::from_utf8_lossy(&sha_output.stdout)
                    .trim()
                    .to_string();
                return Some(format!("detached-{}", sha));
            }
        } else {
            return Some(branch);
        }
    }

    None
}

/// Default decay interval: 1 hour
const DECAY_INTERVAL_SECS: u64 = 3600;

/// Run the background decay job that periodically updates relevance scores
async fn run_decay_job(db_path: PathBuf, project_id: String) {
    let interval = Duration::from_secs(
        std::env::var("ENGRAM_DECAY_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DECAY_INTERVAL_SECS),
    );

    loop {
        tokio::time::sleep(interval).await;

        match Database::open(&db_path) {
            Ok(db) => {
                // Get project's decay rate
                let decay_rate = db
                    .get_project(&project_id)
                    .ok()
                    .flatten()
                    .map(|p| p.decay_rate)
                    .unwrap_or(0.01);

                match db.update_relevance_scores(&project_id, decay_rate) {
                    Ok(count) => {
                        if count > 0 {
                            tracing::debug!("Decay job updated {} memories", count);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Decay job failed: {}", e);
                    }
                }

                match db.auto_prune_dead_memories(&project_id) {
                    Ok(pruned_ids) => {
                        if !pruned_ids.is_empty() {
                            tracing::debug!(
                                "Auto-pruned {} dead memories: {:?}",
                                pruned_ids.len(),
                                pruned_ids
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Auto-prune failed: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Decay job failed to open database: {}", e);
            }
        }
    }
}

/// Default re-clustering interval: 6 hours
const RECLUSTER_INTERVAL_SECS: u64 = 21600;

/// Run the background re-clustering job
async fn run_recluster_job(db_path: PathBuf, project_id: String) {
    let interval = Duration::from_secs(
        std::env::var("ENGRAM_RECLUSTER_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(RECLUSTER_INTERVAL_SECS),
    );

    // Wait one interval before first run
    tokio::time::sleep(interval).await;

    loop {
        match Database::open(&db_path) {
            Ok(db) => {
                match recluster_project(&db, &project_id) {
                    Ok((merged, split)) => {
                        if merged > 0 || split > 0 {
                            tracing::debug!(
                                "Re-clustering: merged {} clusters, split {} members",
                                merged,
                                split
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Re-clustering job failed: {}", e);
                    }
                }

                match dedup_within_clusters(&db, &project_id) {
                    Ok(count) => {
                        if count > 0 {
                            tracing::debug!(
                                "Background dedup merged {} duplicate memories within clusters",
                                count
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Background dedup failed: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Re-clustering job failed to open database: {}", e);
            }
        }

        tokio::time::sleep(interval).await;
    }
}

/// Re-cluster a project: merge similar clusters, split divergent members.
fn recluster_project(db: &Database, project_id: &str) -> Result<(usize, usize), MemoryError> {
    use crate::embedding::cosine_similarity;

    let clusters = db.get_clusters_for_project(project_id)?;
    let mut merged_count = 0usize;
    let mut split_count = 0usize;

    // Build embedding map once (reused throughout)
    let all_embeddings = db.get_all_embeddings_for_project(project_id)?;
    let embedding_map: std::collections::HashMap<String, Vec<f32>> =
        all_embeddings.into_iter().collect();

    // Phase 1: Merge similar clusters (centroid similarity >= 0.80)
    let mut skip_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for i in 0..clusters.len() {
        if skip_ids.contains(&clusters[i].id) {
            continue;
        }
        let Some(ref centroid_i) = clusters[i].centroid else {
            continue;
        };

        for j in (i + 1)..clusters.len() {
            if skip_ids.contains(&clusters[j].id) {
                continue;
            }
            let Some(ref centroid_j) = clusters[j].centroid else {
                continue;
            };

            let similarity = cosine_similarity(centroid_i, centroid_j);
            if similarity >= 0.80 {
                // Merge cluster j into cluster i
                let member_ids = db.get_cluster_member_ids(&clusters[j].id)?;
                for member_id in &member_ids {
                    db.remove_from_cluster(member_id)?;
                    db.add_to_cluster(&clusters[i].id, member_id)?;
                }
                skip_ids.insert(clusters[j].id.clone());
                merged_count += 1;
            }
        }
    }

    // Phase 2: Recalculate centroids after merges (before split evaluation)
    recalculate_all_centroids(db, project_id, &embedding_map)?;
    db.delete_empty_clusters(project_id)?;

    // Phase 3: Split divergent members (similarity to centroid < 0.50)
    // Use fresh clusters with updated centroids
    let updated_clusters = db.get_clusters_for_project(project_id)?;
    for cluster in &updated_clusters {
        let Some(ref centroid) = cluster.centroid else {
            continue;
        };

        let member_ids = db.get_cluster_member_ids(&cluster.id)?;
        for member_id in &member_ids {
            if let Some(member_embedding) = embedding_map.get(member_id) {
                let similarity = cosine_similarity(centroid, member_embedding);
                if similarity < 0.50 {
                    db.remove_from_cluster(member_id)?;
                    split_count += 1;

                    // Try to find a better cluster
                    let mut best_match: Option<(String, f32)> = None;
                    for other_cluster in &updated_clusters {
                        if other_cluster.id == cluster.id {
                            continue;
                        }
                        if let Some(ref other_centroid) = other_cluster.centroid {
                            let sim = cosine_similarity(member_embedding, other_centroid);
                            if sim >= 0.75
                                && (best_match.is_none() || sim > best_match.as_ref().unwrap().1)
                            {
                                best_match = Some((other_cluster.id.clone(), sim));
                            }
                        }
                    }

                    if let Some((better_cluster_id, _)) = best_match {
                        db.add_to_cluster(&better_cluster_id, member_id)?;
                    }
                }
            }
        }
    }

    // Phase 4: Final cleanup
    db.delete_empty_clusters(project_id)?;
    recalculate_all_centroids(db, project_id, &embedding_map)?;

    Ok((merged_count, split_count))
}

/// Dedup memories within each cluster: find same-type pairs with similarity >= 0.90 and merge.
/// Global memories always survive: if one is global and the other is local, the global survives.
/// Returns the number of merges performed.
fn dedup_within_clusters(db: &Database, project_id: &str) -> Result<usize, MemoryError> {
    use crate::embedding::cosine_similarity;

    let clusters = db.get_clusters_for_project(project_id)?;
    let all_embeddings = db.get_all_embeddings_for_project_and_global(project_id)?;
    let embedding_map: std::collections::HashMap<String, Vec<f32>> =
        all_embeddings.into_iter().collect();

    let mut merge_count = 0usize;
    // Track consumed IDs so we don't try to merge already-deleted memories
    let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();

    for cluster in &clusters {
        let member_ids = db.get_cluster_member_ids(&cluster.id)?;
        if member_ids.len() < 2 {
            continue;
        }

        for (i, id_i) in member_ids.iter().enumerate() {
            if consumed.contains(id_i) {
                continue;
            }
            let Some(emb_i) = embedding_map.get(id_i) else {
                continue;
            };
            let Some(mem_i) = db.get_memory(id_i)? else {
                continue;
            };

            for id_j in member_ids.iter().skip(i + 1) {
                if consumed.contains(id_j) {
                    continue;
                }
                let Some(emb_j) = embedding_map.get(id_j) else {
                    continue;
                };
                let Some(mem_j) = db.get_memory(id_j)? else {
                    continue;
                };

                // Only merge same-type memories
                if mem_i.memory_type != mem_j.memory_type {
                    continue;
                }

                let similarity = cosine_similarity(emb_i, emb_j);
                if similarity >= 0.90 {
                    // Determine survivor: global always wins; otherwise keep i (first seen)
                    let (survivor_id, consumed_id, consumed_preview) =
                        if mem_j.global && !mem_i.global {
                            // j is global, i is local: j survives
                            let preview: String = mem_i.content.chars().take(100).collect();
                            (mem_j.id.clone(), mem_i.id.clone(), preview)
                        } else {
                            // i survives (i is global, or both same scope)
                            let preview: String = mem_j.content.chars().take(100).collect();
                            (mem_i.id.clone(), mem_j.id.clone(), preview)
                        };

                    db.merge_memories(&survivor_id, &consumed_id, &consumed_preview)?;
                    consumed.insert(consumed_id);
                    merge_count += 1;
                }
            }
        }
    }

    Ok(merge_count)
}

/// Recalculate centroids and summaries for all clusters in a project.
fn recalculate_all_centroids(
    db: &Database,
    project_id: &str,
    embedding_map: &std::collections::HashMap<String, Vec<f32>>,
) -> Result<(), MemoryError> {
    let clusters = db.get_clusters_for_project(project_id)?;
    for cluster in &clusters {
        let member_ids = db.get_cluster_member_ids(&cluster.id)?;
        if member_ids.is_empty() {
            continue;
        }

        // Compute new centroid
        let mut sum: Option<Vec<f32>> = None;
        let mut count = 0usize;
        for member_id in &member_ids {
            if let Some(emb) = embedding_map.get(member_id) {
                count += 1;
                match &mut sum {
                    None => sum = Some(emb.clone()),
                    Some(s) => {
                        for (i, v) in emb.iter().enumerate() {
                            if i < s.len() {
                                s[i] += v;
                            }
                        }
                    }
                }
            }
        }

        if let Some(mut centroid) = sum {
            let c = count as f32;
            for v in &mut centroid {
                *v /= c;
            }
            let members_map = db.get_memories_batch(&member_ids)?;
            let best_member = members_map.values().max_by(|a, b| {
                a.importance
                    .partial_cmp(&b.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let summary = best_member
                .map(|m| crate::summarize::extract_first_sentence(&m.content))
                .unwrap_or_else(|| "Cluster".to_string());

            db.update_cluster_centroid(&cluster.id, &centroid, &summary)?;
        }
    }
    Ok(())
}

struct MemoryServer {
    tool_handler: Arc<RwLock<Option<ToolHandler>>>,
    db_path: PathBuf,
    project_id: String,
    current_branch: Option<String>,
}

impl MemoryServer {
    fn new(db_path: PathBuf, project_id: String, current_branch: Option<String>) -> Self {
        Self {
            tool_handler: Arc::new(RwLock::new(None)),
            db_path,
            project_id,
            current_branch,
        }
    }

    async fn ensure_initialized(&self) -> Result<(), McpError> {
        let mut handler = self.tool_handler.write().await;
        if handler.is_none() {
            let db = Database::open(&self.db_path)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            // Ensure project exists
            db.get_or_create_project(&self.project_id, &self.project_id)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            let embedding = EmbeddingService::new()
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            *handler = Some(ToolHandler::new(
                db,
                embedding,
                self.project_id.clone(),
                self.current_branch.clone(),
            ));
        }
        Ok(())
    }
}

impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_prompts()
                .build(),
        )
        .with_server_info(
            Implementation::new("engram", env!("CARGO_PKG_VERSION"))
                .with_title("Engram MCP Server"),
        )
        .with_instructions(include_str!("instructions.md"))
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
        async move { Ok(ListToolsResult::with_all_items(get_tool_definitions())) }
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, McpError>> + Send + '_ {
        async move {
            self.ensure_initialized().await?;

            let handler = self.tool_handler.read().await;
            let handler = handler
                .as_ref()
                .ok_or_else(|| McpError::internal_error("Server not initialized", None))?;

            let args = request.arguments.map(Value::Object).unwrap_or(Value::Null);
            let result = handler
                .handle_tool(&request.name, args)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            // Human-readable format (not sent to LLM)
            let formatted = format::format_tool_result(&request.name, &result);
            // Compact JSON for LLM consumption
            let compact_json = format::compact_tool_result(&request.name, &result);

            Ok(CallToolResult::success(vec![
                // Human-readable: audience=User means NOT sent to LLM
                Content::text(formatted).with_audience(vec![Role::User]),
                // Compact JSON: audience=Assistant means sent to LLM only
                Content::text(compact_json).with_audience(vec![Role::Assistant]),
            ]))
        }
    }

    fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourcesResult, McpError>> + Send + '_ {
        async move {
            let db = Database::open(&self.db_path)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            let memories = db
                .get_all_memories_for_project(&self.project_id)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            let resources: Vec<_> = memories
                .into_iter()
                .map(|m| {
                    Annotated::new(
                        RawResource {
                            uri: format!("memory://{}/{}", self.project_id, m.id),
                            name: m
                                .summary
                                .unwrap_or_else(|| m.content.chars().take(50).collect()),
                            title: Some(format!("{:?}", m.memory_type)),
                            description: Some(m.content.chars().take(200).collect()),
                            mime_type: Some("text/plain".to_string()),
                            size: Some(m.content.len() as u32),
                            icons: None,
                            meta: None,
                        },
                        None,
                    )
                })
                .collect();

            Ok(ListResourcesResult::with_all_items(resources))
        }
    }

    fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourceTemplatesResult, McpError>> + Send + '_
    {
        async move {
            Ok(ListResourceTemplatesResult::with_all_items(vec![
                Annotated::new(
                    RawResourceTemplate {
                        uri_template: format!("memory://{}/{{memory_id}}", self.project_id),
                        name: "Memory".to_string(),
                        title: Some("Read a specific memory".to_string()),
                        description: Some("Access a memory by its ID".to_string()),
                        mime_type: Some("text/plain".to_string()),
                        icons: None,
                    },
                    None,
                ),
            ]))
        }
    }

    fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ReadResourceResult, McpError>> + Send + '_ {
        async move {
            // Parse URI: memory://{project_id}/{memory_id}
            let uri = &request.uri;
            let memory_id = uri
                .strip_prefix(&format!("memory://{}/", self.project_id))
                .ok_or_else(|| McpError::invalid_params("Invalid memory URI", None))?;

            let db = Database::open(&self.db_path)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            let memory = db
                .get_memory(memory_id)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?
                .ok_or_else(|| McpError::invalid_params("Memory not found", None))?;

            // Record access for reinforcement
            let _ = db.record_access(memory_id);

            // Format memory as readable text
            let content = format!(
                "Type: {:?}\nImportance: {:.2}\nRelevance: {:.2}\nTags: {}\nCreated: {}\n\n{}",
                memory.memory_type,
                memory.importance,
                memory.relevance_score,
                memory.tags.join(", "),
                chrono::DateTime::from_timestamp(memory.created_at, 0)
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                memory.content
            );

            Ok(ReadResourceResult::new(vec![ResourceContents::text(
                content,
                uri.clone(),
            )]))
        }
    }

    fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListPromptsResult, McpError>> + Send + '_ {
        async move {
            Ok(ListPromptsResult::with_all_items(vec![Prompt::new(
                "recall_context",
                Some("Recall relevant memories for a given context or question"),
                Some(vec![
                    PromptArgument::new("context")
                        .with_title("Context")
                        .with_description("The context or question to find relevant memories for")
                        .with_required(true),
                ]),
            )]))
        }
    }

    fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<GetPromptResult, McpError>> + Send + '_ {
        async move {
            if request.name != "recall_context" {
                return Err(McpError::invalid_params(
                    format!("Unknown prompt: {}", request.name),
                    None,
                ));
            }

            // Get context argument
            let context = request
                .arguments
                .as_ref()
                .and_then(|args| args.get("context"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| McpError::invalid_params("Missing 'context' argument", None))?;

            // Ensure initialized (reuses existing ToolHandler with its EmbeddingService)
            self.ensure_initialized().await?;

            let handler = self.tool_handler.read().await;
            let handler = handler
                .as_ref()
                .ok_or_else(|| McpError::internal_error("Server not initialized", None))?;

            let db = handler.database();
            let embedding_service = handler.embedding_service();

            // Generate query embedding (reuses initialized EmbeddingService)
            let query_embedding = embedding_service
                .embed(context)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            // Get all embeddings for the project
            let embeddings = db
                .get_all_embeddings_for_project(&self.project_id)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            // Calculate similarities and sort
            use crate::embedding::cosine_similarity;
            let mut scored_ids: Vec<(String, f32)> = embeddings
                .iter()
                .map(|(id, vec)| (id.clone(), cosine_similarity(&query_embedding, vec)))
                .collect();
            scored_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Get top 5 relevant memories
            let mut memories_text = String::new();
            for (id, score) in scored_ids.into_iter().take(5) {
                if score < 0.3 {
                    break;
                }
                if let Ok(Some(memory)) = db.get_memory(&id) {
                    memories_text.push_str(&format!(
                        "---\n**[{:?}]** (relevance: {:.2}, similarity: {:.2})\n{}\n",
                        memory.memory_type, memory.relevance_score, score, memory.content
                    ));
                    // Record access for reinforcement
                    let _ = db.record_access(&id);
                }
            }

            if memories_text.is_empty() {
                memories_text = "No relevant memories found for this context.".to_string();
            }

            Ok(GetPromptResult::new(vec![PromptMessage::new_text(
                PromptMessageRole::User,
                format!(
                    "Here are relevant memories from the project knowledge base:\n\n{}\n\nUse these memories to inform your response about: {}",
                    memories_text, context
                ),
            )])
            .with_description(format!("Relevant memories for: {}", context)))
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("engram=info".parse()?))
        .with_writer(std::io::stderr)
        .init();

    // Determine database path
    let db_path = std::env::var("ENGRAM_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("engram")
                .join("memories.db")
        });

    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Determine project ID (from env, git root, or current directory)
    let project_id = std::env::var("ENGRAM_PROJECT").unwrap_or_else(|_| {
        if let Some(git_root) = find_git_root() {
            git_root.to_string_lossy().to_string()
        } else {
            std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "default".to_string())
        }
    });

    // Detect current git branch
    let current_branch = get_current_branch();

    tracing::info!("Starting Engram MCP server");
    tracing::info!("Database: {}", db_path.display());
    tracing::info!("Project: {}", project_id);
    if let Some(ref branch) = current_branch {
        tracing::info!("Branch: {}", branch);
    }

    // Spawn background decay job
    let decay_db_path = db_path.clone();
    let decay_project_id = project_id.clone();
    tokio::spawn(async move {
        run_decay_job(decay_db_path, decay_project_id).await;
    });

    // Spawn background re-clustering job
    let recluster_db_path = db_path.clone();
    let recluster_project_id = project_id.clone();
    tokio::spawn(async move {
        run_recluster_job(recluster_db_path, recluster_project_id).await;
    });

    let server = MemoryServer::new(db_path, project_id, current_branch);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
