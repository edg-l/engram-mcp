#![allow(clippy::manual_async_fn)]

mod db;
mod decay;
mod embedding;
mod error;
mod export;
mod memory;
mod summarize;
mod tools;

use rmcp::ErrorData as McpError;
use rmcp::model::{
    Annotated, CallToolRequestParams, CallToolResult, Content, GetPromptRequestParams,
    GetPromptResult, Implementation, ListPromptsResult, ListResourceTemplatesResult,
    ListResourcesResult, ListToolsResult, PaginatedRequestParams, Prompt, PromptArgument,
    PromptMessage, PromptMessageRole, PromptsCapability, RawResource, RawResourceTemplate,
    ReadResourceRequestParams, ReadResourceResult, ResourceContents, ResourcesCapability,
    ServerCapabilities, ServerInfo, ToolsCapability,
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
            }
            Err(e) => {
                tracing::warn!("Decay job failed to open database: {}", e);
            }
        }
    }
}

struct MemoryServer {
    tool_handler: Arc<RwLock<Option<ToolHandler>>>,
    db_path: PathBuf,
    project_id: String,
}

impl MemoryServer {
    fn new(db_path: PathBuf, project_id: String) -> Self {
        Self {
            tool_handler: Arc::new(RwLock::new(None)),
            db_path,
            project_id,
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

            *handler = Some(ToolHandler::new(db, embedding, self.project_id.clone()));
        }
        Ok(())
    }
}

impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: Some(false),
                }),
                resources: Some(ResourcesCapability {
                    subscribe: Some(false),
                    list_changed: Some(false),
                }),
                prompts: Some(PromptsCapability {
                    list_changed: Some(false),
                }),
                ..Default::default()
            },
            server_info: Implementation {
                name: "engram".to_string(),
                title: Some("Engram MCP Server".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "A persistent memory system for AI agents. Use memory_store to save facts, \
                decisions, and patterns. Use memory_query to search for relevant memories."
                    .to_string(),
            ),
        }
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
        async move {
            Ok(ListToolsResult {
                tools: get_tool_definitions(),
                next_cursor: None,
                meta: None,
            })
        }
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

            Ok(CallToolResult {
                content: vec![Content::text(
                    serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string()),
                )],
                structured_content: None,
                is_error: Some(false),
                meta: None,
            })
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

            Ok(ListResourcesResult {
                resources,
                next_cursor: None,
                meta: None,
            })
        }
    }

    fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourceTemplatesResult, McpError>> + Send + '_
    {
        async move {
            Ok(ListResourceTemplatesResult {
                resource_templates: vec![Annotated::new(
                    RawResourceTemplate {
                        uri_template: format!("memory://{}/{{memory_id}}", self.project_id),
                        name: "Memory".to_string(),
                        title: Some("Read a specific memory".to_string()),
                        description: Some("Access a memory by its ID".to_string()),
                        mime_type: Some("text/plain".to_string()),
                        icons: None,
                    },
                    None,
                )],
                next_cursor: None,
                meta: None,
            })
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

            Ok(ReadResourceResult {
                contents: vec![ResourceContents::text(content, uri.clone())],
            })
        }
    }

    fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListPromptsResult, McpError>> + Send + '_ {
        async move {
            Ok(ListPromptsResult {
                prompts: vec![Prompt::new(
                    "recall_context",
                    Some("Recall relevant memories for a given context or question"),
                    Some(vec![PromptArgument {
                        name: "context".to_string(),
                        title: Some("Context".to_string()),
                        description: Some(
                            "The context or question to find relevant memories for".to_string(),
                        ),
                        required: Some(true),
                    }]),
                )],
                next_cursor: None,
                meta: None,
            })
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

            // Initialize embedding service and database
            let db = Database::open(&self.db_path)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            let embedding_service = EmbeddingService::new()
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            // Generate query embedding
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

            Ok(GetPromptResult {
                description: Some(format!("Relevant memories for: {}", context)),
                messages: vec![PromptMessage::new_text(
                    PromptMessageRole::User,
                    format!(
                        "Here are relevant memories from the project knowledge base:\n\n{}\n\nUse these memories to inform your response about: {}",
                        memories_text, context
                    ),
                )],
            })
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

    // Determine project ID (from env or current directory)
    let project_id = std::env::var("ENGRAM_PROJECT").unwrap_or_else(|_| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|s| s.to_string_lossy().to_string()))
            .unwrap_or_else(|| "default".to_string())
    });

    tracing::info!("Starting Engram MCP server");
    tracing::info!("Database: {}", db_path.display());
    tracing::info!("Project: {}", project_id);

    // Spawn background decay job
    let decay_db_path = db_path.clone();
    let decay_project_id = project_id.clone();
    tokio::spawn(async move {
        run_decay_job(decay_db_path, decay_project_id).await;
    });

    let server = MemoryServer::new(db_path, project_id);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
