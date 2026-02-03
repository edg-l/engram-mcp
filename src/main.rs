mod db;
mod decay;
mod embedding;
mod error;
mod memory;
mod tools;

use rmcp::ErrorData as McpError;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, ToolsCapability,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::transport::stdio;
use rmcp::{ServerHandler, ServiceExt};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber::{self, EnvFilter};

use crate::db::Database;
use crate::embedding::EmbeddingService;
use crate::error::MemoryError;
use crate::tools::{get_tool_definitions, ToolHandler};

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
                ..Default::default()
            },
            server_info: Implementation {
                name: "agent-memory".to_string(),
                title: Some("Agent Memory MCP Server".to_string()),
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

            let args = request
                .arguments
                .map(Value::Object)
                .unwrap_or(Value::Null);
            let result = handler
                .handle_tool(&request.name, args)
                .map_err(|e: MemoryError| McpError::internal_error(e.to_string(), None))?;

            Ok(CallToolResult {
                content: vec![Content::text(
                    serde_json::to_string_pretty(&result)
                        .unwrap_or_else(|_| result.to_string()),
                )],
                structured_content: None,
                is_error: Some(false),
                meta: None,
            })
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("agent_memory=info".parse()?))
        .with_writer(std::io::stderr)
        .init();

    // Determine database path
    let db_path = std::env::var("AGENT_MEMORY_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("agent-memory")
                .join("memories.db")
        });

    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Determine project ID (from env or current directory)
    let project_id = std::env::var("AGENT_MEMORY_PROJECT").unwrap_or_else(|_| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|s| s.to_string_lossy().to_string()))
            .unwrap_or_else(|| "default".to_string())
    });

    tracing::info!("Starting agent-memory MCP server");
    tracing::info!("Database: {}", db_path.display());
    tracing::info!("Project: {}", project_id);

    let server = MemoryServer::new(db_path, project_id);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
