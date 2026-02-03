use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod cache;
mod db;
mod decay;
mod embedding;
mod error;
mod export;
mod memory;
mod summarize;
mod tools;

use db::Database;
use embedding::EmbeddingService;
use error::MemoryError;
use memory::{Memory, MemoryType, RelationType, Relationship};
use summarize::{generate_summary, should_auto_summarize};

#[derive(Parser)]
#[command(name = "engram-cli")]
#[command(about = "CLI for Engram memory management", long_about = None)]
struct Cli {
    /// Database path (default: ~/.local/share/engram/memories.db)
    #[arg(short, long)]
    database: Option<PathBuf>,

    /// Project ID (default: current directory name)
    #[arg(short, long)]
    project: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search memories by semantic similarity
    Query {
        /// Search query
        query: String,
        /// Maximum results
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Minimum relevance score
        #[arg(short, long, default_value = "0.3")]
        min_relevance: f64,
        /// Filter by type(s)
        #[arg(short, long)]
        types: Vec<String>,
    },
    /// List all memories
    List {
        /// Filter by type
        #[arg(short, long)]
        r#type: Option<String>,
        /// Maximum results
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },
    /// Show a specific memory
    Show {
        /// Memory ID
        id: String,
    },
    /// Store a new memory
    Store {
        /// Memory content
        content: String,
        /// Memory type
        #[arg(short, long, default_value = "fact")]
        r#type: String,
        /// Tags (comma-separated)
        #[arg(short = 'g', long)]
        tags: Option<String>,
        /// Importance (0.0-1.0)
        #[arg(short, long, default_value = "0.5")]
        importance: f64,
        /// Summary
        #[arg(short, long)]
        summary: Option<String>,
    },
    /// Delete a memory
    Delete {
        /// Memory ID
        id: String,
    },
    /// Update a memory
    Update {
        /// Memory ID
        id: String,
        /// New content
        #[arg(short, long)]
        content: Option<String>,
        /// New importance
        #[arg(short, long)]
        importance: Option<f64>,
        /// New tags (comma-separated)
        #[arg(short = 'g', long)]
        tags: Option<String>,
        /// New summary
        #[arg(short, long)]
        summary: Option<String>,
    },
    /// Link two memories
    Link {
        /// Source memory ID
        source: String,
        /// Target memory ID
        target: String,
        /// Relation type
        #[arg(short, long, default_value = "relates_to")]
        relation: String,
        /// Strength (0.0-1.0)
        #[arg(short, long, default_value = "1.0")]
        strength: f64,
    },
    /// Export memories to JSON
    Export {
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Include embeddings
        #[arg(short, long)]
        embeddings: bool,
    },
    /// Import memories from JSON
    Import {
        /// Input file
        file: PathBuf,
        /// Import mode
        #[arg(short, long, default_value = "merge")]
        mode: String,
    },
    /// Show project statistics
    Stats,
    /// Run decay algorithm manually
    Decay,
    /// Prune low-relevance memories
    Prune {
        /// Minimum relevance to keep
        #[arg(short, long, default_value = "0.2")]
        threshold: f64,
        /// Actually delete (dry run by default)
        #[arg(long)]
        confirm: bool,
    },
}

fn get_db_path(cli_path: Option<PathBuf>) -> PathBuf {
    cli_path
        .or_else(|| std::env::var("ENGRAM_DB").ok().map(PathBuf::from))
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("engram")
                .join("memories.db")
        })
}

fn get_project_id(cli_project: Option<String>) -> String {
    cli_project
        .or_else(|| std::env::var("ENGRAM_PROJECT").ok())
        .unwrap_or_else(|| {
            std::env::current_dir()
                .ok()
                .and_then(|p| p.file_name().map(|s| s.to_string_lossy().to_string()))
                .unwrap_or_else(|| "default".to_string())
        })
}

/// Check if command needs embedding service (lazy initialization).
fn needs_embedding_service(cmd: &Commands) -> bool {
    matches!(
        cmd,
        Commands::Query { .. } | Commands::Store { .. } | Commands::Update { .. } | Commands::Import { .. }
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let db_path = get_db_path(cli.database);
    let project_id = get_project_id(cli.project);

    // Ensure database directory exists
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let db = Database::open(&db_path)?;
    db.get_or_create_project(&project_id, &project_id)?;

    // Initialize embedding service once, only if needed (saves ~500ms for commands that don't need it)
    let embedding_service = if needs_embedding_service(&cli.command) {
        Some(EmbeddingService::new()?)
    } else {
        None
    };

    match cli.command {
        Commands::Query {
            query,
            limit,
            min_relevance,
            types,
        } => {
            cmd_query(
                &db,
                &project_id,
                embedding_service.as_ref().unwrap(),
                &query,
                limit,
                min_relevance,
                &types,
            )?;
        }
        Commands::List { r#type, limit } => {
            cmd_list(&db, &project_id, r#type.as_deref(), limit)?;
        }
        Commands::Show { id } => {
            cmd_show(&db, &id)?;
        }
        Commands::Store {
            content,
            r#type,
            tags,
            importance,
            summary,
        } => {
            cmd_store(
                &db,
                &project_id,
                embedding_service.as_ref().unwrap(),
                &content,
                &r#type,
                tags.as_deref(),
                importance,
                summary,
            )?;
        }
        Commands::Delete { id } => {
            cmd_delete(&db, &id)?;
        }
        Commands::Update {
            id,
            content,
            importance,
            tags,
            summary,
        } => {
            cmd_update(
                &db,
                embedding_service.as_ref().unwrap(),
                &id,
                content,
                importance,
                tags,
                summary,
            )?;
        }
        Commands::Link {
            source,
            target,
            relation,
            strength,
        } => {
            cmd_link(&db, &source, &target, &relation, strength)?;
        }
        Commands::Export { output, embeddings } => {
            cmd_export(&db, &project_id, output, embeddings)?;
        }
        Commands::Import { file, mode } => {
            cmd_import(&db, &project_id, embedding_service.as_ref().unwrap(), &file, &mode)?;
        }
        Commands::Stats => {
            cmd_stats(&db, &project_id)?;
        }
        Commands::Decay => {
            cmd_decay(&db, &project_id)?;
        }
        Commands::Prune { threshold, confirm } => {
            cmd_prune(&db, &project_id, threshold, confirm)?;
        }
    }

    Ok(())
}

fn cmd_query(
    db: &Database,
    project_id: &str,
    embedding_service: &EmbeddingService,
    query: &str,
    limit: usize,
    min_relevance: f64,
    types: &[String],
) -> Result<(), MemoryError> {
    let query_embedding = embedding_service.embed(query)?;

    let embeddings = db.get_all_embeddings_for_project(project_id)?;

    let mut scored: Vec<(String, f32)> = embeddings
        .iter()
        .map(|(id, vec)| {
            (
                id.clone(),
                embedding::cosine_similarity(&query_embedding, vec),
            )
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let type_filters: Vec<MemoryType> = types.iter().filter_map(|t| t.parse().ok()).collect();

    let mut count = 0;
    for (id, similarity) in scored {
        if count >= limit {
            break;
        }
        if let Some(memory) = db.get_memory(&id)? {
            if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
                continue;
            }
            let score = (similarity as f64) * memory.relevance_score;
            if score < min_relevance {
                continue;
            }

            println!("─────────────────────────────────────────");
            println!("ID: {}", memory.id);
            println!(
                "Type: {:?} | Score: {:.3} | Importance: {:.2}",
                memory.memory_type, score, memory.importance
            );
            if let Some(summary) = &memory.summary {
                println!("Summary: {}", summary);
            }
            println!("Content: {}", memory.content);
            if !memory.tags.is_empty() {
                println!("Tags: {}", memory.tags.join(", "));
            }
            count += 1;
        }
    }

    if count == 0 {
        println!("No matching memories found.");
    } else {
        println!("─────────────────────────────────────────");
        println!("Found {} memories", count);
    }

    Ok(())
}

fn cmd_list(
    db: &Database,
    project_id: &str,
    type_filter: Option<&str>,
    limit: usize,
) -> Result<(), MemoryError> {
    let type_filters: Option<Vec<MemoryType>> =
        type_filter.and_then(|t| t.parse().ok()).map(|t| vec![t]);

    let memories = db.query_memories(project_id, type_filters.as_deref(), None, None, limit)?;

    if memories.is_empty() {
        println!("No memories found.");
        return Ok(());
    }

    for memory in &memories {
        let summary = memory
            .summary
            .as_deref()
            .unwrap_or_else(|| &memory.content[..memory.content.len().min(60)]);
        println!(
            "{} [{:?}] {:.2} - {}",
            memory.id, memory.memory_type, memory.relevance_score, summary
        );
    }
    println!("\nTotal: {} memories", memories.len());

    Ok(())
}

fn cmd_show(db: &Database, id: &str) -> Result<(), MemoryError> {
    let memory = db
        .get_memory(id)?
        .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

    println!("ID: {}", memory.id);
    println!("Project: {}", memory.project_id);
    println!("Type: {:?}", memory.memory_type);
    println!("Importance: {:.2}", memory.importance);
    println!("Relevance: {:.2}", memory.relevance_score);
    println!("Access count: {}", memory.access_count);
    if let Some(summary) = &memory.summary {
        println!("Summary: {}", summary);
    }
    println!(
        "Tags: {}",
        if memory.tags.is_empty() {
            "(none)".to_string()
        } else {
            memory.tags.join(", ")
        }
    );
    println!("Created: {}", format_timestamp(memory.created_at));
    println!("Updated: {}", format_timestamp(memory.updated_at));
    println!(
        "Last accessed: {}",
        format_timestamp(memory.last_accessed_at)
    );
    println!("\nContent:\n{}", memory.content);

    // Show relationships
    let outgoing = db.get_relationships_from(id)?;
    let incoming = db.get_relationships_to(id)?;

    if !outgoing.is_empty() || !incoming.is_empty() {
        println!("\nRelationships:");
        for rel in outgoing {
            println!("  -> {} ({})", rel.target_id, rel.relation_type.as_str());
        }
        for rel in incoming {
            println!("  <- {} ({})", rel.source_id, rel.relation_type.as_str());
        }
    }

    // Record access
    db.record_access(id)?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_store(
    db: &Database,
    project_id: &str,
    embedding_service: &EmbeddingService,
    content: &str,
    type_str: &str,
    tags: Option<&str>,
    importance: f64,
    summary: Option<String>,
) -> Result<(), MemoryError> {
    let memory_type: MemoryType = type_str
        .parse()
        .map_err(|_| MemoryError::InvalidType(type_str.to_string()))?;

    let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
    let now = chrono::Utc::now().timestamp();

    let summary = if should_auto_summarize(content, summary.as_deref()) {
        Some(generate_summary(content))
    } else {
        summary
    };

    let tags_vec: Vec<String> = tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let memory = Memory {
        id: id.clone(),
        project_id: project_id.to_string(),
        memory_type,
        content: content.to_string(),
        summary,
        tags: tags_vec,
        importance: importance.clamp(0.0, 1.0),
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
    };

    db.store_memory(&memory)?;

    // Generate and store embedding
    let embedding = embedding_service.embed_memory(memory_type, content)?;
    db.store_embedding(&id, &embedding, embedding_service.model_version())?;

    println!("Memory stored: {}", id);

    Ok(())
}

fn cmd_delete(db: &Database, id: &str) -> Result<(), MemoryError> {
    if db.delete_memory(id)? {
        println!("Deleted memory: {}", id);
    } else {
        println!("Memory not found: {}", id);
    }
    Ok(())
}

fn cmd_update(
    db: &Database,
    embedding_service: &EmbeddingService,
    id: &str,
    content: Option<String>,
    importance: Option<f64>,
    tags: Option<String>,
    summary: Option<String>,
) -> Result<(), MemoryError> {
    let mut memory = db
        .get_memory(id)?
        .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

    memory.updated_at = chrono::Utc::now().timestamp();

    if let Some(new_content) = content {
        memory.content = new_content.clone();
        // Re-embed
        let embedding = embedding_service.embed_memory(memory.memory_type, &new_content)?;
        db.store_embedding(id, &embedding, embedding_service.model_version())?;

        // Auto-generate summary if needed
        if summary.is_none() && should_auto_summarize(&new_content, memory.summary.as_deref()) {
            memory.summary = Some(generate_summary(&new_content));
        }
    }

    if let Some(imp) = importance {
        memory.importance = imp.clamp(0.0, 1.0);
    }

    if let Some(tags_str) = tags {
        memory.tags = tags_str.split(',').map(|s| s.trim().to_string()).collect();
    }

    if let Some(sum) = summary {
        memory.summary = Some(sum);
    }

    db.update_memory(&memory)?;
    println!("Updated memory: {}", id);

    Ok(())
}

fn cmd_link(
    db: &Database,
    source: &str,
    target: &str,
    relation: &str,
    strength: f64,
) -> Result<(), MemoryError> {
    let relation_type: RelationType = relation
        .parse()
        .map_err(|_| MemoryError::InvalidRelation(relation.to_string()))?;

    // Verify both exist
    db.get_memory(source)?
        .ok_or_else(|| MemoryError::NotFound(source.to_string()))?;
    db.get_memory(target)?
        .ok_or_else(|| MemoryError::NotFound(target.to_string()))?;

    let rel = Relationship {
        id: format!("rel_{}", uuid::Uuid::new_v4().simple()),
        source_id: source.to_string(),
        target_id: target.to_string(),
        relation_type,
        strength: strength.clamp(0.0, 1.0),
        created_at: chrono::Utc::now().timestamp(),
    };

    db.create_relationship(&rel)?;
    println!(
        "Created relationship: {} -> {} ({})",
        source, target, relation
    );

    Ok(())
}

fn cmd_export(
    db: &Database,
    project_id: &str,
    output: Option<PathBuf>,
    include_embeddings: bool,
) -> Result<(), MemoryError> {
    let memories = db.get_all_memories_for_project(project_id)?;
    let relationships = db.get_all_relationships_for_project(project_id)?;

    let embeddings = if include_embeddings {
        Some(db.get_all_embeddings_for_project(project_id)?)
    } else {
        None
    };

    let export_data = export::create_export(project_id, memories, relationships, embeddings);

    let json = serde_json::to_string_pretty(&export_data)?;

    if let Some(path) = output {
        std::fs::write(&path, &json)?;
        println!("Exported to: {}", path.display());
    } else {
        println!("{}", json);
    }

    Ok(())
}

fn cmd_import(
    db: &Database,
    project_id: &str,
    embedding_service: &EmbeddingService,
    file: &PathBuf,
    mode: &str,
) -> Result<(), MemoryError> {
    let json = std::fs::read_to_string(file)?;
    let export_data: export::ExportData = serde_json::from_str(&json)?;

    export::validate_import(&export_data).map_err(MemoryError::Embedding)?;

    let import_mode: export::ImportMode = mode.parse().unwrap_or(export::ImportMode::Merge);

    if import_mode == export::ImportMode::Replace {
        db.delete_project_data(project_id)?;
        println!("Cleared existing data.");
    }

    let now = chrono::Utc::now().timestamp();
    let mut imported = 0;
    let mut skipped = 0;

    for exported in export_data.memories {
        let mut memory = exported.memory;
        memory.project_id = project_id.to_string();
        memory.updated_at = now;

        if import_mode == export::ImportMode::Merge && db.get_memory(&memory.id)?.is_some() {
            skipped += 1;
            continue;
        }

        db.store_memory(&memory)?;

        // Handle embedding
        if let Some(encoded) = exported.embedding {
            if let Ok(vector) = export::decode_embedding(&encoded) {
                db.store_embedding(&memory.id, &vector, embedding_service.model_version())?;
            }
        } else {
            let embedding = embedding_service.embed_memory(memory.memory_type, &memory.content)?;
            db.store_embedding(&memory.id, &embedding, embedding_service.model_version())?;
        }

        imported += 1;
    }

    // Import relationships
    let mut rel_imported = 0;
    for rel in export_data.relationships {
        let source_exists = db.get_memory(&rel.source_id)?.is_some();
        let target_exists = db.get_memory(&rel.target_id)?.is_some();
        if source_exists && target_exists {
            db.create_relationship(&rel)?;
            rel_imported += 1;
        }
    }

    println!(
        "Imported {} memories, {} relationships ({} skipped)",
        imported, rel_imported, skipped
    );

    Ok(())
}

fn cmd_stats(db: &Database, project_id: &str) -> Result<(), MemoryError> {
    let stats = db.get_project_stats(project_id)?;

    println!("Project: {}", project_id);
    println!("Memories: {}", stats.memory_count);
    println!("Relationships: {}", stats.relationship_count);
    println!("Avg relevance: {:.3}", stats.avg_relevance);

    Ok(())
}

fn cmd_decay(db: &Database, project_id: &str) -> Result<(), MemoryError> {
    let project = db
        .get_project(project_id)?
        .ok_or_else(|| MemoryError::NotFound(project_id.to_string()))?;

    let updated = db.update_relevance_scores(project_id, project.decay_rate)?;
    println!("Updated relevance scores for {} memories", updated);

    Ok(())
}

fn cmd_prune(
    db: &Database,
    project_id: &str,
    threshold: f64,
    confirm: bool,
) -> Result<(), MemoryError> {
    let memories = db.get_all_memories_for_project(project_id)?;
    let low_relevance: Vec<&Memory> = memories
        .iter()
        .filter(|m| m.relevance_score < threshold)
        .collect();

    if low_relevance.is_empty() {
        println!("No memories below threshold {:.2}", threshold);
        return Ok(());
    }

    println!(
        "Found {} memories below threshold {:.2}:",
        low_relevance.len(),
        threshold
    );
    for memory in &low_relevance {
        let summary = memory
            .summary
            .as_deref()
            .unwrap_or_else(|| &memory.content[..memory.content.len().min(50)]);
        println!(
            "  {} ({:.3}): {}",
            memory.id, memory.relevance_score, summary
        );
    }

    if confirm {
        let ids: Vec<String> = low_relevance.iter().map(|m| m.id.clone()).collect();
        let deleted = db.delete_memories_batch(&ids)?;
        println!("Deleted {} memories", deleted);
    } else {
        println!("\nRun with --confirm to delete these memories.");
    }

    Ok(())
}

fn format_timestamp(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
