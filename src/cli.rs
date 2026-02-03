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
        /// Branch mode: "current" (global + current branch), "all", "global", or specific branch name
        #[arg(short, long, default_value = "current")]
        branch_mode: String,
    },
    /// List all memories
    List {
        /// Filter by type
        #[arg(short, long)]
        r#type: Option<String>,
        /// Maximum results
        #[arg(short, long, default_value = "50")]
        limit: usize,
        /// Filter by branch (default: show all)
        #[arg(short, long)]
        branch: Option<String>,
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
        /// Branch: omit for global, "auto" for current branch, or explicit branch name
        #[arg(short, long)]
        branch: Option<String>,
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
    /// Promote a branch-local memory to global
    Promote {
        /// Memory ID to promote
        id: String,
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

/// Determine project ID from git root path or current directory.
fn get_project_id(cli_project: Option<String>) -> String {
    // 1. Explicit override via CLI or env var
    if let Some(project) = cli_project {
        return project;
    }
    if let Ok(project) = std::env::var("ENGRAM_PROJECT") {
        return project;
    }

    // 2. Try git root path
    if let Some(git_root) = find_git_root() {
        return git_root.to_string_lossy().to_string();
    }

    // 3. Fall back to current directory path
    if let Ok(cwd) = std::env::current_dir() {
        return cwd.to_string_lossy().to_string();
    }

    // 4. Ultimate fallback
    "default".to_string()
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

/// Check if command needs embedding service (lazy initialization).
fn needs_embedding_service(cmd: &Commands) -> bool {
    matches!(
        cmd,
        Commands::Query { .. }
            | Commands::Store { .. }
            | Commands::Update { .. }
            | Commands::Import { .. }
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

    // Detect current branch once for commands that need it
    let current_branch = get_current_branch();

    match cli.command {
        Commands::Query {
            query,
            limit,
            min_relevance,
            types,
            branch_mode,
        } => {
            cmd_query(
                &db,
                &project_id,
                embedding_service.as_ref().unwrap(),
                &query,
                limit,
                min_relevance,
                &types,
                &branch_mode,
                current_branch.as_deref(),
            )?;
        }
        Commands::List {
            r#type,
            limit,
            branch,
        } => {
            cmd_list(
                &db,
                &project_id,
                r#type.as_deref(),
                limit,
                branch.as_deref(),
            )?;
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
            branch,
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
                branch.as_deref(),
                current_branch.as_deref(),
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
            cmd_import(
                &db,
                &project_id,
                embedding_service.as_ref().unwrap(),
                &file,
                &mode,
            )?;
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
        Commands::Promote { id } => {
            cmd_promote(&db, &id)?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_query(
    db: &Database,
    project_id: &str,
    embedding_service: &EmbeddingService,
    query: &str,
    limit: usize,
    min_relevance: f64,
    types: &[String],
    branch_mode: &str,
    current_branch: Option<&str>,
) -> Result<(), MemoryError> {
    use std::collections::{HashMap, HashSet};

    // Resolve branch filter based on mode
    let branch_filter = match branch_mode {
        "all" => None,                     // All memories
        "global" => Some(None),            // Global only
        "current" => Some(current_branch), // Global + current branch
        specific => Some(Some(specific)),  // Specific branch
    };

    // Hybrid search: combine semantic (70%) and keyword (30%) scores
    const SEMANTIC_WEIGHT: f64 = 0.7;
    const KEYWORD_WEIGHT: f64 = 0.3;

    // Run semantic search
    let query_embedding = embedding_service.embed(query)?;
    let embeddings = db.get_all_embeddings_for_project(project_id)?;

    let semantic_scores: HashMap<String, f32> = embeddings
        .iter()
        .map(|(id, vec)| {
            (
                id.clone(),
                embedding::cosine_similarity(&query_embedding, vec),
            )
        })
        .collect();

    // Run keyword search (FTS5)
    let keyword_results = db.keyword_search(project_id, query, limit * 5)?;

    // Normalize keyword scores
    let max_keyword_score = keyword_results
        .iter()
        .map(|(_, s)| *s)
        .fold(0.0_f64, f64::max);

    let keyword_scores: HashMap<String, f64> = if max_keyword_score > 0.0 {
        keyword_results
            .into_iter()
            .map(|(id, score)| (id, score / max_keyword_score))
            .collect()
    } else {
        HashMap::new()
    };

    // Collect all candidate IDs
    let mut candidate_ids: HashSet<String> = semantic_scores.keys().cloned().collect();
    candidate_ids.extend(keyword_scores.keys().cloned());

    let type_filters: Vec<MemoryType> = types.iter().filter_map(|t| t.parse().ok()).collect();

    // Calculate hybrid scores
    let mut scored_results: Vec<(String, f64, f64, f64)> = Vec::new(); // (id, combined, semantic, keyword)

    for id in candidate_ids {
        let semantic_score = *semantic_scores.get(&id).unwrap_or(&0.0) as f64;
        let keyword_score = *keyword_scores.get(&id).unwrap_or(&0.0);

        // Hybrid score
        let hybrid_score = SEMANTIC_WEIGHT * semantic_score + KEYWORD_WEIGHT * keyword_score;

        if let Some(memory) = db.get_memory(&id)? {
            if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
                continue;
            }

            // Apply branch filter
            match branch_filter {
                None => {} // All branches - no filter
                Some(None) => {
                    // Global only
                    if memory.branch.is_some() {
                        continue;
                    }
                }
                Some(Some(branch)) => {
                    // Global + specific branch
                    if let Some(ref mem_branch) = memory.branch
                        && mem_branch != branch
                    {
                        continue;
                    }
                    // Global (branch = None) is always included
                }
            }

            let final_score = hybrid_score * memory.relevance_score;
            if final_score >= min_relevance {
                scored_results.push((id, final_score, semantic_score, keyword_score));
            }
        }
    }

    // Sort by combined score descending
    scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut count = 0;
    for (id, score, semantic_score, keyword_score) in scored_results.into_iter().take(limit) {
        if let Some(memory) = db.get_memory(&id)? {
            println!("─────────────────────────────────────────");
            println!("ID: {}", memory.id);
            let branch_str = memory
                .branch
                .as_ref()
                .map(|b| format!(" | Branch: {}", b))
                .unwrap_or_default();
            println!(
                "Type: {:?} | Score: {:.3} | Importance: {:.2}{}",
                memory.memory_type, score, memory.importance, branch_str
            );
            println!(
                "Semantic: {:.3} | Keyword: {:.3}",
                semantic_score, keyword_score
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
    branch_filter: Option<&str>,
) -> Result<(), MemoryError> {
    let type_filters: Option<Vec<MemoryType>> =
        type_filter.and_then(|t| t.parse().ok()).map(|t| vec![t]);

    // Convert branch filter for query
    let branch_query = branch_filter.map(Some);

    let memories = db.query_memories_with_branch(
        project_id,
        type_filters.as_deref(),
        None,
        None,
        limit,
        branch_query,
    )?;

    if memories.is_empty() {
        println!("No memories found.");
        return Ok(());
    }

    for memory in &memories {
        let summary = memory
            .summary
            .as_deref()
            .unwrap_or_else(|| &memory.content[..memory.content.len().min(60)]);
        let branch_info = memory
            .branch
            .as_ref()
            .map(|b| format!(" [{}]", b))
            .unwrap_or_default();
        println!(
            "{} [{:?}]{} {:.2} - {}",
            memory.id, memory.memory_type, branch_info, memory.relevance_score, summary
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
    println!("Branch: {}", memory.branch.as_deref().unwrap_or("(global)"));
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
    branch_arg: Option<&str>,
    current_branch: Option<&str>,
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

    // Resolve branch: omit for global, "auto" for current branch, else explicit
    let branch = match branch_arg {
        None | Some("") => None, // Global
        Some("auto") => current_branch.map(String::from),
        Some(explicit) => Some(explicit.to_string()),
    };

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
        branch: branch.clone(),
    };

    db.store_memory(&memory)?;

    // Generate and store embedding
    let embedding = embedding_service.embed_memory(memory_type, content)?;
    db.store_embedding(&id, &embedding, embedding_service.model_version())?;

    if let Some(ref b) = branch {
        println!("Memory stored: {} (branch: {})", id, b);
    } else {
        println!("Memory stored: {} (global)", id);
    }

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

fn cmd_promote(db: &Database, id: &str) -> Result<(), MemoryError> {
    // Get the memory first to verify it exists and get its current state
    let memory = db
        .get_memory(id)?
        .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

    // Check if already global
    if memory.branch.is_none() {
        println!("Memory {} is already global", id);
        return Ok(());
    }

    let was_branch = memory.branch.clone();

    // Promote to global
    let promoted = db.promote_memory(id)?;

    if promoted {
        println!(
            "Promoted memory {} from branch '{}' to global",
            id,
            was_branch.as_deref().unwrap_or("?")
        );
    } else {
        println!("Failed to promote memory {}", id);
    }

    Ok(())
}

fn format_timestamp(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
