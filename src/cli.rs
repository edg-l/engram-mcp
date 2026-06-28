use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod adr_export;
mod cache;
mod db;
mod decay;
mod embedding;
mod error;
mod export;
mod format;
mod hooks;
mod memory;
mod summarize;
mod tools;

use db::Database;
use embedding::EmbeddingService;
use error::MemoryError;
use hooks::HookEvent;
use memory::{
    AdrSections, AdrStatus, HandoffSections, Memory, MemoryType, RelationType, Relationship,
};
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
        /// Pin this memory so it never decays or gets pruned
        #[arg(long)]
        pinned: bool,
        /// Make this memory visible across all projects (forces branch=null)
        #[arg(long)]
        global: bool,
        /// External artifact references (file paths, URLs, ticket IDs). Repeatable.
        #[arg(long = "artifact", value_name = "PATH")]
        artifacts: Vec<String>,
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
        /// Replace external artifact list. Repeatable. Pass once with empty string to clear.
        #[arg(long = "artifact", value_name = "PATH")]
        artifacts: Vec<String>,
        /// Clear all external artifacts (sets list to empty).
        #[arg(long)]
        clear_artifacts: bool,
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
    /// Wipe all memories for the current project
    Wipe {
        /// Skip confirmation prompt
        #[arg(long)]
        confirm: bool,
    },
    /// Find and merge duplicate memories
    Dedup {
        /// Similarity threshold (default: 0.90)
        #[arg(short, long, default_value = "0.90")]
        threshold: f32,
        /// Actually merge (default: dry run)
        #[arg(long)]
        confirm: bool,
    },
    /// Pin a memory so it never decays or gets pruned
    Pin {
        /// Memory ID to pin
        id: String,
    },
    /// Unpin a memory to allow decay and pruning
    Unpin {
        /// Memory ID to unpin
        id: String,
    },
    /// Load relevant memories for a context (like memory_context MCP tool)
    Context {
        /// Context description (e.g. "working on auth refactor")
        context: String,
        /// Maximum memories to return
        #[arg(short, long, default_value = "5")]
        limit: usize,
        /// Minimum similarity score
        #[arg(short, long, default_value = "0.3")]
        min_score: f64,
        /// Filter by type(s)
        #[arg(short, long)]
        types: Vec<String>,
    },
    /// Show memory usage patterns and effectiveness metrics
    Insights,
    /// Show actionable memory health report
    Health,
    /// Session handoff management
    Handoff {
        #[command(subcommand)]
        cmd: HandoffCmd,
    },
    /// Architecture Decision Record management
    Adr {
        #[command(subcommand)]
        cmd: AdrCmd,
    },
    /// Process a Claude Code lifecycle hook event
    HookEvent {
        /// Hook event name (e.g. SessionStart, UserPromptSubmit, PostToolUse)
        event: String,
        /// JSON payload (reads from stdin if omitted)
        #[arg(long)]
        payload: Option<String>,
        /// Print outcome to stdout instead of persisting
        #[arg(long)]
        dry_run: bool,
    },
    /// Manage engram-cli entries in ~/.claude/settings.json
    Hooks {
        #[command(subcommand)]
        cmd: HooksCmd,
    },
}

/// Subcommands for `engram-cli handoff`.
#[derive(Subcommand)]
enum HandoffCmd {
    /// Create a session handoff (interactive or from a markdown file)
    Create {
        /// High-level session summary
        #[arg(long)]
        summary: Option<String>,
        /// Key decisions made (can be repeated or comma-separated)
        #[arg(long, value_delimiter = ',')]
        decisions: Vec<String>,
        /// Within-session work the next agent should pick up immediately. Concrete, ready-to-execute items. (can be repeated or comma-separated)
        #[arg(long, value_delimiter = ',')]
        todos: Vec<String>,
        /// Things preventing forward motion right now (missing access, failing dependency, unanswered question). (can be repeated or comma-separated)
        #[arg(long, value_delimiter = ',')]
        blockers: Vec<String>,
        /// Architecture/context needed by the next session
        #[arg(long)]
        mental_model: Option<String>,
        /// Post-session follow-ups beyond the current thread. Future-facing, not for immediate pickup. (can be repeated or comma-separated)
        #[arg(long, value_delimiter = ',')]
        next_steps: Vec<String>,
        /// Freeform notes (optional)
        #[arg(long)]
        notes: Option<String>,
        /// Git branch to scope the handoff to (defaults to current branch)
        #[arg(long)]
        branch: Option<String>,
        /// ID of the handoff this session continues from
        #[arg(long)]
        continues_from: Option<String>,
        /// Importance score (0.0-1.0, default 0.85)
        #[arg(long, default_value = "0.85")]
        importance: f64,
        /// Do NOT pin this handoff (it will be pinned by default)
        #[arg(long)]
        no_pin: bool,
        /// Do NOT auto-link to related memories
        #[arg(long)]
        no_auto_link: bool,
        /// Read sections from a markdown file instead of interactive prompts
        #[arg(long)]
        from_file: Option<std::path::PathBuf>,
    },
    /// Resume a session by loading context from recent handoffs
    Resume {
        /// Branch to load handoffs from (defaults to current branch)
        #[arg(long)]
        branch: Option<String>,
        /// Query string for section scoring (defaults to latest handoff summary)
        #[arg(long)]
        query: Option<String>,
        /// Maximum number of top sections to show (default 5)
        #[arg(long, default_value = "5")]
        max: usize,
        /// Include handoffs from all branches
        #[arg(long)]
        include_off_branch: bool,
        /// Truncate each returned section to this many characters (0 = no cap)
        #[arg(long)]
        max_chars_per_section: Option<usize>,
    },
    /// Search handoff sections by content
    Search {
        /// Search query
        query: String,
        /// Limit results to this branch (omit for all branches)
        #[arg(long)]
        branch: Option<String>,
        /// Only show these sections (comma-separated, e.g. blockers,todos)
        #[arg(long, value_delimiter = ',')]
        section: Vec<String>,
        /// Maximum results (default 10)
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Show a handoff memory by ID
    Show {
        /// Memory ID of the handoff
        id: String,
    },
}

/// Subcommands for `engram-cli hooks`.
#[derive(Subcommand)]
enum HooksCmd {
    /// Install engram-cli hook entries into ~/.claude/settings.json
    Install,
    /// Remove engram-cli hook entries from ~/.claude/settings.json
    Uninstall,
    /// Show which events are managed by engram-cli
    Status,
}

/// Subcommands for `engram-cli adr`.
#[derive(Subcommand)]
enum AdrCmd {
    /// Create a new ADR (project-global, pinned by default)
    Create {
        /// Short, imperative-mood title
        #[arg(long)]
        title: Option<String>,
        /// Forces and constraints that drove this decision
        #[arg(long)]
        context: Option<String>,
        /// The decision made
        #[arg(long)]
        decision: Option<String>,
        /// Positive and negative consequences
        #[arg(long)]
        consequences: Option<String>,
        /// Initial lifecycle status
        #[arg(long, default_value = "proposed")]
        status: String,
        /// ADR number this decision supersedes
        #[arg(long)]
        supersedes: Option<u32>,
        /// Importance score (0.0-1.0)
        #[arg(long, default_value_t = 0.85)]
        importance: f64,
        /// Do NOT pin this ADR (it is pinned by default)
        #[arg(long)]
        no_pin: bool,
        /// Read sections from a Markdown file instead of interactive prompts
        #[arg(long)]
        from_file: Option<PathBuf>,
    },
    /// Update the lifecycle status of an ADR
    UpdateStatus {
        /// ADR number to update
        number: u32,
        /// New status (proposed, accepted, deprecated, rejected)
        status: String,
    },
    /// List all ADRs for the current project
    List {
        /// Filter by status
        #[arg(long)]
        status: Option<String>,
    },
    /// Show full details of an ADR by number
    Show {
        /// ADR number
        number: u32,
    },
    /// Export ADRs to Markdown files
    Export {
        /// Export a single ADR by number; omit to export all
        number: Option<u32>,
        /// Output directory (default: docs/adr)
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Actually write files (default: dry run)
        #[arg(long)]
        write: bool,
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
    match cmd {
        Commands::Query { .. }
        | Commands::Store { .. }
        | Commands::Update { .. }
        | Commands::Import { .. }
        | Commands::Dedup { .. }
        | Commands::Context { .. } => true,
        Commands::Handoff { cmd: handoff_cmd } => matches!(
            handoff_cmd,
            HandoffCmd::Create { .. } | HandoffCmd::Resume { .. } | HandoffCmd::Search { .. }
        ),
        Commands::Adr { cmd: adr_cmd } => matches!(adr_cmd, AdrCmd::Create { .. }),
        Commands::HookEvent { .. } => true,
        Commands::Hooks { .. } => false,
        _ => false,
    }
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
            pinned,
            global,
            artifacts,
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
                pinned,
                global,
                if artifacts.is_empty() {
                    None
                } else {
                    Some(artifacts)
                },
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
            artifacts,
            clear_artifacts,
        } => {
            // external_artifacts semantics for CLI:
            //   --clear-artifacts       -> Some([]) (clear)
            //   --artifact PATH ...     -> Some([PATH, ...]) (replace)
            //   neither flag            -> None (preserve)
            let external_artifacts = if clear_artifacts {
                Some(Vec::new())
            } else if !artifacts.is_empty() {
                Some(artifacts)
            } else {
                None
            };
            cmd_update(
                &db,
                embedding_service.as_ref().unwrap(),
                &id,
                content,
                importance,
                tags,
                summary,
                external_artifacts,
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
        Commands::Dedup { threshold, confirm } => {
            cmd_dedup(
                &db,
                &project_id,
                embedding_service.as_ref().unwrap(),
                threshold,
                confirm,
            )?;
        }
        Commands::Pin { id } => {
            if db.set_pinned(&id, true)? {
                println!("Pinned memory: {}", id);
            } else {
                println!("Memory not found: {}", id);
            }
        }
        Commands::Unpin { id } => {
            if db.set_pinned(&id, false)? {
                println!("Unpinned memory: {}", id);
            } else {
                println!("Memory not found: {}", id);
            }
        }
        Commands::Context {
            context,
            limit,
            min_score,
            types,
        } => {
            cmd_context(
                &db,
                &project_id,
                embedding_service.as_ref().unwrap(),
                &context,
                limit,
                min_score,
                &types,
                current_branch.as_deref(),
            )?;
        }
        Commands::Wipe { confirm } => {
            if !confirm {
                let stats = db.get_project_stats(&project_id)?;
                println!(
                    "This will delete all {} memories and {} relationships for project '{}'.",
                    stats.memory_count, stats.relationship_count, project_id
                );
                println!("Run with --confirm to proceed.");
            } else {
                let deleted = db.delete_project_data(&project_id)?;
                // Also clean up clusters
                db.delete_empty_clusters(&project_id)?;
                println!("Wiped {} memories from project '{}'.", deleted, project_id);
            }
        }
        Commands::Insights => {
            cmd_insights(&db, &project_id)?;
        }
        Commands::Health => {
            cmd_health(&db, &project_id)?;
        }
        Commands::Handoff { cmd: handoff_cmd } => {
            cmd_handoff(
                &db,
                &project_id,
                embedding_service.as_ref(),
                current_branch.as_deref(),
                handoff_cmd,
            )?;
        }
        Commands::Adr { cmd: adr_cmd } => {
            cmd_adr(&db, &project_id, embedding_service.as_ref(), adr_cmd)?;
        }
        Commands::HookEvent {
            event,
            payload,
            dry_run,
        } => {
            cmd_hook_event(
                event,
                payload,
                dry_run,
                &db,
                embedding_service.as_ref(),
                &project_id,
            );
        }
        Commands::Hooks { cmd } => {
            cmd_hooks(cmd, &db, &project_id);
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

    // Run semantic search (include global memories from other projects)
    let query_embedding = embedding_service.embed(query)?;
    let embeddings = db.get_all_embeddings_for_project_and_global(project_id)?;

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
                None => {}                                         // All branches - no filter
                Some(None) if memory.branch.is_some() => continue, // Global only
                Some(None) => {}
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
    pinned: bool,
    global: bool,
    external_artifacts: Option<Vec<String>>,
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
    // If global=true, force branch to None
    let branch = if global {
        None
    } else {
        match branch_arg {
            None | Some("") => None, // Global
            Some("auto") => current_branch.map(String::from),
            Some(explicit) => Some(explicit.to_string()),
        }
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
        merged_from: None,
        external_artifacts,
        pinned,
        global,
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

#[allow(clippy::too_many_arguments)]
fn cmd_update(
    db: &Database,
    embedding_service: &EmbeddingService,
    id: &str,
    content: Option<String>,
    importance: Option<f64>,
    tags: Option<String>,
    summary: Option<String>,
    external_artifacts: Option<Vec<String>>,
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

    // external_artifacts: None = preserve, Some([]) = clear, Some([...]) = replace
    if let Some(artifacts) = external_artifacts {
        if artifacts.is_empty() {
            memory.external_artifacts = None;
        } else {
            memory.external_artifacts = Some(artifacts);
        }
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

    // Collect handoff sidecar data for Handoff memories.
    let mut handoff_sidecars: std::collections::HashMap<String, export::HandoffSidecar> =
        std::collections::HashMap::new();
    // Collect ADR sidecar data for ADR memories.
    let mut adr_sidecars: export::AdrSidecarMap = std::collections::HashMap::new();
    for memory in &memories {
        if memory.memory_type == MemoryType::Handoff
            && let Some((sections, section_vecs)) = db.get_handoff_sections(&memory.id)?
        {
            let key_strings: Vec<String> = section_vecs.iter().map(|(k, _)| k.clone()).collect();
            let keys: Vec<&str> = key_strings.iter().map(|s| s.as_str()).collect();
            let vecs: Vec<Vec<f32>> = section_vecs.into_iter().map(|(_, v)| v).collect();
            let (keys_str, bytes) = db::encode_section_embeddings(&keys, &vecs);
            handoff_sidecars.insert(
                memory.id.clone(),
                export::HandoffSidecar {
                    sections,
                    keys: keys_str,
                    bytes,
                },
            );
        }
        if memory.memory_type == MemoryType::Adr
            && let Some((num, status, sections)) = db.get_adr_sections(&memory.id)?
        {
            adr_sidecars.insert(memory.id.clone(), (num, status, sections));
        }
    }

    let export_data = export::create_export(
        project_id,
        memories,
        relationships,
        embeddings,
        handoff_sidecars,
        &adr_sidecars,
        None,
    );

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
        let encoded_embedding = exported.embedding;
        let sections = exported.sections;
        let section_embedding_keys = exported.section_embedding_keys;
        let encoded_section_embeddings = exported.section_embeddings;
        let adr_number = exported.adr_number;
        let adr_status_str = exported.adr_status;
        let adr_sections_data = exported.adr_sections;
        let mem_created_at = memory.created_at;
        let mem_updated_at = memory.updated_at;
        memory.project_id = project_id.to_string();
        memory.updated_at = now;

        if import_mode == export::ImportMode::Merge && db.get_memory(&memory.id)?.is_some() {
            skipped += 1;
            continue;
        }

        // For ADR memories with a known number, pre-check the number BEFORE storing
        // the memory row.  If the number is already taken, skip the entire memory
        // (memory row + embedding + sidecar) to keep them consistent.
        if memory.memory_type == MemoryType::Adr
            && let Some(num) = adr_number
            && db.get_adr_by_number(project_id, num)?.is_some()
        {
            skipped += 1;
            eprintln!(
                "Warning: skipping imported ADR {} — number {} already exists in project",
                memory.id, num
            );
            continue;
        }

        db.store_memory(&memory)?;

        // Handle embedding
        if let Some(encoded) = encoded_embedding {
            if let Ok(vector) = export::decode_embedding(&encoded) {
                db.store_embedding(&memory.id, &vector, embedding_service.model_version())?;
            }
        } else {
            let embedding = embedding_service.embed_memory(memory.memory_type, &memory.content)?;
            db.store_embedding(&memory.id, &embedding, embedding_service.model_version())?;
        }

        // Import handoff sidecar if present.
        // Old exports without sidecar fields skip this step silently.
        if memory.memory_type == MemoryType::Handoff {
            match (sections, section_embedding_keys, encoded_section_embeddings) {
                (Some(sections_data), Some(keys), Some(encoded_bytes)) => {
                    match export::decode_section_embedding_bytes(&encoded_bytes) {
                        Ok(bytes) => {
                            let key_count = if keys.is_empty() {
                                0
                            } else {
                                keys.split(',').count()
                            };
                            if bytes.len() == key_count * 256 * 4 {
                                if let Err(e) = db.insert_handoff_sections(
                                    &memory.id,
                                    &sections_data,
                                    &keys,
                                    &bytes,
                                ) {
                                    eprintln!(
                                        "Warning: failed to import handoff sidecar for {}: {}",
                                        memory.id, e
                                    );
                                }
                            } else {
                                eprintln!(
                                    "Warning: skipping handoff sidecar for {} — byte length mismatch",
                                    memory.id
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: skipping handoff sidecar for {} — decode error: {}",
                                memory.id, e
                            );
                        }
                    }
                }
                _ => {
                    eprintln!(
                        "Notice: handoff {} imported without sidecar (old export format).",
                        memory.id
                    );
                }
            }
        }

        // Import ADR sidecar if present.
        // Number-conflict check above guarantees the number is free at this point.
        if memory.memory_type == MemoryType::Adr
            && let (Some(num), Some(status_str), Some(adr_sec)) =
                (adr_number, adr_status_str, adr_sections_data)
        {
            use std::str::FromStr;
            match AdrStatus::from_str(&status_str) {
                Ok(status) => {
                    if let Err(e) = db.insert_adr_sidecar(
                        &memory.id,
                        project_id,
                        num,
                        status,
                        &adr_sec,
                        mem_created_at,
                        mem_updated_at,
                    ) {
                        eprintln!(
                            "Warning: failed to insert ADR sidecar for {} (number {}): {}",
                            memory.id, num, e
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: skipping ADR sidecar for {} — invalid status '{}': {}",
                        memory.id, status_str, e
                    );
                }
            }
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
    println!("Handoffs: {}", stats.handoff_count);
    if let Some(ts) = stats.latest_handoff_at {
        use chrono::{TimeZone, Utc};
        let dt = Utc.timestamp_opt(ts, 0).single();
        if let Some(dt) = dt {
            println!("Latest handoff: {}", dt.format("%Y-%m-%d %H:%M UTC"));
        }
    }

    Ok(())
}

fn cmd_insights(db: &Database, project_id: &str) -> Result<(), MemoryError> {
    let stats = db.get_project_stats(project_id)?;

    if stats.memory_count == 0 {
        println!("No memories found for this project.");
        return Ok(());
    }

    println!("Insights for project: {}", project_id);
    println!("─────────────────────────────────────────");

    // Top 10 most accessed
    let most_accessed = db.get_most_accessed(project_id, 10)?;
    println!("\nTop accessed memories:");
    if most_accessed.is_empty() || most_accessed.iter().all(|m| m.access_count == 0) {
        println!("  (none accessed yet)");
    } else {
        for memory in &most_accessed {
            if memory.access_count == 0 {
                break;
            }
            let preview = memory
                .summary
                .as_deref()
                .unwrap_or_else(|| &memory.content[..memory.content.len().min(60)]);
            println!(
                "  [{}] {} ({:?}) - {} accesses",
                &memory.id[..memory.id.len().min(8)],
                preview,
                memory.memory_type,
                memory.access_count
            );
        }
    }

    // Never accessed (older than 7 days)
    let never_accessed_count = db.get_never_accessed(project_id, 7)?;
    println!("\nNever accessed:");
    if never_accessed_count == 0 {
        println!("  All memories have been retrieved at least once.");
    } else {
        println!(
            "  {} memories stored 7+ days ago have never been retrieved.",
            never_accessed_count
        );
    }

    // Decaying (below 0.2 relevance)
    let decaying_count = db.get_below_relevance(project_id, 0.2)?;
    println!("\nDecaying memories (relevance < 0.2): {}", decaying_count);

    // Pinned and global counts
    println!("\nPinned: {}", stats.pinned_count);
    println!("Global: {}", stats.global_count);

    // Type distribution
    let type_dist = db.get_type_distribution(project_id)?;
    println!("\nType distribution:");
    for (memory_type, count) in &type_dist {
        println!("  {}: {}", memory_type, count);
    }

    // Storage rate (last 30 days)
    let rate = db.get_storage_rate(project_id, 30)?;
    println!("\nStorage rate (last 30 days): {:.2} memories/day", rate);

    // Health summary: subtract never-accessed + decaying, but add back the overlap to avoid
    // double-counting memories that are both never-accessed and decaying.
    let overlap = db.get_never_accessed_and_below_relevance(project_id, 7, 0.2)?;
    let healthy = stats
        .memory_count
        .saturating_sub(never_accessed_count + decaying_count - overlap);
    println!(
        "\nHealth: {} healthy, {} never accessed, {} decaying, {} pinned",
        healthy, never_accessed_count, decaying_count, stats.pinned_count
    );

    Ok(())
}

fn cmd_health(db: &Database, project_id: &str) -> Result<(), MemoryError> {
    let stats = db.get_project_stats(project_id)?;
    let decaying_count = db.get_below_relevance(project_id, 0.2)?;
    let never_accessed_count = db.get_never_accessed(project_id, 7)?;
    let potential_dupes = db.get_potential_duplicate_count(project_id)?;

    if decaying_count == 0 && never_accessed_count == 0 && potential_dupes == 0 {
        println!("All clear. {} memories, all healthy.", stats.memory_count);
        return Ok(());
    }

    println!("Health report for project: {}", project_id);
    println!("─────────────────────────────────────────");

    if decaying_count > 0 {
        println!(
            "\n{} memories below relevance 0.2 (candidates for pruning).",
            decaying_count
        );
        println!(
            "  Run `engram-cli prune -t 0.2 --confirm` to remove {} decayed memories.",
            decaying_count
        );
    }

    if never_accessed_count > 0 {
        println!(
            "\n{} memories stored 7+ days ago have never been retrieved.",
            never_accessed_count
        );
        println!("  Consider reviewing these with `engram-cli list` and removing unneeded ones.");
    }

    if potential_dupes > 0 {
        println!(
            "\n{} potential duplicate pairs (same cluster + type).",
            potential_dupes
        );
        println!("  Run `engram-cli dedup -t 0.90 --confirm` to merge duplicates.");
    }

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

fn cmd_dedup(
    db: &Database,
    project_id: &str,
    _embedding_service: &EmbeddingService,
    threshold: f32,
    confirm: bool,
) -> Result<(), MemoryError> {
    let all_embeddings = db.get_all_embeddings_for_project(project_id)?;

    // Find duplicate groups
    let mut processed: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut groups: Vec<Vec<(String, f32)>> = Vec::new();

    for i in 0..all_embeddings.len() {
        let (ref id_i, ref vec_i) = all_embeddings[i];
        if processed.contains(id_i) {
            continue;
        }

        let mem_i = match db.get_memory(id_i)? {
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

            let similarity = crate::embedding::cosine_similarity(vec_i, vec_j);
            if similarity >= threshold
                && let Some(mem_j) = db.get_memory(id_j)?
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
        println!("No duplicates found at threshold {:.2}", threshold);
        return Ok(());
    }

    println!("Found {} duplicate groups:", groups.len());
    for (gi, group) in groups.iter().enumerate() {
        println!("\n  Group {}:", gi + 1);
        for (id, sim) in group {
            if let Some(mem) = db.get_memory(id)? {
                let preview: String = mem.content.chars().take(80).collect();
                println!(
                    "    [{:.2}] {} ({}) - {}",
                    sim,
                    id,
                    mem.memory_type.as_str(),
                    preview
                );
            }
        }
    }

    if confirm {
        let mut merged_count = 0;
        for group in &groups {
            let mut with_time: Vec<(String, f32, i64)> = group
                .iter()
                .filter_map(|(id, sim)| {
                    db.get_memory(id)
                        .ok()
                        .flatten()
                        .map(|m| (id.clone(), *sim, m.updated_at))
                })
                .collect();
            with_time.sort_by_key(|(_, _, updated_at)| std::cmp::Reverse(*updated_at));

            if with_time.len() < 2 {
                continue;
            }

            let keeper_id = with_time[0].0.clone();
            for (old_id, _, _) in &with_time[1..] {
                let old_preview: String = db
                    .get_memory(old_id)?
                    .map(|m| m.content.chars().take(100).collect())
                    .unwrap_or_default();
                db.merge_memories(&keeper_id, old_id, &old_preview)?;
                merged_count += 1;
            }
        }
        println!("\nMerged {} duplicate memories.", merged_count);
    } else {
        let total_dups: usize = groups.iter().map(|g| g.len() - 1).sum();
        println!(
            "\n{} duplicates would be merged. Use --confirm to merge.",
            total_dups
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_context(
    db: &Database,
    project_id: &str,
    embedding_service: &EmbeddingService,
    context: &str,
    limit: usize,
    min_score: f64,
    types: &[String],
    current_branch: Option<&str>,
) -> Result<(), MemoryError> {
    let context_embedding = embedding_service.embed(context)?;
    let embeddings = db.get_all_embeddings_for_project_and_global(project_id)?;

    let type_filters: Vec<MemoryType> = types.iter().filter_map(|t| t.parse().ok()).collect();

    let mut scored: Vec<(String, f32)> = embeddings
        .iter()
        .map(|(id, vec)| {
            (
                id.clone(),
                embedding::cosine_similarity(&context_embedding, vec),
            )
        })
        .filter(|(_, score)| *score >= min_score as f32)
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Batch fetch candidates
    let candidate_ids: Vec<String> = scored
        .iter()
        .take(limit * 2)
        .map(|(id, _)| id.clone())
        .collect();
    let memories_map = db.get_memories_batch(&candidate_ids)?;

    let mut count = 0usize;
    for (id, similarity) in &scored {
        if count >= limit {
            break;
        }
        let Some(memory) = memories_map.get(id) else {
            continue;
        };

        // Branch filter: show global + current branch
        match &memory.branch {
            Some(branch) if current_branch.is_some_and(|cb| cb != branch) => continue,
            _ => {}
        }

        if !type_filters.is_empty() && !type_filters.contains(&memory.memory_type) {
            continue;
        }

        if count > 0 {
            println!();
        }
        println!(
            "[{}] ({}, importance: {:.1}, similarity: {:.2})",
            memory.memory_type.as_str(),
            memory.id,
            memory.importance,
            similarity,
        );
        if let Some(ref summary) = memory.summary {
            println!("{}", summary);
        } else {
            println!("{}", memory.content);
        }

        count += 1;
    }

    // Record access
    let accessed_ids: Vec<String> = scored
        .iter()
        .take(count)
        .map(|(id, _)| id.clone())
        .collect();
    if !accessed_ids.is_empty() {
        let _ = db.record_access_batch(&accessed_ids);
    }

    Ok(())
}

fn format_timestamp(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Prompt the user for a line of text on stdin. Returns empty string on EOF or blank.
fn prompt_line(label: &str) -> String {
    use std::io::{self, BufRead, Write};
    print!("{}: ", label);
    io::stdout().flush().ok();
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line).ok();
    line.trim().to_string()
}

/// Prompt for a list of items, one per line, until the user enters a blank line.
fn prompt_list(label: &str) -> Vec<String> {
    use std::io::{self, BufRead, Write};
    println!("{} (enter one per line, blank line to finish):", label);
    let stdin = io::stdin();
    let mut items = Vec::new();
    loop {
        print!("  > ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() {
            break;
        }
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            break;
        }
        items.push(trimmed);
    }
    items
}

/// Dispatch handoff subcommands.
fn cmd_handoff(
    db: &Database,
    project_id: &str,
    embedding_service: Option<&EmbeddingService>,
    current_branch: Option<&str>,
    cmd: HandoffCmd,
) -> Result<(), MemoryError> {
    match cmd {
        HandoffCmd::Create {
            summary,
            decisions,
            todos,
            blockers,
            mental_model,
            next_steps,
            notes,
            branch,
            continues_from,
            importance,
            no_pin,
            no_auto_link,
            from_file,
        } => {
            let embedding = embedding_service.ok_or_else(|| {
                MemoryError::InvalidType("embedding service required".to_string())
            })?;

            let sections = if let Some(path) = from_file {
                let content = std::fs::read_to_string(&path)?;
                let mut s = HandoffSections::parse_markdown(&content)?;
                if continues_from.is_some() {
                    s.continues_from = continues_from.clone();
                }
                s
            } else {
                // Use flags if provided; otherwise prompt interactively for missing required fields.
                let summary_text = if let Some(s) = summary {
                    s
                } else {
                    let s = prompt_line("Summary");
                    if s.is_empty() {
                        return Err(MemoryError::InvalidType(
                            "handoff: summary is required".to_string(),
                        ));
                    }
                    s
                };

                let decisions_list = if !decisions.is_empty() {
                    decisions
                } else {
                    prompt_list("Decisions")
                };

                let todos_list = if !todos.is_empty() {
                    todos
                } else {
                    prompt_list("Todos")
                };

                let blockers_list = if !blockers.is_empty() {
                    blockers
                } else {
                    prompt_list("Blockers")
                };

                let mental_model_text = if let Some(m) = mental_model {
                    m
                } else {
                    prompt_line("Mental model")
                };

                let next_steps_list = if !next_steps.is_empty() {
                    next_steps
                } else {
                    prompt_list("Next steps")
                };

                let notes_text = if let Some(n) = notes {
                    Some(n)
                } else {
                    let n = prompt_line("Notes (optional, blank to skip)");
                    if n.is_empty() { None } else { Some(n) }
                };

                HandoffSections {
                    summary: summary_text,
                    decisions: decisions_list,
                    todos: todos_list,
                    blockers: blockers_list,
                    mental_model: mental_model_text,
                    next_steps: next_steps_list,
                    notes: notes_text,
                    continues_from,
                }
            };

            // Resolve branch: CLI arg > current branch > error
            let resolved_branch = branch.as_deref().or(current_branch).map(str::to_string);

            let result = tools::create_handoff(
                db,
                embedding,
                project_id,
                resolved_branch.as_deref(),
                sections,
                importance,
                !no_pin,
                !no_auto_link,
            )?;

            println!("Handoff created: {}", result.id);
            if let Some(ref cf) = result.continues_from {
                println!("Continues from: {}", cf);
            }
            for w in &result.warnings {
                eprintln!("warning: {}", w);
            }
            if !result.linked_memory_ids.is_empty() {
                println!(
                    "Auto-linked {} memor{}:",
                    result.linked_memory_ids.len(),
                    if result.linked_memory_ids.len() == 1 {
                        "y"
                    } else {
                        "ies"
                    }
                );
                for id in &result.linked_memory_ids {
                    println!("  {}", id);
                }
            }
        }
        HandoffCmd::Resume {
            branch,
            query,
            max,
            include_off_branch,
            max_chars_per_section,
        } => {
            let embedding = embedding_service.ok_or_else(|| {
                MemoryError::InvalidType("embedding service required".to_string())
            })?;

            let resolved_branch = branch.as_deref().or(current_branch);

            let result = tools::resume_handoff(
                db,
                embedding,
                project_id,
                resolved_branch,
                query.as_deref(),
                max,
                include_off_branch,
                max_chars_per_section,
            )?;

            if let Some(ref msg) = result.message {
                println!("Note: {}", msg);
            }

            if result.latest_handoff_id.is_none() {
                println!("No handoffs found.");
                return Ok(());
            }

            println!(
                "Branch: {}",
                result.branch.as_deref().unwrap_or("(all branches)")
            );
            println!(
                "Latest handoff: {}",
                result.latest_handoff_id.as_deref().unwrap_or("none")
            );

            if result.chain.len() > 1 {
                println!("Chain ({} handoffs, oldest to newest):", result.chain.len());
                for id in &result.chain {
                    println!("  {}", id);
                }
            }

            if !result.top_sections.is_empty() {
                println!("\nTop sections:");
                for section in &result.top_sections {
                    println!("─────────────────────────────────────────");
                    println!(
                        "[{}] {} (score: {:.2})",
                        section.handoff_id, section.section_name, section.score
                    );
                    println!("{}", section.section_text);
                }
                println!("─────────────────────────────────────────");
            }

            if !result.linked_memories.is_empty() {
                println!("\nLinked memories:");
                for mem in &result.linked_memories {
                    let preview: String = mem.content.chars().take(80).collect();
                    println!("  [{}] ({:?}) {}", mem.id, mem.memory_type, preview);
                }
            }
        }
        HandoffCmd::Search {
            query,
            branch,
            section,
            limit,
        } => {
            let embedding = embedding_service.ok_or_else(|| {
                MemoryError::InvalidType("embedding service required".to_string())
            })?;

            let section_filter: Option<Vec<String>> = if section.is_empty() {
                None
            } else {
                Some(section)
            };

            let result = tools::search_handoffs(
                db,
                embedding,
                project_id,
                &query,
                branch.as_deref(),
                limit,
                section_filter.as_deref(),
            )?;

            if result.matches.is_empty() {
                println!("No matching handoff sections found.");
                return Ok(());
            }

            println!("{} match(es):", result.matches.len());
            for m in &result.matches {
                println!("─────────────────────────────────────────");
                println!(
                    "[{}] {} (score: {:.2})",
                    m.handoff_id, m.section_name, m.score
                );
                println!("{}", m.section_text);
            }
            println!("─────────────────────────────────────────");
        }
        HandoffCmd::Show { id } => {
            let memory = db
                .get_memory(&id)?
                .ok_or_else(|| MemoryError::NotFound(id.clone()))?;

            if memory.memory_type != MemoryType::Handoff {
                println!(
                    "Warning: memory {} is type {:?}, not handoff",
                    id, memory.memory_type
                );
            }

            // Render via format_handoff if sidecar is available
            match db.get_handoff_sections(&id)? {
                Some((sections, _)) => {
                    println!("{}", format::format_handoff(&memory, &sections));
                }
                None => {
                    // Fall back to plain content display
                    println!("ID: {}", memory.id);
                    println!("Branch: {}", memory.branch.as_deref().unwrap_or("(global)"));
                    println!("Importance: {:.2}", memory.importance);
                    println!("Created: {}", format_timestamp(memory.created_at));
                    println!("\nContent:\n{}", memory.content);
                }
            }
        }
    }

    Ok(())
}

/// Dispatch ADR subcommands.
fn cmd_adr(
    db: &Database,
    project_id: &str,
    embedding_service: Option<&EmbeddingService>,
    cmd: AdrCmd,
) -> Result<(), MemoryError> {
    use std::str::FromStr;

    match cmd {
        AdrCmd::Create {
            title,
            context,
            decision,
            consequences,
            status,
            supersedes,
            importance,
            no_pin,
            from_file,
        } => {
            let embedding = embedding_service.ok_or_else(|| {
                MemoryError::InvalidType("embedding service required".to_string())
            })?;

            let sections = if let Some(path) = from_file {
                let content = std::fs::read_to_string(&path)?;
                AdrSections::parse_markdown(&content)?
            } else {
                let title_text = if let Some(t) = title {
                    t
                } else {
                    let t = prompt_line("Title");
                    if t.is_empty() {
                        return Err(MemoryError::InvalidType(
                            "adr: title is required".to_string(),
                        ));
                    }
                    t
                };

                let context_text = if let Some(c) = context {
                    c
                } else {
                    prompt_line("Context")
                };

                let decision_text = if let Some(d) = decision {
                    d
                } else {
                    let d = prompt_line("Decision");
                    if d.is_empty() {
                        return Err(MemoryError::InvalidType(
                            "adr: decision is required".to_string(),
                        ));
                    }
                    d
                };

                let consequences_text = if let Some(c) = consequences {
                    c
                } else {
                    prompt_line("Consequences")
                };

                AdrSections {
                    title: title_text,
                    context: context_text,
                    decision: decision_text,
                    consequences: consequences_text,
                }
            };

            let parsed_status = AdrStatus::from_str(&status)
                .map_err(|e| MemoryError::InvalidType(e.to_string()))?;

            let result = tools::create_adr(
                db,
                embedding,
                project_id,
                sections,
                parsed_status,
                importance,
                !no_pin,
                supersedes,
            )?;

            println!(
                "ADR-{:04} created (status: {})",
                result.adr_number, result.status
            );
            if let Some(ref sid) = result.superseded_id {
                println!("Superseded: {}", sid);
            }
        }

        AdrCmd::UpdateStatus { number, status } => {
            let target_status = AdrStatus::from_str(&status)
                .map_err(|e| MemoryError::InvalidType(e.to_string()))?;

            if target_status == AdrStatus::Superseded {
                return Err(MemoryError::InvalidType(
                    "use adr create --supersedes to mark an ADR superseded".to_string(),
                ));
            }

            let id = db
                .get_adr_by_number(project_id, number)?
                .ok_or_else(|| MemoryError::NotFound(format!("ADR-{:04} not found", number)))?;

            let (_, current_status, _) = db
                .get_adr_sections(&id)?
                .ok_or_else(|| MemoryError::NotFound(format!("ADR sidecar missing for {}", id)))?;

            if !current_status.can_transition_to(target_status) {
                return Err(MemoryError::InvalidType(format!(
                    "invalid ADR status transition: {} -> {}",
                    current_status, target_status
                )));
            }

            db.update_adr_status(&id, target_status)?;
            println!("ADR-{:04} status updated to {}", number, target_status);
        }

        AdrCmd::List { status } => {
            let status_filter = status
                .as_deref()
                .map(AdrStatus::from_str)
                .transpose()
                .map_err(|e| MemoryError::InvalidType(e.to_string()))?;

            let rows = db.list_adrs(project_id, status_filter)?;

            if rows.is_empty() {
                println!("No ADRs found.");
                return Ok(());
            }

            println!("{:<6}  {:<12}  TITLE", "NUMBER", "STATUS");
            println!("{}", "-".repeat(60));
            for (number, adr_status, title, _id) in rows {
                println!("ADR-{:04}  {:<12}  {}", number, adr_status, title);
            }
        }

        AdrCmd::Show { number } => {
            let id = db
                .get_adr_by_number(project_id, number)?
                .ok_or_else(|| MemoryError::NotFound(format!("ADR-{:04} not found", number)))?;

            let (num, adr_status, sections) = db
                .get_adr_sections(&id)?
                .ok_or_else(|| MemoryError::NotFound(format!("ADR sidecar missing for {}", id)))?;

            let _ = db.record_access(&id);

            println!("ADR-{:04}: {}", num, sections.title);
            println!("Status: {}", adr_status);
            println!("ID: {}", id);
            println!("\n## Context\n\n{}", sections.context);
            println!("\n## Decision\n\n{}", sections.decision);
            println!("\n## Consequences\n\n{}", sections.consequences);
        }

        AdrCmd::Export { number, dir, write } => {
            let dry_run = !write;
            let target_dir =
                adr_export::adr_export_target_dir(dir.as_deref().and_then(|p| p.to_str()));
            let paths =
                adr_export::export_adr_to_disk(db, project_id, &target_dir, number, dry_run)?;
            for path in &paths {
                if dry_run {
                    println!("would write: {}", path.display());
                } else {
                    println!("wrote: {}", path.display());
                }
            }
            if paths.is_empty() {
                println!("No ADRs found for project '{}'.", project_id);
            }
        }
    }

    Ok(())
}

fn cmd_hooks(cmd: HooksCmd, db: &Database, project_id: &str) {
    match cmd {
        HooksCmd::Install => match hooks::install::install() {
            Ok(report) => {
                if report.added.is_empty() && report.skipped.is_empty() {
                    println!("No events to manage.");
                } else {
                    if !report.added.is_empty() {
                        println!(
                            "Installed {} managed entries to {}",
                            report.added.len(),
                            report.settings_path.display()
                        );
                        for ev in &report.added {
                            println!("  + {}", ev);
                        }
                    }
                    if !report.skipped.is_empty() {
                        println!("Already present (skipped):");
                        for ev in &report.skipped {
                            println!("  = {}", ev);
                        }
                    }
                    if let Some(bak) = &report.backup_path {
                        println!("Backup: {}", bak.display());
                    }
                }
            }
            Err(e) => {
                tracing::warn!("hooks install failed: {}", e);
                eprintln!("error: hooks install failed: {}", e);
            }
        },
        HooksCmd::Uninstall => match hooks::install::uninstall() {
            Ok(report) => {
                if report.removed.is_empty() {
                    println!(
                        "No engram-cli entries found in {}.",
                        report.settings_path.display()
                    );
                } else {
                    println!(
                        "Removed {} entries from {}:",
                        report.removed.len(),
                        report.settings_path.display()
                    );
                    for ev in &report.removed {
                        println!("  - {}", ev);
                    }
                    if let Some(bak) = &report.backup_path {
                        println!("Backup: {}", bak.display());
                    }
                }
            }
            Err(e) => {
                tracing::warn!("hooks uninstall failed: {}", e);
                eprintln!("error: hooks uninstall failed: {}", e);
            }
        },
        HooksCmd::Status => match hooks::install::status() {
            Ok(report) => {
                println!("Settings: {}", report.settings_path.display());
                if report.managed.is_empty() {
                    println!("No engram-cli entries installed.");
                } else {
                    println!("Managed events ({}):", report.managed.len());
                    for ev in &report.managed {
                        println!("  {}", ev);
                    }
                }
                if !report.shadowed.is_empty() {
                    println!("Shadowed (other hooks also registered for these events):");
                    for ev in &report.shadowed {
                        println!("  {}", ev);
                    }
                }
                // Show today's hook capture count vs daily cap.
                match db.count_hook_memories_today(project_id) {
                    Ok(n) => {
                        let cap = hooks::filter::hook_daily_cap();
                        if cap == 0 {
                            println!("Hook captures today: {} / unlimited", n);
                        } else {
                            println!("Hook captures today: {} / {}", n, cap);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("hooks status: could not query capture count: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("hooks status failed: {}", e);
                eprintln!("error: hooks status failed: {}", e);
            }
        },
    }
}

fn cmd_hook_event(
    event: String,
    payload: Option<String>,
    dry_run: bool,
    db: &Database,
    embedding_service: Option<&EmbeddingService>,
    project_id: &str,
) {
    use std::io::Read;

    let hook_event = match event.parse::<HookEvent>() {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("hook-event: unknown event name '{}': {}", event, e);
            return;
        }
    };

    let raw = match payload {
        Some(s) => s,
        None => {
            let mut buf = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
                tracing::warn!("hook-event: failed to read stdin: {}", e);
                return;
            }
            buf
        }
    };

    match hooks::dispatch::dispatch(hook_event, &raw, dry_run, db, embedding_service, project_id) {
        Ok(outcome) => {
            if dry_run && let hooks::dispatch::DispatchOutcome::DryRun(_) = &outcome {
                println!("{:?}", outcome);
            }
        }
        Err(e) => {
            tracing::warn!("hook-event dispatch error: {}", e);
        }
    }
}
