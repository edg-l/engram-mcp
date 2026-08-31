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
mod project;
mod summarize;
mod tools;

use db::Database;
use embedding::EmbeddingService;
use error::MemoryError;
use format::format_todo;
use hooks::HookEvent;
use memory::{
    AdrSections, AdrStatus, HandoffSections, Memory, MemoryType, RelationType, Relationship,
};
use summarize::{generate_summary, should_auto_summarize};
use tools::TodoOp;
use tools::curation::{CurationView, MatchedVia, Resolution, supersession_candidates};

#[derive(Parser)]
#[command(name = "engram-cli")]
#[command(about = "CLI for Engram memory management", long_about = None)]
#[command(version)]
struct Cli {
    /// Database path (default: ~/.local/share/engram/memories.db)
    #[arg(short, long)]
    database: Option<PathBuf>,

    /// Project ID (default: current directory name)
    #[arg(short, long)]
    project: Option<String>,

    /// Treat this as the current git branch instead of detecting it from the working
    /// directory. Use when the shell is in a different checkout than the work, e.g. a
    /// worktree. Takes precedence over ENGRAM_BRANCH. Distinct from the per-command
    /// `--branch` flags, which filter or tag a single command; this sets what "current
    /// branch" resolves to for all of them.
    #[arg(long, global = true)]
    current_branch: Option<String>,

    /// Emit machine-readable JSON instead of human-readable text. Supported by the
    /// read commands: query, context, stats, projects, list, show,
    /// handoff resume/search/show, adr list/show.
    #[arg(long, global = true)]
    json: bool,

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
        /// Return superseded and dead memories as they are, with no redirect or
        /// suppression. For auditing the store.
        #[arg(long)]
        include_superseded: bool,
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
        /// Which memories to show: live (default), superseded, dead, all
        #[arg(long, default_value = "live")]
        status: String,
        /// Sort order: relevance (default), created, updated, accessed
        #[arg(long, default_value = "relevance")]
        order: String,
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
        /// Memory ID this one replaces. Repeatable. Superseded memories stop being
        /// returned by search; queries that matched them return this memory instead.
        #[arg(long = "supersedes", value_name = "ID")]
        supersedes: Vec<String>,
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
        /// Mark dead: the subject no longer exists and there is no replacement.
        /// Excluded from retrieval entirely. Prefer `store --supersedes` when a
        /// replacement exists, so searches get redirected instead of nothing.
        #[arg(long)]
        dead: bool,
        /// Undo `--dead`.
        #[arg(long, conflicts_with = "dead")]
        alive: bool,
        /// Why it is dead.
        #[arg(long, requires = "dead")]
        dead_reason: Option<String>,
    },
    /// List recoverable snapshots of destroyed memories
    Trash {
        /// Maximum entries to show
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },
    /// Restore a memory from the trash
    Restore {
        /// Memory ID to restore (uses its most recent snapshot)
        id: Option<String>,
        /// Exact snapshot to restore, from `engram-cli trash`
        #[arg(long)]
        trash_id: Option<i64>,
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
        /// Export every project in the store, not just the current one
        #[arg(long)]
        all_projects: bool,
        /// Only include rows updated at or after this unix timestamp (seconds)
        #[arg(long)]
        since: Option<i64>,
    },
    /// Import memories from JSON
    Import {
        /// Input file, or "-" to read from stdin
        file: PathBuf,
        /// Import mode
        #[arg(short, long, default_value = "merge")]
        mode: String,
    },
    /// Bidirectional incremental sync of the whole store with a remote machine over ssh.
    /// Shells out to `ssh <target> <remote-bin> ...`; deletions never propagate.
    Sync {
        /// ssh target, e.g. "user@host" or an entry from ~/.ssh/config
        target: String,
        /// engram-cli binary to invoke on the remote
        #[arg(long, default_value = "engram-cli")]
        remote_bin: String,
        /// Report what would be transferred without importing or pushing anything
        #[arg(long)]
        dry_run: bool,
        /// Only pull from the remote
        #[arg(long, conflicts_with = "push_only")]
        pull_only: bool,
        /// Only push to the remote
        #[arg(long)]
        push_only: bool,
    },
    /// Show project statistics
    Stats,
    /// List all projects in the memory store
    Projects,
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
    /// Manage the project's durable todo list
    Todo {
        #[command(subcommand)]
        cmd: TodoCmd,
    },
}

/// Subcommands for `engram-cli todo`.
#[derive(Subcommand)]
enum TodoCmd {
    /// Open a new todo
    Add {
        /// Todo text
        text: String,
        /// Scope to a branch. Omit for a project-wide todo; "auto" uses the current branch
        #[arg(long)]
        branch: Option<String>,
        /// Topic tags (repeatable or comma-separated)
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        /// Importance (0.0-1.0, default 0.6)
        #[arg(long)]
        importance: Option<f64>,
    },
    /// List todos
    List {
        /// Lifecycle filter: open (default), done, dropped, all
        #[arg(long, default_value = "open")]
        status: String,
        /// current (default), project, all, or a literal branch name
        #[arg(long, default_value = "current")]
        branch_mode: String,
        /// Maximum results
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },
    /// Mark a todo finished
    Done {
        /// Todo id
        id: String,
    },
    /// Close a todo without doing it
    Drop {
        /// Todo id
        id: String,
        /// Why it was dropped (required)
        #[arg(long)]
        reason: String,
    },
    /// Return a closed todo to the open state
    Reopen {
        /// Todo id
        id: String,
    },
    /// Rewrite a todo's text
    Edit {
        /// Todo id
        id: String,
        /// New text
        text: String,
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
        /// Things preventing forward motion right now (missing access, failing dependency, unanswered question). (can be repeated or comma-separated)
        #[arg(long, value_delimiter = ',')]
        blockers: Vec<String>,
        /// Approaches attempted and abandoned, each with why it failed (can be repeated or comma-separated)
        #[arg(long, value_delimiter = ',')]
        tried: Vec<String>,
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
        /// Never prompt for missing sections; leave them empty
        #[arg(long)]
        non_interactive: bool,
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
        /// Never prompt for missing sections; require them as flags
        #[arg(long)]
        non_interactive: bool,
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

/// Commands that render JSON when `--json` is set. Anything else rejects the flag
/// rather than silently printing prose a caller expected to parse.
fn supports_json(cmd: &Commands) -> bool {
    match cmd {
        Commands::Query { .. }
        | Commands::Context { .. }
        | Commands::Stats
        | Commands::Projects
        | Commands::List { .. }
        | Commands::Trash { .. }
        | Commands::Show { .. } => true,
        Commands::Handoff { cmd } => matches!(
            cmd,
            HandoffCmd::Resume { .. } | HandoffCmd::Search { .. } | HandoffCmd::Show { .. }
        ),
        Commands::Adr { cmd } => matches!(cmd, AdrCmd::List { .. } | AdrCmd::Show { .. }),
        Commands::Todo { cmd } => matches!(cmd, TodoCmd::List { .. }),
        _ => false,
    }
}

/// Curation status for a retrieval command, seeded with previews of the memories it
/// might suppress so a redirect can say what it replaced.
///
/// `raw` returns an empty view: superseded and dead memories then come back untouched.
fn curation_view(db: &Database, project_id: &str, raw: bool) -> Result<CurationView, MemoryError> {
    if raw {
        return Ok(CurationView::empty());
    }
    let mut view = CurationView::load(db, project_id)?;
    for memory in db.get_all_memories_for_project(project_id)? {
        view.note_preview(&memory.id, &memory.content);
    }
    Ok(view)
}

/// Print a JSON document on stdout, pretty-printed so a human can read it too.
fn print_json(value: &serde_json::Value) {
    match serde_json::to_string_pretty(value) {
        Ok(text) => println!("{text}"),
        Err(e) => eprintln!("failed to render JSON: {e}"),
    }
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

/// Whether a command only reads memories, so an explicit `--project` that does
/// not exist yet is a mistake rather than a new project to create.
fn requires_existing_project(cmd: &Commands) -> bool {
    match cmd {
        Commands::Query { .. }
        | Commands::List { .. }
        | Commands::Trash { .. }
        | Commands::Show { .. }
        | Commands::Export { .. }
        | Commands::Stats
        | Commands::Context { .. }
        | Commands::Insights
        | Commands::Health => true,
        Commands::Handoff { cmd } => matches!(
            cmd,
            HandoffCmd::Resume { .. } | HandoffCmd::Search { .. } | HandoffCmd::Show { .. }
        ),
        Commands::Adr { cmd } => matches!(
            cmd,
            AdrCmd::List { .. } | AdrCmd::Show { .. } | AdrCmd::Export { .. }
        ),
        _ => false,
    }
}

/// Check if command needs embedding service (lazy initialization).
fn needs_embedding_service(cmd: &Commands) -> bool {
    match cmd {
        Commands::Query { .. }
        | Commands::Store { .. }
        | Commands::Update { .. }
        | Commands::Import { .. }
        | Commands::Sync { .. }
        | Commands::Dedup { .. }
        | Commands::Context { .. } => true,
        Commands::Handoff { cmd: handoff_cmd } => matches!(
            handoff_cmd,
            HandoffCmd::Create { .. } | HandoffCmd::Resume { .. } | HandoffCmd::Search { .. }
        ),
        Commands::Adr { cmd: adr_cmd } => matches!(adr_cmd, AdrCmd::Create { .. }),
        // Every todo mutation embeds its text (add/edit directly, the rest to reach the
        // shared batch path); listing does not.
        Commands::Todo { cmd: todo_cmd } => !matches!(todo_cmd, TodoCmd::List { .. }),
        Commands::HookEvent { .. } => true,
        Commands::Hooks { .. } => false,
        _ => false,
    }
}

/// Restore the default `SIGPIPE` disposition. Rust ignores `SIGPIPE`, so writing
/// to a closed pipe (`engram-cli projects | head`) raises `EPIPE` and `println!`
/// panics. With the default disposition the process terminates quietly instead,
/// which is what a pipeline expects.
#[cfg(unix)]
fn reset_sigpipe() {
    // SAFETY: called at the top of main, before any thread is spawned.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn reset_sigpipe() {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    reset_sigpipe();

    let cli = Cli::parse();

    let db_path = get_db_path(cli.database);
    let project_was_explicit = cli.project.is_some();
    let project_id = project::resolve_project_id(cli.project);

    // Ensure database directory exists
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if cli.json && !supports_json(&cli.command) {
        eprintln!(
            "--json is not supported for this command. Supported: query, context, stats, \
             projects, list, show, handoff resume/search/show, adr list/show."
        );
        std::process::exit(2);
    }

    let db = Database::open(&db_path)?;

    // An explicit --project that reads from a project which does not exist is
    // reported instead of silently returning nothing.
    if project_was_explicit
        && requires_existing_project(&cli.command)
        && !db.project_exists(&project_id)?
    {
        let known: Vec<String> = db.list_projects()?.into_iter().map(|p| p.id).collect();
        let error = MemoryError::UnknownProject {
            requested: project_id,
            known: if known.is_empty() {
                "(none)".to_string()
            } else {
                known.join(", ")
            },
        };
        eprintln!("{error}");
        std::process::exit(1);
    }

    db.get_or_create_project(&project_id, &project_id)?;

    // Initialize embedding service once, only if needed (saves ~500ms for commands that don't need it)
    let embedding_service = if needs_embedding_service(&cli.command) {
        Some(EmbeddingService::new()?)
    } else {
        None
    };

    // Resolve the current branch once for commands that need it.
    // Precedence: --current-branch > ENGRAM_BRANCH > git detection.
    let current_branch = cli
        .current_branch
        .clone()
        .filter(|b| !b.is_empty())
        .or_else(project::current_branch);

    match cli.command {
        Commands::Query {
            query,
            limit,
            min_relevance,
            types,
            branch_mode,
            include_superseded,
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
                include_superseded,
                cli.json,
            )?;
        }
        Commands::List {
            r#type,
            limit,
            branch,
            status,
            order,
        } => {
            cmd_list(
                &db,
                &project_id,
                r#type.as_deref(),
                limit,
                branch.as_deref(),
                &status,
                &order,
                cli.json,
            )?;
        }
        Commands::Show { id } => {
            cmd_show(&db, &id, cli.json)?;
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
            supersedes,
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
                &supersedes,
            )?;
        }
        Commands::Delete { id } => {
            cmd_delete(&db, &id)?;
        }
        Commands::Trash { limit } => {
            cmd_trash(&db, &project_id, limit, cli.json)?;
        }
        Commands::Restore { id, trash_id } => {
            cmd_restore(&db, id.as_deref(), trash_id)?;
        }
        Commands::Update {
            id,
            content,
            importance,
            tags,
            summary,
            artifacts,
            clear_artifacts,
            dead,
            alive,
            dead_reason,
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
            let dead_flag = if dead {
                Some(true)
            } else if alive {
                Some(false)
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
                dead_flag,
                dead_reason.as_deref(),
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
        Commands::Export {
            output,
            embeddings,
            all_projects,
            since,
        } => {
            cmd_export(&db, &project_id, output, embeddings, all_projects, since)?;
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
        Commands::Sync {
            target,
            remote_bin,
            dry_run,
            pull_only,
            push_only,
        } => {
            cmd_sync(
                &db,
                embedding_service.as_ref().unwrap(),
                &target,
                &remote_bin,
                dry_run,
                pull_only,
                push_only,
            )?;
        }
        Commands::Stats => {
            cmd_stats(&db, &project_id, cli.json)?;
        }
        Commands::Projects => {
            cmd_projects(&db, &project_id, cli.json)?;
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
                cli.json,
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
                cli.json,
            )?;
        }
        Commands::Adr { cmd: adr_cmd } => {
            cmd_adr(
                &db,
                &project_id,
                embedding_service.as_ref(),
                adr_cmd,
                cli.json,
            )?;
        }
        Commands::Todo { cmd: todo_cmd } => {
            cmd_todo(
                &db,
                &project_id,
                embedding_service.as_ref(),
                current_branch.as_deref(),
                todo_cmd,
                cli.json,
            )?;
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
    include_superseded: bool,
    json: bool,
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

    // Same curation pass the MCP `memory_query` applies: superseded matches are replaced
    // by their successor at the superseded memory's rank, dead ones are dropped.
    let curation = curation_view(db, project_id, include_superseded)?;

    // (memory, combined score, semantic score, keyword score, matched_via)
    let mut hits: Vec<(Memory, f64, f64, f64, Option<MatchedVia>)> = Vec::new();
    let mut emitted: HashMap<String, usize> = HashMap::new();
    for (id, score, semantic_score, keyword_score) in scored_results {
        if hits.len() >= limit {
            break;
        }
        let (effective_id, matched_via) = match curation.resolve(&id) {
            Resolution::Keep => (id, None),
            Resolution::Drop => continue,
            Resolution::Redirect { successor_id, via } => (successor_id, Some(via)),
        };
        if let Some(&index) = emitted.get(&effective_id) {
            if let Some(via) = matched_via
                && hits[index].4.is_none()
            {
                hits[index].4 = Some(via);
            }
            continue;
        }
        if let Some(memory) = db.get_memory(&effective_id)? {
            emitted.insert(effective_id, hits.len());
            hits.push((memory, score, semantic_score, keyword_score, matched_via));
        }
    }

    if json {
        let memories: Vec<serde_json::Value> = hits
            .iter()
            .map(
                |(memory, score, semantic_score, keyword_score, matched_via)| {
                    serde_json::json!({
                        "memory": memory,
                        "score": score,
                        "semantic_score": semantic_score,
                        "keyword_score": keyword_score,
                        "matched_via": matched_via,
                    })
                },
            )
            .collect();
        print_json(&serde_json::json!({
            "project": project_id,
            "query": query,
            "count": memories.len(),
            "memories": memories,
        }));
        return Ok(());
    }

    for (memory, score, semantic_score, keyword_score, matched_via) in &hits {
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
        if let Some(via) = matched_via {
            println!(
                "Replaces: {} — \"{}\"",
                via.superseded_id, via.superseded_preview
            );
        }
        if let Some(summary) = &memory.summary {
            println!("Summary: {}", summary);
        }
        println!("Content: {}", memory.content);
        if !memory.tags.is_empty() {
            println!("Tags: {}", memory.tags.join(", "));
        }
    }

    if hits.is_empty() {
        println!("No matching memories found.");
    } else {
        println!("─────────────────────────────────────────");
        println!("Found {} memories", hits.len());
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_list(
    db: &Database,
    project_id: &str,
    type_filter: Option<&str>,
    limit: usize,
    branch_filter: Option<&str>,
    status: &str,
    order: &str,
    json: bool,
) -> Result<(), MemoryError> {
    let type_filters: Option<Vec<MemoryType>> =
        type_filter.and_then(|t| t.parse().ok()).map(|t| vec![t]);

    // Convert branch filter for query
    let branch_query = branch_filter.map(Some);

    // Fetch before the status filter narrows the set, so a page of "live" memories is
    // still a full page when some of the matches turn out to be superseded.
    let dead = db.get_dead_ids(project_id)?;
    let supersession = db.get_supersession_map(project_id)?;

    let mut memories = db.query_memories_with_branch(
        project_id,
        type_filters.as_deref(),
        None,
        None,
        usize::MAX,
        branch_query,
    )?;

    memories.retain(|m| {
        let is_dead = dead.contains(&m.id);
        let is_superseded = supersession.is_superseded(&m.id);
        match status {
            "dead" => is_dead,
            "superseded" => is_superseded,
            "all" => true,
            _ => !is_dead && !is_superseded,
        }
    });

    match order {
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
    memories.truncate(limit);

    if json {
        print_json(&serde_json::json!({
            "project": project_id,
            "status": status,
            "order": order,
            "total": total,
            "count": memories.len(),
            "memories": memories,
        }));
        return Ok(());
    }

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
        let status_info = if dead.contains(&memory.id) {
            " [dead]".to_string()
        } else if let Some(successor) = supersession.terminal_successor(&memory.id) {
            format!(" [superseded by {successor}]")
        } else {
            String::new()
        };
        println!(
            "{} [{:?}]{}{} {:.2} - {}",
            memory.id,
            memory.memory_type,
            branch_info,
            status_info,
            memory.relevance_score,
            summary
        );
    }
    if total > memories.len() {
        println!("\nShowing {} of {} memories", memories.len(), total);
    } else {
        println!("\nTotal: {} memories", total);
    }

    Ok(())
}

fn cmd_trash(db: &Database, project_id: &str, limit: usize, json: bool) -> Result<(), MemoryError> {
    let entries = db.list_trash(project_id, limit)?;
    let total = db.count_trash(project_id)?;

    if json {
        let rows: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "trash_id": e.trash_id,
                    "memory_id": e.memory.id,
                    "op": e.op,
                    "trashed_at": e.trashed_at,
                    "memory": e.memory,
                    "relationships": e.relationships.len(),
                })
            })
            .collect();
        print_json(&serde_json::json!({
            "project": project_id,
            "total": total,
            "count": rows.len(),
            "entries": rows,
        }));
        return Ok(());
    }

    if entries.is_empty() {
        println!("Trash is empty.");
        return Ok(());
    }

    for entry in &entries {
        let preview: String = entry.memory.content.chars().take(70).collect();
        println!(
            "{:>6}  {:<7} {} [{:?}] - {}",
            entry.trash_id, entry.op, entry.memory.id, entry.memory.memory_type, preview
        );
    }
    println!(
        "\n{} of {} entries. Restore with `engram-cli restore <memory-id>` or `--trash-id <n>`.",
        entries.len(),
        total
    );

    Ok(())
}

fn cmd_restore(db: &Database, id: Option<&str>, trash_id: Option<i64>) -> Result<(), MemoryError> {
    let entry = match (trash_id, id) {
        (Some(trash_id), _) => db.get_trash_entry(trash_id)?,
        (None, Some(id)) => db.latest_trash_for_memory(id)?,
        (None, None) => {
            eprintln!("Pass a memory ID, or --trash-id from `engram-cli trash`.");
            std::process::exit(2);
        }
    };

    let Some(entry) = entry else {
        eprintln!("Nothing in the trash matches that.");
        std::process::exit(1);
    };

    let outcome = db.restore_trash_entry(entry.trash_id)?;
    println!(
        "Restored {} (removed by {}).",
        outcome.memory.id, outcome.op
    );
    if outcome.overwrote_existing {
        println!("A live memory with that ID was replaced; it is now in the trash itself.");
    }
    if outcome.edges_restored > 0 {
        println!("Reconnected {} relationship(s).", outcome.edges_restored);
    }
    if outcome.edges_dropped > 0 {
        println!(
            "{} relationship(s) could not be restored: the memory at the other end is gone.",
            outcome.edges_dropped
        );
    }

    Ok(())
}

fn cmd_show(db: &Database, id: &str, json: bool) -> Result<(), MemoryError> {
    let memory = db
        .get_memory(id)?
        .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

    if json {
        let outgoing = db.get_relationships_from(id)?;
        let incoming = db.get_relationships_to(id)?;
        print_json(&serde_json::json!({
            "memory": memory,
            "relationships": {"outgoing": outgoing, "incoming": incoming},
        }));
        db.record_access(id)?;
        return Ok(());
    }

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
    supersedes: &[String],
) -> Result<(), MemoryError> {
    let memory_type: MemoryType = type_str
        .parse()
        .map_err(|_| MemoryError::InvalidType(type_str.to_string()))?;

    // Fail before storing anything if a superseded id is wrong.
    for old_id in supersedes {
        db.get_memory(old_id)?
            .ok_or_else(|| MemoryError::NotFound(old_id.clone()))?;
    }

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

    let embedding = embedding_service.embed_memory(memory_type, content)?;

    // Same path as the MCP `memory_store`, so a memory stored from the CLI gets the same
    // dedup, provenance, and recoverability as one stored by an agent. Memories this one
    // supersedes are exempt: they are distinct by assertion.
    let exempt: std::collections::HashSet<String> = supersedes.iter().cloned().collect();
    let outcome = tools::store::store_with_dedup_exempting(
        db,
        Some(embedding_service),
        project_id,
        memory,
        Some(&embedding),
        tools::schemas::dedup_threshold(),
        None,
        &exempt,
    )?;

    let (id, merged_with) = match outcome {
        tools::store::StoreOutcome::Stored(id) => (id, None),
        tools::store::StoreOutcome::Merged {
            id,
            merged_with,
            similarity,
        } => (id, Some((merged_with, similarity))),
        tools::store::StoreOutcome::SkippedSimilar { .. } => {
            unreachable!("skip_above is None, so nothing can be skipped")
        }
    };

    for old_id in supersedes {
        let rel = Relationship {
            id: format!("rel_{}", uuid::Uuid::new_v4().simple()),
            source_id: id.clone(),
            target_id: old_id.clone(),
            relation_type: RelationType::Supersedes,
            strength: 1.0,
            created_at: now,
        };
        db.create_relationship(&rel)?;
    }

    // New knowledge advanced the project's clock, so every other memory in it is one
    // store-day more displaced. Matches what the MCP store path does.
    if let Some(project) = db.get_project(project_id)? {
        db.update_relevance_scores(project_id, project.decay_rate)?;
    }

    if let Some(ref b) = branch {
        println!("Memory stored: {} (branch: {})", id, b);
    } else {
        println!("Memory stored: {} (global)", id);
    }
    if let Some((merged_with, similarity)) = &merged_with {
        println!(
            "Merged with near-duplicate {merged_with} (similarity {similarity:.2}); its full \
             content is kept in this memory's provenance and in the trash."
        );
    }
    for old_id in supersedes {
        println!("Supersedes {old_id}; searches that matched it now return this memory.");
    }

    // Nothing else tells you that a memory from months ago is about the same subject:
    // it never surfaced, so it never became a candidate for anything.
    if supersedes.is_empty() {
        let mut exclusions: Vec<&str> = vec![id.as_str()];
        if let Some((merged_with, _)) = &merged_with {
            exclusions.push(merged_with.as_str());
        }
        let candidates =
            supersession_candidates(db, project_id, &embedding, memory_type, &exclusions)?;
        if !candidates.is_empty() {
            println!("\nExisting memories on what looks like the same subject:");
            for candidate in &candidates {
                println!(
                    "  {} ({:.2}) {}",
                    candidate.id, candidate.similarity, candidate.preview
                );
            }
            println!(
                "If this replaces one of them, store again with --supersedes <id> instead of \
                 keeping both."
            );
        }
    }

    Ok(())
}

fn cmd_delete(db: &Database, id: &str) -> Result<(), MemoryError> {
    // Read it out first: the result of a delete is the only place the caller still sees
    // what the memory said, and one memory can carry several unrelated claims.
    let doomed = db.get_memory(id)?;

    if db.delete_memory(id)? {
        println!("Deleted memory: {}", id);
        if let Some(memory) = doomed {
            println!("\n--- deleted content ---\n{}", memory.content);
            if let Some(sources) = &memory.merged_from
                && !sources.is_empty()
            {
                println!(
                    "\nThis memory had absorbed {} other memor{} by dedup:",
                    sources.len(),
                    if sources.len() == 1 { "y" } else { "ies" }
                );
                for source in sources {
                    println!("\n  [{}]\n{}", source.id, source.content_or_preview());
                }
            }
            println!("\n--- end ---");
        }
        println!("Recoverable with `engram-cli restore {id}` until the trash is swept.");
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
    dead: Option<bool>,
    dead_reason: Option<&str>,
) -> Result<(), MemoryError> {
    let mut memory = db
        .get_memory(id)?
        .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

    // Content is replaced, not patched, so print what is about to be overwritten.
    let previous_content = memory.content.clone();
    let content_replaced = content.as_ref().is_some_and(|new| *new != memory.content);
    if content_replaced {
        db.trash_memory(id, db::OP_UPDATE)?;
    }

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

    if let Some(dead) = dead {
        db.set_dead(id, dead, dead_reason)?;
    }

    println!("Updated memory: {}", id);
    if content_replaced {
        println!("\n--- replaced content ---\n{previous_content}\n--- end ---");
        println!("Recoverable with `engram-cli restore {id}` until the trash is swept.");
    }
    match dead {
        Some(true) => println!("Marked dead: excluded from all retrieval."),
        Some(false) => println!("Marked live: visible to retrieval again."),
        None => {}
    }

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

/// Build an `ExportData` payload straight from the database. Shared by `cmd_export`
/// (writes it to a file or stdout) and `cmd_sync` (serializes it for a piped `ssh`
/// transfer), so the sidecar-collection logic below exists exactly once.
fn build_export_data(
    db: &Database,
    project_filter: Option<&str>,
    since: Option<i64>,
    include_embeddings: bool,
    scope: export::ExportScope,
    export_project_id: &str,
    exclude_origin: Option<&str>,
) -> Result<export::ExportData, MemoryError> {
    let mut memories = db.export_memories(project_filter, since)?;
    // Sync's push half only: drop memories that arrived unmodified from this exact
    // remote (see `origin_unchanged_since_pull`), so a sync doesn't echo pulled content
    // straight back to its own source on the very next run.
    if let Some(remote) = exclude_origin {
        let candidates: Vec<(String, i64)> = memories
            .iter()
            .map(|m| (m.id.clone(), m.updated_at))
            .collect();
        let exclude_ids = db.origin_unchanged_since_pull(remote, &candidates)?;
        if !exclude_ids.is_empty() {
            memories.retain(|m| !exclude_ids.contains(&m.id));
        }
    }
    let relationships = db.export_relationships(project_filter, since)?;

    let embeddings = if include_embeddings {
        Some(db.export_embeddings(project_filter, since)?)
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

    // Status sidecar: every memory that has ever been marked dead or revived, so a
    // revival converges through export/import the same way a dead-marking does.
    let status_map: export::StatusMap = db
        .export_status_rows(project_filter, since)?
        .into_iter()
        .map(|(id, (dead, reason, marked_at))| {
            (
                id,
                export::ExportedStatus {
                    dead,
                    reason,
                    marked_at,
                },
            )
        })
        .collect();

    // Todo lifecycle sidecar.
    let todo_map: export::TodoMap = db
        .export_todo_rows(project_filter, since)?
        .into_iter()
        .map(|t| {
            (
                t.id,
                export::ExportedTodo {
                    status: t.status,
                    reason: t.reason,
                    closed_at: t.closed_at,
                },
            )
        })
        .collect();

    Ok(export::create_export(
        export_project_id,
        memories,
        relationships,
        embeddings,
        handoff_sidecars,
        &adr_sidecars,
        db.dominant_model_version()?,
        scope,
        &status_map,
        &todo_map,
    ))
}

fn cmd_export(
    db: &Database,
    project_id: &str,
    output: Option<PathBuf>,
    include_embeddings: bool,
    all_projects: bool,
    since: Option<i64>,
) -> Result<(), MemoryError> {
    // `None` draws from every project; the unbounded `export_*` getters (src/db/sync.rs)
    // replace the old `get_all_*_for_project` calls even for the default single-project
    // case, so a store past `query_memories`'s 10000-row cap is never silently truncated.
    let project_filter = if all_projects { None } else { Some(project_id) };

    let scope = if all_projects {
        export::ExportScope::AllProjects
    } else {
        export::ExportScope::Project
    };
    // Each memory carries its own project_id in an all-projects export; the top-level
    // field is meaningless there and left empty rather than naming one arbitrary project.
    let export_project_id = if all_projects {
        String::new()
    } else {
        project_id.to_string()
    };

    let export_data = build_export_data(
        db,
        project_filter,
        since,
        include_embeddings,
        scope,
        &export_project_id,
        None,
    )?;

    let json = serde_json::to_string_pretty(&export_data)?;

    if let Some(path) = output {
        std::fs::write(&path, &json)?;
        println!("Exported to: {}", path.display());
    } else {
        println!("{}", json);
    }

    Ok(())
}

/// Counts from `import_export_data`, reported by both `cmd_import` (as prose) and
/// `cmd_sync`'s pull half.
struct ImportSummary {
    imported: usize,
    updated: usize,
    rel_imported: usize,
    skipped: usize,
}

/// The memory/relationship/sidecar merge loop, shared by `cmd_import` (a payload read
/// from a file or stdin) and `cmd_sync`'s pull half (a payload already parsed from a
/// remote's `export` stdout, always `ImportMode::Merge`). `project_id` only matters when
/// `export_data.scope` is `Project`; an `AllProjects` payload re-homes nothing and keeps
/// each memory's own `project_id`, ignoring this argument entirely.
fn import_export_data(
    db: &Database,
    project_id: &str,
    embedding_service: &EmbeddingService,
    export_data: export::ExportData,
    import_mode: export::ImportMode,
    origin_remote: Option<&str>,
) -> Result<ImportSummary, MemoryError> {
    let all_projects = export_data.scope == export::ExportScope::AllProjects;

    // A payload from a different embedding model is not comparable to locally-stored
    // vectors in cosine search: ignore it and re-embed locally instead of trusting
    // foreign vectors under this store's model label.
    let reembed_due_to_model_mismatch = export_data
        .model_version
        .as_deref()
        .is_some_and(|v| v != embedding_service.model_version());

    let mut imported = 0;
    let mut updated = 0;
    let mut skipped = 0;
    // Every distinct project touched by this import, so relevance scores are recomputed
    // once per project rather than only for the CLI's own `-p`/cwd project. Also doubles
    // as the "already ensured" set for `get_or_create_project` in all-projects mode.
    let mut touched_projects: std::collections::HashSet<String> = std::collections::HashSet::new();

    for exported in export_data.memories {
        let mut memory = exported.memory;
        let encoded_embedding = exported.embedding;
        let sections = exported.sections;
        let section_embedding_keys = exported.section_embedding_keys;
        let encoded_section_embeddings = exported.section_embeddings;
        let adr_number = exported.adr_number;
        let adr_status_str = exported.adr_status;
        let adr_sections_data = exported.adr_sections;
        let status_sidecar = exported.status;
        let todo_sidecar = exported.todo;
        let mem_created_at = memory.created_at;
        let mem_updated_at = memory.updated_at;
        let incoming_updated_at = memory.updated_at;
        // All-projects payloads keep each memory's own project_id and ignore the CLI's
        // resolved `-p`/cwd project entirely; a single-project payload is re-homed to it,
        // matching the pre-4.4 behavior.
        let target_project_id = if all_projects {
            memory.project_id.clone()
        } else {
            project_id.to_string()
        };
        memory.project_id = target_project_id.clone();
        if all_projects && touched_projects.insert(target_project_id.clone()) {
            db.get_or_create_project(&target_project_id, &target_project_id)?;
        }
        // Preserve the source machine's updated_at rather than stamping "now": overwriting
        // it would make the importer's copy look newest on every import and defeat
        // last-write-wins convergence (it would ping-pong forever between two machines).

        // Merge mode is last-write-wins on `updated_at`, ties keep the local copy. Replace
        // mode has already cleared the project, so it never looks the id up at all — the
        // existence check below is Merge-only, matching the pre-LWW behavior for Replace.
        let mut content_differs = false;
        let is_update = if import_mode == export::ImportMode::Merge {
            match db.get_memory(&memory.id)? {
                Some(existing_memory) if incoming_updated_at <= existing_memory.updated_at => {
                    skipped += 1;
                    continue;
                }
                Some(existing_memory) => {
                    content_differs = existing_memory.content != memory.content;
                    if content_differs {
                        db.trash_memory(&memory.id, db::OP_UPDATE)?;
                    }
                    // access_count/last_accessed_at take the max of both sides rather than
                    // LWW — they are usage counters, not a single machine's claim.
                    memory.access_count = memory.access_count.max(existing_memory.access_count);
                    memory.last_accessed_at = memory
                        .last_accessed_at
                        .max(existing_memory.last_accessed_at);
                    true
                }
                None => false,
            }
        } else {
            false
        };

        // For ADR memories with a known number, pre-check the number BEFORE storing
        // the memory row.  If the number is already taken, skip the entire memory
        // (memory row + embedding + sidecar) to keep them consistent. Only applies to
        // brand-new ADRs; an existing ADR being updated already owns its number.
        if !is_update
            && memory.memory_type == MemoryType::Adr
            && let Some(num) = adr_number
            && db.get_adr_by_number(&target_project_id, num)?.is_some()
        {
            skipped += 1;
            eprintln!(
                "Warning: skipping imported ADR {} — number {} already exists in project",
                memory.id, num
            );
            continue;
        }

        if is_update {
            db.update_memory(&memory)?;
            updated += 1;
        } else {
            db.store_memory(&memory)?;
            imported += 1;
        }
        touched_projects.insert(target_project_id.clone());
        // Pin this memory's origin so sync's push half never echoes it straight back to
        // the remote it just arrived from (see `origin_unchanged_since_pull`). Only set
        // during sync's own in-process pull import; a plain `engram-cli import` passes
        // `None` and does no origin tracking, since there is no "remote" to attribute it to.
        if let Some(remote) = origin_remote {
            db.set_memory_origin(&memory.id, remote, memory.updated_at)?;
        }

        // Handle embedding
        let stored_vector: Option<Vec<f32>> = if reembed_due_to_model_mismatch {
            None
        } else {
            encoded_embedding
                .as_deref()
                .and_then(|e| export::decode_embedding(e).ok())
        };
        let final_vector = match stored_vector {
            Some(vector) => vector,
            None => embedding_service.embed_memory(memory.memory_type, &memory.content)?,
        };
        db.store_embedding(&memory.id, &final_vector, embedding_service.model_version())?;

        // Assign to a cluster for every inserted or content-updated memory, matching the
        // per-store call shape `memory_store` uses — the background re-clustering job
        // amortizes the full pairwise merge/split pass separately and on its own schedule.
        // ADR and Todo memories are exempt: `store_adr_atomic`/`store_todo_atomic` never
        // route them through clustering either (ADRs are project-global reference
        // documents, and a task list clustered by topic serves no retrieval purpose).
        if (!is_update || content_differs)
            && memory.memory_type != MemoryType::Adr
            && memory.memory_type != MemoryType::Todo
        {
            // A content-changed update may already belong to a cluster whose centroid no
            // longer represents it. Vacate that membership first — mirroring memory_delete's
            // pattern in handler.rs — so re-assignment can't leave the memory in two
            // clusters at once with a stale, unrecomputed centroid left behind in the old one.
            if is_update
                && content_differs
                && let Ok(Some(old_cluster_id)) = db.remove_from_cluster(&memory.id)
            {
                let member_ids = db.get_cluster_member_ids(&old_cluster_id)?;
                if member_ids.is_empty() {
                    let _ = db.delete_empty_clusters(&target_project_id);
                } else {
                    let new_centroid = tools::cluster::compute_cluster_centroid(db, &member_ids)?;
                    let summary = tools::cluster::generate_cluster_summary(db, &member_ids)?;
                    if let Some(centroid) = new_centroid {
                        let _ = db.update_cluster_centroid(&old_cluster_id, &centroid, &summary);
                    }
                }
            }

            tools::cluster::assign_to_cluster(
                db,
                &target_project_id,
                &memory.id,
                &final_vector,
                &memory.content,
                memory.importance,
            )?;
        }

        // Import handoff sidecar if present. Only for brand-new handoffs: handoff and ADR
        // sidecars are not among the LWW-governed mutable fields, so an update leaves the
        // local sidecar row alone.
        // Old exports without sidecar fields skip this step silently.
        if !is_update && memory.memory_type == MemoryType::Handoff {
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

        // Import ADR sidecar if present (brand-new ADRs only; see note above).
        // Number-conflict check above guarantees the number is free at this point.
        if !is_update
            && memory.memory_type == MemoryType::Adr
            && let (Some(num), Some(status_str), Some(adr_sec)) =
                (adr_number, adr_status_str, adr_sections_data)
        {
            use std::str::FromStr;
            match AdrStatus::from_str(&status_str) {
                Ok(status) => {
                    if let Err(e) = db.insert_adr_sidecar(
                        &memory.id,
                        &target_project_id,
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

        // Apply retrieval-status and todo sidecars for both new and updated memories: both
        // are LWW-governed fields (see the Convergence rule), and `set_dead_at`/
        // `upsert_todo_item` are upsert-safe.
        if let Some(status) = status_sidecar {
            db.set_dead_at(
                &memory.id,
                status.dead,
                status.reason.as_deref(),
                status.marked_at,
            )?;
        }
        if let Some(todo) = todo_sidecar {
            db.upsert_todo_item(
                &memory.id,
                todo.status,
                todo.reason.as_deref(),
                todo.closed_at,
            )?;
        }
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

    // New or changed knowledge advanced each touched project's clock, so relevance needs
    // recomputing there — matches what a store does. Looping `touched_projects` (rather
    // than the single CLI `-p`/cwd project) is what makes an all-projects import recompute
    // every project it actually wrote to, not just the one the command happened to run in.
    for pid in &touched_projects {
        if let Some(project) = db.get_project(pid)? {
            db.update_relevance_scores(pid, project.decay_rate)?;
        }
    }

    Ok(ImportSummary {
        imported,
        updated,
        rel_imported,
        skipped,
    })
}

fn cmd_import(
    db: &Database,
    project_id: &str,
    embedding_service: &EmbeddingService,
    file: &PathBuf,
    mode: &str,
) -> Result<(), MemoryError> {
    let json = if file.as_os_str() == "-" {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        std::fs::read_to_string(file)?
    };
    let export_data: export::ExportData = serde_json::from_str(&json)?;

    export::validate_import(&export_data).map_err(MemoryError::Embedding)?;

    let import_mode: export::ImportMode = mode.parse().unwrap_or(export::ImportMode::Merge);
    let all_projects = export_data.scope == export::ExportScope::AllProjects;

    if all_projects && import_mode == export::ImportMode::Replace {
        return Err(MemoryError::Embedding(
            "--mode replace cannot be used with an all-projects payload (scope: all_projects): \
             replace would wipe every project in this store, not just the one the payload came \
             from. Re-export a single project's scope, or import this payload with --mode merge."
                .to_string(),
        ));
    }

    if import_mode == export::ImportMode::Replace {
        db.delete_project_data(project_id)?;
        println!("Cleared existing data.");
    }

    let summary = import_export_data(
        db,
        project_id,
        embedding_service,
        export_data,
        import_mode,
        None,
    )?;

    println!(
        "Imported {} memories, {} updated, {} relationships ({} skipped)",
        summary.imported, summary.updated, summary.rel_imported, summary.skipped
    );

    Ok(())
}

/// The greatest `memories[].updated_at` in a payload, or `None` if it carried no
/// memories. Watermarks derive from this rather than wall-clock, so clock skew between
/// the two machines can never skip a row.
fn max_memory_updated_at(data: &export::ExportData) -> Option<i64> {
    data.memories.iter().map(|m| m.memory.updated_at).max()
}

/// Bidirectional incremental sync with one remote over `ssh`, one process per direction.
///
/// Order matters: the push payload is built from the *local* store before the pull half
/// imports anything, so rows that just arrived from the remote are not immediately
/// bounced back to it in the same invocation. The pull half runs in-process (the remote
/// only exports; this binary does the merge, reusing `import_export_data` — the same
/// re-embedding and cluster-assignment path a file-based `import` takes). The push half
/// is a genuine subprocess: the remote's own `import --mode merge -` does that side's
/// merge, on its own store.
///
/// The two watermarks are independent: a failed half leaves both untouched for that
/// direction only, and never touches the other direction's watermark.
#[allow(clippy::too_many_arguments)]
fn cmd_sync(
    db: &Database,
    embedding_service: &EmbeddingService,
    target: &str,
    remote_bin: &str,
    dry_run: bool,
    pull_only: bool,
    push_only: bool,
) -> Result<(), MemoryError> {
    let ssh_cmd = std::env::var("ENGRAM_SSH_CMD").unwrap_or_else(|_| "ssh".to_string());

    // Preflight: confirm remote_bin is reachable and understands --version, so a missing
    // binary or a broken PATH under non-interactive ssh fails with a clear cause instead
    // of a confusing parse error further down.
    let preflight = std::process::Command::new(&ssh_cmd)
        .args([target, remote_bin, "--version"])
        .output()?;
    if !preflight.status.success() {
        return Err(MemoryError::Embedding(format!(
            "remote preflight failed: `{ssh_cmd} {target} {remote_bin} --version` exited with \
             {}. Confirm {remote_bin} is on the remote's PATH under non-interactive ssh, or \
             point at it explicitly with --remote-bin.\n{}",
            preflight.status,
            String::from_utf8_lossy(&preflight.stderr).trim()
        )));
    }

    let (pull_wm, push_wm) = db.get_sync_state(target)?;
    let now = chrono::Utc::now().timestamp();

    // Built before the pull half imports anything (see doc comment above).
    let push_payload = if !pull_only {
        let data = build_export_data(
            db,
            None,
            Some(push_wm),
            true,
            export::ExportScope::AllProjects,
            "",
            Some(target),
        )?;
        let count = data.memories.len();
        let max_updated_at = max_memory_updated_at(&data);
        let json = serde_json::to_string(&data)?;
        let bytes = json.len();
        Some((json, bytes, count, max_updated_at))
    } else {
        None
    };

    let pull_payload = if !push_only {
        let output = std::process::Command::new(&ssh_cmd)
            .args([
                target,
                remote_bin,
                "export",
                "--all-projects",
                "--embeddings",
                "--since",
                &pull_wm.to_string(),
            ])
            .output()?;
        if !output.status.success() {
            return Err(MemoryError::Embedding(format!(
                "remote export failed: `{ssh_cmd} {target} {remote_bin} export --all-projects \
                 --embeddings --since {pull_wm}` exited with {}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        let stdout = String::from_utf8(output.stdout).map_err(|e| {
            MemoryError::Embedding(format!("remote export produced invalid UTF-8: {e}"))
        })?;
        let data: export::ExportData = serde_json::from_str(&stdout)?;
        export::validate_import(&data).map_err(MemoryError::Embedding)?;
        // `export --all-projects` is what was requested above; a payload that comes back
        // scoped to one project (including an old --remote-bin whose response predates
        // `scope`, which `#[serde(default)]` would silently read as `Project`) would
        // otherwise re-home every memory under the empty `project_id` sent for
        // all-projects payloads, with no foreign key to stop it.
        if data.scope != export::ExportScope::AllProjects {
            return Err(MemoryError::Embedding(format!(
                "remote export returned scope `{:?}`, expected `all_projects`; refusing to \
                 re-home {} memories under an empty project id. Check --remote-bin.",
                data.scope,
                data.memories.len()
            )));
        }
        let count = data.memories.len();
        let max_updated_at = max_memory_updated_at(&data);
        Some((data, stdout.len(), count, max_updated_at))
    } else {
        None
    };

    if dry_run {
        if let Some((_, bytes, count, _)) = &push_payload {
            println!(
                "push: {count} memory(ies) staged, {bytes} bytes, since watermark {push_wm} \
                 (dry run, not sent)"
            );
        }
        if let Some((_, bytes, count, _)) = &pull_payload {
            println!(
                "pull: {count} memory(ies) available, {bytes} bytes, since watermark {pull_wm} \
                 (dry run, not imported)"
            );
        }
        return Ok(());
    }

    if let Some((data, _bytes, count, max_updated_at)) = pull_payload {
        let summary = import_export_data(
            db,
            "",
            embedding_service,
            data,
            export::ImportMode::Merge,
            Some(target),
        )?;
        println!(
            "pull: {} memories, {} updated, {} relationships ({} skipped) from {count} received",
            summary.imported, summary.updated, summary.rel_imported, summary.skipped
        );
        let new_pull_wm = match max_updated_at {
            Some(m) => pull_wm.max(m),
            None => pull_wm,
        };
        db.set_pull_watermark(target, new_pull_wm, now)?;
    }

    if let Some((json, _bytes, count, max_updated_at)) = push_payload {
        let mut child = std::process::Command::new(&ssh_cmd)
            .args([target, remote_bin, "import", "--mode", "merge", "-"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        // Write stdin on its own thread, concurrently with wait_with_output() draining
        // stdout/stderr below. A payload here can run ~10MB; writing it to completion
        // before touching the child's output pipes risks a classic deadlock if the child
        // (or ssh itself, e.g. a host-key banner) fills its stdout/stderr pipe buffer
        // before it has read all of stdin — parent blocks writing, child blocks writing
        // back, neither ever drains the other.
        let mut stdin = child.stdin.take().expect("piped stdin");
        let writer = std::thread::spawn(move || {
            use std::io::Write;
            stdin.write_all(json.as_bytes())
        });
        let output = child.wait_with_output()?;
        writer
            .join()
            .expect("stdin writer thread panicked")
            .map_err(MemoryError::Io)?;
        if !output.stdout.is_empty() {
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }
        if !output.stderr.is_empty() {
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
        }
        if !output.status.success() {
            return Err(MemoryError::Embedding(format!(
                "remote import failed: `{ssh_cmd} {target} {remote_bin} import --mode merge -` \
                 exited with {}; push watermark left unchanged\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        println!("push: sent {count} memories");
        let new_push_wm = match max_updated_at {
            Some(m) => push_wm.max(m),
            None => push_wm,
        };
        db.set_push_watermark(target, new_push_wm, now)?;
    }

    Ok(())
}

fn cmd_projects(db: &Database, current_project: &str, json: bool) -> Result<(), MemoryError> {
    let projects = db.list_projects()?;

    if json {
        let items: Vec<serde_json::Value> = projects
            .iter()
            .map(|project| {
                let mut value = serde_json::to_value(project).unwrap_or_default();
                if let Some(map) = value.as_object_mut() {
                    map.insert(
                        "current".to_string(),
                        serde_json::json!(project.id == current_project),
                    );
                }
                value
            })
            .collect();
        print_json(&serde_json::json!({
            "current_project": current_project,
            "count": items.len(),
            "projects": items,
        }));
        return Ok(());
    }

    if projects.is_empty() {
        println!("No projects in the memory store.");
        return Ok(());
    }

    println!("Projects ({}):\n", projects.len());
    for project in &projects {
        let marker = if project.id == current_project {
            " *"
        } else {
            ""
        };
        println!("{}{}", project.id, marker);
        println!(
            "  {} memories, {} handoffs, {} ADRs{}",
            project.memory_count,
            project.handoff_count,
            project.adr_count,
            project
                .latest_activity_at
                .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
                .map(|dt| format!(", last activity {}", dt.format("%Y-%m-%d")))
                .unwrap_or_default()
        );
    }
    println!("\n* = current project");

    Ok(())
}

fn cmd_stats(db: &Database, project_id: &str, json: bool) -> Result<(), MemoryError> {
    let stats = db.get_project_stats(project_id)?;

    let dead_count = db.count_dead(project_id)?;
    let trash_count = db.count_trash(project_id)?;

    if json {
        let mut value = serde_json::to_value(&stats)?;
        if let Some(map) = value.as_object_mut() {
            map.insert("project".to_string(), serde_json::json!(project_id));
            map.insert(
                "cluster_count".to_string(),
                serde_json::json!(db.get_clusters_for_project(project_id)?.len()),
            );
            map.insert("dead_count".to_string(), serde_json::json!(dead_count));
            map.insert("trash_count".to_string(), serde_json::json!(trash_count));
        }
        print_json(&value);
        return Ok(());
    }

    println!("Project: {}", project_id);
    println!("Memories: {}", stats.memory_count);
    println!("Relationships: {}", stats.relationship_count);
    println!("Avg relevance: {:.3}", stats.avg_relevance);
    println!("Handoffs: {}", stats.handoff_count);
    if dead_count > 0 {
        println!("Dead (excluded from retrieval): {}", dead_count);
    }
    if trash_count > 0 {
        println!("Recoverable in trash: {}", trash_count);
    }
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

/// Run the same maintenance pass as the server's background decay tick.
///
/// The server only ticks after a full interval of uptime, so a short-lived process never
/// runs any of this. Doing the whole pass here rather than the decay step alone is what
/// makes an external timer a complete substitute -- otherwise trash retention would be
/// reachable only from a server that happened to stay up for an hour.
fn cmd_decay(db: &Database, project_id: &str) -> Result<(), MemoryError> {
    let project = db
        .get_project(project_id)?
        .ok_or_else(|| MemoryError::NotFound(project_id.to_string()))?;

    let updated = db.update_relevance_scores(project_id, project.decay_rate)?;
    println!("Updated relevance scores for {} memories", updated);

    let pruned = db.auto_prune_stale_memories(project_id)?;
    if !pruned.is_empty() {
        println!(
            "Auto-pruned {} stale memories (recoverable from the trash)",
            pruned.len()
        );
    }

    let retention = db::trash_retention_days();
    let swept = db.sweep_trash(retention)?;
    if swept > 0 {
        println!(
            "Swept {} trash entries older than {} days",
            swept, retention
        );
    }

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
        let deleted = db.delete_memories_batch_with_op(&ids, db::OP_PRUNE)?;
        println!("Deleted {} memories", deleted);
        println!("Recoverable with `engram-cli restore <id>` until the trash is swept.");
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
                db.merge_memories(&keeper_id, old_id)?;
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
    json: bool,
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

    // Curation, matching the MCP `memory_context`: context is what an agent treats as
    // established background, so a superseded conclusion must not reach it unqualified.
    let curation = curation_view(db, project_id, false)?;
    let mut emitted: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut selected: Vec<(Memory, f32, Option<MatchedVia>)> = Vec::new();
    for (id, similarity) in &scored {
        if selected.len() >= limit {
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

        let (memory, matched_via) = match curation.resolve(&memory.id) {
            Resolution::Drop => continue,
            Resolution::Keep => (memory.clone(), None),
            Resolution::Redirect { successor_id, via } => {
                let Some(successor) = db.get_memory(&successor_id)? else {
                    continue;
                };
                if !type_filters.is_empty() && !type_filters.contains(&successor.memory_type) {
                    continue;
                }
                (successor, Some(via))
            }
        };
        if !emitted.insert(memory.id.clone()) {
            continue;
        }

        selected.push((memory, *similarity, matched_via));
    }

    if json {
        let memories: Vec<serde_json::Value> = selected
            .iter()
            .map(|(memory, similarity, matched_via)| {
                serde_json::json!({
                    "memory": memory,
                    "similarity": similarity,
                    "matched_via": matched_via,
                })
            })
            .collect();
        print_json(&serde_json::json!({
            "project": project_id,
            "context": context,
            "count": memories.len(),
            "memories": memories,
        }));
    } else {
        for (index, (memory, similarity, matched_via)) in selected.iter().enumerate() {
            if index > 0 {
                println!();
            }
            println!(
                "[{}] ({}, importance: {:.1}, similarity: {:.2})",
                memory.memory_type.as_str(),
                memory.id,
                memory.importance,
                similarity,
            );
            if let Some(via) = matched_via {
                println!("(replaces {})", via.superseded_id);
            }
            if let Some(ref summary) = memory.summary {
                println!("{}", summary);
            } else {
                println!("{}", memory.content);
            }
        }
    }

    // Record access
    let accessed_ids: Vec<String> = selected
        .iter()
        .map(|(memory, _, _)| memory.id.clone())
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

/// Whether stdin can carry an interactive answer. `--non-interactive` and a
/// non-terminal stdin (CI job, agent with a closed pipe) both mean "do not ask".
fn is_interactive(non_interactive: bool) -> bool {
    use std::io::IsTerminal;
    !non_interactive && std::io::stdin().is_terminal()
}

/// Prompt the user for a line of text on stdin. Returns empty string on EOF or blank.
fn prompt_line(label: &str) -> String {
    use std::io::{self, BufRead, Write};
    print!("{}: ", label);
    io::stdout().flush().ok();
    let stdin = io::stdin();
    let mut line = String::new();
    // EOF leaves the prompt unterminated; close the line so following output
    // does not render inside it.
    if matches!(stdin.lock().read_line(&mut line), Ok(0) | Err(_)) {
        println!();
    }
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
        if matches!(stdin.lock().read_line(&mut line), Ok(0) | Err(_)) {
            println!();
            break;
        }
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            println!();
            break;
        }
        items.push(trimmed);
    }
    items
}

/// Dispatch handoff subcommands.
#[allow(clippy::too_many_arguments)]
/// Handle `engram-cli todo <subcommand>`.
fn cmd_todo(
    db: &Database,
    project_id: &str,
    embedding_service: Option<&EmbeddingService>,
    current_branch: Option<&str>,
    cmd: TodoCmd,
    json: bool,
) -> Result<(), MemoryError> {
    let embedding = || -> Result<&EmbeddingService, MemoryError> {
        embedding_service
            .ok_or_else(|| MemoryError::InvalidType("embedding service required".to_string()))
    };

    // Every mutation goes through the same `write_todos` batch path the MCP tool uses, so
    // the two cannot disagree about validation or duplicate reporting.
    let op = match cmd {
        TodoCmd::List {
            status,
            branch_mode,
            limit,
        } => {
            let status_filter = match status.as_str() {
                "all" => None,
                other => Some(other.parse().map_err(|_| {
                    MemoryError::InvalidType(format!(
                        "unknown todo status '{other}'; expected open, done, dropped, or all"
                    ))
                })?),
            };
            let branch_filter = match branch_mode.as_str() {
                "all" => None,
                "project" => Some(None),
                "current" => current_branch.map(Some),
                literal => Some(Some(literal)),
            };
            let result = tools::list_todos(db, project_id, status_filter, branch_filter, limit)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            if result.todos.is_empty() {
                println!("No todos matched.");
            } else {
                for todo in &result.todos {
                    println!("{}", format_todo(todo));
                }
            }
            println!(
                "\n{} open, {} done, {} dropped",
                result.open_count, result.done_count, result.dropped_count
            );
            return Ok(());
        }
        TodoCmd::Add {
            text,
            branch,
            tags,
            importance,
        } => TodoOp::Add {
            text,
            branch,
            tags,
            importance,
        },
        TodoCmd::Done { id } => TodoOp::Done { id },
        TodoCmd::Drop { id, reason } => TodoOp::Drop { id, reason },
        TodoCmd::Reopen { id } => TodoOp::Reopen { id },
        TodoCmd::Edit { id, text } => TodoOp::Edit { id, text },
    };

    let result = tools::write_todos(db, embedding()?, project_id, current_branch, vec![op])?;

    for r in &result.results {
        match &r.error {
            Some(err) => {
                eprintln!("{} failed: {}", r.op, err);
                std::process::exit(1);
            }
            None => println!("{} ok: {}", r.op, r.id),
        }
        for dup in &r.possible_duplicates {
            println!(
                "  possible duplicate: {} ({:.0}%) {}",
                dup.id,
                dup.similarity * 100.0,
                dup.text
            );
        }
    }
    println!("{} open todo(s).", result.open_count);
    Ok(())
}

fn cmd_handoff(
    db: &Database,
    project_id: &str,
    embedding_service: Option<&EmbeddingService>,
    current_branch: Option<&str>,
    cmd: HandoffCmd,
    json: bool,
) -> Result<(), MemoryError> {
    match cmd {
        HandoffCmd::Create {
            summary,
            decisions,
            blockers,
            tried,
            mental_model,
            next_steps,
            notes,
            branch,
            continues_from,
            importance,
            no_pin,
            no_auto_link,
            from_file,
            non_interactive,
        } => {
            let embedding = embedding_service.ok_or_else(|| {
                MemoryError::InvalidType("embedding service required".to_string())
            })?;

            let interactive = is_interactive(non_interactive);

            let sections = if let Some(path) = from_file {
                let content = std::fs::read_to_string(&path)?;
                let mut s = HandoffSections::parse_markdown(&content)?;
                if continues_from.is_some() {
                    s.continues_from = continues_from.clone();
                }
                s
            } else {
                // Use flags if provided; prompt for the rest only when stdin can answer.
                let summary_text = match summary {
                    Some(s) => s,
                    None if interactive => {
                        let s = prompt_line("Summary");
                        if s.is_empty() {
                            return Err(MemoryError::InvalidType(
                                "handoff: summary is required".to_string(),
                            ));
                        }
                        s
                    }
                    None => {
                        return Err(MemoryError::InvalidType(
                            "handoff: summary is required (pass --summary or --from-file)"
                                .to_string(),
                        ));
                    }
                };

                let decisions_list = if !decisions.is_empty() || !interactive {
                    decisions
                } else {
                    prompt_list("Decisions")
                };

                let blockers_list = if !blockers.is_empty() || !interactive {
                    blockers
                } else {
                    prompt_list("Blockers")
                };

                let tried_list = if !tried.is_empty() || !interactive {
                    tried
                } else {
                    prompt_list("Tried (approaches that failed, and why)")
                };

                let mental_model_text = match mental_model {
                    Some(m) => m,
                    None if interactive => prompt_line("Mental model"),
                    None => String::new(),
                };

                let next_steps_list = if !next_steps.is_empty() || !interactive {
                    next_steps
                } else {
                    prompt_list("Next steps")
                };

                let notes_text = match notes {
                    Some(n) => Some(n),
                    None if interactive => {
                        let n = prompt_line("Notes (optional, blank to skip)");
                        if n.is_empty() { None } else { Some(n) }
                    }
                    None => None,
                };

                HandoffSections {
                    summary: summary_text,
                    decisions: decisions_list,
                    // Open work lives in the durable todo list (`engram-cli todo`), which
                    // handoff resume reads directly.
                    todos: vec![],
                    blockers: blockers_list,
                    tried: tried_list,
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
            println!("Project: {} | Branch: {}", result.project, result.branch);
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

            if json {
                print_json(&serde_json::to_value(&result)?);
                return Ok(());
            }

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

            if !result.open_todos.is_empty() {
                println!("\nOpen todos:");
                for todo in &result.open_todos {
                    println!("  - [ ] {}", todo);
                }
            }

            if !result.open_blockers.is_empty() {
                println!("\nOpen blockers:");
                for blocker in &result.open_blockers {
                    println!("  - {}", blocker);
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

            if json {
                print_json(&serde_json::json!({
                    "query": query,
                    "count": result.matches.len(),
                    "matches": result.matches,
                }));
                return Ok(());
            }

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

            if json {
                let sections = db.get_handoff_sections(&id)?.map(|(sections, _)| sections);
                print_json(&serde_json::json!({"memory": memory, "sections": sections}));
                return Ok(());
            }

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
    json: bool,
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
            non_interactive,
        } => {
            let embedding = embedding_service.ok_or_else(|| {
                MemoryError::InvalidType("embedding service required".to_string())
            })?;

            let interactive = is_interactive(non_interactive);

            let sections = if let Some(path) = from_file {
                let content = std::fs::read_to_string(&path)?;
                AdrSections::parse_markdown(&content)?
            } else {
                let title_text = match title {
                    Some(t) => t,
                    None if interactive => {
                        let t = prompt_line("Title");
                        if t.is_empty() {
                            return Err(MemoryError::InvalidType(
                                "adr: title is required".to_string(),
                            ));
                        }
                        t
                    }
                    None => {
                        return Err(MemoryError::InvalidType(
                            "adr: title is required (pass --title or --from-file)".to_string(),
                        ));
                    }
                };

                let context_text = match context {
                    Some(c) => c,
                    None if interactive => prompt_line("Context"),
                    None => String::new(),
                };

                let decision_text = match decision {
                    Some(d) => d,
                    None if interactive => {
                        let d = prompt_line("Decision");
                        if d.is_empty() {
                            return Err(MemoryError::InvalidType(
                                "adr: decision is required".to_string(),
                            ));
                        }
                        d
                    }
                    None => {
                        return Err(MemoryError::InvalidType(
                            "adr: decision is required (pass --decision or --from-file)"
                                .to_string(),
                        ));
                    }
                };

                let consequences_text = match consequences {
                    Some(c) => c,
                    None if interactive => prompt_line("Consequences"),
                    None => String::new(),
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

            println!("Project: {}", result.project);
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

            if json {
                let items: Vec<serde_json::Value> = rows
                    .iter()
                    .map(|(number, adr_status, title, id)| {
                        serde_json::json!({
                            "number": number,
                            "status": adr_status.as_str(),
                            "title": title,
                            "id": id,
                        })
                    })
                    .collect();
                print_json(&serde_json::json!({
                    "project": project_id,
                    "count": items.len(),
                    "adrs": items,
                }));
                return Ok(());
            }

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

            if json {
                print_json(&serde_json::json!({
                    "id": id,
                    "number": num,
                    "status": adr_status.as_str(),
                    "title": sections.title,
                    "context": sections.context,
                    "decision": sections.decision,
                    "consequences": sections.consequences,
                }));
                return Ok(());
            }

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
