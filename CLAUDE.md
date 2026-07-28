# Engram MCP

MCP server for AI agent persistent memory. Crate name: `engram_mcp` (binaries: `engram`, `engram-cli`). SQLite + local embeddings (mdbr-leaf-ir q8, 256-dim MRL).

## Development Rules
- **Always run `cargo clippy` before committing and fix all warnings**
- Run `cargo test` to verify changes
- Use `cargo build --release` for production binaries

## Structure
```
src/
  main.rs      - MCP server entry, stdio transport, resources, prompts
  cli.rs       - CLI binary (engram-cli)
  lib.rs       - module exports
  db.rs        - SQLite ops (memories, embeddings, relationships, batch ops)
  memory.rs    - Memory, MemoryType, MergeSource, MemoryCluster, Relationship, RelationType, ProjectStats
  embedding.rs - mdbr-leaf-ir ONNX wrapper (256-dim MRL vectors)
  decay.rs     - relevance decay algorithm
  tools.rs     - MCP tool handlers + dedup + clustering
  format.rs    - human-readable output formatting for MCP results
  summarize.rs - extractive summarization for large content
  export.rs    - import/export JSON format
  error.rs     - MemoryError enum
  hooks/       - Claude Code lifecycle hook handlers: dispatch, filter, redact, install, payload structs
```

## Key Types
- `MemoryType`: fact, decision, preference, pattern, debug, entity, handoff, adr
- `RelationType`: relates_to, supersedes, derived_from
- `Memory`: id, project_id, content, tags, importance, relevance_score, timestamps, branch, merged_from
- `MergeSource`: id, content_preview, merged_at (provenance tracking for dedup merges)
- `MemoryCluster`: id, project_id, summary, member_count, centroid, timestamps
- `MemoryError`: Database, Json, Embedding, NotFound, InvalidType, InvalidRelation, Io

## MCP Capabilities
### Tools
- `memory_store` - store memory + embedding, auto-summarize, auto-dedup (0.90+), auto-cluster
- `memory_query` - semantic search with pagination, branch filtering
- `memory_update` - update content/tags/importance, regenerate summary
- `memory_delete` - remove memory and relationships
- `memory_link` - create relationship between memories
- `memory_graph` - traverse relationship graph
- `memory_store_batch` - store up to 100 memories atomically
- `memory_delete_batch` - delete multiple memories by ID
- `memory_export` - export project to JSON (optional embeddings)
- `memory_import` - import from JSON (merge/replace modes)
- `memory_stats` - get project statistics (includes cluster count and adr_count)
- `memory_context` - get relevant memories for context (hierarchical retrieval via clusters, flat fallback)
- `memory_prune` - remove low-relevance memories (dry run by default)
- `memory_dedup` - find and merge duplicate memories (dry run by default, threshold configurable)
- `handoff_create` - create a session handoff with structured sections + per-section embeddings, pinned by default
- `handoff_resume` - retrieve top sections from recent handoffs on a branch, with linked memories
- `handoff_search` - search handoff sections by content, with optional branch and section filters
- `adr_create` - create a numbered ADR (Nygard-style), optionally superseding an existing ADR
- `adr_update_status` - advance an ADR through its lifecycle (proposed/accepted/deprecated/rejected; superseded only via adr_create)
- `adr_list` - list all ADRs for the project, optionally filtered by status
- `adr_show` - retrieve full details of a single ADR by number
- `adr_export` - export ADRs to Nygard-style Markdown files on disk (dry-run by default)

### Resources
- `memory://{project}/{id}` - read individual memories

### Prompts
- `recall_context` - retrieve relevant memories for a context

## CLI (engram-cli)
```bash
engram-cli query "search text"     # semantic search
engram-cli list                    # list all memories
engram-cli show <id>               # show specific memory
engram-cli store "content" -t fact # store new memory
engram-cli store "content" -t preference --pinned --global  # pinned global memory
engram-cli delete <id>             # delete memory
engram-cli update <id> -c "new"    # update memory
engram-cli link <src> <tgt> -r relates_to  # link memories
engram-cli export -o backup.json   # export to file
engram-cli import backup.json      # import from file
engram-cli context "auth refactor"  # load relevant context
engram-cli context "auth refactor" --global  # include global memories in context
engram-cli stats                   # show statistics
engram-cli decay                   # run decay manually
engram-cli prune -t 0.2 --confirm  # remove low-relevance
engram-cli dedup -t 0.90           # find duplicates (dry run)
engram-cli dedup -t 0.90 --confirm # merge duplicates
engram-cli wipe                    # show what would be wiped
engram-cli wipe --confirm          # delete all project memories
engram-cli pin <id>                # pin a memory (exempt from decay/prune)
engram-cli unpin <id>              # unpin a memory
engram-cli insights                # show memory health insights
engram-cli health                  # check memory store health
engram-cli handoff create          # interactive handoff capture (prompts per section)
engram-cli handoff create --from-file session.md  # ingest pre-written markdown handoff
engram-cli handoff resume [--branch X] [--query Q] [--max N]  # load context from recent handoffs
engram-cli handoff search <query> [--branch X] [--section blockers,todos] [--limit N]  # search sections
engram-cli hook-event <Event>              # process a Claude Code lifecycle hook event (reads stdin JSON)
engram-cli hooks install/uninstall/status  # manage Claude Code settings.json integration
engram-cli adr create                      # create a new ADR interactively
engram-cli adr update-status <N> <status>  # advance ADR lifecycle status
engram-cli adr list [--status <s>]         # list ADRs (optionally filtered by status)
engram-cli adr show <N>                    # show full details of ADR number N
engram-cli adr export [--write] [--number N] [--dir D]  # export ADRs to Markdown files
```

## Features
- Background decay job (hourly, configurable via ENGRAM_DECAY_INTERVAL)
- Background re-clustering job (6-hourly, configurable via ENGRAM_RECLUSTER_INTERVAL)
- Semantic deduplication on store (similarity >= 0.90, same type: auto-merge with provenance)
- Hierarchical memory clustering (auto-assign to clusters, centroid-based retrieval in memory_context)
- Branch-aware queries (branch_mode: "current", "global", "all", or specific branch name)
- Access tracking for memory reinforcement
- Auto-summarization for content > 500 chars
- Batch operations with transactions
- Query pagination and empty-query optimization
- Human-readable formatted output (markdown) + JSON in collapsible block
- Claude Code lifecycle hooks: passive capture via `hook-event` subcommand + `hooks install` one-liner. Managed events: `UserPromptSubmit, SubagentStop, SessionEnd` (Stop/PreCompact/PostToolUse are explicit no-ops; tool-call outcomes are never captured — low-signal noise). `SessionEnd` reads `transcript_path` and stores the last assistant message as a `session_summary` Fact. All hook stores route through `store_with_dedup`, so near-duplicate captures are silently skipped. Per-project daily cap on hook captures prevents runaway logging. Hook importance is clamped to `0.5` regardless of `ENGRAM_HOOK_MIN_IMPORTANCE`.

## ADRs
Architecture Decision Records, stored as `MemoryType::Adr` memories with an `adr_sections` sidecar (title, context, decision, consequences).

- **Numbering**: per-project integers starting at 1, allocated inside a transaction as `MAX(existing)+1`. Numbers are monotonic; deleting any ADR can leave gaps. Deleting the current highest-numbered ADR lets that number be reused on the next create.
- **Status lifecycle**: proposed → accepted → deprecated or superseded; proposed → rejected; rejected → proposed (retry); deprecated → accepted (reinstate); superseded is a terminal state. Transition rules are enforced at the tool layer; the DB accepts any value.
- **Supersession**: set only via `adr_create --supersedes N` (or the `supersedes` field on the MCP tool). `adr_update_status` rejects `"superseded"` directly; the caller must create a new ADR. Supersession flips the old ADR to `Superseded` and creates a `Supersedes` relationship edge from the new ADR to the old one.
- **Project-global**: ADRs have no branch scope (`branch = NULL`). They are visible in all branch-mode queries.
- **Bypasses dedup and clustering**: ADRs are stored via `store_adr_atomic`, which skips the deduplication and cluster-assignment paths used by `memory_store`.
- **Pinned by default**: `pinned = true` makes ADRs exempt from relevance decay and `memory_prune`.
- **File export**: `adr_export` renders Nygard-style Markdown (`docs/adr/NNNN-slug.md` by default). Default is dry-run; set `dry_run: false` (CLI: `--write`) to write files. Existing files are overwritten silently. Target directory precedence: explicit `dir` argument → `ENGRAM_ADR_DIR` env var → `docs/adr`.
- **`memory_stats` includes `adr_count`**: the total number of ADR sidecar rows for the project.

## Handoffs
Section-based session capture (`summary, decisions, todos, blockers, mental_model, next_steps, notes`). Handoffs are branch-aware `MemoryType::Handoff` memories, pinned by default, with a `handoff_sections` sidecar table holding per-section embeddings (256-dim f32 LE, prefix-free). Branch chaining via `continues_from` lives in the sidecar only (not a graph edge). Auto-links to `decision/pattern/debug` memories at cosine similarity >= 0.75 (cap 10 links). Bypasses dedup entirely. Two MCP prompts: `handoff` (capture) and `resume` (restore). CLI: `engram-cli handoff create/resume/search`.

Section semantics: **todos** — Within-session work the next agent should pick up immediately. Concrete, ready-to-execute items. **blockers** — Things preventing forward motion right now (missing access, failing dependency, unanswered question). **next_steps** — Post-session follow-ups beyond the current thread. Future-facing, not for immediate pickup.

## Config (env vars)
- `ENGRAM_DB` - SQLite path (default: ~/.local/share/engram/memories.db)
- `ENGRAM_PROJECT` - project scope (default: cwd name)
- `ENGRAM_DECAY_INTERVAL` - decay job interval in seconds (default: 3600)
- `ENGRAM_RECLUSTER_INTERVAL` - re-clustering job interval in seconds (default: 21600)
- `ENGRAM_MAX_CANDIDATES` - max candidate memories to score during search (default: 200)
- `ENGRAM_HOOK_DEDUP_SKIP` - similarity threshold above which hook captures are silently dropped (default: 0.95, clamped to [0.5, 1.0])
- `ENGRAM_HOOK_DAILY_CAP` - max hook-captured memories per project per UTC day; `0` = unlimited (default: 50)
- `ENGRAM_HOOK_MIN_IMPORTANCE` - importance floor for hook captures (default: 0.5; values above 0.5 have no effect because dispatch caps importance at 0.5)
- `ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED` - opt-in flag for the `UserPromptSubmit` hook (default off; even when on, captures require an explicit `#remember` cue)
- `ENGRAM_ADR_DIR` - output directory for `adr_export` file writes (default: `docs/adr` relative to server cwd). Overridden by the `dir` argument on the tool/CLI call.
- `ENGRAM_MCP_TOOL_PROFILE` - advertised MCP tool surface: `full` (23 tools, default), `core` (14; includes adr_create/adr_show/adr_list), or `minimal` (3: memory_context, memory_store, handoff_resume). `adr_update_status` and `adr_export` are full-only. Dispatch stays permissive — non-advertised tools still execute with a one-time `[engram]` warning per process.

## Commands
```bash
cargo build --release    # binaries: target/release/engram, target/release/engram-cli
cargo test               # run all tests
cargo clippy             # lint (must pass with no warnings)
```

## Memory
Engram MCP available. Store decisions/patterns, query before architectural changes.
