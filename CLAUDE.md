# Engram MCP

MCP server for AI agent persistent memory. Crate name: `engram_mcp` (binaries: `engram`, `engram-cli`). SQLite + local embeddings (mdbr-leaf-ir q8, 256-dim MRL).

`engram` is the MCP server only: JSON-RPC over stdio, no subcommands. It serves when stdin is a pipe (or with `--stdio`/`--serve`); any argument or a terminal on stdin prints a usage notice pointing at `engram-cli` instead (exit 0 for `--help`, 2 for an unsupported argument). All human/agent CLI use goes through `engram-cli`.

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
  memory.rs    - Memory, MemoryType, MergeSource, MemoryCluster, Relationship, RelationType, ProjectStats, ProjectSummary
  embedding.rs - mdbr-leaf-ir ONNX wrapper (256-dim MRL vectors)
  decay.rs     - relevance decay algorithm (single source of truth; registered as the RELEVANCE() SQL function)
  tools.rs     - MCP tool handlers + dedup + clustering
  format.rs    - human-readable output formatting for MCP results
  summarize.rs - extractive summarization for large content
  export.rs    - import/export JSON format
  error.rs     - MemoryError enum
  hooks/       - Claude Code lifecycle hook handlers: dispatch, filter, redact, install, payload structs
  db/status.rs   - retrieval status: supersession map (from edges) + dead flags
  db/trash.rs    - recoverable snapshots for every destructive operation
  tools/curation.rs - applying supersession/dead status to a ranked result set
```

## Key Types
- `MemoryType`: fact, decision, preference, pattern, debug, entity, handoff, adr
- `RelationType`: relates_to, supersedes, derived_from
- `Memory`: id, project_id, content, tags, importance, relevance_score, timestamps, branch, merged_from
- `MergeSource`: id, content_preview, content (full text of the consumed memory), merged_at
- `MemoryCluster`: id, project_id, summary, member_count, centroid, timestamps
- `MemoryError`: Database, Json, Embedding, NotFound, InvalidType, InvalidRelation, UnknownTool, UnknownProject, InvalidArguments, Io

## MCP Capabilities
### Tools
- `memory_store` - store memory + embedding, auto-summarize, auto-dedup (0.90+), auto-cluster; `supersedes` replaces existing memories, result reports `possible_supersedes`
- `memory_query` - semantic search with pagination, branch filtering
- `memory_update` - update content/tags/importance/dead, regenerate summary; returns the previous version
- `memory_delete` - remove memory and relationships; returns what it deleted, recoverable from the trash
- `memory_link` - create relationship between memories
- `memory_graph` - traverse relationship graph
- `memory_store_batch` - store up to 100 memories atomically
- `memory_delete_batch` - delete multiple memories by ID
- `memory_export` - export project to JSON (optional embeddings)
- `memory_import` - import from JSON (merge/replace modes)
- `memory_stats` - get project statistics (includes cluster count, adr_count, dead_count, trash_count)
- `memory_projects` - list every project in the store with memory/handoff/ADR counts
- `memory_context` - get relevant memories for context (hierarchical retrieval via clusters, flat fallback)
- `memory_list` - enumerate memories without a query; filters by type/tag/status, orders by relevance/created/updated/accessed
- `memory_trash` - list recoverable snapshots of destroyed memories
- `memory_restore` - restore a memory from the trash with its embedding and relationships
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

All tools accept an optional `project` argument that overrides the server's own project for that call; see **Project scoping** below.

### Resources
- `memory://{project}/{id}` - read individual memories from any project in the store (the URI's project must match the memory's owner)

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
engram-cli projects                # list all projects in the store
engram-cli -p <project> <cmd>      # scope any command to another project
engram-cli --current-branch feat/x <cmd>  # override the detected branch (worktrees, wrong cwd)
engram-cli --json <read cmd>       # machine-readable JSON instead of text
engram-cli decay                   # run decay manually
engram-cli prune -t 0.2 --confirm  # remove low-relevance
engram-cli dedup -t 0.90           # find duplicates (dry run)
engram-cli dedup -t 0.90 --confirm # merge duplicates
engram-cli wipe                    # show what would be wiped
engram-cli wipe --confirm          # delete all project memories
engram-cli list --status superseded # show what retrieval hides (live|superseded|dead|all)
engram-cli list --order created    # relevance (default) | created | updated | accessed
engram-cli store "..." --supersedes <id>  # replace an existing memory
engram-cli update <id> --dead --dead-reason "service retired"  # exclude from retrieval
engram-cli update <id> --alive     # undo --dead
engram-cli trash                   # list recoverable snapshots
engram-cli restore <id>            # restore a memory (or --trash-id <n>)
engram-cli pin <id>                # pin a memory (exempt from decay/prune)
engram-cli unpin <id>              # unpin a memory
engram-cli insights                # show memory health insights
engram-cli health                  # check memory store health
engram-cli handoff create          # interactive handoff capture (prompts per section)
engram-cli handoff create --non-interactive --summary "..."  # never prompt; missing sections stay empty
engram-cli handoff create --tried "bulk UPDATE, locked the table for 90s"  # record a dead end
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
- Claude Code lifecycle hooks: passive capture via `hook-event` subcommand + `hooks install` one-liner. `hooks status`/`uninstall` detect a block by its `_source` marker **or** by the command being an `engram-cli hook-event` invocation; marker-only matching made status report a false negative and uninstall a no-op for hand-written blocks. Managed events: `UserPromptSubmit, SubagentStop, SessionEnd` (Stop/PreCompact/PostToolUse are explicit no-ops; tool-call outcomes are never captured — low-signal noise). `SessionEnd` reads `transcript_path` and stores the last assistant message as a `session_summary` Fact. All hook stores route through `store_with_dedup`, so near-duplicate captures are silently skipped. Per-project daily cap on hook captures prevents runaway logging. Hook importance is clamped to `0.5` regardless of `ENGRAM_HOOK_MIN_IMPORTANCE`.

## Tool arguments
All MCP tool arguments are deserialized through `parse_args` (`src/tools/handler.rs`), which reports the tool name and the field names actually received on failure. The input structs ignore unknown fields, so a misnamed field would otherwise surface as a bare `missing field \`x\`` that reads like the server dropped a field the caller did send. Field order never matters — arguments arrive as a JSON object.

`memory_store` (and each item of `memory_store_batch`) accepts `memory_type` as an alias for `type`.

Store results report where they landed: `memory_store`/`memory_store_batch` include `project`, `handoff_create` includes `project` and `branch`, and `adr_create` includes `project`.

## JSON output
The global `--json` flag makes a read command print one JSON document on stdout. Supported: `query`, `context`, `stats`, `projects`, `list`, `show`, `handoff resume/search/show`, `adr list/show` (see `supports_json` in `src/cli.rs`). Unsupported commands exit 2 with a message instead of ignoring the flag, so a caller never parses prose by accident. Empty results still render as JSON with `count: 0`. Memory objects are the `Memory` serde shape, identical to export/MCP payloads.

## Interactive prompts
`engram-cli handoff create` and `adr create` prompt only when stdin is a terminal and `--non-interactive` was not passed. Otherwise missing optional sections stay empty and missing required ones (`--summary`; `--title`/`--decision` for ADRs) are errors naming the flag to pass. This keeps CI jobs and agents with an open stdin from blocking on a prompt.

## Branch scoping

The current branch is detected from the working directory's git state, which is the directory the process was started in — not necessarily where the work is happening. A Claude session launched in the main clone while work proceeds in a worktree would otherwise file every handoff and branch-tagged memory under the main clone's branch.

Three overrides, in precedence order:

- **Per-call `branch` argument** on `memory_store`, `handoff_create`, `handoff_resume`, `handoff_search` (CLI: the per-command `--branch` flags). Correct when the branch varies call to call; this is what the `handoff` and `read-handoffs` skills pass.
- **`engram-cli --current-branch <name>`**, a global flag that sets what "current branch" resolves to for every subcommand, including `--branch auto` on `store`. Distinct from the per-command `--branch` flags so the two compose rather than shadow each other.
- **`ENGRAM_BRANCH`**, the same override as an env var. The right choice for a persistently mismatched setup, since it applies to the MCP server too — the server takes no arguments.

Worktrees are detected correctly on their own: `.git` is a file rather than a directory there, so the fast `.git/HEAD` read fails and detection falls back to `git rev-parse`. The problem is only ever *which directory* the process is reading from.

## Project scoping
Every MCP tool takes an optional `project` argument; omitted, it resolves to the server's own project (`ENGRAM_PROJECT` or cwd). `engram-cli` has the equivalent global `-p/--project` flag.

- **Unknown projects are rejected**, not created: `resolve_project` returns `MemoryError::UnknownProject` listing the known project IDs. A project counts as known if it has a `projects` row or owns at least one memory. `engram-cli` applies the same check for read-only commands when `-p` is explicit; write commands still create the project via `get_or_create_project`.
- **Branch handling for foreign projects**: branch names belong to a repository, so the server's current branch is not used outside its own project. `branch_mode: "current"` widens to all branches, `branch: "auto"` on `memory_store` resolves to global, and `handoff_create` for another project requires an explicit `branch`.
- **`memory_projects`** lists project IDs with memory/handoff/ADR counts and last activity, ordered by most recent activity. It is the discovery path for the `project` argument and takes no arguments itself.

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

## Curation: supersession, dead, trash

Retrieval status is kept outside the memory row, alongside clusters and relationships, so the `Memory` export shape is unchanged.

- **Superseded** is derived from the existing `Supersedes` relationship edge (`src/db/status.rs`), never duplicated onto the memory. Set it with `supersedes: [id]` on `memory_store` (CLI: `store --supersedes`), or by creating the edge directly with `memory_link`.
- **Retrieval redirects, it does not drop.** A superseded memory is suppressed and *its successor is returned at the superseded memory's rank*, annotated with `matched_via: {superseded_id, superseded_preview}`. A successor reached both on its own merit and by redirect appears once. Chains are followed to the terminal successor, depth-capped at 5 with a cycle guard (a revert makes A→B→A reachable). Returning nothing instead would read as "nobody investigated this" and invites the stale conclusion to be re-derived.
- **Dead** (`memory_status.dead`) means the subject is gone with no successor, so there is nothing to redirect to; those are excluded outright. Set via `memory_update dead: true` (CLI: `update --dead`). A chain ending in a dead memory is also dropped.
- **`include_superseded: true`** on `memory_query`, and `memory_list status=superseded|dead|all`, show what retrieval hides. Curation needs to see the suppressed set.
- **Redirect targets must satisfy the caller's type filter** on their own; a successor of a different type is dropped rather than smuggled past the filter.

### Trash

Every destructive path snapshots the memory to `memory_trash` in the same transaction as the destruction: `memory_delete`, `memory_delete_batch`, `memory_prune`, `auto_prune_stale_memories` (the unattended one), `delete_project_data` (wipe), dedup merges, and content-replacing updates. Snapshots carry the full `Memory` plus every edge touching it, and the embedding blob, so `memory_restore` puts a memory back searchable and reconnected. Edges whose other end is also gone are skipped and counted in the result.

Results say what they destroyed: `memory_update` returns `previous`, `memory_delete` returns `deleted`, `memory_delete_batch` returns `memories`. `MergeSource` carries the consumed memory's **full content**, not only `content_preview` — a merge deletes the only other copy.

Retention is `ENGRAM_TRASH_RETENTION_DAYS` (default 30), swept on the decay tick.


### Dedup exemptions

Auto-merge already requires matching `memory_type`, so a `debug` finding and the `pattern` lesson derived from it can never collapse. On top of that: `find_duplicates` refuses candidates the caller passed in `related_to`/`supersedes`, and refuses any candidate that is already a merge composite (non-empty `merged_from`) — merging into a memory that has already absorbed others is how claims with different lifetimes end up in one record. The `memory_dedup` scan additionally skips pairs joined by a `derived_from` or `supersedes` edge.

A refused pair still shows up in `possible_supersedes`, so the caller is asked rather than left with two contradicting memories.

`engram-cli store` runs the same `store_with_dedup_exempting` path as the MCP tool.

### Supersession candidates

`memory_store` returns `possible_supersedes` (CLI: printed after `store`): existing same-type memories at similarity >= `SUPERSESSION_CANDIDATE_MIN` (0.75), top 5, excluding the new memory and anything it just merged with. There is deliberately no upper bound — a pair can sit above the dedup threshold and still not merge, because dedup refuses composites and caller-exempted memories, and those are the *most* likely supersessions. Never applied automatically: cosine cannot separate "contradicts" from "elaborates".

The similarities are computed for dedup anyway. Suppressed when the caller already passed `supersedes`.

## Relevance decay

`crate::decay::relevance_from_parts` is the only definition of the algorithm. It is registered as the `RELEVANCE()` SQLite scalar function on every connection, and `update_relevance_scores` calls it rather than reimplementing the formula in SQL, so the two cannot drift.

```
score = clamp(0.1, 1.0, exp(-decay_rate·elapsed)·(0.5 + importance·0.5) + usage_boost)
usage_boost = 0.1 · min(1, ln(1+access_count)/ln(1+50)) · exp(-0.02·elapsed)
```

**`elapsed` is store-days, not calendar days** (`src/db/activity.rs`). A store-day is a day on which the memory's project received a store; `elapsed` counts those recorded after the memory was last accessed.

Wall-clock decay is wrong for this. A project untouched for two years is exactly as you left it, and its memories are no less current than the day you stopped — but calendar decay drives them all to the floor, so returning to a project is when its knowledge is worth least. What makes a memory stale is *newer knowledge about the same project arriving to displace it*. With a store-day clock a dormant project freezes, and an active one ages at its own pace regardless of the calendar.

- **The unit is a day, not a store.** Counting raw stores would let one heavy session (100 memories across 8 days is real here) age everything else at once, and would put `decay_rate` on an uninterpretable scale. Per store-day keeps 0.01 readable as a half-life of ~69 active days.
- **Hook captures do not advance the clock** (`CLOCK_ADVANCING_STORE`). They are stored without anyone deciding to; a session where automatic capture was the only thing that happened has not made curated knowledge staler. This matters: in one project here 39 total activity-days were only 16 once hook captures were excluded.
- **Ranking uses the same axis.** `compute_hybrid_score` and `compute_context_score` take elapsed store-days too. Measuring recency on the wall clock in the scorers would re-import exactly the behaviour decay was moved off.
- **The clock is derived, not stored.** It is computed from `created_at` on existing rows, so there is no counter to migrate or keep in sync. Deleting memories can retract store-days, which raises the relevance of what remains — correct, since displacement is by memories that still exist.

The usage boost is bounded (`USAGE_BOOST_MAX` = 0.1, one step on the documented importance scale), saturates at 50 accesses, and fades on the same clock. Both clamps apply. Decay skips pinned rows, so a pinned memory holds at 1.0; the ceiling is what keeps a heavily-retrieved unpinned memory from outranking it.

`memory_query` ranks by `(base + tag_boost) · relevance_score`, so this column is a direct ranking multiplier. `memory_context` ranks by `compute_hybrid_score`/`compute_context_score` and uses `relevance_score` only as a `min_score` gate.

## Background maintenance

The MCP server spawns two jobs: decay (`ENGRAM_DECAY_INTERVAL`, default 3600s) and re-clustering (`ENGRAM_RECLUSTER_INTERVAL`, default 21600s). The decay tick runs `update_relevance_scores`, then `auto_prune_stale_memories`, then `sweep_trash`.

**Both loops sleep before their first pass**, so a server that lives less than one interval never runs any of it. When the server is started per editor session, maintenance effectively depends on session length. `engram-cli decay` runs the full decay-tick pass (scores, auto-prune, trash sweep), so an external timer is a complete substitute; it is not a decay-only command.

**Decay runs at store time**, not on a schedule. Relevance is a function of store-days, so it changes at exactly one moment: when a store advances the project's clock. `memory_store`, `memory_store_batch`, and `engram-cli store` recompute the project's scores after a successful store (best-effort — the memory is already committed, so a scoring failure must not report the store as failed). A periodic timer is not recommended: it would recompute values that cannot have changed.

The background job still has work a store cannot do. `sweep_trash` deletes on `trashed_at < now - retention_days` and `auto_prune_stale_memories` gates on `created_at`, both genuinely wall-clock. It also acts as a safety net for the decay step.

A consequence worth knowing: a freshly stored memory's relevance is now its importance factor (`0.5 + importance/2`), not a flat 1.0. Ranking respects importance from the first query rather than from whenever a decay pass happened to run.

## Handoffs
Section-based session capture (`summary, decisions, todos, blockers, tried, mental_model, next_steps, notes`). Handoffs are branch-aware `MemoryType::Handoff` memories, pinned by default, with a `handoff_sections` sidecar table holding per-section embeddings (256-dim f32 LE, prefix-free). Branch chaining via `continues_from` lives in the sidecar only (not a graph edge). Auto-links to `decision/pattern/debug` memories at cosine similarity >= 0.75 (cap 10 links). Bypasses dedup entirely. Two MCP prompts: `handoff` (capture) and `resume` (restore). CLI: `engram-cli handoff create/resume/search`.

Section semantics: **todos** — Within-session work the next agent should pick up immediately. Concrete, ready-to-execute items. **blockers** — Things preventing forward motion right now (missing access, failing dependency, unanswered question). **tried** — Approaches attempted and abandoned, each with the reason it failed. **next_steps** — Post-session follow-ups beyond the current thread. Future-facing, not for immediate pickup.

### Carry-over

`handoff_resume` returns `open_todos` and `open_blockers`: the newest handoff's todos and blockers verbatim, outside the similarity ranking and unaffected by `max_sections`. Ranked retrieval alone loses long-running work — a task spanning several sessions competes for section slots against whatever the latest session happened to be about, and drops out exactly when it has been open longest.

That guarantee only holds if each handoff restates what is still open, which is what the tool schema and the `handoff` prompt instruct. An omitted todo therefore reads as done; that is the intended encoding, since a snapshot cannot distinguish "finished" from "forgotten" any other way.

`tried` is deliberately excluded from carry-over. A dead end is a permanent fact rather than open state, so it is recorded once and reached later via `handoff_search` with `section_filter: ["tried"]`. Restating dead ends in every subsequent handoff would grow without bound.

## Config (env vars)
- `ENGRAM_DB` - SQLite path (default: ~/.local/share/engram/memories.db)
- `ENGRAM_PROJECT` - project scope (default: cwd name)
- `ENGRAM_BRANCH` - what "current branch" resolves to, for both binaries (default: detected from the working directory's git state)
- `ENGRAM_DECAY_INTERVAL` - decay job interval in seconds (default: 3600)
- `ENGRAM_RECLUSTER_INTERVAL` - re-clustering job interval in seconds (default: 21600)
- `ENGRAM_MAX_CANDIDATES` - max candidate memories to score during search (default: 200)
- `ENGRAM_HOOK_DEDUP_SKIP` - similarity threshold above which hook captures are silently dropped (default: 0.95, clamped to [0.5, 1.0])
- `ENGRAM_HOOK_DAILY_CAP` - max hook-captured memories per project per UTC day; `0` = unlimited (default: 50)
- `ENGRAM_HOOK_MIN_IMPORTANCE` - importance floor for hook captures (default: 0.5; values above 0.5 have no effect because dispatch caps importance at 0.5)
- `ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED` - opt-in flag for the `UserPromptSubmit` hook (default off; even when on, captures require an explicit `#remember` cue)
- `ENGRAM_TRASH_RETENTION_DAYS` - days a destroyed memory stays recoverable in `memory_trash`; `0` = keep forever (default: 30). Swept on the decay tick.
- `ENGRAM_ADR_DIR` - output directory for `adr_export` file writes (default: `docs/adr` relative to server cwd). Overridden by the `dir` argument on the tool/CLI call.
- `ENGRAM_MCP_TOOL_PROFILE` - advertised MCP tool surface: `full` (27 tools, default), `core` (17; includes adr_create/adr_show/adr_list, memory_projects, memory_list and memory_restore), or `minimal` (3: memory_context, memory_store, handoff_resume). `adr_update_status` and `adr_export` are full-only. Dispatch stays permissive — non-advertised tools still execute with a one-time `[engram]` warning per process.

## Commands
```bash
cargo build --release    # binaries: target/release/engram, target/release/engram-cli
cargo test               # run all tests
cargo clippy             # lint (must pass with no warnings)
```

## Memory
Engram MCP available. Store decisions/patterns, query before architectural changes.
