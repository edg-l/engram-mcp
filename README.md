<p align="center">
  <img src="assets/logo.svg" alt="Engram Logo" width="180" height="180">
</p>

<h1 align="center">Engram</h1>

<p align="center"><strong>Git-aware session memory for coding agents.</strong></p>

<p align="center">
  <a href="https://crates.io/crates/engram_mcp"><img src="https://img.shields.io/crates/v/engram_mcp.svg" alt="Crates.io"></a>
  <a href="https://github.com/edg-l/engram-mcp/actions/workflows/ci.yml"><img src="https://github.com/edg-l/engram-mcp/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/edg-l/engram-mcp/actions/workflows/release.yml"><img src="https://github.com/edg-l/engram-mcp/actions/workflows/release.yml/badge.svg" alt="Release"></a>
</p>

Engram is an [MCP](https://modelcontextprotocol.io/) server that gives coding agents persistent memory scoped to your git branches. Capture a session as a structured handoff, switch branches, and pick up later with the prior session's decisions, blockers, and todos surfaced automatically. Backed by local SQLite and ONNX embeddings; no cloud, no LLM calls for storage.

## Why this exists

- Coding agents lose context between sessions. Notes in scratch files don't surface at the right moment.
- Git branches encode work-in-progress boundaries that prior memory tools ignore — context from `feat/auth` leaks into `fix/billing`.
- Engineers resume work by re-reading their own structured notes (decisions, blockers, mental model). Handoffs match that workflow instead of dumping a chat transcript.

## Quickstart: session handoffs

End of session, capture state on the current branch:

```bash
engram-cli handoff create
# Prompts for: summary, decisions, todos, blockers, mental_model, next_steps, notes
```

Or invoke the MCP prompt from inside Claude Code: `/mcp__engram__handoff`.

Next session (same or different branch), restore context:

```bash
engram-cli handoff resume
# Resuming `feat/auth`, 3 handoffs in chain, latest from mem_xyz
# Top sections (summary, blockers, todos)
# Linked decisions/patterns/debug memories
```

Or: `/mcp__engram__resume` from Claude Code.

Each new handoff sets `continues_from` to the previous one on the same branch, forming a chain. `handoff_resume` walks the chain (depth 5, cycle-detected) and returns the top-scoring sections via hybrid similarity + recency. Sections automatically link to existing `decision` / `pattern` / `debug` memories at cosine ≥ 0.75.

When a branch has only one handoff, `handoff_resume` supplements `linked_memories` with related `decision` / `pattern` / `debug` memories surfaced via vector search against the query — so even a fresh branch with a single capture comes back with cross-cutting context, not five slices of the same document.

## Installation

```bash
cargo install engram_mcp
```

Installs `engram` (MCP server) and `engram-cli` (command-line tool).

From source:

```bash
git clone https://github.com/edg-l/engram-mcp.git
cd engram-mcp
cargo build --release
```

## Setup

### Claude Code

```bash
claude mcp add -s user engram $(which engram)
```

Allow all Engram tools without permission prompts:

```json
{
  "permissions": {
    "allow": ["mcp__engram__*"]
  }
}
```

### Claude Desktop

Add to your config file:
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram"
    }
  }
}
```

### Auto-capture decisions and session summaries (Claude Code)

Passively capture decisions and session summaries into Engram without any
explicit tool calls. One-liner install:

```bash
engram-cli hooks install
```

This registers handlers for three lifecycle events: `UserPromptSubmit`,
`SubagentStop`, and `SessionEnd`. `Stop`, `PreCompact`, and `PostToolUse` are
explicit no-ops (per-turn noise, mid-session duplicates of `SessionEnd`, and
low-signal tool-call outcomes respectively); tool-call outcomes are never
captured.
All hook stores route through the same dedup path as `memory_store`, so
near-identical captures are silently skipped (`ENGRAM_HOOK_DEDUP_SKIP`,
default 0.95). A per-project daily cap (`ENGRAM_HOOK_DAILY_CAP`, default 50)
prevents runaway logging. See [hooks/README.md](hooks/README.md) for the full
per-event reference, env vars, and secret redaction details.

### Auto-load context on session start (Claude Code)

A hook script loads relevant memories at the start of every conversation, building a semantic query from recent git activity so the LLM gets context without needing to call `memory_context`.

```bash
cp scripts/engram-hook.sh ~/.claude/hooks/engram-hook.sh
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          { "type": "command", "command": "~/.claude/hooks/engram-hook.sh" }
        ]
      }
    ]
  }
}
```

Works in non-git directories. Exits silently if `engram-cli` is not on PATH.

### Handoff skills (Claude Code)

Two opinionated skills wrap the handoff tools:

- `handoff` — gathers session state and calls `handoff_create`
- `read-handoffs` — calls `handoff_resume`, summarizes, pairs with `memory_context`

```bash
scripts/install-skills.sh
```

Skip if you prefer the bundled MCP prompts (`/mcp__engram__handoff` and `/mcp__engram__resume`) — they cover the same flow.

### Importing legacy markdown handoffs

```bash
scripts/port_md_handoffs.py ~/.claude/handoff /path/to/repo/.claude/handoff   # dry run
scripts/port_md_handoffs.py --apply ~/.claude/handoff /path/to/repo/.claude/handoff
```

Maps old section headings to the new schema; resolves `Continues from:` chains. Lossy mapping (`Dead ends → blockers`); originals stay on disk as backup.

## Memory beyond handoffs

Handoffs are the lead workflow, but Engram is a full memory system underneath:

- **Semantic search** with hybrid scoring (cosine + recency + importance)
- **Local embeddings** via [mdbr-leaf-ir](https://huggingface.co/onnx-community/mdbr-leaf-ir-ONNX) (256-dim MRL, quantized ONNX)
- **Memory decay** with reinforcement on access and auto-pruning of dead memories
- **Pinned memories** that never decay
- **Global memories** visible across projects
- **Semantic deduplication** at store time (≥ 0.90 auto-merge) plus periodic background dedup
- **Hierarchical clustering** with centroid-based retrieval at scale
- **Relationship graphs** (`relates_to`, `supersedes`, `derived_from`)
- **Contradiction detection** flags conflicts at cosine > 0.85, scoped to the same non-handoff `MemoryType` (cross-type and handoff-touching matches are suppressed to avoid topical-overlap false positives)
- **Branch-aware queries** filter by git branch scope
- **External artifacts**: memories can declare referenced files / URLs / identifiers via `external_artifacts`. Retrieval lists them inline and tags local paths with `[missing]` if absent on the server's filesystem
- **Pre-filtered retrieval** caps embedding scans for performance at scale
- **Import/export** for backup and migration

### Memory types

| Type | Description | Example |
|------|-------------|---------|
| `fact` | Objective information | "The API uses JWT authentication" |
| `decision` | Architectural choices and rationale | "Chose SQLite over Postgres for simplicity" |
| `preference` | User or project preferences | "Prefer explicit error handling over unwrap" |
| `pattern` | Recurring approaches | "All handlers return Result<Json<T>, AppError>" |
| `debug` | Past issues and solutions | "OOM was caused by unbounded channel buffer" |
| `entity` | People, systems, services | "UserService handles all auth logic" |
| `handoff` | Session snapshots with structured sections | Created via `handoff_create`; not available in `memory_store` |
| `adr` | Architecture Decision Records (numbered, status-tracked) | Created via `adr_create`; not available in `memory_store` |

## MCP tool reference

| Tool | Description |
|------|-------------|
| `handoff_create` | Capture a session handoff with structured sections (summary, decisions, todos, blockers, mental_model, next_steps, notes) |
| `handoff_resume` | Retrieve top sections from recent handoffs on the current branch, plus linked memories |
| `handoff_search` | Search handoff sections by content; filter by branch or section name |
| `memory_store` | Store a memory with embedding, auto-dedup, auto-cluster |
| `memory_query` | Semantic search with hybrid scoring, pagination, branch filtering |
| `memory_context` | Load relevant memories for a task (hierarchical retrieval via clusters) |
| `memory_update` | Update content, tags, importance, pinned status |
| `memory_delete` | Remove a memory and its relationships |
| `memory_link` | Create typed relationships between memories |
| `memory_graph` | Traverse relationship graph from a root memory |
| `memory_store_batch` | Store up to 100 memories atomically |
| `memory_delete_batch` | Delete multiple memories by ID |
| `memory_export` | Export project memories to JSON |
| `memory_import` | Import from JSON (merge or replace modes) |
| `memory_stats` | Project statistics (counts, types, pinned, global, clusters) |
| `memory_prune` | Remove low-relevance memories (dry run by default) |
| `memory_dedup` | Find and merge duplicate memories (dry run by default) |
| `memory_promote` | Promote a branch-local memory to global scope |
| `adr_create` | Create a numbered, Nygard-style ADR; optionally supersede an existing ADR |
| `adr_update_status` | Advance an ADR through its lifecycle (proposed/accepted/deprecated/rejected) |
| `adr_list` | List the project's ADRs, optionally filtered by status |
| `adr_show` | Show a single ADR by number |
| `adr_export` | Export ADRs to Nygard-style Markdown files (dry run by default) |

### Storing memories

```json
{
  "content": "We chose PostgreSQL over SQLite for the API because of concurrent write requirements",
  "type": "decision",
  "tags": ["database", "api", "architecture"],
  "importance": 0.7,
  "pinned": true,
  "global": false,
  "external_artifacts": ["docs/adr/0007-postgres.md", "https://github.com/foo/bar/pull/42"]
}
```

- `pinned: true` — never decays or gets pruned
- `global: true` — visible in all projects (forces `branch` to null)
- `importance` — 0.3 minor, 0.5 normal, 0.7 important, 0.9 critical
- `external_artifacts` — optional list of file paths, URLs, or opaque IDs the memory references. Local-looking paths are existence-checked at retrieval and marked `[missing]` if absent. URLs and opaque IDs print unmarked (no network calls). Use `memory_update` with `[]` to clear; omit the field to preserve existing

### Querying

```json
{
  "query": "what database do we use and why",
  "limit": 10,
  "min_relevance": 0.3,
  "types": ["decision", "fact"],
  "branch_mode": "current"
}
```

Branch modes: `current` (global + current branch), `global` (global only), `all`, or a specific branch name.

## CLI

```bash
# Handoffs
engram-cli handoff create                        # interactive section prompts
engram-cli handoff create --from-file session.md # ingest pre-written markdown
engram-cli handoff resume --branch feat/x        # load context from recent handoffs
engram-cli handoff search "auth refactor" --section blockers,todos

# Search
engram-cli query "how does authentication work"
engram-cli context "working on auth refactor"
engram-cli context "auth refactor" --global

# CRUD
engram-cli store "The API uses rate limiting" -t fact --tags api,security
engram-cli store "Always use snake_case" -t preference --pinned --global
engram-cli store "Bench results show BM25 wins" -t fact \
  --artifact benchmarks/longmemeval/RESULTS.md \
  --artifact https://github.com/edg-l/engram-mcp/pull/42
engram-cli show mem_abc123
engram-cli list
engram-cli update mem_abc123 -c "Updated content" --importance 0.9
engram-cli update mem_abc123 --artifact /new/path.md          # replace artifact list
engram-cli update mem_abc123 --clear-artifacts                # clear artifact list
engram-cli delete mem_abc123

# Pinning
engram-cli pin mem_abc123
engram-cli unpin mem_abc123

# Relationships
engram-cli link mem_abc123 mem_def456 -r relates_to

# Import/Export
engram-cli export -o backup.json
engram-cli import backup.json

# Maintenance
engram-cli stats
engram-cli decay
engram-cli prune -t 0.2 --confirm
engram-cli dedup -t 0.90
engram-cli dedup -t 0.90 --confirm
engram-cli wipe --confirm

# Observability
engram-cli insights
engram-cli health
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_DB` | SQLite database path | `~/.local/share/engram/memories.db` |
| `ENGRAM_PROJECT` | Project scope identifier | Git root directory name |
| `ENGRAM_DECAY_INTERVAL` | Decay job interval (seconds) | `3600` (1 hour) |
| `ENGRAM_RECLUSTER_INTERVAL` | Re-clustering job interval (seconds) | `21600` (6 hours) |
| `ENGRAM_MAX_CANDIDATES` | Max candidate embeddings to score during context retrieval | `200` |
| `ENGRAM_ADR_DIR` | Target directory for `adr_export` Markdown files | `docs/adr` |
| `ENGRAM_MCP_TOOL_PROFILE` | Advertised MCP tool surface: `full` (23 tools), `core` (14), or `minimal` (3: `memory_context`, `memory_store`, `handoff_resume`). Dispatch stays permissive — non-advertised tools still execute with a one-time `[engram]` warning per process | `full` |
| `ENGRAM_HOOK_DEDUP_SKIP` | Similarity threshold above which hook captures are silently dropped (clamped to `[0.5, 1.0]`) | `0.95` |
| `ENGRAM_HOOK_DAILY_CAP` | Max hook-captured memories per project per UTC day; `0` = unlimited | `50` |

## Retrieval benchmark

Retrieval quality on a 30-question seeded slice of [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) (cleaned), full haystack ingested (~225–275 turn-pairs per question). See [benchmarks/longmemeval/RESULTS.md](benchmarks/longmemeval/RESULTS.md) for the full table, methodology, and caveats.

| Mode | partial-R@5 | full-R@5 | MRR |
|---|---:|---:|---:|
| Vector | 50.0% | 23.3% | 0.326 |
| BM25 | **93.3%** | **73.3%** | **0.883** |
| Hybrid (RRF k=60, default) | 93.3% | 70.0% | 0.828 |

LongMemEval queries reference specific entities, dates, and numbers in the haystack; FTS5 catches them exactly, which is why BM25 leads on this slice. Hybrid (Reciprocal Rank Fusion) is the published-best baseline and remains the default; flipping defaults on a 30-question slice would over-fit. The headline finding: **embedder tuning is the highest-leverage retrieval improvement on this workload, not the fusion algorithm.** Full 500-question run is future work.

## How it compares

| Tool | Branch-aware | Structured handoffs | Per-section embeddings | Local-only | Storage |
|------|:---:|:---:|:---:|:---:|---|
| **Engram** | ✓ | ✓ | ✓ | ✓ | SQLite + ONNX |
| [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) | — | — | — | ✓ | SQLite-vec |
| [Mem0](https://mem0.ai) | — | — | — | partial | Vector DB + LLM extraction |
| [Zep / Graphiti](https://getzep.com) | — | — | — | — | Neo4j |
| [Letta (MemGPT)](https://letta.com) | — | OS-style blocks | — | partial | Pluggable |

Engram is opinionated for coding agents using git; the others target broader assistant memory. *Comparison as of 2026-05.*

## Why not just use X?

**Why not Mem0?** Mem0 is cloud-default, runs LLM extraction per write, and has no branch model. Great for chat personalization; not aimed at coding agents who switch git contexts mid-day.

**Why not vanilla mcp-memory-service?** Solid generic memory MCP, but no handoff structure, no branch awareness. Engram trades breadth for opinionation around coding-agent workflows.

**Why not a vector DB directly?** A vector DB is one component. Engram adds decay, dedup, clustering, and the MCP layer — none of which a raw Qdrant/Chroma instance gives you.

**Why SQLite over Postgres/libSQL?** Zero-ops local-first. Embeddings are 256-dim MRL (quantized ONNX), small enough that SQLite's row size and query plan are fine for the corpus sizes a single developer accumulates. If you need multi-tenant, this isn't the tool.

## How it works

### Hybrid scoring

`memory_context` scores memories using three signals:

```
score = 0.6 * cosine_similarity + 0.2 * recency + 0.2 * importance
```

Where `recency = exp(-0.02 * days_since_access)`. A recently accessed, important memory can outrank a slightly more similar but old, low-importance one.

### Memory decay

```
relevance = (time_decay * importance_factor) + usage_boost

  time_decay        = exp(-decay_rate * days_since_access)
  importance_factor = 0.5 + (importance * 0.5)
  usage_boost       = ln(1 + access_count) * 0.1
```

- Accessing a memory boosts its score by 0.1
- Pinned memories skip decay entirely
- Memories at the floor (0.1), never accessed, older than 30 days are auto-pruned

### Deduplication

- **At store time**: new memories with ≥ 0.90 cosine similarity to an existing memory of the same type are auto-merged (tags combined, max importance kept, provenance tracked)
- **Background**: the 6-hourly recluster job dedups within clusters
- **Global wins**: when a global and local memory are duplicates, global survives

### Clustering

Related memories are auto-grouped into clusters with centroid summaries. `memory_context` uses hierarchical retrieval: score cluster centroids first, then fetch the best members from top clusters. Falls back to flat retrieval below 10 memories.

### Pre-filtered retrieval

For large memory stores, `memory_context` pre-filters via SQL before loading embeddings:

```sql
SELECT ... FROM embeddings
WHERE memory_id IN (
    SELECT id FROM memories
    WHERE (project_id = ? OR global = 1)
    ORDER BY last_accessed_at DESC LIMIT 500
)
UNION  -- pinned memories always included
SELECT ... FROM embeddings
WHERE memory_id IN (SELECT id FROM memories WHERE pinned = 1)
```

Cap is configurable via `ENGRAM_MAX_CANDIDATES`. `memory_query` always does a full scan for comprehensive results.

### Handoff internals

Each handoff has seven named sections: `summary`, `decisions`, `todos`, `blockers`, `mental_model`, `next_steps`, `notes`. Stored in a `handoff_sections` sidecar table with per-section embeddings (256-dim f32 LE, prefix-free) alongside the rendered markdown in the main `memories` row.

- **Branch chaining** via `continues_from` in the sidecar (no graph edge). `handoff_resume` walks the chain up to depth 5 with cycle detection.
- **Auto-linking**: each section is scored against existing `decision` / `pattern` / `debug` memories. Matches at cosine ≥ 0.75 get a `derived_from` edge, capped at 10 links per handoff.
- **Bypass rules**: handoffs skip dedup. Pinned by default.

## Status and limits

- 100+ tests across 8 binaries; clippy clean; fmt clean.
- Single-node, single-user. No auth (it's a local MCP server).
- Embedding model: mdbr-leaf-ir, 256-dim MRL.
- Hybrid retrieval: SQLite FTS5 keyword + cosine.
- Cross-PC sync is not supported (local SQLite only). Use `engram-cli export` / `import` for manual sync.

## Development

```bash
cargo build --release    # binaries: target/release/engram, target/release/engram-cli
cargo test               # run all tests
cargo clippy             # lint
cargo fmt --check        # format check
cargo bench              # criterion benches; see BENCHMARKS.md
```

## License

MIT OR Apache-2.0
