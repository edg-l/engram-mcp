<p align="center">
  <img src="assets/logo.svg" alt="Engram Logo" width="180" height="180">
</p>

<h1 align="center">Engram</h1>

<p align="center">
  <a href="https://crates.io/crates/engram_mcp"><img src="https://img.shields.io/crates/v/engram_mcp.svg" alt="Crates.io"></a>
  <a href="https://github.com/edg-l/engram-mcp/actions/workflows/ci.yml"><img src="https://github.com/edg-l/engram-mcp/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/edg-l/engram-mcp/actions/workflows/release.yml"><img src="https://github.com/edg-l/engram-mcp/actions/workflows/release.yml/badge.svg" alt="Release"></a>
</p>

A persistent memory system for AI agents, built as an [MCP](https://modelcontextprotocol.io/) server. Gives LLMs long-term, project-scoped knowledge with semantic search, automatic decay, deduplication, and relationship graphs. Everything runs locally: SQLite for storage, ONNX embeddings (256-dim MRL vectors) for retrieval.

## Features

- **Semantic search** with hybrid scoring (cosine similarity + recency + importance)
- **Local embeddings** via [mdbr-leaf-ir](https://huggingface.co/onnx-community/mdbr-leaf-ir-ONNX) (256-dim MRL, quantized ONNX, runs locally)
- **Memory decay** with automatic relevance scoring, reinforcement on access, and auto-pruning of dead memories
- **Pinned memories** that never decay or get pruned, for permanent knowledge
- **Global memories** visible across all projects, for cross-project knowledge
- **Semantic deduplication** at store time (0.90+ similarity auto-merge) and periodic background dedup
- **Hierarchical clustering** with centroid-based retrieval for large memory stores
- **Relationship graphs** linking memories (supersedes, relates_to, derived_from, contradicts)
- **Contradiction detection** automatically flags conflicts (similarity > 0.85)
- **Pre-filtered retrieval** caps embedding scans at 500 candidates (configurable) for performance at scale
- **Branch-aware queries** filter by git branch scope
- **Import/export** for backup and migration
- **Claude Code hook** for automatic context injection at session start

## Installation

### From crates.io

```bash
cargo install engram_mcp
```

This installs both `engram` (MCP server) and `engram-cli` (command-line tool).

### From source

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

### Auto-load context on session start (Claude Code)

Engram includes a hook script that loads relevant memories at the start of every conversation. It uses recent git activity to build a semantic query, so the LLM gets project context without needing to call `memory_context` explicitly.

**1. Copy the hook script:**

```bash
cp scripts/engram-hook.sh ~/.claude/hooks/engram-hook.sh
```

**2. Add to your settings** (`~/.claude/settings.json`):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/engram-hook.sh"
          }
        ]
      }
    ]
  }
}
```

Works in non-git directories (falls back to directory name). Exits silently if `engram-cli` is not on PATH.

### Handoff skills (Claude Code)

Two opinionated skills wrap the handoff tools so capture and resume become single-command flows:

- `handoff` — gathers session state and calls `handoff_create`
- `read-handoffs` — calls `handoff_resume`, summarizes, and pairs with `memory_context`

Install both into `~/.claude/skills/` (backs up any existing copies):

```bash
scripts/install-skills.sh
```

Skip this if you prefer the bundled MCP prompts (`/mcp__engram__handoff` and `/mcp__engram__resume`) — they cover the same flow without needing local skill files.

### Importing legacy markdown handoffs

If you have pre-existing `.claude/handoff/*.md` files written by older skills, port them into Engram with:

```bash
scripts/port_md_handoffs.py ~/.claude/handoff /path/to/repo/.claude/handoff   # dry run
scripts/port_md_handoffs.py --apply ~/.claude/handoff /path/to/repo/.claude/handoff
```

The script maps old section headings to the new schema and resolves `Continues from:` chains. The mapping is lossy (`Dead ends → blockers`, etc.); the original files stay on disk as a backup.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_DB` | SQLite database path | `~/.local/share/engram/memories.db` |
| `ENGRAM_PROJECT` | Project scope identifier | Git root directory name |
| `ENGRAM_DECAY_INTERVAL` | Decay job interval (seconds) | `3600` (1 hour) |
| `ENGRAM_RECLUSTER_INTERVAL` | Re-clustering job interval (seconds) | `21600` (6 hours) |
| `ENGRAM_MAX_CANDIDATES` | Max candidate embeddings to score during context retrieval | `200` |

## Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `fact` | Objective information | "The API uses JWT authentication" |
| `decision` | Architectural choices and rationale | "Chose SQLite over Postgres for simplicity" |
| `preference` | User or project preferences | "Prefer explicit error handling over unwrap" |
| `pattern` | Recurring approaches | "All handlers return Result<Json<T>, AppError>" |
| `debug` | Past issues and solutions | "OOM was caused by unbounded channel buffer" |
| `entity` | People, systems, services | "UserService handles all auth logic" |
| `handoff` | Session snapshots with structured sections | Created via `handoff_create`; not available in `memory_store` |

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a memory with embedding, auto-dedup, auto-cluster, contradiction detection |
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
| `handoff_create` | Capture a session handoff with structured sections (summary, decisions, todos, blockers, mental model, next steps, notes) |
| `handoff_resume` | Retrieve the most relevant sections from recent handoffs on the current branch, plus linked memories |
| `handoff_search` | Search handoff sections by content; filter by branch or section name |

### Storing memories

```json
{
  "content": "We chose PostgreSQL over SQLite for the API because of concurrent write requirements",
  "type": "decision",
  "tags": ["database", "api", "architecture"],
  "importance": 0.7,
  "pinned": true,
  "global": false
}
```

- `pinned: true` -- memory never decays or gets pruned
- `global: true` -- memory is visible in all projects (forces `branch` to null)
- `importance` -- 0.3 minor, 0.5 normal, 0.7 important, 0.9 critical

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

### Relationships

```json
{
  "source_id": "mem_abc123",
  "target_id": "mem_def456",
  "relation": "supersedes",
  "strength": 1.0
}
```

Types: `relates_to`, `supersedes`, `derived_from`, `contradicts`.

## CLI

```bash
# Search
engram-cli query "how does authentication work"
engram-cli context "working on auth refactor"    # broad context loading
engram-cli context "auth refactor" --global      # include global memories

# CRUD
engram-cli store "The API uses rate limiting" -t fact --tags api,security
engram-cli store "Always use snake_case" -t preference --pinned --global
engram-cli show mem_abc123
engram-cli list
engram-cli update mem_abc123 -c "Updated content" --importance 0.9
engram-cli delete mem_abc123

# Pinning
engram-cli pin mem_abc123       # exempt from decay and pruning
engram-cli unpin mem_abc123

# Relationships
engram-cli link mem_abc123 mem_def456 -r relates_to

# Import/Export
engram-cli export -o backup.json
engram-cli import backup.json

# Maintenance
engram-cli stats
engram-cli decay                        # run decay manually
engram-cli prune -t 0.2 --confirm       # remove low-relevance memories
engram-cli dedup -t 0.90                # find duplicates (dry run)
engram-cli dedup -t 0.90 --confirm      # merge duplicates
engram-cli wipe                         # show what would be deleted
engram-cli wipe --confirm               # delete all project memories

# Observability
engram-cli insights     # usage patterns, top accessed, never accessed, health summary
engram-cli health       # actionable maintenance report with suggested commands
```

## How It Works

### Hybrid Scoring

`memory_context` scores memories using three signals:

```
score = 0.6 * cosine_similarity + 0.2 * recency + 0.2 * importance
```

Where `recency = exp(-0.02 * days_since_access)`. This means a recently accessed, important memory can outrank a slightly more similar but old, low-importance one.

### Memory Decay

Memories have a relevance score (0.0-1.0) that evolves over time:

```
relevance = (time_decay * importance_factor) + usage_boost

  time_decay       = exp(-decay_rate * days_since_access)
  importance_factor = 0.5 + (importance * 0.5)
  usage_boost      = ln(1 + access_count) * 0.1
```

- Accessing a memory boosts its score by 0.1
- Pinned memories skip decay entirely
- Memories that hit the floor (0.1), were never accessed, and are older than 30 days are auto-pruned

### Deduplication

- **At store time**: new memories with >= 0.90 cosine similarity to an existing memory of the same type are automatically merged (tags combined, max importance kept, provenance tracked)
- **Background**: the 6-hourly recluster job also deduplicates within clusters
- **Global wins**: when a global and local memory are duplicates, the global one always survives

### Clustering

Related memories are automatically grouped into clusters with centroid summaries. `memory_context` uses hierarchical retrieval: score cluster centroids first, then fetch the best members from top clusters. Falls back to flat retrieval when fewer than 10 memories exist.

### Pre-filtered Retrieval

For large memory stores, `memory_context` pre-filters candidates via SQL before loading embeddings:

```sql
SELECT ... FROM embeddings
WHERE memory_id IN (
    SELECT id FROM memories
    WHERE (project_id = ? OR global = 1)
    ORDER BY last_accessed_at DESC LIMIT 500
)
UNION  -- pinned memories always included
SELECT ... FROM embeddings
WHERE memory_id IN (
    SELECT id FROM memories WHERE pinned = 1
)
```

The cap is configurable via `ENGRAM_MAX_CANDIDATES`. `memory_query` always does a full scan for comprehensive results.

### Handoffs

Handoffs capture structured session state for high-fidelity resume across sessions. Each handoff has seven named sections: `summary`, `decisions`, `todos`, `blockers`, `mental_model`, `next_steps`, `notes`. Sections are stored in a `handoff_sections` sidecar table with per-section embeddings (256-dim f32, prefix-free) alongside the full markdown in the main `memories` row.

**Branch chaining**: `continues_from` in the sidecar links a handoff to its predecessor on the same branch. This is sidecar-only; no graph edge is created. `handoff_resume` walks the chain up to depth 5 and returns the top-scoring sections against your query.

**Auto-linking**: on creation, each section is scored against existing `decision`, `pattern`, and `debug` memories. Matches at cosine similarity >= 0.75 get a `derived_from` edge, capped at 10 links per handoff.

**Bypass rules**: handoffs skip dedup and contradiction detection entirely. They are pinned by default (exempt from decay and prune).

**MCP prompts**: The `handoff` and `resume` MCP prompts surface as `/mcp__engram__handoff` and `/mcp__engram__resume` in Claude Code (other MCP clients may surface them differently). `/mcp__engram__handoff` guides the model through capturing a handoff; `/mcp__engram__resume` calls `handoff_resume` and proposes the next action. The existing `/mcp__engram__recall_context` prompt is unchanged.

**CLI**:
```bash
engram-cli handoff create                        # interactive section prompts
engram-cli handoff create --from-file session.md # ingest pre-written markdown
engram-cli handoff resume --branch feat/x        # load context from recent handoffs
engram-cli handoff search "auth refactor" --section blockers,todos
```

Cross-PC sync is not supported (local SQLite only).

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    MCP Server                         │
│                                                      │
│  Tools: store, query, context, update, delete,       │
│         link, graph, batch, export/import,            │
│         stats, prune, dedup, promote                  │
│                                                      │
│  ┌────────────┐ ┌──────────┐ ┌───────────────────┐  │
│  │ Embedding  │ │  Decay   │ │    Clustering      │  │
│  │  Service   │ │ + Prune  │ │   + Dedup          │  │
│  └────────────┘ └──────────┘ └───────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │              SQLite Database                   │   │
│  │  memories | embeddings | relationships        │   │
│  │  projects | clusters   | cluster_members      │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

## Development

```bash
cargo build --release    # binaries: target/release/engram, target/release/engram-cli
cargo test               # run all tests
cargo clippy             # lint
cargo fmt --check        # format check
```

## License

MIT OR Apache-2.0
