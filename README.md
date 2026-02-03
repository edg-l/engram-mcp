# Engram

[![CI](https://github.com/edg-l/engram-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/edg-l/engram-mcp/actions/workflows/ci.yml)
[![Release](https://github.com/edg-l/engram-mcp/actions/workflows/release.yml/badge.svg)](https://github.com/edg-l/engram-mcp/actions/workflows/release.yml)

A persistent memory system for AI agents built as an MCP (Model Context Protocol) server. Provides project-scoped knowledge storage with semantic search, memory decay, and relationship graphs.

## Overview

Engram enables AI agents to maintain long-term memory across sessions, remembering:

- **Facts** about the codebase and project
- **Decisions** and their rationale
- **Preferences** for coding style and patterns
- **Patterns** and idioms used in the codebase
- **Debug history** and solutions
- **Entities** like important files, functions, or concepts

Memories are stored locally in SQLite, embedded using a local model (all-MiniLM-L6-v2), and retrieved via semantic similarity search.

## Features

- **Semantic Search**: Find relevant memories using natural language queries
- **Local Embeddings**: Uses fastembed with all-MiniLM-L6-v2 (384 dimensions, runs locally)
- **Memory Decay**: Relevance scores decay over time, with reinforcement on access
- **Relationship Graphs**: Link memories with typed relationships (supersedes, relates_to, derived_from, contradicts)
- **Contradiction Detection**: Automatically flags potential contradictions (similarity > 0.85)
- **Auto-Summarization**: Long content (> 500 chars) is automatically summarized
- **Batch Operations**: Store or delete up to 100 memories atomically
- **Import/Export**: Backup and restore memories in JSON format
- **Project Scoping**: Memories are isolated per project
- **MCP Protocol**: Standard interface for AI assistant integration

## Installation

### Prerequisites

- Rust 1.70+ (uses edition 2024)
- SQLite (bundled with rusqlite)

### Build from Source

```bash
git clone https://github.com/edg-l/engram-mcp.git
cd engram-mcp
cargo build --release
```

The binary will be at `target/release/engram`.

### Configure with Claude Code

```bash
claude mcp add -s user engram $(which engram)
```

#### Allow All Permissions

By default, Claude Code will ask for permission before using each MCP tool. To allow all Engram tools without prompts, add to your `~/.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "mcp__engram__memory_store",
      "mcp__engram__memory_query",
      "mcp__engram__memory_update",
      "mcp__engram__memory_delete",
      "mcp__engram__memory_link",
      "mcp__engram__memory_graph",
      "mcp__engram__memory_store_batch",
      "mcp__engram__memory_delete_batch",
      "mcp__engram__memory_export",
      "mcp__engram__memory_import",
      "mcp__engram__memory_stats"
    ]
  }
}
```

Or allow all Engram tools with a single pattern:

```json
{
  "permissions": {
    "allow": ["mcp__engram__*"]
  }
}
```

### Configure with Claude Desktop

Add to the config file:
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

## Configuration

Configuration is done via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_DB` | Path to SQLite database file | `~/.local/share/engram/memories.db` |
| `ENGRAM_PROJECT` | Project identifier for scoping memories | Git root path (or current directory if not in a git repo) |
| `ENGRAM_DECAY_INTERVAL` | Background decay job interval in seconds | `3600` (1 hour) |

**Project Detection:** Engram automatically detects the git repository root, so running from any subdirectory uses the same project scope.

**Background Decay:** The MCP server runs a background job that periodically applies decay to all memories based on time since last access.

### Tip: Add a CLAUDE.md hint

Claude discovers Engram tools automatically via MCP, but adding a brief hint to your project's `CLAUDE.md` encourages proactive use:

```markdown
## Memory
Engram MCP available. Store decisions/patterns, query before architectural changes.
```

## MCP Tools

### memory_store

Store a new memory.

```json
{
  "content": "The API uses JWT tokens for authentication with RS256 signing",
  "type": "fact",
  "tags": ["auth", "api", "security"],
  "importance": 0.8,
  "summary": "JWT auth with RS256",
  "related_to": ["mem_abc123"]
}
```

**Parameters:**
- `content` (required): The memory content
- `type` (required): One of `fact`, `decision`, `preference`, `pattern`, `debug`, `entity`
- `tags`: Array of categorization tags
- `importance`: 0.0-1.0, affects decay rate (default: 0.5)
- `summary`: Short version for listings
- `related_to`: IDs of related memories to link

**Returns:**
```json
{
  "id": "mem_a1b2c3d4...",
  "message": "Memory stored successfully"
}
```

### memory_query

Search for relevant memories using semantic similarity.

```json
{
  "query": "how does authentication work",
  "limit": 10,
  "min_relevance": 0.3,
  "types": ["fact", "decision"],
  "tags": ["auth"]
}
```

**Parameters:**
- `query` (required): Natural language search query
- `limit`: Maximum results (default: 10, max: 100)
- `min_relevance`: Minimum score threshold (default: 0.3)
- `types`: Filter by memory types
- `tags`: Filter by tags (matches any)

**Returns:**
```json
{
  "count": 2,
  "memories": [
    {
      "memory": {
        "id": "mem_...",
        "content": "...",
        "memory_type": "fact",
        "tags": ["auth"],
        "importance": 0.8,
        "relevance_score": 0.95,
        ...
      },
      "score": 0.87
    }
  ]
}
```

### memory_update

Update an existing memory.

```json
{
  "id": "mem_abc123",
  "content": "Updated content...",
  "importance": 0.9,
  "tags": ["new", "tags"],
  "summary": "New summary"
}
```

**Parameters:**
- `id` (required): Memory ID to update
- `content`: New content (re-generates embedding)
- `importance`: New importance score
- `tags`: New tags (replaces existing)
- `summary`: New summary

### memory_delete

Delete a memory and its relationships.

```json
{
  "id": "mem_abc123"
}
```

### memory_link

Create a relationship between memories.

```json
{
  "source_id": "mem_abc123",
  "target_id": "mem_def456",
  "relation": "supersedes",
  "strength": 1.0
}
```

**Relation types:**
- `relates_to`: General association
- `supersedes`: Source replaces/updates target
- `derived_from`: Source is derived from target
- `contradicts`: Source conflicts with target

### memory_graph

Retrieve a memory with its related memories via graph traversal.

```json
{
  "id": "mem_abc123",
  "depth": 2,
  "relation_types": ["relates_to", "derived_from"]
}
```

**Parameters:**
- `id` (required): Root memory ID
- `depth`: Traversal depth (default: 2, max: 5)
- `relation_types`: Filter by relationship types

**Returns:**
```json
{
  "root": { "id": "mem_abc123", ... },
  "related": [
    {
      "memory": { ... },
      "relation": "relates_to",
      "direction": "outgoing",
      "depth": 1
    }
  ]
}
```

### memory_store_batch

Store multiple memories atomically (up to 100).

```json
{
  "memories": [
    {
      "content": "First memory content",
      "type": "fact",
      "tags": ["tag1"]
    },
    {
      "content": "Second memory content",
      "type": "decision",
      "importance": 0.9
    }
  ]
}
```

### memory_delete_batch

Delete multiple memories by ID.

```json
{
  "ids": ["mem_abc123", "mem_def456", "mem_ghi789"]
}
```

### memory_export

Export all project memories to JSON.

```json
{
  "include_embeddings": false
}
```

**Parameters:**
- `include_embeddings`: Include embedding vectors in export (default: false, increases size significantly)

### memory_import

Import memories from JSON.

```json
{
  "data": { ... },
  "mode": "merge"
}
```

**Parameters:**
- `data` (required): JSON data from memory_export
- `mode`: `merge` (add new, skip existing) or `replace` (clear and import all)

### memory_stats

Get statistics for the current project.

**Returns:**
```json
{
  "total_memories": 42,
  "by_type": {
    "fact": 20,
    "decision": 10,
    "pattern": 12
  },
  "avg_relevance": 0.75,
  "total_relationships": 15
}
```

## MCP Resources

Access individual memories via URI:

```
memory://{project}/{id}
```

## MCP Prompts

### recall_context

Retrieve relevant memories for a given context. Useful for pre-loading context at the start of a conversation.

## Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `fact` | Objective information | "The API uses JWT authentication" |
| `decision` | Architectural choices | "Chose SQLite over Postgres for simplicity" |
| `preference` | User/project preferences | "Prefer explicit error handling over unwrap" |
| `pattern` | Recurring code patterns | "All handlers return Result<Json<T>, AppError>" |
| `debug` | Past issues and solutions | "OOM was caused by unbounded channel buffer" |
| `entity` | Named entities | "UserService handles all auth logic" |

## Memory Decay Algorithm

Memories have a relevance score (0.0-1.0) that evolves over time:

```
relevance = (time_decay * importance_factor) + usage_boost

where:
  time_decay = exp(-decay_rate * days_since_access)
  importance_factor = 0.5 + (importance * 0.5)
  usage_boost = ln(1 + access_count) * 0.1
```

- **Decay rate**: Configurable per project (default: 0.01/day)
- **Reinforcement**: Accessing a memory boosts its score by 0.1
- **Floor**: Memories never drop below 0.1

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     MCP Server                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │                    Tools                         │   │
│  │  memory_store  memory_query  memory_update      │   │
│  │  memory_delete memory_link   memory_graph       │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                              │
│  ┌──────────────────────┴──────────────────────────┐   │
│  │                 Core Engine                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌───────────┐  │   │
│  │  │ Embedding  │  │   Decay    │  │   Graph   │  │   │
│  │  │  Service   │  │  Manager   │  │ Resolver  │  │   │
│  │  └────────────┘  └────────────┘  └───────────┘  │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                              │
│  ┌──────────────────────┴──────────────────────────┐   │
│  │              SQLite Database                     │   │
│  │  memories | embeddings | relationships | projects│   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Database Schema

```sql
-- Core memory storage
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    tags TEXT,  -- JSON array
    importance REAL DEFAULT 0.5,
    relevance_score REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_accessed_at INTEGER NOT NULL
);

-- Vector embeddings
CREATE TABLE embeddings (
    memory_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,  -- 384 x f32 = 1.5KB
    model_version TEXT NOT NULL
);

-- Relationship graph
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    created_at INTEGER NOT NULL
);

-- Project configuration
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    root_path TEXT,
    decay_rate REAL DEFAULT 0.01,
    created_at INTEGER NOT NULL
);
```

## CLI

Engram includes a CLI tool (`engram-cli`) for direct interaction with the memory database.

```bash
# Semantic search
engram-cli query "how does authentication work"

# List all memories
engram-cli list

# Show a specific memory
engram-cli show mem_abc123

# Store a new memory
engram-cli store "The API uses rate limiting" -t fact --tags api,security

# Update a memory
engram-cli update mem_abc123 -c "Updated content" --importance 0.9

# Delete a memory
engram-cli delete mem_abc123

# Link two memories
engram-cli link mem_abc123 mem_def456 -r relates_to

# Export to file
engram-cli export -o backup.json

# Import from file
engram-cli import backup.json

# Show project statistics
engram-cli stats

# Run decay manually
engram-cli decay

# Prune low-relevance memories
engram-cli prune -t 0.2 --confirm
```

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture
```

### Project Structure

```
src/
├── main.rs        # MCP server entry point, stdio transport, resources, prompts
├── cli.rs         # CLI binary (engram-cli)
├── lib.rs         # Library exports
├── db.rs          # SQLite operations (memories, embeddings, relationships, batch ops)
├── memory.rs      # Memory, MemoryType, Relationship, RelationType, ProjectStats
├── embedding.rs   # fastembed wrapper (384-dim vectors)
├── decay.rs       # Relevance decay algorithm
├── tools.rs       # MCP tool handlers + contradiction detection
├── summarize.rs   # Extractive summarization for large content
├── export.rs      # Import/export JSON format
└── error.rs       # MemoryError enum

tests/
└── integration.rs # Integration tests
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| rmcp | 0.14 | MCP protocol implementation |
| tokio | 1.49 | Async runtime |
| rusqlite | 0.38 | SQLite database |
| fastembed | 5 | Local ONNX embeddings |
| serde | 1.0 | Serialization |
| uuid | 1.20 | ID generation |
| chrono | 0.4 | Timestamps |

## License

MIT OR Apache-2.0

## Contributing

Contributions welcome! Please ensure tests pass before submitting PRs:

```bash
cargo test
cargo clippy
cargo fmt --check
```
