# Engram

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

Add to your Claude Code MCP settings (`~/.claude/claude_desktop_config.json` or similar):

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram",
      "env": {
        "ENGRAM_PROJECT": "my-project"
      }
    }
  }
}
```

## Configuration

Configuration is done via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENGRAM_DB` | Path to SQLite database file | `~/.local/share/engram/memories.db` |
| `ENGRAM_PROJECT` | Project identifier for scoping memories | Current directory name |

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
├── main.rs        # MCP server entry point
├── lib.rs         # Library exports
├── db.rs          # SQLite operations
├── memory.rs      # Memory types and structs
├── embedding.rs   # Local embedding service
├── decay.rs       # Relevance decay algorithm
├── tools.rs       # MCP tool handlers
└── error.rs       # Error types

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
