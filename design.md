# Engram - Design Document

A persistent memory system for AI agents, providing project-scoped knowledge storage with semantic search, memory decay, and relationship graphs.

## Vision

Enable AI agents to maintain long-term memory across sessions, remembering:
- Facts about the codebase and project
- Architectural decisions and their rationale
- User preferences and coding style
- Common patterns and idioms used
- Debugging history and solutions
- Relationships between concepts

## Core Features

### Memory Types
| Type | Description | Example |
|------|-------------|---------|
| `fact` | Objective information about the project | "The API uses JWT authentication" |
| `decision` | Architectural/design choices with rationale | "Chose SQLite over Postgres for simplicity" |
| `preference` | User/project preferences | "Prefer explicit error handling over unwrap" |
| `pattern` | Recurring code patterns | "All handlers follow the Result<Json<T>, AppError> pattern" |
| `debug` | Past issues and solutions | "OOM was caused by unbounded channel buffer" |
| `entity` | Named entities (files, functions, concepts) | "UserService handles all auth logic" |

### Retrieval System
- **Semantic search** using local embeddings (sentence-transformers)
- Similarity threshold filtering
- Optional tag/type filtering
- Results ranked by: `similarity * relevance_score`

### Memory Decay
Memories have a `relevance_score` (0.0 - 1.0) that decays over time:
- Decay rate: configurable (default: 0.01/day)
- Reinforcement: accessing a memory boosts its score
- Floor: memories never drop below 0.1 (can be archived manually)
- High-importance memories decay slower

### Relationship Graph
Memories can link to other memories:
- `relates_to` - general association
- `supersedes` - newer info replacing old
- `derived_from` - conclusion from evidence
- `contradicts` - conflicting information (flag for review)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCP Server                              │
│  ┌──────────────────┐  ┌─────────────┐  ┌─────────────────┐     │
│  │      Tools       │  │  Resources  │  │    Prompts      │     │
│  │                  │  │             │  │                 │     │
│  │ memory_store     │  │ /memories/* │  │ recall_context  │     │
│  │ memory_query     │  │             │  │                 │     │
│  │ memory_update    │  │             │  │                 │     │
│  │ memory_delete    │  │             │  │                 │     │
│  │ memory_link      │  │             │  │                 │     │
│  │ memory_graph     │  │             │  │                 │     │
│  │ memory_store_batch│ │             │  │                 │     │
│  │ memory_delete_batch││             │  │                 │     │
│  │ memory_export    │  │             │  │                 │     │
│  │ memory_import    │  │             │  │                 │     │
│  │ memory_stats     │  │             │  │                 │     │
│  └────────┬─────────┘  └──────┬──────┘  └────────┬───────┘     │
│           │                   │                   │             │
│  ┌────────┴───────────────────┴───────────────────┴──────────┐  │
│  │                      Core Engine                          │  │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐  │  │
│  │  │ Embedding  │  │   Decay    │  │   Graph / Export    │  │  │
│  │  │  Service   │  │  Manager   │  │   Summarization     │  │  │
│  │  └────────────┘  └────────────┘  └─────────────────────┘  │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                     SQLite Database                        │  │
│  │    memories | embeddings | relationships | projects        │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

CLI: engram-cli (query, list, show, store, delete, update, link, export, import, stats, decay, prune)
```

## Data Model (SQLite Schema)

```sql
-- Core memory storage
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,  -- fact, decision, preference, pattern, debug, entity
    content TEXT NOT NULL,
    summary TEXT,               -- short version for listings
    tags TEXT,                  -- JSON array
    importance REAL DEFAULT 0.5,
    relevance_score REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_accessed_at INTEGER NOT NULL
);

-- Vector embeddings (stored separately for efficiency)
CREATE TABLE embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,       -- serialized f32 array
    model_version TEXT NOT NULL
);

-- Relationship graph
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,  -- relates_to, supersedes, derived_from, contradicts
    strength REAL DEFAULT 1.0,
    created_at INTEGER NOT NULL,
    UNIQUE(source_id, target_id, relation_type)
);

-- Project configuration
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    root_path TEXT,
    decay_rate REAL DEFAULT 0.01,
    created_at INTEGER NOT NULL
);

-- Indexes
CREATE INDEX idx_memories_project ON memories(project_id);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_relevance ON memories(relevance_score DESC);
CREATE INDEX idx_relationships_source ON relationships(source_id);
CREATE INDEX idx_relationships_target ON relationships(target_id);
```

## MCP Tools

### `memory_store`
Store a new memory.

```json
{
    "content": "The auth module uses bcrypt with cost factor 12",
    "type": "fact",
    "tags": ["auth", "security"],
    "importance": 0.8,
    "related_to": ["mem_abc123"]  // optional links
}
```

### `memory_query`
Semantic search for relevant memories.

```json
{
    "query": "how does authentication work",
    "limit": 10,
    "min_relevance": 0.3,
    "types": ["fact", "decision"],  // optional filter
    "tags": ["auth"]                // optional filter
}
```

Returns memories ranked by `similarity * relevance_score`.

### `memory_update`
Update an existing memory.

```json
{
    "id": "mem_abc123",
    "content": "Updated content...",
    "importance": 0.9,
    "tags": ["new", "tags"]
}
```

### `memory_delete`
Remove a memory (and its relationships).

```json
{
    "id": "mem_abc123"
}
```

### `memory_link`
Create/update relationships between memories.

```json
{
    "source_id": "mem_abc123",
    "target_id": "mem_def456",
    "relation": "supersedes"
}
```

### `memory_graph`
Retrieve a memory with its related memories (graph traversal).

```json
{
    "id": "mem_abc123",
    "depth": 2,
    "relation_types": ["relates_to", "derived_from"]
}
```

### `memory_store_batch`
Store multiple memories atomically (max 100 per batch).

```json
{
    "memories": [
        {"content": "First memory", "type": "fact"},
        {"content": "Second memory", "type": "decision", "importance": 0.8}
    ]
}
```

### `memory_delete_batch`
Delete multiple memories by ID.

```json
{
    "ids": ["mem_abc123", "mem_def456"]
}
```

### `memory_export`
Export all project memories to JSON.

```json
{
    "include_embeddings": true  // optional, increases size
}
```

### `memory_import`
Import memories from JSON export.

```json
{
    "data": { /* export data object */ },
    "mode": "merge"  // or "replace"
}
```

### `memory_stats`
Get project statistics.

```json
{}
```

Returns: `{project_id, memory_count, relationship_count, avg_relevance}`

## Embedding Strategy

### Local Model
- **Model**: `all-MiniLM-L6-v2` (384 dimensions, fast, good quality)
- **Library**: `fastembed` - actively maintained, simple API, ONNX-based
- Supports multiple models out of the box
- No Python dependencies

### Embedding Process
1. Concatenate: `{type}: {content}` (type provides context)
2. Generate embedding
3. Normalize to unit vector
4. Store as binary blob (384 × 4 bytes = 1.5KB per memory)

### Similarity Search
- Cosine similarity (dot product on normalized vectors)
- In-memory for small projects (<10K memories)
- Consider `sqlite-vss` extension for larger scale

## Memory Decay Algorithm

```rust
fn calculate_relevance(memory: &Memory, now: Timestamp) -> f64 {
    let days_since_access = (now - memory.last_accessed_at).as_days();
    let days_since_created = (now - memory.created_at).as_days();

    // Base decay
    let time_decay = (-decay_rate * days_since_access).exp();

    // Importance modifier (high importance = slower decay)
    let importance_factor = 0.5 + (memory.importance * 0.5);

    // Usage boost (frequently accessed = more relevant)
    let usage_boost = (memory.access_count as f64).ln_1p() * 0.1;

    // Calculate final score
    let score = (time_decay * importance_factor + usage_boost).clamp(0.1, 1.0);

    score
}
```

### Reinforcement
When a memory is accessed:
1. Increment `access_count`
2. Update `last_accessed_at`
3. Boost `relevance_score` by 0.1 (capped at 1.0)

## Roadmap

### Phase 1: Foundation ✅
- [x] SQLite database setup with schema
- [x] Basic CRUD operations for memories
- [x] MCP server with `memory_store` and `memory_query` (text search)
- [x] Project scoping

### Phase 2: Semantic Search ✅
- [x] Integrate local embedding model
- [x] Vector storage and retrieval
- [x] Similarity-based search
- [x] Hybrid search (semantic + filters)

### Phase 3: Relevance & Decay ✅
- [x] Implement decay algorithm
- [x] Automatic relevance updates
- [x] Access tracking and reinforcement
- [x] Background decay job

### Phase 4: Relationships ✅
- [x] Relationship table and operations
- [x] `memory_link` tool
- [x] Graph traversal queries
- [x] Contradiction detection

### Phase 5: Polish & Optimization ✅
- [x] Memory summarization for large content (auto-generated for >500 chars)
- [x] Batch operations (`memory_store_batch`, `memory_delete_batch`)
- [x] Import/export functionality (JSON with optional embeddings)
- [x] Performance optimization (new indexes, empty query optimization, pagination)
- [x] CLI for manual memory management (`engram-cli`)

## Future Considerations

### Potential Features
- **Memory compression**: Summarize old/low-relevance memories
- **Clustering**: Auto-group related memories
- **Conflict resolution**: UI for resolving contradictions
- **Multi-project**: Share memories across related projects
- **Sync**: Cloud backup/sync for memories
- **Hooks**: Trigger actions when memories match patterns

### Scaling
- Current design targets: <50K memories per project
- For larger scale: consider pgvector or dedicated vector DB
- Memory pruning: archive memories below threshold

## Dependencies

```toml
[dependencies]
rmcp = { version = "0.14", features = ["server", "transport-io"] }
tokio = { version = "1.49", features = ["full"] }
rusqlite = { version = "0.38", features = ["bundled"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.20", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
fastembed = "5"           # local ONNX-based embeddings (all-MiniLM-L6-v2)
thiserror = "2.0"
dirs = "6.0"              # XDG paths
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
base64 = "0.22"           # embedding export encoding
clap = { version = "4", features = ["derive"] }  # CLI
```
