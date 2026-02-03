# Engram

MCP server for AI agent persistent memory. SQLite + local embeddings (fastembed/all-MiniLM-L6-v2).

## Structure
```
src/
  main.rs      - MCP server entry, stdio transport, resources, prompts
  lib.rs       - module exports
  db.rs        - SQLite ops (memories, embeddings, relationships)
  memory.rs    - Memory, MemoryType, Relationship, RelationType structs
  embedding.rs - fastembed wrapper (384-dim vectors)
  decay.rs     - relevance decay algorithm
  tools.rs     - MCP tool handlers + contradiction detection
  error.rs     - MemoryError enum
```

## Key Types
- `MemoryType`: fact, decision, preference, pattern, debug, entity
- `RelationType`: relates_to, supersedes, derived_from, contradicts
- `Memory`: id, project_id, content, tags, importance, relevance_score, timestamps
- `MemoryError`: Database, Json, Embedding, NotFound, InvalidType, InvalidRelation, Io

## MCP Capabilities
### Tools
- `memory_store` - store memory + embedding, detect contradictions
- `memory_query` - semantic search, returns contradiction warnings
- `memory_update` - update content/tags/importance
- `memory_delete` - remove memory and relationships
- `memory_link` - create relationship between memories
- `memory_graph` - traverse relationship graph

### Resources
- `memory://{project}/{id}` - read individual memories

### Prompts
- `recall_context` - retrieve relevant memories for a context

## Features
- Background decay job (hourly, configurable via ENGRAM_DECAY_INTERVAL)
- Contradiction detection on store (similarity > 0.85 flagged)
- Access tracking for memory reinforcement

## Config (env vars)
- `ENGRAM_DB` - SQLite path (default: ~/.local/share/engram/memories.db)
- `ENGRAM_PROJECT` - project scope (default: cwd name)
- `ENGRAM_DECAY_INTERVAL` - decay job interval in seconds (default: 3600)

## Commands
```bash
cargo build --release    # binary: target/release/engram
cargo test
cargo clippy
```
