# Engram

MCP server for AI agent persistent memory. SQLite + local embeddings (fastembed/all-MiniLM-L6-v2).

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
  memory.rs    - Memory, MemoryType, Relationship, RelationType, ProjectStats
  embedding.rs - fastembed wrapper (384-dim vectors)
  decay.rs     - relevance decay algorithm
  tools.rs     - MCP tool handlers + contradiction detection
  summarize.rs - extractive summarization for large content
  export.rs    - import/export JSON format
  error.rs     - MemoryError enum
```

## Key Types
- `MemoryType`: fact, decision, preference, pattern, debug, entity
- `RelationType`: relates_to, supersedes, derived_from, contradicts
- `Memory`: id, project_id, content, tags, importance, relevance_score, timestamps
- `MemoryError`: Database, Json, Embedding, NotFound, InvalidType, InvalidRelation, Io

## MCP Capabilities
### Tools
- `memory_store` - store memory + embedding, detect contradictions, auto-summarize
- `memory_query` - semantic search with pagination, returns contradiction warnings
- `memory_update` - update content/tags/importance, regenerate summary
- `memory_delete` - remove memory and relationships
- `memory_link` - create relationship between memories
- `memory_graph` - traverse relationship graph
- `memory_store_batch` - store up to 100 memories atomically
- `memory_delete_batch` - delete multiple memories by ID
- `memory_export` - export project to JSON (optional embeddings)
- `memory_import` - import from JSON (merge/replace modes)
- `memory_stats` - get project statistics

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
engram-cli delete <id>             # delete memory
engram-cli update <id> -c "new"    # update memory
engram-cli link <src> <tgt> -r relates_to  # link memories
engram-cli export -o backup.json   # export to file
engram-cli import backup.json      # import from file
engram-cli stats                   # show statistics
engram-cli decay                   # run decay manually
engram-cli prune -t 0.2 --confirm  # remove low-relevance
```

## Features
- Background decay job (hourly, configurable via ENGRAM_DECAY_INTERVAL)
- Contradiction detection on store (similarity > 0.85 flagged)
- Access tracking for memory reinforcement
- Auto-summarization for content > 500 chars
- Batch operations with transactions
- Query pagination and empty-query optimization

## Config (env vars)
- `ENGRAM_DB` - SQLite path (default: ~/.local/share/engram/memories.db)
- `ENGRAM_PROJECT` - project scope (default: cwd name)
- `ENGRAM_DECAY_INTERVAL` - decay job interval in seconds (default: 3600)

## Commands
```bash
cargo build --release    # binaries: target/release/engram, target/release/engram-cli
cargo test               # run all tests
cargo clippy             # lint (must pass with no warnings)
```
