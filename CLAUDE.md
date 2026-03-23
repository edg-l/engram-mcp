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
  embedding.rs - fastembed wrapper (384-dim vectors)
  decay.rs     - relevance decay algorithm
  tools.rs     - MCP tool handlers + contradiction detection + dedup + clustering
  format.rs    - human-readable output formatting for MCP results
  summarize.rs - extractive summarization for large content
  export.rs    - import/export JSON format
  error.rs     - MemoryError enum
```

## Key Types
- `MemoryType`: fact, decision, preference, pattern, debug, entity
- `RelationType`: relates_to, supersedes, derived_from, contradicts
- `Memory`: id, project_id, content, tags, importance, relevance_score, timestamps, branch, merged_from
- `MergeSource`: id, content_preview, merged_at (provenance tracking for dedup merges)
- `MemoryCluster`: id, project_id, summary, member_count, centroid, timestamps
- `MemoryError`: Database, Json, Embedding, NotFound, InvalidType, InvalidRelation, Io

## MCP Capabilities
### Tools
- `memory_store` - store memory + embedding, detect contradictions, auto-summarize, auto-dedup (0.90+), auto-cluster
- `memory_query` - semantic search with pagination, branch filtering, returns contradiction warnings
- `memory_update` - update content/tags/importance, regenerate summary
- `memory_delete` - remove memory and relationships
- `memory_link` - create relationship between memories
- `memory_graph` - traverse relationship graph
- `memory_store_batch` - store up to 100 memories atomically
- `memory_delete_batch` - delete multiple memories by ID
- `memory_export` - export project to JSON (optional embeddings)
- `memory_import` - import from JSON (merge/replace modes)
- `memory_stats` - get project statistics (includes cluster count)
- `memory_context` - get relevant memories for context (hierarchical retrieval via clusters, flat fallback)
- `memory_prune` - remove low-relevance memories (dry run by default)
- `memory_dedup` - find and merge duplicate memories (dry run by default, threshold configurable)

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
```

## Features
- Background decay job (hourly, configurable via ENGRAM_DECAY_INTERVAL)
- Background re-clustering job (6-hourly, configurable via ENGRAM_RECLUSTER_INTERVAL)
- Contradiction detection on store (similarity > 0.85 flagged)
- Semantic deduplication on store (similarity >= 0.90, same type: auto-merge with provenance)
- Hierarchical memory clustering (auto-assign to clusters, centroid-based retrieval in memory_context)
- Branch-aware queries (branch_mode: "current", "global", "all", or specific branch name)
- Access tracking for memory reinforcement
- Auto-summarization for content > 500 chars
- Batch operations with transactions
- Query pagination and empty-query optimization
- Human-readable formatted output (markdown) + JSON in collapsible block

## Config (env vars)
- `ENGRAM_DB` - SQLite path (default: ~/.local/share/engram/memories.db)
- `ENGRAM_PROJECT` - project scope (default: cwd name)
- `ENGRAM_DECAY_INTERVAL` - decay job interval in seconds (default: 3600)
- `ENGRAM_RECLUSTER_INTERVAL` - re-clustering job interval in seconds (default: 21600)
- `ENGRAM_MAX_CANDIDATES` - max candidate memories to score during search (default: 200)

## Commands
```bash
cargo build --release    # binaries: target/release/engram, target/release/engram-cli
cargo test               # run all tests
cargo clippy             # lint (must pass with no warnings)
```

## Memory
Engram MCP available. Store decisions/patterns, query before architectural changes.
