# Changelog

## Unreleased

### Added
- `external_artifacts` field on memories (`memory_store` / `memory_update` / `memory_store_batch` schemas, `--artifact <PATH>` CLI flag). Retrieval surfaces a `[missing]` marker for absent local paths.
- `ENGRAM_MCP_TOOL_PROFILE` env var (`full` | `core` | `minimal`, default `full`) to reduce the advertised MCP tool surface. Profiles:
  - **Minimal (3)**: memory_context, memory_store, handoff_resume.
  - **Core (11)**: Minimal + memory_query, memory_update, memory_delete, memory_link, memory_graph, handoff_create, memory_store_batch, memory_delete_batch.
  - **Full (18)**: every tool (default).
  Dispatch remains permissive: non-advertised tool names still execute, with a one-time `[engram]` warning per process.

### Changed
- Contradiction detection now applies only within the same non-handoff `MemoryType`. Cross-type and handoff-touching matches no longer raise warnings.
- `handoff_resume` on a single-handoff branch supplements `linked_memories` with related Decision/Pattern/Debug memories via vector search.
- Clarified `todos` / `blockers` / `next_steps` handoff section semantics across CLI help, MCP schemas, and docs.

### Migration
- Schema migration 5 auto-adds the `external_artifacts TEXT` column on first startup. No manual action needed.
