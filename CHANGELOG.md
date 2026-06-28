# Changelog

## [0.7.0] - 2026-06-28

### Added
- **Architecture Decision Records (ADRs).** A new `MemoryType::Adr` with an `adr_sections` sidecar table (migration 6). ADRs have fixed Nygard-style sections (title, context, decision, consequences), per-project sequential numbering (`MAX(existing)+1`, allocated in-transaction with a `UNIQUE(project_id, adr_number)` guard), and a status lifecycle: `proposed → accepted → superseded/deprecated`, plus `rejected → proposed` and `deprecated → accepted`. Transitions are validated; `superseded` is only reachable via supersession.
- **5 MCP tools**: `adr_create`, `adr_update_status`, `adr_list`, `adr_show`, `adr_export`. `adr_create`/`adr_show`/`adr_list` are in the `core` tool profile (now 14 tools); `adr_update_status`/`adr_export` are `full`-only (now 23). None are in `minimal`.
- **CLI**: `engram-cli adr create/update-status/list/show/export`.
- **File export**: `adr_export` writes Nygard-style `docs/adr/NNNN-kebab-title.md` files (target dir via `ENGRAM_ADR_DIR`, default `docs/adr`). Dry-run by default; pass `--write` (CLI) / `dry_run: false` (MCP) to write to disk.
- `memory_stats` now reports `adr_count`.

### Changed
- ADRs are project-global (never branch-scoped), pinned by default (exempt from decay/prune), and bypass both deduplication and clustering. Superseding an ADR flips the old one to `superseded` and creates a `Supersedes` edge, atomically with the new ADR's creation.
- JSON export format bumped to `1.2` (adds optional ADR sidecar fields). Import still accepts `1.0`, `1.1`, and `1.2`; ADR number collisions on import are skipped with a warning rather than aborting.

## [0.6.0] - 2026-06-20

### Removed
- **Contradiction detection, entirely.** Store-time auto-detection on `memory_store` (the `potential_contradictions` scan at cosine similarity ≥ 0.85 within a type) is gone — it false-positived on legitimate supersession (e.g. a new handoff continuing a chain flagged its predecessors). Query-time `contradiction_warnings` on `memory_query` and the underlying relationship batch check are also removed, along with the `potential_contradictions` / `contradiction_warnings` response fields.
- **`contradicts` relation type.** `RelationType::Contradicts` is removed from `memory_link` / `memory_graph` (valid relations are now `relates_to`, `supersedes`, `derived_from`). Existing stored `contradicts` edges load safely as `relates_to` (the DB read path falls back via `unwrap_or(RelationType::RelatesTo)`), so no migration is required.

### Changed
- **`PostToolUse` hook is now a no-op.** Tool-call outcomes — including failures — are no longer captured as `Debug` memories. They were low-signal noise that bloated the store. The handler validates its payload and returns immediately, matching `Stop` / `PreCompact`.
- Removed the now-unused `ENGRAM_HOOK_TOOL_ALLOWLIST` and `ENGRAM_HOOK_TOOL_DENYLIST` env vars (they only gated the PostToolUse capture path).

## [0.5.5] - 2026-06-04

### Fixed
- Tool dispatch errors now carry the correct JSON-RPC error code. A malformed `tools/call` (e.g. `memory_store` with an empty arguments object, which fails deserialization with `missing field 'content'`) previously returned `-32603 Internal error`, implying a server fault. Client-side faults (bad arguments, invalid type/relation names, unknown memory IDs) now return `-32602 Invalid params`; only genuine server faults (database, embedding, IO) stay `-32603`.
- Calling an unknown tool returned a `success` result wrapping `{"error": "Unknown tool: ..."}` instead of a protocol error. It now returns `-32602 Invalid params` with an `Unknown tool` message, so callers can detect the failure.

## [0.5.2] - 2026-05-21

### Added
- `handoff_resume` accepts `max_chars_per_section` (and `engram-cli handoff resume --max-chars-per-section <N>`). When set and > 0, each returned `section_text` is char-truncated with a `… [truncated, N chars total]` marker. Default behavior unchanged. Use this when a previous resume response was rejected as too large by the caller.
- `handoff_create` returns advisory `warnings: Vec<String>` for oversized sections (> 5000 chars) or oversized list items (> 1000 chars). The handoff is still stored; the warning points the writer at storing long content as separate `memory_store` entries that auto-link instead of being dumped into sections. CLI prints warnings to stderr.

### Changed
- `handoff_create` MCP tool description and `prompts/handoff.md` rewritten to forbid transcript dumps and direct long content to separate memories (auto-linked back via `derived_from`).

## [0.5.1] - 2026-05-15

### Fixed
- Decay never ran in production builds. `Database::open` (the on-disk constructor used by every real install) did not register the `EXP()` and `LN()` SQLite scalar functions that the decay query depends on; only `Database::open_in_memory` (test-only) did. The `update_relevance_scores` query silently failed with `no such function: EXP`, leaving every non-pinned memory's `relevance_score` stuck at the initial `1.0` forever. This made `memory_prune` ineffective, `min_relevance` filters inert, and removed the recency/importance contribution from hybrid scoring in `memory_context`. Math-function registration is now hoisted into a shared `register_math_scalar_functions` helper called from both constructors. New regression test `tests/decay_production_path.rs` exercises decay through `Database::open` against a tempfile-backed DB so this can't silently re-break.

### Action required after upgrade
- Run `engram-cli decay` once to apply the long-overdue relevance update to memories stored under v0.4.x / v0.5.0.

## [0.5.0] - 2026-05-15

### Added
- BM25 hybrid retrieval with Reciprocal Rank Fusion (k=60) alongside the existing vector path. New LongMemEval-S benchmark harness under `benchmarks/longmemeval/`.
- Claude Code lifecycle hook capture: `engram-cli hook-event <Event>` consumes stdin JSON; `engram-cli hooks install/uninstall/status` manages `settings.json` wiring. Captured events flow through redaction, filtering, and dedup.
- `external_artifacts` field on memories (`memory_store` / `memory_update` / `memory_store_batch` schemas, `--artifact <PATH>` CLI flag). Retrieval surfaces a `[missing]` marker for local-looking absent paths; URLs and opaque identifiers print unmarked.
- `ENGRAM_MCP_TOOL_PROFILE` env var (`full` | `core` | `minimal`, default `full`) to reduce the advertised MCP tool surface. Profiles:
  - **Minimal (3)**: `memory_context`, `memory_store`, `handoff_resume`.
  - **Core (11)**: Minimal + `memory_query`, `memory_update`, `memory_delete`, `memory_link`, `memory_graph`, `handoff_create`, `memory_store_batch`, `memory_delete_batch`.
  - **Full (18)**: every tool (default).
  Dispatch remains permissive: non-advertised tool names still execute, with a one-time `[engram]` warning per process.
- Hook tuning knobs: `ENGRAM_HOOK_DEDUP_SKIP` (default 0.95), `ENGRAM_HOOK_DAILY_CAP` (default 50, 0 = unlimited), `ENGRAM_HOOK_MIN_IMPORTANCE` (default 0.5).

### Changed
- Hooks overhaul: payload structs rewritten to verified Claude Code schemas. `SessionEnd` reads `transcript_path` and stores the last assistant message as a `session_summary` Fact. `Stop` and `PreCompact` become explicit no-ops. `UserPromptSubmit` defaults off, opt-in via `#remember` cue or `ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED`. `MANAGED_EVENTS` trimmed to four events. All hook stores route through `store_with_dedup`, so near-duplicate captures are silently skipped. Hook importance clamped to ≤ 0.5.
- Contradiction detection now applies only within the same non-handoff `MemoryType`. Cross-type matches and handoff-touching matches no longer raise warnings.
- `handoff_resume` on a single-handoff branch supplements `linked_memories` with related `Decision` / `Pattern` / `Debug` memories via vector search against the query embedding.
- Clarified `todos` / `blockers` / `next_steps` handoff section semantics across CLI help, MCP tool schemas, prompts, the `HandoffSections` struct docs, and CLAUDE.md.

### Fixed
- Several byte-slice panic bugs in hook content truncation via `floor_char_boundary` (Phase 2 of the hooks overhaul).
- `StoreOutcome::Merged` now returns the surviving record's id when the existing global memory wins the merge (Phase 3 code-review fix).

### Internal
- Split `src/tools.rs` into `src/tools/` module tree; split `src/db.rs` into `src/db/` module tree.
- Added a Criterion benchmark suite (`benches/`).

### Migration
- Schema migration 5 auto-adds the `external_artifacts TEXT` column on first startup. Idempotent on existing databases. No manual action needed.

## [0.4.0] and earlier

See git history.
