# Changelog

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
