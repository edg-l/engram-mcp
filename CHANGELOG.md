# Changelog

## [Unreleased]

### Added
- **`engram-cli sync <ssh-target>`**: bidirectional, incremental replication of the whole store (every project) with another machine over plain `ssh` — no hub, no daemon, no new auth. One `ssh` process per direction, JSON over stdin/stdout. Two independent per-remote watermarks (`sync_state.pull_watermark`/`push_watermark`, migration 11) track what has already moved in each direction, each derived from the payload's own `max(updated_at)` rather than wall-clock, so clock skew between machines can't skip a row. `--dry-run` previews both payload sizes without importing or pushing; `--pull-only`/`--push-only` run a single direction; `--remote-bin` points at a non-default `engram-cli` on the remote `PATH`. A new `memory_origin` table (migration 12) records which remote a memory was last pulled from, so the push half never echoes content straight back to the remote it just arrived from — without it, two machines with asymmetric usage patterns would re-transmit the same pulled rows on every run and never converge to a no-op. See `CLAUDE.md`'s **Sync** section for the full convergence rule and what's out of scope (deletions, trash, clusters).
- **`export`/`import` gained a whole-store mode.** `export --all-projects [--since <unix-seconds>]` emits every project in one payload (`scope: "all_projects"`); `import -` reads from stdin so the two compose over an `ssh` pipe. Export format bumps to **1.3** (1.0–1.2 still accepted): `status`/`todo` sidecar fields carry each memory's dead/revival state and todo lifecycle, which the format previously dropped entirely. `import` now assigns imported memories to clusters (`tools/cluster.rs`, extracted from `ToolHandler` so both the MCP path and the CLI import path share it) — without this, imported memories were invisible to `memory_context`'s hierarchical retrieval, which only returns cluster members once a project has 10+ memories.
- Every `export` (including the existing project-scoped default, not only `--all-projects`) now draws from new unbounded getters in `src/db/sync.rs` instead of the old `query_memories`-backed helpers, which silently capped results at 10000 rows.

### Changed
- **Portable project ids.** `project_id` is now derived from the repo's normalized git remote URL (`git:host/path`) when it has one, else a `~/`-relative path, else used verbatim — instead of the previous absolute filesystem path, which didn't survive a clone landing in a different directory or under a different username on another machine. One shared derivation (`src/project.rs`) is used by `engram`, `engram-cli`, and the hook dispatcher. **Migration 10** rewrites every existing `projects.id`/`root_path` and every `project_id` column to the portable form, in one transaction, after a `VACUUM INTO <db>.pre-portable-ids.bak` backup; two legacy ids that fold to the same portable id merge into one `projects` row. **Restart any running `engram` MCP server after upgrading** — it resolves its project id once at startup, so it keeps writing under the old, now-orphaned id until restarted. Likewise, a hardcoded `--project`/`ENGRAM_PROJECT` override still naming a pre-migration legacy id is not rewritten by the migration and silently creates a new orphan project on its next write; `engram-cli projects` shows the real portable id to switch to.
- **`import --mode merge` is last-write-wins, not skip-if-present.** The old behavior skipped any id already present locally, so a todo closed or a memory edited on one machine never reached another via merge import. Merge now compares `updated_at`: strictly newer wins (content, tags, importance, pinned, global, dead status, todo lifecycle all update; `access_count`/`last_accessed_at` take the max of both sides instead), a tie or older keeps the local copy. `import` also stopped stamping `updated_at = now()` on every arrival, which previously made the importer's copy look newest regardless of actual recency and would have made last-write-wins ping-pong forever between two machines.

### Fixed
- `memory_status.dead`/`reason` toggles (dead-marking and revival) now bump the owning memory's `updated_at`, so a dead-only change with no content edit is still visible to `--since` incremental exports and still resolves correctly under last-write-wins. Revival is an upsert on `memory_status`, never a delete, so it converges through export/import the same way a dead-marking does.

## [0.10.1] - 2026-07-31

### Fixed
- **`handoff_create` rejected the payload its own schema described.** The tool advertised `sections.required = ["summary"]`, but `HandoffSections` carried no serde defaults, so anything omitting `decisions`, `todos`, `blockers`, `mental_model` or `next_steps` failed with `missing field blockers`. A caller reading the published schema had no way to construct a legal call, and the documented behaviour of leaving empty sections out was impossible. Those five sections now default, leaving `summary` as the only requirement — which is what the schema always claimed.
- **`memory_restore` marked neither `id` nor `trash_id` as required** while the handler requires one of them, so the published contract said an empty argument object was legal. Expressed as an `anyOf` so the constraint is part of the schema rather than only of the error message.

### Added
- **A contract test tying every tool schema to its deserializer** (`tests/schema_contract.rs`). Nothing connects a hand-written JSON schema to the Rust struct beside it, so the two drift silently and the failure lands on callers. The test walks every advertised tool, builds the minimal payload that tool's own schema declares legal, and asserts the server accepts it; a deserializer demanding more than the published `required` list fails with the offending payload printed. It caught the `memory_restore` case on its first run.

## [0.10.0] - 2026-07-31

### Changed
- **Relevance decay runs on project activity, not wall-clock time.** A project untouched for two years is exactly as you left it, and its memories are no less current than the day you stopped — but calendar decay drove them all to the floor, so returning to a project was precisely when its knowledge was worth least. What makes a memory stale is newer knowledge about the same project arriving to displace it. The elapsed term is now **store-days**: days on which the memory's project received a store, counted after the memory was last accessed (`src/db/activity.rs`). A dormant project's clock freezes; an active one advances at its own pace regardless of the calendar. `compute_hybrid_score` and `compute_context_score` moved to the same axis — leaving ranking on wall-clock recency would have re-imported the behaviour decay was moved off. The unit is a day rather than a store because storage is bursty (100 memories across 8 days is typical), so raw counts would let one session age everything else at once and would make `decay_rate` uninterpretable; per store-day keeps 0.01 readable as a half-life of ~69 active days. The clock is derived from `created_at` on existing rows, so there is no counter to migrate.
- **Hook-captured stores do not advance the decay clock.** They are stored without anyone deciding to, and a session where automatic capture was the only thing that happened has not made deliberately curated knowledge staler. Measured on a real store, this is the difference between 39 activity-days and 16 for one project.
- **Relevance is recomputed when a store lands**, in `memory_store`, `memory_store_batch`, and `engram-cli store`. Since relevance is a function of store-days it changes at exactly one moment, and the background job cannot be relied on for it: both background loops sleep a full interval before their first pass, so a server that lives less than an hour never runs any of them. The job remains for `sweep_trash` and `auto_prune_stale_memories`, which gate on wall-clock timestamps that no store-time hook can replace. One consequence: a freshly stored memory's relevance is now its importance factor (`0.5 + importance/2`) rather than a placeholder 1.0, so ranking respects importance from the first query.
- **`engram-cli decay` runs the full maintenance pass** (relevance, auto-prune, trash sweep) instead of relevance alone. `ENGRAM_TRASH_RETENTION_DAYS` was previously reachable only from a server that happened to stay up for an hour.

### Fixed
- **Curation fields never reached MCP callers.** The MCP transport returns only the compact rendering and discards the result JSON, so `possible_supersedes`, `matched_via`, the `memory_update`/`memory_delete` echoes, and the new stats counts existed in the payload but were invisible to an agent — the features whose entire purpose is to inform the caller were the ones the caller could not see. `memory_list`, `memory_trash` and `memory_restore` fell through to a raw-JSON fallback. Each now has a renderer, the shared update/delete handler is split so both show what they destroyed (including `merged_from` predecessors, since one id can stand for several claims), and redirected results are marked in query and context output. Tests assert each field survives rendering rather than only that it reaches the JSON.
- **`engram-cli hooks status` reported a false negative, and `uninstall` was a silent no-op.** Both matched hook blocks only by the `_source: "engram-cli"` marker the installer writes, so a block written by hand or by an older version was invisible: `status` said "No engram-cli entries installed" while three hooks were live and capturing, and `uninstall` would have removed nothing while reporting success. Detection now also matches a command that invokes `engram-cli hook-event`, however it is spelled.

## [0.9.0] - 2026-07-31

### Added
- **Supersession, as a retrieval behaviour rather than a note in the content.** `memory_store` accepts `supersedes: [id]` (CLI: `store --supersedes`), recording a `supersedes` edge from the new memory to the old. A superseded memory is no longer returned by `memory_query` or `memory_context`; instead **its successor is returned in its place, at the superseded memory's rank**, annotated with `matched_via: {superseded_id, superseded_preview}`. Marking a memory superseded in its own text left it ranking exactly where it did, which is what pushed callers to delete it instead. Dropping the match outright was rejected for the opposite reason: a narrow query then returns nothing, which reads as "nobody has investigated this" and invites the stale conclusion to be derived from scratch. Chains resolve to the terminal successor, capped at 5 hops with a cycle guard, since reverting a decision makes A→B→A reachable. A successor reached both on its own merit and by redirect appears once, at its better rank.
- **`dead` status** on `memory_update` (CLI: `update --dead [--dead-reason]`, `--alive` to undo), for a memory whose subject is gone with no replacement. Dead memories are excluded from retrieval outright, because there is nothing current to redirect to. Stored in a `memory_status` sidecar; supersession is read from the relationship edge, so neither is duplicated onto the memory row and the `Memory` export shape is unchanged.
- **A trash, making every destructive operation undoable.** `memory_delete`, `memory_delete_batch`, `memory_prune`, the unattended auto-prune, `wipe`, dedup merges, and content-replacing updates each snapshot the memory to `memory_trash` in the same transaction as the destruction. Snapshots carry the full memory, every edge touching it, and its embedding, so `memory_restore` (CLI: `restore <id>` or `--trash-id`) puts it back searchable and reconnected; edges whose other end is also gone are skipped and counted. `memory_trash` (CLI: `trash`) lists what is recoverable. Retention is `ENGRAM_TRASH_RETENTION_DAYS`, default 30, swept on the decay tick.
- **Destructive tools now report what they destroyed.** `memory_update` returns `previous`, `memory_delete` returns `deleted`, `memory_delete_batch` returns `memories`; the CLI prints the replaced or deleted content and the `restore` command to undo it. A memory frequently carries more than the claim it appears to be about, and the caller cannot check what it cannot see.
- **`memory_list`** (CLI: `list --status --order`): enumerate memories without a query, filtered by type/tag/status and ordered by relevance, creation, update, or access. Search only shows what a query matches, so the memories most in need of curation are the ones it hides; `status: superseded|dead|all` and `memory_query include_superseded` show what retrieval suppresses.
- **`possible_supersedes` on every store result** (CLI: printed after `store`): existing same-type memories at cosine >= 0.75, excluding the new memory and anything it just merged with. These similarities are computed for dedup anyway. Without them, a caller storing "X is now Y" has no way to learn that a three-month-old memory says "X is Z", because that memory never surfaced and so never became a candidate for anything. Never applied automatically: cosine cannot separate "contradicts" from "elaborates". There is deliberately no upper bound on the band — a pair can sit above the dedup threshold and still not merge, and those are the most likely supersessions, not the least.
- `memory_stats` reports `dead_count` and `trash_count`. `memory_list` and `memory_restore` join the `core` profile (now 17 tools; `full` is now 27).

### Fixed
- **Relevance scores could exceed 1.0, inverting what pinning does.** `update_relevance_scores` clamped with `MAX(0.1, …)` — a floor and no ceiling — and its usage term `LN(1 + access_count) * 0.1` was unbounded and time-independent. A memory retrieved 100 times recomputed to roughly 1.46 and stayed there. Decay skips pinned rows, so a pinned memory sits at 1.0: **pinning capped a memory's ranking multiplier below what churn could reach**, and `memory_query` ranks by `(base + tag_boost) * relevance_score`. The clamp in `decay.rs` was correct but applied to a reference implementation that production never called, and the test asserting the ceiling tested only that copy. `relevance_from_parts` is now the single definition, registered as the `RELEVANCE()` SQLite scalar function and called by the job, so the two cannot drift. The usage boost is capped at 0.1 (one step on the documented importance scale), saturates at 50 accesses, and decays with the same `exp(-0.02 * days)` the retrieval scorers use, so "retrieved often a year ago" no longer outranks "stored last week" forever.
- **Dedup merges recorded the wrong memory when a global memory won.** The merge keeps the global copy and consumes the local one, but `merged_from` provenance was built from the caller-supplied preview of the *existing* memory, so it described the survivor rather than the memory that was destroyed. `merge_memories` now reads the consumed memory's content itself inside the transaction.
- **`MergeSource` carries the consumed memory's full content**, not a 100-character preview. Auto-dedup deletes the predecessor's row, so the preview was the only surviving copy: a merge silently truncated anything past the first hundred characters of a memory nobody had asked to modify. `content_preview` is retained for display.
- **`engram-cli query` and `context` applied no curation, and `store` never deduplicated.** Each had its own implementation rather than sharing the MCP path, so a superseded memory still outranked its successor on the CLI, and CLI-stored duplicates accumulated. All three now run the same code as the corresponding tools.

### Changed
- **Dedup refuses candidates that should not be collapsed.** `find_duplicates` skips memories the caller passed in `related_to`/`supersedes` (distinct by assertion) and any memory that is already a merge composite — merging into a memory that has absorbed others is how claims with different lifetimes end up in one record, which a later delete then cannot verify claim by claim. The `memory_dedup` scan additionally skips pairs joined by a `derived_from` or `supersedes` edge. A refused pair is reported as `possible_supersedes` instead of being silently left as two contradicting memories.
- **Server instructions cover curation.** Three rules with tests that can be applied mid-task: point at committed files instead of restating them (*if someone edits that file, does my memory become wrong?*); one memory, one lifetime (*would this claim survive a rewrite of the subsystem?*), splitting the durable half into a `pattern` linked by `derived_from`; and supersede rather than appending a contradicting memory.
- `auto_prune_dead_memories` is now `auto_prune_stale_memories`. It means "decayed to the floor and never accessed", which collides with `dead` now meaning "the subject is gone".
- The row mapper for `SELECT`s over `memories` is shared (`MEMORY_COLUMNS` + `map_memory_row`) instead of being written out at each of six call sites.

## [0.8.1] - 2026-07-28

### Fixed
- **Piping CLI output into a reader that exits early no longer panics.** `engram-cli projects | head` printed `failed printing to stdout: Broken pipe (os error 32)` and exited 101, because Rust ignores `SIGPIPE` and `println!` panics on `EPIPE`. Both binaries now restore the default `SIGPIPE` disposition, so the writer dies quietly (exit 141) as a pipeline expects. Adds a unix-only `libc` dependency for the one `signal()` call.

### Changed
- **Dependencies updated.** `rmcp` 1.6 → 2.2, which aligns the model types with the MCP 2025-11-25 spec: the `Annotated<RawResource>` / `RawResourceTemplate` wrappers are now flat `Resource` / `ResourceTemplate`, `Content` is `ContentBlock`, and `PromptMessageRole` is `Role`. No change to engram's own tool, resource, or prompt surface. Also `rusqlite` 0.39 → 0.40, `base64` 0.22 → 0.23, `rand` 0.9 → 0.10, `criterion` 0.5 → 0.8 (benches use `std::hint::black_box`), plus in-range updates for the rest of the tree.
- `hf-hub` 0.4 → 0.5, matching what `fastembed` already depends on. The tree previously carried two copies of the crate that resolves and caches the ONNX model; it now carries one.
- Integration tests resolve binaries via `CARGO_BIN_EXE_*` instead of guessing `target/release` before `target/debug`, so a stale binary from another profile can no longer be tested by accident.

## [0.8.0] - 2026-07-28

### Added
- **Optional `project` argument on every MCP tool.** Omitted, it resolves to the server's own project (`ENGRAM_PROJECT` or cwd) exactly as before; passed, that single call reads or writes another project's memories. Previously only the finer-grained `branch` was settable, so a server rooted above the repo being worked on silently stored memories under the parent directory. An unknown project ID is rejected with `MemoryError::UnknownProject` listing the known IDs rather than silently returning an empty result; a project counts as known if it has a `projects` row or owns at least one memory.
- **`memory_projects` MCP tool** and **`engram-cli projects`**: every project in the store with memory, handoff, and ADR counts plus last activity, ordered by recency. This is the discovery path for the `project` argument. `memory_projects` joins the `core` profile (now 15 tools; `full` is now 24).
- **`engram-cli --json`** for machine-readable output on `query`, `context`, `stats`, `projects`, `list`, `show`, `handoff resume`/`search`/`show`, and `adr list`/`show`. Memory objects use the same serde shape as export and MCP payloads. Empty results still render as JSON with `count: 0`. Commands that cannot render JSON exit 2 with a message instead of ignoring the flag, so a caller never parses prose by accident.
- **`--non-interactive` on `engram-cli handoff create` and `adr create`.** Prompting now also requires stdin to be a terminal, so CI jobs and agents holding an open stdin no longer block on a section prompt. Missing optional sections stay empty; missing required ones fail immediately naming the flag to pass (`--summary`, `--title`, `--decision`).
- Store results report where they landed: `memory_store` and `memory_store_batch` include `project`, `handoff_create` includes `project` and `branch`, `adr_create` includes `project`. The compact MCP output reads `Stored mem_… in <project>`.
- `memory_stats` accepts `project`; `memory://{project}/{id}` resources resolve any project in the store (a URI whose project does not match the memory's owner is an error).

### Changed
- **`engram` no longer starts a server when given arguments or a terminal.** It speaks MCP over stdio and has no subcommands, so `engram --help`, `engram store …`, or a bare run in a shell previously failed with `ConnectionClosed("initialize request")` — which reads as "there is no CLI". It now prints a usage notice pointing at `engram-cli` (exit 0 for `--help`/`-h`, 2 for an unsupported argument), prints the version for `--version`/`-V`, and serves as before when stdin is a pipe or `--stdio`/`--serve` is passed.
- Branch handling for a project that is not the server's own: the current git branch describes only the server's checkout, so `branch_mode: "current"` widens to all branches, `branch: "auto"` on `memory_store` resolves to global, and `handoff_create` for another project requires an explicit `branch`.
- `engram-cli` read commands with an explicit `--project` that does not exist now report the known projects and exit 1 instead of creating an empty project and returning nothing. Write commands still create on demand.

### Fixed
- **Misleading `missing field` errors on tool calls.** Input structs ignore unknown fields, so a misnamed argument was dropped silently and the failure surfaced as a bare `missing field \`type\`` on a call that looked like it did send `type`. All tool arguments now parse through `parse_args`, which names the tool and lists the fields actually received: `Invalid arguments for memory_store: missing field \`type\`. Fields received: content, kind, tags`. Field order never mattered — arguments arrive as a JSON object — and there is now a regression test asserting that with an 18KB content field in both orders.
- `memory_store` (and each `memory_store_batch` item) accepts `memory_type` as an alias for `type`, the spelling callers most often reach for.
- Interactive prompts terminate their line on EOF, so output no longer renders inside a `  > ` prompt.

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
