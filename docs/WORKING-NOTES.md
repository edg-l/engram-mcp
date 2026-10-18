# Working notes

## Current state

`master` holds a set of usability fixes driven by 30 days of Claude Code session logs
(2,221 engram calls, ~150 failures), from `d4bdf75` through the review-fix commit. Each commit passed
`cargo test`, `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`. The
installed binaries and any running `engram` servers predate all of it.

What each commit is meant to contain, for checking completeness:

| Commit | Contents |
|---|---|
| `d4bdf75` | `SUPERSESSION_CANDIDATE_MIN` 0.75 → 0.88, from measured similarities (see its doc comment) |
| `e151d65` | `handoff_resume` `open_todos` as `{id, title}` + `open_todo_count`; renderer order status → count → blockers → sections → linked → todos, capped at 30 with "…and K more"; `todo_list` compact by default (`full_text` / `--full`); importance ordering applied before the cap |
| `b182756` | todo `text` capped at 200 chars on add/edit; `detail` stored as a linked `fact` (`detail_id`, `--detail`); stale todos (30+ store-days) counted in resume and `todo_list` |
| `9176d45` | `resolve_link_target`: `related_to`/`supersedes`/`memory_link` ids checked before any write, merged-away ids followed to the survivor, `redirected_links`; batch items now create their links |
| `39d486e` | `merge_memories` folds a consumed composite's `merged_from` into the survivor |
| `46119fd` | `memory_update` takes a partial `sections` patch for handoffs; parse failures name the canonical headings |
| `010fff7` | one update path (`src/tools/update.rs`) for MCP `memory_update` and `engram-cli update`; CLI `--sections-json` |
| `98d028e` | argument tolerance: section string/array, flat handoff fields, `type` defaults to `fact`, `memory_context` aliases, `todo_write` op aliases and `todos` error, comma-string `tags` |
| `3023f67` | `resolve_known_project`: short names (`antworld`) and legacy absolute paths resolve; ambiguous names list candidates |
| `7ee358b` | project identity: `root_path` recorded, `reconcile_identity` merges a repo's old id into its new one on startup/CLI run, `projects merge`, `project_aliases`, import redirects aliased projects |
| `f741bbc` | aliases followed on resolve; ADR numbers rendered from the sidecar; import renumbers colliding ADRs |
| "Route memory_import through the shared import path" | one import path, `import_payload` (`src/tools/import.rs`), for MCP `memory_import`, `engram-cli import` and `sync`; MCP import now merges LWW, preserves source `updated_at`, renumbers colliding ADRs, re-embeds foreign-model vectors; invalid `mode` is an error on both paths |
| "Fix the final review's findings; release 0.12.0" | reconcile absorbs only into `git:` ids and records a worktree's main checkout; startup ids follow aliases; import never edits another project and applies handoff/ADR sidecars on LWW updates, plus edge projects to the cache; `sections` patch keeps `continues_from` and multi-line items; dedup merge moves the consumed memory's edges to the survivor; ADR content edits rebuild the sidecar; todo `detail` failures reported as `detail_error`, `detail_id` rendered; `handoff_create` rejects empty summary and nested+flat sections; unknown `sections` patch fields rejected; CLI/resource ADR display from the sidecar |

## Next steps

1. **Install and restart.** `cargo build --release`, install both binaries, restart
   every running `engram` server (it resolves its project once at startup).
2. **Check on the live store.** Run `engram-cli stats` inside `~/dev/Gardener` and
   confirm the one-line merge note (`~/dev/Gardener` → `git:github.com/XivHub/Gardener`,
   12 memories). Run `handoff_resume` in antworld and confirm the output fits inline
   (it was ~62 KB, 55 KB of it todos).
3. **Update the `read-handoffs` skill** (`~/.claude/skills/read-handoffs/SKILL.md`):
   `open_todos` is now `{id, title}` with `open_todo_count` and `stale_todo_count` /
   `stale_todo_ids`; step 5 should say to offer closing or dropping the stale ones.
4. **Release.** Tag `v0.12.0` and publish once 1–2 are done; the version bump and changelog are committed.

## How to work here

- Session-log analysis: logs are `~/.claude/projects/*/*.jsonl`, kept 30 days.
  Extract engram calls with jq over `.message.content[] | select(.type=="tool_use")`
  joined to `tool_result` by `tool_use_id`; results over ~60 KB show as
  `persisted-output` with a 2 KB preview.
- Threshold changes are measured against the live store's embeddings, never picked by
  feel; record the numbers in the constant's doc comment.
- Never run anything that writes against `~/.local/share/engram/memories.db` from
  tests; use `ENGRAM_DB` with a temp path.
