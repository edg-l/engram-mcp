# Working notes

## Current state

`master` holds a set of usability fixes driven by 30 days of Claude Code session logs
(2,221 engram calls, ~150 failures), from `d4bdf75` through `f741bbc`. Each commit passed
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

## Next steps

1. **Unify the import paths.** The MCP `memory_import` tool (`src/tools/handler.rs`,
   `fn memory_import`) is a separate implementation from `cmd_import` (`src/cli.rs`,
   used by `import` and `sync`). It skips ADRs whose number collides, ignores
   `project_aliases`, and likely diverges from the Sync rules in CLAUDE.md (LWW on
   `updated_at` with ties kept local, source `updated_at` preserved, max access
   counts, status/todo sidecars, model-mismatch re-embed, cluster re-assignment).
   Move the logic into one function in `src/tools/import.rs` taking db, embedding
   service, `ExportData` and a typed options struct (mode, target project,
   all-projects handling, sync origin). `cmd_import`, sync's pull, and the MCP
   handler all call it; delete the duplicate. List every behaviour difference and
   resolve each per CLAUDE.md. Existing import/sync tests must pass unchanged; add
   MCP-path tests for ADR renumbering, alias redirection, and LWW. Opus executor:
   this touches sync convergence.
   A partial attempt is in `git stash list` ("E2 partial …"): `cmd_import` moved
   into `src/tools/import.rs`, MCP handler not yet routed. Inspect with
   `git stash show -p stash@{0}`; apply it or drop it and start fresh.
2. **Final review.** One `code-reviewer` run on Opus over `70bb160..HEAD`, scoped to
   cross-commit seams (e.g. todo `detail` vs link resolution, argument tolerance vs
   the `sections` patch, aliases vs short-name resolution vs reconcile) and to the
   table above as the completeness checklist.
3. **Install and restart.** `cargo build --release`, install both binaries, restart
   every running `engram` server (it resolves its project once at startup).
4. **Check on the live store.** Run `engram-cli stats` inside `~/dev/Gardener` and
   confirm the one-line merge note (`~/dev/Gardener` → `git:github.com/XivHub/Gardener`,
   12 memories). Run `handoff_resume` in antworld and confirm the output fits inline
   (it was ~62 KB, 55 KB of it todos).
5. **Update the `read-handoffs` skill** (`~/.claude/skills/read-handoffs/SKILL.md`):
   `open_todos` is now `{id, title}` with `open_todo_count` and `stale_todo_count` /
   `stale_todo_ids`; step 5 should say to offer closing or dropping the stale ones.
6. **Release.** Bump the version and write the release commit once 1–4 are done.

## How to work here

- Session-log analysis: logs are `~/.claude/projects/*/*.jsonl`, kept 30 days.
  Extract engram calls with jq over `.message.content[] | select(.type=="tool_use")`
  joined to `tool_result` by `tool_use_id`; results over ~60 KB show as
  `persisted-output` with a 2 KB preview.
- Threshold changes are measured against the live store's embeddings, never picked by
  feel; record the numbers in the constant's doc comment.
- Never run anything that writes against `~/.local/share/engram/memories.db` from
  tests; use `ENGRAM_DB` with a temp path.
