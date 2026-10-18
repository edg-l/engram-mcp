---
name: read-handoffs
description: Resume a session by loading recent handoffs from Engram. Use at the start of a session or when the user wants to review what was done previously.
disable-model-invocation: true
---

Load session context from Engram via the `mcp__engram__handoff_resume` MCP tool. Engram is the canonical store; legacy `.claude/handoff/*.md` files were ported via `engram-port-handoffs` and are no longer read.

## Steps

1. **Call `mcp__engram__handoff_resume`**, passing `branch` from `git branch --show-current`. Other defaults: `max_sections: 5`, `include_off_branch: false`.

   Pass the branch explicitly for the same reason `handoff` writes it explicitly: the server falls back to the branch of the directory it was launched in, so a session started in the main clone but working in a worktree would resume the wrong branch and report "no prior handoffs". Omit `branch` only when not in a git repo.

2. **Inspect the result.** Engram returns:
   - `branch` — the resolved branch (or `null` if detached HEAD)
   - `latest_handoff_id` — most recent handoff on this branch
   - `chain` — handoff ids ordered oldest-to-newest via `continues_from` (capped at depth 5, cycle-detected)
   - `open_todos` — the project's live open todo list for this branch, from the durable store rather than any handoff, as `{id, title}` items ordered by importance, then most recently updated. Returned outside the similarity ranking, and present even when no handoff exists
   - `open_todo_count` — the true number of open todos; `open_todos` is capped, so this can be larger
   - `stale_todo_count`, `stale_todo_ids` — open todos untouched for 30+ active store-days, oldest first (at most 5 ids)
   - `open_blockers` — the newest handoff's unresolved blockers verbatim, also outside the ranking
   - `top_sections` — highest-scoring sections across the chain, each with `handoff_id`, `section_name`, `section_text`, `score`
   - `linked_memories` — decision/pattern/debug memories the latest handoff links to via `derived_from`
   - `message` — only present when branch could not be resolved

3. **Handle the empty case.** If `chain` is empty AND `latest_handoff_id` is `null`, say "No prior handoffs on this branch." Then call `mcp__engram__handoff_resume` again with `include_off_branch: true` to surface handoffs from other branches as background. Note that `open_todos` is still meaningful here — a project can have todos before its first handoff, so report them even when no handoff was found.

4. **Handle detached HEAD.** If `message` is set, tell the user no current branch was detected and present whatever off-branch results came back, flagged as such.

5. **Present to the user**, in this order:
   - **Status line**: `Resuming \`<branch>\`, <chain length> handoff(s) in chain, latest from <handoff id>`
   - **Open todos / blockers**: print `open_todos` (title and id) and `open_blockers` in full, before anything else. They are the open state of the work and are not subject to the ranking. When `open_todo_count` exceeds the items returned, say how many more there are and that `mcp__engram__todo_list` lists them. `open_todos` is the durable list — treat it as the authoritative record of outstanding work, and reconcile it with `mcp__engram__todo_write` as you go rather than only at session end.
   - **Stale todos**: when `stale_todo_count` is above 0, name the stale ids and offer to finish, edit, or drop each one (a drop needs a reason). A todo nobody touches for a month of active work is usually done or dead.
   - **Top sections**: summarize each `top_sections` entry. Group by `handoff_id` if multiple sections come from the same handoff. Quote the strongest section text verbatim; paraphrase weaker matches. Call out any `tried` section as approaches already ruled out.
   - **Linked memories**: bullet list of `linked_memories` with their type and content preview ("Related decision: ...", "Related debug: ...").

6. **Pair with `mcp__engram__memory_context`.** Call it with a short description of the inferred current task (derived from `top_sections` and `linked_memories`). Surface any additional memories that didn't come through the handoff chain.

7. **Closing note.** End with a one-liner: which handoff in the chain is the working starting point, and whether you followed any cross-references the user might want expanded.

## Why this matters

The handoff chain encodes session-to-session continuity: each `continues_from` link means "the next agent should pick up from here". The top-sections retrieval is hybrid (similarity + recency); a single old but highly-relevant section can outrank newer but generic content. Trust the ranking; do not just present the latest handoff verbatim.

## Specific lookups

When several efforts share a branch and the user names one, pass its handoff id as `handoff_id` to `mcp__engram__handoff_resume`: it walks that handoff's chain instead of the newest on the branch.

If the user asks about a specific past handoff (by date, by topic), use `mcp__engram__handoff_search` with a query string. Filter by section with `section_filter: ["blockers"]` etc. when the user is asking targeted questions like "have we hit this kind of error before?".
