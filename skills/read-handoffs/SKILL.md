---
name: read-handoffs
description: Resume a session by loading recent handoffs from Engram. Use at the start of a session or when the user wants to review what was done previously.
disable-model-invocation: true
---

Load session context from Engram via the `mcp__engram__handoff_resume` MCP tool. Engram is the canonical store; legacy `.claude/handoff/*.md` files were ported via `engram-port-handoffs` and are no longer read.

## Steps

1. **Call `mcp__engram__handoff_resume`** with no arguments. Defaults: current branch, `max_sections: 5`, `include_off_branch: false`.

2. **Inspect the result.** Engram returns:
   - `branch` — the resolved branch (or `null` if detached HEAD)
   - `latest_handoff_id` — most recent handoff on this branch
   - `chain` — handoff ids ordered oldest-to-newest via `continues_from` (capped at depth 5, cycle-detected)
   - `top_sections` — highest-scoring sections across the chain, each with `handoff_id`, `section_name`, `section_text`, `score`
   - `linked_memories` — decision/pattern/debug memories the latest handoff links to via `derived_from`
   - `message` — only present when branch could not be resolved

3. **Handle the empty case.** If `chain` is empty AND `latest_handoff_id` is `null`, say "No prior handoffs on this branch." Then call `mcp__engram__handoff_resume` again with `include_off_branch: true` to surface handoffs from other branches as background.

4. **Handle detached HEAD.** If `message` is set, tell the user no current branch was detected and present whatever off-branch results came back, flagged as such.

5. **Present to the user**, in this order:
   - **Status line**: `Resuming \`<branch>\`, <chain length> handoff(s) in chain, latest from <handoff id>`
   - **Top sections**: summarize each `top_sections` entry. Group by `handoff_id` if multiple sections come from the same handoff. Quote the strongest section text verbatim; paraphrase weaker matches.
   - **Open todos / blockers** from the latest handoff: surface these explicitly so the user immediately sees what's pending.
   - **Linked memories**: bullet list of `linked_memories` with their type and content preview ("Related decision: ...", "Related debug: ...").

6. **Pair with `mcp__engram__memory_context`.** Call it with a short description of the inferred current task (derived from `top_sections` and `linked_memories`). Surface any additional memories that didn't come through the handoff chain.

7. **Closing note.** End with a one-liner: which handoff in the chain is the working starting point, and whether you followed any cross-references the user might want expanded.

## Why this matters

The handoff chain encodes session-to-session continuity: each `continues_from` link means "the next agent should pick up from here". The top-sections retrieval is hybrid (similarity + recency); a single old but highly-relevant section can outrank newer but generic content. Trust the ranking; do not just present the latest handoff verbatim.

## Specific lookups

If the user asks about a specific past handoff (by date, by topic), use `mcp__engram__handoff_search` with a query string. Filter by section with `section_filter: ["blockers"]` etc. when the user is asking targeted questions like "have we hit this kind of error before?".
