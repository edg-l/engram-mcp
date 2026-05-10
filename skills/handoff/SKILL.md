---
name: handoff
description: Save a session handoff to Engram for future sessions. Captures decisions, todos, blockers, mental model, and next steps as a structured memory. Use when the user wants to preserve session context before ending work.
disable-model-invocation: true
---

Capture session state into Engram via the `mcp__engram__handoff_create` MCP tool. Engram is now the canonical store; no markdown file is written.

## Steps

1. **Gather working state** (skip if not in a git repo). Run in parallel:
   - `git branch --show-current`
   - `git log --oneline -5`
   - `git status --short`
   - `git diff --stat`
   - `gh pr view --json url,number,state 2>/dev/null`

2. **Build the handoff payload.** Map session content to these sections. `summary` is required; everything else is optional. Omit sections with nothing real to say; do not pad.

   - `summary` (string): 1–3 sentences naming what was worked on.
   - `decisions` (string array): architectural / design choices made this session, one per item, with rationale.
   - `todos` (string array): remaining tasks. Specific: file path, function name, the next concrete action.
   - `blockers` (string array): approaches tried that did not work, with why each failed. So the next agent does not retry them.
   - `mental_model` (string): prose paragraph describing how the system / problem fits together right now. Use when the next agent needs the model in their head.
   - `next_steps` (string array): ordered/prioritized actions, closely related to `todos`. Use `next_steps` for "what to do next"; use `todos` for "what's still open".
   - `notes` (string, optional): open questions for the user, env quirks, pre-existing failures, anything that doesn't fit the other sections.

3. **Detect `continues_from`.** If the `read-handoffs` skill or `mcp__engram__handoff_resume` ran earlier this session and returned a `latest_handoff_id`, pass that id as `continues_from`. Otherwise omit.

4. **Sensitive data filter.** Before calling the tool, scrub: API tokens, passwords, private URLs, customer data, internal hostnames. If unsure, ask the user.

5. **Call `mcp__engram__handoff_create`** with the payload. Defaults: `pinned: true`, `importance: 0.85`, `auto_link: true`. Branch is auto-detected from git; pass an explicit `branch` only if you want to override.

6. **Report to the user.** Print the new handoff id, the auto-linked memory count (decisions/patterns/debug memories Engram associated by similarity), and the `continues_from` link if one was set. Ask if any section should be revised; if so, the user can dictate edits and you can call `mcp__engram__memory_update` on the just-created handoff.

7. **Escalate single insights worth keeping outside the handoff.** Only do this for content that should surface in `memory_context` queries on unrelated future tasks. Call `mcp__engram__memory_store` separately for:
   - `type: decision` — architectural choice with rationale (importance 0.7)
   - `type: pattern` — recurring gotcha / convention discovered (importance 0.7)
   - `type: debug` — root cause of a tricky bug (importance 0.6)
   The handoff already captures session-level context; only escalate insights with cross-session value.

## Failure modes

- **Detached HEAD / non-git workspace**: `handoff_create` rejects with `MemoryError::InvalidType` because branch is required. Ask the user for a branch name and pass it explicitly via the `branch` argument.
- **Empty payload**: at least one section must be non-empty. If you have nothing real to say, do not call the tool — tell the user the session has no state worth handing off.

## Viewing handoffs later

- `engram-cli handoff show <id>` — render a specific handoff
- `engram-cli handoff resume` — load context for the current branch
- `engram-cli handoff search "query"` — semantic search across handoff sections
- `/mcp__engram__resume` — same as `handoff resume`, via slash command in Claude Code
