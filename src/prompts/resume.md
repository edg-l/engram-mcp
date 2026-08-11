You are resuming a work session. Call `handoff_resume` to load context from the most recent handoff(s) on this branch, then summarize what you learned and propose the next concrete action.

## Steps

1. Call `handoff_resume` with no arguments to use the current branch automatically:
   ```json
   {}
   ```
   Or pass an explicit branch if needed:
   ```json
   { "branch": "feat/my-feature" }
   ```

2. Read the result carefully:
   - `open_todos` / `open_blockers`: the newest handoff's todos and blockers verbatim, outside the similarity ranking. These are the open state of the work; treat them as still-pending unless you can verify otherwise.
   - `top_sections`: the most relevant section excerpts from recent handoffs, ranked by similarity to the session summary.
   - `chain`: the ordered list of handoff IDs from oldest to newest. If the chain has multiple entries, the work has spanned multiple sessions.
   - `linked_memories`: decisions, patterns, and debug notes that were auto-linked to the latest handoff. These provide broader project context.
   - `message`: if present, explains any branch detection issues (e.g. detached HEAD).

3. Summarize what you learned in two to four sentences covering:
   - What the previous session accomplished.
   - What is still open, from `open_todos` and `open_blockers`. Surface these explicitly even when `top_sections` did not rank them; a todo carried across several handoffs is the most likely thing to be dropped.
   - Any `tried` section in the results: approaches already ruled out. Do not re-attempt them.
   - The mental model needed to continue.

4. Propose the next concrete action: the exact first step the user should take, referencing specific files, functions, or commands where possible.

5. If the chain is long or there are many linked memories, offer to search for specific context with `handoff_search`.
