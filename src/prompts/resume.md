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
   - `open_todos`: the project's live open todo list for this branch, from the durable store rather than any handoff — `{id, title}` per item, ordered by importance then recency. `title` is a one-line summary; if a todo needs a longer finding recorded, it goes in `detail` via `todo_write`, not in `title`. This is the authoritative record of outstanding work. Present even when no handoff exists. The rendered checklist is capped at 30 items; `open_todo_count` is the true total, and `todo_list` shows the rest.
   - `stale_todo_count`/`stale_todo_ids`: open todos untouched for 30+ active store-days (oldest first, capped at 5). Call these out specifically — finish, edit, or drop them with a reason rather than letting them accumulate.
   - `open_blockers`: the newest handoff's unresolved blockers verbatim, outside the similarity ranking.
   - `top_sections`: the most relevant section excerpts from recent handoffs, ranked by similarity to the session summary.
   - `chain`: the ordered list of handoff IDs from oldest to newest. If the chain has multiple entries, the work has spanned multiple sessions.
   - `linked_memories`: decisions, patterns, and debug notes that were auto-linked to the latest handoff. These provide broader project context.
   - `message`: if present, explains any branch detection issues (e.g. detached HEAD).

3. Summarize what you learned in two to four sentences covering:
   - What the previous session accomplished.
   - What is still open, from `open_todos` and `open_blockers`. List the open todos shown — they are not subject to the similarity ranking, so they are complete up to the 30-item cap; call `todo_list` if `open_todo_count` says there are more.
   - Any `tried` section in the results: approaches already ruled out. Do not re-attempt them.
   - The mental model needed to continue.

4. Propose the next concrete action: the exact first step the user should take, referencing specific files, functions, or commands where possible.

5. **Reconcile the todo list as you work, not only at the end.** Call `todo_write` to mark items `done` as you finish them, and `drop` with a reason when one no longer applies. A list that only grows stops being trusted and then stops being read, so leaving a finished item open is worse than never having added it. Add new items for work that a *later* session should pick up — not for steps you are about to take in this one.

6. If the chain is long or there are many linked memories, offer to search for specific context with `handoff_search`.
