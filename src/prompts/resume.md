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
   - `top_sections`: the most relevant section excerpts from recent handoffs, ranked by similarity to the session summary. Start here.
   - `chain`: the ordered list of handoff IDs from oldest to newest. If the chain has multiple entries, the work has spanned multiple sessions.
   - `linked_memories`: decisions, patterns, and debug notes that were auto-linked to the latest handoff. These provide broader project context.
   - `message`: if present, explains any branch detection issues (e.g. detached HEAD).

3. Summarize what you learned in two to four sentences covering:
   - What the previous session accomplished.
   - What blockers exist: things preventing forward motion right now (missing access, failing dependency, unanswered question).
   - The mental model needed to continue.

4. Propose the next concrete action: the exact first step the user should take, referencing specific files, functions, or commands where possible.

5. If the chain is long or there are many linked memories, offer to search for specific context with `handoff_search`.

6. **If the result doesn't match what you were asked to resume**, don't assume it's right just because it's the latest. Handoffs are keyed by branch, not by task — two independent sessions on the same branch will otherwise collide. Check for other recent handoffs (`handoff_search`, or ask the user) and, if there's more than one plausible thread, ask which to resume rather than silently picking the newest, e.g.: "I found 2 recent handoffs on this branch: topic `auth-refactor` (2h ago) and topic `db-migration` (10m ago) — which should I resume?" Once you know, call `handoff_resume` again with `topic: "<slug>"` to scope to that thread, or `handoff_id: "<id>"` to resume a specific handoff directly.
