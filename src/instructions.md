Engram is your persistent memory. Use it to remember knowledge across conversations.

# When to use memory

## Always call `memory_context` first
At the start of every conversation or task, call `memory_context` with a brief description of what you're about to do. This loads relevant background knowledge -- decisions, preferences, patterns, and facts from prior conversations. Do this before reading code or making plans. Think of it like checking your notes before starting work.

## Store memories as you learn
Whenever you discover something worth remembering across conversations, call `memory_store`. Good things to store:

- **Decisions and rationale**: "We chose PostgreSQL over SQLite for the API because of concurrent write requirements" (type: decision, importance: 0.7)
- **User preferences**: "User prefers short commit messages with no Co-Authored-By lines" (type: preference, importance: 0.7)
- **Project facts**: "The auth service runs on port 8443 and requires mTLS" (type: fact)
- **Recurring patterns**: "Integration tests in this repo need Docker running for the database container" (type: pattern)
- **Debug findings**: "The flaky CI test was caused by a race condition in the connection pool teardown" (type: debug)
- **People and systems**: "Alice owns the payments service, Bob owns the auth service" (type: entity)

Do NOT store things that are obvious from reading the code, git history, or documentation. Store the *why* behind decisions, the context that would be lost, and knowledge that took effort to discover.

## Query for specific lookups
Use `memory_query` when you need to recall a specific piece of knowledge -- "what database does this project use?" or "why did we choose this approach?" This performs semantic search across stored memories. Use `content_length` to control how much content is shown per result (default 300). Increase it when you need full details, decrease it when scanning many results.

# How to store well

- **Be specific and self-contained**: Write content that will make sense when retrieved later without surrounding context. Bad: "we decided to go with option 2". Good: "We chose server-side rendering over client-side SPA because the app is content-heavy with minimal interactivity."
- **Use 2-5 lowercase tags**: Tags improve search ranking. Use domain terms like "database", "auth", "deployment", "testing".
- **Set importance appropriately**: 0.3 = minor detail, 0.5 = normal (default), 0.7 = important decision or preference, 0.9 = critical constraint or blocker.
- **Choose the right type**: fact, decision, preference, pattern, debug, or entity. This helps with filtering and retrieval.
- **Use `pinned: true`** for permanent knowledge that should never decay or be pruned -- critical constraints, foundational decisions, or standing user preferences.
- **Use `global: true`** for knowledge that applies across all projects -- user preferences, environment facts, or universal conventions.

## Point at files, do not copy them

If the claim already lives in a committed file, store the pointer and only the part the file does not say. Put the path in `external_artifacts` and write the reasoning, the count, or the lesson in the content.

The test: **if someone edits that file, does my memory become wrong?** If yes, you copied instead of pointing. A memory restating a document goes stale the moment the document changes, silently, and nothing will tell you.

## One memory, one lifetime

A finding about current code and the general lesson it taught are two memories, not one. They stop being true at different times: the finding dies when the code is rewritten, the lesson does not.

The test: **would this claim survive a rewrite of the subsystem?** If part of it would and part of it would not, split it. Store the durable half as `pattern` and link it with `derived_from`. Memories joined that way are never auto-merged, and neither are memories of different types.

This matters most when something is later deleted. Checking "is this recoverable from git?" is only valid if every claim in the memory is the same kind of claim; a record that mixes a stale conclusion with a durable rule will pass the check on the conclusion and lose the rule.

# Working across projects

Every tool defaults to the project the server was launched in. To read or write another project's memories, pass its ID as the `project` argument (e.g. `memory_context` with `project: "/home/me/dev/other-repo"`). Call `memory_projects` to list the available project IDs with their memory, handoff, and ADR counts; an unknown ID is rejected with the known ones listed. Since branch names are per-repository, `branch_mode: "current"` covers all branches when the target is another project, and `handoff_create` for another project requires an explicit `branch`.

# When something you stored is no longer true

Do not store a second, contradicting memory. Two memories that disagree both keep surfacing, and the next agent has no way to tell which one is current.

- **A replacement exists** -> store the new memory with `supersedes: ["<old id>"]`. The old memory stops being returned, and searches that would have matched it return the new one instead, marked with what it replaced. The old text stays readable via `memory_list status=superseded`, so the history is not lost.
- **The subject is gone entirely** (service retired, file deleted, approach abandoned with nothing in its place) -> `memory_update` with `dead: true`. Dead memories are excluded from retrieval outright. Prefer supersession whenever there is something to point at: returning nothing reads as "nobody looked into this", which invites the same wrong conclusion to be reached again.
- **It is simply wrong or badly worded** -> `memory_update` with new content. This replaces the content wholesale rather than patching it; the result hands back the previous version, and the old one stays recoverable.

Every `memory_store` result may include `possible_supersedes`: existing memories close enough to be about the same subject but not close enough to have been merged. Read it. That is usually the only way you will learn that a memory from months ago contradicts what you just stored, because a memory that never surfaces in search never becomes a candidate for anything.

# Memory maintenance

- Memories automatically decay in relevance if not accessed -- important memories persist, trivial ones fade.
- Duplicates are automatically detected and merged when stored.
- Use `memory_list` to see what is actually in the store. Search only shows what a query matches, so the memories most in need of attention are the ones it never surfaces. `status: "superseded"` and `status: "dead"` show what retrieval is hiding.
- Use `memory_prune` periodically to clean up low-relevance memories.
- Use `memory_dedup` to find and merge similar memories that weren't caught automatically.

## Deleting is a last resort, and it is not free

Prefer supersession or `dead` over deletion. A memory is often carrying more than the claim it appears to be about -- a lesson, a count of how often something has happened, context that exists nowhere else. Verifying "this conclusion is in git" does not verify the rest of it.

Before deleting, read the whole memory, including `merged_from`: dedup may have folded several other memories into it, and one id can stand for several distinct claims.

`memory_delete`, `memory_delete_batch`, `memory_prune`, and dedup merges all return what they destroyed and keep a recoverable snapshot. `memory_trash` lists them; `memory_restore` puts one back with its embedding and relationships. Snapshots expire, so recovering something is only possible for as long as the retention window.

# Session handoffs

Use `handoff_create` at session end and `handoff_resume` at session start to preserve and restore working context across sessions. Handoffs capture structured sections (summary, decisions, todos, blockers, mental model, next steps, notes) and are pinned so they never decay. Use `handoff_search` to find specific section content across past sessions.

Section semantics: **todos** — Within-session work the next agent should pick up immediately. Concrete, ready-to-execute items. **blockers** — Things preventing forward motion right now (missing access, failing dependency, unanswered question). **next_steps** — Post-session follow-ups beyond the current thread. Future-facing, not for immediate pickup.

# Architecture decisions

Use `adr_create` for formal, numbered, status-tracked architecture decisions that warrant a durable record (technology choices, structural constraints, API contracts). ADRs are project-global, pinned, and exempt from decay. Use `memory_store type=decision` for lightweight rationale or in-the-moment choices that do not need a formal lifecycle (proposed/accepted/superseded/deprecated/rejected).
