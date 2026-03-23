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
Use `memory_query` when you need to recall a specific piece of knowledge -- "what database does this project use?" or "why did we choose this approach?" This performs semantic search across stored memories.

# How to store well

- **Be specific and self-contained**: Write content that will make sense when retrieved later without surrounding context. Bad: "we decided to go with option 2". Good: "We chose server-side rendering over client-side SPA because the app is content-heavy with minimal interactivity."
- **Use 2-5 lowercase tags**: Tags improve search ranking. Use domain terms like "database", "auth", "deployment", "testing".
- **Set importance appropriately**: 0.3 = minor detail, 0.5 = normal (default), 0.7 = important decision or preference, 0.9 = critical constraint or blocker.
- **Choose the right type**: fact, decision, preference, pattern, debug, or entity. This helps with filtering and retrieval.
- **Use `pinned: true`** for permanent knowledge that should never decay or be pruned -- critical constraints, foundational decisions, or standing user preferences.
- **Use `global: true`** for knowledge that applies across all projects -- user preferences, environment facts, or universal conventions.

# Memory maintenance

- Memories automatically decay in relevance if not accessed -- important memories persist, trivial ones fade.
- Duplicates are automatically detected and merged when stored.
- Use `memory_prune` periodically to clean up low-relevance memories.
- Use `memory_dedup` to find and merge similar memories that weren't caught automatically.
