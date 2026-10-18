Engram is your persistent memory across conversations. These instructions say when to reach for each tool; each tool's description says how to use it. The full guide is the `engram://guide` resource: read it before your first `memory_store` in a session, or when unsure where something belongs.

# At the start of every task
- Call `memory_context` with one sentence describing the task, before reading code or planning.
- When continuing earlier work, call `handoff_resume` with the current git branch. Its `open_todos` is the authoritative list of open work: close what you finish with `todo_write` as you go.

# Where each kind of knowledge goes
| You have | Use |
|---|---|
| A fact, decision and its reason, preference, pattern, debug finding, or person/system | `memory_store` with that `type` |
| Work a later session should pick up | `todo_write` op `add`; one-line `text`, long notes in `detail` |
| The state of a session that is ending | `handoff_create` |
| A formal, numbered architecture decision with a lifecycle (technology choice, structural constraint, API contract) | `adr_create`; lighter rationale is `memory_store type=decision` |

Do not store what the code, git history or a committed doc already says. Store the why, and what took effort to find out.

# Storing well
- Write each memory so it makes sense alone, months later: "Chose SSR over a SPA because the app is content-heavy", not "went with option 2".
- If the claim lives in a committed file, put the path in `external_artifacts` and store only what the file does not say. A copy goes stale when the file changes.
- Split a finding about current code from the general lesson it taught: store the lesson as `pattern` and link it to the finding with `memory_link` relation `derived_from`. They stop being true at different times.
- 2-5 lowercase tags. Importance 0.3 minor, 0.5 normal, 0.7 important, 0.9 critical. `pinned: true` for knowledge that must never fade, `global: true` for knowledge that applies to every project.

# When something stored is no longer true
- A replacement exists: `memory_store` the new memory with `supersedes: ["<old id>"]`. Never store a second, contradicting memory.
- The subject is gone with nothing replacing it: `memory_update` with `dead: true`.
- It is wrong or badly worded: `memory_update` with new `content`.
- Read `possible_supersedes` in every `memory_store` result: it is how you learn an old memory contradicts the new one.

# Before deleting
Prefer `supersedes` or `dead`. A memory may carry more than the claim it seems to be about, and `merged_from` may hold several absorbed memories. Deletions go to `memory_trash` and `memory_restore` brings them back, until retention expires.

# Other projects
Every tool takes `project`: a full id from `memory_projects` or its last segment (`payments` for `git:github.com/owner/payments`).
