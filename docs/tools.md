# MCP tools

Every tool takes an optional `project` argument; see [Project scoping](#project-scoping). The server's instructions tell an agent when to use which tool; the `engram://guide` resource has the full guidance on storing, superseding and curating memories.

## Handoffs

| Tool | What it does |
|---|---|
| `handoff_create` | Write a handoff with sections `summary` (required), `decisions`, `blockers`, `tried`, `mental_model`, `next_steps`, `notes`. Chains to the previous handoff on the branch with `continues_from`. |
| `handoff_resume` | The top-scoring sections across the branch's recent handoff chain, plus open todos, the newest handoff's blockers, and linked memories. `handoff_id` resumes a given handoff's chain instead of the newest, for two efforts sharing a branch. |
| `handoff_search` | Search handoff sections by content, optionally limited to a branch or to sections such as `tried`. |

## Todos

| Tool | What it does |
|---|---|
| `todo_write` | Apply `add`, `done`, `drop` (needs a `reason`), `reopen` and `edit` ops. Each op reports its own result, so one bad id does not discard the rest. `detail` on `add`/`edit` stores a longer note as a linked memory. |
| `todo_list` | List todos by state (`open`, `done`, `dropped`, `all`) and branch. Titles by default; `full_text: true` for everything. |

## Memories

| Tool | What it does |
|---|---|
| `memory_store` | Store a memory. Merges into a near-duplicate of the same type (cosine ≥ 0.90), and lists existing memories it may supersede. |
| `memory_store_batch` | Store up to 100 memories in one transaction. |
| `memory_context` | The memories most relevant to a description of the current task. |
| `memory_query` | Search, with filters on type, branch and relevance, and pagination. |
| `memory_list` | List without a query, filtered by type, tag or status (`live`, `superseded`, `dead`, `all`). |
| `memory_update` | Change content, tags, importance, pinned, global or dead status; patch a handoff's `sections`. Returns the previous version. |
| `memory_delete`, `memory_delete_batch` | Delete memories. Both go to the trash. |
| `memory_trash`, `memory_restore` | List the trash; restore a memory with its embedding and links. |
| `memory_link`, `memory_graph` | Link two memories (`relates_to`, `supersedes`, `derived_from`); walk the links from one. |
| `memory_promote` | Make a branch-local memory visible on every branch of its project. |
| `memory_prune`, `memory_dedup` | Remove low-relevance memories; merge duplicates. Both dry-run by default. |
| `memory_export`, `memory_import` | Export a project to JSON; import one, merging by last write wins on `updated_at`, or replacing. |
| `memory_stats`, `memory_projects` | Counts for one project; every project in the store. |

## ADRs

| Tool | What it does |
|---|---|
| `adr_create` | Create a numbered architecture decision record, optionally superseding an earlier one. |
| `adr_update_status` | Move an ADR between `proposed`, `accepted`, `deprecated` and `rejected`. |
| `adr_list`, `adr_show` | List ADRs, optionally by status; show one by number. |
| `adr_export` | Write ADRs as Markdown files (`docs/adr/` by default). Dry-run by default. |

## Memory types

| Type | For | Example |
|---|---|---|
| `fact` | Objective information (the default type) | "The API uses JWT authentication" |
| `decision` | A choice and its reason | "Chose SQLite over Postgres: one writer, no ops" |
| `preference` | How the user or project likes things done | "Prefer explicit error handling over unwrap" |
| `pattern` | A recurring approach or a lesson | "All handlers return Result<Json<T>, AppError>" |
| `debug` | A past issue and its cause | "OOM was caused by an unbounded channel buffer" |
| `entity` | People, systems, services | "UserService owns all auth logic" |
| `handoff` | A session snapshot | Written with `handoff_create` |
| `adr` | A numbered decision record | Written with `adr_create` |
| `todo` | An open work item | Written with `todo_write` |

## Storing

```json
{
  "content": "We chose PostgreSQL over SQLite for the API because of concurrent write requirements",
  "type": "decision",
  "tags": ["database", "api"],
  "importance": 0.7,
  "pinned": true,
  "external_artifacts": ["docs/adr/0007-postgres.md", "https://github.com/foo/bar/pull/42"]
}
```

- `importance`: 0.3 minor, 0.5 normal (default), 0.7 important, 0.9 critical.
- `pinned: true` exempts the memory from decay and pruning.
- `global: true` makes it visible in every project and clears its branch. `memory_update` sets or clears it on an existing memory.
- `external_artifacts` lists files, URLs or ids the memory refers to. Local paths that no longer exist are marked `[missing]` when the memory is retrieved. Pass `[]` to `memory_update` to clear the list.
- `supersedes: [id]` replaces an existing memory; retrieval then returns the new one in its place.

## Querying

```json
{
  "query": "what database do we use and why",
  "limit": 10,
  "types": ["decision", "fact"],
  "branch_mode": "current"
}
```

`branch_mode` is `current` (this branch plus global memories), `global`, `all`, or a branch name.

## Project scoping

A tool works on the project the server started in unless `project` names another:

```json
{ "context": "how does the payments service handle retries", "project": "payments" }
```

A project id is the repo's normalized git remote (`git:github.com/me/payments`), or its home-relative path (`~/dev/payments`) when it has no remote. A short name that matches the last segment of exactly one id resolves to it; an ambiguous one fails and lists the candidates. An unknown id is rejected with the known ids listed, so a typo cannot return an empty result. `memory_projects` lists them all.

Branch names belong to a repository, so for another project `branch_mode: "current"` covers every branch, and `handoff_create` needs an explicit `branch`.

## Tool profiles

`ENGRAM_MCP_TOOL_PROFILE` sets how many tools the server advertises, for clients where a long tool list costs context:

| Profile | Tools |
|---|---|
| `full` (default) | All 29 |
| `core` | 19: leaves out `handoff_search`, `memory_stats`, `memory_trash`, `memory_promote`, `memory_prune`, `memory_dedup`, `memory_export`, `memory_import`, `adr_update_status`, `adr_export` |
| `minimal` | 4: `memory_context`, `memory_store`, `handoff_resume`, `todo_write` |

A tool left out of the profile still runs when called, with a one-time warning.
