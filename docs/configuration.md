# Configuration

Engram reads environment variables only; there is no config file. The MCP server takes no arguments, so set these in the environment your client starts it with.

| Variable | Default | Effect |
|---|---|---|
| `ENGRAM_DB` | `~/.local/share/engram/memories.db` | Path of the SQLite store. |
| `ENGRAM_PROJECT` | derived from the working directory | Fixes the project id instead of deriving it from the git remote or path. |
| `ENGRAM_BRANCH` | detected from git | What "current branch" means, for a server or shell started in a different checkout than the work. |
| `ENGRAM_SEARCH_MODE` | `hybrid` | Retrieval for the MCP server's `memory_query` and `memory_context`: `hybrid`, `bm25` or `vector`. |
| `ENGRAM_MAX_CANDIDATES` | `500` | Memories `memory_context` scores, most recently accessed first. Pinned memories are always included. |
| `ENGRAM_DEDUP_THRESHOLD` | `0.90` | Cosine similarity at which a new memory merges into an existing one of the same type. Clamped to 0.5–1.0. |
| `ENGRAM_DECAY_INTERVAL` | `3600` | Seconds between the server's maintenance passes: relevance, auto-prune, trash sweep. |
| `ENGRAM_RECLUSTER_INTERVAL` | `21600` | Seconds between re-clustering passes. |
| `ENGRAM_TRASH_RETENTION_DAYS` | `30` | Days a deleted memory stays restorable. `0` keeps it forever. |
| `ENGRAM_ADR_DIR` | `docs/adr` | Where `adr_export` writes files, unless the call passes `dir`. |
| `ENGRAM_EMBED_SHARED` | on | `0` loads the embedding model in every process instead of sharing one background process. |
| `ENGRAM_EMBED_IDLE_SECS` | `900` | Seconds without a request before the shared embedding process exits. |
| `ENGRAM_MCP_TOOL_PROFILE` | `full` | How many tools the server advertises: `full`, `core` or `minimal`. See [tool profiles](tools.md#tool-profiles). |

Both maintenance passes wait one interval before they first run, so a server that lives less than an hour never runs one. `engram-cli decay` runs the same pass on demand, for example from a timer.

The hook variables (`ENGRAM_HOOK_*`) are listed in [hooks/README.md](../hooks/README.md#env-vars).
