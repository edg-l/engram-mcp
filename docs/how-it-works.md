# How it works

## Handoffs and todos

A handoff is a memory with seven sections: `summary`, `decisions`, `blockers`, `tried`, `mental_model`, `next_steps`, `notes`. Each section gets its own embedding, so resuming can return the one `tried` entry that matters without the rest of the document. A new handoff points at the previous one on its branch through `continues_from`, and `handoff_resume` walks that chain back up to five handoffs, ranking sections by similarity and recency.

Each section is also compared with the project's `decision`, `pattern` and `debug` memories; matches at cosine ≥ 0.75 are linked, up to 10 per handoff, and come back with the resume.

Open work is not a handoff section. A handoff is a snapshot, and restating todos in every snapshot cannot tell a finished item from a forgotten one: both just stop appearing. Todos therefore live on their own list, where each has an id and is closed explicitly, as `done` or as `dropped` with a reason. `handoff_resume` returns the open list every time, outside the ranking, along with the newest handoff's blockers. A todo untouched for 30 active days is flagged as stale.

`tried` is not carried forward. A dead end stays a dead end, so it is recorded once and found later with `handoff_search` filtered to `tried`.

## Retrieval

`memory_query` and `memory_context` rank with BM25 (SQLite FTS5) and cosine similarity over local embeddings, fused by reciprocal rank (k = 60). In the default hybrid mode `memory_context` then weights each result by recency and importance:

```
score = base × (0.6 + 0.2 × recency + 0.2 × importance)
recency = exp(−0.02 × elapsed)
```

`memory_query` multiplies its score by the memory's relevance (see [decay](#decay)).

For a project with 10 or more memories, `memory_context` scores cluster centroids first and draws from the best clusters, which spreads results across topics instead of returning five near-copies. It scores at most `ENGRAM_MAX_CANDIDATES` memories, the most recently accessed, plus every pinned one.

Embeddings come from [mdbr-leaf-ir](https://huggingface.co/onnx-community/mdbr-leaf-ir-ONNX), quantized, cut to 256 dimensions. It runs in-process through ONNX Runtime.

### Benchmark

On a 30-question seeded slice of [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval), with each question's full haystack ingested (about 250 turn pairs), measured at crate version 0.4.0:

| Mode | partial-R@5 | full-R@5 | MRR |
|---|---:|---:|---:|
| Vector | 50.0% | 23.3% | 0.326 |
| BM25 | **93.3%** | **73.3%** | **0.883** |
| Hybrid (default) | 93.3% | 70.0% | 0.828 |

LongMemEval questions name specific entities, dates and numbers, which FTS5 matches exactly; that is why BM25 leads here. Hybrid stays the default because 30 questions are too few to justify switching, and the weak vector result points at the embedder rather than the fusion. [benchmarks/longmemeval/RESULTS.md](../benchmarks/longmemeval/RESULTS.md) has the full table and method.

## Decay

Every unpinned memory has a relevance between 0.1 and 1.0:

```
relevance = exp(−decay_rate × elapsed) × (0.5 + 0.5 × importance) + usage_boost
usage_boost = 0.1 × min(1, ln(1 + accesses) / ln(51)) × exp(−0.02 × elapsed)
```

`elapsed` counts store-days, not calendar days: days on which the memory's project received a store, after the memory was last accessed. What makes a memory stale is newer knowledge about the same project arriving, so a project nobody touches does not age at all, and an active one ages at its own pace. Captures made by hooks do not advance the clock. With the default `decay_rate` of 0.01, relevance halves over about 69 active days.

Pinned memories stay at 1.0. A memory at the 0.1 floor that was never accessed, is neither pinned nor global, and is over 30 days old is pruned automatically, into the trash.

## Deduplication and supersession

A new memory at cosine ≥ 0.90 to an existing memory of the same type merges into it: tags combine, the higher importance wins, and the absorbed text is kept in `merged_from`. Between a global and a local memory, the global one survives. The server also merges duplicates within each cluster on its re-clustering pass.

Some pairs never merge: memories of different types, a memory into one that has already absorbed others, and any memory the caller linked through `related_to` or `supersedes`. A near-match that is not merged is returned as `possible_supersedes`, because similarity cannot tell "contradicts" from "elaborates"; the caller decides.

Superseding a memory keeps it. Retrieval returns its successor in its place, marked with what it replaced, so a question answered once does not get re-derived from the stale answer. A memory marked dead, whose subject is gone with no successor, is left out of retrieval entirely.

Deleting, pruning, merging and overwriting all put a snapshot in the trash, with the memory's embedding and links, for 30 days by default.

## Project identity

A project id is the repo's git remote, normalized so that `git@github.com:me/repo.git` and `https://github.com/me/repo` are the same id (`git:github.com/me/repo`). A repo without a remote uses its path relative to the home directory (`~/dev/repo`), which matches across machines with different user names.

Every worktree of a repo resolves to its main checkout, so all of them are one project. Engram records the directory each id came from. When a repo gains a remote, or its remote changes, the next run in that directory merges the old project into the new one and prints a line saying so. A directory without a remote never absorbs another project.

## Sync

`engram-cli sync user@host` runs one `ssh` process per direction and moves JSON over stdin and stdout. There is no daemon and no hub. Each remote has two watermarks, one per direction, taken from the newest `updated_at` that actually moved, so a clock difference between the machines cannot skip a row.

Conflicts resolve by last write wins on `updated_at`; on a tie the local copy stays. Access counts take the larger of the two sides. Relevance is recomputed locally. A memory that just arrived from a host is not sent back to it on the next push.

Deletions, the trash and clusters do not sync. A project merge is local to the machine it ran on; the other machine merges the same way the next time Engram runs in that repo there.
