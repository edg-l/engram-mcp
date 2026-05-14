# LongMemEval-S benchmark harness

Measures engram-mcp's retrieval quality (R@k, MRR) on the LongMemEval-S cleaned dataset.

## Quick start

```bash
bash scripts/fetch-longmemeval.sh
cargo build --release --bin longmemeval-bench
./target/release/longmemeval-bench \
  --dataset benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  --questions 30 --session-limit 0 --seed 42 --mode hybrid
```

The dataset (~277 MB) is fetched on first run, hash-verified, and `.gitignore`d. SHA256 is pinned in `scripts/fetch-longmemeval.sh`.

## CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--dataset <PATH>` | required | Path to `longmemeval_s_cleaned.json`. |
| `--questions <N\|all>` | `30` | Number of seeded-sampled questions to evaluate. |
| `--session-limit <N>` | `10` | Cap sessions per question (0 = use all). See "Session limit" below. |
| `--api <query\|context>` | `query` | Which MCP API to evaluate. `context` is stubbed until Phase 6 ships. |
| `--mode <vector\|bm25\|hybrid>` | `hybrid` | Search mode passed to `ToolHandler` via constructor injection. |
| `--seed <u64>` | `42` | Sampling seed for question selection. |
| `--out <DIR>` | `benchmarks/longmemeval/results/` | Output directory for markdown + JSON reports. |

## Output

Two files per run, timestamped:

- `{mode}-{api}-{rfc3339}.md` — human-readable summary table.
- `{mode}-{api}-{rfc3339}.json` — flat object containing all metrics and run metadata.

## Session limit

LongMemEval-S questions have 50-55 sessions in their haystack (~225-275 turn-pairs total per question). The answer can be in any of them. Truncating with `--session-limit 10` will cause most questions to miss their answer entirely — initial smoke testing with `session_limit=5` gave 0% R@5. Use `session_limit=0` for honest numbers.

## Runtime estimates

Measured locally on release build, mdbr-leaf-ir q8 embedder:

| Slice | Per-mode wall | 3-mode wall |
|---|---|---|
| 30 questions × `session_limit=0` | ~130-180s | ~7-9 min |
| 500 questions × `session_limit=0` | ~35-50 min | ~2-2.5 hours |

The dominant cost is ingest embedding (each turn-pair embedded once); querying is fast.

## What gets measured

- `partial-R@k` — fraction of questions where any answer session appears in top-k retrieved.
- `full-R@k` — fraction where all answer sessions appear in top-k. 324 of 500 questions have ≥2 answer sessions.
- `MRR` — mean reciprocal rank over top-10, using the earliest matching session.

The "session id" of a retrieved memory is extracted from its `session:{id}` tag (assigned at ingest).

## Ingest bypass

The harness calls `Database::store_memory` + `Database::store_embedding` directly, NOT the MCP tool handler. The tool handler fires auto-dedup at cosine similarity ≥ 0.90 and contradiction detection at ≥ 0.85, which would silently merge topically-similar turn-pairs and corrupt recall numbers. The bypass is documented in `src/ingest.rs:1-17` and guarded by the regression test `tests/longmemeval_ingest_bypass.rs`.

## Dataset license

LongMemEval is MIT-licensed. We do not vendor it; the fetch script downloads from Hugging Face on demand.

## Reproducibility

- Same `--seed` produces the same sampled questions.
- Embedding model is deterministic.
- Ingest order is deterministic (haystack array order).
- RRF tie-breaks deterministically by ascending memory id.
- Numbers should match within rounding across runs on the same machine; cross-machine variance is bounded by embedder numerics.

Per-run metadata (crate version, dataset SHA, optional `GIT_SHA` env var) is captured in both the markdown footer and the JSON output.
