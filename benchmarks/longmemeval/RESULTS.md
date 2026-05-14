# LongMemEval-S benchmark results

Retrieval quality for engram-mcp's three search modes, measured on a 30-question seeded subset of the LongMemEval-S cleaned dataset (500 total questions).

## Run configuration

- Dataset: `longmemeval_s_cleaned.json`, SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- Sample: 30 questions, seed 42 (deterministic), `session_limit = 0` (full haystack ingested, ~225-275 turn-pairs per question)
- API: `memory_query` (Phase 6 adds `memory_context` coverage)
- Embedder: mdbr-leaf-ir q8 (256-dim MRL)
- Crate version: 0.4.0

## Results

| Mode | partial-R@1 | partial-R@5 | partial-R@10 | full-R@1 | full-R@5 | full-R@10 | MRR | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vector | 23.3% | 50.0% | 60.0% | 6.7% | 23.3% | 40.0% | 0.326 | 176s |
| BM25 | **83.3%** | **93.3%** | **96.7%** | **13.3%** | **73.3%** | **83.3%** | **0.883** | 131s |
| Hybrid (RRF k=60) | 76.7% | 93.3% | 96.7% | 6.7% | 70.0% | 76.7% | 0.828 | 133s |
| Hybrid context (RRF k=60) | 76.7% | 93.3% | 96.7% | 6.7% | 66.7% | 76.7% | 0.826 | 139s |

Partial hit = any answer session present in top-k. Full hit = all answer sessions present. MRR uses the rank of the earliest matching session in top-10.

The "Hybrid context" row uses `--api context` (`memory_context` tool, flat path, no clusters). All other rows use `--api query` (`memory_query` tool). API, seed, and session-limit are identical across rows.

## Findings

1. **BM25 dominates on this slice.** LongMemEval questions reference specific entities, dates, and numbers from the haystack; FTS5 catches them exactly. BM25 alone scores partial-R@5 = 93.3%, MRR = 0.883 — the headline numbers.

2. **Vector at 50.0% is surprisingly weak.** The mdbr-leaf-ir q8 256-dim MRL embedder appears under-tuned for chat-style conversation memory retrieval. Concrete causes to investigate: dimension may be too small for fine-grained chat semantics; the model may favor short documents over multi-turn text; the asymmetric retrieval prefix may not match what LongMemEval evaluates.

3. **Hybrid (RRF) does NOT outperform BM25.** It matches R@5 but loses 5.5pp on MRR (0.828 vs 0.883) and 6.6pp on partial-R@1 (76.7% vs 83.3%). RRF's averaging dilutes BM25's high-precision lexical hits with lower-ranked vector candidates.

4. **Plan's kill-switch (hybrid vs vector) did NOT trigger.** Hybrid beats Vector by 43.3pp on partial-R@5 — far above the 1pp threshold. The default stays at `SearchMode::Hybrid`.

5. **`memory_context` (Hybrid, flat path) matches `memory_query` (Hybrid) on R@5/R@10 and is within 0.2pp on MRR (0.826 vs 0.828).** The context path's recency/importance scoring (`memory_context` uses `compute_context_score` multiplicative form) does not degrade retrieval quality on this slice. Full-R@5 is slightly lower (66.7% vs 70.0%), a 3.3pp difference within sampling noise at n=30. The kill-switch conclusion for `memory_query` holds equally for `memory_context`: Hybrid does not outperform BM25 (MRR 0.826 vs 0.883), so the same recommendation applies — keep the default at `SearchMode::Hybrid` pending a full 500-question run.

## Default mode decision

**Keep `SearchMode::Hybrid` as the default.** Rationale:

- RRF is the published-best hybrid baseline; flipping to BM25 on a 30-question slice would over-fit to this sample.
- BM25's MRR lead (5.5pp) is meaningful but not categorical — the modes are close on R@5 and R@10.
- This result is a strong signal that **embedder tuning is the highest-leverage retrieval improvement**, not the fusion algorithm. Future work item.

The plan's mandated kill-switch (hybrid < vector by ≥1pp on partial-R@5) is honored: it did not fire, so no code change is required. `parse_search_mode` in `src/tools/handler.rs` continues to default to `SearchMode::Hybrid`.

## Caveats

- **Sample size**: 30 of 500 questions. Confidence intervals on R@5 percentages are wide (~±15pp at 30 samples). A full 500-question run is recommended before drawing strong conclusions or merging the BM25-tuning follow-up.
- **FTS5 OR-join**: `escape_fts_query` at `src/db/embeddings.rs:299` joins query tokens with `OR`, so multi-word queries match documents sharing any single token. This inflates BM25 recall and may be part of the lift over Vector.
- **Ingest path bypasses the MCP tool handler**: `Database::store_memory` + `Database::store_embedding` are called directly to avoid auto-dedup (≥0.90) and contradiction detection (≥0.85) firing on topically-similar turn-pairs. See `benchmarks/longmemeval/src/ingest.rs:1-17` and the regression test at `tests/longmemeval_ingest_bypass.rs`.
- **`memory_context` coverage**: Phase 6 extends hybrid retrieval into `memory_context`. The "Hybrid context" row in the table reflects the flat-path (no clusters) result.
- **Per-question project isolation**: each LongMemEval question is ingested into a fresh tempdir DB tagged `lme-{question_id}` to prevent cross-question candidate pollution.

## Reproducing

```bash
bash scripts/fetch-longmemeval.sh
cargo build --release --bin longmemeval-bench
./target/release/longmemeval-bench \
  --dataset benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  --questions 30 --session-limit 0 --seed 42 \
  --mode vector
# repeat for --mode bm25 and --mode hybrid
```

Artifacts land in `benchmarks/longmemeval/results/{mode}-{api}-{rfc3339}.{md,json}`.

## Future work

- Run the full 500-question benchmark to tighten confidence intervals.
- Investigate the Vector mode's weakness: try a larger embedder (768-dim or 1024-dim), reconsider the asymmetric retrieval prefix, evaluate alternate models.
- Per-field BM25 weighting (`bm25(memories_fts, w_content, w_summary, w_tags)`) — currently uniform.
- AND-joined FTS query option for high-precision queries.
- Run BM25 and Vector modes with `--api context` for a complete comparison.
- Investigate hierarchical path numbers (requires enough questions with clusterable memories).
