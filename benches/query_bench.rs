mod common;

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use serde_json::json;

fn bench_query(c: &mut Criterion) {
    // Corpus 100: real embeddings so query embedding is in the same vector space.
    let fixture_100 = common::build_fixture_real_embeddings(100);

    // Corpus 1k and 10k: synthetic embeddings — measures retrieval throughput /
    // SQL plan cost, not recall quality. See tests/retrieval_quality.rs for recall.
    let fixture_1k = common::build_fixture_with_dummy_embeddings(1_000);
    let fixture_10k = common::build_fixture_with_dummy_embeddings(10_000);

    let mut group = c.benchmark_group("memory_query");

    // Corpus 100 — current branch
    group.bench_with_input(
        BenchmarkId::new("corpus_100/branch_current", ""),
        &fixture_100,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_query",
                        black_box(json!({
                            "query": "authenticate user session",
                            "project_id": "bench",
                            "branch_mode": "current",
                            "limit": 10
                        })),
                    )
                    .expect("query")
            })
        },
    );

    // Corpus 100 — all branches
    group.bench_with_input(
        BenchmarkId::new("corpus_100/branch_all", ""),
        &fixture_100,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_query",
                        black_box(json!({
                            "query": "authenticate user session",
                            "project_id": "bench",
                            "branch_mode": "all",
                            "limit": 10
                        })),
                    )
                    .expect("query")
            })
        },
    );

    // Corpus 1k — current branch
    group.bench_with_input(
        BenchmarkId::new("corpus_1k/branch_current", ""),
        &fixture_1k,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_query",
                        black_box(json!({
                            "query": "database connection pool",
                            "project_id": "bench",
                            "branch_mode": "current",
                            "limit": 10
                        })),
                    )
                    .expect("query")
            })
        },
    );

    // Corpus 1k — all branches
    group.bench_with_input(
        BenchmarkId::new("corpus_1k/branch_all", ""),
        &fixture_1k,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_query",
                        black_box(json!({
                            "query": "database connection pool",
                            "project_id": "bench",
                            "branch_mode": "all",
                            "limit": 10
                        })),
                    )
                    .expect("query")
            })
        },
    );

    // Corpus 10k — current branch (reduce sample count to keep wall-time reasonable)
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    group.bench_with_input(
        BenchmarkId::new("corpus_10k/branch_current", ""),
        &fixture_10k,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_query",
                        black_box(json!({
                            "query": "cache eviction policy",
                            "project_id": "bench",
                            "branch_mode": "current",
                            "limit": 10
                        })),
                    )
                    .expect("query")
            })
        },
    );

    // Corpus 10k — all branches
    group.bench_with_input(
        BenchmarkId::new("corpus_10k/branch_all", ""),
        &fixture_10k,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_query",
                        black_box(json!({
                            "query": "cache eviction policy",
                            "project_id": "bench",
                            "branch_mode": "all",
                            "limit": 10
                        })),
                    )
                    .expect("query")
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_query);
criterion_main!(benches);
