mod common;

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use serde_json::json;

fn bench_context(c: &mut Criterion) {
    // Flat fallback fixtures (no clusters).
    let flat_1k = common::build_fixture_with_dummy_embeddings(1_000);
    let flat_10k = common::build_fixture_with_dummy_embeddings(10_000);

    // Hierarchical fixtures with clusters seeded.
    let hier_1k = common::build_fixture_with_clusters(1_000, 20);
    let hier_10k = common::build_fixture_with_clusters(10_000, 50);

    let mut group = c.benchmark_group("memory_context");

    // 1k flat
    group.bench_with_input(
        BenchmarkId::new("corpus_1k/hierarchical_false", ""),
        &flat_1k,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_context",
                        black_box(json!({
                            "query": "authentication refactor",
                            "project_id": "bench",
                            "hierarchical": false,
                            "limit": 10
                        })),
                    )
                    .expect("context")
            })
        },
    );

    // 1k hierarchical
    group.bench_with_input(
        BenchmarkId::new("corpus_1k/hierarchical_true", ""),
        &hier_1k,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_context",
                        black_box(json!({
                            "query": "authentication refactor",
                            "project_id": "bench",
                            "hierarchical": true,
                            "limit": 10
                        })),
                    )
                    .expect("context")
            })
        },
    );

    // 10k — reduce sample count for reasonable wall-time
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    // 10k flat
    group.bench_with_input(
        BenchmarkId::new("corpus_10k/hierarchical_false", ""),
        &flat_10k,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_context",
                        black_box(json!({
                            "query": "authentication refactor",
                            "project_id": "bench",
                            "hierarchical": false,
                            "limit": 10
                        })),
                    )
                    .expect("context")
            })
        },
    );

    // 10k hierarchical
    group.bench_with_input(
        BenchmarkId::new("corpus_10k/hierarchical_true", ""),
        &hier_10k,
        |b, f| {
            b.iter(|| {
                f.handler
                    .handle_tool(
                        "memory_context",
                        black_box(json!({
                            "query": "authentication refactor",
                            "project_id": "bench",
                            "hierarchical": true,
                            "limit": 10
                        })),
                    )
                    .expect("context")
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_context);
criterion_main!(benches);
