mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use engram_mcp::tools::{compute_hybrid_score, compute_tag_boost};
use std::hint::black_box;

fn bench_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("scoring");

    let query_words: Vec<String> = vec![
        "auth".to_string(),
        "database".to_string(),
        "cache".to_string(),
    ];
    let tags_hit: Vec<String> = vec!["auth".to_string(), "memory".to_string()];
    let tags_miss: Vec<String> = vec!["network".to_string(), "parsing".to_string()];

    group.bench_function("compute_tag_boost/hit", |b| {
        b.iter(|| compute_tag_boost(black_box(&query_words), black_box(&tags_hit)))
    });

    group.bench_function("compute_tag_boost/miss", |b| {
        b.iter(|| compute_tag_boost(black_box(&query_words), black_box(&tags_miss)))
    });

    group.bench_function("compute_hybrid_score/recent", |b| {
        b.iter(|| {
            compute_hybrid_score(
                black_box(0.85_f32),
                black_box(0.0_f64), // undisplaced
                black_box(0.7_f64),
            )
        })
    });

    group.bench_function("compute_hybrid_score/stale", |b| {
        b.iter(|| {
            compute_hybrid_score(
                black_box(0.85_f32),
                black_box(30.0_f64), // displaced by 30 store-days
                black_box(0.7_f64),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_scoring);
criterion_main!(benches);
