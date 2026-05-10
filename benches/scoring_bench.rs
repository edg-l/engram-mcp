mod common;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use engram_mcp::tools::{compute_hybrid_score, compute_tag_boost};

fn bench_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("scoring");

    let query_words: Vec<String> = vec![
        "auth".to_string(),
        "database".to_string(),
        "cache".to_string(),
    ];
    let tags_hit: Vec<String> = vec!["auth".to_string(), "memory".to_string()];
    let tags_miss: Vec<String> = vec!["network".to_string(), "parsing".to_string()];
    let now = chrono::Utc::now().timestamp();

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
                black_box(now), // just accessed
                black_box(0.7_f64),
            )
        })
    });

    group.bench_function("compute_hybrid_score/stale", |b| {
        b.iter(|| {
            compute_hybrid_score(
                black_box(0.85_f32),
                black_box(now - 30 * 86_400), // 30 days ago
                black_box(0.7_f64),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_scoring);
criterion_main!(benches);
