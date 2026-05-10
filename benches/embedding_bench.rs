mod common;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

fn bench_embedding(c: &mut Criterion) {
    // Model load is excluded — get_or_init runs once before any iteration.
    let svc = common::embedding_service();

    let mut group = c.benchmark_group("embedding");

    group.bench_function("embed_single", |b| {
        b.iter(|| {
            svc.embed(black_box("authenticate user session with JWT token"))
                .expect("embed")
        })
    });

    for &batch_size in &[10usize, 100usize] {
        let texts: Vec<String> = (0..batch_size)
            .map(|i| format!("sample document number {} about topic {}", i, i % 8))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("embed_batch", batch_size),
            &texts,
            |b, texts| {
                b.iter(|| {
                    svc.embed_batch(black_box(texts.clone()))
                        .expect("embed_batch")
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_embedding);
criterion_main!(benches);
