mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_mcp::db::{Database, encode_section_embeddings};
use engram_mcp::memory::{HandoffSections, Memory, MemoryType};
use engram_mcp::tools::{SearchMode, score_handoff_sections};
use serde_json::json;
use tempfile::TempDir;

// ============================================================
// Handoff seeding helpers (no real EmbeddingService needed)
// ============================================================

fn make_sections(i: usize, continues_from: Option<String>) -> HandoffSections {
    HandoffSections {
        summary: format!("Session {} summary: worked on auth module", i),
        decisions: vec![format!("decision {} use SQLite", i)],
        todos: vec![format!("todo {} write tests", i)],
        blockers: vec![format!("blocker {} migration pending", i)],
        mental_model: format!("mental model {}: layered service architecture", i),
        next_steps: vec![format!("next {} deploy to staging", i)],
        notes: Some(format!("notes for session {}", i)),
        continues_from,
    }
}

fn make_handoff_memory(
    id: &str,
    project_id: &str,
    branch: &str,
    sections: &HandoffSections,
) -> Memory {
    let now = chrono::Utc::now().timestamp();
    Memory {
        id: id.to_string(),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Handoff,
        content: sections.render_markdown(),
        summary: None,
        tags: vec![],
        importance: 0.85,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: Some(branch.to_string()),
        merged_from: None,
        pinned: true,
        global: false,
    }
}

/// Section keys and byte-blob for a given handoff index (all dummy embeddings).
fn encode_sections(i: usize) -> (String, Vec<u8>) {
    let keys = [
        "summary",
        "decisions",
        "todos",
        "blockers",
        "mental_model",
        "next_steps",
        "notes",
    ];
    let key_strs: Vec<&str> = keys.into_iter().collect();
    let vecs: Vec<Vec<f32>> = (0..keys.len())
        .map(|j| common::dummy_vec((i * 7 + j) as f32 * 0.05 + 0.1))
        .collect();
    encode_section_embeddings(&key_strs, &vecs)
}

/// Seed `n` handoffs into `db`, chained via `continues_from` so each handoff
/// continues from the previous one (oldest first). Returns the seeded Memory list.
fn seed_handoffs(db: &Database, project_id: &str, n: usize) -> Vec<Memory> {
    let mut prev_id: Option<String> = None;
    let mut memories = Vec::with_capacity(n);

    for i in 0..n {
        let id = format!("hoff_{:04}", i);
        let sections = make_sections(i, prev_id.clone());
        let memory = make_handoff_memory(&id, project_id, "main", &sections);
        let (keys_str, key_bytes) = encode_sections(i);
        let full_emb = common::dummy_vec(i as f32 * 0.03 + 0.5);

        db.store_handoff_atomic(
            &memory, &full_emb, "dummy-v0", &sections, &keys_str, &key_bytes,
        )
        .expect("store_handoff_atomic");

        prev_id = Some(id);
        memories.push(memory);
    }
    memories
}

struct HandoffFixture {
    db: Database,
    memories: Vec<Memory>,
    _tempdir: TempDir,
}

fn build_handoff_fixture(n: usize) -> HandoffFixture {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("bench.db");
    let db = Database::open(&db_path).expect("open db");
    db.get_or_create_project("bench", "bench").expect("project");

    let memories = seed_handoffs(&db, "bench", n);
    HandoffFixture {
        db,
        memories,
        _tempdir: tempdir,
    }
}

// ============================================================
// Bench functions
// ============================================================

fn bench_score_sections(c: &mut Criterion) {
    // Pre-build fixture with 10 handoffs; scoring is per-call, fast microbench.
    let fixture = build_handoff_fixture(10);
    let query_vec = common::dummy_vec(0.42);

    c.bench_function("score_handoff_sections/10_handoffs", |b| {
        b.iter(|| {
            score_handoff_sections(
                black_box(&query_vec),
                black_box(&fixture.memories),
                &fixture.db,
            )
            .expect("score")
        })
    });
}

fn bench_handoff_resume(c: &mut Criterion) {
    // Fixtures with different chain depths (1, 3, 5 handoffs).
    let fix1 = {
        let f = build_handoff_fixture(1);
        let svc = engram_mcp::embedding::EmbeddingService::new().expect("svc");
        let handler = engram_mcp::tools::ToolHandler::new(
            f.db,
            svc,
            "bench".to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );
        (handler, f._tempdir)
    };
    let fix3 = {
        let f = build_handoff_fixture(3);
        let svc = engram_mcp::embedding::EmbeddingService::new().expect("svc");
        let handler = engram_mcp::tools::ToolHandler::new(
            f.db,
            svc,
            "bench".to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );
        (handler, f._tempdir)
    };
    let fix5 = {
        let f = build_handoff_fixture(5);
        let svc = engram_mcp::embedding::EmbeddingService::new().expect("svc");
        let handler = engram_mcp::tools::ToolHandler::new(
            f.db,
            svc,
            "bench".to_string(),
            Some("main".to_string()),
            SearchMode::default(),
        );
        (handler, f._tempdir)
    };

    let mut group = c.benchmark_group("handoff_resume");

    for (depth, (handler, _dir)) in [("depth_1", &fix1), ("depth_3", &fix3), ("depth_5", &fix5)] {
        group.bench_with_input(BenchmarkId::new("chain_depth", depth), handler, |b, h| {
            b.iter(|| {
                h.handle_tool(
                    "handoff_resume",
                    black_box(json!({
                        "project_id": "bench",
                        "query": "authentication refactor",
                        "branch": "main",
                        "max_sections": 5
                    })),
                )
                .expect("handoff_resume")
            })
        });
    }

    group.finish();
}

fn bench_handoff_search(c: &mut Criterion) {
    // 50 handoffs, search by content.
    let fixture = build_handoff_fixture(50);
    let svc = engram_mcp::embedding::EmbeddingService::new().expect("svc");
    let handler = engram_mcp::tools::ToolHandler::new(
        fixture.db,
        svc,
        "bench".to_string(),
        Some("main".to_string()),
        SearchMode::default(),
    );
    let _dir = fixture._tempdir;

    c.bench_function("handoff_search/50_handoffs", |b| {
        b.iter(|| {
            handler
                .handle_tool(
                    "handoff_search",
                    black_box(json!({
                        "query": "migrate database schema",
                        "project_id": "bench",
                        "limit": 10
                    })),
                )
                .expect("handoff_search")
        })
    });
}

criterion_group!(
    benches,
    bench_score_sections,
    bench_handoff_resume,
    bench_handoff_search
);
criterion_main!(benches);
