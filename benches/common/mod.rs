//! Bench-only helpers; mirrors src/tools/test_utils.rs::dummy_vec but normalized for cosine.

#![allow(dead_code)]

use std::sync::OnceLock;

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::memory::{Memory, MemoryCluster, MemoryType};
use engram_mcp::tools::ToolHandler;
use tempfile::TempDir;

/// Build a dummy 256-element unit vector for use in benches that need pre-computed
/// vectors without a real `EmbeddingService`.
///
/// Uses a simple LCG-derived pattern so different seeds produce genuinely different
/// directions. The result is L2-normalized so cosine similarity is well-defined.
pub fn dummy_vec(seed: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..256)
        .map(|i| seed * (i as f32 + 1.0).cos() + (i as f32 * 0.1).sin())
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

pub struct BenchFixture {
    pub handler: ToolHandler,
    pub _tempdir: TempDir,
}

/// Open a fresh on-disk DB under `tempdir`, create project "bench", seed `corpus_size`
/// memories using synthetic embeddings (fast, no model load needed).
pub fn build_fixture_with_dummy_embeddings(corpus_size: usize) -> BenchFixture {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("bench.db");
    let db = Database::open(&db_path).expect("open db");
    db.get_or_create_project("bench", "bench").expect("project");

    let now = chrono::Utc::now().timestamp();
    let topics = [
        "auth", "database", "cache", "network", "parsing", "memory", "embed", "index",
    ];
    let types = [
        MemoryType::Fact,
        MemoryType::Decision,
        MemoryType::Pattern,
        MemoryType::Debug,
    ];

    let memories: Vec<Memory> = (0..corpus_size)
        .map(|i| {
            let topic = topics[i % topics.len()];
            let mtype = types[i % types.len()];
            Memory {
                id: format!("mem_{:08}", i),
                project_id: "bench".to_string(),
                memory_type: mtype,
                content: format!("memory {} about topic {} with index {}", i, topic, i % 37),
                summary: None,
                tags: vec![topic.to_string()],
                importance: 0.5 + (i % 5) as f64 * 0.1,
                relevance_score: 1.0,
                access_count: 0,
                created_at: now - i as i64,
                updated_at: now - i as i64,
                last_accessed_at: now - i as i64,
                branch: Some("main".to_string()),
                merged_from: None,
                pinned: false,
                global: false,
            }
        })
        .collect();

    db.store_memories_batch(&memories).expect("store memories");

    let embeddings: Vec<(String, Vec<f32>, String)> = (0..corpus_size)
        .map(|i| {
            (
                format!("mem_{:08}", i),
                dummy_vec(i as f32 * 0.01 + 0.1),
                "dummy-v0".to_string(),
            )
        })
        .collect();

    db.store_embeddings_batch(&embeddings)
        .expect("store embeddings");

    let svc = EmbeddingService::new().expect("embedding service");
    let handler = ToolHandler::new(db, svc, "bench".to_string(), Some("main".to_string()));
    BenchFixture {
        handler,
        _tempdir: tempdir,
    }
}

/// Same as `build_fixture_with_dummy_embeddings` but also seeds `num_clusters` clusters
/// and assigns corpus memories round-robin to them. Used by `context_bench` to exercise
/// the hierarchical retrieval path.
pub fn build_fixture_with_clusters(corpus_size: usize, num_clusters: usize) -> BenchFixture {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("bench.db");
    let db = Database::open(&db_path).expect("open db");
    db.get_or_create_project("bench", "bench").expect("project");

    let now = chrono::Utc::now().timestamp();
    let topics = [
        "auth", "database", "cache", "network", "parsing", "memory", "embed", "index",
    ];
    let types = [
        MemoryType::Fact,
        MemoryType::Decision,
        MemoryType::Pattern,
        MemoryType::Debug,
    ];

    let memories: Vec<Memory> = (0..corpus_size)
        .map(|i| {
            let topic = topics[i % topics.len()];
            let mtype = types[i % types.len()];
            Memory {
                id: format!("mem_{:08}", i),
                project_id: "bench".to_string(),
                memory_type: mtype,
                content: format!("memory {} about topic {} with index {}", i, topic, i % 37),
                summary: None,
                tags: vec![topic.to_string()],
                importance: 0.5 + (i % 5) as f64 * 0.1,
                relevance_score: 1.0,
                access_count: 0,
                created_at: now - i as i64,
                updated_at: now - i as i64,
                last_accessed_at: now - i as i64,
                branch: Some("main".to_string()),
                merged_from: None,
                pinned: false,
                global: false,
            }
        })
        .collect();

    db.store_memories_batch(&memories).expect("store memories");

    let embeddings: Vec<(String, Vec<f32>, String)> = (0..corpus_size)
        .map(|i| {
            (
                format!("mem_{:08}", i),
                dummy_vec(i as f32 * 0.01 + 0.1),
                "dummy-v0".to_string(),
            )
        })
        .collect();

    db.store_embeddings_batch(&embeddings)
        .expect("store embeddings");

    // Seed clusters with deterministic centroids.
    let nc = num_clusters.max(1);
    for c in 0..nc {
        let cluster = MemoryCluster {
            id: format!("cluster_{:04}", c),
            project_id: "bench".to_string(),
            summary: format!("cluster {} summary", c),
            member_count: 0,
            centroid: Some(dummy_vec(c as f32 * 0.3 + 0.05)),
            created_at: now,
            updated_at: now,
        };
        db.create_cluster(&cluster).expect("create cluster");
    }

    // Assign memories round-robin to clusters.
    for i in 0..corpus_size {
        let cluster_id = format!("cluster_{:04}", i % nc);
        let mem_id = format!("mem_{:08}", i);
        db.add_to_cluster(&cluster_id, &mem_id)
            .expect("add to cluster");
    }

    let svc = EmbeddingService::new().expect("embedding service");
    let handler = ToolHandler::new(db, svc, "bench".to_string(), Some("main".to_string()));
    BenchFixture {
        handler,
        _tempdir: tempdir,
    }
}

/// Like `build_fixture_with_dummy_embeddings` but uses real `EmbeddingService::embed_batch`.
/// Only suitable for small corpus sizes (≤ 100) due to model inference cost.
pub fn build_fixture_real_embeddings(corpus_size: usize) -> BenchFixture {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("bench.db");
    let db = Database::open(&db_path).expect("open db");
    db.get_or_create_project("bench", "bench").expect("project");

    let svc = EmbeddingService::new().expect("embedding service");

    let now = chrono::Utc::now().timestamp();
    let topics = [
        "auth", "database", "cache", "network", "parsing", "memory", "embed", "index",
    ];
    let types = [
        MemoryType::Fact,
        MemoryType::Decision,
        MemoryType::Pattern,
        MemoryType::Debug,
    ];

    let contents: Vec<String> = (0..corpus_size)
        .map(|i| {
            let topic = topics[i % topics.len()];
            format!("memory {} about topic {} with index {}", i, topic, i % 37)
        })
        .collect();

    let embeddings_raw = svc.embed_batch(contents.clone()).expect("embed batch");

    let memories: Vec<Memory> = (0..corpus_size)
        .map(|i| {
            let topic = topics[i % topics.len()];
            let mtype = types[i % types.len()];
            Memory {
                id: format!("mem_{:08}", i),
                project_id: "bench".to_string(),
                memory_type: mtype,
                content: contents[i].clone(),
                summary: None,
                tags: vec![topic.to_string()],
                importance: 0.5 + (i % 5) as f64 * 0.1,
                relevance_score: 1.0,
                access_count: 0,
                created_at: now - i as i64,
                updated_at: now - i as i64,
                last_accessed_at: now - i as i64,
                branch: Some("main".to_string()),
                merged_from: None,
                pinned: false,
                global: false,
            }
        })
        .collect();

    db.store_memories_batch(&memories).expect("store memories");

    let emb_triples: Vec<(String, Vec<f32>, String)> = (0..corpus_size)
        .map(|i| {
            (
                format!("mem_{:08}", i),
                embeddings_raw[i].clone(),
                svc.model_version().to_string(),
            )
        })
        .collect();

    db.store_embeddings_batch(&emb_triples)
        .expect("store embeddings");

    let handler = ToolHandler::new(db, svc, "bench".to_string(), Some("main".to_string()));
    BenchFixture {
        handler,
        _tempdir: tempdir,
    }
}

static EMBEDDING_SERVICE: OnceLock<EmbeddingService> = OnceLock::new();

/// Lazily-initialized singleton `EmbeddingService` so model load is excluded from
/// measured bench iterations. Uses `std::sync::OnceLock` (no `once_cell` dep needed).
pub fn embedding_service() -> &'static EmbeddingService {
    EMBEDDING_SERVICE.get_or_init(|| EmbeddingService::new().expect("EmbeddingService::new"))
}
