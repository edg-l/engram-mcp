#![cfg(test)]
#![allow(dead_code)]

use crate::db::{Database, encode_section_embeddings};
use crate::memory::{HandoffSections, Memory, MemoryType};

/// Build a dummy 256-element embedding vector for use in tests that need
/// pre-computed vectors without a real `EmbeddingService`.
///
/// Uses a simple pattern so different seeds produce genuinely different directions.
pub(super) fn dummy_vec(seed: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..256)
        .map(|i| seed * (i as f32 + 1.0).cos() + (i as f32 * 0.1).sin())
        .collect();
    // L2-normalize so cosine similarity is well-defined.
    let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Insert a handoff directly into the DB (no `EmbeddingService` required).
pub(super) fn insert_test_handoff(
    db: &Database,
    project_id: &str,
    id: &str,
    branch: &str,
    sections: &HandoffSections,
    section_vecs: &[(&str, Vec<f32>)],
) {
    let now = chrono::Utc::now().timestamp();
    let memory = Memory {
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
        external_artifacts: None,
        pinned: true,
        global: false,
    };
    db.store_memory(&memory).unwrap();

    // Store a dummy full-content embedding.
    let full_emb = dummy_vec(1.0);
    db.store_embedding(id, &full_emb, "test-model").unwrap();

    let keys: Vec<&str> = section_vecs.iter().map(|(k, _)| *k).collect();
    let vecs: Vec<Vec<f32>> = section_vecs.iter().map(|(_, v)| v.clone()).collect();
    let (keys_str, bytes) = encode_section_embeddings(&keys, &vecs);
    db.insert_handoff_sections(id, sections, &keys_str, &bytes)
        .unwrap();
}
