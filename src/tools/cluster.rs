//! Hierarchical memory clustering: centroid-based assignment and summaries.
//!
//! Free functions over `&Database` so both the MCP `memory_store` path (via
//! `ToolHandler`'s thin delegating methods) and `engram-cli import` can assign a memory
//! to a cluster without an importer having to construct a `ToolHandler`.

use crate::db::Database;
use crate::embedding::cosine_similarity;
use crate::error::MemoryError;
use crate::memory::MemoryCluster;

/// Similarity to an existing cluster centroid above which a memory joins that cluster
/// instead of starting a new one.
const CLUSTER_THRESHOLD: f32 = 0.75;

/// Generate a cluster summary from member memories.
/// Uses the first sentence of the highest-importance member + top keywords across all members.
pub fn generate_cluster_summary(
    db: &Database,
    member_ids: &[String],
) -> Result<String, MemoryError> {
    if member_ids.is_empty() {
        return Ok("Empty cluster".to_string());
    }

    let members = db.get_memories_batch(member_ids)?;
    if members.is_empty() {
        return Ok("Empty cluster".to_string());
    }

    // Find highest-importance member
    let best_member = members
        .values()
        .max_by(|a, b| {
            a.importance
                .partial_cmp(&b.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    // Get first sentence from best member
    let first_sentence = crate::summarize::extract_first_sentence(&best_member.content);

    // Collect keywords from all members
    let all_content: String = members
        .values()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let keywords = crate::summarize::extract_keywords(&all_content, 3);

    if keywords.is_empty() {
        Ok(first_sentence)
    } else {
        Ok(format!("{} [{}]", first_sentence, keywords.join(", ")))
    }
}

/// Assign a memory to the best matching cluster, or create a new one.
pub fn assign_to_cluster(
    db: &Database,
    project: &str,
    memory_id: &str,
    embedding: &[f32],
    content: &str,
    _importance: f64,
) -> Result<Option<String>, MemoryError> {
    let clusters = db.get_clusters_for_project(project)?;

    // Find best matching cluster by centroid similarity
    let mut best_match: Option<(String, f32)> = None;

    for cluster in &clusters {
        if let Some(ref centroid) = cluster.centroid {
            let similarity = cosine_similarity(embedding, centroid);
            if similarity >= CLUSTER_THRESHOLD
                && (best_match.is_none() || similarity > best_match.as_ref().unwrap().1)
            {
                best_match = Some((cluster.id.clone(), similarity));
            }
        }
    }

    let now = chrono::Utc::now().timestamp();

    if let Some((cluster_id, _)) = best_match {
        // Add to existing cluster
        db.add_to_cluster(&cluster_id, memory_id)?;

        // Update centroid (running average)
        let member_ids = db.get_cluster_member_ids(&cluster_id)?;
        let new_centroid = compute_cluster_centroid(db, &member_ids)?;
        let summary = generate_cluster_summary(db, &member_ids)?;

        if let Some(centroid) = new_centroid {
            db.update_cluster_centroid(&cluster_id, &centroid, &summary)?;
        }

        Ok(Some(cluster_id))
    } else {
        // Create new cluster
        let cluster_id = format!("clust_{}", uuid::Uuid::new_v4().simple());
        let summary = crate::summarize::extract_first_sentence(content);

        let cluster = MemoryCluster {
            id: cluster_id.clone(),
            project_id: project.to_string(),
            summary,
            member_count: 1,
            centroid: Some(embedding.to_vec()),
            created_at: now,
            updated_at: now,
        };

        db.create_cluster(&cluster)?;
        db.add_to_cluster(&cluster_id, memory_id)?;

        Ok(Some(cluster_id))
    }
}

/// Compute the centroid (average embedding) for a set of memory IDs.
pub fn compute_cluster_centroid(
    db: &Database,
    member_ids: &[String],
) -> Result<Option<Vec<f32>>, MemoryError> {
    if member_ids.is_empty() {
        return Ok(None);
    }

    let member_embeddings = db.get_embeddings_batch(member_ids)?;

    let mut sum: Option<Vec<f32>> = None;
    let mut count = 0usize;

    for (_id, vec) in &member_embeddings {
        count += 1;
        match &mut sum {
            None => sum = Some(vec.clone()),
            Some(s) => {
                for (i, v) in vec.iter().enumerate() {
                    if i < s.len() {
                        s[i] += v;
                    }
                }
            }
        }
    }

    Ok(sum.map(|mut s| {
        let c = count as f32;
        for v in &mut s {
            *v /= c;
        }
        s
    }))
}
