/// Compute a scoring boost based on query word overlap with memory tags.
///
/// For each query word that matches a tag (substring or exact), adds a boost.
/// The boost is capped to avoid overwhelming semantic/keyword scores.
/// Returns a value in [0.0, 0.15].
pub fn compute_tag_boost(query_words: &[String], tags: &[String]) -> f64 {
    if query_words.is_empty() || tags.is_empty() {
        return 0.0;
    }

    let tags_lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();

    let mut matches = 0usize;
    for qw in query_words {
        for tag in &tags_lower {
            // Match if query word contains the tag or tag contains the query word
            // e.g. "database" matches tag "database", "errors" matches tag "errors",
            // "auth" matches tag "auth" or "authentication"
            if qw == tag || tag.contains(qw.as_str()) || qw.contains(tag.as_str()) {
                matches += 1;
                break; // One match per query word
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Scale: 1 match = 0.05, 2 = 0.10, 3+ = 0.15 (capped)
    (matches as f64 * 0.05).min(0.15)
}

/// Compute a hybrid relevance score combining semantic similarity, recency, and importance.
///
/// `score = 0.6 * semantic + 0.2 * recency + 0.2 * importance`
///
/// where `recency = exp(-0.02 * days_since_access)`.
pub fn compute_hybrid_score(similarity: f32, last_accessed_at: i64, importance: f64) -> f32 {
    let now = chrono::Utc::now().timestamp();
    let days_since_access = ((now - last_accessed_at).max(0) as f64) / 86_400.0;
    let recency = (-0.02 * days_since_access).exp() as f32;
    0.6 * similarity + 0.2 * recency + 0.2 * importance as f32
}
