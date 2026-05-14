use std::collections::HashMap;
use std::str::FromStr;

/// Controls which retrieval strategy is active for `memory_query`.
///
/// Parsed from `ENGRAM_SEARCH_MODE` in Phase 2; here it is a pure value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    /// Pure vector (cosine) retrieval.
    Vector,
    /// Pure BM25 keyword retrieval.
    Bm25,
    /// Reciprocal Rank Fusion of vector and BM25 rankings (default).
    #[default]
    Hybrid,
}

impl FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "vector" => Ok(Self::Vector),
            "bm25" => Ok(Self::Bm25),
            "hybrid" => Ok(Self::Hybrid),
            other => Err(format!(
                "unknown search mode {:?}; expected one of: vector, bm25, hybrid",
                other
            )),
        }
    }
}

/// Fuse multiple ranked ID lists using Reciprocal Rank Fusion (RRF).
///
/// Each element of `rankings` is one ranker's ordered slice of memory IDs,
/// with rank 0 being the best result. The score contributed by a ranker is
/// `1.0 / (k + rank + 1.0)`, where `rank` is the 0-based position in the
/// slice. This makes rank 0 contribute `1/(k+1)` (1-indexed in the formula).
///
/// The published default for `k` is `60.0` (from the original RRF paper).
/// IDs absent from a ranker contribute 0 from that ranker.
///
/// Returns IDs sorted descending by fused score. **Ties are broken by
/// ascending lexicographic order of the ID** so results are deterministic.
pub fn rrf_fuse(rankings: &[&[String]], k: f64) -> Vec<(String, f64)> {
    let mut scores: HashMap<&str, f64> = HashMap::new();

    for ranker in rankings {
        for (rank, id) in ranker.iter().enumerate() {
            let contribution = 1.0 / (k + rank as f64 + 1.0);
            *scores.entry(id.as_str()).or_insert(0.0) += contribution;
        }
    }

    let mut result: Vec<(String, f64)> = scores
        .into_iter()
        .map(|(id, score)| (id.to_owned(), score))
        .collect();

    // Descending score; ties broken by ascending ID (lexicographic).
    result.sort_by(|(id_a, score_a), (id_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| id_a.cmp(id_b))
    });

    result
}

/// Apply tag boost and relevance decay to a base retrieval score.
///
/// Formula: `(base + tag_boost) * relevance`
///
/// The additive form matches the existing handler logic (handler.rs:578) and
/// keeps tag boost proportional to the base score after relevance scaling.
/// Uses `f64` to align with `compute_tag_boost` and `relevance_score`.
pub fn apply_tag_and_relevance(base: f64, tag_boost: f64, relevance: f64) -> f64 {
    (base + tag_boost) * relevance
}

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

/// Compute a multiplicative recency/importance boost for BM25 and Hybrid context scoring.
///
/// Formula: `base * (0.6 + 0.2 * recency + 0.2 * importance)`
///
/// Used in `memory_context` for Bm25 and Hybrid modes. The multiplicative form ensures that
/// memories absent from the active ranker (base = 0.0) score exactly 0.0, preventing
/// non-matching memories from surfacing due to recency/importance alone.
///
/// For Vector mode, `compute_hybrid_score` (additive form) is used instead, preserving
/// the existing behavior where recency and importance contribute even when cosine similarity
/// is low.
pub fn compute_context_score(base: f32, last_accessed_at: i64, importance: f64) -> f32 {
    let now = chrono::Utc::now().timestamp();
    let days_since_access = ((now - last_accessed_at).max(0) as f64) / 86_400.0;
    let recency = (-0.02 * days_since_access).exp() as f32;
    base * (0.6 + 0.2 * recency + 0.2 * importance as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- rrf_fuse tests ---

    #[test]
    fn rrf_fuse_two_rankers_overlap() {
        // ranker 0: ["a","b","c"]  ranker 1: ["b","c","d"]
        // 0-based ranks in each ranker:
        //   a: r0=0              -> 1/61
        //   b: r0=1, r1=0        -> 1/62 + 1/61
        //   c: r0=2, r1=1        -> 1/63 + 1/62
        //   d: r1=2              -> 1/63
        // Totals: b ≈ 0.03252, c ≈ 0.03200, a ≈ 0.01639, d ≈ 0.01587
        let r0 = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let r1 = vec!["b".to_owned(), "c".to_owned(), "d".to_owned()];
        let result = rrf_fuse(&[r0.as_slice(), r1.as_slice()], 60.0);
        let ids: Vec<&str> = result.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["b", "c", "a", "d"]);

        // Scores must be monotonically descending.
        let scores: Vec<f64> = result.iter().map(|(_, s)| *s).collect();
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "scores not descending: {} < {}", w[0], w[1]);
        }

        // Spot-check exact values.
        let b_score = result.iter().find(|(id, _)| id == "b").unwrap().1;
        let expected_b = 1.0 / 62.0 + 1.0 / 61.0;
        assert!(
            (b_score - expected_b).abs() < 1e-9,
            "b score {b_score} != {expected_b}"
        );
    }

    #[test]
    fn rrf_fuse_empty_ranker() {
        let r0: Vec<String> = vec![];
        let r1 = vec!["a".to_owned(), "b".to_owned()];
        let result = rrf_fuse(&[r0.as_slice(), r1.as_slice()], 60.0);
        assert_eq!(result.len(), 2);
        let ids: Vec<&str> = result.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);

        let a_score = result[0].1;
        let b_score = result[1].1;
        let expected_a = 1.0 / 61.0;
        let expected_b = 1.0 / 62.0;
        assert!((a_score - expected_a).abs() < 1e-9);
        assert!((b_score - expected_b).abs() < 1e-9);
    }

    #[test]
    fn rrf_fuse_single_ranker() {
        let r0 = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let result = rrf_fuse(&[r0.as_slice()], 60.0);
        let ids: Vec<&str> = result.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);

        // Scores must be strictly decreasing for a single ranker (all distinct ranks).
        let scores: Vec<f64> = result.iter().map(|(_, s)| *s).collect();
        for w in scores.windows(2) {
            assert!(
                w[0] > w[1],
                "scores not strictly decreasing: {} <= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn rrf_fuse_k_parameter() {
        // With small k the gap between rank-0 and rank-1 is larger.
        let ranker = vec!["x".to_owned(), "y".to_owned()];
        let result_k1 = rrf_fuse(&[ranker.as_slice()], 1.0);
        let result_k60 = rrf_fuse(&[ranker.as_slice()], 60.0);

        // k=1: scores are 1/2 and 1/3, gap = 1/6 ≈ 0.1667
        // k=60: scores are 1/61 and 1/62, gap ≈ 0.000265
        let gap_k1 = result_k1[0].1 - result_k1[1].1;
        let gap_k60 = result_k60[0].1 - result_k60[1].1;
        assert!(
            gap_k1 > gap_k60,
            "smaller k should produce larger rank gaps"
        );

        let expected_x_k1 = 1.0 / 2.0;
        let expected_y_k1 = 1.0 / 3.0;
        assert!((result_k1[0].1 - expected_x_k1).abs() < 1e-9);
        assert!((result_k1[1].1 - expected_y_k1).abs() < 1e-9);
    }

    #[test]
    fn rrf_fuse_tie_breaks_by_id() {
        // Both "a" and "b" appear at rank 0 in their respective single-element rankers.
        // Score for each: 1/(60+0+1) = 1/61. Tie broken lexicographically ascending.
        let r0 = vec!["b".to_owned()];
        let r1 = vec!["a".to_owned()];
        let result = rrf_fuse(&[r0.as_slice(), r1.as_slice()], 60.0);
        assert_eq!(result.len(), 2);
        let ids: Vec<&str> = result.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"], "tie should be broken by ascending ID");

        // Both scores equal 1/61.
        let expected = 1.0 / 61.0;
        assert!((result[0].1 - expected).abs() < 1e-9);
        assert!((result[1].1 - expected).abs() < 1e-9);
    }

    // --- apply_tag_and_relevance tests ---

    #[test]
    fn apply_tag_and_relevance_zero_tag() {
        let result = apply_tag_and_relevance(0.5, 0.0, 0.8);
        assert!((result - 0.4).abs() < 1e-9, "expected 0.4, got {result}");
    }

    #[test]
    fn apply_tag_and_relevance_zero_relevance() {
        let result = apply_tag_and_relevance(0.9, 0.15, 0.0);
        assert!((result - 0.0).abs() < 1e-9, "expected 0.0, got {result}");
    }

    #[test]
    fn apply_tag_and_relevance_additive_form() {
        // (0.3 + 0.1) * 1.0 = 0.4, not 0.3 * 0.1 * 1.0 = 0.03
        let result = apply_tag_and_relevance(0.3, 0.1, 1.0);
        assert!((result - 0.4).abs() < 1e-9, "expected 0.4, got {result}");
    }
}
