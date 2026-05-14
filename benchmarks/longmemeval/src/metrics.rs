//! Per-question and aggregate retrieval metrics for the LongMemEval-S harness.

/// Retrieval result for a single question at a given cut-off k.
///
/// # Deduplication note
/// If the same session id appears multiple times in the retrieved list (e.g. multiple
/// memories from one session all ranked in the top-k), the **first occurrence** wins for
/// `first_match_rank`. `full_hit` still requires every gold session to appear at least
/// once anywhere in `retrieved[0..k]`.
#[derive(Debug, Clone)]
pub struct HitResult {
    /// At least one gold session id is present in the top-k retrieved ids.
    pub partial_hit: bool,
    /// Every gold session id appears at least once in the top-k retrieved ids.
    pub full_hit: bool,
    /// 1-indexed rank of the first retrieved id (within the top-k) that is a gold session.
    /// `None` when no gold session appears in the top-k.
    pub first_match_rank: Option<usize>,
}

/// Evaluate retrieval quality for one question.
///
/// # Arguments
/// * `retrieved_session_ids` - Ordered list of session ids returned by the retriever
///   (index 0 is rank 1). May contain duplicates if multiple memories share a session.
/// * `answer_session_ids` - Gold set of session ids that contain the answer.
/// * `k` - Cut-off: only the first `k` elements of `retrieved_session_ids` are examined.
pub fn evaluate_topk(
    retrieved_session_ids: &[String],
    answer_session_ids: &[String],
    k: usize,
) -> HitResult {
    let gold: std::collections::HashSet<&str> =
        answer_session_ids.iter().map(String::as_str).collect();

    let mut matched: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut first_match_rank: Option<usize> = None;

    for (rank0, id) in retrieved_session_ids.iter().take(k).enumerate() {
        if gold.contains(id.as_str()) {
            if first_match_rank.is_none() {
                first_match_rank = Some(rank0 + 1); // convert to 1-indexed
            }
            matched.insert(id.as_str());
        }
    }

    let partial_hit = !matched.is_empty();
    let full_hit = !gold.is_empty() && gold.iter().all(|g| matched.contains(*g));

    HitResult {
        partial_hit,
        full_hit,
        first_match_rank,
    }
}

/// Aggregate retrieval metrics over all sampled questions.
///
/// # Arguments
/// * `per_question_top10` - One `HitResult` per question, evaluated at k=10.
/// * `per_question_top5`  - One `HitResult` per question, evaluated at k=5.
/// * `per_question_top1`  - One `HitResult` per question, evaluated at k=1.
///
/// All three slices must have the same length `n`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AggregateMetrics {
    /// Number of questions evaluated.
    pub n: usize,
    /// Partial recall at k=1: fraction of questions where at least one gold session was in top-1.
    pub partial_r_at_1: f64,
    /// Partial recall at k=5.
    pub partial_r_at_5: f64,
    /// Partial recall at k=10.
    pub partial_r_at_10: f64,
    /// Full recall at k=1: fraction of questions where all gold sessions were in top-1.
    pub full_r_at_1: f64,
    /// Full recall at k=5.
    pub full_r_at_5: f64,
    /// Full recall at k=10.
    pub full_r_at_10: f64,
    /// Mean Reciprocal Rank computed from the top-10 results.
    /// For each question: 1/rank if a gold session is in top-10, else 0.
    pub mrr: f64,
}

/// Compute aggregate metrics from per-question hit results.
///
/// All three slices must have the same length.
pub fn aggregate(
    per_question_top10: &[HitResult],
    per_question_top5: &[HitResult],
    per_question_top1: &[HitResult],
) -> AggregateMetrics {
    let n = per_question_top10.len();
    assert_eq!(n, per_question_top5.len(), "slice lengths must match");
    assert_eq!(n, per_question_top1.len(), "slice lengths must match");

    if n == 0 {
        return AggregateMetrics {
            n: 0,
            partial_r_at_1: 0.0,
            partial_r_at_5: 0.0,
            partial_r_at_10: 0.0,
            full_r_at_1: 0.0,
            full_r_at_5: 0.0,
            full_r_at_10: 0.0,
            mrr: 0.0,
        };
    }

    let mean_bool = |hits: &[HitResult], f: fn(&HitResult) -> bool| -> f64 {
        hits.iter().filter(|h| f(h)).count() as f64 / n as f64
    };

    let partial_r_at_1 = mean_bool(per_question_top1, |h| h.partial_hit);
    let partial_r_at_5 = mean_bool(per_question_top5, |h| h.partial_hit);
    let partial_r_at_10 = mean_bool(per_question_top10, |h| h.partial_hit);
    let full_r_at_1 = mean_bool(per_question_top1, |h| h.full_hit);
    let full_r_at_5 = mean_bool(per_question_top5, |h| h.full_hit);
    let full_r_at_10 = mean_bool(per_question_top10, |h| h.full_hit);

    let mrr = per_question_top10
        .iter()
        .map(|h| match h.first_match_rank {
            Some(rank) => 1.0 / rank as f64,
            None => 0.0,
        })
        .sum::<f64>()
        / n as f64;

    AggregateMetrics {
        n,
        partial_r_at_1,
        partial_r_at_5,
        partial_r_at_10,
        full_r_at_1,
        full_r_at_5,
        full_r_at_10,
        mrr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_full_hit_all_in_top_3() {
        let retrieved: Vec<String> = ["s1", "s2", "s3", "s4"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let gold: Vec<String> = ["s2", "s3"].iter().map(|s| s.to_string()).collect();
        let result = evaluate_topk(&retrieved, &gold, 3);
        assert!(result.full_hit, "all gold sessions are in top-3");
        assert!(result.partial_hit);
        assert_eq!(result.first_match_rank, Some(2), "s2 is at rank 2");
    }

    #[test]
    fn evaluate_partial_one_of_two_in_top_5() {
        let retrieved: Vec<String> = ["s1", "s2", "s3", "s4", "s5"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let gold: Vec<String> = ["s2", "s99"].iter().map(|s| s.to_string()).collect();
        let result = evaluate_topk(&retrieved, &gold, 5);
        assert!(result.partial_hit, "s2 is in top-5");
        assert!(!result.full_hit, "s99 is not in top-5");
        assert_eq!(result.first_match_rank, Some(2));
    }

    #[test]
    fn evaluate_no_match() {
        let retrieved: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let gold: Vec<String> = ["x"].iter().map(|s| s.to_string()).collect();
        let result = evaluate_topk(&retrieved, &gold, 2);
        assert!(!result.partial_hit);
        assert!(!result.full_hit);
        assert_eq!(result.first_match_rank, None);
    }

    #[test]
    fn mrr_math() {
        let top10 = vec![
            HitResult {
                partial_hit: true,
                full_hit: true,
                first_match_rank: Some(1),
            },
            HitResult {
                partial_hit: true,
                full_hit: true,
                first_match_rank: Some(2),
            },
            HitResult {
                partial_hit: false,
                full_hit: false,
                first_match_rank: None,
            },
        ];
        let top5 = top10.clone();
        let top1 = top10.clone();
        let agg = aggregate(&top10, &top5, &top1);
        let expected_mrr = (1.0_f64 + 0.5 + 0.0) / 3.0;
        assert!(
            (agg.mrr - expected_mrr).abs() < 1e-10,
            "mrr={} expected={}",
            agg.mrr,
            expected_mrr
        );
    }
}
