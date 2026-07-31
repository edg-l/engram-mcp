#[cfg(test)]
use crate::memory::Memory;

/// Lowest relevance a memory can decay to.
pub const RELEVANCE_FLOOR: f64 = 0.1;

/// Highest relevance a memory can reach. `memory_query` multiplies its ranking by
/// `relevance_score`, so an unbounded score is an unbounded ranking multiplier.
pub const RELEVANCE_CEILING: f64 = 1.0;

/// Most that retrieval frequency alone may add to a relevance score.
///
/// One step on the importance scale documented for `memory_store`
/// (0.3 minor / 0.5 normal / 0.7 important / 0.9 critical) is 0.2, which maps to 0.1
/// of `importance_factor` and therefore 0.1 of score. Usage is allowed to be worth
/// exactly one such step and no more, so no amount of retrieval can outrank a memory
/// that a human marked two importance levels higher.
const USAGE_BOOST_MAX: f64 = 0.1;

/// Access count at which the usage boost reaches `USAGE_BOOST_MAX`.
///
/// The boost grows logarithmically and is normalized so that this many accesses
/// saturates it; further retrieval adds nothing.
const USAGE_SATURATION_COUNT: f64 = 50.0;

/// Time constant for the usage boost's own fade, per store-day.
///
/// Matches the recency term used by `compute_hybrid_score` / `compute_context_score` so a
/// memory's ranking fades at the same rate whichever retrieval path scores it. Without
/// this the boost is permanent and "retrieved often, long ago" outranks "stored
/// recently" forever.
const USAGE_RECENCY_RATE: f64 = 0.02;

/// Relevance score from the raw column values, clamped to
/// `[RELEVANCE_FLOOR, RELEVANCE_CEILING]`.
///
/// ```text
/// time_decay       = exp(-decay_rate * elapsed)
/// importance_factor= 0.5 + importance * 0.5
/// usage_boost      = USAGE_BOOST_MAX * min(1, ln(1+n) / ln(1+SATURATION)) * exp(-0.02 * elapsed)
/// score            = clamp(time_decay * importance_factor + usage_boost)
/// ```
///
/// `elapsed` is **store-days, not calendar days**: the number of days on which the
/// memory's project received a store after this memory was last accessed (see
/// `db::activity`). A memory only goes stale when newer knowledge about the same project
/// arrives to displace it, so a dormant project's memories keep the relevance they had.
///
/// This is the single definition of the decay algorithm: the background job runs it by
/// calling the `RELEVANCE()` SQLite scalar function, which is registered against this
/// function on every connection (see `db::register_math_scalar_functions`).
pub fn relevance_from_parts(
    elapsed_store_days: f64,
    importance: f64,
    access_count: i64,
    decay_rate: f64,
) -> f64 {
    let days = elapsed_store_days.max(0.0);

    // Base decay: exponential in time since last access.
    let time_decay = (-decay_rate * days).exp();

    // Importance modifier: high importance = slower decay (0.5 to 1.0 range).
    let importance_factor = 0.5 + (importance.clamp(0.0, 1.0) * 0.5);

    // Usage boost: logarithmic in access count, saturating, and fading with time.
    let usage_fraction =
        ((access_count.max(0) as f64).ln_1p() / USAGE_SATURATION_COUNT.ln_1p()).min(1.0);
    let usage_boost = USAGE_BOOST_MAX * usage_fraction * (-USAGE_RECENCY_RATE * days).exp();

    (time_decay * importance_factor + usage_boost).clamp(RELEVANCE_FLOOR, RELEVANCE_CEILING)
}

/// Calculate the relevance score for a memory.
///
/// Thin wrapper over [`relevance_from_parts`] for callers that already hold a `Memory`.
#[cfg(test)]
pub fn calculate_relevance(memory: &Memory, now_timestamp: i64, decay_rate: f64) -> f64 {
    let days_since_access = (now_timestamp - memory.last_accessed_at) as f64 / 86400.0;
    relevance_from_parts(
        days_since_access,
        memory.importance,
        memory.access_count,
        decay_rate,
    )
}

/// Immediate relevance bump applied to a memory each time retrieval returns it.
///
/// Applied by `Database::record_access` / `record_access_batch` and capped at
/// [`RELEVANCE_CEILING`]. The next decay pass recomputes the score from scratch via
/// [`relevance_from_parts`], so this is a short-lived boost between passes, not a
/// permanent one.
pub const ACCESS_REINFORCEMENT: f64 = 0.1;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryType;

    fn create_test_memory(importance: f64, access_count: i64, days_ago: i64) -> Memory {
        let now = chrono::Utc::now().timestamp();
        Memory {
            id: "test".to_string(),
            project_id: "project".to_string(),
            memory_type: MemoryType::Fact,
            content: "test content".to_string(),
            summary: None,
            tags: vec![],
            importance,
            relevance_score: 1.0,
            access_count,
            created_at: now - (days_ago * 86400),
            updated_at: now - (days_ago * 86400),
            last_accessed_at: now - (days_ago * 86400),
            branch: None,
            merged_from: None,
            external_artifacts: None,
            pinned: false,
            global: false,
        }
    }

    #[test]
    fn test_fresh_memory_high_relevance() {
        let now = chrono::Utc::now().timestamp();
        let memory = create_test_memory(0.5, 0, 0);
        let relevance = calculate_relevance(&memory, now, 0.01);
        // Fresh memory should have high relevance
        assert!(relevance > 0.7);
    }

    #[test]
    fn test_old_memory_decays() {
        let now = chrono::Utc::now().timestamp();
        let memory = create_test_memory(0.5, 0, 100); // 100 days old
        let relevance = calculate_relevance(&memory, now, 0.01);
        // Old memory should have lower relevance
        assert!(relevance < 0.5);
    }

    #[test]
    fn test_important_memory_decays_slower() {
        let now = chrono::Utc::now().timestamp();
        let low_importance = create_test_memory(0.2, 0, 50);
        let high_importance = create_test_memory(0.9, 0, 50);

        let low_rel = calculate_relevance(&low_importance, now, 0.01);
        let high_rel = calculate_relevance(&high_importance, now, 0.01);

        // High importance memory should have higher relevance
        assert!(high_rel > low_rel);
    }

    #[test]
    fn test_frequently_accessed_memory_stays_relevant() {
        let now = chrono::Utc::now().timestamp();
        let rarely_accessed = create_test_memory(0.5, 1, 50);
        let frequently_accessed = create_test_memory(0.5, 100, 50);

        let rare_rel = calculate_relevance(&rarely_accessed, now, 0.01);
        let freq_rel = calculate_relevance(&frequently_accessed, now, 0.01);

        // Frequently accessed memory should have higher relevance
        assert!(freq_rel > rare_rel);
    }

    #[test]
    fn test_relevance_floor() {
        let now = chrono::Utc::now().timestamp();
        let ancient = create_test_memory(0.1, 0, 1000); // Very old, low importance
        let relevance = calculate_relevance(&ancient, now, 0.01);

        // Should never go below 0.1
        assert!(relevance >= RELEVANCE_FLOOR);
    }

    /// The usage boost is bounded: no access count can lift a memory past the ceiling.
    #[test]
    fn usage_boost_cannot_exceed_ceiling() {
        for access_count in [0, 1, 10, 100, 10_000, 1_000_000] {
            let score = relevance_from_parts(0.0, 1.0, access_count, 0.01);
            assert!(
                score <= RELEVANCE_CEILING,
                "access_count {access_count} produced {score}, above the ceiling"
            );
        }
    }

    /// The usage boost saturates: beyond the saturation count, more accesses add nothing.
    #[test]
    fn usage_boost_saturates() {
        // Use a low importance so the score sits below the ceiling and the boost is visible.
        let at_saturation = relevance_from_parts(10.0, 0.0, USAGE_SATURATION_COUNT as i64, 0.01);
        let far_beyond = relevance_from_parts(10.0, 0.0, 100_000, 0.01);
        assert!(
            (far_beyond - at_saturation).abs() < 1e-9,
            "boost still growing past saturation: {at_saturation} -> {far_beyond}"
        );
    }

    /// The usage boost fades with time, so heavy retrieval long ago does not outrank
    /// a comparable memory touched recently.
    #[test]
    fn usage_boost_decays_with_time() {
        let hot_recently = relevance_from_parts(1.0, 0.0, 200, 0.01);
        let hot_long_ago = relevance_from_parts(365.0, 0.0, 200, 0.01);
        assert!(
            hot_recently > hot_long_ago,
            "usage boost did not fade: {hot_recently} vs {hot_long_ago}"
        );
    }

    /// A pinned memory keeps a relevance of 1.0 while decay skips it. No amount of
    /// retrieval may push an unpinned memory above that, or pinning would lower a
    /// memory's ranking multiplier instead of protecting it.
    #[test]
    fn heavy_retrieval_never_outranks_a_pinned_memory() {
        let pinned_score = 1.0_f64;
        let hottest_possible = relevance_from_parts(0.0, 1.0, i64::MAX, 0.01);
        assert!(
            hottest_possible <= pinned_score,
            "unpinned memory reached {hottest_possible}, above a pinned memory's {pinned_score}"
        );
    }

    /// The decay job must apply the same ceiling as the formula. This runs the real SQL
    /// (`RELEVANCE()` is `relevance_from_parts` registered as a scalar function) rather
    /// than the Rust helper, because the two used to disagree: the SQL had a floor and
    /// no ceiling.
    #[test]
    fn decay_job_applies_the_ceiling() {
        let db = crate::db::Database::open_in_memory().unwrap();
        let project = crate::memory::Project {
            id: "test-project".to_string(),
            name: "Test Project".to_string(),
            root_path: None,
            decay_rate: 0.5,
            created_at: chrono::Utc::now().timestamp(),
        };
        db.create_project(&project).unwrap();

        let now = chrono::Utc::now().timestamp();
        let mut hot = create_test_memory(1.0, 5_000, 0);
        hot.id = "hot".to_string();
        hot.project_id = "test-project".to_string();
        hot.last_accessed_at = now;
        db.store_memory(&hot).unwrap();

        db.update_relevance_scores("test-project", 0.01).unwrap();

        let after = db.get_memory("hot").unwrap().unwrap();
        assert!(
            after.relevance_score <= RELEVANCE_CEILING,
            "decay job produced {}, above the ceiling",
            after.relevance_score
        );
        assert!(
            after.relevance_score >= RELEVANCE_FLOOR,
            "decay job produced {}, below the floor",
            after.relevance_score
        );
    }

    /// Pinned Handoff memories must not have their relevance_score changed by decay.
    ///
    /// The DB-level decay query (`update_relevance_scores`) filters `WHERE pinned = 0`,
    /// so pinned memories are exempt.  This test calls the real production function to
    /// verify that invariant end-to-end.  `Database::open` and `Database::open_in_memory`
    /// both register `EXP()` and `LN()` as custom scalar functions via
    /// `register_math_scalar_functions` (see `src/db/mod.rs`), so the SQL in
    /// `update_relevance_scores` works on both in-memory and on-disk databases.
    #[test]
    fn decay_skips_pinned_handoff() {
        let db = crate::db::Database::open_in_memory().unwrap();
        let project = crate::memory::Project {
            id: "test-project".to_string(),
            name: "Test Project".to_string(),
            root_path: None,
            decay_rate: 0.5,
            created_at: chrono::Utc::now().timestamp(),
        };
        db.create_project(&project).unwrap();

        // Place last_accessed_at 1 year in the past so decay meaningfully reduces the score.
        let far_past = chrono::Utc::now().timestamp() - 365 * 86400;

        // Pinned Handoff — must survive decay unchanged.
        let pinned = Memory {
            id: "handoff_pinned".to_string(),
            project_id: "test-project".to_string(),
            memory_type: MemoryType::Handoff,
            content: "## Summary\n\nSession ended here.".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.85,
            relevance_score: 1.0,
            access_count: 0,
            created_at: far_past,
            updated_at: far_past,
            last_accessed_at: far_past,
            branch: None,
            merged_from: None,
            external_artifacts: None,
            pinned: true,
            global: false,
        };

        // Non-pinned Handoff — decay must reduce its score.
        let unpinned = Memory {
            id: "handoff_unpinned".to_string(),
            project_id: "test-project".to_string(),
            memory_type: MemoryType::Handoff,
            content: "## Summary\n\nAnother session.".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.85,
            relevance_score: 1.0,
            access_count: 0,
            created_at: far_past,
            updated_at: far_past,
            last_accessed_at: far_past,
            branch: None,
            merged_from: None,
            external_artifacts: None,
            pinned: false,
            global: false,
        };

        db.store_memory(&pinned).unwrap();
        db.store_memory(&unpinned).unwrap();

        // Call the real production decay function.
        db.update_relevance_scores("test-project", 0.01).unwrap();

        // Pinned Handoff must be unchanged at its initial value.
        let after_pinned = db.get_memory("handoff_pinned").unwrap().unwrap();
        assert_eq!(
            after_pinned.relevance_score, 1.0,
            "pinned handoff relevance_score must not change when decay skips pinned = 1"
        );

        // Non-pinned Handoff must have had its score reduced by the decay formula.
        let after_unpinned = db.get_memory("handoff_unpinned").unwrap().unwrap();
        assert!(
            after_unpinned.relevance_score < 1.0,
            "unpinned handoff relevance_score ({}) should have been reduced by decay",
            after_unpinned.relevance_score
        );
    }
}
