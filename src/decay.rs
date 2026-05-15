use crate::memory::Memory;

/// Calculate the relevance score for a memory based on decay algorithm.
///
/// The relevance score is calculated as:
/// - Base decay: exponential decay based on time since last access
/// - Importance modifier: high importance memories decay slower
/// - Usage boost: frequently accessed memories get a boost
///
/// The final score is clamped between 0.1 and 1.0.
#[allow(dead_code)] // Reference implementation, tested
pub fn calculate_relevance(memory: &Memory, now_timestamp: i64, decay_rate: f64) -> f64 {
    let days_since_access = (now_timestamp - memory.last_accessed_at) as f64 / 86400.0;

    // Base decay: exponential decay based on time since last access
    let time_decay = (-decay_rate * days_since_access).exp();

    // Importance modifier: high importance = slower decay (0.5 to 1.0 range)
    let importance_factor = 0.5 + (memory.importance * 0.5);

    // Usage boost: logarithmic boost based on access count
    let usage_boost = (memory.access_count as f64).ln_1p() * 0.1;

    // Calculate final score
    let score = time_decay * importance_factor + usage_boost;

    // Clamp between 0.1 (floor) and 1.0 (ceiling)
    score.clamp(0.1, 1.0)
}

/// Reinforce a memory by updating its access count and relevance score.
/// Returns the updated memory.
#[allow(dead_code)] // Reference implementation, tested
pub fn reinforce_memory(mut memory: Memory, now_timestamp: i64) -> Memory {
    memory.access_count += 1;
    memory.last_accessed_at = now_timestamp;
    // Boost relevance score by 0.1, capped at 1.0
    memory.relevance_score = (memory.relevance_score + 0.1).min(1.0);
    memory
}

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
        assert!(relevance >= 0.1);
    }

    #[test]
    fn test_reinforce_memory() {
        let now = chrono::Utc::now().timestamp();
        let memory = create_test_memory(0.5, 5, 10);
        let reinforced = reinforce_memory(memory, now);

        assert_eq!(reinforced.access_count, 6);
        assert_eq!(reinforced.last_accessed_at, now);
        assert!(reinforced.relevance_score > 1.0 - 0.001); // Should be boosted to 1.0
    }

    /// Pinned Handoff memories must not have their relevance_score changed by decay.
    ///
    /// The DB-level decay query (`update_relevance_scores`) filters `WHERE pinned = 0`,
    /// so pinned memories are exempt.  This test calls the real production function to
    /// verify that invariant end-to-end.  rusqlite's bundled SQLite supports `EXP()` and
    /// `LN()`, so no stub is needed.
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
