//! The per-project activity clock that relevance decay runs on.
//!
//! Decay measures how stale a memory is. Wall-clock time is the wrong axis for that: a
//! project untouched for two years is exactly as you left it, and its memories are no
//! less current than the day you stopped. What makes a memory stale is *new knowledge
//! arriving about the same project* that the memory does not account for.
//!
//! So the clock advances on stores, not on seconds. A project with no new memories has a
//! frozen clock and nothing decays; a project under active development advances at its
//! own pace regardless of the calendar.
//!
//! The unit is the **store-day**: a day on which the project received at least one store.
//! Counting raw stores instead would let one heavy session age everything else at once —
//! real projects here store 100 memories across 8 days — and would put `decay_rate` on an
//! uninterpretable scale. Per store-day keeps it readable and keeps the existing
//! constants meaningful.
//!
//! Hook-captured memories do not advance the clock. They are stored without anyone
//! deciding to, and a session where automatic capture was the only thing that happened is
//! not a session that made deliberately curated knowledge any staler.

use rusqlite::params;

use crate::error::MemoryError;

use super::Database;

/// Seconds in a day, the bucket width for a store-day.
pub const SECONDS_PER_DAY: i64 = 86_400;

/// SQL predicate selecting stores that advance the clock.
///
/// Tags are a JSON array, so `["hook","session_summary"]` matches `%"hook"%` exactly.
pub const CLOCK_ADVANCING_STORE: &str = "(tags IS NULL OR tags NOT LIKE '%\"hook\"%')";

/// The distinct store-days of one project, ascending.
///
/// Loaded once per retrieval call, in the same shape as the existing per-project
/// embedding and status loads. Projects here top out at 39 store-days, so this is a few
/// hundred bytes and a binary search per memory.
#[derive(Debug, Clone, Default)]
pub struct StoreDayIndex {
    days: Vec<i64>,
}

impl StoreDayIndex {
    /// Build from raw day buckets. Sorts and de-duplicates.
    pub fn from_days(mut days: Vec<i64>) -> Self {
        days.sort_unstable();
        days.dedup();
        Self { days }
    }

    /// Store-days recorded after the day containing `timestamp`.
    ///
    /// A store on the same day as the access does not count: it is concurrent with the
    /// access, not subsequent to it. Returned as `f64` because it feeds the same decay
    /// formula that used to take fractional days.
    pub fn active_days_since(&self, timestamp: i64) -> f64 {
        let bucket = timestamp.div_euclid(SECONDS_PER_DAY);
        // First index strictly greater than `bucket`; everything from there on counts.
        let idx = self.days.partition_point(|d| *d <= bucket);
        (self.days.len() - idx) as f64
    }

    /// Total store-days on record for the project.
    #[allow(dead_code)] // Used by tests and available to callers reporting clock state
    pub fn len(&self) -> usize {
        self.days.len()
    }

    #[allow(dead_code)] // Used by tests and available to callers reporting clock state
    pub fn is_empty(&self) -> bool {
        self.days.is_empty()
    }
}

impl Database {
    /// Load the activity clock for a project.
    pub fn get_store_day_index(&self, project_id: &str) -> Result<StoreDayIndex, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let sql = format!(
            "SELECT DISTINCT created_at / {SECONDS_PER_DAY}
             FROM memories
             WHERE project_id = ?1 AND {CLOCK_ADVANCING_STORE}
             ORDER BY 1"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![project_id], |row| row.get::<_, i64>(0))?;
        Ok(StoreDayIndex::from_days(rows.flatten().collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: i64 = SECONDS_PER_DAY;

    #[test]
    fn counts_only_store_days_after_the_access_day() {
        let index = StoreDayIndex::from_days(vec![10, 11, 15, 20]);
        // Accessed on day 11: days 15 and 20 came later.
        assert_eq!(index.active_days_since(11 * D), 2.0);
        // Accessed on day 9: everything came later.
        assert_eq!(index.active_days_since(9 * D), 4.0);
        // Accessed after the last store: nothing has displaced it.
        assert_eq!(index.active_days_since(25 * D), 0.0);
    }

    /// A store later on the same day as the access is concurrent with it, not subsequent.
    #[test]
    fn same_day_stores_do_not_count() {
        let index = StoreDayIndex::from_days(vec![10]);
        assert_eq!(index.active_days_since(10 * D), 0.0);
        assert_eq!(index.active_days_since(10 * D + 3600), 0.0);
    }

    /// The property the whole change exists for: a dormant project does not age.
    #[test]
    fn a_project_with_no_new_stores_never_ages() {
        let index = StoreDayIndex::from_days(vec![100]);
        let stored_at = 100 * D;
        // Two years of wall-clock time, no new stores.
        let two_years_later = stored_at + 730 * D;
        assert_eq!(index.active_days_since(stored_at), 0.0);
        // The clock is a property of the project, not of when we ask.
        assert_eq!(index.active_days_since(two_years_later), 0.0);
    }

    /// A burst of stores in one session is one tick, not one per memory.
    #[test]
    fn a_heavy_session_advances_the_clock_once() {
        // Sixty stores, all on day 50.
        let index = StoreDayIndex::from_days(vec![50; 60]);
        assert_eq!(index.len(), 1);
        assert_eq!(index.active_days_since(49 * D), 1.0);
    }

    #[test]
    fn empty_project_has_a_frozen_clock() {
        let index = StoreDayIndex::default();
        assert!(index.is_empty());
        assert_eq!(index.active_days_since(0), 0.0);
    }
}
