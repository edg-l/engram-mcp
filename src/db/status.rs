//! Retrieval status: which memories are no longer the answer, and what replaced them.
//!
//! Two distinct conditions, deliberately not one flag:
//!
//! - **Superseded** — a newer memory replaced this one. Read from the `supersedes`
//!   relationship edge, which is the source of truth; nothing is duplicated onto the
//!   memory row. Retrieval redirects to the successor rather than dropping the match,
//!   because returning nothing reads as "nobody looked into this" and invites the stale
//!   conclusion to be derived again from scratch.
//! - **Dead** — the subject is gone (the service was retired, the file deleted) and
//!   there is no successor to point at. Stored in `memory_status` and excluded outright.

use std::collections::{HashMap, HashSet};

use rusqlite::params;

use crate::error::MemoryError;
use crate::memory::RelationType;

use super::Database;

/// A `memory_status` row's payload, keyed by memory id in [`StatusRowMap`]: `(dead, reason,
/// marked_at)`. Distinct from `export::StatusMap`, which is keyed the same way but holds the
/// wire-format `ExportedStatus` struct instead of this raw tuple.
pub type StatusRow = (bool, Option<String>, i64);

/// `memory_id -> (dead, reason, marked_at)`, the shape returned by
/// [`Database::export_status_rows`](super::sync).
pub type StatusRowMap = HashMap<String, StatusRow>;

/// Longest supersession chain followed when resolving a successor.
///
/// Matches the handoff chain depth. A chain longer than this is almost certainly a
/// mistake; resolution stops and returns the deepest memory reached.
const MAX_SUPERSESSION_DEPTH: usize = 5;

/// Which memories in a project were superseded, and by what.
#[derive(Debug, Clone, Default)]
pub struct SupersessionMap {
    /// Superseded memory id -> the memory that directly superseded it.
    direct: HashMap<String, String>,
}

impl SupersessionMap {
    pub fn is_superseded(&self, memory_id: &str) -> bool {
        self.direct.contains_key(memory_id)
    }

    /// Follow the chain to the memory that is current.
    ///
    /// Returns `None` if `memory_id` was never superseded. Stops at
    /// [`MAX_SUPERSESSION_DEPTH`] hops, and stops on a cycle (A supersedes B, B
    /// supersedes A — reachable in practice by reverting a decision), returning the last
    /// distinct memory reached rather than looping.
    pub fn terminal_successor(&self, memory_id: &str) -> Option<&str> {
        let mut current = self.direct.get(memory_id)?.as_str();
        let mut seen: HashSet<&str> = HashSet::new();
        seen.insert(memory_id);

        for _ in 0..MAX_SUPERSESSION_DEPTH {
            if !seen.insert(current) {
                break;
            }
            match self.direct.get(current) {
                Some(next) => current = next.as_str(),
                None => break,
            }
        }
        Some(current)
    }

    #[cfg(test)]
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        Self {
            direct: pairs
                .iter()
                .map(|(old, new)| ((*old).to_string(), (*new).to_string()))
                .collect(),
        }
    }
}

impl Database {
    /// Build the supersession map for a project from its `supersedes` edges.
    ///
    /// One query per retrieval, in the same shape as the existing per-project embedding
    /// load. When two memories both supersede the same predecessor, the most recent edge
    /// wins.
    pub fn get_supersession_map(&self, project_id: &str) -> Result<SupersessionMap, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT r.source_id, r.target_id
             FROM relationships r
             JOIN memories m ON r.source_id = m.id
             WHERE m.project_id = ?1 AND r.relation_type = ?2
             ORDER BY r.created_at ASC",
        )?;
        let rows = stmt.query_map(
            params![project_id, RelationType::Supersedes.as_str()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )?;

        let mut direct = HashMap::new();
        for (source_id, target_id) in rows.flatten() {
            // The edge points new -> old, so the target is the one that was superseded.
            direct.insert(target_id, source_id);
        }
        Ok(SupersessionMap { direct })
    }

    /// Mark a memory dead (excluded from retrieval) or bring it back, at the current time.
    pub fn set_dead(
        &self,
        memory_id: &str,
        dead: bool,
        reason: Option<&str>,
    ) -> Result<(), MemoryError> {
        self.set_dead_at(memory_id, dead, reason, chrono::Utc::now().timestamp())
    }

    /// Mark a memory dead or bring it back, with an explicit `marked_at`.
    ///
    /// Used by import to preserve the source machine's timestamp instead of stamping the
    /// importer's clock, which would make every dead-only toggle look newer than it is and
    /// defeat last-write-wins convergence. Also bumps `memories.updated_at` so a dead-only
    /// toggle (no content change) is still visible to `--since` incremental exports.
    ///
    /// The row is upserted, never deleted, on revival: a revival with no persisted "alive
    /// since T" signal cannot itself converge through export/import, since there would be
    /// nothing to distinguish "never touched" from "explicitly revived" in the payload.
    pub fn set_dead_at(
        &self,
        memory_id: &str,
        dead: bool,
        reason: Option<&str>,
        marked_at: i64,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let dead_flag = i64::from(dead);
        conn.execute(
            "INSERT INTO memory_status (memory_id, dead, reason, marked_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(memory_id) DO UPDATE SET dead = ?2, reason = ?3, marked_at = ?4",
            params![memory_id, dead_flag, reason, marked_at],
        )?;
        conn.execute(
            "UPDATE memories SET updated_at = MAX(updated_at, ?2) WHERE id = ?1",
            params![memory_id, marked_at],
        )?;
        Ok(())
    }

    /// Ids of every dead memory in a project.
    pub fn get_dead_ids(&self, project_id: &str) -> Result<HashSet<String>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT s.memory_id FROM memory_status s
             JOIN memories m ON s.memory_id = m.id
             WHERE m.project_id = ?1 AND s.dead = 1",
        )?;
        let rows = stmt.query_map(params![project_id], |row| row.get::<_, String>(0))?;
        Ok(rows.flatten().collect())
    }

    /// Whether a single memory is marked dead.
    pub fn is_dead(&self, memory_id: &str) -> Result<bool, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let dead: i64 = conn
            .query_row(
                "SELECT dead FROM memory_status WHERE memory_id = ?1",
                params![memory_id],
                |row| row.get(0),
            )
            .unwrap_or(0);
        Ok(dead != 0)
    }

    /// Count of dead memories in a project.
    pub fn count_dead(&self, project_id: &str) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_status s
             JOIN memories m ON s.memory_id = m.id
             WHERE m.project_id = ?1 AND s.dead = 1",
            params![project_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_successor_follows_a_chain() {
        let map = SupersessionMap::from_pairs(&[("a", "b"), ("b", "c")]);
        assert_eq!(map.terminal_successor("a"), Some("c"));
        assert_eq!(map.terminal_successor("b"), Some("c"));
        assert_eq!(map.terminal_successor("c"), None);
    }

    #[test]
    fn terminal_successor_stops_on_a_cycle() {
        // Reverting a decision makes this reachable: b supersedes a, then a supersedes b.
        let map = SupersessionMap::from_pairs(&[("a", "b"), ("b", "a")]);
        assert!(map.terminal_successor("a").is_some());
        assert!(map.terminal_successor("b").is_some());
    }

    #[test]
    fn terminal_successor_is_depth_capped() {
        let pairs: Vec<(String, String)> = (0..20)
            .map(|i| (format!("m{i}"), format!("m{}", i + 1)))
            .collect();
        let refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let map = SupersessionMap::from_pairs(&refs);
        // Resolution stops at the cap rather than walking the whole chain.
        let reached = map.terminal_successor("m0").unwrap();
        let index: usize = reached.trim_start_matches('m').parse().unwrap();
        assert!(
            index <= MAX_SUPERSESSION_DEPTH + 1,
            "walked too far: {reached}"
        );
        assert!(index >= 1, "did not advance: {reached}");
    }
}
