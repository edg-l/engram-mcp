//! Resolving caller-supplied memory ids that may have been consumed by a dedup merge.
//!
//! `related_to`, `supersedes`, and `memory_link`'s `source_id`/`target_id` all name a
//! memory the caller expects to still exist. A dedup merge deletes the losing side and
//! records it in the survivor's `merged_from`, so an id from before the merge is not
//! garbage — it names a memory that has a documented successor. Resolving it here, before
//! any write, is what keeps a retry after an unrelated merge from failing with a foreign
//! key error against a memory that was already committed.

use serde::Serialize;

use crate::db::Database;
use crate::error::MemoryError;

/// Longest merge chain followed when resolving a caller-supplied id. `merge_memories`
/// never actually nests — a memory that has already absorbed one merge is excluded from
/// auto-dedup as a candidate — so this is defensive rather than reachable today, matching
/// the depth cap `db::status::SupersessionMap` applies to supersession chains.
const MAX_MERGE_DEPTH: usize = 5;

/// A caller-supplied id that resolved to a different, live one.
#[derive(Debug, Clone, Serialize)]
pub struct Redirect {
    pub from: String,
    pub to: String,
}

/// Resolve `id` to a memory that still exists.
///
/// - If `id` names an existing memory, returns it unchanged.
/// - Otherwise, if `id` was consumed by a dedup merge, follows `merged_from` to the
///   memory that absorbed it, transitively (depth-capped, cycle-guarded in case a
///   survivor is itself later consumed).
/// - Otherwise, `MemoryError::NotFound` naming both `id` and `arg` (e.g. `related_to`),
///   so the error reads as "this id, from that argument" rather than a bare id.
///
/// The second element of the result is `Some` only when resolution moved away from the
/// id the caller passed in.
pub fn resolve_link_target(
    db: &Database,
    project_id: &str,
    id: &str,
    arg: &str,
) -> Result<(String, Option<Redirect>), MemoryError> {
    if db.get_memory(id)?.is_some() {
        return Ok((id.to_string(), None));
    }

    let mut current = id.to_string();
    let mut seen = std::collections::HashSet::new();
    seen.insert(current.clone());

    for _ in 0..MAX_MERGE_DEPTH {
        let Some(next) = db.find_merge_survivor(project_id, &current)? else {
            break;
        };
        if !seen.insert(next.clone()) {
            break; // cycle guard
        }
        if db.get_memory(&next)?.is_some() {
            return Ok((
                next.clone(),
                Some(Redirect {
                    from: id.to_string(),
                    to: next,
                }),
            ));
        }
        current = next;
    }

    Err(MemoryError::NotFound(format!("{id} (referenced by {arg})")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Memory, MemoryType};

    fn make_memory(id: &str, project_id: &str, content: &str) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: project_id.to_string(),
            memory_type: MemoryType::Fact,
            content: content.to_string(),
            summary: None,
            tags: vec![],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: 0,
            updated_at: 0,
            last_accessed_at: 0,
            branch: None,
            merged_from: None,
            external_artifacts: None,
            pinned: false,
            global: false,
        }
    }

    #[test]
    fn resolves_an_existing_id_to_itself() {
        let db = Database::open_in_memory().unwrap();
        let mem = make_memory("mem_a", "proj", "hello");
        db.store_memory(&mem).unwrap();

        let (resolved, redirect) = resolve_link_target(&db, "proj", "mem_a", "related_to").unwrap();
        assert_eq!(resolved, "mem_a");
        assert!(redirect.is_none());
    }

    #[test]
    fn resolves_a_merged_away_id_to_its_survivor() {
        let db = Database::open_in_memory().unwrap();
        let survivor = make_memory("mem_survivor", "proj", "current answer");
        let consumed = make_memory("mem_consumed", "proj", "old answer");
        db.store_memory(&survivor).unwrap();
        db.store_memory(&consumed).unwrap();
        db.merge_memories("mem_survivor", "mem_consumed").unwrap();

        let (resolved, redirect) =
            resolve_link_target(&db, "proj", "mem_consumed", "related_to").unwrap();
        assert_eq!(resolved, "mem_survivor");
        let redirect = redirect.expect("redirect should be reported");
        assert_eq!(redirect.from, "mem_consumed");
        assert_eq!(redirect.to, "mem_survivor");
    }

    #[test]
    fn unknown_id_is_not_found() {
        let db = Database::open_in_memory().unwrap();
        let err = resolve_link_target(&db, "proj", "mem_nonexistent", "supersedes").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("mem_nonexistent"));
        assert!(message.contains("supersedes"));
    }

    #[test]
    fn resolution_is_scoped_to_project_and_global() {
        // A merge recorded under a different, non-global project must not resolve for
        // this one: find_merge_survivor's scoping, exercised through the resolver.
        let db = Database::open_in_memory().unwrap();
        let survivor = make_memory("mem_survivor2", "other_project", "current answer");
        let consumed = make_memory("mem_consumed2", "other_project", "old answer");
        db.store_memory(&survivor).unwrap();
        db.store_memory(&consumed).unwrap();
        db.merge_memories("mem_survivor2", "mem_consumed2").unwrap();

        let err = resolve_link_target(&db, "proj", "mem_consumed2", "related_to").unwrap_err();
        assert!(err.to_string().contains("mem_consumed2"));
    }
}
