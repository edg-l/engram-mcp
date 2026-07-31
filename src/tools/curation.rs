//! Applying curation status to a ranked result set.
//!
//! Retrieval must not return a memory that is no longer the answer, but it must also not
//! return silence: an empty result reads as "nobody has looked into this", which is how a
//! stale conclusion gets derived a second time. So a superseded memory is replaced by
//! whatever superseded it, annotated with where the match actually came from, and only a
//! memory whose subject is gone entirely is dropped.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::db::{Database, SupersessionMap};
use crate::embedding::cosine_similarity;
use crate::error::MemoryError;
use crate::memory::{Memory, MemoryType};

/// Why a memory is in a result set that its own score did not earn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedVia {
    /// The superseded memory whose match produced this result.
    pub superseded_id: String,
    /// First line of the superseded memory, so the caller can see what it replaced.
    pub superseded_preview: String,
}

/// What retrieval should do with one candidate id.
pub enum Resolution {
    /// Return this memory as matched.
    Keep,
    /// Return `successor` instead, crediting the match to the superseded memory.
    Redirect {
        successor_id: String,
        via: MatchedVia,
    },
    /// Drop the match: the subject is gone and there is nothing to point at.
    Drop,
}

/// Per-project curation status, loaded once per retrieval call.
pub struct CurationView {
    dead: HashSet<String>,
    supersession: SupersessionMap,
    /// First lines of superseded memories, for the `matched_via` annotation.
    previews: HashMap<String, String>,
}

impl CurationView {
    /// Load status for a project. Two small queries, in the same shape as the existing
    /// per-project embedding load.
    pub fn load(db: &Database, project_id: &str) -> Result<Self, MemoryError> {
        Ok(Self {
            dead: db.get_dead_ids(project_id)?,
            supersession: db.get_supersession_map(project_id)?,
            previews: HashMap::new(),
        })
    }

    /// An empty view: nothing superseded, nothing dead.
    pub fn empty() -> Self {
        Self {
            dead: HashSet::new(),
            supersession: SupersessionMap::default(),
            previews: HashMap::new(),
        }
    }

    /// Record the text of a memory so a redirect can describe what it replaced.
    pub fn note_preview(&mut self, memory_id: &str, content: &str) {
        if self.supersession.is_superseded(memory_id) {
            self.previews
                .insert(memory_id.to_string(), first_line(content));
        }
    }

    /// Decide what to do with a candidate.
    pub fn resolve(&self, memory_id: &str) -> Resolution {
        if self.dead.contains(memory_id) {
            return Resolution::Drop;
        }
        let Some(successor) = self.supersession.terminal_successor(memory_id) else {
            return Resolution::Keep;
        };
        // A chain that ends in a dead memory has nothing current to offer.
        if self.dead.contains(successor) {
            return Resolution::Drop;
        }
        Resolution::Redirect {
            successor_id: successor.to_string(),
            via: MatchedVia {
                superseded_id: memory_id.to_string(),
                superseded_preview: self
                    .previews
                    .get(memory_id)
                    .cloned()
                    .unwrap_or_else(|| memory_id.to_string()),
            },
        }
    }
}

fn first_line(content: &str) -> String {
    let line = content.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    line.chars().take(120).collect()
}

/// Lowest similarity at which an existing memory is worth reporting as something a new
/// memory might supersede.
///
/// Set where same-subject memories stop being distinguishable from merely same-topic
/// ones. There is no upper bound: a pair can sit above the dedup threshold and still not
/// have merged, because dedup refuses composites and caller-exempted memories. Those are
/// the *most* likely supersessions, not the least, so capping the band at the dedup
/// threshold would hide exactly the pairs worth asking about.
pub const SUPERSESSION_CANDIDATE_MIN: f32 = 0.75;

/// Most supersession candidates reported per store, highest similarity first.
pub const SUPERSESSION_CANDIDATE_LIMIT: usize = 5;

/// An existing memory that a newly stored one might be replacing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupersessionCandidate {
    pub id: String,
    pub similarity: f32,
    #[serde(rename = "type")]
    pub memory_type: String,
    pub preview: String,
    pub updated_at: i64,
}

/// Existing same-type memories similar enough to be about the same subject as
/// `embedding`, excluding the new memory itself and anything it already merged with.
///
/// These similarities are computed for dedup anyway and otherwise discarded. Reporting
/// them is the only way a caller storing "X is now Y" learns that a memory from months
/// ago says "X is Z": that memory never surfaced, so it never became a candidate for
/// anything. Never acted on automatically — cosine cannot separate "contradicts" from
/// "elaborates", and at this range it frequently means the latter.
pub fn supersession_candidates(
    db: &Database,
    project_id: &str,
    embedding: &[f32],
    memory_type: MemoryType,
    exclude: &[&str],
) -> Result<Vec<SupersessionCandidate>, MemoryError> {
    let embeddings = db.get_all_embeddings_for_project_and_global(project_id)?;
    let memories: HashMap<String, Memory> = db
        .get_all_memories_for_project(project_id)?
        .into_iter()
        .map(|m| (m.id.clone(), m))
        .collect();

    let mut candidates: Vec<(f32, &Memory)> = embeddings
        .iter()
        .filter(|(id, _)| !exclude.contains(&id.as_str()))
        .filter_map(|(id, vec)| {
            let memory = memories.get(id)?;
            if memory.memory_type != memory_type {
                return None;
            }
            let similarity = cosine_similarity(embedding, vec);
            (similarity >= SUPERSESSION_CANDIDATE_MIN).then_some((similarity, memory))
        })
        .collect();

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(SUPERSESSION_CANDIDATE_LIMIT);

    Ok(candidates
        .into_iter()
        .map(|(similarity, memory)| SupersessionCandidate {
            id: memory.id.clone(),
            similarity,
            memory_type: memory.memory_type.as_str().to_string(),
            preview: memory.content.chars().take(160).collect(),
            updated_at: memory.updated_at,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(dead: &[&str], pairs: &[(&str, &str)]) -> CurationView {
        CurationView {
            dead: dead.iter().map(|s| s.to_string()).collect(),
            supersession: SupersessionMap::from_pairs(pairs),
            previews: HashMap::new(),
        }
    }

    #[test]
    fn live_memory_is_kept() {
        let v = view(&[], &[]);
        assert!(matches!(v.resolve("a"), Resolution::Keep));
    }

    #[test]
    fn dead_memory_is_dropped() {
        let v = view(&["a"], &[]);
        assert!(matches!(v.resolve("a"), Resolution::Drop));
    }

    #[test]
    fn superseded_memory_redirects_to_its_successor() {
        let v = view(&[], &[("a", "b")]);
        match v.resolve("a") {
            Resolution::Redirect { successor_id, via } => {
                assert_eq!(successor_id, "b");
                assert_eq!(via.superseded_id, "a");
            }
            _ => panic!("expected a redirect"),
        }
    }

    #[test]
    fn redirect_follows_the_chain_to_the_current_memory() {
        let v = view(&[], &[("a", "b"), ("b", "c")]);
        match v.resolve("a") {
            Resolution::Redirect { successor_id, .. } => assert_eq!(successor_id, "c"),
            _ => panic!("expected a redirect"),
        }
    }

    #[test]
    fn redirect_to_a_dead_successor_is_dropped() {
        let v = view(&["b"], &[("a", "b")]);
        assert!(matches!(v.resolve("a"), Resolution::Drop));
    }
}
