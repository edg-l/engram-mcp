//! Caching layer for Engram performance optimization.
//!
//! Provides LRU caches with TTL for:
//! - Query embeddings: Avoid re-computing embeddings for repeated queries
//! - Search results: Cache similarity search results for hot queries

use moka::sync::Cache;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

/// Cache for query text -> embedding vector mappings.
/// Avoids re-computing embeddings for repeated queries.
///
/// Configuration:
/// - Capacity: 1000 entries
/// - TTL: 5 minutes
/// - Memory overhead: ~1.5MB (1000 * 384 floats * 4 bytes)
pub struct QueryEmbeddingCache {
    cache: Cache<String, Arc<Vec<f32>>>,
}

impl QueryEmbeddingCache {
    /// Create a new query embedding cache with default settings.
    pub fn new() -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(1000)
                .time_to_live(Duration::from_secs(300)) // 5 minutes
                .build(),
        }
    }

    /// Get a cached embedding for a query string.
    pub fn get(&self, query: &str) -> Option<Vec<f32>> {
        self.cache.get(query).map(|arc| (*arc).clone())
    }

    /// Store an embedding for a query string.
    pub fn insert(&self, query: String, embedding: Vec<f32>) {
        self.cache.insert(query, Arc::new(embedding));
    }
}

impl Default for QueryEmbeddingCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash helper for embedding vectors (used as part of search result cache key).
fn hash_embedding(embedding: &[f32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // Hash the first 32 components for speed (sufficient for uniqueness)
    for &val in embedding.iter().take(32) {
        val.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Key for search result cache: (project_id, embedding_hash)
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct SearchResultKey {
    project_id: String,
    embedding_hash: u64,
}

impl SearchResultKey {
    pub fn new(project_id: &str, embedding: &[f32]) -> Self {
        Self {
            project_id: project_id.to_string(),
            embedding_hash: hash_embedding(embedding),
        }
    }
}

/// Cached search result: vector of (memory_id, similarity) pairs.
pub type SearchResultValue = Vec<(String, f32)>;

/// Cache for search results to avoid repeated vector similarity computations.
///
/// Configuration:
/// - Capacity: 500 entries
/// - TTL: 60 seconds (short to avoid stale results)
///
/// Invalidation triggers:
/// - memory_store
/// - memory_update
/// - memory_delete
/// - memory_import
pub struct SearchResultCache {
    cache: Cache<SearchResultKey, Arc<SearchResultValue>>,
}

impl SearchResultCache {
    /// Create a new search result cache with default settings.
    pub fn new() -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(500)
                .time_to_live(Duration::from_secs(60))
                .build(),
        }
    }

    /// Get cached search results for a project and query embedding.
    pub fn get(&self, project_id: &str, embedding: &[f32]) -> Option<SearchResultValue> {
        let key = SearchResultKey::new(project_id, embedding);
        self.cache.get(&key).map(|arc| (*arc).clone())
    }

    /// Store search results for a project and query embedding.
    pub fn insert(&self, project_id: &str, embedding: &[f32], results: SearchResultValue) {
        let key = SearchResultKey::new(project_id, embedding);
        self.cache.insert(key, Arc::new(results));
    }

    /// Invalidate all cached results for a project.
    /// Call this when memories are added, updated, or deleted.
    pub fn invalidate_project(&self, project_id: &str) {
        // Collect keys to invalidate first, then invalidate them
        // This is needed because invalidate_entries_if may be deferred
        let keys_to_remove: Vec<SearchResultKey> = self
            .cache
            .iter()
            .filter(|(k, _)| k.project_id == project_id)
            .map(|(k, _)| (*k).clone())
            .collect();

        for key in keys_to_remove {
            self.cache.invalidate(&key);
        }
    }
}

impl Default for SearchResultCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_embedding_cache() {
        let cache = QueryEmbeddingCache::new();

        // Test miss
        assert!(cache.get("test query").is_none());

        // Test insert and hit
        let embedding = vec![0.1, 0.2, 0.3];
        cache.insert("test query".to_string(), embedding.clone());

        let cached = cache.get("test query");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), embedding);
    }

    #[test]
    fn test_search_result_cache() {
        let cache = SearchResultCache::new();
        let embedding = vec![0.1; 384];
        let results = vec![("mem_1".to_string(), 0.95), ("mem_2".to_string(), 0.85)];

        // Test miss
        assert!(cache.get("project1", &embedding).is_none());

        // Test insert and hit
        cache.insert("project1", &embedding, results.clone());

        let cached = cache.get("project1", &embedding);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), results);

        // Test invalidation
        cache.invalidate_project("project1");
        assert!(cache.get("project1", &embedding).is_none());
    }

    #[test]
    fn test_embedding_hash_consistency() {
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let hash1 = hash_embedding(&embedding);
        let hash2 = hash_embedding(&embedding);
        assert_eq!(hash1, hash2);

        let different = vec![0.1, 0.2, 0.3, 0.5];
        let hash3 = hash_embedding(&different);
        assert_ne!(hash1, hash3);
    }
}
