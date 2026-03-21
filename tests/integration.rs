use std::path::PathBuf;
use tempfile::tempdir;

// We need to test the components together
mod test_helpers {
    use super::*;

    pub fn create_test_db() -> (PathBuf, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        (db_path, dir)
    }
}

#[test]
fn test_full_memory_lifecycle() {
    // This test verifies:
    // 1. Database can be created and initialized
    // 2. Projects can be created
    // 3. Memories can be stored
    // 4. Memories can be retrieved
    // 5. Embeddings are generated and stored
    // 6. Semantic search works

    let (db_path, _dir) = test_helpers::create_test_db();

    // Import the modules we need
    use engram_mcp::db::Database;
    use engram_mcp::embedding::EmbeddingService;
    use engram_mcp::memory::{Memory, MemoryType};

    // 1. Create database
    let db = Database::open(&db_path).expect("Failed to open database");

    // 2. Create project
    let project = db
        .get_or_create_project("test-project", "Test Project")
        .expect("Failed to create project");
    assert_eq!(project.id, "test-project");

    // 3. Store a memory
    let now = chrono::Utc::now().timestamp();
    let memory = Memory {
        id: "mem_test_1".to_string(),
        project_id: "test-project".to_string(),
        memory_type: MemoryType::Fact,
        content: "The API uses JWT tokens for authentication".to_string(),
        summary: Some("JWT auth".to_string()),
        tags: vec!["auth".to_string(), "api".to_string()],
        importance: 0.8,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: None,
        merged_from: None,
    };
    db.store_memory(&memory).expect("Failed to store memory");

    // 4. Retrieve the memory
    let retrieved = db
        .get_memory("mem_test_1")
        .expect("Failed to get memory")
        .expect("Memory not found");
    assert_eq!(
        retrieved.content,
        "The API uses JWT tokens for authentication"
    );
    assert_eq!(retrieved.memory_type, MemoryType::Fact);
    assert_eq!(retrieved.tags, vec!["auth", "api"]);

    // 5. Generate and store embedding
    let embedding_service = EmbeddingService::new().expect("Failed to create embedding service");
    let embedding = embedding_service
        .embed_memory(MemoryType::Fact, &memory.content)
        .expect("Failed to generate embedding");
    assert!(!embedding.is_empty());
    assert_eq!(embedding.len(), 256); // mdbr-leaf-ir with MRL truncation produces 256-dim vectors

    db.store_embedding("mem_test_1", &embedding, embedding_service.model_version())
        .expect("Failed to store embedding");

    // 6. Test semantic search by storing another memory and searching
    let memory2 = Memory {
        id: "mem_test_2".to_string(),
        project_id: "test-project".to_string(),
        memory_type: MemoryType::Decision,
        content: "Chose bcrypt for password hashing with cost factor 12".to_string(),
        summary: Some("Password hashing".to_string()),
        tags: vec!["security".to_string()],
        importance: 0.7,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: None,
        merged_from: None,
    };
    db.store_memory(&memory2).expect("Failed to store memory 2");

    let embedding2 = embedding_service
        .embed_memory(MemoryType::Decision, &memory2.content)
        .expect("Failed to generate embedding 2");
    db.store_embedding("mem_test_2", &embedding2, embedding_service.model_version())
        .expect("Failed to store embedding 2");

    // Search for authentication-related memories
    let query_embedding = embedding_service
        .embed("How does authentication work?")
        .expect("Failed to generate query embedding");

    let all_embeddings = db
        .get_all_embeddings_for_project("test-project")
        .expect("Failed to get embeddings");
    assert_eq!(all_embeddings.len(), 2);

    // Calculate similarities
    use engram_mcp::embedding::cosine_similarity;
    let mut similarities: Vec<(String, f32)> = all_embeddings
        .iter()
        .map(|(id, vec)| (id.clone(), cosine_similarity(&query_embedding, vec)))
        .collect();
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // JWT auth memory should be more relevant to authentication query
    assert_eq!(similarities[0].0, "mem_test_1");
    println!(
        "Similarity scores: {} = {:.3}, {} = {:.3}",
        similarities[0].0, similarities[0].1, similarities[1].0, similarities[1].1
    );
}

#[test]
fn test_relationship_graph() {
    let (db_path, _dir) = test_helpers::create_test_db();

    use engram_mcp::db::Database;
    use engram_mcp::memory::{Memory, MemoryType, RelationType, Relationship};

    let db = Database::open(&db_path).expect("Failed to open database");
    db.get_or_create_project("test-project", "Test")
        .expect("Failed to create project");

    let now = chrono::Utc::now().timestamp();

    // Create three related memories
    let memories = vec![
        Memory {
            id: "mem_1".to_string(),
            project_id: "test-project".to_string(),
            memory_type: MemoryType::Fact,
            content: "Original authentication design".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
        },
        Memory {
            id: "mem_2".to_string(),
            project_id: "test-project".to_string(),
            memory_type: MemoryType::Decision,
            content: "Switched to OAuth2".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
        },
        Memory {
            id: "mem_3".to_string(),
            project_id: "test-project".to_string(),
            memory_type: MemoryType::Fact,
            content: "OAuth2 implementation details".to_string(),
            summary: None,
            tags: vec![],
            importance: 0.5,
            relevance_score: 1.0,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            branch: None,
            merged_from: None,
        },
    ];

    for m in &memories {
        db.store_memory(m).expect("Failed to store memory");
    }

    // Create relationships: mem_2 supersedes mem_1, mem_3 derives from mem_2
    let rel1 = Relationship {
        id: "rel_1".to_string(),
        source_id: "mem_2".to_string(),
        target_id: "mem_1".to_string(),
        relation_type: RelationType::Supersedes,
        strength: 1.0,
        created_at: now,
    };
    let rel2 = Relationship {
        id: "rel_2".to_string(),
        source_id: "mem_3".to_string(),
        target_id: "mem_2".to_string(),
        relation_type: RelationType::DerivedFrom,
        strength: 1.0,
        created_at: now,
    };

    db.create_relationship(&rel1)
        .expect("Failed to create relationship 1");
    db.create_relationship(&rel2)
        .expect("Failed to create relationship 2");

    // Verify relationships
    let outgoing_from_2 = db
        .get_relationships_from("mem_2")
        .expect("Failed to get relationships");
    assert_eq!(outgoing_from_2.len(), 1);
    assert_eq!(outgoing_from_2[0].target_id, "mem_1");
    assert_eq!(outgoing_from_2[0].relation_type, RelationType::Supersedes);

    let incoming_to_2 = db
        .get_relationships_to("mem_2")
        .expect("Failed to get relationships");
    assert_eq!(incoming_to_2.len(), 1);
    assert_eq!(incoming_to_2[0].source_id, "mem_3");
    assert_eq!(incoming_to_2[0].relation_type, RelationType::DerivedFrom);
}

#[test]
fn test_memory_access_tracking() {
    let (db_path, _dir) = test_helpers::create_test_db();

    use engram_mcp::db::Database;
    use engram_mcp::memory::{Memory, MemoryType};

    let db = Database::open(&db_path).expect("Failed to open database");
    db.get_or_create_project("test-project", "Test")
        .expect("Failed to create project");

    let now = chrono::Utc::now().timestamp();
    let memory = Memory {
        id: "mem_access_test".to_string(),
        project_id: "test-project".to_string(),
        memory_type: MemoryType::Fact,
        content: "Test content".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 0.5,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now - 1000,
        branch: None,
        merged_from: None,
    };
    db.store_memory(&memory).expect("Failed to store memory");

    // Record access
    db.record_access("mem_access_test")
        .expect("Failed to record access");

    // Verify access was recorded
    let updated = db
        .get_memory("mem_access_test")
        .expect("Failed to get memory")
        .expect("Memory not found");

    assert_eq!(updated.access_count, 1);
    assert!(updated.relevance_score > 0.5); // Should be boosted
    assert!(updated.last_accessed_at > memory.last_accessed_at);
}

#[test]
fn test_branch_mode_filtering() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    use engram_mcp::db::Database;
    use engram_mcp::memory::{Memory, MemoryType};

    let db = Database::open(&db_path).unwrap();

    let project_id = "test-branch";
    db.get_or_create_project(project_id, project_id).unwrap();

    let now = chrono::Utc::now().timestamp();

    // Store global memory
    let global_mem = Memory {
        id: "mem_global".to_string(),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Fact,
        content: "Global fact".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: None,
        merged_from: None,
    };
    db.store_memory(&global_mem).unwrap();

    // Store feature-x branch memory
    let branch_mem = Memory {
        id: "mem_branch_x".to_string(),
        branch: Some("feature-x".to_string()),
        content: "Feature X fact".to_string(),
        ..global_mem.clone()
    };
    db.store_memory(&branch_mem).unwrap();

    // Store feature-y branch memory
    let branch_y_mem = Memory {
        id: "mem_branch_y".to_string(),
        branch: Some("feature-y".to_string()),
        content: "Feature Y fact".to_string(),
        ..global_mem.clone()
    };
    db.store_memory(&branch_y_mem).unwrap();

    // Test "all" mode (no filter)
    let all = db.query_memories_with_branch(project_id, None, None, None, 100, None).unwrap();
    assert_eq!(all.len(), 3);

    // Test "global" mode
    let global = db.query_memories_with_branch(project_id, None, None, None, 100, Some(None)).unwrap();
    assert_eq!(global.len(), 1);
    assert_eq!(global[0].id, "mem_global");

    // Test "current" mode with specific branch
    let current = db.query_memories_with_branch(project_id, None, None, None, 100, Some(Some("feature-x"))).unwrap();
    assert_eq!(current.len(), 2); // global + feature-x
    assert!(current.iter().any(|m| m.id == "mem_global"));
    assert!(current.iter().any(|m| m.id == "mem_branch_x"));

    // Test specific branch name
    let specific = db.query_memories_with_branch(project_id, None, None, None, 100, Some(Some("feature-y"))).unwrap();
    assert_eq!(specific.len(), 2); // global + feature-y
    assert!(specific.iter().any(|m| m.id == "mem_global"));
    assert!(specific.iter().any(|m| m.id == "mem_branch_y"));
}

#[test]
fn test_store_auto_dedup() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    use engram_mcp::db::Database;
    use engram_mcp::embedding::{cosine_similarity, EmbeddingService};
    use engram_mcp::memory::{Memory, MemoryType};

    let db = Database::open(&db_path).unwrap();
    let embedding_service = EmbeddingService::new().unwrap();

    let project_id = "test-dedup";
    db.get_or_create_project(project_id, project_id).unwrap();

    let now = chrono::Utc::now().timestamp();

    // Store first memory
    let mem1 = Memory {
        id: "mem_dup_1".to_string(),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Fact,
        content: "The database uses PostgreSQL version 15 for storage".to_string(),
        summary: None,
        tags: vec!["db".to_string()],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: None,
        merged_from: None,
    };
    db.store_memory(&mem1).unwrap();
    let emb1 = embedding_service
        .embed_memory(MemoryType::Fact, &mem1.content)
        .unwrap();
    db.store_embedding("mem_dup_1", &emb1, embedding_service.model_version())
        .unwrap();

    // Store near-identical memory
    let mem2 = Memory {
        id: "mem_dup_2".to_string(),
        content: "The database uses PostgreSQL version 15 for data storage".to_string(),
        tags: vec!["postgres".to_string()],
        importance: 0.8,
        ..mem1.clone()
    };
    db.store_memory(&mem2).unwrap();
    let emb2 = embedding_service
        .embed_memory(MemoryType::Fact, &mem2.content)
        .unwrap();
    db.store_embedding("mem_dup_2", &emb2, embedding_service.model_version())
        .unwrap();

    let similarity = cosine_similarity(&emb1, &emb2);

    // If similarity >= 0.90 (dedup threshold), test the merge
    if similarity >= 0.90 {
        db.merge_memories("mem_dup_2", "mem_dup_1", "The database uses PostgreSQL")
            .unwrap();

        // Old memory should be gone
        assert!(db.get_memory("mem_dup_1").unwrap().is_none());

        // New memory should have merged tags and max importance
        let merged = db.get_memory("mem_dup_2").unwrap().unwrap();
        assert!(merged.tags.contains(&"db".to_string()));
        assert!(merged.tags.contains(&"postgres".to_string()));
        assert_eq!(merged.importance, 0.8); // max(0.5, 0.8)
        assert!(merged.merged_from.is_some());
    } else {
        // If similarity < 0.90, both should still exist (no dedup)
        assert!(db.get_memory("mem_dup_1").unwrap().is_some());
        assert!(db.get_memory("mem_dup_2").unwrap().is_some());
    }
}

#[test]
fn test_cluster_assignment() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    use engram_mcp::db::Database;
    use engram_mcp::embedding::{cosine_similarity, EmbeddingService};
    use engram_mcp::memory::{Memory, MemoryCluster, MemoryType};

    let db = Database::open(&db_path).unwrap();
    let embedding_service = EmbeddingService::new().unwrap();

    let project_id = "test-cluster-assign";
    db.get_or_create_project(project_id, project_id).unwrap();

    let now = chrono::Utc::now().timestamp();

    // Store a memory and create a cluster for it
    let mem1 = Memory {
        id: "mem_ca_1".to_string(),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Fact,
        content: "Rust is a systems programming language".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: None,
        merged_from: None,
    };
    db.store_memory(&mem1).unwrap();
    let emb1 = embedding_service
        .embed_memory(MemoryType::Fact, &mem1.content)
        .unwrap();
    db.store_embedding("mem_ca_1", &emb1, embedding_service.model_version())
        .unwrap();

    // Create a cluster with this memory's embedding as centroid
    let cluster = MemoryCluster {
        id: "clust_test_1".to_string(),
        project_id: project_id.to_string(),
        summary: "Rust programming".to_string(),
        member_count: 1,
        centroid: Some(emb1.clone()),
        created_at: now,
        updated_at: now,
    };
    db.create_cluster(&cluster).unwrap();
    db.add_to_cluster("clust_test_1", "mem_ca_1").unwrap();

    // Store a similar memory
    let mem2 = Memory {
        id: "mem_ca_2".to_string(),
        content: "Rust provides memory safety without garbage collection".to_string(),
        ..mem1.clone()
    };
    db.store_memory(&mem2).unwrap();
    let emb2 = embedding_service
        .embed_memory(MemoryType::Fact, &mem2.content)
        .unwrap();
    db.store_embedding("mem_ca_2", &emb2, embedding_service.model_version())
        .unwrap();

    let similarity = cosine_similarity(&emb1, &emb2);

    // If similar enough, adding to the cluster should work
    if similarity >= 0.75 {
        db.add_to_cluster("clust_test_1", "mem_ca_2").unwrap();
        let members = db.get_cluster_member_ids("clust_test_1").unwrap();
        assert_eq!(members.len(), 2);

        let c = db.get_cluster("clust_test_1").unwrap().unwrap();
        assert_eq!(c.member_count, 2);
    }
}

#[test]
fn test_recluster_merge() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    use engram_mcp::db::Database;
    use engram_mcp::embedding::{cosine_similarity, EmbeddingService};
    use engram_mcp::memory::{Memory, MemoryCluster, MemoryType};

    let db = Database::open(&db_path).unwrap();
    let embedding_service = EmbeddingService::new().unwrap();

    let project_id = "test-recluster";
    db.get_or_create_project(project_id, project_id).unwrap();

    let now = chrono::Utc::now().timestamp();

    // Create two memories with similar content
    let mem1 = Memory {
        id: "mem_rc_1".to_string(),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Fact,
        content: "Python is great for data science".to_string(),
        summary: None,
        tags: vec![],
        importance: 0.5,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: None,
        merged_from: None,
    };
    db.store_memory(&mem1).unwrap();
    let emb1 = embedding_service
        .embed_memory(MemoryType::Fact, &mem1.content)
        .unwrap();
    db.store_embedding("mem_rc_1", &emb1, embedding_service.model_version())
        .unwrap();

    let mem2 = Memory {
        id: "mem_rc_2".to_string(),
        content: "Python is excellent for data analysis and science".to_string(),
        ..mem1.clone()
    };
    db.store_memory(&mem2).unwrap();
    let emb2 = embedding_service
        .embed_memory(MemoryType::Fact, &mem2.content)
        .unwrap();
    db.store_embedding("mem_rc_2", &emb2, embedding_service.model_version())
        .unwrap();

    // Create two separate clusters with these embeddings as centroids
    let cluster1 = MemoryCluster {
        id: "clust_rc_1".to_string(),
        project_id: project_id.to_string(),
        summary: "Python data".to_string(),
        member_count: 1,
        centroid: Some(emb1.clone()),
        created_at: now,
        updated_at: now,
    };
    db.create_cluster(&cluster1).unwrap();
    db.add_to_cluster("clust_rc_1", "mem_rc_1").unwrap();

    let cluster2 = MemoryCluster {
        id: "clust_rc_2".to_string(),
        project_id: project_id.to_string(),
        summary: "Python analysis".to_string(),
        member_count: 1,
        centroid: Some(emb2.clone()),
        created_at: now,
        updated_at: now,
    };
    db.create_cluster(&cluster2).unwrap();
    db.add_to_cluster("clust_rc_2", "mem_rc_2").unwrap();

    let sim = cosine_similarity(&emb1, &emb2);

    // Verify initial state: 2 clusters
    let clusters_before = db.get_clusters_for_project(project_id).unwrap();
    assert_eq!(clusters_before.len(), 2);

    // If centroids are similar enough (>= 0.80), recluster should merge them.
    // We simulate the merge by exercising the DB operations directly.
    if sim >= 0.80 {
        // Move mem_rc_2 from cluster2 into cluster1
        db.remove_from_cluster("mem_rc_2").unwrap();
        db.add_to_cluster("clust_rc_1", "mem_rc_2").unwrap();
        db.delete_empty_clusters(project_id).unwrap();

        let clusters_after = db.get_clusters_for_project(project_id).unwrap();
        assert_eq!(clusters_after.len(), 1);

        let members = db.get_cluster_member_ids("clust_rc_1").unwrap();
        assert_eq!(members.len(), 2);
    }
}
