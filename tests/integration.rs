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
    use agent_memory::db::Database;
    use agent_memory::embedding::EmbeddingService;
    use agent_memory::memory::{Memory, MemoryType};

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
    };
    db.store_memory(&memory).expect("Failed to store memory");

    // 4. Retrieve the memory
    let retrieved = db
        .get_memory("mem_test_1")
        .expect("Failed to get memory")
        .expect("Memory not found");
    assert_eq!(retrieved.content, "The API uses JWT tokens for authentication");
    assert_eq!(retrieved.memory_type, MemoryType::Fact);
    assert_eq!(retrieved.tags, vec!["auth", "api"]);

    // 5. Generate and store embedding
    let embedding_service = EmbeddingService::new().expect("Failed to create embedding service");
    let embedding = embedding_service
        .embed_memory(MemoryType::Fact, &memory.content)
        .expect("Failed to generate embedding");
    assert!(!embedding.is_empty());
    assert_eq!(embedding.len(), 384); // all-MiniLM-L6-v2 produces 384-dim vectors

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
    use agent_memory::embedding::cosine_similarity;
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

    use agent_memory::db::Database;
    use agent_memory::memory::{Memory, MemoryType, Relationship, RelationType};

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

    use agent_memory::db::Database;
    use agent_memory::memory::{Memory, MemoryType};

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
