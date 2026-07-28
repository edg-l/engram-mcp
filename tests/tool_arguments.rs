//! Tests for tool argument deserialization.
//!
//! Field order never affects parsing (arguments arrive as a JSON object), a
//! misnamed `type` is accepted via alias, and a genuinely absent field produces
//! an error that lists the fields actually received.

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::error::MemoryError;
use engram_mcp::tools::ToolHandler;
use engram_mcp::tools::scoring::SearchMode;
use serde_json::{Value, json};

fn setup() -> (ToolHandler, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path().join("test.db")).unwrap();
    db.get_or_create_project("proj", "proj").unwrap();
    let handler = ToolHandler::new(
        db,
        EmbeddingService::new().unwrap(),
        "proj".to_string(),
        Some("main".to_string()),
        SearchMode::default(),
    );
    (handler, dir)
}

#[test]
fn field_order_does_not_affect_parsing() {
    let (h, _dir) = setup();

    // A long content field ahead of `type` is the shape that was reported as
    // failing; both orders must behave identically.
    let long_content = "The migration runbook is long. ".repeat(600);

    let type_last = h
        .handle_tool(
            "memory_store",
            json!({"content": long_content.clone(), "type": "decision", "tags": ["migration"]}),
        )
        .expect("type after a long content field must parse");
    let type_first = h
        .handle_tool(
            "memory_store",
            json!({"type": "decision", "content": format!("{long_content} Variant two."), "tags": ["migration"]}),
        )
        .expect("type before content must parse");

    for result in [&type_last, &type_first] {
        assert!(result["id"].as_str().unwrap().starts_with("mem_"));
        assert_eq!(result["project"].as_str().unwrap(), "proj");
    }
}

#[test]
fn memory_type_is_accepted_as_alias_for_type() {
    let (h, _dir) = setup();

    let result = h
        .handle_tool(
            "memory_store",
            json!({"content": "Retries use exponential backoff", "memory_type": "pattern"}),
        )
        .expect("memory_type must be accepted as an alias for type");

    let id = result["id"].as_str().unwrap();
    let stored = h.database().get_memory(id).unwrap().unwrap();
    assert_eq!(stored.memory_type.as_str(), "pattern");
}

#[test]
fn batch_items_accept_the_type_alias() {
    let (h, _dir) = setup();

    let result = h
        .handle_tool(
            "memory_store_batch",
            json!({"memories": [
                {"content": "First batch item", "type": "fact"},
                {"content": "Second batch item", "memory_type": "fact"},
            ]}),
        )
        .expect("batch items must accept both spellings");

    assert_eq!(result["count"].as_u64().unwrap(), 2);
    assert_eq!(result["project"].as_str().unwrap(), "proj");
}

#[test]
fn missing_field_error_lists_received_fields() {
    let (h, _dir) = setup();

    let err = h
        .handle_tool(
            "memory_store",
            json!({"content": "No type anywhere", "kind": "fact", "tags": []}),
        )
        .expect_err("a genuinely absent type must be rejected");

    match err {
        MemoryError::InvalidArguments {
            tool,
            message,
            received,
        } => {
            assert_eq!(tool, "memory_store");
            assert!(message.contains("missing field"), "message: {message}");
            // The received list is what makes the error diagnosable: the caller
            // can see `type` really was not among the fields sent.
            assert!(received.contains("content"), "received: {received}");
            assert!(received.contains("kind"), "received: {received}");
            assert!(!received.contains("type"), "received: {received}");
        }
        other => panic!("expected InvalidArguments, got {other:?}"),
    }
}

#[test]
fn non_object_arguments_are_reported_as_such() {
    let (h, _dir) = setup();

    let err = h
        .handle_tool("memory_query", Value::Null)
        .expect_err("null arguments must be rejected");

    match err {
        MemoryError::InvalidArguments { received, .. } => {
            assert!(received.contains("null"), "received: {received}");
        }
        other => panic!("expected InvalidArguments, got {other:?}"),
    }
}

#[test]
fn store_result_reports_the_project_it_landed_in() {
    let (h, _dir) = setup();

    let result = h
        .handle_tool(
            "memory_store",
            json!({"content": "Where did this land", "type": "fact"}),
        )
        .unwrap();
    assert_eq!(result["project"].as_str().unwrap(), "proj");

    let formatted = engram_mcp::format::compact_tool_result("memory_store", &result, 300);
    assert!(
        formatted.contains("in proj"),
        "formatted output should name the project: {formatted}"
    );
}
