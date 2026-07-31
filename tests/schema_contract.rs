//! The advertised tool schemas must agree with what the deserializers actually accept.
//!
//! Every tool carries a hand-written JSON schema next to a Rust input struct, and nothing
//! ties the two together. They drifted once already: `handoff_create` advertised
//! `sections.required = ["summary"]` while `HandoffSections` had no serde defaults, so a
//! payload following the published contract was rejected with `missing field blockers`.
//! A caller reading the schema had no way to get it right.
//!
//! This walks every advertised tool, builds the minimal payload its own schema says is
//! legal, and asserts the server accepts it. Anything the deserializer demands beyond the
//! schema's `required` list is a lie in the published contract.

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::error::MemoryError;
use engram_mcp::tools::schemas::get_tool_definitions;
use engram_mcp::tools::{SearchMode, ToolHandler};
use serde_json::{Map, Value, json};

/// A value satisfying `spec`, preferring a legal enum member so the call gets past type
/// parsing and reaches the real handler.
fn sample_value(spec: &Value) -> Value {
    if let Some(first) = spec
        .get("enum")
        .and_then(|e| e.as_array())
        .and_then(|a| a.first())
    {
        return first.clone();
    }
    match spec.get("type").and_then(|t| t.as_str()) {
        Some("string") => json!("x"),
        Some("integer") => json!(1),
        Some("number") => json!(1.0),
        Some("boolean") => json!(true),
        Some("array") => match spec.get("items") {
            // One element, so a required array is not trivially empty.
            Some(items) => json!([sample_value(items)]),
            None => json!([]),
        },
        Some("object") => minimal_payload(spec),
        // Untyped property: an empty object is the least presumptuous thing to send.
        _ => json!({}),
    }
}

/// The smallest object satisfying `schema`: exactly its `required` properties, recursively.
fn minimal_payload(schema: &Value) -> Value {
    let mut out = Map::new();
    let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) else {
        return Value::Object(out);
    };
    let required = schema
        .get("required")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();

    let mut needed: Vec<String> = required
        .iter()
        .filter_map(|r| r.as_str())
        .map(str::to_string)
        .collect();

    // `anyOf` branches carry requirements too: "one of id or trash_id" is a real
    // constraint the handler enforces, so satisfy the first branch.
    if let Some(branches) = schema.get("anyOf").and_then(|a| a.as_array())
        && let Some(first) = branches.first()
        && let Some(branch_required) = first.get("required").and_then(|r| r.as_array())
    {
        needed.extend(
            branch_required
                .iter()
                .filter_map(|r| r.as_str())
                .map(str::to_string),
        );
    }

    for name in needed {
        let spec = properties.get(&name).cloned().unwrap_or(json!({}));
        out.insert(name, sample_value(&spec));
    }
    Value::Object(out)
}

#[test]
fn every_tool_accepts_the_minimal_payload_its_schema_advertises() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path().join("schema_contract.db")).expect("open db");
    let project = "schema-contract";
    db.get_or_create_project(project, project).expect("project");

    let embedding = EmbeddingService::new().expect("model must be available");
    let handler = ToolHandler::new(
        db,
        embedding,
        project.to_string(),
        Some("main".to_string()),
        SearchMode::default(),
    );

    let mut violations: Vec<String> = Vec::new();

    for tool in get_tool_definitions() {
        let schema = Value::Object((*tool.input_schema).clone());
        let args = minimal_payload(&schema);

        // The call may legitimately fail — a fabricated id is NotFound, a stub import
        // payload is a Json error. Only an argument-shape rejection means the published
        // schema disagrees with the deserializer.
        if let Err(MemoryError::InvalidArguments { message, .. }) =
            handler.handle_tool(&tool.name, args.clone())
        {
            violations.push(format!(
                "{}: schema says these fields suffice ({}), but the deserializer rejected \
                 the payload: {}",
                tool.name,
                serde_json::to_string(&args).unwrap_or_default(),
                message
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "tool schemas disagree with their deserializers; a caller following the published \
         contract would be rejected:\n  {}",
        violations.join("\n  ")
    );
}

/// The specific regression: a handoff with only a summary is a legal payload.
#[test]
fn handoff_sections_need_only_a_summary() {
    let sections: engram_mcp::memory::HandoffSections =
        serde_json::from_value(json!({"summary": "Just the summary."}))
            .expect("summary alone must be a legal HandoffSections payload");

    assert_eq!(sections.summary, "Just the summary.");
    assert!(sections.decisions.is_empty());
    assert!(sections.blockers.is_empty());
    assert!(sections.mental_model.is_empty());
    assert!(sections.next_steps.is_empty());
    assert!(sections.notes.is_none());
}
