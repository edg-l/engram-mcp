//! Tests for the optional `project` argument on MCP tools.
//!
//! Each tool may target a project other than the one the server was launched
//! with; unknown project IDs are rejected instead of silently reading an empty
//! store.

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::error::MemoryError;
use engram_mcp::tools::ToolHandler;
use engram_mcp::tools::scoring::SearchMode;
use serde_json::{Value, json};

const HOME: &str = "home-project";
const OTHER: &str = "other-project";

/// Create a ToolHandler scoped to `HOME`, with `OTHER` also present in the DB.
fn setup(branch: Option<&str>) -> (ToolHandler, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path().join("test.db")).unwrap();
    db.get_or_create_project(HOME, HOME).unwrap();
    db.get_or_create_project(OTHER, OTHER).unwrap();
    let handler = ToolHandler::new(
        db,
        EmbeddingService::new().unwrap(),
        HOME.to_string(),
        branch.map(String::from),
        SearchMode::default(),
    );
    (handler, dir)
}

fn call(handler: &ToolHandler, tool: &str, args: Value) -> Value {
    handler
        .handle_tool(tool, args)
        .unwrap_or_else(|e| panic!("{tool} failed: {e}"))
}

#[test]
fn store_and_query_target_another_project() {
    let (h, _dir) = setup(Some("main"));

    call(
        &h,
        "memory_store",
        json!({
            "content": "The other project pins its Redis client to version 0.27",
            "type": "fact",
            "project": OTHER,
        }),
    );

    // Visible when querying that project.
    let hits = call(
        &h,
        "memory_query",
        json!({"query": "Redis client version", "project": OTHER}),
    );
    assert_eq!(hits["count"].as_u64().unwrap(), 1);
    assert_eq!(
        hits["memories"][0]["memory"]["project_id"]
            .as_str()
            .unwrap(),
        OTHER
    );

    // Not visible from the server's own project.
    let own = call(&h, "memory_query", json!({"query": "Redis client version"}));
    assert_eq!(own["count"].as_u64().unwrap(), 0);
}

#[test]
fn branch_scoped_memory_in_another_project_is_readable() {
    let (h, _dir) = setup(Some("main"));

    // "auto" cannot mean the server's branch for a foreign project, so the
    // memory lands as project-global and stays readable.
    call(
        &h,
        "memory_store",
        json!({
            "content": "Deployment runs through the staging pipeline first",
            "type": "pattern",
            "branch": "auto",
            "project": OTHER,
        }),
    );

    // A branch-scoped memory written explicitly is still found under the
    // default branch_mode, because "current" widens to all branches when the
    // target project is not the server's own.
    call(
        &h,
        "memory_store",
        json!({
            "content": "The feature flag rollout is gated behind an env var",
            "type": "decision",
            "branch": "feature/rollout",
            "project": OTHER,
        }),
    );

    let hits = call(
        &h,
        "memory_query",
        json!({"query": "feature flag rollout env var", "project": OTHER}),
    );
    assert!(hits["count"].as_u64().unwrap() >= 1);
}

#[test]
fn unknown_project_is_rejected_with_known_projects() {
    let (h, _dir) = setup(None);

    let err = h
        .handle_tool(
            "memory_query",
            json!({"query": "anything", "project": "nope"}),
        )
        .expect_err("unknown project must be rejected");

    match err {
        MemoryError::UnknownProject { requested, known } => {
            assert_eq!(requested, "nope");
            assert!(known.contains(HOME), "known projects listed: {known}");
            assert!(known.contains(OTHER), "known projects listed: {known}");
        }
        other => panic!("expected UnknownProject, got {other:?}"),
    }
}

#[test]
fn empty_project_argument_falls_back_to_own_project() {
    let (h, _dir) = setup(None);

    call(
        &h,
        "memory_store",
        json!({"content": "Home project fact about the build cache", "type": "fact"}),
    );

    let stats = call(&h, "memory_stats", json!({"project": ""}));
    assert_eq!(stats["project_id"].as_str().unwrap(), HOME);
    assert_eq!(stats["memory_count"].as_u64().unwrap(), 1);
}

#[test]
fn memory_stats_reports_target_project() {
    let (h, _dir) = setup(None);

    call(
        &h,
        "memory_store",
        json!({"content": "Only the other project knows this", "type": "fact", "project": OTHER}),
    );

    let stats = call(&h, "memory_stats", json!({"project": OTHER}));
    assert_eq!(stats["project_id"].as_str().unwrap(), OTHER);
    assert_eq!(stats["memory_count"].as_u64().unwrap(), 1);

    let own = call(&h, "memory_stats", json!({}));
    assert_eq!(own["project_id"].as_str().unwrap(), HOME);
    assert_eq!(own["memory_count"].as_u64().unwrap(), 0);
}

#[test]
fn memory_projects_lists_every_project() {
    let (h, _dir) = setup(None);

    call(
        &h,
        "memory_store",
        json!({"content": "A fact in the home project", "type": "fact"}),
    );
    call(
        &h,
        "memory_store",
        json!({"content": "A fact in the other project", "type": "fact", "project": OTHER}),
    );
    call(
        &h,
        "adr_create",
        json!({
            "title": "Adopt structured logging",
            "context": "Log lines are unparseable across services.",
            "decision": "Emit JSON logs from every service.",
            "consequences": "Slightly larger log volume.",
            "project": OTHER,
        }),
    );

    let listing = call(&h, "memory_projects", json!({}));
    assert_eq!(listing["current_project"].as_str().unwrap(), HOME);

    let projects = listing["projects"].as_array().unwrap();
    let home = projects
        .iter()
        .find(|p| p["project_id"] == HOME)
        .expect("home project listed");
    assert_eq!(home["memory_count"].as_u64().unwrap(), 1);
    assert!(home["current"].as_bool().unwrap());

    let other = projects
        .iter()
        .find(|p| p["project_id"] == OTHER)
        .expect("other project listed");
    assert_eq!(other["memory_count"].as_u64().unwrap(), 2);
    assert_eq!(other["adr_count"].as_u64().unwrap(), 1);
    assert!(!other["current"].as_bool().unwrap());
}

#[test]
fn handoffs_can_be_written_and_read_across_projects() {
    let (h, _dir) = setup(Some("main"));

    call(
        &h,
        "handoff_create",
        json!({
            "branch": "release/2.0",
            "project": OTHER,
            "sections": {
                "summary": "Cut the 2.0 release branch and froze the migration schema.",
                "decisions": ["Freeze the migration schema before tagging"],
                "blockers": ["Waiting on the signed changelog"],
                "mental_model": "Release branches only take cherry-picked fixes.",
                "next_steps": ["Plan the 2.1 milestone"],
            },
        }),
    );

    let resumed = call(
        &h,
        "handoff_resume",
        json!({"project": OTHER, "branch": "release/2.0"}),
    );
    let sections = resumed["top_sections"].as_array().unwrap();
    assert!(!sections.is_empty(), "expected sections: {resumed}");

    let found = call(
        &h,
        "handoff_search",
        json!({"query": "signed changelog", "project": OTHER}),
    );
    assert!(
        found["matches"]
            .as_array()
            .map(|m| !m.is_empty())
            .unwrap_or(false),
        "expected handoff matches: {found}"
    );

    // The server's own project has no handoffs.
    let own = call(&h, "handoff_resume", json!({}));
    assert!(
        own["top_sections"]
            .as_array()
            .map(|s| s.is_empty())
            .unwrap_or(true),
        "own project should have no handoff sections: {own}"
    );
}

#[test]
fn cross_project_handoff_requires_explicit_branch() {
    let (h, _dir) = setup(Some("main"));

    let err = h
        .handle_tool(
            "handoff_create",
            json!({
                "project": OTHER,
                "sections": {
                    "summary": "No branch given for a foreign project.",
                    "decisions": [],
                    "todos": [],
                    "blockers": [],
                    "mental_model": "",
                    "next_steps": [],
                },
            }),
        )
        .expect_err("cross-project handoff without a branch must be rejected");

    match err {
        MemoryError::InvalidType(msg) => {
            assert!(msg.contains("explicit branch"), "message was: {msg}");
        }
        other => panic!("expected InvalidType, got {other:?}"),
    }
}

#[test]
fn adrs_are_numbered_per_project() {
    let (h, _dir) = setup(None);

    let home_adr = call(
        &h,
        "adr_create",
        json!({
            "title": "Use SQLite for local storage",
            "context": "Local-first tool with a single writer.",
            "decision": "Store everything in one SQLite file.",
            "consequences": "No server to operate.",
        }),
    );
    let other_adr = call(
        &h,
        "adr_create",
        json!({
            "title": "Use Postgres for the shared service",
            "context": "Concurrent writers across regions.",
            "decision": "Run managed Postgres.",
            "consequences": "Operational cost.",
            "project": OTHER,
        }),
    );

    // Numbering is per project, so both are ADR 1.
    assert_eq!(home_adr["adr_number"].as_u64().unwrap(), 1);
    assert_eq!(other_adr["adr_number"].as_u64().unwrap(), 1);

    let listed = call(&h, "adr_list", json!({"project": OTHER}));
    let items = listed.as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0]["title"].as_str().unwrap(),
        "Use Postgres for the shared service"
    );

    let shown = call(&h, "adr_show", json!({"number": 1, "project": OTHER}));
    assert_eq!(
        shown["title"].as_str().unwrap(),
        "Use Postgres for the shared service"
    );
}

#[test]
fn context_and_export_target_another_project() {
    let (h, _dir) = setup(None);

    for content in [
        "The other project uses tokio for its async runtime",
        "Rate limiting in the other project is token-bucket based",
        "The other project's CI runs on self-hosted ARM runners",
    ] {
        call(
            &h,
            "memory_store",
            json!({"content": content, "type": "fact", "project": OTHER}),
        );
    }

    let context = call(
        &h,
        "memory_context",
        json!({"context": "async runtime choices", "project": OTHER}),
    );
    assert!(context["count"].as_u64().unwrap() >= 1);

    let export = call(&h, "memory_export", json!({"project": OTHER}));
    assert_eq!(export["project_id"].as_str().unwrap(), OTHER);
    assert_eq!(export["memories"].as_array().unwrap().len(), 3);
}
