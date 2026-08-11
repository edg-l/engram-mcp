//! Integration tests for the durable todo list and its wiring into handoff resume.

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::memory::{MemoryType, TodoStatus};
use engram_mcp::tools::{TodoOp, create_handoff, list_todos, resume_handoff, write_todos};

fn setup(project: &str) -> (Database, EmbeddingService) {
    let db = Database::open_in_memory().expect("in-memory DB must open");
    db.get_or_create_project(project, project)
        .expect("project creation must succeed");
    let embedding = EmbeddingService::new().expect("embedding model must be available");
    (db, embedding)
}

fn add(text: &str, branch: Option<&str>) -> TodoOp {
    TodoOp::Add {
        text: text.to_string(),
        branch: branch.map(str::to_string),
        tags: vec![],
        importance: None,
    }
}

/// add → done → reopen, and the id stays stable across the lifecycle.
#[test]
fn todo_lifecycle_round_trip() {
    let project = "todo-lifecycle";
    let (db, embedding) = setup(project);

    let result = write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![add("Migrate the legacy subscriptions", None)],
    )
    .expect("add must succeed");
    let id = result.results[0].id.clone();
    assert!(result.results[0].error.is_none());
    assert_eq!(result.open_count, 1);

    let todo = db.get_todo(&id).unwrap().expect("todo must exist");
    assert_eq!(todo.status, TodoStatus::Open);
    assert!(todo.branch.is_none(), "no branch means project-wide");

    write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![TodoOp::Done { id: id.clone() }],
    )
    .expect("done must succeed");
    let todo = db.get_todo(&id).unwrap().unwrap();
    assert_eq!(todo.status, TodoStatus::Done);
    assert!(todo.closed_at.is_some(), "closing must stamp closed_at");
    assert!(
        db.is_dead(&id).unwrap(),
        "a finished todo must be dead so retrieval stops returning it"
    );

    write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![TodoOp::Reopen { id: id.clone() }],
    )
    .expect("reopen must succeed");
    let todo = db.get_todo(&id).unwrap().unwrap();
    assert_eq!(todo.status, TodoStatus::Open);
    assert!(todo.closed_at.is_none(), "reopening must clear closed_at");
    assert!(
        !db.is_dead(&id).unwrap(),
        "a reopened todo must be retrievable again"
    );
}

/// Dropping without a reason is refused: a todo closed silently is indistinguishable from
/// one that was forgotten, which is the failure the reason exists to prevent.
#[test]
fn drop_requires_a_reason() {
    let project = "todo-drop-reason";
    let (db, embedding) = setup(project);

    let added = write_todos(&db, &embedding, project, None, vec![add("Something", None)]).unwrap();
    let id = added.results[0].id.clone();

    let result = write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![TodoOp::Drop {
            id: id.clone(),
            reason: "   ".to_string(),
        }],
    )
    .expect("the batch itself must not fail");
    assert!(
        result.results[0].error.is_some(),
        "a blank reason must be rejected"
    );
    assert_eq!(
        db.get_todo(&id).unwrap().unwrap().status,
        TodoStatus::Open,
        "a refused drop must leave the todo open"
    );

    let result = write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![TodoOp::Drop {
            id: id.clone(),
            reason: "upstream removed the endpoint".to_string(),
        }],
    )
    .unwrap();
    assert!(result.results[0].error.is_none());
    let todo = db.get_todo(&id).unwrap().unwrap();
    assert_eq!(todo.status, TodoStatus::Dropped);
    assert_eq!(
        todo.reason.as_deref(),
        Some("upstream removed the endpoint")
    );
}

/// One bad id must not discard the valid ops beside it.
#[test]
fn a_failing_op_does_not_abort_the_batch() {
    let project = "todo-partial-batch";
    let (db, embedding) = setup(project);

    let result = write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![
            add("First real todo", None),
            TodoOp::Done {
                id: "mem_does_not_exist".to_string(),
            },
            add("Second real todo", None),
        ],
    )
    .expect("batch must return results rather than erroring");

    assert_eq!(result.results.len(), 3);
    assert!(result.results[0].error.is_none());
    assert!(result.results[1].error.is_some(), "unknown id must report");
    assert!(result.results[2].error.is_none());
    assert_eq!(result.open_count, 2, "both valid adds must have landed");
}

/// Branch scoping: a project-wide todo shows on every branch, a branch todo only on its own.
#[test]
fn branch_scoping_matches_memory_semantics() {
    let project = "todo-branches";
    let (db, embedding) = setup(project);

    write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![
            add("Project-wide: upgrade the toolchain", None),
            add("Branch work: finish the parser", Some("feat/parser")),
            add("Other branch: tune the cache", Some("feat/cache")),
        ],
    )
    .expect("adds must succeed");

    let on_parser = list_todos(
        &db,
        project,
        Some(TodoStatus::Open),
        Some(Some("feat/parser")),
        100,
    )
    .unwrap();
    let texts: Vec<&str> = on_parser.todos.iter().map(|t| t.text.as_str()).collect();
    assert!(texts.iter().any(|t| t.contains("Project-wide")));
    assert!(texts.iter().any(|t| t.contains("finish the parser")));
    assert!(
        !texts.iter().any(|t| t.contains("tune the cache")),
        "another branch's todo must not leak in, got {texts:?}"
    );

    let project_only = list_todos(&db, project, Some(TodoStatus::Open), Some(None), 100).unwrap();
    assert_eq!(project_only.todos.len(), 1);

    let everything = list_todos(&db, project, Some(TodoStatus::Open), None, 100).unwrap();
    assert_eq!(everything.todos.len(), 3);
}

/// A near-identical todo is reported, never merged: two similar todos can be separate work.
#[test]
fn adding_a_similar_todo_reports_it_without_merging() {
    let project = "todo-dupes";
    let (db, embedding) = setup(project);

    write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![add("Migrate the remaining legacy subscriptions", None)],
    )
    .unwrap();

    let result = write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![add("Migrate the remaining legacy subscriptions", None)],
    )
    .unwrap();

    assert!(
        !result.results[0].possible_duplicates.is_empty(),
        "an identical todo must be flagged as a possible duplicate"
    );
    assert_eq!(
        result.open_count, 2,
        "both must still exist — reporting is not merging"
    );
}

/// Resume reads open work from the todo list, and does so even with no handoff at all:
/// a project can have todos before it has a handoff.
#[test]
fn resume_reads_open_todos_with_no_handoff() {
    let project = "todo-resume-empty";
    let (db, embedding) = setup(project);

    write_todos(
        &db,
        &embedding,
        project,
        Some("feat/x"),
        vec![add("Wire up the retry budget", None)],
    )
    .unwrap();

    let result = resume_handoff(
        &db,
        &embedding,
        project,
        Some("feat/x"),
        Some("anything"),
        5,
        false,
        None,
    )
    .expect("resume must succeed with no handoffs");

    assert!(result.latest_handoff_id.is_none(), "no handoff exists");
    assert_eq!(
        result.open_todos,
        vec!["Wire up the retry budget".to_string()],
        "open todos must surface without a handoff to hang them on"
    );
}

/// A finished todo drops out of resume; a still-open one survives regardless of ranking.
#[test]
fn resume_reflects_todo_state_not_handoff_snapshots() {
    let project = "todo-resume-state";
    let (db, embedding) = setup(project);

    let added = write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![
            add("Still open: migrate subscriptions", None),
            add("Will finish: rename the module", None),
        ],
    )
    .unwrap();
    let finish_id = added.results[1].id.clone();

    let sections = engram_mcp::memory::HandoffSections {
        summary: "Reworked the dispatcher".to_string(),
        decisions: vec![],
        todos: vec![],
        blockers: vec!["Waiting on prod credentials".to_string()],
        tried: vec![],
        mental_model: String::new(),
        next_steps: vec![],
        notes: None,
        continues_from: None,
    };
    create_handoff(
        &db,
        &embedding,
        project,
        Some("feat/dispatch"),
        sections,
        0.85,
        true,
        false,
    )
    .expect("handoff must be created");

    write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![TodoOp::Done { id: finish_id }],
    )
    .unwrap();

    // One section slot, so the ranking cannot carry open work even if it wanted to.
    let result = resume_handoff(
        &db,
        &embedding,
        project,
        Some("feat/dispatch"),
        Some("dispatcher rework"),
        1,
        false,
        None,
    )
    .unwrap();

    assert_eq!(
        result.open_todos,
        vec!["Still open: migrate subscriptions".to_string()],
        "resume must show live open todos only"
    );
    assert_eq!(
        result.open_blockers,
        vec!["Waiting on prod credentials".to_string()],
        "blockers still come from the handoff snapshot"
    );
}

/// A closed todo must vanish from ordinary retrieval, via the same `dead` mechanism
/// curation already uses — no todo-specific filtering in the query path.
#[test]
fn finished_todos_leave_retrieval() {
    let project = "todo-retrieval";
    let (db, embedding) = setup(project);

    let added = write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![add("Investigate the flaky connection pool test", None)],
    )
    .unwrap();
    let id = added.results[0].id.clone();

    let live = db
        .query_memories(project, Some(&[MemoryType::Todo]), None, None, 50)
        .unwrap();
    assert_eq!(live.len(), 1);

    write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![TodoOp::Done { id: id.clone() }],
    )
    .unwrap();

    assert!(
        db.get_dead_ids(project).unwrap().contains(&id),
        "a finished todo must be in the dead set that curation filters on"
    );
}

/// A different task that merely shares a subject must not be reported as a duplicate.
/// This is the false positive that matters: a field that flags everything is one the
/// caller stops reading, and this model puts unrelated short todos as high as 0.93.
#[test]
fn a_different_task_on_the_same_subject_is_not_a_duplicate() {
    let project = "todo-dupe-precision";
    let (db, embedding) = setup(project);

    write_todos(
        &db,
        &embedding,
        project,
        None,
        vec![add("Migrate the remaining 40 legacy subscriptions", None)],
    )
    .unwrap();

    for text in [
        "Delete the legacy subscription table",
        "Retroactively tag v0.8.0 and v0.8.1",
        "Profile the todo list query",
    ] {
        let result = write_todos(&db, &embedding, project, None, vec![add(text, None)]).unwrap();
        assert!(
            result.results[0].possible_duplicates.is_empty(),
            "{text:?} must not be reported as a duplicate, got {:?}",
            result.results[0].possible_duplicates
        );
    }
}
