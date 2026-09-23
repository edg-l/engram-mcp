use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::db::{Database, StoreDayIndex};
use crate::embedding::{EmbeddingService, cosine_similarity};
use crate::error::MemoryError;
use crate::memory::{Memory, MemoryType, RelationType, Relationship, TodoItem, TodoStatus};
use crate::tools::schemas::dedup_threshold;
use crate::tools::store::{StoreOutcome, store_with_dedup_exempting};

/// Similarity at or above which an existing open todo is reported as a possible duplicate
/// of one being added. Never merged automatically: two todos that read alike can still be
/// separate work, and collapsing them would silently drop a task.
///
/// Derived from measured pairs on this embedding model rather than picked by feel. Short
/// texts sit in a narrow, high band: against "Migrate the remaining 40 legacy
/// subscriptions", an identical string scores 1.00, a reworded version of the same work
/// 0.98, a *different* task on the same subject ("Delete the legacy subscription table")
/// 0.95, unrelated todos 0.86-0.93, and an unrelated non-todo sentence 0.84. The threshold
/// therefore has to clear 0.95, not the ~0.85 that would look generous elsewhere in this
/// codebase: reporting a same-topic-different-work pair is a false positive, and a field
/// that cries duplicate on everything is one the caller learns to skip.
pub const TODO_DUPLICATE_MIN: f32 = 0.97;

/// Cap on reported duplicates per added todo.
const TODO_DUPLICATE_MAX: usize = 3;

/// Default importance for a new todo. Mid-scale: a todo earns attention by being open,
/// not by being scored highly.
pub const TODO_DEFAULT_IMPORTANCE: f64 = 0.6;

/// Cap, in characters, on the text written by an `add` or `edit` op. A todo is a one-line
/// title; findings, measurements, and dead ends belong in `detail`, which has no cap.
/// Only text an op writes is checked; a stored todo over the cap still reads and closes.
pub const TODO_TEXT_MAX: usize = 200;

/// Default importance for a todo's linked detail memory: an ordinary fact, not elevated
/// just because the todo it hangs off is pinned.
const TODO_DETAIL_IMPORTANCE: f64 = 0.5;

/// One requested change to the todo list.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum TodoOp {
    /// Open a new todo.
    Add {
        text: String,
        /// Branch this todo belongs to. Omit for a project-wide todo; `"auto"` resolves to
        /// the caller's current branch.
        #[serde(default)]
        branch: Option<String>,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        importance: Option<f64>,
        /// A finding, measurement, or dead end too long for the title. Stored as a linked
        /// `fact` memory rather than appended to the todo's own text.
        #[serde(default)]
        detail: Option<String>,
    },
    /// Mark a todo finished.
    Done { id: String },
    /// Close a todo without doing it. The reason is mandatory.
    Drop { id: String, reason: String },
    /// Return a closed todo to the open state.
    Reopen { id: String },
    /// Rewrite a todo's text.
    Edit {
        id: String,
        text: String,
        /// A new finding to link, on top of any recorded by earlier edits. Details
        /// accumulate; the todo itself stays a title.
        #[serde(default)]
        detail: Option<String>,
    },
}

/// Outcome of a single [`TodoOp`].
#[derive(Debug, Clone, Serialize)]
pub struct TodoOpResult {
    /// The operation that was applied, e.g. `"add"`.
    pub op: String,
    pub id: String,
    /// Present on failure; the op is reported rather than aborting the whole batch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Existing open todos that look like the one just added.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub possible_duplicates: Vec<TodoDuplicate>,
    /// Id of the linked fact memory created from `detail`, if one was given.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail_id: Option<String>,
}

/// An existing open todo similar to one being added.
#[derive(Debug, Clone, Serialize)]
pub struct TodoDuplicate {
    pub id: String,
    pub text: String,
    pub similarity: f32,
}

/// A todo reduced to what a compact renderer needs: enough to display and close it without
/// a second lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenTodoItem {
    pub id: String,
    pub title: String,
}

/// Cap, in characters, on a derived todo title. Long enough to keep the lead sentence of a
/// typical todo intact, short enough that a hundred of them stay skimmable in one list.
const TODO_TITLE_MAX_CHARS: usize = 160;

/// Derive a short display title from a todo's full text.
///
/// The single title source for every compact todo renderer (`handoff_resume`'s
/// `open_todos`, `todo_list`'s compact mode, `engram-cli todo list`'s default), so they
/// cannot disagree about what a todo is called. Takes the first line, cut at the first
/// sentence end ("`. `") when that falls before the char cap — a natural stopping point
/// rather than a mid-sentence fragment — otherwise hard-truncated at the cap with a
/// trailing "…".
pub fn todo_title(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or("").trim();
    let chars: Vec<char> = first_line.chars().collect();

    let sentence_end = first_line.find(". ").and_then(|byte_idx| {
        let char_idx = first_line[..byte_idx].chars().count();
        (char_idx < TODO_TITLE_MAX_CHARS).then_some(char_idx)
    });

    if let Some(idx) = sentence_end {
        return chars[..=idx].iter().collect();
    }

    if chars.len() > TODO_TITLE_MAX_CHARS {
        let truncated: String = chars[..TODO_TITLE_MAX_CHARS].iter().collect();
        format!("{truncated}…")
    } else {
        first_line.to_string()
    }
}

/// Result returned by `todo_write`.
#[derive(Debug, Clone, Serialize)]
pub struct TodoWriteResult {
    pub project: String,
    pub results: Vec<TodoOpResult>,
    /// Open todo count for the project after applying the batch.
    pub open_count: usize,
}

/// Result returned by `todo_list`.
#[derive(Debug, Clone, Serialize)]
pub struct TodoListResult {
    pub project: String,
    pub count: usize,
    pub todos: Vec<TodoItem>,
    /// Counts across every lifecycle state, so a caller filtering to `open` still learns
    /// that closed work exists.
    pub open_count: usize,
    pub done_count: usize,
    pub dropped_count: usize,
    /// Open todos, within this listing's branch filter, idle for `STALE_TODO_STORE_DAYS`+
    /// active store-days. Reported regardless of the request's own `status` filter — a
    /// caller listing `done` todos still learns whether the open list needs attention.
    pub stale_todo_count: usize,
    /// Echoes the request's `full_text` so the compact text renderer, which only sees this
    /// JSON, knows whether to print each todo's full text or its derived title. `todos`
    /// itself always carries the full `TodoItem`s regardless.
    pub full_text: bool,
}

/// Open todos for a branch, as `{id, title}` items ordered by importance descending, then
/// by most recently updated — the items most worth a resuming agent's attention first.
///
/// This is the single source of open work for `handoff_resume`. `branch` follows the
/// "branch plus project-wide" shape: a todo with no branch applies everywhere.
pub fn open_todo_titles(
    db: &Database,
    project_id: &str,
    branch: Option<&str>,
    limit: usize,
) -> Result<Vec<OpenTodoItem>, MemoryError> {
    let filter = Some(branch);
    // The DB orders by creation time, so the importance ordering needs every open todo
    // before `limit` applies; capping first would drop an old, important todo unseen.
    let total = db.count_open_todos(project_id, filter)?;
    let mut todos = db.list_todos(project_id, Some(TodoStatus::Open), filter, total)?;
    todos.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.updated_at.cmp(&a.updated_at))
    });
    Ok(todos
        .into_iter()
        .take(limit)
        .map(|t| OpenTodoItem {
            title: todo_title(&t.text),
            id: t.id,
        })
        .collect())
}

/// Store-days an open todo can go untouched before `handoff_resume` and `todo_list` flag
/// it as stale. Store days, not calendar days (`StoreDayIndex`): a project nobody has
/// worked on has a frozen clock, so its todos never go stale for merely sitting there
/// while the project itself is dormant. 30 is about a month of days the project was
/// actually active.
pub const STALE_TODO_STORE_DAYS: f64 = 30.0;

/// Cap on stale ids reported at once — enough to name concretely without dumping the
/// whole idle set.
const STALE_TODO_ID_LIMIT: usize = 5;

/// Open todos idle long enough to flag, and the total count.
pub struct StaleTodos {
    pub count: usize,
    /// Oldest-updated first, capped at [`STALE_TODO_ID_LIMIT`].
    pub ids: Vec<String>,
}

/// Scan every open todo matching `branch` (the three-way filter `Database::list_todos`
/// takes) for staleness against `store_days`.
///
/// Separate from `open_todo_titles`: staleness must see the full open set regardless of
/// any rendering cap, and needs `updated_at`, which the display-only `OpenTodoItem`
/// doesn't carry.
pub fn stale_todos(
    db: &Database,
    project_id: &str,
    branch: Option<Option<&str>>,
    store_days: &StoreDayIndex,
) -> Result<StaleTodos, MemoryError> {
    let total = db.count_open_todos(project_id, branch)?;
    let mut todos = db.list_todos(project_id, Some(TodoStatus::Open), branch, total)?;
    todos.retain(|t| store_days.active_days_since(t.updated_at) >= STALE_TODO_STORE_DAYS);
    todos.sort_by_key(|t| t.updated_at);
    let count = todos.len();
    let ids = todos
        .into_iter()
        .take(STALE_TODO_ID_LIMIT)
        .map(|t| t.id)
        .collect();
    Ok(StaleTodos { count, ids })
}

/// Existing open todos similar to `text`, most similar first.
fn find_similar_open(
    db: &Database,
    project_id: &str,
    new_vec: &[f32],
) -> Result<Vec<TodoDuplicate>, MemoryError> {
    let open = db.list_todos(project_id, Some(TodoStatus::Open), None, 500)?;
    if open.is_empty() {
        return Ok(Vec::new());
    }

    let mut scored: Vec<TodoDuplicate> = Vec::new();
    for todo in open {
        let Some(vec) = db.get_embedding(&todo.id)? else {
            continue;
        };
        let similarity = cosine_similarity(new_vec, &vec);
        if similarity >= TODO_DUPLICATE_MIN {
            scored.push(TodoDuplicate {
                id: todo.id,
                text: todo.text,
                similarity,
            });
        }
    }
    scored.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(TODO_DUPLICATE_MAX);
    Ok(scored)
}

/// Apply a batch of todo operations.
///
/// A failing op is recorded in its own result rather than aborting the batch: a bad id in
/// one entry must not discard the four valid closes next to it.
pub fn write_todos(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
    current_branch: Option<&str>,
    ops: Vec<TodoOp>,
) -> Result<TodoWriteResult, MemoryError> {
    let mut results = Vec::with_capacity(ops.len());

    for op in ops {
        let result = apply_op(db, embedding, project_id, current_branch, op);
        results.push(result);
    }

    let (open_count, _, _) = db.todo_counts(project_id)?;
    Ok(TodoWriteResult {
        project: project_id.to_string(),
        results,
        open_count,
    })
}

fn apply_op(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
    current_branch: Option<&str>,
    op: TodoOp,
) -> TodoOpResult {
    match op {
        TodoOp::Add {
            text,
            branch,
            tags,
            importance,
            detail,
        } => {
            let name = "add";
            match add_todo(
                db,
                embedding,
                project_id,
                current_branch,
                &text,
                branch,
                tags,
                importance,
                detail,
            ) {
                Ok((id, dups, detail_id)) => TodoOpResult {
                    op: name.to_string(),
                    id,
                    error: None,
                    possible_duplicates: dups,
                    detail_id,
                },
                Err(e) => fail(name, String::new(), e),
            }
        }
        TodoOp::Done { id } => transition(db, "done", id, TodoStatus::Done, None),
        TodoOp::Drop { id, reason } => {
            if reason.trim().is_empty() {
                return fail(
                    "drop",
                    id,
                    MemoryError::InvalidType(
                        "drop requires a non-empty reason; a todo closed without one is \
                         indistinguishable from one that was forgotten"
                            .to_string(),
                    ),
                );
            }
            transition(db, "drop", id, TodoStatus::Dropped, Some(reason))
        }
        TodoOp::Reopen { id } => transition(db, "reopen", id, TodoStatus::Open, None),
        TodoOp::Edit { id, text, detail } => {
            let name = "edit";
            match edit_todo(db, embedding, project_id, &id, &text, detail) {
                Ok(detail_id) => TodoOpResult {
                    op: name.to_string(),
                    id,
                    error: None,
                    possible_duplicates: Vec::new(),
                    detail_id,
                },
                Err(e) => fail(name, id, e),
            }
        }
    }
}

fn fail(op: &str, id: String, e: MemoryError) -> TodoOpResult {
    TodoOpResult {
        op: op.to_string(),
        id,
        error: Some(e.to_string()),
        possible_duplicates: Vec::new(),
        detail_id: None,
    }
}

fn transition(
    db: &Database,
    op: &str,
    id: String,
    status: TodoStatus,
    reason: Option<String>,
) -> TodoOpResult {
    match db.get_todo(&id) {
        Ok(Some(_)) => match db.set_todo_status(&id, status, reason.as_deref()) {
            Ok(()) => TodoOpResult {
                op: op.to_string(),
                id,
                error: None,
                possible_duplicates: Vec::new(),
                detail_id: None,
            },
            Err(e) => fail(op, id, e),
        },
        Ok(None) => fail(
            op,
            id.clone(),
            MemoryError::InvalidType(format!("{id} is not a todo")),
        ),
        Err(e) => fail(op, id, e),
    }
}

/// Reject text over [`TODO_TEXT_MAX`] chars, naming `detail` as where the rest belongs.
fn check_text_len(text: &str) -> Result<(), MemoryError> {
    let len = text.chars().count();
    if len > TODO_TEXT_MAX {
        return Err(MemoryError::InvalidType(format!(
            "todo text is {len} chars, over the {TODO_TEXT_MAX}-char cap; keep it a \
             one-line title and put findings, measurements, or dead ends in `detail`"
        )));
    }
    Ok(())
}

/// Store `detail` as a `fact` memory linked to `todo_id` via a `relates_to` edge, through
/// the normal dedup path so a repeated finding merges instead of piling up duplicates.
/// Returns the id that survived — the new memory, or the existing one it merged into.
fn store_todo_detail(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
    todo_id: &str,
    branch: Option<&str>,
    tags: &[String],
    detail: &str,
) -> Result<String, MemoryError> {
    let detail = detail.trim();
    let now = chrono::Utc::now().timestamp();
    let vector = embedding.embed_memory(MemoryType::Fact, detail)?;
    let memory = Memory {
        id: format!("mem_{}", uuid::Uuid::new_v4().simple()),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Fact,
        content: detail.to_string(),
        summary: None,
        tags: tags.to_vec(),
        importance: TODO_DETAIL_IMPORTANCE,
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: branch.map(str::to_string),
        merged_from: None,
        external_artifacts: None,
        pinned: false,
        global: false,
    };

    // A todo is a different memory_type, so type-matched dedup would never fold into it
    // anyway; exempting it here just applies the same "caller declared this connected"
    // rule store_with_dedup_exempting already applies to related_to/supersedes targets.
    let exempt: HashSet<String> = [todo_id.to_string()].into_iter().collect();
    let outcome = store_with_dedup_exempting(
        db,
        Some(embedding),
        project_id,
        memory,
        Some(&vector),
        dedup_threshold(),
        None, // never skip: a detail is always worth recording, merged or not
        &exempt,
    )?;

    let detail_id = match outcome {
        StoreOutcome::Stored(id) => id,
        StoreOutcome::Merged { id, .. } => id,
        StoreOutcome::SkippedSimilar { .. } => {
            unreachable!("skip_above=None; SkippedSimilar cannot occur")
        }
    };

    db.create_relationship(&Relationship {
        id: format!("rel_{}", uuid::Uuid::new_v4().simple()),
        source_id: todo_id.to_string(),
        target_id: detail_id.clone(),
        relation_type: RelationType::RelatesTo,
        strength: 1.0,
        created_at: now,
    })?;

    Ok(detail_id)
}

#[allow(clippy::too_many_arguments)]
fn add_todo(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
    current_branch: Option<&str>,
    text: &str,
    branch: Option<String>,
    tags: Vec<String>,
    importance: Option<f64>,
    detail: Option<String>,
) -> Result<(String, Vec<TodoDuplicate>, Option<String>), MemoryError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(MemoryError::InvalidType(
            "todo text must not be empty".to_string(),
        ));
    }
    check_text_len(text)?;

    // `None` means the todo applies to the whole project; "auto" opts into branch scoping.
    let resolved_branch = match branch.as_deref() {
        Some("auto") => current_branch.map(str::to_string),
        Some(b) if !b.is_empty() => Some(b.to_string()),
        _ => None,
    };

    let vector = embedding.embed_memory(MemoryType::Todo, text)?;
    let duplicates = find_similar_open(db, project_id, &vector)?;

    let now = chrono::Utc::now().timestamp();
    let memory = Memory {
        id: format!("mem_{}", uuid::Uuid::new_v4().simple()),
        project_id: project_id.to_string(),
        memory_type: MemoryType::Todo,
        content: text.to_string(),
        summary: None,
        tags: tags.clone(),
        importance: importance
            .unwrap_or(TODO_DEFAULT_IMPORTANCE)
            .clamp(0.0, 1.0),
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: resolved_branch.clone(),
        merged_from: None,
        external_artifacts: None,
        // Pinned so decay and prune leave it alone: an old todo is not less true, it is
        // more overdue.
        pinned: true,
        global: false,
    };

    db.store_todo_atomic(&memory, &vector, embedding.model_version())?;

    let detail_id = match detail {
        Some(d) if !d.trim().is_empty() => Some(store_todo_detail(
            db,
            embedding,
            project_id,
            &memory.id,
            resolved_branch.as_deref(),
            &tags,
            &d,
        )?),
        _ => None,
    };

    Ok((memory.id, duplicates, detail_id))
}

fn edit_todo(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
    id: &str,
    text: &str,
    detail: Option<String>,
) -> Result<Option<String>, MemoryError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(MemoryError::InvalidType(
            "todo text must not be empty".to_string(),
        ));
    }
    check_text_len(text)?;
    let Some(todo) = db.get_todo(id)? else {
        return Err(MemoryError::InvalidType(format!("{id} is not a todo")));
    };
    db.update_todo_text(id, text)?;
    let vector = embedding.embed_memory(MemoryType::Todo, text)?;
    db.store_embedding(id, &vector, embedding.model_version())?;

    match detail {
        Some(d) if !d.trim().is_empty() => Ok(Some(store_todo_detail(
            db,
            embedding,
            project_id,
            id,
            todo.branch.as_deref(),
            &todo.tags,
            &d,
        )?)),
        _ => Ok(None),
    }
}

/// List todos for a project.
///
/// `full_text` only affects how a compact text renderer displays `todos`; the returned
/// `TodoItem`s themselves always carry their full text.
pub fn list_todos(
    db: &Database,
    project_id: &str,
    status: Option<TodoStatus>,
    branch: Option<Option<&str>>,
    limit: usize,
    full_text: bool,
) -> Result<TodoListResult, MemoryError> {
    let todos = db.list_todos(project_id, status, branch, limit)?;
    let (open_count, done_count, dropped_count) = db.todo_counts(project_id)?;
    // Scanned over the open set for this branch filter regardless of `status`, so a
    // caller listing `done` todos still learns whether the open list needs attention.
    let store_days = db.get_store_day_index(project_id)?;
    let stale = stale_todos(db, project_id, branch, &store_days)?;
    Ok(TodoListResult {
        project: project_id.to_string(),
        count: todos.len(),
        todos,
        open_count,
        done_count,
        dropped_count,
        stale_todo_count: stale.count,
        full_text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_is_the_first_line() {
        assert_eq!(
            todo_title("Migrate the legacy subscriptions\nDetails below."),
            "Migrate the legacy subscriptions"
        );
    }

    #[test]
    fn title_cuts_at_the_first_sentence_end() {
        assert_eq!(
            todo_title("Migrate the subscriptions. Also check the billing job for stragglers."),
            "Migrate the subscriptions."
        );
    }

    #[test]
    fn title_keeps_a_short_line_with_no_sentence_end_intact() {
        assert_eq!(todo_title("Fix the flaky test"), "Fix the flaky test");
    }

    #[test]
    fn title_hard_truncates_past_the_cap_with_an_ellipsis() {
        let long = "x".repeat(200);
        let title = todo_title(&long);
        assert_eq!(title.chars().count(), TODO_TITLE_MAX_CHARS + 1);
        assert!(title.ends_with('…'));
    }

    /// A sentence end past the cap does not count as "earlier": the line still gets a hard
    /// truncation with an ellipsis rather than a title longer than the cap.
    #[test]
    fn title_ignores_a_sentence_end_beyond_the_cap() {
        let long = format!("{}. tail", "x".repeat(200));
        let title = todo_title(&long);
        assert_eq!(title.chars().count(), TODO_TITLE_MAX_CHARS + 1);
        assert!(title.ends_with('…'));
    }

    fn insert_todo(db: &Database, project: &str, id: &str, updated_at: i64) {
        let memory = Memory {
            id: id.to_string(),
            project_id: project.to_string(),
            memory_type: MemoryType::Todo,
            content: "Some todo".to_string(),
            summary: None,
            tags: vec![],
            importance: TODO_DEFAULT_IMPORTANCE,
            relevance_score: 1.0,
            access_count: 0,
            created_at: updated_at,
            updated_at,
            last_accessed_at: updated_at,
            branch: None,
            merged_from: None,
            external_artifacts: None,
            pinned: true,
            global: false,
        };
        db.store_todo_atomic(&memory, &[0.0; 4], "test-model")
            .expect("todo must store");
    }

    /// Only the todo that store-days have actually displaced counts as stale; one that
    /// still sits inside the window does not, regardless of how old it looks in isolation.
    #[test]
    fn stale_todos_flags_only_what_store_days_have_displaced() {
        use crate::db::SECONDS_PER_DAY;

        let db = Database::open_in_memory().expect("in-memory db must open");
        let project = "todo-stale-unit";
        db.get_or_create_project(project, project).unwrap();

        insert_todo(&db, project, "mem_stale", 0);
        insert_todo(&db, project, "mem_fresh", 89 * SECONDS_PER_DAY);

        // Store-days 1..=90: 90 days after mem_stale, only 1 after mem_fresh.
        let store_days = StoreDayIndex::from_days((1..=90).collect());

        let result = stale_todos(&db, project, Some(None), &store_days).unwrap();
        assert_eq!(result.count, 1);
        assert_eq!(result.ids, vec!["mem_stale".to_string()]);
    }
}
