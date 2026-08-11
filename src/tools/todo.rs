use serde::{Deserialize, Serialize};

use crate::db::Database;
use crate::embedding::{EmbeddingService, cosine_similarity};
use crate::error::MemoryError;
use crate::memory::{Memory, MemoryType, TodoItem, TodoStatus};

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
    },
    /// Mark a todo finished.
    Done { id: String },
    /// Close a todo without doing it. The reason is mandatory.
    Drop { id: String, reason: String },
    /// Return a closed todo to the open state.
    Reopen { id: String },
    /// Rewrite a todo's text.
    Edit { id: String, text: String },
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
}

/// An existing open todo similar to one being added.
#[derive(Debug, Clone, Serialize)]
pub struct TodoDuplicate {
    pub id: String,
    pub text: String,
    pub similarity: f32,
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
}

/// Open todos for a branch, as plain text, newest first.
///
/// This is the single source of open work for `handoff_resume`. `branch` follows the
/// "branch plus project-wide" shape: a todo with no branch applies everywhere.
pub fn open_todo_texts(
    db: &Database,
    project_id: &str,
    branch: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, MemoryError> {
    let filter = Some(branch);
    let todos = db.list_todos(project_id, Some(TodoStatus::Open), filter, limit)?;
    Ok(todos.into_iter().map(|t| t.text).collect())
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
            ) {
                Ok((id, dups)) => TodoOpResult {
                    op: name.to_string(),
                    id,
                    error: None,
                    possible_duplicates: dups,
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
        TodoOp::Edit { id, text } => {
            let name = "edit";
            match edit_todo(db, embedding, &id, &text) {
                Ok(()) => TodoOpResult {
                    op: name.to_string(),
                    id,
                    error: None,
                    possible_duplicates: Vec::new(),
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
) -> Result<(String, Vec<TodoDuplicate>), MemoryError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(MemoryError::InvalidType(
            "todo text must not be empty".to_string(),
        ));
    }

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
        tags,
        importance: importance
            .unwrap_or(TODO_DEFAULT_IMPORTANCE)
            .clamp(0.0, 1.0),
        relevance_score: 1.0,
        access_count: 0,
        created_at: now,
        updated_at: now,
        last_accessed_at: now,
        branch: resolved_branch,
        merged_from: None,
        external_artifacts: None,
        // Pinned so decay and prune leave it alone: an old todo is not less true, it is
        // more overdue.
        pinned: true,
        global: false,
    };

    db.store_todo_atomic(&memory, &vector, embedding.model_version())?;
    Ok((memory.id, duplicates))
}

fn edit_todo(
    db: &Database,
    embedding: &EmbeddingService,
    id: &str,
    text: &str,
) -> Result<(), MemoryError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(MemoryError::InvalidType(
            "todo text must not be empty".to_string(),
        ));
    }
    if db.get_todo(id)?.is_none() {
        return Err(MemoryError::InvalidType(format!("{id} is not a todo")));
    }
    db.update_todo_text(id, text)?;
    let vector = embedding.embed_memory(MemoryType::Todo, text)?;
    db.store_embedding(id, &vector, embedding.model_version())?;
    Ok(())
}

/// List todos for a project.
pub fn list_todos(
    db: &Database,
    project_id: &str,
    status: Option<TodoStatus>,
    branch: Option<Option<&str>>,
    limit: usize,
) -> Result<TodoListResult, MemoryError> {
    let todos = db.list_todos(project_id, status, branch, limit)?;
    let (open_count, done_count, dropped_count) = db.todo_counts(project_id)?;
    Ok(TodoListResult {
        project: project_id.to_string(),
        count: todos.len(),
        todos,
        open_count,
        done_count,
        dropped_count,
    })
}
