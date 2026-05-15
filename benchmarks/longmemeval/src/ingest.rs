//! INGEST PATH BYPASSES THE TOOL HANDLER.
//!
//! The MCP tool's memory_store and memory_store_batch fire auto-dedup
//! (similarity >= 0.90) and contradiction detection (>= 0.85) which would
//! silently merge our ~260 topically-similar turn-pairs per session,
//! corrupting recall numbers. We call Database::store_memory at
//! src/db/memories.rs:60 and Database::store_embedding at
//! src/db/embeddings.rs:9 directly to skip both.
//!
//! Verified bypass: confirmed by reading handler.rs lines 301-314 that
//! find_duplicates and the contradiction loop are only called inside
//! memory_store (the ToolHandler method), not in the db-tier methods we
//! invoke here.

use chrono::Utc;
use engram_mcp::{
    db::Database,
    embedding::EmbeddingService,
    memory::{Memory, MemoryType},
};

use crate::dataset::LmeQuestion;

/// Ingest all haystack turns from one question into the database.
///
/// # Project isolation
/// The caller passes `project_id = format!("lme-{}", question.question_id)`
/// so each question's haystack lives in its own isolated pool. This prevents
/// cross-question similarity from polluting recall measurements.
///
/// Returns the number of turn-pair memories ingested.
pub async fn ingest_question(
    db: &Database,
    embedding: &EmbeddingService,
    q: &LmeQuestion,
    project_id: &str,
    session_limit: usize,
) -> anyhow::Result<usize> {
    let sessions_iter = q
        .haystack_sessions
        .iter()
        .zip(q.haystack_session_ids.iter())
        .zip(q.haystack_dates.iter());

    let mut total = 0usize;

    let limited: Box<dyn Iterator<Item = _>> = if session_limit > 0 {
        Box::new(sessions_iter.take(session_limit))
    } else {
        Box::new(sessions_iter)
    };

    for ((turns, session_id), date_str) in limited {
        let created_at = match crate::dataset::parse_haystack_date(date_str) {
            Ok(dt) => dt.timestamp(),
            Err(e) => {
                tracing::warn!(
                    session_id = %session_id,
                    date = %date_str,
                    error = %e,
                    "Failed to parse haystack date, falling back to now"
                );
                Utc::now().timestamp()
            }
        };

        // Walk turns pairwise: user turn followed by assistant turn => one memory.
        let mut i = 0;
        while i < turns.len() {
            let turn = &turns[i];

            if turn.content.trim().is_empty() {
                i += 1;
                continue;
            }

            if turn.role != "user" {
                i += 1;
                continue;
            }

            let user_content = &turn.content;

            // Look ahead for an assistant follow-up.
            let (assistant_content, advance) = if i + 1 < turns.len()
                && turns[i + 1].role == "assistant"
                && !turns[i + 1].content.trim().is_empty()
            {
                (turns[i + 1].content.as_str(), 2)
            } else {
                ("<none>", 1)
            };

            let content = format!("USER: {}\nASSISTANT: {}", user_content, assistant_content);

            let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
            let memory = Memory {
                id: id.clone(),
                project_id: project_id.to_string(),
                memory_type: MemoryType::Fact,
                content: content.clone(),
                summary: None,
                tags: vec![format!("session:{}", session_id)],
                importance: 0.5,
                relevance_score: 1.0,
                access_count: 0,
                created_at,
                updated_at: created_at,
                last_accessed_at: created_at,
                branch: None,
                merged_from: None,
                external_artifacts: None,
                pinned: false,
                global: false,
            };

            // Embed as a document (document prefix matches retrieval-time query prefix asymmetry).
            let vec = embedding.embed_memory(MemoryType::Fact, &content)?;

            // Direct db-tier insert — bypasses ToolHandler dedup and contradiction checks.
            db.store_memory(&memory)?;
            db.store_embedding(&id, &vec, embedding.model_version())?;

            total += 1;
            i += advance;
        }
    }

    Ok(total)
}
