//! Compact text formatting for MCP tool results.
//!
//! Converts JSON tool results into compact text for token-efficient LLM consumption.

use serde_json::Value;

/// Truncate a string to a maximum length, adding "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max_len).collect::<String>())
    }
}

/// Format a tool result as compact text for LLM consumption.
/// Optimized for readability and token efficiency.
pub fn compact_tool_result(tool_name: &str, result: &Value, content_length: usize) -> String {
    match tool_name {
        "memory_store" => compact_store(result),
        "memory_query" => compact_query(result, content_length),
        "memory_context" => compact_context(result, content_length),
        "memory_graph" => compact_graph(result),
        "memory_store_batch" => compact_batch_store(result),
        "memory_prune" => compact_prune(result),
        "memory_promote" => compact_promote(result),
        "memory_dedup" => compact_dedup(result),
        "memory_stats" => compact_stats(result),
        "memory_update" | "memory_delete" | "memory_delete_batch" => compact_simple(result),
        _ => compact_fallback(result),
    }
}

fn compact_store(result: &Value) -> String {
    let id = result.get("id").and_then(|v| v.as_str()).unwrap_or("?");
    let mut out = format!("Stored {}", id);

    if let Some(branch) = result.get("branch").and_then(|v| v.as_str()) {
        out.push_str(&format!(" (branch: {})", branch));
    }

    if let Some(merge) = result.get("merge_info")
        && !merge.is_null()
    {
        let merged_with = merge
            .get("merged_with")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let sim = merge
            .get("similarity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        out.push_str(&format!(
            "\nMerged with duplicate {} (similarity: {:.2})",
            merged_with, sim
        ));
    }

    let contradictions = result
        .get("potential_contradictions")
        .and_then(|v| v.as_array());
    if let Some(arr) = contradictions
        && !arr.is_empty()
    {
        out.push_str(&format!(
            "\nWarning: {} potential contradiction(s):",
            arr.len()
        ));
        for c in arr {
            let cid = c.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
            let summary = c.get("summary").and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!("\n  {} - {}", cid, truncate_str(summary, 80)));
        }
    }

    out
}

fn compact_query(result: &Value, content_length: usize) -> String {
    let memories = result.get("memories").and_then(|v| v.as_array());
    let Some(arr) = memories else {
        return "No results.".to_string();
    };
    if arr.is_empty() {
        return "No results.".to_string();
    }

    let mut out = format!("{} result(s):\n", arr.len());

    for mem in arr {
        let memory = mem.get("memory").unwrap_or(mem);
        let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let mem_type = memory
            .get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let content = memory.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let score = mem.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let tags = memory.get("tags").and_then(|v| v.as_array());
        let importance = memory
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        out.push_str(&format!("\n[{}] {} ({}", id, mem_type, format_score(score)));
        if importance >= 0.7 {
            out.push_str(&format!(", importance: {:.1}", importance));
        }
        out.push(')');
        if let Some(tags) = tags
            && !tags.is_empty()
        {
            let tag_strs: Vec<&str> = tags.iter().filter_map(|t| t.as_str()).collect();
            if !tag_strs.is_empty() {
                out.push_str(&format!(" [{}]", tag_strs.join(", ")));
            }
        }
        out.push('\n');
        out.push_str(&truncate_str(content, content_length));
        out.push('\n');
    }

    // Contradiction warnings
    if let Some(warnings) = result
        .get("contradiction_warnings")
        .and_then(|v| v.as_array())
        && !warnings.is_empty()
    {
        out.push_str("\nContradictions detected:");
        for w in warnings {
            let a = w.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
            let b = w
                .get("contradicts_id")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            out.push_str(&format!("\n  {} contradicts {}", a, b));
        }
    }

    out
}

fn compact_context(result: &Value, content_length: usize) -> String {
    let memories = result.get("memories").and_then(|v| v.as_array());
    let Some(arr) = memories else {
        return "No relevant memories.".to_string();
    };
    if arr.is_empty() {
        return "No relevant memories.".to_string();
    }

    let mode = result
        .get("retrieval_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("flat");
    let mut out = format!(
        "{} relevant memory/memories ({} retrieval):\n",
        arr.len(),
        mode
    );

    for mem in arr {
        let id = mem.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let mem_type = mem.get("type").and_then(|v| v.as_str()).unwrap_or("?");
        let content = mem.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let sim = mem
            .get("similarity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let tags = mem.get("tags").and_then(|v| v.as_array());

        out.push_str(&format!("\n[{}] {} ({})", id, mem_type, format_score(sim)));
        if let Some(tags) = tags
            && !tags.is_empty()
        {
            let tag_strs: Vec<&str> = tags.iter().filter_map(|t| t.as_str()).collect();
            if !tag_strs.is_empty() {
                out.push_str(&format!(" [{}]", tag_strs.join(", ")));
            }
        }
        out.push('\n');
        out.push_str(&truncate_str(content, content_length));
        out.push('\n');
    }

    out
}

fn compact_graph(result: &Value) -> String {
    let root = result.get("root");
    let root_id = root
        .and_then(|r| r.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let root_type = root
        .and_then(|r| r.get("memory_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let root_content = root
        .and_then(|r| r.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut out = format!(
        "Root: [{}] {} - {}\n",
        root_id,
        root_type,
        truncate_str(root_content, 100)
    );

    if let Some(related) = result.get("related").and_then(|v| v.as_array()) {
        if related.is_empty() {
            out.push_str("No related memories.");
        } else {
            out.push_str(&format!("\n{} related:", related.len()));
            for rel in related {
                let memory = rel.get("memory").unwrap_or(rel);
                let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let mem_type = memory
                    .get("memory_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let content = memory.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let relation = rel.get("relation").and_then(|v| v.as_str()).unwrap_or("?");
                let direction = rel.get("direction").and_then(|v| v.as_str()).unwrap_or("?");
                let depth = rel.get("depth").and_then(|v| v.as_u64()).unwrap_or(0);

                let indent = "  ".repeat(depth as usize);
                let arrow = if direction == "outgoing" { "->" } else { "<-" };
                out.push_str(&format!(
                    "\n{}{} {} [{}] {} - {}",
                    indent,
                    arrow,
                    relation,
                    id,
                    mem_type,
                    truncate_str(content, 80)
                ));
            }
        }
    }

    out
}

fn compact_batch_store(result: &Value) -> String {
    let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
    let ids = result.get("ids").and_then(|v| v.as_array());
    let mut out = format!("Stored {} memories", count);
    if let Some(ids) = ids {
        let id_strs: Vec<&str> = ids.iter().filter_map(|v| v.as_str()).collect();
        if !id_strs.is_empty() {
            out.push_str(&format!(": {}", id_strs.join(", ")));
        }
    }
    out
}

fn compact_prune(result: &Value) -> String {
    let dry_run = result
        .get("dry_run")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let candidates = result
        .get("candidates")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let deleted = result.get("deleted").and_then(|v| v.as_u64()).unwrap_or(0);
    let threshold = result
        .get("threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.2);

    if dry_run {
        format!(
            "Prune dry run: {} memories below {:.2} threshold. Set confirm=true to delete.",
            candidates, threshold
        )
    } else {
        format!(
            "Pruned {} memories below {:.2} threshold.",
            deleted, threshold
        )
    }
}

fn compact_promote(result: &Value) -> String {
    let success = result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let message = result.get("message").and_then(|v| v.as_str()).unwrap_or("");
    if success {
        message.to_string()
    } else {
        format!("Failed: {}", message)
    }
}

fn compact_dedup(result: &Value) -> String {
    let dry_run = result
        .get("dry_run")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let groups = result
        .get("duplicate_groups")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total = result
        .get("total_duplicates")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let merged = result.get("merged").and_then(|v| v.as_u64()).unwrap_or(0);

    if dry_run {
        format!(
            "Dedup dry run: {} duplicate groups ({} total). Set confirm=true to merge.",
            groups, total
        )
    } else {
        format!("Dedup complete: merged {} duplicate memories.", merged)
    }
}

fn compact_stats(result: &Value) -> String {
    let count = result
        .get("memory_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let rels = result
        .get("relationship_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let rel = result
        .get("avg_relevance")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let clusters = result
        .get("cluster_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let project = result
        .get("project_id")
        .and_then(|v| v.as_str())
        .unwrap_or("?");

    format!(
        "Project: {}\nMemories: {}, Relationships: {}, Clusters: {}, Avg relevance: {:.2}",
        project, count, rels, clusters, rel
    )
}

fn compact_simple(result: &Value) -> String {
    let success = result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let message = result.get("message").and_then(|v| v.as_str()).unwrap_or("");
    if success {
        message.to_string()
    } else {
        format!("Failed: {}", message)
    }
}

fn compact_fallback(result: &Value) -> String {
    // For unhandled tools, use minimal JSON
    serde_json::to_string(result).unwrap_or_else(|_| result.to_string())
}

fn format_score(score: f64) -> String {
    format!("{:.0}%", score * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_compact_query_result() {
        let result = json!({
            "count": 2,
            "memories": [
                {
                    "memory": {
                        "id": "mem_123",
                        "memory_type": "fact",
                        "content": "This is a test memory",
                        "summary": "Test memory",
                        "tags": ["test"],
                        "importance": 0.8
                    },
                    "score": 0.95,
                    "semantic_score": 0.9,
                    "keyword_score": 0.5
                },
                {
                    "memory": {
                        "id": "mem_456",
                        "memory_type": "decision",
                        "content": "Another memory",
                        "tags": [],
                        "importance": 0.5
                    },
                    "score": 0.75,
                    "semantic_score": 0.7,
                    "keyword_score": 0.3
                }
            ],
            "contradiction_warnings": []
        });

        let compact = compact_tool_result("memory_query", &result, 300);

        // Plain text format should contain memory IDs, types, content
        assert!(
            compact.contains("mem_123"),
            "Should contain first memory ID"
        );
        assert!(
            compact.contains("mem_456"),
            "Should contain second memory ID"
        );
        assert!(compact.contains("fact"), "Should contain memory type");
        assert!(compact.contains("decision"), "Should contain memory type");
        assert!(
            compact.contains("This is a test memory"),
            "Should contain content"
        );
        assert!(compact.contains("test"), "Should contain tags");
        assert!(compact.contains("2 result"), "Should show result count");
    }

    #[test]
    fn test_compact_store_result() {
        let result = json!({
            "id": "mem_789",
            "message": "Memory stored successfully",
            "potential_contradictions": []
        });

        let compact = compact_tool_result("memory_store", &result, 300);

        assert!(compact.contains("mem_789"), "Should contain memory ID");
        assert!(compact.starts_with("Stored"), "Should start with 'Stored'");
        assert!(
            !compact.contains("contradiction"),
            "No contradictions to show"
        );
    }

    #[test]
    fn test_compact_vs_full_size() {
        let full_result = json!({
            "count": 3,
            "memories": [
                {
                    "memory": {
                        "id": "mem_001",
                        "project_id": "test-project",
                        "memory_type": "fact",
                        "content": "The project uses Rust for the backend implementation",
                        "summary": "Backend is Rust",
                        "tags": ["architecture", "rust"],
                        "importance": 0.8,
                        "relevance_score": 0.95,
                        "access_count": 5,
                        "created_at": 1700000000,
                        "updated_at": 1700000000,
                        "last_accessed_at": 1700001000
                    },
                    "score": 0.92,
                    "semantic_score": 0.88,
                    "keyword_score": 0.45
                }
            ],
            "contradiction_warnings": []
        });

        let full_json = serde_json::to_string(&full_result).unwrap();
        let compact = compact_tool_result("memory_query", &full_result, 300);

        // Plain text should be significantly smaller than full JSON
        assert!(
            compact.len() < full_json.len(),
            "Compact ({}) should be smaller than full ({})",
            compact.len(),
            full_json.len()
        );
    }

    #[test]
    fn test_compact_query_content_length() {
        let long_content = "a".repeat(500);
        let result = json!({
            "memories": [
                {
                    "memory": {
                        "id": "mem_001",
                        "memory_type": "fact",
                        "content": long_content,
                        "tags": [],
                        "importance": 0.5
                    },
                    "score": 0.8
                }
            ]
        });

        let compact_100 = compact_tool_result("memory_query", &result, 100);
        let compact_300 = compact_tool_result("memory_query", &result, 300);

        assert!(compact_100.len() < compact_300.len());
        assert!(compact_100.contains("..."));
    }
}
