//! Human-readable formatting for MCP tool results.
//!
//! Converts JSON tool results into markdown-formatted text for display,
//! and compact JSON for token-efficient LLM consumption.

use serde_json::{Value, json};

/// Format a tool result as human-readable markdown.
pub fn format_tool_result(tool_name: &str, result: &Value) -> String {
    match tool_name {
        "memory_store" => format_store_result(result),
        "memory_query" => format_query_result(result),
        "memory_update" => format_simple_result(result),
        "memory_delete" => format_simple_result(result),
        "memory_link" => format_link_result(result),
        "memory_graph" => format_graph_result(result),
        "memory_store_batch" => format_batch_store_result(result),
        "memory_delete_batch" => format_batch_delete_result(result),
        "memory_stats" => format_stats_result(result),
        "memory_export" => format_export_result(result),
        "memory_import" => format_import_result(result),
        "memory_context" => format_context_result(result),
        "memory_prune" => format_prune_result(result),
        _ => format_fallback(result),
    }
}

fn format_store_result(result: &Value) -> String {
    let id = result.get("id").and_then(|v| v.as_str()).unwrap_or("?");
    let message = result
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("Stored");

    let mut out = format!("✓ Stored: {}\n  {}", id, message);

    if let Some(contradictions) = result
        .get("potential_contradictions")
        .and_then(|v| v.as_array())
        && !contradictions.is_empty()
    {
        out.push_str("\n\n⚠️ Potential contradictions:\n");
        for c in contradictions {
            let mem_id = c.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
            let summary = c.get("summary").and_then(|v| v.as_str()).unwrap_or("");
            let sim = c.get("similarity").and_then(|v| v.as_f64()).unwrap_or(0.0);
            out.push_str(&format!(
                "  • {} ({:.0}%): {}\n",
                mem_id,
                sim * 100.0,
                summary
            ));
        }
    }

    out
}

fn format_query_result(result: &Value) -> String {
    let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
    let memories = result.get("memories").and_then(|v| v.as_array());

    if count == 0 {
        return "No memories found.".to_string();
    }

    let mut out = format!(
        "Found {} memor{}:\n",
        count,
        if count == 1 { "y" } else { "ies" }
    );

    if let Some(memories) = memories {
        for mem in memories {
            let memory = mem.get("memory").unwrap_or(mem);
            let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let mem_type = memory
                .get("memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let content = memory.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let summary = memory.get("summary").and_then(|v| v.as_str());
            let tags = memory
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            let score = mem.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let importance = memory
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);

            // Type icon
            let icon = match mem_type {
                "fact" => "📋",
                "decision" => "🎯",
                "preference" => "⭐",
                "pattern" => "🔄",
                "debug" => "🐛",
                "entity" => "👤",
                _ => "📝",
            };

            out.push_str(&format!(
                "\n{} [{}] {} (score: {:.2}, importance: {:.1})\n",
                icon, mem_type, id, score, importance
            ));

            // Show summary or truncated content
            let display_content = summary.unwrap_or(content);
            let truncated = if display_content.len() > 200 {
                format!(
                    "{}...",
                    &display_content.chars().take(200).collect::<String>()
                )
            } else {
                display_content.to_string()
            };
            out.push_str(&format!("   {}\n", truncated.replace('\n', "\n   ")));

            if !tags.is_empty() {
                out.push_str(&format!("   Tags: {}\n", tags));
            }
        }
    }

    // Contradiction warnings
    if let Some(warnings) = result
        .get("contradiction_warnings")
        .and_then(|v| v.as_array())
        && !warnings.is_empty()
    {
        out.push_str("\n⚠️ Contradiction warnings:\n");
        for w in warnings {
            let mem_id = w.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
            let contra_id = w
                .get("contradicts_id")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            out.push_str(&format!("  • {} contradicts {}\n", mem_id, contra_id));
        }
    }

    out
}

fn format_simple_result(result: &Value) -> String {
    let success = result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let message = result
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or(if success { "Success" } else { "Failed" });

    if success {
        format!("✓ {}", message)
    } else {
        format!("✗ {}", message)
    }
}

fn format_link_result(result: &Value) -> String {
    let success = result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let id = result.get("id").and_then(|v| v.as_str()).unwrap_or("?");
    let message = result.get("message").and_then(|v| v.as_str()).unwrap_or("");

    if success {
        format!("✓ Link created: {}\n  {}", id, message)
    } else {
        format!("✗ {}", message)
    }
}

fn format_graph_result(result: &Value) -> String {
    let root = result.get("root");
    let related = result.get("related").and_then(|v| v.as_array());

    let mut out = String::new();

    if let Some(root) = root {
        let id = root.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let mem_type = root
            .get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let content = root.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let summary = root.get("summary").and_then(|v| v.as_str());

        out.push_str(&format!("Root: [{}] {}\n", mem_type, id));
        let display = summary.unwrap_or(content);
        let truncated = if display.len() > 150 {
            format!("{}...", &display.chars().take(150).collect::<String>())
        } else {
            display.to_string()
        };
        out.push_str(&format!("  {}\n", truncated));
    }

    if let Some(related) = related {
        if related.is_empty() {
            out.push_str("\nNo related memories.");
        } else {
            out.push_str(&format!("\nRelated ({}):\n", related.len()));
            for rel in related {
                let memory = rel.get("memory").unwrap_or(rel);
                let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let mem_type = memory
                    .get("memory_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let relation = rel.get("relation").and_then(|v| v.as_str()).unwrap_or("?");
                let direction = rel.get("direction").and_then(|v| v.as_str()).unwrap_or("?");
                let depth = rel.get("depth").and_then(|v| v.as_u64()).unwrap_or(0);
                let summary = memory.get("summary").and_then(|v| v.as_str());
                let content = memory.get("content").and_then(|v| v.as_str()).unwrap_or("");

                let arrow = if direction == "outgoing" {
                    "→"
                } else {
                    "←"
                };
                out.push_str(&format!(
                    "  {} [{}] {} ({}, depth {})\n",
                    arrow, mem_type, id, relation, depth
                ));

                let display = summary.unwrap_or(content);
                if !display.is_empty() {
                    let truncated = if display.len() > 100 {
                        format!("{}...", &display.chars().take(100).collect::<String>())
                    } else {
                        display.to_string()
                    };
                    out.push_str(&format!("    {}\n", truncated));
                }
            }
        }
    }

    out
}

fn format_batch_store_result(result: &Value) -> String {
    let success = result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
    let ids = result.get("ids").and_then(|v| v.as_array());

    if success {
        let mut out = format!("✓ Stored {} memories\n", count);
        if let Some(ids) = ids {
            out.push_str("IDs: ");
            let id_strs: Vec<_> = ids.iter().filter_map(|v| v.as_str()).collect();
            if id_strs.len() <= 5 {
                out.push_str(&id_strs.join(", "));
            } else {
                out.push_str(&format!(
                    "{}, ... (+{} more)",
                    id_strs[..3].join(", "),
                    id_strs.len() - 3
                ));
            }
        }
        out
    } else {
        let error = result
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown error");
        format!("✗ Batch store failed: {}", error)
    }
}

fn format_batch_delete_result(result: &Value) -> String {
    let deleted = result.get("deleted").and_then(|v| v.as_u64()).unwrap_or(0);
    format!("✓ Deleted {} memories", deleted)
}

fn format_stats_result(result: &Value) -> String {
    let project = result
        .get("project_id")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let mem_count = result
        .get("memory_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let rel_count = result
        .get("relationship_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let avg_rel = result
        .get("avg_relevance")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    format!(
        "Project: {}\n\
         Memories: {}\n\
         Relationships: {}\n\
         Avg relevance: {:.2}",
        project, mem_count, rel_count, avg_rel
    )
}

fn format_export_result(result: &Value) -> String {
    let mem_count = result
        .get("memories")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let rel_count = result
        .get("relationships")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    format!(
        "✓ Exported {} memories, {} relationships\n(Full JSON data available in structured output)",
        mem_count, rel_count
    )
}

fn format_import_result(result: &Value) -> String {
    let success = result
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let message = result.get("message").and_then(|v| v.as_str()).unwrap_or("");

    if let Some(stats) = result.get("stats") {
        let mem_imported = stats
            .get("memories_imported")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let mem_skipped = stats
            .get("memories_skipped")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let rel_imported = stats
            .get("relationships_imported")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        format!(
            "✓ Import complete\n\
             Memories: {} imported, {} skipped\n\
             Relationships: {} imported",
            mem_imported, mem_skipped, rel_imported
        )
    } else if success {
        format!("✓ {}", message)
    } else {
        format!("✗ Import failed: {}", message)
    }
}

fn format_context_result(result: &Value) -> String {
    let context = result
        .get("context")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
    let memories = result.get("memories").and_then(|v| v.as_array());

    if count == 0 {
        return format!(
            "Context: \"{}\"\n\nNo relevant memories found.",
            truncate_str(context, 80)
        );
    }

    let mut out = format!(
        "Context: \"{}\"\nFound {} relevant memor{}:\n",
        truncate_str(context, 80),
        count,
        if count == 1 { "y" } else { "ies" }
    );

    if let Some(memories) = memories {
        for mem in memories {
            let id = mem.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let mem_type = mem.get("type").and_then(|v| v.as_str()).unwrap_or("?");
            let content = mem.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let summary = mem.get("summary").and_then(|v| v.as_str());
            let similarity = mem
                .get("similarity")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let importance = mem
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);
            let tags = mem
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();

            let icon = match mem_type {
                "fact" => "📋",
                "decision" => "🎯",
                "preference" => "⭐",
                "pattern" => "🔄",
                "debug" => "🐛",
                "entity" => "👤",
                _ => "📝",
            };

            out.push_str(&format!(
                "\n{} [{}] {} (similarity: {:.0}%, importance: {:.1})\n",
                icon,
                mem_type,
                id,
                similarity * 100.0,
                importance
            ));

            let display = summary.unwrap_or(content);
            let truncated = truncate_str(display, 200);
            out.push_str(&format!("   {}\n", truncated.replace('\n', "\n   ")));

            if !tags.is_empty() {
                out.push_str(&format!("   Tags: {}\n", tags));
            }
        }
    }

    out
}

fn format_prune_result(result: &Value) -> String {
    let dry_run = result
        .get("dry_run")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let threshold = result
        .get("threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.2);
    let candidates = result
        .get("candidates")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let deleted = result.get("deleted").and_then(|v| v.as_u64()).unwrap_or(0);
    let memories = result.get("memories").and_then(|v| v.as_array());

    if candidates == 0 {
        return format!(
            "✓ No memories below threshold {:.2} - nothing to prune.",
            threshold
        );
    }

    let mut out = if dry_run {
        format!(
            "🔍 Dry run - Found {} memor{} below threshold {:.2}:\n",
            candidates,
            if candidates == 1 { "y" } else { "ies" },
            threshold
        )
    } else {
        format!(
            "🗑️ Pruned {} memor{} below threshold {:.2}:\n",
            deleted,
            if deleted == 1 { "y" } else { "ies" },
            threshold
        )
    };

    if let Some(memories) = memories {
        for mem in memories.iter().take(10) {
            let id = mem.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let mem_type = mem.get("type").and_then(|v| v.as_str()).unwrap_or("?");
            let relevance = mem
                .get("relevance_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let summary = mem.get("summary").and_then(|v| v.as_str()).unwrap_or("?");

            out.push_str(&format!(
                "\n  • {} [{}] (relevance: {:.2})\n    {}\n",
                id,
                mem_type,
                relevance,
                truncate_str(summary, 60)
            ));
        }

        if memories.len() > 10 {
            out.push_str(&format!("\n... and {} more\n", memories.len() - 10));
        }
    }

    if dry_run {
        out.push_str("\n⚠️ Set confirm=true to actually delete these memories.");
    }

    out
}

/// Truncate a string to a maximum length, adding "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max_len).collect::<String>())
    }
}

fn format_fallback(result: &Value) -> String {
    // For unknown tools, just show a compact summary
    if let Some(obj) = result.as_object()
        && let Some(success) = obj.get("success").and_then(|v| v.as_bool())
    {
        let msg = obj.get("message").and_then(|v| v.as_str()).unwrap_or("");
        return if success {
            format!("✓ {}", msg)
        } else {
            format!("✗ {}", msg)
        };
    }
    // Fallback to pretty JSON
    serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string())
}

/// Produce compact JSON for LLM consumption.
/// Uses short field names and omits unnecessary data.
pub fn compact_tool_result(tool_name: &str, result: &Value) -> String {
    let compact = match tool_name {
        "memory_query" => compact_query_result(result),
        "memory_context" => compact_context_result(result),
        "memory_store" => compact_store_result(result),
        "memory_graph" => compact_graph_result(result),
        "memory_store_batch" => compact_batch_store_result(result),
        "memory_prune" => compact_prune_result(result),
        // Simple ops: just pass through (already small)
        _ => result.clone(),
    };
    serde_json::to_string(&compact).unwrap_or_else(|_| result.to_string())
}

fn compact_store_result(result: &Value) -> Value {
    let id = result.get("id").cloned().unwrap_or(json!("?"));
    let contradictions = result
        .get("potential_contradictions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|c| {
                    json!({
                        "i": c.get("memory_id"),
                        "s": c.get("similarity")
                    })
                })
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty());

    match contradictions {
        Some(c) => json!({"i": id, "c": c}),
        None => json!({"i": id}),
    }
}

fn compact_query_result(result: &Value) -> Value {
    let memories = result
        .get("memories")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|mem| {
                    let memory = mem.get("memory").unwrap_or(mem);
                    let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                    let mem_type = memory
                        .get("memory_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    let summary = memory
                        .get("summary")
                        .and_then(|v| v.as_str())
                        .or_else(|| memory.get("content").and_then(|v| v.as_str()))
                        .map(|s| truncate_str(s, 150))
                        .unwrap_or_default();
                    let score = mem.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let tags = memory
                        .get("tags")
                        .and_then(|v| v.as_array())
                        .filter(|t| !t.is_empty())
                        .cloned();

                    let mut obj = json!({
                        "i": id,
                        "t": mem_type,
                        "s": summary,
                        "r": (score * 100.0).round() / 100.0
                    });
                    if let Some(t) = tags {
                        obj["g"] = json!(t);
                    }
                    obj
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let count = memories.len();

    // Include contradiction warnings if present
    let warnings = result
        .get("contradiction_warnings")
        .and_then(|v| v.as_array())
        .filter(|w| !w.is_empty())
        .map(|arr| {
            arr.iter()
                .map(|w| {
                    json!({
                        "a": w.get("memory_id"),
                        "b": w.get("contradicts_id")
                    })
                })
                .collect::<Vec<_>>()
        });

    match warnings {
        Some(w) => json!({"n": count, "m": memories, "w": w}),
        None => json!({"n": count, "m": memories}),
    }
}

fn compact_context_result(result: &Value) -> Value {
    let memories = result
        .get("memories")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|mem| {
                    let id = mem.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                    let mem_type = mem.get("type").and_then(|v| v.as_str()).unwrap_or("?");
                    let summary = mem
                        .get("summary")
                        .and_then(|v| v.as_str())
                        .or_else(|| mem.get("content").and_then(|v| v.as_str()))
                        .map(|s| truncate_str(s, 150))
                        .unwrap_or_default();
                    let sim = mem
                        .get("similarity")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let tags = mem
                        .get("tags")
                        .and_then(|v| v.as_array())
                        .filter(|t| !t.is_empty())
                        .cloned();

                    let mut obj = json!({
                        "i": id,
                        "t": mem_type,
                        "s": summary,
                        "r": (sim * 100.0).round() / 100.0
                    });
                    if let Some(t) = tags {
                        obj["g"] = json!(t);
                    }
                    obj
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    json!({"n": memories.len(), "m": memories})
}

fn compact_graph_result(result: &Value) -> Value {
    let root_id = result
        .get("root")
        .and_then(|r| r.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("?");

    let related = result
        .get("related")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|rel| {
                    let memory = rel.get("memory").unwrap_or(rel);
                    let id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                    let relation = rel.get("relation").and_then(|v| v.as_str()).unwrap_or("?");
                    let depth = rel.get("depth").and_then(|v| v.as_u64()).unwrap_or(0);
                    json!({"i": id, "rel": relation, "d": depth})
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    json!({"root": root_id, "rel": related})
}

fn compact_batch_store_result(result: &Value) -> Value {
    let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
    let ids = result
        .get("ids")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    json!({"n": count, "ids": ids})
}

fn compact_prune_result(result: &Value) -> Value {
    let dry_run = result
        .get("dry_run")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let candidates = result
        .get("candidates")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let deleted = result.get("deleted").and_then(|v| v.as_u64()).unwrap_or(0);

    if dry_run {
        json!({"dry": true, "n": candidates})
    } else {
        json!({"del": deleted})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_format_query_empty() {
        let result = json!({"count": 0, "memories": []});
        assert_eq!(format_query_result(&result), "No memories found.");
    }

    #[test]
    fn test_format_store() {
        let result = json!({
            "id": "mem_123",
            "message": "Memory stored successfully",
            "potential_contradictions": []
        });
        let formatted = format_store_result(&result);
        assert!(formatted.contains("mem_123"));
        assert!(formatted.contains("Stored"));
    }

    #[test]
    fn test_format_stats() {
        let result = json!({
            "project_id": "test-project",
            "memory_count": 42,
            "relationship_count": 10,
            "avg_relevance": 0.85
        });
        let formatted = format_stats_result(&result);
        assert!(formatted.contains("test-project"));
        assert!(formatted.contains("42"));
        assert!(formatted.contains("0.85"));
    }

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

        let compact = compact_tool_result("memory_query", &result);
        let parsed: Value = serde_json::from_str(&compact).unwrap();

        // Should have count as "n"
        assert_eq!(parsed["n"], 2);
        // Should have memories as "m"
        assert!(parsed["m"].is_array());
        assert_eq!(parsed["m"].as_array().unwrap().len(), 2);
        // First memory should use short field names
        assert_eq!(parsed["m"][0]["i"], "mem_123");
        assert_eq!(parsed["m"][0]["t"], "fact");
        assert_eq!(parsed["m"][0]["s"], "Test memory");
        assert!(parsed["m"][0]["r"].as_f64().unwrap() > 0.9);
        // Tags present for first, absent for second (empty)
        assert!(parsed["m"][0]["g"].is_array());
        assert!(parsed["m"][1].get("g").is_none());
        // No contradiction warnings field when empty
        assert!(parsed.get("w").is_none());
    }

    #[test]
    fn test_compact_store_result() {
        let result = json!({
            "id": "mem_789",
            "message": "Memory stored successfully",
            "potential_contradictions": []
        });

        let compact = compact_tool_result("memory_store", &result);
        let parsed: Value = serde_json::from_str(&compact).unwrap();

        // Should just have id
        assert_eq!(parsed["i"], "mem_789");
        // No contradictions field when empty
        assert!(parsed.get("c").is_none());
    }

    #[test]
    fn test_compact_vs_full_size() {
        // Simulate a typical query result
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
        let compact_json = compact_tool_result("memory_query", &full_result);

        // Compact should be significantly smaller
        assert!(
            compact_json.len() < full_json.len() / 2,
            "Compact ({}) should be less than half of full ({})",
            compact_json.len(),
            full_json.len()
        );
    }
}
