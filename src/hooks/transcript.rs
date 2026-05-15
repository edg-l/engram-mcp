//! Parses Claude Code session transcript JSONL files.
//!
//! Claude Code writes transcripts as JSONL (one JSON object per line). Each line
//! has a top-level `type` field (`"user"` / `"assistant"` / others) and a
//! `message` object. For assistant entries, `message.content` is primarily a
//! content-block array `[{"type":"text","text":"..."}]` and occasionally a plain
//! string. This module is fail-open: I/O errors and malformed lines are silently
//! skipped rather than propagated.

use serde::Deserialize;

#[allow(dead_code)]
#[derive(Deserialize)]
struct TranscriptLine {
    #[serde(rename = "type")]
    entry_type: Option<String>,
    message: Option<TranscriptMessage>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct TranscriptMessage {
    content: Option<MessageContent>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Blocks(Vec<ContentBlock>), // primary
    Text(String),              // fallback
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: Option<String>,
    text: Option<String>,
}

/// Extract the last `max_messages` assistant messages from a Claude Code transcript
/// file. Returns `None` on I/O error, missing file, or if no assistant text is found.
/// Never panics; parse errors on individual lines are silently skipped.
#[allow(dead_code)]
pub fn extract_last_assistant_text(transcript_path: &str, max_messages: usize) -> Option<String> {
    let contents = std::fs::read_to_string(transcript_path).ok()?;

    let mut assistant_texts: Vec<String> = Vec::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let entry: TranscriptLine = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if entry.entry_type.as_deref() != Some("assistant") {
            continue;
        }

        let message = match entry.message {
            Some(m) => m,
            None => continue,
        };

        let text = match message.content {
            Some(MessageContent::Blocks(blocks)) => blocks
                .iter()
                .filter(|b| b.block_type.as_deref() == Some("text"))
                .filter_map(|b| b.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n"),
            Some(MessageContent::Text(s)) => s,
            None => continue,
        };

        if text.is_empty() {
            continue;
        }

        assistant_texts.push(text);
    }

    if assistant_texts.is_empty() {
        return None;
    }

    let start = assistant_texts.len().saturating_sub(max_messages);
    Some(assistant_texts[start..].join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(content.as_bytes()).expect("write");
        f
    }

    #[test]
    fn well_formed_multi_line_returns_last_assistant() {
        let jsonl = r#"{"type":"user","message":{"content":"hello"}}
{"type":"assistant","message":{"content":[{"type":"text","text":"first response"}]}}
{"type":"user","message":{"content":"follow up"}}
{"type":"assistant","message":{"content":[{"type":"text","text":"second response"}]}}
"#;
        let f = write_temp(jsonl);
        let result =
            extract_last_assistant_text(f.path().to_str().unwrap(), 1).expect("should return Some");
        assert_eq!(result, "second response");
    }

    #[test]
    fn nonexistent_path_returns_none() {
        let result = extract_last_assistant_text("/nonexistent/path/transcript.jsonl", 10);
        assert!(result.is_none());
    }

    #[test]
    fn only_user_entries_returns_none() {
        let jsonl = r#"{"type":"user","message":{"content":"hello"}}
{"type":"user","message":{"content":"still user"}}
"#;
        let f = write_temp(jsonl);
        let result = extract_last_assistant_text(f.path().to_str().unwrap(), 10);
        assert!(result.is_none());
    }

    #[test]
    fn malformed_line_in_middle_still_extracts_valid() {
        let jsonl = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"before bad line"}]}}
this is not json at all }{
{"type":"assistant","message":{"content":[{"type":"text","text":"after bad line"}]}}
"#;
        let f = write_temp(jsonl);
        let result =
            extract_last_assistant_text(f.path().to_str().unwrap(), 1).expect("should return Some");
        assert_eq!(result, "after bad line");
    }

    #[test]
    fn plain_string_content_fallback() {
        let jsonl = r#"{"type":"assistant","message":{"content":"plain string response"}}
"#;
        let f = write_temp(jsonl);
        let result = extract_last_assistant_text(f.path().to_str().unwrap(), 10)
            .expect("should return Some");
        assert_eq!(result, "plain string response");
    }

    #[test]
    fn max_messages_limits_output() {
        let jsonl = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"msg one"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"msg two"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"msg three"}]}}
"#;
        let f = write_temp(jsonl);
        let result =
            extract_last_assistant_text(f.path().to_str().unwrap(), 2).expect("should return Some");
        assert_eq!(result, "msg two\n\nmsg three");
    }
}
