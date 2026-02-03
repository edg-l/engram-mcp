//! Extractive summarization for memories.
//!
//! Strategy: First sentence (≤150 chars) + top 3 keywords from content.
//! Auto-triggered when content > 500 chars and no summary is provided.

use std::collections::HashMap;

/// Threshold for auto-generating summaries
pub const AUTO_SUMMARY_THRESHOLD: usize = 500;

/// Maximum length for the summary
const MAX_SUMMARY_LENGTH: usize = 150;

/// Stop words to exclude from keyword extraction
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "what",
    "which", "who", "when", "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there", "then", "once", "if", "while",
    "although", "because", "until", "unless", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "again", "further", "any", "am", "being",
];

/// Generate an extractive summary from content.
///
/// Returns the first sentence (up to MAX_SUMMARY_LENGTH chars) followed by
/// top keywords if there's room.
pub fn generate_summary(content: &str) -> String {
    let first_sentence = extract_first_sentence(content);
    let keywords = extract_keywords(content, 3);

    // If first sentence is short enough, append keywords
    if first_sentence.len() < MAX_SUMMARY_LENGTH - 20 && !keywords.is_empty() {
        let keyword_suffix = format!(" [{}]", keywords.join(", "));
        let combined = format!("{}{}", first_sentence, keyword_suffix);
        if combined.len() <= MAX_SUMMARY_LENGTH {
            return combined;
        }
    }

    // Just return the truncated first sentence
    if first_sentence.len() > MAX_SUMMARY_LENGTH {
        let truncated = &first_sentence[..MAX_SUMMARY_LENGTH - 3];
        // Try to break at word boundary
        if let Some(last_space) = truncated.rfind(' ') {
            return format!("{}...", &truncated[..last_space]);
        }
        return format!("{}...", truncated);
    }

    first_sentence
}

/// Extract the first sentence from content.
fn extract_first_sentence(content: &str) -> String {
    let content = content.trim();

    // Find sentence boundary (., !, ?)
    for (i, c) in content.char_indices() {
        if (c == '.' || c == '!' || c == '?') && i > 0 {
            // Check it's not part of an abbreviation (e.g., "Dr.", "etc.")
            let before = &content[..i];
            let words: Vec<&str> = before.split_whitespace().collect();
            if let Some(last_word) = words.last() {
                // Skip common abbreviations
                if last_word.len() <= 4
                    && last_word.chars().next().is_some_and(|c| c.is_uppercase())
                {
                    continue;
                }
            }
            return content[..=i].to_string();
        }
    }

    // No sentence boundary found, return first line or truncated content
    if let Some(newline_pos) = content.find('\n') {
        return content[..newline_pos].to_string();
    }

    // Return first MAX_SUMMARY_LENGTH chars if no boundary found
    if content.len() > MAX_SUMMARY_LENGTH {
        content[..MAX_SUMMARY_LENGTH].to_string()
    } else {
        content.to_string()
    }
}

/// Extract top N keywords from content using simple TF scoring.
fn extract_keywords(content: &str, n: usize) -> Vec<String> {
    let mut word_counts: HashMap<String, usize> = HashMap::new();

    // Tokenize and count words
    for word in content.split_whitespace() {
        // Clean word: remove punctuation, lowercase
        let word: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();

        // Skip short words and stop words
        if word.len() < 3 || STOP_WORDS.contains(&word.as_str()) {
            continue;
        }

        *word_counts.entry(word).or_insert(0) += 1;
    }

    // Sort by count descending
    let mut word_vec: Vec<(String, usize)> = word_counts.into_iter().collect();
    word_vec.sort_by(|a, b| b.1.cmp(&a.1));

    // Return top N
    word_vec.into_iter().take(n).map(|(word, _)| word).collect()
}

/// Check if content should have an auto-generated summary.
pub fn should_auto_summarize(content: &str, existing_summary: Option<&str>) -> bool {
    existing_summary.is_none() && content.len() > AUTO_SUMMARY_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_first_sentence() {
        assert_eq!(
            extract_first_sentence("Hello world. This is a test."),
            "Hello world."
        );
        assert_eq!(
            extract_first_sentence("Single line with no period"),
            "Single line with no period"
        );
        assert_eq!(extract_first_sentence("Question? Answer."), "Question?");
    }

    #[test]
    fn test_extract_keywords() {
        let content = "The database uses SQLite for persistence. SQLite is fast and reliable. The database schema is simple.";
        let keywords = extract_keywords(content, 3);
        assert!(keywords.contains(&"database".to_string()));
        assert!(keywords.contains(&"sqlite".to_string()));
    }

    #[test]
    fn test_generate_summary() {
        let short = "This is short.";
        let short_summary = generate_summary(short);
        // Short content may get keywords appended if they fit
        assert!(short_summary.starts_with("This is short."));

        let long = "This is a much longer piece of content that talks about various things. \
                    It mentions databases multiple times. The database is important. \
                    Database performance matters a lot in production systems.";
        let summary = generate_summary(long);
        assert!(summary.len() <= MAX_SUMMARY_LENGTH);
        assert!(summary.starts_with("This is a much longer"));
    }

    #[test]
    fn test_should_auto_summarize() {
        let short = "Short content";
        let long = "x".repeat(600);

        assert!(!should_auto_summarize(short, None));
        assert!(should_auto_summarize(&long, None));
        assert!(!should_auto_summarize(&long, Some("existing")));
    }
}
