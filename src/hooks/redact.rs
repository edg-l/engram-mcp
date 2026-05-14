//! Secret-redaction for hook payloads.
//!
//! All six patterns are compiled exactly once via `OnceLock<regex::Regex>`.

use regex::Regex;
use std::sync::OnceLock;

// ---- pattern 1: AWS access key ID ----
fn re_aws_key_id() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\bAKIA[0-9A-Z]{16}\b").unwrap())
}

// ---- pattern 2: AWS secret access key (context-gated) ----
fn re_aws_secret() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?i)aws[_-]?secret[_-]?access[_-]?key["'\s:=]+[A-Za-z0-9/+=]{40}"#).unwrap()
    })
}

// ---- pattern 3: GitHub tokens (PAT / fine-grained / OAuth / server / refresh) ----
fn re_github_token() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b").unwrap())
}

// ---- pattern 4: generic env assignment of sensitive variable names ----
fn re_env_assignment() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:[A-Z][A-Z0-9_]*_(?:KEY|SECRET|TOKEN)|PASSWORD)\s*=\s*\S+").unwrap()
    })
}

// ---- pattern 5: JWT ----
fn re_jwt() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+").unwrap()
    })
}

// ---- pattern 6: OpenAI secret key ----
fn re_openai_key() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\bsk-[A-Za-z0-9]{20,}\b").unwrap())
}

/// Redact secrets from `input`.
///
/// For patterns 1, 2, 3, 5, 6 the entire match is replaced with `[REDACTED]`.
/// For pattern 4 (env assignment), only the value portion is replaced:
/// `FOO_KEY=secret` → `FOO_KEY=[REDACTED]`.
pub fn redact(input: &str) -> String {
    // Patterns 1, 3, 5, 6: replace full match
    let s = re_aws_key_id().replace_all(input, "[REDACTED]");
    let s = re_aws_secret().replace_all(&s, "[REDACTED]");
    let s = re_github_token().replace_all(&s, "[REDACTED]");
    let s = re_jwt().replace_all(&s, "[REDACTED]");
    let s = re_openai_key().replace_all(&s, "[REDACTED]");

    // Pattern 4: replace only the value after '=' — keep the variable name visible.
    // We capture the variable name part and the `=` to preserve them, replacing `\S+`.
    let result = re_env_assignment().replace_all(&s, |caps: &regex::Captures| {
        let full_match = caps.get(0).unwrap().as_str();
        // Find the '=' character and replace everything after it
        if let Some(eq_pos) = full_match.find('=') {
            let lhs = &full_match[..=eq_pos];
            format!("{}[REDACTED]", lhs)
        } else {
            "[REDACTED]".to_string()
        }
    });

    result.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_aws_access_key() {
        let input = "The key is AKIAIOSFODNN7EXAMPLE and it's live";
        let output = redact(input);
        assert!(
            !output.contains("AKIAIOSFODNN7EXAMPLE"),
            "key should be redacted"
        );
        assert!(output.contains("[REDACTED]"), "placeholder missing");
    }

    #[test]
    fn redact_aws_secret_context() {
        let input = r#"aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY""#;
        let output = redact(input);
        assert!(
            !output.contains("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
            "secret should be redacted"
        );
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn redact_github_pat() {
        let input = "token: ghp_1234567890abcdefghij1234567890abcdef12";
        let output = redact(input);
        assert!(
            !output.contains("ghp_1234567890abcdefghij1234567890abcdef12"),
            "GitHub PAT should be redacted"
        );
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn redact_openai_key() {
        let input = "key=sk-abcdefghijklmnopqrstuvwxyzABCD";
        let output = redact(input);
        assert!(!output.contains("sk-abcdefghijklmnopqrstuvwxyzABCD"));
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn redact_jwt() {
        let input = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        let output = redact(input);
        assert!(!output.contains("eyJhbGciOiJIUzI1NiJ9"));
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn redact_env_assignment() {
        let input = "DB_TOKEN=supersecret123 and API_KEY=abc123xyz";
        let output = redact(input);
        assert!(
            !output.contains("supersecret123"),
            "token value should be redacted"
        );
        assert!(
            !output.contains("abc123xyz"),
            "key value should be redacted"
        );
        // Variable names should remain visible
        assert!(
            output.contains("DB_TOKEN="),
            "variable name should be preserved"
        );
        assert!(
            output.contains("API_KEY="),
            "variable name should be preserved"
        );
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn redact_preserves_non_secret_content() {
        let url = "https://example.com/page#foobar_KEY-anchor";
        let sentence = "the cake is a lie";
        let log_line = "2024-01-01T00:00:00Z INFO engram_mcp: store_memory id=mem_abc count=42";
        let input = format!("{}\n{}\n{}", url, sentence, log_line);
        let output = redact(&input);
        assert_eq!(
            output, input,
            "non-secret content should be unchanged byte-for-byte"
        );
    }
}
