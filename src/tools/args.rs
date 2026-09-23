//! Tolerant deserialization for MCP tool arguments sent in more than one shape.

use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// Splits a comma-separated string into trimmed, non-empty tags. Agents send
/// `"tags": "a, b, c"` often enough that rejecting it costs a retry for something the
/// caller clearly meant unambiguously.
fn tags_from_value(value: Value) -> Result<Vec<String>, String> {
    match value {
        Value::Array(_) => serde_json::from_value(value).map_err(|e| e.to_string()),
        Value::String(s) => Ok(s
            .split(',')
            .map(str::trim)
            .filter(|tag| !tag.is_empty())
            .map(str::to_string)
            .collect()),
        other => Err(format!(
            "expected an array of strings or a comma-separated string, got {other}"
        )),
    }
}

/// Deserializes `tags`, accepting a JSON array of strings or a comma-separated string.
pub fn deserialize_tags<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    tags_from_value(value).map_err(serde::de::Error::custom)
}

/// `Option<Vec<String>>` counterpart of [`deserialize_tags`], for fields where omitting
/// `tags` means "leave unchanged" rather than "clear".
pub fn deserialize_opt_tags<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    if value.is_null() {
        return Ok(None);
    }
    tags_from_value(value)
        .map(Some)
        .map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize)]
    struct Tags {
        #[serde(default, deserialize_with = "deserialize_tags")]
        tags: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    struct OptTags {
        #[serde(default, deserialize_with = "deserialize_opt_tags")]
        tags: Option<Vec<String>>,
    }

    #[test]
    fn accepts_a_json_array() {
        let t: Tags = serde_json::from_value(serde_json::json!({"tags": ["a", "b"]})).unwrap();
        assert_eq!(t.tags, vec!["a", "b"]);
    }

    #[test]
    fn splits_a_comma_separated_string() {
        let t: Tags = serde_json::from_value(serde_json::json!({
            "tags": "ffmarket, censored-demand, newsvendor"
        }))
        .unwrap();
        assert_eq!(t.tags, vec!["ffmarket", "censored-demand", "newsvendor"]);
    }

    #[test]
    fn missing_tags_defaults_to_empty() {
        let t: Tags = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(t.tags.is_empty());
    }

    #[test]
    fn opt_tags_missing_stays_none() {
        let t: OptTags = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(t.tags, None);
    }

    #[test]
    fn opt_tags_accepts_comma_string() {
        let t: OptTags = serde_json::from_value(serde_json::json!({"tags": "a, b"})).unwrap();
        assert_eq!(t.tags, Some(vec!["a".to_string(), "b".to_string()]));
    }
}
