use std::io::BufReader;
use std::path::Path;

use serde::Deserialize;

/// Deserializes a JSON value (string or number) into a String.
fn answer_as_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = serde_json::Value::deserialize(deserializer)?;
    match v {
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        other => Err(serde::de::Error::custom(format!(
            "expected string or number for answer, got {:?}",
            other
        ))),
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LmeQuestion {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    #[serde(deserialize_with = "answer_as_string")]
    pub answer: String,
    pub answer_session_ids: Vec<String>,
    pub haystack_session_ids: Vec<String>,
    pub haystack_dates: Vec<String>,
    pub haystack_sessions: Vec<Vec<LmeTurn>>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LmeTurn {
    pub role: String,
    pub content: String,
}

/// Loads the LongMemEval-S dataset from a JSON file.
///
/// The file is a JSON array of questions (~277 MB). Uses a buffered reader
/// to avoid loading the entire file into a String.
pub fn load_dataset(path: &Path) -> anyhow::Result<Vec<LmeQuestion>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let questions: Vec<LmeQuestion> = serde_json::from_reader(reader)?;
    Ok(questions)
}

#[allow(dead_code)]
/// Parses a LongMemEval haystack date string.
///
/// Format: `"%Y/%m/%d (%a) %H:%M"`, e.g., `"2023/05/20 (Sat) 02:21"`.
pub fn parse_haystack_date(s: &str) -> anyhow::Result<chrono::DateTime<chrono::Utc>> {
    let naive = chrono::NaiveDateTime::parse_from_str(s, "%Y/%m/%d (%a) %H:%M")?;
    Ok(naive.and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_saturday_date() {
        let dt = parse_haystack_date("2023/05/20 (Sat) 02:21").unwrap();
        assert_eq!(dt.to_rfc3339(), "2023-05-20T02:21:00+00:00");
    }

    #[test]
    fn parse_tuesday_date() {
        let dt = parse_haystack_date("2023/05/30 (Tue) 23:40").unwrap();
        assert_eq!(dt.to_rfc3339(), "2023-05-30T23:40:00+00:00");
    }

    #[test]
    fn parse_monday_date() {
        let dt = parse_haystack_date("2023/05/22 (Mon) 14:27").unwrap();
        assert_eq!(dt.to_rfc3339(), "2023-05-22T14:27:00+00:00");
    }

    #[test]
    fn parse_invalid_date_returns_error() {
        assert!(parse_haystack_date("not a date").is_err());
    }
}
