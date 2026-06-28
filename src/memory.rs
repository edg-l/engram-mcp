use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::error::MemoryError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Fact,
    Decision,
    Preference,
    Pattern,
    Debug,
    Entity,
    Handoff,
    Adr,
}

impl MemoryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Decision => "decision",
            Self::Preference => "preference",
            Self::Pattern => "pattern",
            Self::Debug => "debug",
            Self::Entity => "entity",
            Self::Handoff => "handoff",
            Self::Adr => "adr",
        }
    }
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseMemoryTypeError(pub String);

impl fmt::Display for ParseMemoryTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid memory type: {}", self.0)
    }
}

impl std::error::Error for ParseMemoryTypeError {}

impl FromStr for MemoryType {
    type Err = ParseMemoryTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fact" => Ok(Self::Fact),
            "decision" => Ok(Self::Decision),
            "preference" => Ok(Self::Preference),
            "pattern" => Ok(Self::Pattern),
            "debug" => Ok(Self::Debug),
            "entity" => Ok(Self::Entity),
            "handoff" => Ok(Self::Handoff),
            "adr" => Ok(Self::Adr),
            _ => Err(ParseMemoryTypeError(s.to_string())),
        }
    }
}

/// Status lifecycle for an ADR memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)] // Used by ADR create/update/query tools (Phase 2)
pub enum AdrStatus {
    Proposed,
    Accepted,
    Superseded,
    Deprecated,
    Rejected,
}

#[allow(dead_code)] // Used by ADR create/update/query tools (Phase 2)
impl AdrStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Accepted => "accepted",
            Self::Superseded => "superseded",
            Self::Deprecated => "deprecated",
            Self::Rejected => "rejected",
        }
    }

    /// Returns true when transitioning from `self` to `to` is a valid lifecycle move.
    pub fn can_transition_to(self, to: AdrStatus) -> bool {
        if self == to {
            return true;
        }
        match self {
            Self::Proposed => matches!(to, Self::Accepted | Self::Rejected | Self::Deprecated),
            Self::Accepted => matches!(to, Self::Deprecated | Self::Superseded),
            Self::Rejected => matches!(to, Self::Proposed),
            Self::Deprecated => matches!(to, Self::Accepted),
            Self::Superseded => false,
        }
    }
}

impl fmt::Display for AdrStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // Used by ADR create/update/query tools (Phase 2)
pub struct ParseAdrStatusError(pub String);

impl fmt::Display for ParseAdrStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid ADR status: {}", self.0)
    }
}

impl std::error::Error for ParseAdrStatusError {}

impl FromStr for AdrStatus {
    type Err = ParseAdrStatusError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "proposed" => Ok(Self::Proposed),
            "accepted" => Ok(Self::Accepted),
            "superseded" => Ok(Self::Superseded),
            "deprecated" => Ok(Self::Deprecated),
            "rejected" => Ok(Self::Rejected),
            _ => Err(ParseAdrStatusError(s.to_string())),
        }
    }
}

/// Structured fields for a `MemoryType::Handoff` memory.
///
/// The canonical text representation (stored in `memories.content`) is produced by
/// `render_markdown` and can be round-tripped via `parse_markdown`.  The sidecar
/// `handoff_sections` table stores these fields individually plus per-section embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Used by handoff create/resume/search tools (Phase 3)
pub struct HandoffSections {
    /// High-level summary of the session's work.
    pub summary: String,
    /// Key decisions made during the session (each item is one decision).
    pub decisions: Vec<String>,
    /// Within-session work the next agent should pick up immediately. Concrete, ready-to-execute items.
    pub todos: Vec<String>,
    /// Things preventing forward motion right now (missing access, failing dependency, unanswered question).
    pub blockers: Vec<String>,
    /// Mental model: architecture, invariants, or context the next session needs.
    pub mental_model: String,
    /// Post-session follow-ups beyond the current thread. Future-facing, not for immediate pickup.
    pub next_steps: Vec<String>,
    /// Freeform notes that don't fit the other sections (optional).
    pub notes: Option<String>,
    /// ID of the handoff this one continues from (optional, sidecar-only chain link).
    pub continues_from: Option<String>,
}

#[allow(dead_code)] // Used by handoff create/resume/search tools (Phase 3)
impl HandoffSections {
    /// Render sections into canonical markdown stored in `memories.content`.
    ///
    /// Sections are emitted in fixed order.  Empty `Vec` fields and `None` optional
    /// fields are omitted entirely so the stored text stays compact.
    pub fn render_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str("## Summary\n\n");
        out.push_str(&self.summary);
        out.push_str("\n\n");

        if !self.decisions.is_empty() {
            out.push_str("## Decisions\n\n");
            for d in &self.decisions {
                out.push_str(&format!("- {}\n", d));
            }
            out.push('\n');
        }

        if !self.todos.is_empty() {
            out.push_str("## Todos\n\n");
            for t in &self.todos {
                out.push_str(&format!("- [ ] {}\n", t));
            }
            out.push('\n');
        }

        if !self.blockers.is_empty() {
            out.push_str("## Blockers\n\n");
            for b in &self.blockers {
                out.push_str(&format!("- {}\n", b));
            }
            out.push('\n');
        }

        if !self.mental_model.is_empty() {
            out.push_str("## Mental Model\n\n");
            out.push_str(&self.mental_model);
            out.push_str("\n\n");
        }

        if !self.next_steps.is_empty() {
            out.push_str("## Next Steps\n\n");
            for s in &self.next_steps {
                out.push_str(&format!("- {}\n", s));
            }
            out.push('\n');
        }

        if let Some(notes) = &self.notes
            && !notes.is_empty()
        {
            out.push_str("## Notes\n\n");
            out.push_str(notes);
            out.push_str("\n\n");
        }

        // Trim trailing whitespace
        out.trim_end().to_string()
    }

    /// Parse canonical markdown back into `HandoffSections`.
    ///
    /// Returns `MemoryError::InvalidType("handoff: malformed sections")` if the
    /// content cannot be parsed (e.g. missing Summary section).
    pub fn parse_markdown(content: &str) -> Result<HandoffSections, MemoryError> {
        let malformed = || MemoryError::InvalidType("handoff: malformed sections".to_string());

        let mut summary = String::new();
        let mut decisions: Vec<String> = Vec::new();
        let mut todos: Vec<String> = Vec::new();
        let mut blockers: Vec<String> = Vec::new();
        let mut mental_model = String::new();
        let mut next_steps: Vec<String> = Vec::new();
        let mut notes: Option<String> = None;

        #[derive(PartialEq)]
        enum Section {
            None,
            Summary,
            Decisions,
            Todos,
            Blockers,
            MentalModel,
            NextSteps,
            Notes,
        }

        let mut current = Section::None;
        let mut body_lines: Vec<&str> = Vec::new();

        let flush = |current: &Section,
                     body_lines: &[&str],
                     summary: &mut String,
                     decisions: &mut Vec<String>,
                     todos: &mut Vec<String>,
                     blockers: &mut Vec<String>,
                     mental_model: &mut String,
                     next_steps: &mut Vec<String>,
                     notes: &mut Option<String>| {
            let body = body_lines.join("\n").trim().to_string();
            match current {
                Section::Summary => *summary = body,
                Section::Decisions => {
                    *decisions = body
                        .lines()
                        .filter_map(|l| {
                            let l = l.trim();
                            l.strip_prefix("- ").map(|s| s.trim().to_string())
                        })
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                Section::Todos => {
                    *todos = body
                        .lines()
                        .filter_map(|l| {
                            let l = l.trim();
                            // Strip "- [ ] " or "- [x] " or "- "
                            let stripped = l
                                .strip_prefix("- [ ] ")
                                .or_else(|| l.strip_prefix("- [x] "))
                                .or_else(|| l.strip_prefix("- [X] "))
                                .or_else(|| l.strip_prefix("- "));
                            stripped.map(|s| s.trim().to_string())
                        })
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                Section::Blockers => {
                    *blockers = body
                        .lines()
                        .filter_map(|l| {
                            let l = l.trim();
                            l.strip_prefix("- ").map(|s| s.trim().to_string())
                        })
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                Section::MentalModel => *mental_model = body,
                Section::NextSteps => {
                    *next_steps = body
                        .lines()
                        .filter_map(|l| {
                            let l = l.trim();
                            l.strip_prefix("- ").map(|s| s.trim().to_string())
                        })
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                Section::Notes => {
                    if !body.is_empty() {
                        *notes = Some(body);
                    }
                }
                Section::None => {}
            }
        };

        for line in content.lines() {
            if let Some(heading) = line.strip_prefix("## ") {
                // Flush current section
                flush(
                    &current,
                    &body_lines,
                    &mut summary,
                    &mut decisions,
                    &mut todos,
                    &mut blockers,
                    &mut mental_model,
                    &mut next_steps,
                    &mut notes,
                );
                body_lines.clear();

                current = match heading.trim() {
                    "Summary" => Section::Summary,
                    "Decisions" => Section::Decisions,
                    "Todos" => Section::Todos,
                    "Blockers" => Section::Blockers,
                    "Mental Model" => Section::MentalModel,
                    "Next Steps" => Section::NextSteps,
                    "Notes" => Section::Notes,
                    _ => Section::None,
                };
            } else {
                body_lines.push(line);
            }
        }

        // Flush last section
        flush(
            &current,
            &body_lines,
            &mut summary,
            &mut decisions,
            &mut todos,
            &mut blockers,
            &mut mental_model,
            &mut next_steps,
            &mut notes,
        );

        if summary.is_empty() {
            return Err(malformed());
        }

        Ok(HandoffSections {
            summary,
            decisions,
            todos,
            blockers,
            mental_model,
            next_steps,
            notes,
            continues_from: None,
        })
    }
}

/// Structured sections for a `MemoryType::Adr` memory (Nygard-style).
///
/// The canonical text (stored in `memories.content`) is produced by `render_markdown`
/// and can be partially round-tripped via `parse_markdown` (status/number/date are not
/// recovered from markdown — those live in the `adr_sections` sidecar table).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Used by ADR create/update/query tools (Phase 2)
pub struct AdrSections {
    pub title: String,
    pub context: String,
    pub decision: String,
    pub consequences: String,
}

#[allow(dead_code)] // Used by ADR create/update/query tools (Phase 2)
impl AdrSections {
    /// Render sections into Nygard-style markdown stored in `memories.content`.
    pub fn render_markdown(&self, number: u32, status: AdrStatus, date_unix: i64) -> String {
        let date = chrono::DateTime::from_timestamp(date_unix, 0)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "1970-01-01".to_string());

        format!(
            "# {:04}. {}\n\n## Status\n\n{} \u{2014} {}\n\n## Context\n\n{}\n\n## Decision\n\n{}\n\n## Consequences\n\n{}",
            number, self.title, status, date, self.context, self.decision, self.consequences,
        )
    }

    /// Parse Nygard-style markdown back into `AdrSections`.
    ///
    /// Recovers `title` (from `# NNNN. {title}` heading), `context`, `decision`,
    /// and `consequences`. Status, number, and date are NOT parsed; they live in
    /// the `adr_sections` sidecar table.
    ///
    /// Returns `MemoryError::InvalidType("adr: malformed sections")` if Title or
    /// Decision is empty.
    pub fn parse_markdown(content: &str) -> Result<AdrSections, MemoryError> {
        let malformed = || MemoryError::InvalidType("adr: malformed sections".to_string());

        let mut title = String::new();
        let mut context = String::new();
        let mut decision = String::new();
        let mut consequences = String::new();

        #[derive(PartialEq)]
        enum Section {
            None,
            Status,
            Context,
            Decision,
            Consequences,
        }

        let mut current = Section::None;
        let mut body_lines: Vec<&str> = Vec::new();

        let flush = |current: &Section,
                     body_lines: &[&str],
                     context: &mut String,
                     decision: &mut String,
                     consequences: &mut String| {
            let body = body_lines.join("\n").trim().to_string();
            match current {
                Section::Context => *context = body,
                Section::Decision => *decision = body,
                Section::Consequences => *consequences = body,
                Section::Status | Section::None => {}
            }
        };

        for line in content.lines() {
            // Top-level heading: # NNNN. Title
            if let Some(heading) = line.strip_prefix("# ") {
                // Extract title after optional "NNNN. " prefix
                let raw = heading.trim();
                let extracted = if let Some(dot_pos) = raw.find(". ") {
                    let prefix = &raw[..dot_pos];
                    if prefix.chars().all(|c| c.is_ascii_digit()) {
                        raw[dot_pos + 2..].to_string()
                    } else {
                        raw.to_string()
                    }
                } else {
                    raw.to_string()
                };
                title = extracted;
                continue;
            }

            if let Some(heading) = line.strip_prefix("## ") {
                flush(
                    &current,
                    &body_lines,
                    &mut context,
                    &mut decision,
                    &mut consequences,
                );
                body_lines.clear();

                current = match heading.trim() {
                    "Status" => Section::Status,
                    "Context" => Section::Context,
                    "Decision" => Section::Decision,
                    "Consequences" => Section::Consequences,
                    _ => Section::None,
                };
            } else {
                body_lines.push(line);
            }
        }

        // Flush last section
        flush(
            &current,
            &body_lines,
            &mut context,
            &mut decision,
            &mut consequences,
        );

        if title.is_empty() || decision.is_empty() {
            return Err(malformed());
        }

        Ok(AdrSections {
            title,
            context,
            decision,
            consequences,
        })
    }
}

/// Convert a title string into a URL-friendly kebab-case slug.
///
/// Lowercases the input, keeps ASCII alphanumerics, collapses runs of other chars to a
/// single `-`, trims leading/trailing `-`, caps at 60 chars on a `-` boundary (last `-`
/// before 60), or hard-caps at 60 if no boundary exists. Returns an empty string for
/// all-punctuation input.
#[allow(dead_code)] // Used by ADR create tool (Phase 2)
pub fn kebab_title(title: &str) -> String {
    let lower = title.to_lowercase();

    // Build slug: alphanumerics pass through; anything else becomes a separator.
    let mut slug = String::new();
    let mut last_was_sep = true; // suppress leading `-`
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_was_sep = false;
        } else if !last_was_sep {
            slug.push('-');
            last_was_sep = true;
        }
    }
    // Trim trailing `-`
    let slug = slug.trim_end_matches('-').to_string();

    if slug.len() <= 60 {
        return slug;
    }

    // Cap at 60 on a `-` boundary.
    let prefix = &slug[..60];
    if let Some(pos) = prefix.rfind('-') {
        slug[..pos].to_string()
    } else {
        prefix.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    RelatesTo,
    Supersedes,
    DerivedFrom,
}

impl RelationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RelatesTo => "relates_to",
            Self::Supersedes => "supersedes",
            Self::DerivedFrom => "derived_from",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseRelationTypeError(pub String);

impl fmt::Display for ParseRelationTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid relation type: {}", self.0)
    }
}

impl std::error::Error for ParseRelationTypeError {}

impl FromStr for RelationType {
    type Err = ParseRelationTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "relates_to" => Ok(Self::RelatesTo),
            "supersedes" => Ok(Self::Supersedes),
            "derived_from" => Ok(Self::DerivedFrom),
            _ => Err(ParseRelationTypeError(s.to_string())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSource {
    pub id: String,
    pub content_preview: String,
    pub merged_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub project_id: String,
    pub memory_type: MemoryType,
    pub content: String,
    pub summary: Option<String>,
    pub tags: Vec<String>,
    pub importance: f64,
    pub relevance_score: f64,
    pub access_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
    pub last_accessed_at: i64,
    /// Git branch this memory belongs to. NULL means global (visible on all branches).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// Provenance tracking: IDs and previews of memories that were merged into this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_from: Option<Vec<MergeSource>>,
    /// Optional list of external artifact references (file paths, URLs, ticket IDs).
    /// Surfaced at retrieval; local-looking paths are checked for existence and marked
    /// `[missing]` if absent on the server's filesystem.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_artifacts: Option<Vec<String>>,
    /// Whether this memory is pinned (exempt from decay and auto-prune).
    #[serde(default)]
    pub pinned: bool,
    /// Whether this memory is visible across all projects.
    #[serde(default)]
    pub global: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation_type: RelationType,
    pub strength: f64,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub root_path: Option<String>,
    pub decay_rate: f64,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Used by MCP server tools
pub struct MemoryWithScore {
    pub memory: Memory,
    /// Combined final score after tag boost and relevance scaling.
    pub score: f64,
    /// Raw cosine similarity from the embedding model (0.0-1.0). Diagnostic.
    pub semantic_score: f64,
    /// Raw normalized BM25 keyword score (0.0 if no match). Diagnostic.
    pub keyword_score: f64,
    /// Fused RRF score before tag boost and relevance scaling (0.0 in Vector/Bm25 modes).
    #[serde(default)]
    pub rrf_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Used by MCP server and CLI
pub struct ProjectStats {
    pub memory_count: usize,
    pub relationship_count: usize,
    pub avg_relevance: f64,
    pub pinned_count: usize,
    pub global_count: usize,
    /// Number of `MemoryType::Handoff` memories in this project.
    pub handoff_count: usize,
    /// Unix timestamp of the most recent handoff in this project, or `None` if no handoffs exist.
    pub latest_handoff_at: Option<i64>,
    /// Number of `MemoryType::Adr` memories in this project.
    #[serde(default)]
    pub adr_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCluster {
    pub id: String,
    pub project_id: String,
    pub summary: String,
    pub member_count: usize,
    pub centroid: Option<Vec<f32>>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_sections() -> HandoffSections {
        HandoffSections {
            summary: "Session went well".to_string(),
            decisions: vec!["Use Rust".to_string(), "SQLite for storage".to_string()],
            todos: vec!["Write more tests".to_string()],
            blockers: vec!["Awaiting review".to_string()],
            mental_model: "Layered architecture with clear boundaries".to_string(),
            next_steps: vec!["Deploy to staging".to_string()],
            notes: Some("Remember to update the README".to_string()),
            continues_from: Some("prev-id".to_string()),
        }
    }

    #[test]
    fn test_render_markdown_includes_all_sections() {
        let s = full_sections();
        let md = s.render_markdown();
        assert!(md.contains("## Summary"), "missing Summary header");
        assert!(md.contains("Session went well"), "missing summary body");
        assert!(md.contains("## Decisions"), "missing Decisions header");
        assert!(md.contains("- Use Rust"), "missing decision");
        assert!(md.contains("## Todos"), "missing Todos header");
        assert!(md.contains("- [ ] Write more tests"), "missing todo");
        assert!(md.contains("## Blockers"), "missing Blockers header");
        assert!(md.contains("- Awaiting review"), "missing blocker");
        assert!(
            md.contains("## Mental Model"),
            "missing Mental Model header"
        );
        assert!(md.contains("## Next Steps"), "missing Next Steps header");
        assert!(md.contains("## Notes"), "missing Notes header");
    }

    #[test]
    fn test_render_skips_empty_sections() {
        let s = HandoffSections {
            summary: "Minimal".to_string(),
            decisions: vec![],
            todos: vec![],
            blockers: vec![],
            mental_model: String::new(),
            next_steps: vec![],
            notes: None,
            continues_from: None,
        };
        let md = s.render_markdown();
        assert!(md.contains("## Summary"));
        assert!(
            !md.contains("## Decisions"),
            "empty decisions should be omitted"
        );
        assert!(!md.contains("## Todos"), "empty todos should be omitted");
        assert!(
            !md.contains("## Mental Model"),
            "empty mental model should be omitted"
        );
        assert!(!md.contains("## Notes"), "None notes should be omitted");
    }

    #[test]
    fn test_parse_markdown_round_trip() {
        let original = full_sections();
        let md = original.render_markdown();
        let parsed = HandoffSections::parse_markdown(&md).unwrap();

        assert_eq!(parsed.summary, original.summary);
        assert_eq!(parsed.decisions, original.decisions);
        assert_eq!(parsed.todos, original.todos);
        assert_eq!(parsed.blockers, original.blockers);
        assert_eq!(parsed.mental_model, original.mental_model);
        assert_eq!(parsed.next_steps, original.next_steps);
        assert_eq!(parsed.notes, original.notes);
        // continues_from is sidecar-only, not encoded in markdown
        assert_eq!(parsed.continues_from, None);
    }

    #[test]
    fn test_parse_markdown_missing_summary_returns_error() {
        let md = "## Decisions\n\n- Some decision\n";
        let result = HandoffSections::parse_markdown(md);
        assert!(
            matches!(result, Err(MemoryError::InvalidType(_))),
            "expected InvalidType error for missing summary"
        );
    }

    #[test]
    fn test_memory_type_handoff_roundtrip() {
        let mt = MemoryType::Handoff;
        assert_eq!(mt.as_str(), "handoff");
        assert_eq!(mt.to_string(), "handoff");
        let parsed: MemoryType = "handoff".parse().unwrap();
        assert_eq!(parsed, MemoryType::Handoff);
    }

    #[test]
    fn test_memory_type_unknown_returns_error() {
        let result: Result<MemoryType, _> = "unknown_type".parse();
        assert!(result.is_err());
    }

    #[test]
    fn memory_type_adr_roundtrip() {
        let mt = MemoryType::Adr;
        assert_eq!(mt.as_str(), "adr");
        assert_eq!(mt.to_string(), "adr");
        let parsed: MemoryType = "adr".parse().unwrap();
        assert_eq!(parsed, MemoryType::Adr);
    }

    #[test]
    fn adr_status_from_str_and_display() {
        for (s, expected) in [
            ("proposed", AdrStatus::Proposed),
            ("accepted", AdrStatus::Accepted),
            ("superseded", AdrStatus::Superseded),
            ("deprecated", AdrStatus::Deprecated),
            ("rejected", AdrStatus::Rejected),
        ] {
            let parsed: AdrStatus = s.parse().unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.to_string(), s);
        }

        let err = "unknown".parse::<AdrStatus>();
        assert!(err.is_err());
    }

    #[test]
    fn adr_status_transition_rules() {
        // Idempotent
        for status in [
            AdrStatus::Proposed,
            AdrStatus::Accepted,
            AdrStatus::Superseded,
            AdrStatus::Deprecated,
            AdrStatus::Rejected,
        ] {
            assert!(
                status.can_transition_to(status),
                "{status} -> {status} should be allowed"
            );
        }

        // Allowed transitions
        assert!(AdrStatus::Proposed.can_transition_to(AdrStatus::Accepted));
        assert!(AdrStatus::Proposed.can_transition_to(AdrStatus::Rejected));
        assert!(AdrStatus::Proposed.can_transition_to(AdrStatus::Deprecated));
        assert!(AdrStatus::Accepted.can_transition_to(AdrStatus::Deprecated));
        assert!(AdrStatus::Accepted.can_transition_to(AdrStatus::Superseded));
        assert!(AdrStatus::Rejected.can_transition_to(AdrStatus::Proposed));
        assert!(AdrStatus::Deprecated.can_transition_to(AdrStatus::Accepted));

        // Denied transitions
        assert!(!AdrStatus::Proposed.can_transition_to(AdrStatus::Superseded));
        assert!(!AdrStatus::Accepted.can_transition_to(AdrStatus::Proposed));
        assert!(!AdrStatus::Accepted.can_transition_to(AdrStatus::Rejected));
        assert!(!AdrStatus::Rejected.can_transition_to(AdrStatus::Accepted));
        assert!(!AdrStatus::Rejected.can_transition_to(AdrStatus::Deprecated));
        assert!(!AdrStatus::Rejected.can_transition_to(AdrStatus::Superseded));
        assert!(!AdrStatus::Deprecated.can_transition_to(AdrStatus::Proposed));
        assert!(!AdrStatus::Deprecated.can_transition_to(AdrStatus::Rejected));
        assert!(!AdrStatus::Deprecated.can_transition_to(AdrStatus::Superseded));
        assert!(!AdrStatus::Superseded.can_transition_to(AdrStatus::Proposed));
        assert!(!AdrStatus::Superseded.can_transition_to(AdrStatus::Accepted));
        assert!(!AdrStatus::Superseded.can_transition_to(AdrStatus::Rejected));
        assert!(!AdrStatus::Superseded.can_transition_to(AdrStatus::Deprecated));
    }

    #[test]
    fn adr_sections_render_parse_round_trip() {
        let sections = AdrSections {
            title: "Use SQLite for storage".to_string(),
            context: "We need a lightweight embedded database.".to_string(),
            decision: "We will use SQLite via rusqlite.".to_string(),
            consequences: "Simple deployment; limited concurrency.".to_string(),
        };
        let md = sections.render_markdown(1, AdrStatus::Accepted, 1_700_000_000);
        let parsed = AdrSections::parse_markdown(&md).unwrap();

        assert_eq!(parsed.title, sections.title);
        assert_eq!(parsed.context, sections.context);
        assert_eq!(parsed.decision, sections.decision);
        assert_eq!(parsed.consequences, sections.consequences);
    }

    #[test]
    fn adr_render_includes_status_and_date() {
        let sections = AdrSections {
            title: "Choose database".to_string(),
            context: "We need persistence.".to_string(),
            decision: "Use SQLite.".to_string(),
            consequences: "Simple.".to_string(),
        };
        // 2023-11-14 22:13:20 UTC
        let md = sections.render_markdown(42, AdrStatus::Accepted, 1_700_000_000);

        assert!(
            md.contains("# 0042. Choose database"),
            "heading missing: {md}"
        );
        assert!(md.contains("## Status"), "status section missing");
        assert!(md.contains("accepted"), "status value missing");
        assert!(md.contains("2023-11-14"), "date missing");
        assert!(md.contains("\u{2014}"), "em-dash separator missing");
        assert!(md.contains("## Context"), "context section missing");
        assert!(md.contains("## Decision"), "decision section missing");
        assert!(
            md.contains("## Consequences"),
            "consequences section missing"
        );
    }

    #[test]
    fn adr_parse_missing_decision_errors() {
        let md = "# 0001. Some ADR\n\n## Status\n\nproposed\n\n## Context\n\nsome context\n";
        let result = AdrSections::parse_markdown(md);
        assert!(
            matches!(result, Err(MemoryError::InvalidType(_))),
            "expected InvalidType for missing decision"
        );
    }

    #[test]
    fn kebab_title_spaces_punct_length_and_empty() {
        // Basic conversion
        assert_eq!(kebab_title("Hello World"), "hello-world");

        // Punctuation collapse
        assert_eq!(kebab_title("Foo, Bar! Baz"), "foo-bar-baz");

        // Leading/trailing separators trimmed
        assert_eq!(kebab_title("  hello  "), "hello");

        // All-punctuation returns empty
        assert_eq!(kebab_title("!!!"), "");

        // Empty string
        assert_eq!(kebab_title(""), "");

        // Length cap at 60-char boundary
        // "aaaa-bbbb" repeated until > 60 chars
        let long = "word-".repeat(15); // 75 chars
        let slug = kebab_title(&long);
        assert!(slug.len() <= 60, "slug too long: {}", slug.len());
        assert!(!slug.ends_with('-'), "slug should not end with '-'");

        // No '-' boundary before 60: hard cap at 60
        let no_boundary = "a".repeat(70);
        let slug2 = kebab_title(&no_boundary);
        assert_eq!(slug2.len(), 60);
    }
}
