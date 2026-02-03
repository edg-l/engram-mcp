use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Fact,
    Decision,
    Preference,
    Pattern,
    Debug,
    Entity,
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
        }
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
            _ => Err(ParseMemoryTypeError(s.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    RelatesTo,
    Supersedes,
    DerivedFrom,
    Contradicts,
}

impl RelationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RelatesTo => "relates_to",
            Self::Supersedes => "supersedes",
            Self::DerivedFrom => "derived_from",
            Self::Contradicts => "contradicts",
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
            "contradicts" => Ok(Self::Contradicts),
            _ => Err(ParseRelationTypeError(s.to_string())),
        }
    }
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
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Used by MCP server and CLI
pub struct ProjectStats {
    pub memory_count: usize,
    pub relationship_count: usize,
    pub avg_relevance: f64,
}
