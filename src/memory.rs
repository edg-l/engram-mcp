use serde::{Deserialize, Serialize};

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

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "fact" => Some(Self::Fact),
            "decision" => Some(Self::Decision),
            "preference" => Some(Self::Preference),
            "pattern" => Some(Self::Pattern),
            "debug" => Some(Self::Debug),
            "entity" => Some(Self::Entity),
            _ => None,
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

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "relates_to" => Some(Self::RelatesTo),
            "supersedes" => Some(Self::Supersedes),
            "derived_from" => Some(Self::DerivedFrom),
            "contradicts" => Some(Self::Contradicts),
            _ => None,
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
pub struct MemoryWithScore {
    pub memory: Memory,
    pub score: f64,
}
