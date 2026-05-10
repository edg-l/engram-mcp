use std::collections::HashMap;

use rusqlite::params;

use crate::error::MemoryError;
use crate::memory::{RelationType, Relationship};

use super::Database;

impl Database {
    // Relationship operations
    pub fn create_relationship(&self, rel: &Relationship) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO relationships (id, source_id, target_id, relation_type, strength, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                rel.id,
                rel.source_id,
                rel.target_id,
                rel.relation_type.as_str(),
                rel.strength,
                rel.created_at,
            ],
        )?;
        Ok(())
    }

    #[allow(dead_code)] // Used by CLI binary
    pub fn get_relationships_from(
        &self,
        source_id: &str,
    ) -> Result<Vec<Relationship>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, relation_type, strength, created_at FROM relationships WHERE source_id = ?1"
        )?;
        let rows = stmt.query_map(params![source_id], |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    #[allow(dead_code)] // Used by CLI binary
    pub fn get_relationships_to(&self, target_id: &str) -> Result<Vec<Relationship>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, relation_type, strength, created_at FROM relationships WHERE target_id = ?1"
        )?;
        let rows = stmt.query_map(params![target_id], |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Get relationships from multiple source IDs in a single query.
    /// Returns a HashMap from source_id to Vec<Relationship>.
    pub fn get_relationships_from_batch(
        &self,
        source_ids: &[String],
    ) -> Result<HashMap<String, Vec<Relationship>>, MemoryError> {
        if source_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let conn = self.conn.lock().unwrap();

        let placeholders: Vec<&str> = source_ids.iter().map(|_| "?").collect();
        let sql = format!(
            "SELECT id, source_id, target_id, relation_type, strength, created_at FROM relationships WHERE source_id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> = source_ids
            .iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();

        let rows = stmt.query_map(params.as_slice(), |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        let mut result: HashMap<String, Vec<Relationship>> = HashMap::new();
        for rel in rows.flatten() {
            result.entry(rel.source_id.clone()).or_default().push(rel);
        }

        Ok(result)
    }

    /// Get relationships to multiple target IDs in a single query.
    /// Returns a HashMap from target_id to Vec<Relationship>.
    pub fn get_relationships_to_batch(
        &self,
        target_ids: &[String],
    ) -> Result<HashMap<String, Vec<Relationship>>, MemoryError> {
        if target_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let conn = self.conn.lock().unwrap();

        let placeholders: Vec<&str> = target_ids.iter().map(|_| "?").collect();
        let sql = format!(
            "SELECT id, source_id, target_id, relation_type, strength, created_at FROM relationships WHERE target_id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> = target_ids
            .iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();

        let rows = stmt.query_map(params.as_slice(), |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        let mut result: HashMap<String, Vec<Relationship>> = HashMap::new();
        for rel in rows.flatten() {
            result.entry(rel.target_id.clone()).or_default().push(rel);
        }

        Ok(result)
    }

    /// Get all relationships for a project
    pub fn get_all_relationships_for_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<Relationship>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT r.id, r.source_id, r.target_id, r.relation_type, r.strength, r.created_at
             FROM relationships r
             JOIN memories m ON r.source_id = m.id
             WHERE m.project_id = ?1",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            let rel_type_str: String = row.get(3)?;
            Ok(Relationship {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relation_type: rel_type_str.parse().unwrap_or(RelationType::RelatesTo),
                strength: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }
}
