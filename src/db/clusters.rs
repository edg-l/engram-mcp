use rusqlite::params;

use crate::error::MemoryError;
use crate::memory::MemoryCluster;

use super::Database;

impl Database {
    /// Create a new memory cluster.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn create_cluster(&self, cluster: &MemoryCluster) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let centroid_bytes: Option<Vec<u8>> = cluster
            .centroid
            .as_ref()
            .map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect());
        conn.execute(
            "INSERT INTO memory_clusters (id, project_id, summary, member_count, centroid, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                cluster.id,
                cluster.project_id,
                cluster.summary,
                cluster.member_count as i64,
                centroid_bytes,
                cluster.created_at,
                cluster.updated_at,
            ],
        )?;
        Ok(())
    }

    /// Add a memory to a cluster.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn add_to_cluster(&self, cluster_id: &str, memory_id: &str) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO cluster_members (cluster_id, memory_id) VALUES (?1, ?2)",
            params![cluster_id, memory_id],
        )?;
        conn.execute(
            "UPDATE memory_clusters SET member_count = (SELECT COUNT(*) FROM cluster_members WHERE cluster_id = ?1), updated_at = ?2 WHERE id = ?1",
            params![cluster_id, chrono::Utc::now().timestamp()],
        )?;
        Ok(())
    }

    /// Remove a memory from its cluster.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn remove_from_cluster(&self, memory_id: &str) -> Result<Option<String>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        // Get the cluster_id before removing
        let cluster_id: Option<String> = conn
            .query_row(
                "SELECT cluster_id FROM cluster_members WHERE memory_id = ?1",
                params![memory_id],
                |row| row.get(0),
            )
            .ok();

        if let Some(ref cid) = cluster_id {
            conn.execute(
                "DELETE FROM cluster_members WHERE memory_id = ?1",
                params![memory_id],
            )?;
            // Update member count
            let now = chrono::Utc::now().timestamp();
            conn.execute(
                "UPDATE memory_clusters SET member_count = (SELECT COUNT(*) FROM cluster_members WHERE cluster_id = ?1), updated_at = ?2 WHERE id = ?1",
                params![cid, now],
            )?;
        }

        Ok(cluster_id)
    }

    /// Get a cluster by ID.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn get_cluster(&self, id: &str) -> Result<Option<MemoryCluster>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, project_id, summary, member_count, centroid, created_at, updated_at FROM memory_clusters WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            let centroid_bytes: Option<Vec<u8>> = row.get(4)?;
            let centroid = centroid_bytes.map(|bytes| {
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            });
            Ok(Some(MemoryCluster {
                id: row.get(0)?,
                project_id: row.get(1)?,
                summary: row.get(2)?,
                member_count: row.get::<_, i64>(3)? as usize,
                centroid,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get all clusters for a project.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn get_clusters_for_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<MemoryCluster>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, project_id, summary, member_count, centroid, created_at, updated_at FROM memory_clusters WHERE project_id = ?1"
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            let centroid_bytes: Option<Vec<u8>> = row.get(4)?;
            let centroid = centroid_bytes.map(|bytes| {
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            });
            Ok(MemoryCluster {
                id: row.get(0)?,
                project_id: row.get(1)?,
                summary: row.get(2)?,
                member_count: row.get::<_, i64>(3)? as usize,
                centroid,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Update a cluster's centroid.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn update_cluster_centroid(
        &self,
        id: &str,
        centroid: &[f32],
        summary: &str,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let centroid_bytes: Vec<u8> = centroid.iter().flat_map(|f| f.to_le_bytes()).collect();
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "UPDATE memory_clusters SET centroid = ?1, summary = ?2, updated_at = ?3 WHERE id = ?4",
            params![centroid_bytes, summary, now, id],
        )?;
        Ok(())
    }

    /// Delete clusters with no members.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn delete_empty_clusters(&self, project_id: &str) -> Result<usize, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let rows = conn.execute(
            "DELETE FROM memory_clusters WHERE project_id = ?1 AND NOT EXISTS (SELECT 1 FROM cluster_members WHERE cluster_id = memory_clusters.id)",
            params![project_id],
        )?;
        Ok(rows)
    }

    /// Get all memory IDs in a cluster.
    #[allow(dead_code)] // Used by clustering pipeline
    pub fn get_cluster_member_ids(&self, cluster_id: &str) -> Result<Vec<String>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT memory_id FROM cluster_members WHERE cluster_id = ?1")?;
        let rows = stmt.query_map(params![cluster_id], |row| row.get(0))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }
}
