//! Merging projects, and reconciling a directory-derived project id with the ids the
//! same directory was known by earlier.
//!
//! A repo's id changes whenever its remote does (`~/dev/foo` becomes
//! `git:host/owner/foo` when a remote is added, `git:old/…` becomes `git:new/…` when
//! it moves), which orphans every memory stored under the earlier id. The directory is
//! what stays put, so `projects.root_path` records it and reconciliation folds any
//! project claiming the same directory into the current id.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use rusqlite::{Connection, OptionalExtension, params};

use crate::error::MemoryError;
use crate::project::ProjectIdentity;

use super::Database;

/// Decay rate given to a `projects` row created here, matching `get_or_create_project`.
const DEFAULT_DECAY_RATE: f64 = 0.01;

/// Rows one source project contributed to a merge.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectMerge {
    pub from: String,
    pub to: String,
    pub memories: usize,
    pub adrs: usize,
    pub trash: usize,
    pub clusters: usize,
    /// Whether `from` had its own `projects` row, now folded into `to`'s.
    pub project_row: bool,
}

/// Outcome of merging a set of projects.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub merges: Vec<ProjectMerge>,
    /// ADRs whose number changed to keep `UNIQUE(project_id, adr_number)` in a merged
    /// project; counted across every source and the target.
    pub adrs_renumbered: usize,
}

impl MergeReport {
    pub fn is_empty(&self) -> bool {
        self.merges.is_empty()
    }
}

impl fmt::Display for MergeReport {
    /// One line naming every source, its target, and what moved.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .merges
            .iter()
            .map(|m| {
                format!(
                    "'{}' into '{}' ({} memories, {} ADRs, {} trash entries, {} clusters)",
                    m.from, m.to, m.memories, m.adrs, m.trash, m.clusters
                )
            })
            .collect();
        write!(f, "merged project {}", parts.join("; "))?;
        if self.adrs_renumbered > 0 {
            write!(f, "; {} ADRs renumbered", self.adrs_renumbered)?;
        }
        Ok(())
    }
}

fn invalid_merge(from: &str, to: &str, message: &str) -> MemoryError {
    MemoryError::InvalidArguments {
        tool: "projects merge".to_string(),
        message: message.to_string(),
        received: format!("from={from}, to={to}"),
    }
}

/// Merge every `from` project in `map` into its `to`, on `conn`, which must be inside a
/// transaction the caller commits or rolls back.
///
/// Covers every table that carries a project id: `projects` (the row itself),
/// `memories`, `adr_sections`, `memory_trash` (column and the snapshot payload a restore
/// reads it back from) and `memory_clusters`. Every other sidecar keys on `memory_id`
/// and follows its memory without a rewrite. Each source is also recorded in
/// `project_aliases` so an import still labelled with a merged-away id lands in the
/// survivor instead of recreating it.
///
/// ADR numbers are unique per project, so a target fed by two or more projects that own
/// ADRs is renumbered 1..n in `(created_at, memory_id)` order; a target with a single
/// ADR-owning source keeps its numbering, gaps included.
///
/// `projects` rows: the target keeps its own row when it has one, taking the earliest
/// `created_at` and, when it has none, the source's `root_path`. A target with no row
/// gets one built from the first source (in id order) that has one.
pub(super) fn merge_projects_in(
    conn: &Connection,
    map: &BTreeMap<String, String>,
) -> Result<MergeReport, MemoryError> {
    for (from, to) in map {
        if from == to || map.contains_key(to) {
            return Err(invalid_merge(
                from,
                to,
                "a merge target must be a different project that is not itself being \
                 merged away",
            ));
        }
    }
    let target_of = |pid: &str| -> String { map.get(pid).cloned().unwrap_or_else(|| pid.into()) };

    // Negative temporary numbers first, so the project_id rewrite below cannot trip
    // UNIQUE(project_id, adr_number) mid-flight.
    let mut original_numbers: HashMap<String, i64> = HashMap::new();
    {
        let mut stmt =
            conn.prepare("SELECT memory_id, project_id, adr_number, created_at FROM adr_sections")?;
        let rows: Vec<(String, String, i64, i64)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);

        let mut groups: BTreeMap<String, Vec<(String, String, i64, i64)>> = BTreeMap::new();
        for row in rows {
            groups.entry(target_of(&row.1)).or_default().push(row);
        }

        for group in groups.values_mut() {
            let distinct_sources: BTreeSet<&String> =
                group.iter().map(|(_, pid, _, _)| pid).collect();
            if distinct_sources.len() < 2 {
                continue;
            }
            group.sort_by(|a, b| a.3.cmp(&b.3).then_with(|| a.0.cmp(&b.0)));
            for (k, (memory_id, _, number, _)) in group.iter().enumerate() {
                original_numbers.insert(memory_id.clone(), *number);
                conn.execute(
                    "UPDATE adr_sections SET adr_number = ?1 WHERE memory_id = ?2",
                    params![-(k as i64 + 1), memory_id],
                )?;
            }
        }
    }

    let mut merges = Vec::with_capacity(map.len());
    for (from, to) in map {
        let source_row: Option<(Option<String>, f64, i64)> = conn
            .query_row(
                "SELECT root_path, decay_rate, created_at FROM projects WHERE id = ?1",
                params![from],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if let Some((root_path, decay_rate, created_at)) = &source_row {
            let updated = conn.execute(
                "UPDATE projects SET created_at = MIN(created_at, ?2), \
                 root_path = COALESCE(root_path, ?3) WHERE id = ?1",
                params![to, created_at, root_path],
            )?;
            if updated == 0 {
                conn.execute(
                    "INSERT INTO projects (id, name, root_path, decay_rate, created_at) \
                     VALUES (?1, ?1, ?2, ?3, ?4)",
                    params![to, root_path, decay_rate, created_at],
                )?;
            }
            conn.execute("DELETE FROM projects WHERE id = ?1", params![from])?;
        }

        let rewrite = |table: &str| -> rusqlite::Result<usize> {
            conn.execute(
                &format!("UPDATE {table} SET project_id = ?1 WHERE project_id = ?2"),
                params![to, from],
            )
        };
        let memories = rewrite("memories")?;
        let adrs = rewrite("adr_sections")?;
        let clusters = rewrite("memory_clusters")?;
        // `restore_trash_entry` reinserts the snapshot's own `memory.project_id`, so the
        // payload has to move with the column or a restore would recreate `from`.
        let trash = conn.execute(
            "UPDATE memory_trash SET project_id = ?1, \
             payload = CASE WHEN json_valid(payload) \
                 THEN json_set(payload, '$.memory.project_id', ?1) ELSE payload END \
             WHERE project_id = ?2",
            params![to, from],
        )?;

        merges.push(ProjectMerge {
            from: from.clone(),
            to: to.clone(),
            memories,
            adrs,
            trash,
            clusters,
            project_row: source_row.is_some(),
        });
    }

    // Final numbering, keyed by the rewritten project_id. Sorting the temporary negative
    // numbers descending recovers the (created_at, memory_id) order, since -1 > -2 > …
    let mut adrs_renumbered = 0;
    {
        let mut stmt = conn.prepare(
            "SELECT memory_id, project_id, adr_number FROM adr_sections WHERE adr_number < 0",
        )?;
        let rows: Vec<(String, String, i64)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);

        let mut groups: BTreeMap<String, Vec<(String, i64)>> = BTreeMap::new();
        for (memory_id, project_id, adr_number) in rows {
            groups
                .entry(project_id)
                .or_default()
                .push((memory_id, adr_number));
        }
        for group in groups.values_mut() {
            group.sort_by_key(|a| std::cmp::Reverse(a.1));
            for (k, (memory_id, _)) in group.iter().enumerate() {
                let number = k as i64 + 1;
                conn.execute(
                    "UPDATE adr_sections SET adr_number = ?1 WHERE memory_id = ?2",
                    params![number, memory_id],
                )?;
                if original_numbers.get(memory_id) != Some(&number) {
                    adrs_renumbered += 1;
                }
            }
        }
    }

    // Aliases pointing at a merged-away id follow it to its new target, so an alias never
    // names a project that no longer exists; a target is live again, so it stops being
    // an alias for anything.
    let now = chrono::Utc::now().timestamp();
    for (from, to) in map {
        conn.execute(
            "UPDATE project_aliases SET project_id = ?1 WHERE project_id = ?2",
            params![to, from],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO project_aliases (alias, project_id, merged_at) \
             VALUES (?1, ?2, ?3)",
            params![from, to, now],
        )?;
    }
    for to in map.values().collect::<BTreeSet<_>>() {
        conn.execute("DELETE FROM project_aliases WHERE alias = ?1", params![to])?;
    }

    Ok(MergeReport {
        merges,
        adrs_renumbered,
    })
}

impl Database {
    /// Merge project `from` into `to`. With `apply = false` the identical merge runs
    /// inside a transaction that is rolled back, so a dry run reports exactly what
    /// applying would do.
    ///
    /// Both ids must be known to the store (see [`Database::project_exists`]) and differ.
    #[allow(dead_code)] // Used by engram-cli, not the engram MCP-server binary
    pub fn merge_project(
        &self,
        from: &str,
        to: &str,
        apply: bool,
    ) -> Result<MergeReport, MemoryError> {
        if from == to {
            return Err(invalid_merge(
                from,
                to,
                "a project cannot be merged into itself",
            ));
        }
        for id in [from, to] {
            if !self.project_exists(id)? {
                let known: Vec<String> = self.list_projects()?.into_iter().map(|p| p.id).collect();
                return Err(MemoryError::UnknownProject {
                    requested: id.to_string(),
                    known: known.join(", "),
                });
            }
        }

        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let map = BTreeMap::from([(from.to_string(), to.to_string())]);
        let report = merge_projects_in(&tx, &map)?;
        if apply {
            tx.commit()?;
        }
        Ok(report)
    }

    /// Bind the directory-derived project `id` to its directory `root` (the
    /// `home_relative` form of its git root, or of the directory itself), merging into
    /// `id` every other project the same directory was known by: one whose `root_path`
    /// is `root` (the remote changed) or whose id is `root` (a remote was added to a
    /// repo that had none). One transaction; prints a one-line note to stderr when
    /// anything was merged.
    ///
    /// Runs on every `engram-cli` invocation and hook event, so the common case is two
    /// reads and no write: a scan of `projects` (one row per project, no index needed)
    /// and an indexed `EXISTS` on `memories.project_id`. It writes only when something
    /// needs merging or `root_path` is not yet `root`.
    pub fn reconcile_project_root(&self, id: &str, root: &str) -> Result<MergeReport, MemoryError> {
        let mut conn = self.conn.lock().unwrap();

        let recorded_root: Option<Option<String>> = conn
            .query_row(
                "SELECT root_path FROM projects WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?;

        let mut others: BTreeSet<String> = {
            let mut stmt = conn.prepare(
                "SELECT id FROM projects WHERE id <> ?1 AND (root_path = ?2 OR id = ?2)",
            )?;
            stmt.query_map(params![id, root], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<_>>()?
        };
        // A project can be known only through its memories, with no `projects` row.
        if root != id && !others.contains(root) {
            let has_memories: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM memories WHERE project_id = ?1)",
                params![root],
                |row| row.get(0),
            )?;
            if has_memories {
                others.insert(root.to_string());
            }
        }

        if others.is_empty() && recorded_root.flatten().as_deref() == Some(root) {
            return Ok(MergeReport::default());
        }

        let tx = conn.transaction()?;
        let map: BTreeMap<String, String> = others
            .into_iter()
            .map(|other| (other, id.to_string()))
            .collect();
        let report = if map.is_empty() {
            MergeReport::default()
        } else {
            merge_projects_in(&tx, &map)?
        };
        tx.execute(
            "INSERT INTO projects (id, name, root_path, decay_rate, created_at) \
             VALUES (?1, ?1, ?2, ?3, ?4) \
             ON CONFLICT(id) DO UPDATE SET root_path = excluded.root_path",
            params![id, root, DEFAULT_DECAY_RATE, chrono::Utc::now().timestamp()],
        )?;
        tx.commit()?;

        if !report.is_empty() {
            eprintln!("[engram] {report}");
        }
        Ok(report)
    }

    /// The entry point every binary calls after resolving its project: runs
    /// [`Database::reconcile_project_root`] when the id was derived from a directory,
    /// and does nothing for an explicit `--project`/`ENGRAM_PROJECT` id, which names a
    /// project rather than a place.
    ///
    /// A failure is reported on stderr instead of returned: the resolved id is valid
    /// either way, and the next invocation retries.
    pub fn reconcile_identity(&self, identity: &ProjectIdentity) {
        let Some(root) = &identity.root else {
            return;
        };
        if let Err(error) = self.reconcile_project_root(&identity.id, root) {
            eprintln!(
                "[engram] could not reconcile project '{}' with directory '{root}': {error}",
                identity.id
            );
        }
    }

    /// Every merged-away project id mapped to the project it was merged into.
    #[allow(dead_code)] // Used by engram-cli import, not the engram MCP-server binary
    pub fn project_aliases(&self) -> Result<HashMap<String, String>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT alias, project_id FROM project_aliases")?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<HashMap<String, String>>>()?;
        Ok(rows)
    }
}
