use std::collections::{BTreeMap, BTreeSet};

use rusqlite::params;

use crate::error::MemoryError;

use super::Database;

impl Database {
    /// Add branch column to memories table if it doesn't exist.
    pub(super) fn migrate_branch_column(&self) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();

        // Check if branch column already exists
        let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
        let has_branch = stmt
            .query_map([], |row| {
                let name: String = row.get(1)?;
                Ok(name)
            })?
            .filter_map(|r| r.ok())
            .any(|name| name == "branch");

        if !has_branch {
            conn.execute_batch(
                r#"
                ALTER TABLE memories ADD COLUMN branch TEXT;
                CREATE INDEX IF NOT EXISTS idx_memories_project_branch ON memories(project_id, branch);
                "#,
            )?;
        }

        Ok(())
    }

    /// Migrate existing memories to FTS index if empty.
    pub(super) fn migrate_fts(&self) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();

        // Check if FTS table is empty
        let fts_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))?;

        if fts_count == 0 {
            // Check if there are memories to migrate
            let memory_count: i64 =
                conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

            if memory_count > 0 {
                // Populate FTS from existing memories
                conn.execute(
                    "INSERT INTO memories_fts(rowid, content, summary, tags)
                     SELECT rowid, content, summary, tags FROM memories",
                    [],
                )?;
            }
        }

        Ok(())
    }

    pub(super) fn run_migrations(&self) -> Result<(), MemoryError> {
        let mut conn = self.conn.lock().unwrap();

        // Ensure schema_version table exists
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);",
        )?;

        let current_version: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        // Migration 1: Add merged_from, memory_clusters, cluster_members
        if current_version < 1 {
            // Check if merged_from column already exists
            let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
            let has_merged_from = stmt
                .query_map([], |row| {
                    let name: String = row.get(1)?;
                    Ok(name)
                })?
                .filter_map(|r| r.ok())
                .any(|name| name == "merged_from");

            if !has_merged_from {
                conn.execute_batch("ALTER TABLE memories ADD COLUMN merged_from TEXT;")?;
            }

            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS memory_clusters (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    member_count INTEGER NOT NULL DEFAULT 0,
                    centroid BLOB,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cluster_members (
                    cluster_id TEXT NOT NULL REFERENCES memory_clusters(id) ON DELETE CASCADE,
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    PRIMARY KEY (cluster_id, memory_id)
                );

                CREATE INDEX IF NOT EXISTS idx_clusters_project ON memory_clusters(project_id);
                CREATE INDEX IF NOT EXISTS idx_cluster_members_memory ON cluster_members(memory_id);
            "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![1],
            )?;
        }

        // Migration 2: wipe embeddings table due to dimension change (384 -> 256)
        if current_version < 2 {
            conn.execute_batch("DELETE FROM embeddings;")?;
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![2],
            )?;
        }

        // Migration 3: add pinned and global columns with partial indexes
        if current_version < 3 {
            let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
            let columns: Vec<String> = stmt
                .query_map([], |row| row.get(1))?
                .filter_map(|r| r.ok())
                .collect();

            if !columns.iter().any(|c| c == "pinned") {
                conn.execute_batch(
                    "ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;",
                )?;
            }
            if !columns.iter().any(|c| c == "global") {
                conn.execute_batch(
                    "ALTER TABLE memories ADD COLUMN global INTEGER NOT NULL DEFAULT 0;",
                )?;
            }

            conn.execute_batch(
                r#"
                CREATE INDEX IF NOT EXISTS idx_memories_global ON memories(global) WHERE global = 1;
                CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned) WHERE pinned = 1;
                "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![3],
            )?;
        }

        // Migration 4: add handoff_sections sidecar table and continuation index.
        //
        // Wire format for section_embeddings:
        //   section_embedding_keys: comma-separated section names in canonical order,
        //     omitting empty sections (matches render_markdown order).
        //   section_embeddings: concatenated little-endian f32 bytes,
        //     256 dims × N sections × 4 bytes per float.
        //   Decoder validates bytes.len() == count * 256 * 4.
        if current_version < 4 {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS handoff_sections (
                    memory_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    decisions TEXT NOT NULL,
                    todos TEXT NOT NULL,
                    blockers TEXT NOT NULL,
                    mental_model TEXT NOT NULL,
                    next_steps TEXT NOT NULL,
                    notes TEXT,
                    continues_from TEXT,
                    section_embedding_keys TEXT NOT NULL,
                    section_embeddings BLOB NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_handoff_continues
                    ON handoff_sections(continues_from);
                "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![4],
            )?;
        }

        // Migration 5: add external_artifacts column (nullable JSON array of strings).
        if current_version < 5 {
            let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
            let has_col = stmt
                .query_map([], |row| {
                    let name: String = row.get(1)?;
                    Ok(name)
                })?
                .filter_map(|r| r.ok())
                .any(|name| name == "external_artifacts");

            if !has_col {
                conn.execute_batch("ALTER TABLE memories ADD COLUMN external_artifacts TEXT;")?;
            }

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![5],
            )?;
        }

        // Migration 6: add adr_sections sidecar table for ADR memories.
        if current_version < 6 {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS adr_sections (
                    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                    project_id TEXT NOT NULL,
                    adr_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    context TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    consequences TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(project_id, adr_number)
                );
                CREATE INDEX IF NOT EXISTS idx_adr_project_number ON adr_sections(project_id, adr_number);
                CREATE INDEX IF NOT EXISTS idx_adr_status ON adr_sections(project_id, status);
                "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![6],
            )?;
        }

        // Migration 7: curation sidecars.
        //
        // `memory_status` holds retrieval status that is not part of the memory itself.
        // Supersession is read from the `supersedes` relationship edge, so the only
        // status stored here is `dead`: the subject is gone and the memory should not
        // surface, with no successor to redirect to.
        //
        // `memory_trash` holds snapshots of memories removed or overwritten by any
        // destructive operation. It deliberately has no foreign key to `memories`,
        // because its whole purpose is to outlive the row.
        if current_version < 7 {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS memory_status (
                    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                    dead INTEGER NOT NULL DEFAULT 0,
                    reason TEXT,
                    marked_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memory_status_dead
                    ON memory_status(dead) WHERE dead = 1;

                CREATE TABLE IF NOT EXISTS memory_trash (
                    trash_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    op TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    embedding BLOB,
                    model_version TEXT,
                    trashed_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trash_project
                    ON memory_trash(project_id, trashed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_trash_memory
                    ON memory_trash(memory_id, trashed_at DESC);
                "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![7],
            )?;
        }

        // Migration 8: add the `tried` handoff section (approaches attempted and abandoned).
        // Existing rows default to an empty JSON array so `get_handoff_sections` decodes
        // them unchanged.
        if current_version < 8 {
            let has_tried = conn
                .prepare("PRAGMA table_info(handoff_sections)")?
                .query_map([], |row| row.get::<_, String>(1))?
                .filter_map(|r| r.ok())
                .any(|name| name == "tried");

            if !has_tried {
                conn.execute_batch(
                    "ALTER TABLE handoff_sections ADD COLUMN tried TEXT NOT NULL DEFAULT '[]';",
                )?;
            }

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![8],
            )?;
        }

        // Migration 9: todo_items, the lifecycle sidecar for `MemoryType::Todo` memories.
        //
        // Branch scoping comes from `memories.branch` (NULL = applies to the whole project),
        // so it is not duplicated here. Closing a todo also sets `memory_status.dead`, which
        // is what keeps finished work out of query and context results.
        if current_version < 9 {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS todo_items (
                    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    reason TEXT,
                    closed_at INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_todo_items_status
                    ON todo_items(status);
                "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![9],
            )?;
        }

        // Migration 10: rewrite absolute-filesystem-path project ids to the portable
        // `crate::project` identity (normalized git remote, else `~`-relative path).
        //
        // `root_path` may no longer name a real directory by the time this runs — a repo
        // can be renamed or deleted long after it was last used — so
        // `portable_id_from_legacy` falls back to a pure string fold in that case rather
        // than requiring the directory to exist.
        //
        // Folding is lossy by design: two legacy ids can land on the same portable id (a
        // repo and its own subdirectory, or two different machines' home directories for
        // the same user). Every affected row is backed up via `VACUUM INTO` before the
        // rewrite, and merges are reported so they can be sanity-checked.
        if current_version < 10 {
            let mut legacy_ids: BTreeSet<String> = BTreeSet::new();
            for query in [
                "SELECT id FROM projects",
                "SELECT project_id FROM memories",
                "SELECT project_id FROM adr_sections",
                "SELECT project_id FROM memory_trash",
                "SELECT project_id FROM memory_clusters",
            ] {
                let mut stmt = conn.prepare(query)?;
                let ids = stmt
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                legacy_ids.extend(ids);
            }

            let mut map: BTreeMap<String, String> = BTreeMap::new();
            for legacy in &legacy_ids {
                let portable = crate::project::portable_id_from_legacy(legacy);
                if &portable != legacy {
                    map.insert(legacy.clone(), portable);
                }
            }

            if map.is_empty() {
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                    params![10],
                )?;
            } else {
                let target_of = |pid: &str| -> String {
                    map.get(pid).cloned().unwrap_or_else(|| pid.to_string())
                };

                // Group every legacy id (not just the ones that changed) by its target,
                // so the merge summary below can name every source that folded together.
                let mut source_groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
                for legacy in &legacy_ids {
                    source_groups
                        .entry(target_of(legacy))
                        .or_default()
                        .insert(legacy.clone());
                }

                // Task 2.2: back up the whole store before any rewrite. `VACUUM INTO`
                // needs a real file target, so this is skipped for in-memory connections
                // (`conn.path()` returns `Some("")` for those).
                let backup_path = match conn.path().filter(|p| !p.is_empty()) {
                    Some(db_path) => {
                        let db_path = db_path.to_string();
                        let mut backup = format!("{db_path}.pre-portable-ids.bak");
                        if std::path::Path::new(&backup).exists() {
                            backup = format!("{db_path}.{}.bak", chrono::Utc::now().timestamp());
                        }
                        conn.execute("VACUUM INTO ?1", params![backup])?;
                        Some(backup)
                    }
                    None => None,
                };

                let tx = conn.transaction()?;

                // Task 2.3(a): negative-temporary ADR renumbering, grouped by the
                // post-migration target project id. Only groups fed by more than one
                // distinct legacy project id can collide under
                // UNIQUE(project_id, adr_number); single-source groups are left alone so
                // their existing numbering (and any gaps from prior deletions) survives.
                {
                    let mut stmt = tx.prepare(
                        "SELECT memory_id, project_id, adr_number, created_at FROM adr_sections",
                    )?;
                    let rows: Vec<(String, String, i64, i64)> = stmt
                        .query_map([], |row| {
                            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                        })?
                        .collect::<rusqlite::Result<Vec<_>>>()?;
                    drop(stmt);

                    let mut groups: BTreeMap<String, Vec<(String, String, i64, i64)>> =
                        BTreeMap::new();
                    for (memory_id, project_id, adr_number, created_at) in rows {
                        let target = target_of(&project_id);
                        groups
                            .entry(target)
                            .or_default()
                            .push((memory_id, project_id, adr_number, created_at));
                    }

                    for group in groups.values_mut() {
                        let distinct_sources: BTreeSet<&String> =
                            group.iter().map(|(_, pid, _, _)| pid).collect();
                        if distinct_sources.len() < 2 {
                            continue;
                        }
                        group.sort_by(|a, b| a.3.cmp(&b.3).then_with(|| a.0.cmp(&b.0)));
                        for (k, (memory_id, _, _, _)) in group.iter().enumerate() {
                            tx.execute(
                                "UPDATE adr_sections SET adr_number = ?1 WHERE memory_id = ?2",
                                params![-(k as i64 + 1), memory_id],
                            )?;
                        }
                    }
                }

                // Task 2.4: rewrite `projects` wholesale rather than per-row, since a
                // merge would otherwise collide on the primary key mid-rewrite.
                {
                    let mut stmt =
                        tx.prepare("SELECT id, root_path, decay_rate, created_at FROM projects")?;
                    let rows: Vec<(String, Option<String>, f64, i64)> = stmt
                        .query_map([], |row| {
                            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                        })?
                        .collect::<rusqlite::Result<Vec<_>>>()?;
                    drop(stmt);

                    // new_id -> (decay_rate carried from any source row, MIN(created_at))
                    let mut merged: BTreeMap<String, (f64, i64)> = BTreeMap::new();
                    for (id, _root_path, decay_rate, created_at) in &rows {
                        let target = target_of(id);
                        merged
                            .entry(target)
                            .and_modify(|(_, min_created)| {
                                if *created_at < *min_created {
                                    *min_created = *created_at;
                                }
                            })
                            .or_insert((*decay_rate, *created_at));
                    }

                    tx.execute("DELETE FROM projects", [])?;
                    for (new_id, (decay_rate, created_at)) in &merged {
                        tx.execute(
                            "INSERT INTO projects (id, name, root_path, decay_rate, created_at) \
                             VALUES (?1, ?1, ?1, ?2, ?3)",
                            params![new_id, decay_rate, created_at],
                        )?;
                    }
                }

                // The four sidecar tables carrying a `project_id` column.
                for (legacy, new_id) in &map {
                    for table in [
                        "memories",
                        "adr_sections",
                        "memory_trash",
                        "memory_clusters",
                    ] {
                        tx.execute(
                            &format!("UPDATE {table} SET project_id = ?1 WHERE project_id = ?2"),
                            params![new_id, legacy],
                        )?;
                    }
                }

                // Task 2.3(b): final renumber pass, now keyed by the rewritten
                // project_id. Sorting the temporary negative numbers descending recovers
                // the original (created_at, memory_id) order, since -1 > -2 > -3 …
                {
                    let mut stmt = tx.prepare(
                        "SELECT memory_id, project_id, adr_number FROM adr_sections \
                         WHERE adr_number < 0",
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
                            tx.execute(
                                "UPDATE adr_sections SET adr_number = ?1 WHERE memory_id = ?2",
                                params![k as i64 + 1, memory_id],
                            )?;
                        }
                    }
                }

                tx.commit()?;

                conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                    params![10],
                )?;

                // Task 2.6: summary of what happened, and an explicit warning about the
                // one thing this migration cannot fix on its own.
                eprintln!(
                    "[engram] migration 10: rewrote {} legacy project id(s) to portable ids.",
                    map.len()
                );
                eprintln!(
                    "[engram] migration 10: backup written to {}",
                    backup_path
                        .as_deref()
                        .unwrap_or("<none, in-memory database>")
                );
                for (target, sources) in &source_groups {
                    if sources.len() > 1 {
                        let list: Vec<&str> = sources.iter().map(String::as_str).collect();
                        eprintln!(
                            "[engram] migration 10: merged into '{target}': {}",
                            list.join(", ")
                        );
                    }
                }
                eprintln!(
                    "[engram] migration 10: WARNING — a hardcoded --project/ENGRAM_PROJECT \
                     override that still points at a pre-migration legacy project id is not \
                     rewritten by this migration. Its next write will silently create a new, \
                     orphaned project under the old id. Run `engram-cli projects` to find the \
                     new portable id and update the override."
                );
            }
        }

        // Migration 11: sync_state, per-remote watermarks for `engram-cli sync`.
        //
        // Two independent watermarks, not one: a failed push must never advance the pull
        // watermark, and a failed pull must never advance the push watermark. Both are
        // derived from the payload's own max(updated_at) over rows actually transferred
        // (never wall-clock), so clock skew between the two machines cannot skip a row.
        if current_version < 11 {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS sync_state (
                    remote TEXT PRIMARY KEY,
                    pull_watermark INTEGER NOT NULL DEFAULT 0,
                    push_watermark INTEGER NOT NULL DEFAULT 0,
                    last_pull_at INTEGER,
                    last_push_at INTEGER
                );
                "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![11],
            )?;
        }

        // Migration 12: memory_origin, so `sync`'s push half can tell a memory it just
        // pulled from a remote apart from one that was genuinely edited locally.
        //
        // Without this, the push query (`updated_at >= push_wm`) cannot distinguish the
        // two: an imported memory keeps its source's `updated_at` (by design, for LWW
        // convergence — see `src/db/sync.rs`'s module doc), so once pulled it looks
        // exactly like fresh local content and gets pushed straight back to the remote it
        // came from. `origin_updated_at` pins the exact `updated_at` value the memory had
        // at the moment it was pulled; the push query excludes a memory only while its
        // current `updated_at` still equals that pinned value — any later local edit
        // (which always advances `updated_at`) makes the row eligible for push again with
        // no explicit clearing needed.
        if current_version < 12 {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS memory_origin (
                    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                    remote TEXT NOT NULL,
                    origin_updated_at INTEGER NOT NULL
                );
                "#,
            )?;

            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                params![12],
            )?;
        }

        Ok(())
    }
}
