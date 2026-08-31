//! Migration 10 rewrites legacy absolute-path project ids to the portable
//! `crate::project` identity (normalized git remote, else `~`-relative path) on
//! `projects.id`/`root_path` and every `project_id` column.

use rusqlite::{Connection, OptionalExtension, params};

use engram_mcp::db::Database;

/// Seed a fresh sqlite file at `path` with pre-migration-10 legacy project data,
/// then leave the schema at version 9 so the next `Database::open` runs migration
/// 10 against it. Returns the base timestamp used for `created_at` ordering.
fn seed_pre_migration(path: &std::path::Path) -> i64 {
    // Bring the file up to the full current schema (including every table
    // migration 10 touches), then roll the version marker back and seed legacy
    // rows directly, bypassing the crate's own id derivation.
    drop(Database::open(path).expect("initial open to create schema"));

    let conn = Connection::open(path).expect("raw open");
    conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
    conn.execute("DELETE FROM schema_version WHERE version >= 10", [])
        .unwrap();

    let now = chrono::Utc::now().timestamp();

    let insert_project_and_memory = |id: &str, created_at: i64| {
        conn.execute(
            "INSERT INTO projects (id, name, root_path, decay_rate, created_at) \
             VALUES (?1, ?1, ?1, 0.01, ?2)",
            params![id, created_at],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories \
             (id, project_id, memory_type, content, summary, tags, importance, \
              relevance_score, access_count, created_at, updated_at, last_accessed_at, \
              branch, merged_from, pinned, global) \
             VALUES (?1, ?2, 'fact', 'legacy memory', NULL, '[]', 0.5, 1.0, 0, \
                     ?3, ?3, ?3, NULL, NULL, 0, 0)",
            params![format!("mem-{id}"), id, created_at],
        )
        .unwrap();
    };

    // Plain legacy ids: a directory and its own subdirectory (distinct portable
    // ids — they only merge via a shared git root, not tested here), plus two ids
    // that must survive completely untouched.
    insert_project_and_memory("/home/testuser/dev/foo", now);
    insert_project_and_memory("/home/testuser/dev/foo/sub", now + 1);
    insert_project_and_memory("/tmp/x", now + 2);
    insert_project_and_memory("smoke_test_temp", now + 3);

    // Two legacy ids that fold to the same portable id (`~/dev/shared`) because
    // `home_relative` strips any `/home/<user>` prefix regardless of username —
    // a genuine merge. Each carries an ADR sidecar numbered 1, which must not
    // collide once both rows share a project_id.
    for (idx, (owner, created_at)) in [
        ("/home/alice/dev/shared", now + 10),
        ("/home/bob/dev/shared", now + 20),
    ]
    .into_iter()
    .enumerate()
    {
        conn.execute(
            "INSERT INTO projects (id, name, root_path, decay_rate, created_at) \
             VALUES (?1, ?1, ?1, 0.01, ?2)",
            params![owner, created_at],
        )
        .unwrap();
        let mem_id = format!("adr-mem-{idx}");
        conn.execute(
            "INSERT INTO memories \
             (id, project_id, memory_type, content, summary, tags, importance, \
              relevance_score, access_count, created_at, updated_at, last_accessed_at, \
              branch, merged_from, pinned, global) \
             VALUES (?1, ?2, 'adr', 'adr content', NULL, '[]', 0.7, 1.0, 0, \
                     ?3, ?3, ?3, NULL, NULL, 1, 0)",
            params![mem_id, owner, created_at],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO adr_sections \
             (memory_id, project_id, adr_number, status, title, context, decision, \
              consequences, created_at, updated_at) \
             VALUES (?1, ?2, 1, 'proposed', ?3, 'ctx', 'dec', 'cons', ?4, ?4)",
            params![mem_id, owner, format!("ADR from {owner}"), created_at],
        )
        .unwrap();
    }

    now
}

fn memory_project_id_count(conn: &Connection, project_id: &str) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM memories WHERE project_id = ?1",
        params![project_id],
        |row| row.get(0),
    )
    .unwrap()
}

fn adr_number(conn: &Connection, mem_id: &str) -> i64 {
    conn.query_row(
        "SELECT adr_number FROM adr_sections WHERE memory_id = ?1",
        params![mem_id],
        |row| row.get(0),
    )
    .unwrap()
}

fn adr_project_id(conn: &Connection, mem_id: &str) -> String {
    conn.query_row(
        "SELECT project_id FROM adr_sections WHERE memory_id = ?1",
        params![mem_id],
        |row| row.get(0),
    )
    .unwrap()
}

#[test]
fn migration_rewrites_legacy_ids_and_handles_adr_collisions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("legacy.db");
    let now = seed_pre_migration(&path);

    // Opening triggers migration 10 against the seeded legacy data.
    let db = Database::open(&path).expect("open triggers migration");

    let conn = Connection::open(&path).expect("verify connection");

    // A pre-migration `.pre-portable-ids.bak` file was written before the rewrite.
    let backup_path = format!("{}.pre-portable-ids.bak", path.display());
    assert!(
        std::path::Path::new(&backup_path).exists(),
        "expected backup file at {backup_path}"
    );

    // Plain rewrites: legacy id gone, portable id present.
    for (legacy, expected) in [
        ("/home/testuser/dev/foo", "~/dev/foo"),
        ("/home/testuser/dev/foo/sub", "~/dev/foo/sub"),
    ] {
        assert_eq!(
            memory_project_id_count(&conn, legacy),
            0,
            "legacy id {legacy} must not remain on memories"
        );
        assert_eq!(
            memory_project_id_count(&conn, expected),
            1,
            "expected portable id {expected} on memories"
        );

        let (proj_id, root_path): (String, Option<String>) = conn
            .query_row(
                "SELECT id, root_path FROM projects WHERE id = ?1",
                params![expected],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .unwrap()
            .unwrap_or_else(|| panic!("projects row for portable id {expected}"));
        assert_eq!(proj_id, expected);
        assert_eq!(root_path.as_deref(), Some(expected));
    }

    // `/tmp/x` and `smoke_test_temp` are not under any home prefix — unchanged.
    for untouched in ["/tmp/x", "smoke_test_temp"] {
        assert_eq!(
            memory_project_id_count(&conn, untouched),
            1,
            "{untouched} must survive unchanged"
        );
    }

    // Merged group: one `projects` row for `~/dev/shared`, name = root_path = id,
    // created_at = MIN over the two sources (now+10, now+20), decay_rate carried
    // from either source (both 0.01 here).
    let merged_id = "~/dev/shared";
    let rows: Vec<(String, String, Option<String>, f64, i64)> = {
        let mut stmt = conn
            .prepare(
                "SELECT id, name, root_path, decay_rate, created_at FROM projects \
                 WHERE id = ?1",
            )
            .unwrap();
        stmt.query_map(params![merged_id], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
            ))
        })
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap()
    };
    assert_eq!(
        rows.len(),
        1,
        "merged group must collapse to exactly one projects row"
    );
    let (id, name, root_path, decay_rate, created_at) = &rows[0];
    assert_eq!(id, merged_id);
    assert_eq!(name, merged_id);
    assert_eq!(root_path.as_deref(), Some(merged_id));
    assert_eq!(*decay_rate, 0.01);
    assert_eq!(
        *created_at,
        now + 10,
        "created_at must be MIN over the merged group"
    );

    assert_eq!(memory_project_id_count(&conn, merged_id), 2);

    // ADR numbers: 1 and 2, no UNIQUE(project_id, adr_number) failure, order
    // preserved by created_at (alice at now+10 comes first).
    assert_eq!(
        adr_number(&conn, "adr-mem-0"),
        1,
        "alice's ADR (earlier) should be 1"
    );
    assert_eq!(
        adr_number(&conn, "adr-mem-1"),
        2,
        "bob's ADR (later) should be 2"
    );
    assert_eq!(adr_project_id(&conn, "adr-mem-0"), merged_id);
    assert_eq!(adr_project_id(&conn, "adr-mem-1"), merged_id);

    // Total memory count is unchanged by the migration (rewritten, not dropped).
    let total_memories: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap();
    assert_eq!(total_memories, 6, "6 seeded memories, none lost");

    drop(db);
    drop(conn);

    // Re-opening a second time is a no-op: schema version stays at the current head, no
    // new backup is written, and every row is exactly as migration 10 left it.
    let db2 = Database::open(&path).expect("second open");
    let conn2 = Connection::open(&path).expect("verify connection after second open");

    let version: i64 = conn2
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, 12);

    assert_eq!(memory_project_id_count(&conn2, merged_id), 2);
    assert_eq!(adr_number(&conn2, "adr-mem-0"), 1);
    assert_eq!(adr_number(&conn2, "adr-mem-1"), 2);

    let total_memories_after: i64 = conn2
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap();
    assert_eq!(total_memories_after, 6);

    // No timestamp-suffixed fallback backup was written by the second, no-op run.
    let stray_backups = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".bak"))
        .count();
    assert_eq!(
        stray_backups, 1,
        "second open must not write another backup file"
    );

    drop(db2);
}
