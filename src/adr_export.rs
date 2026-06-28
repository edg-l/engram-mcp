//! ADR file-export utilities.
//!
//! Converts `adr_sections` sidecar rows into Nygard-style Markdown files.
//! Target directory precedence: explicit override → `ENGRAM_ADR_DIR` env var → `docs/adr`.

use std::path::{Path, PathBuf};

use crate::db::Database;
use crate::error::MemoryError;
use crate::memory::{AdrSections, AdrStatus, kebab_title};

/// Resolve the ADR output directory.
///
/// Precedence: `override_dir` argument → `ENGRAM_ADR_DIR` env var → `docs/adr`.
pub fn adr_export_target_dir(override_dir: Option<&str>) -> PathBuf {
    if let Some(d) = override_dir {
        return PathBuf::from(d);
    }
    if let Ok(env_dir) = std::env::var("ENGRAM_ADR_DIR")
        && !env_dir.is_empty()
    {
        return PathBuf::from(env_dir);
    }
    PathBuf::from("docs/adr")
}

/// Build the canonical filename for an ADR.
///
/// Format: `{:04}-{slug}.md` where `slug` is the kebab-case title.
/// Falls back to `"adr"` when the title produces an empty slug.
pub fn adr_file_name(number: u32, title: &str) -> String {
    let slug = kebab_title(title);
    let slug = if slug.is_empty() {
        "adr".to_string()
    } else {
        slug
    };
    format!("{:04}-{}.md", number, slug)
}

/// Render the canonical Markdown content for a single ADR file.
pub fn render_adr_file(
    number: u32,
    status: AdrStatus,
    sections: &AdrSections,
    created_at: i64,
) -> String {
    sections.render_markdown(number, status, created_at)
}

/// Export one or all ADRs to Markdown files under `dir`.
///
/// - `number = Some(n)` — export only ADR #n; `MemoryError::NotFound` if it doesn't exist.
/// - `number = None`   — export all ADRs for the project.
/// - `dry_run = true`  — collect the target paths without touching the filesystem.
/// - `dry_run = false` — create `dir` if needed, then write each file (overwriting silently).
///
/// Returns the list of paths that were (or would be) written.
pub fn export_adr_to_disk(
    db: &Database,
    project_id: &str,
    dir: &Path,
    number: Option<u32>,
    dry_run: bool,
) -> Result<Vec<PathBuf>, MemoryError> {
    // Collect (memory_id, adr_number) pairs to export.
    let memory_ids: Vec<(String, u32)> = if let Some(n) = number {
        let mid = db
            .get_adr_by_number(project_id, n)?
            .ok_or_else(|| MemoryError::NotFound(format!("ADR-{:04}", n)))?;
        vec![(mid, n)]
    } else {
        db.list_adrs(project_id, None)?
            .into_iter()
            .map(|(num, _status, _title, mid)| (mid, num))
            .collect()
    };

    if !dry_run && !memory_ids.is_empty() {
        std::fs::create_dir_all(dir).map_err(MemoryError::Io)?;
    }

    let mut written: Vec<PathBuf> = Vec::new();

    for (memory_id, num) in memory_ids {
        let (adr_num, status, sections) = db
            .get_adr_sections(&memory_id)?
            .ok_or_else(|| MemoryError::NotFound(memory_id.clone()))?;

        // Fetch created_at from the memory row.
        let memory = db
            .get_memory(&memory_id)?
            .ok_or_else(|| MemoryError::NotFound(memory_id.clone()))?;

        let _ = num; // adr_num from sidecar is canonical
        let file_name = adr_file_name(adr_num, &sections.title);
        let path = dir.join(&file_name);

        if !dry_run {
            let content = render_adr_file(adr_num, status, &sections, memory.created_at);
            std::fs::write(&path, content).map_err(MemoryError::Io)?;
        }

        written.push(path);
    }

    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use crate::embedding::EmbeddingService;
    use crate::memory::{AdrSections, AdrStatus};

    fn make_adr_sections(title: &str) -> AdrSections {
        AdrSections {
            title: title.to_string(),
            context: "Some context".to_string(),
            decision: "The decision".to_string(),
            consequences: "Some consequences".to_string(),
        }
    }

    fn store_adr_for_test(
        db: &Database,
        embedding: &EmbeddingService,
        project_id: &str,
        title: &str,
    ) {
        let sections = make_adr_sections(title);
        let id = format!("mem_{}", uuid::Uuid::new_v4().simple());
        let now = chrono::Utc::now().timestamp();
        let combined = format!(
            "{}\n{}\n{}\n{}",
            sections.title, sections.context, sections.decision, sections.consequences
        );
        let emb = embedding
            .embed_memory(crate::memory::MemoryType::Adr, &combined)
            .expect("embed must succeed");
        db.store_adr_atomic(
            &id,
            project_id,
            &sections,
            AdrStatus::Proposed,
            0.7,
            true,
            &emb,
            embedding.model_version(),
            now,
            None,
        )
        .expect("adr store must succeed");
    }

    /// 5.2 test 1: adr_file_name_zero_pads_kebabs_and_empty_fallback
    /// Assert adr_file_name(1, "Use SQLite!") == "0001-use-sqlite.md"
    /// and an all-punctuation title falls back to "0001-adr.md".
    #[test]
    fn adr_file_name_zero_pads_kebabs_and_empty_fallback() {
        assert_eq!(adr_file_name(1, "Use SQLite!"), "0001-use-sqlite.md");
        // All-punctuation title produces empty slug — fallback to "adr".
        assert_eq!(adr_file_name(1, "!!!"), "0001-adr.md");
        // Verify zero-padding for larger numbers.
        assert_eq!(adr_file_name(42, "Foo Bar"), "0042-foo-bar.md");
    }

    /// 5.2 test 2: target_dir_precedence
    /// Override arg beats ENGRAM_ADR_DIR beats default "docs/adr".
    #[test]
    fn target_dir_precedence() {
        // Override arg takes precedence over everything.
        let result = adr_export_target_dir(Some("/custom/path"));
        assert_eq!(result, std::path::PathBuf::from("/custom/path"));

        // Default when no override and no env var.
        // Temporarily clear the env var to avoid interference from other tests.
        let saved = std::env::var("ENGRAM_ADR_DIR").ok();
        // Safety: single-threaded test context; we restore immediately after.
        unsafe { std::env::remove_var("ENGRAM_ADR_DIR") };
        let default_result = adr_export_target_dir(None);
        assert_eq!(default_result, std::path::PathBuf::from("docs/adr"));

        // Set env var and verify it is picked up over the default.
        unsafe { std::env::set_var("ENGRAM_ADR_DIR", "/from/env") };
        let env_result = adr_export_target_dir(None);
        assert_eq!(env_result, std::path::PathBuf::from("/from/env"));

        // Restore original value.
        unsafe { std::env::remove_var("ENGRAM_ADR_DIR") };
        if let Some(orig) = saved {
            unsafe { std::env::set_var("ENGRAM_ADR_DIR", orig) };
        }
    }

    /// 5.2 test 3: export_dry_run_writes_nothing
    /// In-memory DB + a tempdir; create an ADR; call export_adr_to_disk with dry_run=true;
    /// assert returned paths non-empty AND the tempdir contains no files.
    #[test]
    fn export_dry_run_writes_nothing() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "adr-dry-run-proj";
        db.get_or_create_project(project_id, "Dry Run Test")
            .unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");
        store_adr_for_test(&db, &embedding, project_id, "Use SQLite");

        let tmpdir = tempfile::TempDir::new().expect("tempdir must be created");
        let paths = export_adr_to_disk(&db, project_id, tmpdir.path(), None, true)
            .expect("dry_run export must succeed");

        assert!(
            !paths.is_empty(),
            "dry_run must return non-empty paths list"
        );

        // No files should have been written.
        let files: Vec<_> = std::fs::read_dir(tmpdir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            files.is_empty(),
            "dry_run must not write any files, but found: {:?}",
            files.iter().map(|e| e.path()).collect::<Vec<_>>()
        );
    }

    /// 5.2 test 4: export_write_creates_one_file_per_adr
    /// dry_run=false; assert N files exist with expected names and each contains "## Decision".
    #[test]
    fn export_write_creates_one_file_per_adr() {
        let db = Database::open_in_memory().unwrap();
        let project_id = "adr-write-proj";
        db.get_or_create_project(project_id, "Write Test").unwrap();

        let embedding = EmbeddingService::new().expect("model must be available");
        store_adr_for_test(&db, &embedding, project_id, "Use SQLite");
        store_adr_for_test(&db, &embedding, project_id, "Use PostgreSQL");

        let tmpdir = tempfile::TempDir::new().expect("tempdir must be created");
        let paths = export_adr_to_disk(&db, project_id, tmpdir.path(), None, false)
            .expect("export must succeed");

        assert_eq!(paths.len(), 2, "must export exactly 2 ADRs");

        for path in &paths {
            assert!(path.exists(), "exported file must exist: {:?}", path);
            let content = std::fs::read_to_string(path).expect("file must be readable");
            assert!(
                content.contains("## Decision"),
                "exported file must contain '## Decision', path={:?}",
                path
            );
        }

        // Verify expected filenames.
        let file_names: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(
            file_names.contains(&"0001-use-sqlite.md".to_string()),
            "expected 0001-use-sqlite.md, got {:?}",
            file_names
        );
        assert!(
            file_names.contains(&"0002-use-postgresql.md".to_string()),
            "expected 0002-use-postgresql.md, got {:?}",
            file_names
        );
    }
}
