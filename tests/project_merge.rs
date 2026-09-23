//! Project identity across a git remote being added, `engram-cli projects merge`, and
//! imports that still name a project merged away on this machine.

use std::path::{Path, PathBuf};
use std::process::Command;

fn cli() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_engram-cli"))
}

#[derive(Debug)]
struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Run `engram-cli` in `cwd` against `db`. `project = None` leaves the id to be derived
/// from `cwd`, the way an agent session in a repo resolves it.
fn run(db: &Path, cwd: &Path, project: Option<&str>, args: &[&str]) -> Run {
    let mut command = Command::new(cli());
    command
        .current_dir(cwd)
        .env("ENGRAM_DB", db)
        .env("ENGRAM_BRANCH", "main")
        .env_remove("ENGRAM_PROJECT")
        .args(args);
    if let Some(project) = project {
        command.env("ENGRAM_PROJECT", project);
    }
    let out = command.output().expect("failed to spawn engram-cli");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn ok(run: Run) -> Run {
    assert_eq!(
        run.code,
        Some(0),
        "stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    run
}

fn git(repo: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .status()
        .expect("git");
    assert!(status.success(), "git {args:?} failed");
}

/// `(id, memory_count)` for every project in the store.
fn projects(db: &Path, cwd: &Path) -> Vec<(String, u64)> {
    let out = ok(run(db, cwd, Some("lister"), &["--json", "projects"]));
    let value: serde_json::Value = serde_json::from_str(&out.stdout).expect("projects JSON");
    value["projects"]
        .as_array()
        .expect("projects array")
        .iter()
        .map(|p| {
            (
                p["id"].as_str().unwrap().to_string(),
                p["memory_count"].as_u64().unwrap(),
            )
        })
        .filter(|(id, _)| id != "lister")
        .collect()
}

fn count_of(projects: &[(String, u64)], id: &str) -> Option<u64> {
    projects.iter().find(|(p, _)| p == id).map(|(_, n)| *n)
}

#[test]
fn adding_a_remote_merges_the_directory_id_on_the_next_derived_run() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("store.db");
    let repo = dir.path().join("repo");
    std::fs::create_dir(&repo).unwrap();
    git(&repo, &["init", "-q"]);
    let repo = repo.canonicalize().unwrap();
    let dir_id = engram_mcp::project::home_relative(&repo.to_string_lossy());

    ok(run(
        &db,
        &repo,
        None,
        &["store", "Stored before the remote", "-t", "fact"],
    ));
    assert_eq!(count_of(&projects(&db, &repo), &dir_id), Some(1));

    git(
        &repo,
        &[
            "remote",
            "add",
            "origin",
            "https://example.com/owner/repo.git",
        ],
    );
    let remote_id = "git:example.com/owner/repo";

    // An explicit project names a project, not a directory: nothing is reconciled.
    let pinned = ok(run(&db, &repo, Some("pinned"), &["stats"]));
    assert!(
        !pinned.stderr.contains("merged project"),
        "{}",
        pinned.stderr
    );
    let after_pinned = projects(&db, &repo);
    assert_eq!(count_of(&after_pinned, &dir_id), Some(1));
    assert_eq!(count_of(&after_pinned, remote_id), None);

    let derived = ok(run(&db, &repo, None, &["stats"]));
    assert!(
        derived.stderr.contains(&format!(
            "merged project '{dir_id}' into '{remote_id}' (1 memories"
        )),
        "{}",
        derived.stderr
    );
    let after = projects(&db, &repo);
    assert_eq!(count_of(&after, &dir_id), None);
    assert_eq!(count_of(&after, remote_id), Some(1));

    // Reconciled once; a further run has nothing left to merge.
    let again = ok(run(&db, &repo, None, &["stats"]));
    assert!(!again.stderr.contains("merged project"), "{}", again.stderr);
}

#[test]
fn projects_merge_dry_run_then_confirm() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("store.db");
    let cwd = dir.path();
    ok(run(
        &db,
        cwd,
        Some("old"),
        &["store", "Old fact", "-t", "fact"],
    ));
    ok(run(
        &db,
        cwd,
        Some("old"),
        &["store", "Another old fact", "-t", "decision"],
    ));
    ok(run(
        &db,
        cwd,
        Some("new"),
        &["store", "New fact", "-t", "fact"],
    ));

    let dry = ok(run(
        &db,
        cwd,
        Some("new"),
        &["projects", "merge", "old", "new"],
    ));
    assert!(
        dry.stdout.contains("Would merge 'old' into 'new'"),
        "{}",
        dry.stdout
    );
    assert!(dry.stdout.contains("memories:        2"), "{}", dry.stdout);
    assert!(dry.stdout.contains("--confirm"), "{}", dry.stdout);
    let before = projects(&db, cwd);
    assert_eq!(count_of(&before, "old"), Some(2));
    assert_eq!(count_of(&before, "new"), Some(1));

    let applied = ok(run(
        &db,
        cwd,
        Some("new"),
        &["projects", "merge", "old", "new", "--confirm"],
    ));
    assert!(
        applied.stdout.contains("Merged 'old' into 'new'"),
        "{}",
        applied.stdout
    );
    let after = projects(&db, cwd);
    assert_eq!(count_of(&after, "old"), None);
    assert_eq!(count_of(&after, "new"), Some(3));

    let unknown = run(&db, cwd, Some("new"), &["projects", "merge", "nope", "new"]);
    assert_eq!(unknown.code, Some(1));
    assert!(
        unknown.stderr.contains("Unknown project 'nope'"),
        "{}",
        unknown.stderr
    );

    let itself = run(&db, cwd, Some("new"), &["projects", "merge", "new", "new"]);
    assert_eq!(itself.code, Some(1));
    assert!(itself.stderr.contains("into itself"), "{}", itself.stderr);
}

#[test]
fn import_of_a_merged_away_project_lands_in_the_survivor() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path();
    let adr = |db: &Path, project: &str, title: &str| {
        ok(run(
            db,
            cwd,
            Some(project),
            &[
                "adr",
                "create",
                "--non-interactive",
                "--title",
                title,
                "--decision",
                "Decided",
            ],
        ))
    };

    // The other machine never merged: it still writes under `old`.
    let source = dir.path().join("source.db");
    ok(run(
        &source,
        cwd,
        Some("old"),
        &["store", "Written after the merge elsewhere", "-t", "fact"],
    ));
    adr(&source, "old", "Source ADR");
    let payload = dir.path().join("payload.json");
    ok(run(
        &source,
        cwd,
        Some("old"),
        &[
            "export",
            "--all-projects",
            "--embeddings",
            "-o",
            payload.to_str().unwrap(),
        ],
    ));

    let local = dir.path().join("local.db");
    ok(run(
        &local,
        cwd,
        Some("old"),
        &["store", "Local old fact", "-t", "fact"],
    ));
    adr(&local, "old", "Local old ADR");
    ok(run(
        &local,
        cwd,
        Some("new"),
        &["store", "Local new fact", "-t", "decision"],
    ));
    adr(&local, "new", "Local new ADR");
    ok(run(
        &local,
        cwd,
        Some("new"),
        &["projects", "merge", "old", "new", "--confirm"],
    ));

    let imported = ok(run(
        &local,
        cwd,
        Some("new"),
        &["import", payload.to_str().unwrap()],
    ));
    assert!(
        !imported.stderr.contains("skipping imported ADR"),
        "{}",
        imported.stderr
    );

    let after = projects(&local, cwd);
    assert_eq!(count_of(&after, "old"), None, "{after:?}");
    assert_eq!(count_of(&after, "new"), Some(6), "{after:?}");

    // The source's ADR 1 collided with the survivor's renumbered 1 and 2, so it took 3.
    let adrs = ok(run(&local, cwd, Some("new"), &["--json", "adr", "list"]));
    let value: serde_json::Value = serde_json::from_str(&adrs.stdout).unwrap();
    let mut numbers: Vec<u64> = value["adrs"]
        .as_array()
        .expect("adrs array")
        .iter()
        .map(|a| a["number"].as_u64().unwrap())
        .collect();
    numbers.sort_unstable();
    assert_eq!(numbers, vec![1, 2, 3]);
}
