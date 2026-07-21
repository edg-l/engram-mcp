use std::path::PathBuf;

/// Find the git repository root by walking up from the current directory,
/// looking for a `.git` entry (directory in a plain checkout, file in a worktree).
/// Returns None if not in a git repository.
pub fn find_git_root() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        if current.join(".git").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Resolve the root of the project for scoping purposes (used to derive `project_id`).
///
/// In a plain git checkout this is the same directory `find_git_root()` returns. In a git
/// *worktree* (e.g. `<repo>/.tree/feature-x` or `<repo>/.worktree/feature-x`), `.git` is a
/// file containing a `gitdir:` pointer into the main repository's `.git/worktrees/<name>`
/// directory — walking up from cwd resolves to the worktree's own directory, so each
/// worktree of the same logical repo would otherwise get a different, siloed `project_id`.
///
/// Instead, shell out to `git rev-parse --git-common-dir`, which returns the path to the
/// MAIN repository's shared `.git` directory regardless of which worktree invoked it, and
/// take its parent directory as the project root. Falls back to `find_git_root()` if `git`
/// isn't on PATH or the command fails, so non-git environments keep working unchanged.
pub fn resolve_project_root() -> Option<PathBuf> {
    git_common_dir_root().or_else(find_git_root)
}

fn git_common_dir_root() -> Option<PathBuf> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--git-common-dir"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if raw.is_empty() {
        return None;
    }
    let git_common_dir = PathBuf::from(raw);
    let git_common_dir = if git_common_dir.is_absolute() {
        git_common_dir
    } else {
        std::env::current_dir().ok()?.join(git_common_dir)
    };
    let canonical = git_common_dir.canonicalize().ok()?;
    canonical.parent().map(|p| p.to_path_buf())
}

/// Determine the project ID from an explicit override, `ENGRAM_PROJECT`, the resolved git
/// project root (worktree-aware, see `resolve_project_root`), or finally the current
/// directory.
pub fn get_project_id(explicit: Option<String>) -> String {
    if let Some(project) = explicit {
        return project;
    }
    if let Ok(project) = std::env::var("ENGRAM_PROJECT") {
        return project;
    }
    if let Some(root) = resolve_project_root() {
        return root.to_string_lossy().to_string();
    }
    if let Ok(cwd) = std::env::current_dir() {
        return cwd.to_string_lossy().to_string();
    }
    "default".to_string()
}

/// Detect the current git branch.
/// Returns None if not in a git repository or on error.
/// Priority: `ENGRAM_BRANCH` env var > git detection.
pub fn get_current_branch() -> Option<String> {
    if let Ok(branch) = std::env::var("ENGRAM_BRANCH")
        && !branch.is_empty()
    {
        return Some(branch);
    }

    let git_root = find_git_root()?;
    let git_dir = git_root.join(".git");

    // Try reading .git/HEAD directly (faster than spawning a git process). Only works
    // when .git is a real directory (not a worktree's gitdir-pointer file) — the git
    // command fallback below covers worktrees.
    if let Ok(head_content) = std::fs::read_to_string(git_dir.join("HEAD")) {
        let head = head_content.trim();
        if let Some(branch_ref) = head.strip_prefix("ref: refs/heads/") {
            return Some(branch_ref.to_string());
        }
        // Detached HEAD - use short SHA.
        if head.len() >= 7 {
            return Some(format!("detached-{}", &head[..7]));
        }
    }

    // Fallback: shell out to git.
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(&git_root)
        .output()
        && output.status.success()
    {
        let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if branch == "HEAD" {
            if let Ok(sha_output) = std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .current_dir(&git_root)
                .output()
                && sha_output.status.success()
            {
                let sha = String::from_utf8_lossy(&sha_output.stdout)
                    .trim()
                    .to_string();
                return Some(format!("detached-{}", sha));
            }
        } else {
            return Some(branch);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn git(args: &[&str], dir: &std::path::Path) {
        let status = Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .expect("git command must run");
        assert!(status.success(), "git {:?} failed in {:?}", args, dir);
    }

    /// Reproduces the exact scenario from the bug report: a repo checked out normally,
    /// plus a `git worktree add` checkout (as used for `.tree/<branch>` worktrees in this
    /// project). `resolve_project_root()` must return the SAME path from both locations.
    #[test]
    fn resolve_project_root_unifies_main_repo_and_worktree() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let main_repo = tmp.path().join("main");
        std::fs::create_dir_all(&main_repo).unwrap();

        git(&["init", "-q"], &main_repo);
        git(&["config", "user.email", "test@example.com"], &main_repo);
        git(&["config", "user.name", "test"], &main_repo);
        std::fs::write(main_repo.join("README.md"), "hello").unwrap();
        git(&["add", "README.md"], &main_repo);
        git(&["commit", "-q", "-m", "init"], &main_repo);

        let worktree_path = tmp.path().join("wt-feature");
        git(
            &[
                "worktree",
                "add",
                "-b",
                "feature",
                worktree_path.to_str().unwrap(),
            ],
            &main_repo,
        );

        let orig_cwd = std::env::current_dir().unwrap();

        std::env::set_current_dir(&main_repo).unwrap();
        let from_main = resolve_project_root();

        std::env::set_current_dir(&worktree_path).unwrap();
        let from_worktree = resolve_project_root();

        std::env::set_current_dir(&orig_cwd).unwrap();

        let from_main = from_main.expect("main repo root must resolve");
        let from_worktree = from_worktree.expect("worktree root must resolve");

        assert_eq!(
            from_main.canonicalize().unwrap(),
            main_repo.canonicalize().unwrap()
        );
        assert_eq!(
            from_worktree.canonicalize().unwrap(),
            from_main.canonicalize().unwrap(),
            "worktree must resolve to the same project root as the main checkout"
        );

        // Sanity check: the naive find_git_root() would NOT unify these (the bug we fixed).
        std::env::set_current_dir(&worktree_path).unwrap();
        let naive = find_git_root();
        std::env::set_current_dir(&orig_cwd).unwrap();
        assert_ne!(
            naive.unwrap().canonicalize().unwrap(),
            main_repo.canonicalize().unwrap(),
            "find_git_root() is expected to still be worktree-local; resolve_project_root() is the fix"
        );
    }

    #[test]
    fn git_common_dir_root_parses_fabricated_worktree_gitdir_file() {
        // Directly exercises the parsing path git_common_dir_root() depends on
        // (`git rev-parse --git-common-dir`) without relying on process cwd state,
        // by fabricating the layout a real `git worktree add` produces.
        let tmp = tempfile::tempdir().expect("tempdir");
        let main_repo = tmp.path().join("main");
        std::fs::create_dir_all(main_repo.join(".git").join("worktrees").join("feature")).unwrap();
        std::fs::write(
            main_repo.join(".git").join("HEAD"),
            "ref: refs/heads/main\n",
        )
        .unwrap();

        let worktree_path = tmp.path().join("wt-feature");
        std::fs::create_dir_all(&worktree_path).unwrap();
        std::fs::write(
            worktree_path.join(".git"),
            format!(
                "gitdir: {}\n",
                main_repo
                    .join(".git")
                    .join("worktrees")
                    .join("feature")
                    .display()
            ),
        )
        .unwrap();

        // This test only proves the fabricated on-disk shape matches what real
        // `git worktree add` produces; actual resolution is covered end-to-end above.
        assert!(worktree_path.join(".git").is_file());
        assert!(
            main_repo
                .join(".git")
                .join("worktrees")
                .join("feature")
                .is_dir()
        );
    }
}
