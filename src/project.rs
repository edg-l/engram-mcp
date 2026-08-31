//! Portable project identity: git-root discovery, remote-URL normalization,
//! branch detection, and id derivation, shared by `engram`, `engram-cli`, and
//! `hooks::dispatch` so the three binaries never derive an id differently.
//!
//! Ids take one of three shapes:
//! - `git:github.com/owner/repo` — normalized remote (host lowercased, userinfo/
//!   port/`.git`/trailing slash stripped, scp and URL syntax unified).
//! - `~/dev/foo` — home-relative fallback for a repo or directory with no remote.
//! - verbatim — anything else (`/tmp/…`, `ENGRAM_PROJECT` values, non-path ids).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Walk up from `start` looking for a `.git` entry (file or directory, so
/// worktrees — where `.git` is a file pointing at the real gitdir — resolve to
/// the worktree root rather than failing).
pub fn find_git_root_from(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current.join(".git").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// [`find_git_root_from`] rooted at the process's current directory.
pub fn find_git_root() -> Option<PathBuf> {
    find_git_root_from(&env::current_dir().ok()?)
}

/// The repo's remote URL: `remote.origin.url` if set, else the
/// lexicographically first `remote.<name>.url` among all configured remotes.
/// `None` if the repo has no remote configured.
pub fn git_remote_url(repo: &Path) -> Option<String> {
    if let Ok(output) = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["config", "--get", "remote.origin.url"])
        .output()
        && output.status.success()
    {
        let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !url.is_empty() {
            return Some(url);
        }
    }

    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["config", "--get-regexp", r"^remote\..*\.url$"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut entries: Vec<(&str, &str)> = text
        .lines()
        .filter_map(|line| line.split_once(' '))
        .collect();
    entries.sort_by_key(|(key, _)| *key);
    entries.into_iter().next().map(|(_, url)| url.to_string())
}

/// Normalize a git remote URL to `git:{host}/{path}`. Accepts
/// `<scheme>://[user[:pass]@]host[:port]/<path>` for `ssh`, `git+ssh`, `git`,
/// `http`, `https`, and scp syntax `[user@]host:<path>` (no `/` before the
/// `:`). Returns `None` for local paths (`/…`, `./…`, `../…`, `file://…`), a
/// Windows drive-letter path (`C:\…`, `C:/…` — indistinguishable from scp
/// syntax without this guard), or a URL whose host or path would be empty.
pub fn normalize_remote_url(url: &str) -> Option<String> {
    let url = url.trim();

    if url.starts_with('/') || url.starts_with("./") || url.starts_with("../") {
        return None;
    }
    if url.starts_with("file://") {
        return None;
    }

    let bytes = url.as_bytes();
    let is_drive_letter = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/');
    if is_drive_letter {
        return None;
    }

    let (host, path) = if let Some(scheme_end) = url.find("://") {
        let scheme = &url[..scheme_end];
        if !matches!(scheme, "ssh" | "git+ssh" | "git" | "http" | "https") {
            return None;
        }
        let rest = &url[scheme_end + 3..];
        let (authority, path) = rest.split_once('/')?;
        let host_part = authority
            .rsplit_once('@')
            .map_or(authority, |(_, host)| host);
        let host = host_part.split(':').next().unwrap_or(host_part);
        (host.to_string(), path.to_string())
    } else {
        // scp syntax: [user@]host:path — a '/' before the ':' means this is a
        // local path with a colon in it, not scp syntax.
        let colon_idx = url.find(':')?;
        if url[..colon_idx].contains('/') {
            return None;
        }
        let user_host = &url[..colon_idx];
        let path = &url[colon_idx + 1..];
        let host = user_host
            .rsplit_once('@')
            .map_or(user_host, |(_, host)| host);
        (host.to_string(), path.to_string())
    };

    let host = host.to_lowercase();
    let path = path.trim_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path);

    if host.is_empty() || path.is_empty() {
        return None;
    }

    Some(format!("git:{host}/{path}"))
}

/// Fold a path under a home directory to `~`-relative form. First checks the
/// local `$HOME` exactly; failing that, folds **any** `/home/<user>`,
/// `/Users/<user>`, or `C:\Users\<user>` prefix (case-insensitive drive
/// letter, backslashes normalized to `/`) to `~`, regardless of whether
/// `<user>` matches the local `$HOME`'s basename — the whole point is
/// convergence across two machines with different account names. Anything
/// else is returned verbatim.
pub fn home_relative(path: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let home = home.to_string_lossy().to_string();
        if path == home {
            return "~".to_string();
        }
        if let Some(rest) = path.strip_prefix(&home)
            && let Some(rest) = rest.strip_prefix(['/', '\\'])
        {
            return format!("~/{}", rest.replace('\\', "/"));
        }
    }

    let normalized = path.replace('\\', "/");

    for prefix in ["/home/", "/Users/"] {
        if let Some(rest) = normalized.strip_prefix(prefix) {
            return match rest.find('/') {
                Some(idx) => format!("~{}", &rest[idx..]),
                None => "~".to_string(),
            };
        }
    }

    // C:/Users/<user>/… after backslash normalization above (drive letter
    // case-insensitive).
    let bytes = normalized.as_bytes();
    if bytes.len() >= 2
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && let Some(rest) = normalized[2..].strip_prefix("/Users/")
    {
        return match rest.find('/') {
            Some(idx) => format!("~{}", &rest[idx..]),
            None => "~".to_string(),
        };
    }

    path.to_string()
}

/// Derive a project id for `dir`: the normalized git remote of its git root
/// if it has one, else the git root's home-relative path, else `dir`'s own
/// home-relative path if there is no git root at all.
pub fn project_id_for_dir(dir: &Path) -> String {
    if let Some(git_root) = find_git_root_from(dir) {
        if let Some(id) = git_remote_url(&git_root).and_then(|url| normalize_remote_url(&url)) {
            return id;
        }
        return home_relative(&git_root.to_string_lossy());
    }
    home_relative(&dir.to_string_lossy())
}

/// Resolve the project id for this invocation: `explicit` wins outright, else
/// a non-empty `ENGRAM_PROJECT`, else [`project_id_for_dir`] of the current
/// directory, else `"default"`.
pub fn resolve_project_id(explicit: Option<String>) -> String {
    if let Some(project) = explicit {
        return project;
    }
    if let Ok(project) = env::var("ENGRAM_PROJECT")
        && !project.is_empty()
    {
        return project;
    }
    if let Ok(cwd) = env::current_dir() {
        return project_id_for_dir(&cwd);
    }
    "default".to_string()
}

/// Re-derive a portable id from a legacy (pre-migration, absolute-path)
/// project id. If the legacy string is still a real directory,
/// [`project_id_for_dir`] can discover its git remote; otherwise it is folded
/// with [`home_relative`] as a bare string, since there is no directory left
/// to walk up from.
pub fn portable_id_from_legacy(legacy: &str) -> String {
    let path = Path::new(legacy);
    if path.is_dir() {
        project_id_for_dir(path)
    } else {
        home_relative(legacy)
    }
}

/// Detect the branch checked out in `repo`. Reads `.git/HEAD` directly
/// (faster than spawning `git`) and falls back to `git rev-parse` — the
/// fallback also covers worktrees, where `.git` is a file rather than a
/// directory, so the direct `HEAD` read misses.
pub fn current_branch_in(repo: &Path) -> Option<String> {
    let git_dir = repo.join(".git");

    if let Ok(head_content) = std::fs::read_to_string(git_dir.join("HEAD")) {
        let head = head_content.trim();
        if let Some(branch_ref) = head.strip_prefix("ref: refs/heads/") {
            return Some(branch_ref.to_string());
        }
        // Detached HEAD - use short SHA
        if head.len() >= 7 {
            return Some(format!("detached-{}", &head[..7]));
        }
    }

    // Fallback: try git command
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(repo)
        .output()
        && output.status.success()
    {
        let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if branch == "HEAD" {
            // Detached HEAD - get short SHA
            if let Ok(sha_output) = Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .current_dir(repo)
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

/// Detect the current git branch. Priority: `ENGRAM_BRANCH` env var > git
/// detection at the current directory's git root.
pub fn current_branch() -> Option<String> {
    if let Ok(branch) = env::var("ENGRAM_BRANCH")
        && !branch.is_empty()
    {
        return Some(branch);
    }

    let git_root = find_git_root()?;
    current_branch_in(&git_root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_remote_url_variants_agree() {
        let expected = "git:github.com/owner/repo";
        assert_eq!(
            normalize_remote_url("git@github.com:owner/repo.git").as_deref(),
            Some(expected)
        );
        assert_eq!(
            normalize_remote_url("https://github.com/owner/repo.git").as_deref(),
            Some(expected)
        );
        assert_eq!(
            normalize_remote_url("https://github.com/owner/repo/").as_deref(),
            Some(expected)
        );
        assert_eq!(
            normalize_remote_url("ssh://git@github.com:22/owner/repo.git").as_deref(),
            Some(expected)
        );
        assert_eq!(
            normalize_remote_url("git://github.com/owner/repo").as_deref(),
            Some(expected)
        );
    }

    #[test]
    fn normalize_remote_url_rejects_local_and_unsupported_paths() {
        assert_eq!(normalize_remote_url("/srv/git/repo.git"), None);
        assert_eq!(normalize_remote_url("file:///srv/repo"), None);
        // Windows drive-letter local path must not be misparsed as scp syntax
        // (host=`C`, path=`\repos\foo`).
        assert_eq!(normalize_remote_url("C:\\repos\\foo"), None);
    }

    #[test]
    fn home_relative_folds_any_username() {
        assert_eq!(home_relative("/home/alice/dev/foo"), "~/dev/foo");
        assert_eq!(home_relative("/home/bob/dev/foo"), "~/dev/foo");
        assert_eq!(home_relative("/Users/alice/dev/foo"), "~/dev/foo");
        assert_eq!(home_relative("/Users/bob/dev/foo"), "~/dev/foo");
    }

    #[test]
    fn home_relative_leaves_non_home_paths_untouched() {
        assert_eq!(home_relative("/tmp/x"), "/tmp/x");
        assert_eq!(home_relative("smoke_test_temp"), "smoke_test_temp");
    }

    #[test]
    fn home_relative_folds_windows_users_prefix() {
        assert_eq!(home_relative("C:\\Users\\alice\\dev\\foo"), "~/dev/foo");
        assert_eq!(home_relative("c:\\Users\\alice\\dev\\foo"), "~/dev/foo");
    }
}
