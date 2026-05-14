//! Report writers: markdown table and JSON output for a benchmark run.

use std::io::Write;
use std::path::Path;

use crate::metrics::AggregateMetrics;

/// Metadata about one benchmark run.
#[derive(Debug, serde::Serialize)]
pub struct RunMeta {
    /// Retrieval mode used: "vector", "bm25", or "hybrid".
    pub mode: String,
    /// Query API used: "query" or "context".
    pub api: String,
    /// Number of questions evaluated.
    pub n_questions: usize,
    /// Maximum sessions ingested per question (0 = all sessions).
    pub session_limit: usize,
    /// RNG seed used for question sampling.
    pub seed: u64,
    /// Total wall-clock time for the entire run in seconds.
    pub wall_time_secs: f64,
    /// Path to the dataset file.
    pub dataset_path: String,
    /// Crate version from `Cargo.toml` at build time.
    pub crate_version: String,
    /// Git commit SHA, if available via the `GIT_SHA` env variable at build time.
    pub git_sha: Option<String>,
}

/// Write a markdown report to `path`.
///
/// Percentages are rounded to 1 decimal place (e.g. `95.2`). Raw 0-1 float values
/// are in the JSON output; the markdown shows human-readable percentages.
pub fn write_markdown(path: &Path, agg: &AggregateMetrics, meta: &RunMeta) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;

    writeln!(f, "# LongMemEval-S Benchmark Results")?;
    writeln!(f)?;
    writeln!(
        f,
        "| mode | api | partial-R@1 | partial-R@5 | partial-R@10 | full-R@1 | full-R@5 | full-R@10 | MRR | n | session_limit | wall_time_s |"
    )?;
    writeln!(
        f,
        "|------|-----|------------|------------|-------------|---------|---------|----------|-----|---|--------------|------------|"
    )?;
    writeln!(
        f,
        "| {} | {} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.3} | {} | {} | {:.1} |",
        meta.mode,
        meta.api,
        agg.partial_r_at_1 * 100.0,
        agg.partial_r_at_5 * 100.0,
        agg.partial_r_at_10 * 100.0,
        agg.full_r_at_1 * 100.0,
        agg.full_r_at_5 * 100.0,
        agg.full_r_at_10 * 100.0,
        agg.mrr,
        agg.n,
        meta.session_limit,
        meta.wall_time_secs,
    )?;

    writeln!(f)?;
    writeln!(f, "---")?;
    writeln!(f)?;
    writeln!(f, "**Dataset:** `{}`", meta.dataset_path)?;
    writeln!(f, "**Crate version:** `{}`", meta.crate_version)?;
    if let Some(sha) = &meta.git_sha {
        writeln!(f, "**Git SHA:** `{}`", sha)?;
    }
    writeln!(f, "**Seed:** `{}`", meta.seed)?;

    Ok(())
}

/// Combined struct for JSON serialization containing both run metadata and metrics.
#[derive(serde::Serialize)]
struct JsonReport<'a> {
    #[serde(flatten)]
    meta: &'a RunMeta,
    #[serde(flatten)]
    metrics: &'a AggregateMetrics,
}

/// Write a JSON report to `path`.
///
/// The output is a single flat object containing all `RunMeta` fields and all
/// `AggregateMetrics` fields. Metric values are raw 0-1 floats.
pub fn write_json(path: &Path, agg: &AggregateMetrics, meta: &RunMeta) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = std::fs::File::create(path)?;
    let report = JsonReport { meta, metrics: agg };
    serde_json::to_writer_pretty(f, &report)?;
    Ok(())
}
