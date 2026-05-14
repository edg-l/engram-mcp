mod dataset;
mod ingest;
mod metrics;
mod report;
mod runner;
mod setup;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use rand::seq::SliceRandom;

use engram_mcp::tools::{SearchMode, ToolHandler};

#[derive(Debug, Clone, clap::ValueEnum)]
enum Api {
    Query,
    Context,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum Mode {
    Vector,
    Bm25,
    Hybrid,
}

impl Mode {
    fn to_search_mode(&self) -> SearchMode {
        match self {
            Mode::Vector => SearchMode::Vector,
            Mode::Bm25 => SearchMode::Bm25,
            Mode::Hybrid => SearchMode::Hybrid,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Mode::Vector => "vector",
            Mode::Bm25 => "bm25",
            Mode::Hybrid => "hybrid",
        }
    }
}

impl Api {
    fn as_str(&self) -> &'static str {
        match self {
            Api::Query => "query",
            Api::Context => "context",
        }
    }
}

/// Number of questions to run — accepts an integer or the string "all".
#[derive(Debug, Clone)]
struct QuestionCount(Option<usize>);

impl std::str::FromStr for QuestionCount {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("all") {
            return Ok(QuestionCount(None));
        }
        s.parse::<usize>()
            .map(|n| QuestionCount(Some(n)))
            .map_err(|_| format!("expected a non-negative integer or \"all\", got {:?}", s))
    }
}

#[derive(Debug, Parser)]
#[command(name = "longmemeval-bench", about = "LongMemEval-S benchmark harness")]
struct Args {
    /// Path to the LongMemEval-S JSON dataset file.
    #[arg(long)]
    dataset: PathBuf,

    /// Number of questions to evaluate, or "all".
    #[arg(long, default_value = "30")]
    questions: QuestionCount,

    /// Maximum sessions per question to ingest (0 = all sessions).
    #[arg(long, default_value = "10")]
    session_limit: usize,

    /// Which retrieval API to use.
    #[arg(long, value_enum, default_value = "query")]
    api: Api,

    /// Search mode.
    #[arg(long, value_enum, default_value = "hybrid")]
    mode: Mode,

    /// RNG seed for reproducible question sampling.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Directory for result output files.
    #[arg(long, default_value = "benchmarks/longmemeval/results/")]
    out: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let mut questions = dataset::load_dataset(&args.dataset)?;
    let total = questions.len();
    println!("Loaded {} questions from {}", total, args.dataset.display());

    let target = match args.questions.0 {
        None => total,
        Some(n) => n.min(total),
    };

    if target < total {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        questions.shuffle(&mut rng);
        questions.truncate(target);
        println!("Sampled {} questions (seed={})", target, args.seed);
    }

    // Initialize the embedding service once — model load is expensive (~500ms cold).
    // EmbeddingService::clone() shares the Arc<Mutex<TextEmbedding>>, no reload.
    println!("Initializing embedding service...");
    let embedding = engram_mcp::embedding::EmbeddingService::new()?;
    println!("Embedding service ready.");

    let search_mode = args.mode.to_search_mode();
    let n = questions.len();

    // Per-question results: top-10 retrieved session ids.
    let mut all_top10_hits: Vec<metrics::HitResult> = Vec::with_capacity(n);
    let mut all_top5_hits: Vec<metrics::HitResult> = Vec::with_capacity(n);
    let mut all_top1_hits: Vec<metrics::HitResult> = Vec::with_capacity(n);

    let wall_start = Instant::now();

    for (i, q) in questions.iter().enumerate() {
        let tempdir = tempfile::TempDir::new()?;
        let db = setup::setup_db(&tempdir)?;

        let project_id = format!("lme-{}", q.question_id);
        db.get_or_create_project(&project_id, &project_id)?;

        let t0 = Instant::now();
        let count =
            ingest::ingest_question(&db, &embedding, q, &project_id, args.session_limit).await?;
        let elapsed = t0.elapsed();

        println!(
            "[{}/{}] question_id={}: ingested {} turn-pairs in {:?}",
            i + 1,
            n,
            q.question_id,
            count,
            elapsed
        );

        // Build a ToolHandler for this question's isolated database.
        // EmbeddingService::clone() shares the loaded ONNX model via Arc.
        let handler =
            ToolHandler::new(db, embedding.clone(), project_id.clone(), None, search_mode);

        // Retrieve top-10 session ids using the chosen API.
        let top10_ids = match args.api {
            Api::Query => runner::run_query(&handler, &q.question, 10).await?,
            Api::Context => runner::run_context(&handler, &q.question, 10).await?,
        };

        // Evaluate at k=1, k=5, k=10 by calling evaluate_topk with different k values.
        let hit10 = metrics::evaluate_topk(&top10_ids, &q.answer_session_ids, 10);
        let hit5 = metrics::evaluate_topk(&top10_ids, &q.answer_session_ids, 5);
        let hit1 = metrics::evaluate_topk(&top10_ids, &q.answer_session_ids, 1);

        all_top10_hits.push(hit10);
        all_top5_hits.push(hit5);
        all_top1_hits.push(hit1);

        // tempdir drops here, cleaning up the per-question SQLite database.
    }

    let wall_secs = wall_start.elapsed().as_secs_f64();

    // Aggregate metrics.
    let agg = metrics::aggregate(&all_top10_hits, &all_top5_hits, &all_top1_hits);

    // Build output filename: {mode}-{api}-{timestamp}.{ext}
    let timestamp = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
        .replace(':', "-");
    let stem = format!("{}-{}-{}", args.mode.as_str(), args.api.as_str(), timestamp);

    std::fs::create_dir_all(&args.out)?;
    let md_path = args.out.join(format!("{}.md", stem));
    let json_path = args.out.join(format!("{}.json", stem));

    let meta = report::RunMeta {
        mode: args.mode.as_str().to_string(),
        api: args.api.as_str().to_string(),
        n_questions: agg.n,
        session_limit: args.session_limit,
        seed: args.seed,
        wall_time_secs: wall_secs,
        dataset_path: args.dataset.display().to_string(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: option_env!("GIT_SHA").map(str::to_owned),
    };

    report::write_markdown(&md_path, &agg, &meta)?;
    report::write_json(&json_path, &agg, &meta)?;

    println!(
        "partial-R@5 = {:.1}%  mrr = {:.3}  n = {}  wall = {:.1}s",
        agg.partial_r_at_5 * 100.0,
        agg.mrr,
        agg.n,
        wall_secs,
    );
    println!("markdown: {}", md_path.display());
    println!("json:     {}", json_path.display());

    Ok(())
}
