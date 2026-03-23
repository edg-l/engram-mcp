//! Retrieval quality benchmark for Engram.
//!
//! Stores a corpus of realistic memories, runs queries, and measures how well
//! the system retrieves the expected results. Outputs a scorecard with:
//!   - Hit@1, Hit@3, Hit@5 (was the expected memory in the top K?)
//!   - MRR (Mean Reciprocal Rank)
//!   - Per-query breakdown
//!
//! Run with: cargo test --test bench_retrieval -- --nocapture
//!
//! This is NOT a pass/fail test by default. It prints a report. Set the env var
//! ENGRAM_BENCH_MIN_MRR=0.6 to fail if MRR drops below that threshold.

use engram_mcp::db::Database;
use engram_mcp::embedding::EmbeddingService;
use engram_mcp::tools::ToolHandler;
use serde_json::{Value, json};

fn setup() -> (ToolHandler, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("bench.db");
    let db = Database::open(&db_path).unwrap();
    let embedding = EmbeddingService::new().unwrap();
    let project_id = "bench".to_string();
    db.get_or_create_project(&project_id, &project_id).unwrap();
    let handler = ToolHandler::new(db, embedding, project_id, None);
    (handler, dir)
}

fn call(handler: &ToolHandler, tool: &str, args: Value) -> Value {
    handler
        .handle_tool(tool, args)
        .expect(&format!("{} failed", tool))
}

/// A memory to store in the corpus.
struct CorpusMemory {
    id_tag: &'static str, // human-readable tag for matching
    content: &'static str,
    memory_type: &'static str,
    tags: &'static [&'static str],
    importance: f64,
}

/// A query with the expected matching memory id_tag.
struct QueryCase {
    name: &'static str,
    query: &'static str,
    /// The id_tag of the memory that SHOULD rank highest (or near top)
    expected: &'static str,
    /// Optional: query with type filter
    type_filter: Option<&'static str>,
}

fn corpus() -> Vec<CorpusMemory> {
    vec![
        // -- Backend / API --
        CorpusMemory {
            id_tag: "backend-framework",
            content: "The backend API is built with Rust using the Axum framework on port 8080, with tower middleware for rate limiting and tracing",
            memory_type: "fact",
            tags: &["backend", "rust", "axum"],
            importance: 0.8,
        },
        CorpusMemory {
            id_tag: "api-versioning",
            content: "API versioning uses URL path prefix /v1/, /v2/ with a custom middleware that routes to the correct handler module",
            memory_type: "decision",
            tags: &["api", "versioning"],
            importance: 0.7,
        },
        CorpusMemory {
            id_tag: "error-handling",
            content: "All API errors return RFC 7807 Problem Details JSON with a correlation ID for tracing through the request pipeline",
            memory_type: "pattern",
            tags: &["api", "errors"],
            importance: 0.6,
        },
        // -- Database --
        CorpusMemory {
            id_tag: "database-primary",
            content: "PostgreSQL 16 is the primary database running on AWS RDS in us-east-1 with Multi-AZ failover enabled",
            memory_type: "fact",
            tags: &["database", "postgres", "aws"],
            importance: 0.9,
        },
        CorpusMemory {
            id_tag: "database-migrations",
            content: "Database schema migrations are managed with sqlx-cli, applied automatically during deployment via a Kubernetes init container",
            memory_type: "fact",
            tags: &["database", "migrations"],
            importance: 0.6,
        },
        CorpusMemory {
            id_tag: "database-pooling",
            content: "Connection pooling uses PgBouncer in transaction mode with max 100 connections, sitting between the app and RDS",
            memory_type: "fact",
            tags: &["database", "performance"],
            importance: 0.5,
        },
        CorpusMemory {
            id_tag: "cache-redis",
            content: "Redis 7 cluster is used for caching API responses and session storage with a default TTL of 5 minutes",
            memory_type: "fact",
            tags: &["cache", "redis"],
            importance: 0.6,
        },
        // -- Auth --
        CorpusMemory {
            id_tag: "auth-jwt",
            content: "Authentication uses JWT tokens signed with RS256, issued by our auth service, with 1 hour expiry and refresh tokens stored in httpOnly cookies",
            memory_type: "decision",
            tags: &["auth", "security", "jwt"],
            importance: 0.9,
        },
        CorpusMemory {
            id_tag: "auth-rbac",
            content: "Authorization is role-based with three roles: admin, editor, viewer. Permissions are checked via a tower middleware layer",
            memory_type: "decision",
            tags: &["auth", "rbac"],
            importance: 0.8,
        },
        // -- Frontend --
        CorpusMemory {
            id_tag: "frontend-stack",
            content: "The frontend is a React 18 SPA using TypeScript, built with Vite, and communicates with the backend through a GraphQL gateway powered by Apollo",
            memory_type: "fact",
            tags: &["frontend", "react", "graphql"],
            importance: 0.7,
        },
        CorpusMemory {
            id_tag: "frontend-state",
            content: "Client-side state management uses Zustand for global state and TanStack Query for server state with optimistic updates",
            memory_type: "decision",
            tags: &["frontend", "state"],
            importance: 0.6,
        },
        // -- Deployment --
        CorpusMemory {
            id_tag: "deploy-k8s",
            content: "Production runs on Kubernetes (EKS) with Helm charts in the deploy/ directory, using rolling deployments with 25% max surge",
            memory_type: "fact",
            tags: &["deployment", "kubernetes"],
            importance: 0.8,
        },
        CorpusMemory {
            id_tag: "deploy-ci",
            content: "CI/CD pipeline runs on GitHub Actions: lint, test, build Docker image, push to ECR, deploy to EKS via ArgoCD",
            memory_type: "fact",
            tags: &["ci", "deployment"],
            importance: 0.7,
        },
        CorpusMemory {
            id_tag: "deploy-envs",
            content: "Three environments: dev (auto-deploy on push to main), staging (manual promote), production (requires approval from two reviewers)",
            memory_type: "decision",
            tags: &["deployment", "environments"],
            importance: 0.7,
        },
        // -- Monitoring --
        CorpusMemory {
            id_tag: "monitoring-metrics",
            content: "Application metrics are exported via Prometheus /metrics endpoint and visualized in Grafana dashboards",
            memory_type: "fact",
            tags: &["monitoring", "prometheus"],
            importance: 0.6,
        },
        CorpusMemory {
            id_tag: "monitoring-logs",
            content: "Structured JSON logs are shipped via Filebeat to Elasticsearch, searchable through Kibana with retention of 30 days",
            memory_type: "fact",
            tags: &["logging", "elasticsearch"],
            importance: 0.5,
        },
        CorpusMemory {
            id_tag: "monitoring-errors",
            content: "Error tracking uses Sentry with source maps uploaded on each deploy, alerting to Slack #incidents channel",
            memory_type: "fact",
            tags: &["monitoring", "sentry"],
            importance: 0.7,
        },
        // -- Testing --
        CorpusMemory {
            id_tag: "testing-integration",
            content: "Integration tests use testcontainers-rs to spin up ephemeral Postgres and Redis instances, running in parallel via cargo-nextest",
            memory_type: "fact",
            tags: &["testing"],
            importance: 0.6,
        },
        CorpusMemory {
            id_tag: "testing-coverage",
            content: "Code coverage is measured with cargo-llvm-cov, enforced at 80% minimum in CI, reported to Codecov",
            memory_type: "decision",
            tags: &["testing", "ci"],
            importance: 0.5,
        },
        // -- Debug notes --
        CorpusMemory {
            id_tag: "debug-oom",
            content: "The payment service hit OutOfMemoryError under load testing at 500 req/s due to unbounded channel buffer in the event processor",
            memory_type: "debug",
            tags: &["payment", "oom"],
            importance: 0.8,
        },
        CorpusMemory {
            id_tag: "debug-oom-fix",
            content: "Fixed OOM by switching to bounded channel with capacity 1000 and adding backpressure via tokio::sync::mpsc",
            memory_type: "pattern",
            tags: &["payment", "fix"],
            importance: 0.7,
        },
    ]
}

fn queries() -> Vec<QueryCase> {
    vec![
        // -- Direct topic match --
        QueryCase {
            name: "backend framework",
            query: "what web framework does the backend use",
            expected: "backend-framework",
            type_filter: None,
        },
        QueryCase {
            name: "primary database",
            query: "which database is used and where is it hosted",
            expected: "database-primary",
            type_filter: None,
        },
        QueryCase {
            name: "auth mechanism",
            query: "how does authentication work",
            expected: "auth-jwt",
            type_filter: None,
        },
        QueryCase {
            name: "frontend framework",
            query: "what is the frontend built with",
            expected: "frontend-stack",
            type_filter: None,
        },
        QueryCase {
            name: "deployment platform",
            query: "where does the app run in production",
            expected: "deploy-k8s",
            type_filter: None,
        },
        // -- Semantic / rephrased queries --
        QueryCase {
            name: "caching layer",
            query: "how are API responses cached",
            expected: "cache-redis",
            type_filter: None,
        },
        QueryCase {
            name: "error format",
            query: "what format do error responses use",
            expected: "error-handling",
            type_filter: None,
        },
        QueryCase {
            name: "user permissions",
            query: "how are user roles and permissions managed",
            expected: "auth-rbac",
            type_filter: None,
        },
        QueryCase {
            name: "log aggregation",
            query: "where do logs go and how are they searched",
            expected: "monitoring-logs",
            type_filter: None,
        },
        QueryCase {
            name: "deploy pipeline",
            query: "what is the CI/CD pipeline",
            expected: "deploy-ci",
            type_filter: None,
        },
        // -- Cross-domain / indirect queries --
        QueryCase {
            name: "schema changes",
            query: "how do we apply database schema changes",
            expected: "database-migrations",
            type_filter: None,
        },
        QueryCase {
            name: "connection limits",
            query: "how many database connections can the app use",
            expected: "database-pooling",
            type_filter: None,
        },
        QueryCase {
            name: "client state",
            query: "how is state managed in the React app",
            expected: "frontend-state",
            type_filter: None,
        },
        QueryCase {
            name: "crash reporting",
            query: "how are production errors reported and alerted on",
            expected: "monitoring-errors",
            type_filter: None,
        },
        QueryCase {
            name: "test infrastructure",
            query: "how do integration tests get a database",
            expected: "testing-integration",
            type_filter: None,
        },
        // -- Debug / incident queries --
        QueryCase {
            name: "payment OOM",
            query: "what caused the memory issue in the payment service",
            expected: "debug-oom",
            type_filter: None,
        },
        QueryCase {
            name: "OOM resolution",
            query: "how was the out of memory bug fixed",
            expected: "debug-oom-fix",
            type_filter: None,
        },
        // -- Type-filtered queries --
        QueryCase {
            name: "decisions about auth",
            query: "authentication and authorization decisions",
            expected: "auth-jwt",
            type_filter: Some("decision"),
        },
        QueryCase {
            name: "deployment decisions",
            query: "how are deployments promoted between environments",
            expected: "deploy-envs",
            type_filter: Some("decision"),
        },
        // -- Harder / ambiguous queries --
        QueryCase {
            name: "observability stack",
            query: "what is the observability and monitoring setup",
            expected: "monitoring-metrics",
            type_filter: None,
        },
    ]
}

struct QueryResult {
    name: &'static str,
    query: &'static str,
    expected: &'static str,
    rank: Option<usize>, // 1-indexed position, None if not found
    top_result: String,
    _score: f64,
}

#[test]
fn bench_retrieval_quality() {
    let (h, _dir) = setup();

    // Store corpus
    let corpus = corpus();
    let mut id_map: std::collections::HashMap<&str, String> = std::collections::HashMap::new();

    for mem in &corpus {
        let tags: Vec<String> = mem.tags.iter().map(|t| t.to_string()).collect();
        let r = call(
            &h,
            "memory_store",
            json!({
                "content": mem.content,
                "type": mem.memory_type,
                "tags": tags,
                "importance": mem.importance,
            }),
        );
        id_map.insert(mem.id_tag, r["id"].as_str().unwrap().to_string());
    }

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("ENGRAM RETRIEVAL BENCHMARK");
    eprintln!(
        "Corpus: {} memories | Model: mdbr-leaf-ir q8 d256 | Queries: {}",
        corpus.len(),
        queries().len()
    );
    eprintln!("{}", "=".repeat(80));

    // Run queries
    let queries = queries();
    let mut results: Vec<QueryResult> = Vec::new();

    for q in &queries {
        let mut args = json!({
            "query": q.query,
            "limit": 10,
        });
        if let Some(type_filter) = q.type_filter {
            args["types"] = json!([type_filter]);
        }

        let r = call(&h, "memory_query", args);
        let memories = r["memories"].as_array().unwrap();

        let expected_id = id_map.get(q.expected).unwrap();

        // Find rank of expected memory
        let rank = memories
            .iter()
            .position(|m| m["memory"]["id"].as_str().unwrap() == expected_id.as_str())
            .map(|pos| pos + 1); // 1-indexed

        let top_result = memories
            .first()
            .map(|m| {
                let content = m["memory"]["content"].as_str().unwrap();
                if content.len() > 70 {
                    format!("{}...", &content[..70])
                } else {
                    content.to_string()
                }
            })
            .unwrap_or_else(|| "(no results)".to_string());

        let score = memories
            .first()
            .and_then(|m| m["score"].as_f64())
            .unwrap_or(0.0);

        results.push(QueryResult {
            name: q.name,
            query: q.query,
            expected: q.expected,
            rank,
            top_result,
            _score: score,
        });
    }

    // Print results
    eprintln!("\n{:<25} {:<6} {}", "QUERY", "RANK", "TOP RESULT");
    eprintln!("{}", "-".repeat(80));

    let mut hit_at_1 = 0usize;
    let mut hit_at_3 = 0usize;
    let mut hit_at_5 = 0usize;
    let mut reciprocal_ranks: Vec<f64> = Vec::new();

    for r in &results {
        let rank_str = match r.rank {
            Some(rank) => format!("#{}", rank),
            None => "MISS".to_string(),
        };

        let indicator = match r.rank {
            Some(1) => "++",
            Some(2..=3) => "+ ",
            Some(4..=5) => "~ ",
            Some(_) => "- ",
            None => "X ",
        };

        eprintln!(
            "{} {:<23} {:<6} {}",
            indicator, r.name, rank_str, r.top_result
        );

        match r.rank {
            Some(1) => {
                hit_at_1 += 1;
                hit_at_3 += 1;
                hit_at_5 += 1;
                reciprocal_ranks.push(1.0);
            }
            Some(2..=3) => {
                hit_at_3 += 1;
                hit_at_5 += 1;
                reciprocal_ranks.push(1.0 / r.rank.unwrap() as f64);
            }
            Some(4..=5) => {
                hit_at_5 += 1;
                reciprocal_ranks.push(1.0 / r.rank.unwrap() as f64);
            }
            Some(rank) => {
                reciprocal_ranks.push(1.0 / rank as f64);
            }
            None => {
                reciprocal_ranks.push(0.0);
            }
        }
    }

    let total = results.len();
    let mrr = reciprocal_ranks.iter().sum::<f64>() / total as f64;

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("SCORECARD");
    eprintln!("{}", "-".repeat(80));
    eprintln!(
        "  Hit@1:  {}/{} ({:.0}%)",
        hit_at_1,
        total,
        hit_at_1 as f64 / total as f64 * 100.0
    );
    eprintln!(
        "  Hit@3:  {}/{} ({:.0}%)",
        hit_at_3,
        total,
        hit_at_3 as f64 / total as f64 * 100.0
    );
    eprintln!(
        "  Hit@5:  {}/{} ({:.0}%)",
        hit_at_5,
        total,
        hit_at_5 as f64 / total as f64 * 100.0
    );
    eprintln!("  MRR:    {:.3}", mrr);
    eprintln!("{}", "=".repeat(80));

    // Print misses for debugging
    let misses: Vec<&QueryResult> = results
        .iter()
        .filter(|r| r.rank.is_none() || r.rank.unwrap() > 3)
        .collect();
    if !misses.is_empty() {
        eprintln!("\nMISSES / LOW RANKS (rank > 3):");
        for r in &misses {
            eprintln!("  [{}] \"{}\"", r.expected, r.query);
            eprintln!("    expected: {}", r.expected);
            eprintln!("    got rank: {:?}, top: {}", r.rank, r.top_result);
        }
    }

    // Optional threshold gate
    if let Ok(min_mrr_str) = std::env::var("ENGRAM_BENCH_MIN_MRR") {
        if let Ok(min_mrr) = min_mrr_str.parse::<f64>() {
            assert!(
                mrr >= min_mrr,
                "MRR {:.3} is below minimum threshold {:.3}",
                mrr,
                min_mrr
            );
        }
    }
}
