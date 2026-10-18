//! Text embeddings: [mdbr-leaf-ir](https://huggingface.co/onnx-community/mdbr-leaf-ir-ONNX),
//! quantized, CLS-pooled, cut to 256 dimensions (Matryoshka truncation) and normalized.
//!
//! [`EmbeddingService::new`] runs the model in-process. [`EmbeddingService::shared`] asks one
//! per-user embedding process (`embed_server`) instead, so concurrent sessions hold one copy
//! of the model rather than one each, and falls back to the in-process model when that
//! process cannot be reached.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use tokenizers::{AddedToken, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::embed_server;
use crate::error::MemoryError;
use crate::memory::MemoryType;

/// Target embedding dimension (MRL truncation from the model's 384).
const EMBED_DIM: usize = 256;

/// Identifies the vector space stored embeddings belong to. Vectors from a different value
/// are not comparable and are re-embedded on import.
pub const MODEL_VERSION: &str = "mdbr-leaf-ir-q8-d256";

const QUERY_PREFIX: &str = "search_query: ";
const DOCUMENT_PREFIX: &str = "search_document: ";

/// Texts per inference call. Matches the batch size the vectors were originally produced
/// with: padding is to the longest text in a batch, so a different split changes the
/// padded inputs.
const BATCH_SIZE: usize = 256;

/// Tokens per text after truncation; the model's `model_max_length`.
const MAX_LENGTH: usize = 512;

/// The loaded ONNX session and its tokenizer.
pub struct Model {
    session: Session,
    tokenizer: Tokenizer,
    need_token_type_ids: bool,
}

struct ModelFiles {
    dir: PathBuf,
    onnx_model: PathBuf,
}

fn embedding_error(context: &str) -> impl Fn(String) -> MemoryError + '_ {
    move |e| MemoryError::Embedding(format!("{context}: {e}"))
}

/// Download the model files into the Hugging Face cache, or find them there.
fn model_files() -> Result<ModelFiles, MemoryError> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| embedding_error("Failed to init HF API")(e.to_string()))?;
    let repo = api.model("onnx-community/mdbr-leaf-ir-ONNX".to_string());
    let get = |file: &str| {
        repo.get(file)
            .map_err(|e| MemoryError::Embedding(format!("Failed to download {file}: {e}")))
    };
    let onnx_model = get("onnx/model_quantized.onnx")?;
    // The model references its weights as an external file next to it; ONNX Runtime opens
    // it from there, so it only has to be present.
    get("onnx/model_quantized.onnx_data")?;
    let tokenizer = get("tokenizer.json")?;
    for file in [
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ] {
        get(file)?;
    }
    let dir = tokenizer
        .parent()
        .ok_or_else(|| MemoryError::Embedding("model cache has no parent directory".into()))?
        .to_path_buf();
    Ok(ModelFiles { dir, onnx_model })
}

fn read_json(path: PathBuf) -> Result<serde_json::Value, MemoryError> {
    let bytes = std::fs::read(&path)
        .map_err(|e| MemoryError::Embedding(format!("Failed to read {}: {e}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| MemoryError::Embedding(format!("Failed to parse {}: {e}", path.display())))
}

/// The tokenizer with batch-longest padding, truncation at `model_max_length`, and the
/// special tokens from `special_tokens_map.json` registered.
fn load_tokenizer(dir: &std::path::Path) -> Result<Tokenizer, MemoryError> {
    let config = read_json(dir.join("config.json"))?;
    let tokenizer_config = read_json(dir.join("tokenizer_config.json"))?;
    let special_tokens = read_json(dir.join("special_tokens_map.json"))?;

    let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| embedding_error("Failed to load tokenizer")(e.to_string()))?;
    let max_length = tokenizer_config["model_max_length"]
        .as_f64()
        .map_or(MAX_LENGTH, |n| MAX_LENGTH.min(n as usize));
    let pad_token = tokenizer_config["pad_token"]
        .as_str()
        .ok_or_else(|| MemoryError::Embedding("tokenizer_config.json has no pad_token".into()))?
        .to_string();
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_token,
            pad_id: config["pad_token_id"].as_u64().unwrap_or(0) as u32,
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| embedding_error("Failed to configure truncation")(e.to_string()))?;

    if let serde_json::Value::Object(map) = special_tokens {
        for value in map.values() {
            let token = match value {
                serde_json::Value::String(content) => Some(AddedToken {
                    content: content.clone(),
                    special: true,
                    ..Default::default()
                }),
                serde_json::Value::Object(_) => match (
                    value["content"].as_str(),
                    value["single_word"].as_bool(),
                    value["lstrip"].as_bool(),
                    value["rstrip"].as_bool(),
                    value["normalized"].as_bool(),
                ) {
                    (
                        Some(content),
                        Some(single_word),
                        Some(lstrip),
                        Some(rstrip),
                        Some(normalized),
                    ) => Some(AddedToken {
                        content: content.to_string(),
                        special: true,
                        single_word,
                        lstrip,
                        rstrip,
                        normalized,
                    }),
                    _ => None,
                },
                _ => None,
            };
            if let Some(token) = token {
                tokenizer.add_special_tokens(&[token]);
            }
        }
    }
    Ok(tokenizer)
}

/// L2-normalize, with the epsilon the stored vectors were produced with.
fn normalize_eps(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / (norm + 1e-12)).collect()
}

/// Truncate embedding to target dimension and L2-normalize.
fn truncate_and_normalize(embedding: &mut Vec<f32>) {
    embedding.truncate(EMBED_DIM);
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }
}

impl Model {
    /// Load the model from the Hugging Face cache, downloading it on first use.
    ///
    /// The session is built from the file path rather than from bytes in memory: handed a
    /// buffer, ONNX Runtime keeps its own copy of the weights next to the caller's, which
    /// measured as 29 MB more peak and 15 MB more resident memory per process.
    pub fn load() -> Result<Self, MemoryError> {
        let files = model_files()?;
        let ort_error = |e: ort::Error| embedding_error("Failed to load model")(e.to_string());
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let session = Session::builder()
            .map_err(ort_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ort_error(e.into()))?
            .with_intra_threads(threads)
            .map_err(|e| ort_error(e.into()))?
            .commit_from_file(&files.onnx_model)
            .map_err(ort_error)?;
        let need_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        Ok(Self {
            session,
            tokenizer: load_tokenizer(&files.dir)?,
            need_token_type_ids,
        })
    }

    /// Embed already-prefixed texts: CLS pooling, normalization, then truncation to
    /// [`EMBED_DIM`] and normalization again.
    pub fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, MemoryError> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH_SIZE) {
            out.extend(self.embed_batch(batch)?);
        }
        Ok(out)
    }

    fn embed_batch(&mut self, batch: &[String]) -> Result<Vec<Vec<f32>>, MemoryError> {
        let fail = |e: String| MemoryError::Embedding(e);
        let encodings = self
            .tokenizer
            .encode_batch(batch.iter().map(String::as_str).collect::<Vec<_>>(), true)
            .map_err(|e| fail(e.to_string()))?;
        let len = encodings.first().map_or(0, |e| e.len());
        let shape = (batch.len(), len);
        let column = |field: fn(&tokenizers::Encoding) -> &[u32]| {
            let flat: Vec<i64> = encodings
                .iter()
                .flat_map(|e| field(e).iter().map(|&x| x as i64))
                .collect();
            Array2::from_shape_vec(shape, flat).map_err(|e| fail(e.to_string()))
        };
        let ids = column(tokenizers::Encoding::get_ids)?;
        let mask = column(tokenizers::Encoding::get_attention_mask)?;
        let types = column(tokenizers::Encoding::get_type_ids)?;

        let value = |a: Array2<i64>| Value::from_array(a).map_err(|e| fail(e.to_string()));
        let mut inputs = ort::inputs![
            "input_ids" => value(ids)?,
            "attention_mask" => value(mask)?,
        ];
        if self.need_token_type_ids {
            inputs.push(("token_type_ids".into(), value(types)?.into()));
        }
        let outputs = self.session.run(inputs).map_err(|e| fail(e.to_string()))?;

        // The sole output, else the first of these names, as the vectors were made with.
        let output = if outputs.len() == 1 {
            Some(&outputs[0])
        } else {
            ["text_embeds", "last_hidden_state", "sentence_embedding"]
                .iter()
                .find_map(|name| outputs.get(*name))
        }
        .ok_or_else(|| fail("model has no embedding output".to_string()))?;
        let tensor = output
            .try_extract_array::<f32>()
            .map_err(|e| fail(e.to_string()))?;
        let pooled = match tensor.ndim() {
            2 => tensor.to_owned(),
            3 => tensor.index_axis(Axis(1), 0).to_owned().into_dyn(),
            n => return Err(fail(format!("unexpected {n}-dimensional model output"))),
        };
        Ok(pooled
            .rows()
            .into_iter()
            .map(|row| {
                let mut v = normalize_eps(&row.to_vec());
                truncate_and_normalize(&mut v);
                v
            })
            .collect())
    }
}

struct Inner {
    /// Loaded on first use: in shared mode only when the shared process cannot answer.
    local: Mutex<Option<Model>>,
    shared: Option<embed_server::Client>,
}

/// Cloning shares the model (or the connection settings), never reloads it.
#[derive(Clone)]
pub struct EmbeddingService {
    inner: Arc<Inner>,
}

impl EmbeddingService {
    /// Run the model in this process, loading it now.
    pub fn new() -> Result<Self, MemoryError> {
        Ok(Self {
            inner: Arc::new(Inner {
                local: Mutex::new(Some(Model::load()?)),
                shared: None,
            }),
        })
    }

    /// Use the shared embedding process, starting it if needed. Loads nothing up front;
    /// the model is loaded in-process only if the shared process cannot be reached. For
    /// the `engram` and `engram-cli` binaries, which can run that process; other callers
    /// use [`EmbeddingService::new`]. `ENGRAM_EMBED_SHARED=0` turns it off.
    pub fn shared() -> Result<Self, MemoryError> {
        match embed_server::Client::from_env() {
            Some(client) => Ok(Self {
                inner: Arc::new(Inner {
                    local: Mutex::new(None),
                    shared: Some(client),
                }),
            }),
            None => Self::new(),
        }
    }

    pub fn model_version(&self) -> &str {
        MODEL_VERSION
    }

    fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, MemoryError> {
        if let Some(client) = &self.inner.shared {
            match client.embed(&texts) {
                Ok(vectors) => return Ok(vectors),
                Err(e) => tracing::warn!(
                    "shared embedding process unavailable ({e}); embedding in-process"
                ),
            }
        }
        let mut local = self
            .inner
            .local
            .lock()
            .map_err(|e| MemoryError::Embedding(format!("Failed to acquire lock: {e}")))?;
        if local.is_none() {
            *local = Some(Model::load()?);
        }
        local.as_mut().expect("loaded above").embed(&texts)
    }

    fn embed_raw(&self, text: String) -> Result<Vec<f32>, MemoryError> {
        self.embed_texts(vec![text])?
            .into_iter()
            .next()
            .ok_or_else(|| MemoryError::Embedding("No embedding generated".to_string()))
    }

    /// Embed a search query (adds query prefix for asymmetric retrieval).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        self.embed_raw(format!("{QUERY_PREFIX}{text}"))
    }

    /// Embed a memory for storage (adds document prefix for asymmetric retrieval).
    pub fn embed_memory(
        &self,
        memory_type: MemoryType,
        content: &str,
    ) -> Result<Vec<f32>, MemoryError> {
        self.embed_raw(format!(
            "{DOCUMENT_PREFIX}{}: {content}",
            memory_type.as_str()
        ))
    }

    /// Embed text using the same document prefix as `embed_memory` for a given type.
    ///
    /// Used by `auto_link_handoff_sections` to compare section text against target-type
    /// memory embeddings in a shared vector space.  The prefix matches the one applied
    /// during storage so similarity scores are meaningful across types.
    pub fn embed_memory_text(
        &self,
        memory_type: MemoryType,
        text: &str,
    ) -> Result<Vec<f32>, MemoryError> {
        self.embed_memory(memory_type, text)
    }

    #[allow(dead_code)] // Used by MCP server batch tools
    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, MemoryError> {
        self.embed_texts(
            texts
                .into_iter()
                .map(|t| format!("{DOCUMENT_PREFIX}{t}"))
                .collect(),
        )
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_truncate_and_normalize() {
        let mut v2 = vec![3.0, 4.0];
        let norm = (3.0_f32 * 3.0 + 4.0 * 4.0).sqrt(); // 5.0
        truncate_and_normalize(&mut v2); // won't truncate since len < EMBED_DIM
        // After normalize: [0.6, 0.8]
        assert!((v2[0] - 3.0 / norm).abs() < 0.001);
        assert!((v2[1] - 4.0 / norm).abs() < 0.001);
    }

    /// Vectors already in every store were made by this pipeline; a change to
    /// tokenization, pooling or normalization would silently make new ones incomparable.
    /// Reference values recorded from it (identical to fastembed 5.17 at the time).
    #[test]
    fn vectors_match_the_stored_vector_space() {
        let mut model = Model::load().unwrap();
        let texts = [
            "search_query: how does authentication work".to_string(),
            "search_document: decision: Chose SQLite over Postgres because one writer".to_string(),
        ];
        let vectors = model.embed(&texts).unwrap();
        let expected: [[f32; 4]; 2] = [
            [-0.028242199, 0.026272101, 0.023451738, 0.00312708],
            [-0.00750964, 0.014411634, 0.026033089, 0.019142812],
        ];
        for (v, e) in vectors.iter().zip(expected) {
            assert_eq!(v.len(), EMBED_DIM);
            for (a, b) in v.iter().zip(e) {
                assert!((a - b).abs() < 1e-5, "{:?} vs {e:?}", &v[..4]);
            }
        }
    }
}
