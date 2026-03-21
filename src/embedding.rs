use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::error::MemoryError;
use crate::memory::MemoryType;

/// Target embedding dimension (MRL truncation from 1024)
const EMBED_DIM: usize = 256;

const QUERY_PREFIX: &str = "search_query: ";
const DOCUMENT_PREFIX: &str = "search_document: ";

pub struct EmbeddingService {
    model: Arc<Mutex<TextEmbedding>>,
    model_version: String,
}

struct ModelFiles {
    onnx_model: PathBuf,
    onnx_data: PathBuf,
    tokenizer: PathBuf,
    config: PathBuf,
    special_tokens_map: PathBuf,
    tokenizer_config: PathBuf,
}

/// Download model files from HuggingFace and return paths to each cached file.
fn download_model_files() -> Result<ModelFiles, MemoryError> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| MemoryError::Embedding(format!("Failed to init HF API: {}", e)))?;
    let repo = api.model("onnx-community/mdbr-leaf-ir-ONNX".to_string());

    let map_err = |file: &str| {
        let file = file.to_string();
        move |e: hf_hub::api::sync::ApiError| {
            MemoryError::Embedding(format!("Failed to download {}: {}", file, e))
        }
    };

    Ok(ModelFiles {
        onnx_model: repo
            .get("onnx/model_quantized.onnx")
            .map_err(map_err("onnx/model_quantized.onnx"))?,
        onnx_data: repo
            .get("onnx/model_quantized.onnx_data")
            .map_err(map_err("onnx/model_quantized.onnx_data"))?,
        tokenizer: repo
            .get("tokenizer.json")
            .map_err(map_err("tokenizer.json"))?,
        config: repo.get("config.json").map_err(map_err("config.json"))?,
        special_tokens_map: repo
            .get("special_tokens_map.json")
            .map_err(map_err("special_tokens_map.json"))?,
        tokenizer_config: repo
            .get("tokenizer_config.json")
            .map_err(map_err("tokenizer_config.json"))?,
    })
}

fn load_model(files: &ModelFiles) -> Result<TextEmbedding, MemoryError> {
    let onnx_file = std::fs::read(&files.onnx_model)
        .map_err(|e| MemoryError::Embedding(format!("Failed to read ONNX model: {}", e)))?;
    let onnx_data = std::fs::read(&files.onnx_data)
        .map_err(|e| MemoryError::Embedding(format!("Failed to read ONNX data: {}", e)))?;

    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(&files.tokenizer)
            .map_err(|e| MemoryError::Embedding(format!("Failed to read tokenizer: {}", e)))?,
        config_file: std::fs::read(&files.config)
            .map_err(|e| MemoryError::Embedding(format!("Failed to read config: {}", e)))?,
        special_tokens_map_file: std::fs::read(&files.special_tokens_map)
            .map_err(|e| MemoryError::Embedding(format!("Failed to read special tokens: {}", e)))?,
        tokenizer_config_file: std::fs::read(&files.tokenizer_config).map_err(|e| {
            MemoryError::Embedding(format!("Failed to read tokenizer config: {}", e))
        })?,
    };

    let user_model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files)
        .with_pooling(Pooling::Cls)
        .with_external_initializer("model_quantized.onnx_data".to_string(), onnx_data);

    let options = InitOptionsUserDefined::new();

    TextEmbedding::try_new_from_user_defined(user_model, options)
        .map_err(|e| MemoryError::Embedding(format!("Failed to load model: {}", e)))
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

impl EmbeddingService {
    pub fn new() -> Result<Self, MemoryError> {
        let files = download_model_files()?;
        let model = load_model(&files)?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_version: "mdbr-leaf-ir-q8-d256".to_string(),
        })
    }

    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    /// Embed raw text without any prefix.
    fn embed_raw(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let mut model = self
            .model
            .lock()
            .map_err(|e| MemoryError::Embedding(format!("Failed to acquire lock: {}", e)))?;

        let embeddings = model
            .embed(vec![text], None)
            .map_err(|e| MemoryError::Embedding(e.to_string()))?;

        let mut embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| MemoryError::Embedding("No embedding generated".to_string()))?;

        truncate_and_normalize(&mut embedding);
        Ok(embedding)
    }

    /// Embed a search query (adds query prefix for asymmetric retrieval).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let prefixed = format!("{}{}", QUERY_PREFIX, text);
        self.embed_raw(&prefixed)
    }

    /// Embed a memory for storage (adds document prefix for asymmetric retrieval).
    pub fn embed_memory(
        &self,
        memory_type: MemoryType,
        content: &str,
    ) -> Result<Vec<f32>, MemoryError> {
        let text = format!("{}{}: {}", DOCUMENT_PREFIX, memory_type.as_str(), content);
        self.embed_raw(&text)
    }

    #[allow(dead_code)] // Used by MCP server batch tools
    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, MemoryError> {
        let prefixed: Vec<String> = texts
            .into_iter()
            .map(|t| format!("{}{}", DOCUMENT_PREFIX, t))
            .collect();

        let mut model = self
            .model
            .lock()
            .map_err(|e| MemoryError::Embedding(format!("Failed to acquire lock: {}", e)))?;

        let mut embeddings = model
            .embed(prefixed, None)
            .map_err(|e| MemoryError::Embedding(e.to_string()))?;

        for embedding in &mut embeddings {
            truncate_and_normalize(embedding);
        }

        Ok(embeddings)
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
}
