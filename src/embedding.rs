use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::error::MemoryError;
use crate::memory::MemoryType;

pub struct EmbeddingService {
    model: Arc<Mutex<TextEmbedding>>,
    model_version: String,
}

fn get_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("fastembed")
}

impl EmbeddingService {
    pub fn new() -> Result<Self, MemoryError> {
        let options =
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_cache_dir(get_cache_dir());
        let model =
            TextEmbedding::try_new(options).map_err(|e| MemoryError::Embedding(e.to_string()))?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_version: "all-MiniLM-L6-v2".to_string(),
        })
    }

    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let mut model = self
            .model
            .lock()
            .map_err(|e| MemoryError::Embedding(format!("Failed to acquire lock: {}", e)))?;

        let embeddings = model
            .embed(vec![text], None)
            .map_err(|e| MemoryError::Embedding(e.to_string()))?;

        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| MemoryError::Embedding("No embedding generated".to_string()))
    }

    pub fn embed_memory(
        &self,
        memory_type: MemoryType,
        content: &str,
    ) -> Result<Vec<f32>, MemoryError> {
        // Concatenate type with content for better context
        let text = format!("{}: {}", memory_type.as_str(), content);
        self.embed(&text)
    }

    #[allow(dead_code)] // Used by MCP server batch tools
    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, MemoryError> {
        let mut model = self
            .model
            .lock()
            .map_err(|e| MemoryError::Embedding(format!("Failed to acquire lock: {}", e)))?;

        model
            .embed(texts, None)
            .map_err(|e| MemoryError::Embedding(e.to_string()))
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
}
