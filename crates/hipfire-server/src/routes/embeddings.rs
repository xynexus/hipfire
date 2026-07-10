use axum::{
    extract::{Extension, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use hipfire_daemon_adapter::{EmbedRequest, RerankRequest};
use serde::Deserialize;
use serde_json::json;

use crate::routes::chat::ensure_model_loaded;
use crate::state::SharedState;

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub dims: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

impl EmbeddingInput {
    fn into_texts(self) -> Vec<String> {
        match self {
            Self::One(text) => vec![text],
            Self::Many(texts) => texts,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RerankHttpRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<RerankDocumentInput>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum RerankDocumentInput {
    Text(String),
    Object { text: String },
}

impl RerankDocumentInput {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) | Self::Object { text } => text,
        }
    }
}

pub async fn post_embeddings(
    State(state): State<SharedState>,
    accounting: Option<Extension<crate::accounting::RequestAccounting>>,
    Json(body): Json<EmbeddingsRequest>,
) -> Response {
    let model = body.model;
    let texts = body.input.into_texts();
    let input_tokens = estimate_text_tokens(texts.iter().map(String::as_str));
    let dims = body.dimensions.or(body.dims);
    if texts.iter().any(|text| text.is_empty()) {
        return bad_request("embedding input entries must be non-empty");
    }
    let loaded = match ensure_model_loaded(&state, &model, 0).await {
        Ok(loaded) => loaded,
        Err(e) => return server_error(format!("load failed: {e}")),
    };
    let mut engine_guard = state.engine.lock().await;
    let Some(engine) = engine_guard.as_mut() else {
        return server_error("daemon engine unavailable after model load");
    };
    let embeddings = match engine
        .embed(EmbedRequest {
            texts,
            dims,
            worker_key_id: loaded.worker_key_id,
        })
        .await
    {
        Ok(embeddings) => embeddings,
        Err(e) => return server_error(e.to_string()),
    };
    let data = embeddings
        .into_iter()
        .map(|item| {
            json!({
                "object": "embedding",
                "index": item.index,
                "embedding": item.embedding,
            })
        })
        .collect::<Vec<_>>();
    if let Some(Extension(accounting)) = accounting {
        accounting.report_text(input_tokens, 0, 0);
    }
    Json(json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": input_tokens,
            "total_tokens": input_tokens,
        },
    }))
    .into_response()
}

pub async fn post_rerank(
    State(state): State<SharedState>,
    accounting: Option<Extension<crate::accounting::RequestAccounting>>,
    Json(body): Json<RerankHttpRequest>,
) -> Response {
    let model = body.model;
    let query = body.query;
    let documents = body
        .documents
        .into_iter()
        .map(RerankDocumentInput::into_text)
        .collect::<Vec<_>>();
    let input_tokens = estimate_text_tokens(
        std::iter::once(query.as_str()).chain(documents.iter().map(String::as_str)),
    );
    if query.is_empty() {
        return bad_request("rerank query must be non-empty");
    }
    if documents.iter().any(|text| text.is_empty()) {
        return bad_request("rerank documents must be non-empty");
    }
    let loaded = match ensure_model_loaded(&state, &model, 0).await {
        Ok(loaded) => loaded,
        Err(e) => return server_error(format!("load failed: {e}")),
    };
    let mut engine_guard = state.engine.lock().await;
    let Some(engine) = engine_guard.as_mut() else {
        return server_error("daemon engine unavailable after model load");
    };
    let results = match engine
        .rerank(RerankRequest {
            query,
            documents: documents.clone(),
            worker_key_id: loaded.worker_key_id,
        })
        .await
    {
        Ok(results) => results,
        Err(e) => return server_error(e.to_string()),
    };
    let results = results
        .into_iter()
        .map(|item| {
            json!({
                "index": item.index,
                "relevance_score": item.relevance_score,
                "document": {
                    "text": documents.get(item.index).cloned().unwrap_or_default(),
                },
            })
        })
        .collect::<Vec<_>>();
    if let Some(Extension(accounting)) = accounting {
        accounting.report_text(input_tokens, 0, 0);
    }
    Json(json!({
        "object": "list",
        "results": results,
        "model": model,
    }))
    .into_response()
}

fn estimate_text_tokens<'a>(texts: impl Iterator<Item = &'a str>) -> u64 {
    texts
        .map(|text| (text.len() as u64).saturating_add(3) / 4)
        .sum()
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": message.into(),
            }
        })),
    )
        .into_response()
}

fn server_error(message: impl Into<String>) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "message": message.into(),
            }
        })),
    )
        .into_response()
}
