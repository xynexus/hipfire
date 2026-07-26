use axum::{
    extract::{Extension, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use hipfire_daemon_adapter::{EmbedRequest, EmbeddingVector, RerankRequest};
use hipfire_model::embedding::EmbeddingInputType;
use hipfire_scheduler::{
    server_prefill_batch_enabled, SchedulerPolicyEnv, WorkloadClass, WorkloadResources,
    WorkloadSpec,
};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::batch_runner::{EmbedJob, ScheduledJob};
use crate::routes::chat::ensure_model_loaded;
use crate::state::SharedState;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub dims: Option<usize>,
    #[serde(default)]
    pub input_type: EmbeddingInputType,
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
    let req = EmbedRequest {
        texts,
        input_type: body.input_type,
        dims,
        worker_key_id: loaded.worker_key_id,
    };

    // Route through the runner (the one GPU arbiter) when it is active, so embed
    // shares priority admission with text/image and never races the runner's
    // engine `take`. Kill switch off (HIPFIRE_SERVER_PREFILL_BATCH=0) falls back
    // to locking the engine directly.
    let embeddings: Vec<EmbeddingVector> =
        if server_prefill_batch_enabled(&SchedulerPolicyEnv::from_pairs(std::env::vars())) {
            let req_id = Uuid::new_v4().to_string();
            let (tx, rx) = tokio::sync::oneshot::channel();
            state
                .batch_inbox
                .lock()
                .await
                .insert(req_id.clone(), ScheduledJob::Embed(EmbedJob { req, tx }));
            // Embed is its own lease (no coalescing yet): distinct microbatch key,
            // size 1, Maintenance class. Priority 64 ~ interactive default.
            let workload = WorkloadSpec::microbatchable(
                req_id.clone(),
                WorkloadClass::Maintenance,
                64,
                now_ms(),
                WorkloadResources::default(),
                format!("embed:{model}"),
                1,
            );
            if let Err(e) = state.work_scheduler.lock().await.enqueue(workload) {
                state.batch_inbox.lock().await.remove(&req_id);
                return server_error(format!("scheduler admission: {e}"));
            }
            state.prefill_notify.notify_waiters();
            match rx.await {
                Ok(Ok(embeddings)) => embeddings,
                Ok(Err(e)) => return embedding_error(e),
                Err(_) => return server_error("embed job dropped before completion"),
            }
        } else {
            let mut engine_guard = state.engine.lock().await;
            let Some(engine) = engine_guard.as_mut() else {
                return server_error("daemon engine unavailable after model load");
            };
            match engine.embed(req).await {
                Ok(embeddings) => embeddings,
                Err(e) => return embedding_error(e.to_string()),
            }
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

fn embedding_error(message: String) -> Response {
    if message.contains("maximum supported length")
        || message.contains("no compiled sequence bucket")
        || message.contains("unsupported embedding dimensions")
        || message.contains("must be non-empty")
    {
        bad_request(message)
    } else {
        server_error(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_input_type_defaults_to_document_and_accepts_query() {
        let default: EmbeddingsRequest = serde_json::from_value(json!({
            "model": "embedding",
            "input": ["one", "two"]
        }))
        .unwrap();
        assert_eq!(default.input_type, EmbeddingInputType::Document);
        assert!(matches!(default.input, EmbeddingInput::Many(values) if values.len() == 2));

        let query: EmbeddingsRequest = serde_json::from_value(json!({
            "model": "embedding",
            "input": "question",
            "input_type": "query",
            "dimensions": 256
        }))
        .unwrap();
        assert_eq!(query.input_type, EmbeddingInputType::Query);
        assert_eq!(query.dimensions, Some(256));
    }

    #[test]
    fn embedding_input_type_rejects_unknown_roles() {
        let error = serde_json::from_value::<EmbeddingsRequest>(json!({
            "model": "embedding",
            "input": "text",
            "input_type": "passage"
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown variant"));
    }

    #[test]
    fn oversized_embedding_errors_are_client_errors() {
        assert_eq!(
            embedding_error(
                "embedding input has 2049 tokens; maximum supported length is 2048".into()
            )
            .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            embedding_error("missing NPU image".into()).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
