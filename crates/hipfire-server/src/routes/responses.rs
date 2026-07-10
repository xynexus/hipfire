use axum::{
    extract::{Extension, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};
use hipfire_prompt::{Message, Role};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::routes::chat::{
    effective_request_max_tokens, ensure_model_loaded, execute_blocking_chat_owned,
    extract_request_image_base64, generate_request_from_chat, normalize_stop_sequences,
    request_generation_controls, required_load_max_seq, scheduler_owner_from_principal,
    strip_visible_thinking, wait_for_prefill_scheduler_turn, AssistantDelta, ChatMessage,
    ChatRequest, ThinkStreamFilter,
};
use crate::state::{SharedState, StoredResponsesContext};
use hipfire_auth::{RequestPrincipal, ResponseContextRecord};
use hipfire_daemon_adapter::{GenerateStreamControl, GenerateStreamEvent};
use hipfire_generate::GenerationSamplingPolicy;

const AUTHENTICATED_RESPONSE_TTL_SECS: u64 = 30 * 24 * 60 * 60;
const AUTHENTICATED_RESPONSE_MAX: usize = 128;
const RESPONSE_CHAIN_MAX_DEPTH: usize = 128;

#[derive(Debug, Clone)]
enum ResponsesOwner {
    AnonymousLocal,
    User(String),
}

impl ResponsesOwner {
    fn from_principal(principal: Option<Extension<RequestPrincipal>>) -> Self {
        principal
            .and_then(|Extension(principal)| principal.user_id)
            .map(Self::User)
            .unwrap_or(Self::AnonymousLocal)
    }
}

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    pub input: Value,
    pub previous_response_id: Option<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub max_output_tokens: Option<u32>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    pub reasoning_effort: Option<String>,
    pub reasoning: Option<Value>,
    pub chat_template_kwargs: Option<Value>,
}

pub async fn post_responses(
    State(state): State<SharedState>,
    principal: Option<Extension<RequestPrincipal>>,
    accounting: Option<Extension<crate::accounting::RequestAccounting>>,
    Json(body): Json<ResponsesRequest>,
) -> Response {
    let accounting = accounting.map(|Extension(accounting)| accounting);
    let scheduler_owner = principal
        .as_ref()
        .map(|Extension(principal)| scheduler_owner_from_principal(principal))
        .unwrap_or_default();
    let owner = ResponsesOwner::from_principal(principal);
    if body.stream {
        return stream_responses(state, body, owner, scheduler_owner, accounting)
            .await
            .into_response();
    }
    match execute_responses_owned(state, body, owner, scheduler_owner).await {
        Ok(body) => {
            if let Some(accounting) = &accounting {
                report_response_usage(accounting, &body);
            }
            Json(body).into_response()
        }
        Err(error) => (error_status(&error), Json(error)).into_response(),
    }
}

fn error_status(error: &Value) -> StatusCode {
    if error
        .get("error")
        .and_then(|inner| inner.get("type"))
        .and_then(Value::as_str)
        == Some("invalid_request_error")
    {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

pub(crate) async fn execute_responses(
    state: SharedState,
    body: ResponsesRequest,
) -> Result<Value, Value> {
    execute_responses_owned(
        state,
        body,
        ResponsesOwner::AnonymousLocal,
        hipfire_scheduler::WorkloadOwner::default(),
    )
    .await
}

async fn execute_responses_owned(
    state: SharedState,
    body: ResponsesRequest,
    owner: ResponsesOwner,
    scheduler_owner: hipfire_scheduler::WorkloadOwner,
) -> Result<Value, Value> {
    let messages = prepare_response_messages(&state, &body, &owner).await?;

    let mut chat_body = ChatRequest {
        model: body.model.clone(),
        messages: messages
            .iter()
            .map(prompt_message_to_chat_message)
            .collect::<Vec<_>>(),
        stream: false,
        temperature: body.temperature,
        top_p: body.top_p,
        repeat_penalty: body.repeat_penalty,
        presence_penalty: body.presence_penalty,
        frequency_penalty: body.frequency_penalty,
        max_tokens: body.max_output_tokens.or(body.max_tokens),
        stop: body.stop.clone(),
        priority: None,
        tools: None,
        system: None,
        reasoning_effort: body.reasoning_effort.clone(),
        reasoning: body.reasoning.clone(),
        stream_options: None,
        chat_template_kwargs: body.chat_template_kwargs.clone(),
    };
    // Forward the current turn's image (if any) so the chat vision path and its
    // embedding cache pick it up; non-stream extraction happens inside
    // execute_blocking_chat.
    apply_vision_content(&mut chat_body.messages, &body.input);

    let generated =
        match execute_blocking_chat_owned(state.clone(), chat_body, scheduler_owner).await {
            Ok(generated) => generated,
            Err(error) => return Err(error),
        };

    let response_id = format!("resp_{}", Uuid::new_v4().simple());
    let mut stored = messages;
    stored.push(Message {
        role: Role::Assistant,
        content: generated.text.clone(),
        tool_calls: Vec::new(),
        tool_call_id: None,
    });
    store_responses_context(
        &state,
        &owner,
        response_id.clone(),
        body.previous_response_id.clone(),
        stored,
    )
    .await?;

    Ok(response_json(
        &response_id,
        &generated.model,
        &generated.text,
        &generated.done,
    ))
}

async fn prepare_response_messages(
    state: &SharedState,
    body: &ResponsesRequest,
    owner: &ResponsesOwner,
) -> Result<Vec<Message>, Value> {
    let mut messages = match responses_input_to_chat_messages(&body.input) {
        Ok(messages) => messages,
        Err(message) => {
            return Err(json!({"error": {"message": message, "type": "invalid_request_error"}}));
        }
    };

    if let Some(previous_id) = &body.previous_response_id {
        match load_responses_context(state, owner, previous_id).await {
            Ok(Some(mut previous)) => {
                previous.extend(messages);
                messages = previous;
            }
            Ok(None) => {
                return Err(json!({
                    "error": {
                        "message": format!("previous_response_id not found: {previous_id}"),
                        "type": "invalid_request_error"
                    }
                }));
            }
            Err(error) => return Err(server_context_error(error)),
        }
    }
    Ok(messages)
}

fn response_json(
    response_id: &str,
    model: &str,
    text: &str,
    done: &hipfire_generate::DoneEvent,
) -> Value {
    let prompt_tokens = done.prefill_tokens.unwrap_or(0);
    let completion_tokens = done.tokens;
    json!({
        "id": response_id,
        "object": "response",
        "model": model,
        "status": "completed",
        "output": [{
            "id": format!("msg_{response_id}"),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": []
            }]
        }],
        "output_text": text,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
}

async fn stream_responses(
    state: SharedState,
    body: ResponsesRequest,
    owner: ResponsesOwner,
    scheduler_owner: hipfire_scheduler::WorkloadOwner,
    accounting: Option<crate::accounting::RequestAccounting>,
) -> impl IntoResponse {
    let (tx, mut rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let response_id = format!("resp_{}", Uuid::new_v4().simple());
        let message_id = format!("msg_{response_id}");
        let req_id = response_id.clone();

        let messages = match prepare_response_messages(&state, &body, &owner).await {
            Ok(messages) => messages,
            Err(error) => {
                let _ = tx.send(Ok(sse_json_event("error", error))).await;
                return;
            }
        };

        let model_arg = {
            let cfg = state.config.lock().await;
            body.model.clone().or(cfg.default_model.clone())
        };
        let Some(model_arg) = model_arg else {
            let _ = tx
                .send(Ok(sse_json_event(
                    "error",
                    json!({"error": {"message": "no model specified", "type": "invalid_request_error"}}),
                )))
                .await;
            return;
        };

        let stop = match normalize_stop_sequences(body.stop.as_ref()) {
            Ok(stop) => stop,
            Err(message) => {
                let _ = tx
                    .send(Ok(sse_json_event(
                        "error",
                        json!({"error": {"message": message, "type": "invalid_request_error"}}),
                    )))
                    .await;
                return;
            }
        };

        let mut chat_messages = messages
            .iter()
            .map(prompt_message_to_chat_message)
            .collect::<Vec<_>>();
        apply_vision_content(&mut chat_messages, &body.input);
        let image_base64 = match extract_request_image_base64(&chat_messages) {
            Ok(image) => image,
            Err(message) => {
                let _ = tx
                    .send(Ok(sse_json_event(
                        "error",
                        json!({"error": {"message": message, "type": "invalid_request_error"}}),
                    )))
                    .await;
                return;
            }
        };

        let (request_max_tokens, required_max_seq) = {
            let cfg = state.config.lock().await;
            let requested = body.max_output_tokens.or(body.max_tokens);
            let request_max_tokens = effective_request_max_tokens(cfg.max_tokens, requested);
            (
                request_max_tokens,
                required_load_max_seq(cfg.max_seq, request_max_tokens, image_base64.is_some()),
            )
        };

        let loaded = match ensure_model_loaded(&state, &model_arg, required_max_seq).await {
            Ok(loaded) => loaded,
            Err(e) => {
                let _ = tx
                    .send(Ok(sse_json_event(
                        "error",
                        json!({"error": {"message": e, "type": "server_error"}}),
                    )))
                    .await;
                return;
            }
        };

        if let Err(e) = wait_for_prefill_scheduler_turn(
            &state,
            &req_id,
            &loaded.model_path,
            &chat_messages,
            None,
            scheduler_owner,
        )
        .await
        {
            let _ = tx
                .send(Ok(sse_json_event(
                    "error",
                    json!({"error": {"message": e, "type": "server_error"}}),
                )))
                .await;
            return;
        }

        let gen_req = {
            let cfg = state.config.lock().await;
            let controls = request_generation_controls(
                &cfg,
                body.chat_template_kwargs.as_ref(),
                body.reasoning_effort.as_deref(),
                body.reasoning.as_ref(),
                body.presence_penalty,
                body.frequency_penalty,
            );
            generate_request_from_chat(
                req_id.clone(),
                &chat_messages,
                GenerationSamplingPolicy::from_defaults(
                    cfg.temperature,
                    cfg.top_p,
                    cfg.repeat_penalty,
                    cfg.max_tokens,
                    body.temperature,
                    body.top_p,
                    body.repeat_penalty,
                    Some(request_max_tokens),
                ),
                loaded.worker_key_id,
                None,
                None,
                stop,
                image_base64,
                controls,
            )
        };

        let _ = tx
            .send(Ok(sse_json_event(
                "response.created",
                response_created_json(&response_id, &model_arg),
            )))
            .await;
        let _ = tx
            .send(Ok(sse_json_event(
                "response.output_item.added",
                response_output_item_added_json(&response_id, &message_id),
            )))
            .await;

        let mut engine_guard = state.engine.lock().await;
        let mut engine = match engine_guard.take() {
            Some(engine) => engine,
            None => {
                let _ = tx
                    .send(Ok(sse_json_event(
                        "error",
                        json!({"error": {"message": "daemon not running", "type": "server_error"}}),
                    )))
                    .await;
                return;
            }
        };

        if !loaded.cache_capable {
            if let Err(e) = engine.reset().await {
                *engine_guard = Some(engine);
                let _ = tx
                    .send(Ok(sse_json_event(
                        "error",
                        json!({"error": {"message": e.to_string(), "type": "server_error"}}),
                    )))
                    .await;
                return;
            }
        }

        let mut output_text = String::new();
        let mut think_filter = ThinkStreamFilter::default();
        let result = engine
            .generate_streaming_events_controlled(gen_req, |event| {
                if tx.is_closed() {
                    return GenerateStreamControl::Cancel;
                }
                if let GenerateStreamEvent::Token(token) = event {
                    for delta in think_filter.observe(&token, false) {
                        if let AssistantDelta::Content(text) = delta {
                            output_text.push_str(&text);
                            let event =
                                response_output_text_delta_json(&response_id, &message_id, &text);
                            match tx
                                .try_send(Ok(sse_json_event("response.output_text.delta", event)))
                            {
                                Ok(()) => {}
                                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                                    return GenerateStreamControl::Cancel;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {}
                            }
                        }
                    }
                }
                GenerateStreamControl::Continue
            })
            .await;

        match result {
            Ok(Some(done)) => {
                if let Some(accounting) = &accounting {
                    accounting.report_text(
                        done.prefill_tokens.unwrap_or(0) as u64,
                        done.tokens as u64,
                        0,
                    );
                }
                *engine_guard = Some(engine);
                let output_text = strip_visible_thinking(output_text, false, true);
                let mut stored = messages;
                stored.push(Message {
                    role: Role::Assistant,
                    content: output_text.clone(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                });
                if let Err(error) = store_responses_context(
                    &state,
                    &owner,
                    response_id.clone(),
                    body.previous_response_id.clone(),
                    stored,
                )
                .await
                {
                    let _ = tx.send(Ok(sse_json_event("error", error))).await;
                    return;
                }

                let _ = tx
                    .send(Ok(sse_json_event(
                        "response.output_text.done",
                        response_output_text_done_json(&response_id, &message_id, &output_text),
                    )))
                    .await;
                let _ = tx
                    .send(Ok(sse_json_event(
                        "response.output_item.done",
                        response_output_item_done_json(&response_id, &message_id, &output_text),
                    )))
                    .await;
                let _ = tx
                    .send(Ok(sse_json_event(
                        "response.completed",
                        response_json(&response_id, &model_arg, &output_text, &done),
                    )))
                    .await;
            }
            Ok(None) => {
                tracing::info!(
                    response_id = %response_id,
                    "responses stream client disconnected; dropping daemon"
                );
                *state.loaded_model_path.lock().await = None;
                *state.loaded_model_cache_capable.lock().await = None;
                *state.loaded_model_max_seq.lock().await = None;
                drop(engine);
            }
            Err(e) => {
                *engine_guard = Some(engine);
                let _ = tx
                    .send(Ok(sse_json_event(
                        "error",
                        json!({"error": {"message": e.to_string(), "type": "server_error"}}),
                    )))
                    .await;
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(item) = rx.recv().await {
            yield item;
        }
    };
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(10))
            .text("prefill"),
    )
}

fn report_response_usage(accounting: &crate::accounting::RequestAccounting, body: &Value) {
    let usage = &body["usage"];
    accounting.report_text(
        usage["input_tokens"].as_u64().unwrap_or(0),
        usage["output_tokens"].as_u64().unwrap_or(0),
        0,
    );
}

fn sse_json_event(event: &str, data: Value) -> Event {
    Event::default()
        .event(event)
        .data(serde_json::to_string(&data).unwrap())
}

fn response_created_json(response_id: &str, model: &str) -> Value {
    json!({
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "model": model,
            "status": "in_progress",
            "output": [],
        }
    })
}

fn response_output_item_added_json(response_id: &str, message_id: &str) -> Value {
    json!({
        "type": "response.output_item.added",
        "response_id": response_id,
        "output_index": 0,
        "item": {
            "id": message_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
    })
}

fn response_output_text_delta_json(response_id: &str, message_id: &str, delta: &str) -> Value {
    json!({
        "type": "response.output_text.delta",
        "response_id": response_id,
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    })
}

fn response_output_text_done_json(response_id: &str, message_id: &str, text: &str) -> Value {
    json!({
        "type": "response.output_text.done",
        "response_id": response_id,
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "text": text,
    })
}

fn response_output_item_done_json(response_id: &str, message_id: &str, text: &str) -> Value {
    json!({
        "type": "response.output_item.done",
        "response_id": response_id,
        "output_index": 0,
        "item": {
            "id": message_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": []
            }],
        }
    })
}

/// The raw content of the last user item in `input`, returned only when it
/// carries an image part. Forwarding this (rather than the flattened text) to
/// the chat path is what gives the Responses API vision — and, downstream, the
/// shared vision-embedding cache.
fn last_user_image_content(input: &Value) -> Option<Value> {
    let items = match input {
        Value::Array(items) => items.as_slice(),
        Value::Object(obj) => obj.get("messages").and_then(Value::as_array)?.as_slice(),
        _ => return None,
    };
    let last_user = items
        .iter()
        .rev()
        .find(|it| it.get("role").and_then(Value::as_str) == Some("user"))?;
    let content = last_user.get("content")?;
    let has_image = content
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .any(|p| p.get("type").and_then(Value::as_str) == Some("image_url"))
        })
        .unwrap_or(false);
    has_image.then(|| content.clone())
}

/// Override the last user message's content with the original multimodal input
/// (text + image parts) so the chat vision path can extract the image. Image
/// constraints match chat: a single png/jpeg data URL in the last user turn.
fn apply_vision_content(messages: &mut [ChatMessage], input: &Value) {
    if let Some(content) = last_user_image_content(input) {
        if let Some(message) = messages.iter_mut().rev().find(|m| m.role == "user") {
            message.content = Some(content);
        }
    }
}

fn responses_input_to_chat_messages(input: &Value) -> Result<Vec<Message>, String> {
    match input {
        Value::String(text) => Ok(vec![user_message(text.clone())]),
        Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                out.push(response_item_to_message(item)?);
            }
            if out.is_empty() {
                return Err("responses input must contain at least one item".to_string());
            }
            Ok(out)
        }
        Value::Object(obj) => {
            let Some(Value::Array(messages)) = obj.get("messages") else {
                return Err("responses object input must include messages array".to_string());
            };
            let mut out = Vec::new();
            for item in messages {
                out.push(response_item_to_message(item)?);
            }
            if out.is_empty() {
                return Err("responses input.messages must contain at least one item".to_string());
            }
            Ok(out)
        }
        _ => Err("responses input must be a string, array, or messages object".to_string()),
    }
}

fn response_item_to_message(item: &Value) -> Result<Message, String> {
    let Some(obj) = item.as_object() else {
        return Err("responses input array items must be objects".to_string());
    };
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
    let content = obj
        .get("content")
        .map(response_content_to_text)
        .transpose()?
        .unwrap_or_default();
    let role = match role {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => return Err(format!("unsupported responses role: {other}")),
    };
    Ok(Message {
        role,
        content,
        tool_calls: Vec::new(),
        tool_call_id: None,
    })
}

fn response_content_to_text(content: &Value) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if let Some(s) = part.as_str() {
                    text.push_str(s);
                    continue;
                }
                let Some(obj) = part.as_object() else {
                    continue;
                };
                let part_type = obj.get("type").and_then(Value::as_str);
                if matches!(
                    part_type,
                    Some("input_text" | "output_text" | "text") | None
                ) {
                    if let Some(s) = obj
                        .get("text")
                        .or_else(|| obj.get("content"))
                        .and_then(Value::as_str)
                    {
                        text.push_str(s);
                    }
                }
            }
            Ok(text)
        }
        Value::Null => Ok(String::new()),
        other => Err(format!("unsupported responses content shape: {}", other)),
    }
}

fn user_message(content: String) -> Message {
    Message {
        role: Role::User,
        content,
        tool_calls: Vec::new(),
        tool_call_id: None,
    }
}

fn prompt_message_to_chat_message(message: &Message) -> ChatMessage {
    let role = match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    ChatMessage {
        role: role.to_string(),
        content: Some(Value::String(message.content.clone())),
        ..Default::default()
    }
}

async fn load_responses_context(
    state: &SharedState,
    owner: &ResponsesOwner,
    response_id: &str,
) -> Result<Option<Vec<Message>>, String> {
    match owner {
        ResponsesOwner::AnonymousLocal => Ok(state
            .responses_contexts
            .lock()
            .await
            .get(response_id)
            .map(|ctx| ctx.messages.clone())),
        ResponsesOwner::User(user_id) => {
            let store = state.access.store()?;
            let user_id = user_id.clone();
            let response_id = response_id.to_string();
            let record = tokio::task::spawn_blocking(move || {
                let now = response_now_secs();
                store.prune_responses_expired(now)?;
                load_owned_record_with_chain(&store, &user_id, &response_id, now)
            })
            .await
            .map_err(|error| format!("Responses storage task failed: {error}"))?
            .map_err(|error| error.to_string())?;
            record
                .map(|record| {
                    serde_json::from_slice::<Vec<Message>>(&record.payload)
                        .map_err(|error| format!("persisted Responses context is invalid: {error}"))
                })
                .transpose()
        }
    }
}

async fn store_responses_context(
    state: &SharedState,
    owner: &ResponsesOwner,
    response_id: String,
    parent_response_id: Option<String>,
    messages: Vec<Message>,
) -> Result<(), Value> {
    match owner {
        ResponsesOwner::AnonymousLocal => {
            let max = responses_state_max();
            if max == 0 {
                return Ok(());
            }
            {
                let mut contexts = state.responses_contexts.lock().await;
                contexts.insert(response_id.clone(), StoredResponsesContext { messages });
            }
            let mut order = state.responses_order.lock().await;
            order.retain(|id| id != &response_id);
            order.push_back(response_id);
            while order.len() > max {
                if let Some(evicted) = order.pop_front() {
                    state.responses_contexts.lock().await.remove(&evicted);
                }
            }
            Ok(())
        }
        ResponsesOwner::User(user_id) => {
            let payload = serde_json::to_vec(&messages).map_err(|error| {
                server_context_error(format!("failed to serialize Responses context: {error}"))
            })?;
            let store = state.access.store().map_err(server_context_error)?;
            let user_id = user_id.clone();
            let now = response_now_secs();
            tokio::task::spawn_blocking(move || {
                if let Some(parent) = &parent_response_id {
                    ensure_chain_can_extend(&store, &user_id, parent, now)?;
                }
                store.put_response_bounded(
                    &ResponseContextRecord {
                        user_id,
                        response_id,
                        parent_response_id,
                        created_at: now,
                        updated_at: now,
                        expires_at: now.saturating_add(AUTHENTICATED_RESPONSE_TTL_SECS),
                        payload,
                    },
                    AUTHENTICATED_RESPONSE_MAX,
                )
            })
            .await
            .map_err(|error| {
                server_context_error(format!("Responses storage task failed: {error}"))
            })?
            .map_err(context_store_error)
        }
    }
}

fn load_owned_record_with_chain(
    store: &hipfire_auth::AccessStore,
    user_id: &str,
    response_id: &str,
    now: u64,
) -> Result<Option<ResponseContextRecord>, hipfire_auth::AuthError> {
    let Some(root) = store.get_response(user_id, response_id)? else {
        return Ok(None);
    };
    if root.expires_at <= now {
        return Ok(None);
    }
    let mut parent = root.parent_response_id.clone();
    let mut depth = 1usize;
    while let Some(parent_id) = parent {
        depth += 1;
        if depth > RESPONSE_CHAIN_MAX_DEPTH {
            return Ok(None);
        }
        let Some(record) = store.get_response(user_id, &parent_id)? else {
            return Ok(None);
        };
        if record.expires_at <= now {
            return Ok(None);
        }
        parent = record.parent_response_id;
    }
    Ok(Some(root))
}

fn ensure_chain_can_extend(
    store: &hipfire_auth::AccessStore,
    user_id: &str,
    parent_id: &str,
    now: u64,
) -> Result<(), hipfire_auth::AuthError> {
    let Some(parent) = load_owned_record_with_chain(store, user_id, parent_id, now)? else {
        return Err(hipfire_auth::AuthError::Invalid(
            "previous_response_id not found".into(),
        ));
    };
    let mut depth = 1usize;
    let mut cursor = parent.parent_response_id;
    while let Some(parent_id) = cursor {
        depth += 1;
        if depth >= RESPONSE_CHAIN_MAX_DEPTH {
            return Err(hipfire_auth::AuthError::Invalid(format!(
                "Responses chain depth exceeds {RESPONSE_CHAIN_MAX_DEPTH}"
            )));
        }
        let Some(record) = store.get_response(user_id, &parent_id)? else {
            return Err(hipfire_auth::AuthError::Invalid(
                "previous_response_id not found".into(),
            ));
        };
        cursor = record.parent_response_id;
    }
    Ok(())
}

fn context_store_error(error: hipfire_auth::AuthError) -> Value {
    match error {
        hipfire_auth::AuthError::Invalid(message) => json!({
            "error": {"message": message, "type": "invalid_request_error"}
        }),
        other => server_context_error(other.to_string()),
    }
}

fn server_context_error(message: impl Into<String>) -> Value {
    json!({"error": {"message": message.into(), "type": "server_error"}})
}

fn response_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn responses_state_max() -> usize {
    std::env::var("HIPFIRE_RESPONSES_STATE_MAX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use hipfire_auth::{AuthKind, NewUser, RatePolicyOverride, Scope};
    use hipfire_config::{HipfireConfig, LoadedConfig};

    fn test_state(directory: &std::path::Path) -> SharedState {
        crate::AppState::new_loaded_with_directories(
            LoadedConfig::from_config(HipfireConfig::default()),
            directory.join("training"),
            directory.join("access"),
        )
    }

    fn create_user(state: &SharedState, name: &str) -> String {
        state
            .access
            .store()
            .unwrap()
            .create_user(
                NewUser {
                    name: name.into(),
                    rate_policy: RatePolicyOverride::default(),
                },
                1,
            )
            .unwrap()
            .id
    }

    #[test]
    fn responses_string_input_becomes_user_message() {
        let messages = responses_input_to_chat_messages(&json!("hello")).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].content, "hello");
    }

    #[test]
    fn invalid_responses_errors_return_bad_request_status() {
        let error = json!({"error": {"message": "bad", "type": "invalid_request_error"}});
        assert_eq!(error_status(&error), StatusCode::BAD_REQUEST);

        let error = json!({"error": {"message": "bad", "type": "server_error"}});
        assert_eq!(error_status(&error), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn responses_array_input_extracts_text_parts() {
        let messages = responses_input_to_chat_messages(&json!([
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                    {"type": "input_text", "text": " world"}
                ]
            }
        ]))
        .unwrap();
        assert_eq!(messages[0].content, "hello world");
    }

    #[test]
    fn responses_object_input_reads_messages() {
        let messages = responses_input_to_chat_messages(&json!({
            "messages": [{
                "role": "user",
                "content": "hello"
            }]
        }))
        .unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "hello");
    }

    #[test]
    fn response_json_matches_openai_responses_shape() {
        let done = hipfire_generate::DoneEvent {
            id: "req".to_string(),
            tokens: 3,
            tok_s: None,
            prefill_tokens: Some(7),
            prefill_ms: None,
            prefill_tok_s: None,
            decode_tok_s: None,
            ttft_ms: None,
            finish_reason: Some("stop".to_string()),
            response_id: None,
            extra: Default::default(),
        };
        let body = response_json("resp_1", "qwen", "hi", &done);
        assert_eq!(body["id"], "resp_1");
        assert_eq!(body["object"], "response");
        assert_eq!(body["output_text"], "hi");
        assert_eq!(body["usage"]["input_tokens"], 7);
        assert_eq!(body["usage"]["output_tokens"], 3);
    }

    #[test]
    fn response_stream_events_match_responses_shape() {
        let created = response_created_json("resp_1", "qwen");
        assert_eq!(created["type"], "response.created");
        assert_eq!(created["response"]["status"], "in_progress");

        let delta = response_output_text_delta_json("resp_1", "msg_1", "hi");
        assert_eq!(delta["type"], "response.output_text.delta");
        assert_eq!(delta["response_id"], "resp_1");
        assert_eq!(delta["item_id"], "msg_1");
        assert_eq!(delta["delta"], "hi");

        let done = response_output_item_done_json("resp_1", "msg_1", "hello");
        assert_eq!(done["type"], "response.output_item.done");
        assert_eq!(done["item"]["status"], "completed");
        assert_eq!(done["item"]["content"][0]["text"], "hello");
    }

    #[tokio::test]
    async fn authenticated_contexts_are_user_scoped_and_survive_restart() {
        let directory = tempfile::tempdir().unwrap();
        let state = test_state(directory.path());
        let alice = create_user(&state, "alice");
        let bob = create_user(&state, "bob");
        let messages = vec![Message {
            role: Role::User,
            content: "private".into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }];
        store_responses_context(
            &state,
            &ResponsesOwner::User(alice.clone()),
            "resp_private".into(),
            None,
            messages.clone(),
        )
        .await
        .unwrap();
        let loaded =
            load_responses_context(&state, &ResponsesOwner::User(alice.clone()), "resp_private")
                .await
                .unwrap()
                .unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "private");
        assert!(
            load_responses_context(&state, &ResponsesOwner::User(bob), "resp_private")
                .await
                .unwrap()
                .is_none()
        );

        drop(state);
        let restarted = test_state(directory.path());
        let loaded =
            load_responses_context(&restarted, &ResponsesOwner::User(alice), "resp_private")
                .await
                .unwrap()
                .unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "private");
    }

    #[tokio::test]
    async fn anonymous_contexts_remain_memory_only_and_missing_parents_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let state = test_state(directory.path());
        store_responses_context(
            &state,
            &ResponsesOwner::AnonymousLocal,
            "resp_local".into(),
            None,
            vec![Message {
                role: Role::User,
                content: "local".into(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
        )
        .await
        .unwrap();
        assert!(state
            .access
            .store()
            .unwrap()
            .list_user_responses("anonymous-local")
            .unwrap()
            .is_empty());
        drop(state);
        let restarted = test_state(directory.path());
        assert!(
            load_responses_context(&restarted, &ResponsesOwner::AnonymousLocal, "resp_local")
                .await
                .unwrap()
                .is_none()
        );

        let user = create_user(&restarted, "owner");
        let now = response_now_secs();
        restarted
            .access
            .store()
            .unwrap()
            .put_response(&ResponseContextRecord {
                user_id: user.clone(),
                response_id: "resp_orphan".into(),
                parent_response_id: Some("resp_missing".into()),
                created_at: now,
                updated_at: now,
                expires_at: now + 60,
                payload: serde_json::to_vec(&Vec::<Message>::new()).unwrap(),
            })
            .unwrap();
        assert!(
            load_responses_context(&restarted, &ResponsesOwner::User(user), "resp_orphan")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn token_rotation_keeps_the_same_response_owner() {
        let principal = |token: &str| RequestPrincipal {
            user_id: Some("user-1".into()),
            token_id: Some(token.into()),
            scopes: BTreeSet::from([Scope::Text]),
            auth_kind: AuthKind::ApiToken,
        };
        assert!(matches!(
            ResponsesOwner::from_principal(Some(Extension(principal("old")))),
            ResponsesOwner::User(ref id) if id == "user-1"
        ));
        assert!(matches!(
            ResponsesOwner::from_principal(Some(Extension(principal("new")))),
            ResponsesOwner::User(ref id) if id == "user-1"
        ));
    }

    #[tokio::test]
    async fn cross_user_and_unknown_parent_return_the_same_error_shape() {
        fn request(previous_response_id: &str) -> ResponsesRequest {
            serde_json::from_value(json!({
                "input": "next",
                "previous_response_id": previous_response_id
            }))
            .unwrap()
        }

        let with_record = tempfile::tempdir().unwrap();
        let state = test_state(with_record.path());
        let alice = create_user(&state, "alice");
        store_responses_context(
            &state,
            &ResponsesOwner::User(alice),
            "resp_same".into(),
            None,
            Vec::new(),
        )
        .await
        .unwrap();
        let cross_user = prepare_response_messages(
            &state,
            &request("resp_same"),
            &ResponsesOwner::User("bob".into()),
        )
        .await
        .unwrap_err();

        let without_record = tempfile::tempdir().unwrap();
        let empty_state = test_state(without_record.path());
        let unknown = prepare_response_messages(
            &empty_state,
            &request("resp_same"),
            &ResponsesOwner::User("bob".into()),
        )
        .await
        .unwrap_err();
        assert_eq!(cross_user, unknown);
        assert_eq!(cross_user["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn authenticated_chain_depth_is_bounded() {
        let directory = tempfile::tempdir().unwrap();
        let store = hipfire_auth::AccessStore::open_in(directory.path()).unwrap();
        let user = "owner";
        let now = response_now_secs();
        let mut parent = None;
        for index in 0..RESPONSE_CHAIN_MAX_DEPTH {
            let id = format!("resp_{index}");
            store
                .put_response(&ResponseContextRecord {
                    user_id: user.into(),
                    response_id: id.clone(),
                    parent_response_id: parent,
                    created_at: now,
                    updated_at: now,
                    expires_at: now + 60,
                    payload: Vec::new(),
                })
                .unwrap();
            parent = Some(id);
        }
        let error =
            ensure_chain_can_extend(&store, user, parent.as_deref().unwrap(), now).unwrap_err();
        assert!(matches!(error, hipfire_auth::AuthError::Invalid(_)));
    }
}
