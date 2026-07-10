//! Shared browser networking for the hipfire Leptos UIs (chat + admin).
//!
//! Factors out the plumbing both apps would otherwise duplicate: JSON
//! `GET`/`POST` helpers and a cancellable Server-Sent-Events `POST` reader
//! built on `fetch` + `ReadableStreamDefaultReader` + `TextDecoder`. App-
//! specific event interpretation stays in each app via the `on_data` callback.
//!
//! wasm-only by construction (web-sys); the consuming crates are standalone
//! trunk builds that path-depend on this one.

use js_sys::{Reflect, Uint8Array};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    AbortSignal, ReadableStreamDefaultReader, Request, RequestInit, RequestMode, Response,
    TextDecodeOptions,
};

/// Best-effort human string for a `JsValue` error (string, else JSON, else a
/// generic fallback).
pub fn js_error(value: JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            js_sys::JSON::stringify(&value)
                .ok()
                .and_then(|s| s.as_string())
        })
        .unwrap_or_else(|| "JavaScript error".to_string())
}

/// Structured browser-facing HTTP failure. Admin surfaces use the status to
/// distinguish an expired login from validation and server failures while the
/// message remains suitable for inline display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: Option<u16>,
    pub message: String,
}

impl ApiError {
    pub fn is_unauthorized(&self) -> bool {
        self.status == Some(401)
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

fn error_message(status: u16, text: &str) -> String {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| {
                    error
                        .as_str()
                        .map(str::to_owned)
                        .or_else(|| error.get("message")?.as_str().map(str::to_owned))
                })
                .or_else(|| value.get("message")?.as_str().map(str::to_owned))
        })
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| format!("request returned HTTP {status}"))
}

async fn decode_json<T: DeserializeOwned>(
    response: gloo_net::http::Response,
) -> Result<T, ApiError> {
    let status = response.status();
    let text = response.text().await.map_err(|error| ApiError {
        status: Some(status),
        message: error.to_string(),
    })?;
    if !(200..300).contains(&status) {
        return Err(ApiError {
            status: Some(status),
            message: error_message(status, &text),
        });
    }
    serde_json::from_str(&text).map_err(|error| ApiError {
        status: Some(status),
        message: format!("invalid JSON response: {error}"),
    })
}

/// Typed JSON helpers for authenticated same-origin applications.
pub async fn get_json_typed<T: DeserializeOwned>(url: &str) -> Result<T, ApiError> {
    let response = gloo_net::http::Request::get(url)
        .send()
        .await
        .map_err(|error| ApiError {
            status: None,
            message: error.to_string(),
        })?;
    decode_json(response).await
}

pub async fn post_json_typed<B: Serialize, T: DeserializeOwned>(
    url: &str,
    body: &B,
) -> Result<T, ApiError> {
    let request = gloo_net::http::Request::post(url)
        .json(body)
        .map_err(|error| ApiError {
            status: None,
            message: error.to_string(),
        })?;
    let response = request.send().await.map_err(|error| ApiError {
        status: None,
        message: error.to_string(),
    })?;
    decode_json(response).await
}

pub async fn patch_json<B: Serialize, T: DeserializeOwned>(
    url: &str,
    body: &B,
) -> Result<T, ApiError> {
    let request = gloo_net::http::Request::patch(url)
        .json(body)
        .map_err(|error| ApiError {
            status: None,
            message: error.to_string(),
        })?;
    let response = request.send().await.map_err(|error| ApiError {
        status: None,
        message: error.to_string(),
    })?;
    decode_json(response).await
}

pub async fn delete_json<T: DeserializeOwned>(url: &str) -> Result<T, ApiError> {
    let response = gloo_net::http::Request::delete(url)
        .send()
        .await
        .map_err(|error| ApiError {
            status: None,
            message: error.to_string(),
        })?;
    decode_json(response).await
}

/// `GET url` and deserialize the JSON body into `T`.
pub async fn get_json<T: DeserializeOwned>(url: &str) -> Result<T, String> {
    get_json_typed(url).await.map_err(|error| error.to_string())
}

/// `POST url` with a JSON body, returning the parsed JSON response (regardless
/// of status, so callers can inspect an `error` payload).
pub async fn post_json(url: &str, body: &Value) -> Result<PostResult, String> {
    let resp = gloo_net::http::Request::post(url)
        .json(body)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status();
    let payload = resp.json::<Value>().await.map_err(|e| e.to_string())?;
    Ok(PostResult { status, payload })
}

/// Result of [`post_json`]: HTTP status plus the parsed body.
pub struct PostResult {
    pub status: u16,
    pub payload: Value,
}

/// `POST url` with a JSON body and stream the SSE response, invoking `on_data`
/// once per `data:` payload (trimmed; `[DONE]` and empty payloads skipped).
///
/// Pass an [`AbortSignal`] (from a `web_sys::AbortController`) to make the
/// stream cancellable — aborting rejects the in-flight read, surfaced here as
/// an `Err`. `on_data` returning `Err` stops the stream early.
pub async fn sse_post(
    url: &str,
    body: &Value,
    signal: Option<&AbortSignal>,
    mut on_data: impl FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::SameOrigin);
    if let Some(s) = signal {
        opts.set_signal(Some(s));
    }
    let serialized = serde_json::to_string(body).map_err(|e| e.to_string())?;
    opts.set_body(&JsValue::from_str(&serialized));
    let request = Request::new_with_str_and_init(url, &opts).map_err(js_error)?;
    request
        .headers()
        .set("Content-Type", "application/json")
        .map_err(js_error)?;

    let window = web_sys::window().ok_or_else(|| "window unavailable".to_string())?;
    let resp = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(js_error)?
        .dyn_into::<Response>()
        .map_err(js_error)?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let stream = resp
        .body()
        .ok_or_else(|| "response body stream unavailable".to_string())?;
    let reader = stream
        .get_reader()
        .dyn_into::<ReadableStreamDefaultReader>()
        .map_err(|e| js_error(e.into()))?;
    let decoder = web_sys::TextDecoder::new().map_err(js_error)?;
    let decode_opts = TextDecodeOptions::new();
    decode_opts.set_stream(true);
    let mut buffer = String::new();

    loop {
        let chunk = JsFuture::from(reader.read()).await.map_err(js_error)?;
        let done = Reflect::get(&chunk, &JsValue::from_str("done"))
            .map_err(js_error)?
            .as_bool()
            .unwrap_or(false);
        if done {
            break;
        }
        let value = Reflect::get(&chunk, &JsValue::from_str("value")).map_err(js_error)?;
        let bytes = Uint8Array::new(&value);
        let mut bytes_vec = vec![0; bytes.length() as usize];
        bytes.copy_to(&mut bytes_vec);
        let text = decoder
            .decode_with_u8_array_and_options(&bytes_vec, &decode_opts)
            .map_err(js_error)?;
        buffer.push_str(&text);
        drain_events(&mut buffer, &mut on_data)?;
    }
    if !buffer.trim().is_empty() {
        let event = std::mem::take(&mut buffer);
        dispatch_event(&event, &mut on_data)?;
    }
    Ok(())
}

/// Pull complete `\n\n`-delimited SSE events out of `buffer`.
fn drain_events(
    buffer: &mut String,
    on_data: &mut impl FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    while let Some(split) = buffer.find("\n\n") {
        let event = buffer[..split].to_string();
        buffer.drain(..split + 2);
        dispatch_event(&event, on_data)?;
    }
    Ok(())
}

/// Hand each `data:` payload in one SSE event to `on_data`.
fn dispatch_event(
    event: &str,
    on_data: &mut impl FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    for line in event.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim_start();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        on_data(data)?;
    }
    Ok(())
}
