use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub(crate) struct ServerChatResult {
    pub(crate) text: String,
    pub(crate) timings: Value,
}

pub(crate) fn eval_server_url() -> Option<String> {
    std::env::var("HIPFIRE_EVAL_SERVER_URL")
        .ok()
        .filter(|url| !url.trim().is_empty())
        .map(|url| url.trim_end_matches('/').to_string())
}

pub(crate) fn server_chat_completion(
    server_url: &str,
    model: &str,
    prompt: &str,
    system: Option<&str>,
    tools: Option<Value>,
    max_tokens: usize,
) -> Result<ServerChatResult, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("build HTTP client: {e}"))?;
    let mut messages = Vec::new();
    if let Some(system) = system.filter(|s| !s.is_empty()) {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    let mut body = json!({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    });
    if let Some(tools) = tools {
        body["tools"] = tools;
    }

    let response = client
        .post(format!("{server_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .map_err(|e| format!("POST /v1/chat/completions: {e}"))?;
    let status = response.status();
    let value: Value = response
        .json()
        .map_err(|e| format!("decode chat response ({status}): {e}"))?;
    if !status.is_success() {
        let message = value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("server error");
        return Err(format!("server returned {status}: {message}"));
    }
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| "server response missing choices[0]".to_string())?;
    let text = choice
        .pointer("/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Ok(ServerChatResult {
        text,
        timings: value.get("timings").cloned().unwrap_or(Value::Null),
    })
}

pub(crate) fn server_reset(server_url: &str) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| format!("build HTTP client: {e}"))?;
    let mut request = client.post(format!("{server_url}/admin/runtime/reset"));
    if let Some(secret) = hipfire_config::read_admin_secret() {
        request = request.bearer_auth(secret);
    }
    let response = request
        .send()
        .map_err(|e| format!("POST /admin/runtime/reset: {e}"))?;
    let status = response.status();
    let value: Value = response
        .json()
        .map_err(|e| format!("decode reset response ({status}): {e}"))?;
    if !status.is_success() {
        let message = value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("server error");
        return Err(format!("server returned {status}: {message}"));
    }
    Ok(())
}

pub(crate) fn timing_f64(timings: &Value, key: &str) -> Option<f64> {
    timings.get(key).and_then(Value::as_f64)
}

pub(crate) fn timing_u64(timings: &Value, key: &str) -> Option<u64> {
    timings.get(key).and_then(Value::as_u64)
}
