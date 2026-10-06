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
    // Streamed: a streamed request reports full timings (prefill and decode
    // tok/s, TTFT) and runs on its own; a non-streamed one batched beside other
    // clients' requests reports token counts only.
    let mut body = json!({
        "model": server_model_id(model),
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
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
    if !status.is_success() {
        let value: Value = response.json().unwrap_or(Value::Null);
        let message = value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("server error");
        return Err(format!("server returned {status}: {message}"));
    }
    read_chat_stream(std::io::BufReader::new(response))
}

/// The id a server knows a model by: a path to a `.hfq` file is its stem
/// (`.../Qwen3.6-35B-A3B--oq4.25++.hfq` -> `Qwen3.6-35B-A3B--oq4.25++`); the
/// server resolves names, not paths.
fn server_model_id(model: &str) -> &str {
    let name = std::path::Path::new(model)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(model);
    name.strip_suffix(".hfq").unwrap_or(name)
}

/// A streamed chat completion's text (its content deltas, joined) and the
/// `timings` its final chunk carries.
fn read_chat_stream(reader: impl std::io::BufRead) -> Result<ServerChatResult, String> {
    let mut text = String::new();
    let mut timings = Value::Null;
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read chat stream: {e}"))?;
        let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else {
            continue;
        };
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            continue; // `[DONE]`
        };
        // A stream that fails after its 200 says so in-band.
        if let Some(message) = event.pointer("/error/message").and_then(Value::as_str) {
            return Err(format!("server error: {message}"));
        }
        if let Some(delta) = event
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            text.push_str(delta);
        }
        if let Some(t) = event.get("timings").filter(|t| !t.is_null()) {
            timings = t.clone();
        }
    }
    Ok(ServerChatResult { text, timings })
}

pub(crate) fn timing_f64(timings: &Value, key: &str) -> Option<f64> {
    timings.get(key).and_then(Value::as_f64)
}

pub(crate) fn timing_u64(timings: &Value, key: &str) -> Option<u64> {
    timings.get(key).and_then(Value::as_u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chat_stream_folds_to_its_text_and_final_timings() {
        let stream =
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n\
                      data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n\
                      data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\
                      \"timings\":{\"tokens\":2,\"decode_tok_s\":41.5,\"ttft_ms\":12.0}}\n\n\
                      data: [DONE]\n";
        let result = read_chat_stream(stream.as_bytes()).unwrap();
        assert_eq!(result.text, "Hello");
        assert_eq!(timing_u64(&result.timings, "tokens"), Some(2));
        assert_eq!(timing_f64(&result.timings, "decode_tok_s"), Some(41.5));
    }

    #[test]
    fn a_stream_that_fails_after_its_200_is_an_error() {
        let stream = "data: {\"error\":{\"message\":\"model not found: x\"}}\n\ndata: [DONE]\n";
        let err = read_chat_stream(stream.as_bytes()).unwrap_err();
        assert!(err.contains("model not found: x"), "{err}");
    }

    #[test]
    fn a_model_path_is_sent_as_the_name_the_server_knows() {
        assert_eq!(
            server_model_id("/m/Qwen3.6-35B-A3B--oq4.25++.hfq"),
            "Qwen3.6-35B-A3B--oq4.25++"
        );
        assert_eq!(server_model_id("qwen3.5:9b"), "qwen3.5:9b");
    }
}
