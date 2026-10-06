use std::io::Write;
use std::path::{Path, PathBuf};

use base64::Engine;
use clap::Args;
use hipfire_config::{HipfireConfig, LoadedConfig};
use hipfire_daemon_adapter::{find_daemon_bin_or_error, DaemonEngine};
use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};
use hipfire_model::ModelLoadParams;
use uuid::Uuid;

use crate::model::find_model;

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  hipfire chat --model Qwen3.5-30B-A3B \"Explain ROCm in one paragraph\"\n  hipfire chat \"hello\" --max-tokens 64\n  hipfire chat --attach image.png \"describe this image\"\n"
)]
pub struct ChatArgs {
    /// Model name, shorthand, alias, or path. Falls back to the
    /// `default_model` config value when omitted.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Prompt text
    pub prompt: String,
    /// Max tokens to generate
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature
    #[arg(long)]
    pub temperature: Option<f64>,
    /// Attach a file to the prompt (repeatable). The type is detected from the
    /// extension. Only images are wired today (PNG/JPEG/WebP/GIF/BMP); text,
    /// video, and audio are recognized but not yet supported and will error.
    #[arg(long, value_name = "FILE")]
    pub attach: Vec<PathBuf>,
}

/// Recognized attachment categories. Generic on purpose so new modalities slot
/// in without reshaping the CLI; only [`AttachKind::Image`] is wired so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachKind {
    Image,
    Text,
    Video,
    Audio,
}

/// Classify an attachment by file extension. Errors on unknown/missing types.
fn classify_attachment(path: &Path) -> anyhow::Result<AttachKind> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    Ok(match ext.as_str() {
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" => AttachKind::Image,
        "txt" | "md" | "markdown" | "json" | "csv" | "log" | "rs" | "py" => AttachKind::Text,
        "mp4" | "mov" | "webm" | "mkv" | "avi" => AttachKind::Video,
        "wav" | "mp3" | "flac" | "ogg" | "m4a" => AttachKind::Audio,
        "" => anyhow::bail!(
            "--attach {}: cannot determine file type (no extension)",
            path.display()
        ),
        other => anyhow::bail!(
            "--attach {}: unsupported file type '.{other}'",
            path.display()
        ),
    })
}

/// Validate CLI attachments and resolve the single supported image (if any).
/// Runs before the model loads so bad/unwired/unknown attachments fail fast:
/// every non-image kind errors "recognized but not yet wired", unknown types
/// error in [`classify_attachment`], and >1 image errors.
fn resolve_image_attachment(attach: &[PathBuf]) -> anyhow::Result<Option<PathBuf>> {
    let mut image: Option<PathBuf> = None;
    for path in attach {
        match classify_attachment(path)? {
            AttachKind::Image => {
                if let Some(prev) = &image {
                    anyhow::bail!(
                        "--attach: multiple image attachments are not yet supported \
                         (got {} and {})",
                        prev.display(),
                        path.display()
                    );
                }
                if !path.exists() {
                    anyhow::bail!("--attach {}: file not found", path.display());
                }
                image = Some(path.clone());
            }
            kind => anyhow::bail!(
                "--attach {}: {kind:?} attachments are recognized but not yet wired \
                 (only images are supported today)",
                path.display()
            ),
        }
    }
    Ok(image)
}

/// Read an image attachment and set it on the request as base64.
fn attach_image(req: GenerateTextRequest, image: &Path) -> anyhow::Result<GenerateTextRequest> {
    let bytes = std::fs::read(image)
        .map_err(|e| anyhow::anyhow!("--attach {}: read failed: {e}", image.display()))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(req.with_image_base64(Some(b64)))
}

fn load_params_from_config(config: &HipfireConfig) -> ModelLoadParams {
    ModelLoadParams::from_hipfire_config(config)
}

fn generate_request_from_prompt(
    id: String,
    prompt: &str,
    sampling: GenerationSamplingPolicy,
    worker_key_id: Option<String>,
) -> GenerateTextRequest {
    GenerateTextRequest::from_prompt(id, prompt, sampling).with_worker_key_id(worker_key_id)
}

fn thinking_controls_from_config(
    config: &HipfireConfig,
) -> (Option<String>, Option<String>, Option<u32>) {
    let thinking_disabled = config.thinking.eq_ignore_ascii_case("off");
    let thinking_mode = if thinking_disabled {
        "chat"
    } else {
        "thinking"
    };
    let assistant_prefix = if thinking_disabled {
        "closed_think"
    } else {
        "open_think"
    };
    (
        Some(thinking_mode.to_string()),
        Some(assistant_prefix.to_string()),
        thinking_disabled.then_some(1),
    )
}

/// Drive a model-load future to completion while rendering an indicatif bar
/// fed by the daemon adapter's `model_load_progress()` global. Starts as a
/// spinner (pre-first-frame), then switches to a `pos/len` bar once the daemon
/// reports a phase total; the bar restarts per phase (see `model_load_progress`).
async fn load_with_progress<T>(
    fut: impl std::future::Future<Output = anyhow::Result<T>>,
    model_path: &Path,
) -> anyhow::Result<T> {
    use indicatif::{ProgressBar, ProgressStyle};

    let bar = ProgressBar::new_spinner();
    bar.set_message(format!("Loading {}…", model_path.display()));
    tokio::pin!(fut);
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(100));
    loop {
        tokio::select! {
            r = &mut fut => {
                bar.finish_and_clear();
                return r;
            }
            _ = tick.tick() => {
                let (cur, total, phase) = hipfire_daemon_adapter::model_load_progress();
                if total > 0 {
                    if bar.length() != Some(total as u64) {
                        bar.set_length(total as u64);
                        bar.set_style(
                            ProgressStyle::with_template("{msg} [{bar:30}] {pos}/{len}")
                                .unwrap()
                                .progress_chars("=>-"),
                        );
                    }
                    bar.set_position(cur as u64);
                    bar.set_message(format!("Loading {phase}"));
                } else {
                    bar.tick();
                }
            }
        }
    }
}

/// The id a server knows a resolved model by: its file's stem. The server
/// resolves names, not paths, so `--model ./x.hfq` must not reach it as given.
pub(crate) fn server_model_id(model_path: &Path) -> String {
    let name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    name.strip_suffix(".hfq").unwrap_or(name).to_string()
}

/// The chat completion request `hipfire chat` sends a running server.
fn server_chat_request(model: &str, args: &ChatArgs, image_b64: Option<&str>) -> serde_json::Value {
    let content = match image_b64 {
        Some(b64) => serde_json::json!([
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}},
        ]),
        None => serde_json::json!(args.prompt),
    };
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(n) = args.max_tokens {
        body["max_tokens"] = serde_json::json!(n);
    }
    if let Some(t) = args.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    body
}

/// What a streamed chat completion ended with.
pub(crate) struct StreamedCompletion {
    /// The final chunk's `timings` (prefill/decode tokens, ms, tok/s, TTFT).
    pub timings: serde_json::Value,
    pub usage: serde_json::Value,
}

/// POST `body` (a streaming request) to `{url}/v1/chat/completions`, handing
/// each content delta to `on_text` as it arrives.
pub(crate) async fn stream_chat_completion(
    client: &reqwest::Client,
    url: &str,
    body: &serde_json::Value,
    mut on_text: impl FnMut(&str),
) -> anyhow::Result<StreamedCompletion> {
    let mut response = client
        .post(format!("{url}/v1/chat/completions"))
        .json(body)
        .send()
        .await?;
    if !response.status().is_success() {
        let status = response.status();
        anyhow::bail!(
            "{url}: {status}: {}",
            response.text().await.unwrap_or_default()
        );
    }
    let mut done = StreamedCompletion {
        timings: serde_json::Value::Null,
        usage: serde_json::Value::Null,
    };
    let mut buffered = String::new();
    while let Some(chunk) = response.chunk().await? {
        buffered.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(end) = buffered.find('\n') {
            let line: String = buffered.drain(..=end).collect();
            let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else {
                continue;
            };
            let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
                continue; // `[DONE]`
            };
            // A stream that fails after its 200 says so in-band.
            if let Some(message) = event["error"]["message"].as_str() {
                anyhow::bail!("{url}: {message}");
            }
            if let Some(text) = event["choices"][0]["delta"]["content"].as_str() {
                on_text(text);
            }
            for (key, slot) in [("timings", &mut done.timings), ("usage", &mut done.usage)] {
                if !event[key].is_null() {
                    *slot = event[key].clone();
                }
            }
        }
    }
    Ok(done)
}

/// Stream one chat turn from a running server's `/v1/chat/completions`.
async fn chat_via_server(
    url: &str,
    model: &str,
    args: &ChatArgs,
    image: Option<&Path>,
) -> anyhow::Result<()> {
    let image_b64 =
        match image {
            Some(path) => Some(base64::engine::general_purpose::STANDARD.encode(
                std::fs::read(path).map_err(|e| {
                    anyhow::anyhow!("--attach {}: read failed: {e}", path.display())
                })?,
            )),
            None => None,
        };
    let done = stream_chat_completion(
        &reqwest::Client::new(),
        url,
        &server_chat_request(model, args, image_b64.as_deref()),
        |text| {
            print!("{text}");
            let _ = std::io::stdout().flush();
        },
    )
    .await?;
    println!();
    let tokens = done.usage["completion_tokens"].as_u64().unwrap_or(0);
    let tok_s = done.timings["decode_tok_s"].as_f64().unwrap_or(0.0);
    eprintln!("\n[{tokens} tokens, {tok_s:.2} tok/s, via {url}]");
    Ok(())
}

pub async fn run(args: ChatArgs, loaded: LoadedConfig) -> anyhow::Result<()> {
    // Resolve the model first (using the global config for `default_model`),
    // then re-resolve the config with that model's tag so per-model overrides
    // (e.g. a model-specific `max_seq`/`kv_cache` in `[model_overrides]`) apply.
    let model = args
        .model
        .as_deref()
        .or(loaded.config.default_model.as_deref())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model specified and no `default_model` configured; \
                 pass --model <name> or set `default_model` in {}",
                hipfire_config::config_path().display()
            )
        })?
        .to_string();
    let model_path = find_model(&model, &loaded.config)
        .ok_or_else(|| anyhow::anyhow!("model not found: {model}"))?;
    let config: HipfireConfig = loaded.resolve_for_model(&model).config;

    // Validate attachments up front so an unsupported file type fails before the
    // (potentially large) model load.
    let image_attachment = resolve_image_attachment(&args.attach)?;

    // With `hipfire serve` up, chat through it like any client: its worker is
    // shared, and a request on the worker's socket that names no worker lands
    // on whatever the server has active -- a reset or load there wipes other
    // clients' sessions (the worker now refuses them). Through the HTTP API the
    // server schedules this request beside everything else, on the model it
    // already has resident.
    if let Some(url) = crate::commands::bench::running_server_url(&loaded).await {
        let name = server_model_id(&model_path);
        return chat_via_server(&url, &name, &args, image_attachment.as_deref()).await;
    }

    // No server: a private worker of our own. Never the shared socket.
    let bin = find_daemon_bin_or_error()?;
    let mut engine = DaemonEngine::spawn(&bin).await?;

    let model_path_str = model_path.to_string_lossy().into_owned();
    let load = engine.load(&model_path_str, load_params_from_config(&config));
    load_with_progress(load, &model_path).await?;

    let (thinking_mode, assistant_prefix, max_think_tokens) =
        thinking_controls_from_config(&config);
    let gen_req = generate_request_from_prompt(
        Uuid::new_v4().to_string(),
        &args.prompt,
        GenerationSamplingPolicy::from_defaults(
            config.temperature,
            config.top_p,
            config.repeat_penalty,
            config.max_tokens,
            args.temperature,
            None,
            None,
            None,
            args.max_tokens,
        ),
        engine.worker_key_id.clone(),
    )
    .with_thinking_controls(None, thinking_mode, assistant_prefix, max_think_tokens);
    let gen_req = match &image_attachment {
        Some(img) => attach_image(gen_req, img)?,
        None => gen_req,
    };

    let done = engine
        .generate_streaming(gen_req, |token| {
            print!("{token}");
            let _ = std::io::stdout().flush();
        })
        .await?;

    println!();
    eprintln!(
        "\n[{} tokens, {:.2} tok/s]",
        done.tokens,
        done.decode_tok_s.unwrap_or(0.0)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Through a server, chat streams (full timings), sends an image as a data-URI
    // content part, and leaves sampling to the server's config unless given.
    #[test]
    fn the_server_request_streams_and_carries_only_what_was_asked() {
        let mut args = ChatArgs {
            model: None,
            prompt: "hi".to_string(),
            max_tokens: None,
            temperature: None,
            attach: Vec::new(),
        };
        let body = server_chat_request("m", &args, None);
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["content"], "hi");
        assert!(body.get("max_tokens").is_none() && body.get("temperature").is_none());
        args.max_tokens = Some(7);
        args.temperature = Some(0.0);
        let body = server_chat_request("m", &args, Some("QUJD"));
        assert_eq!(body["max_tokens"], 7);
        assert_eq!(body["temperature"], 0.0);
        let parts = &body["messages"][0]["content"];
        assert_eq!(parts[0]["text"], "hi");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,QUJD");
    }

    #[test]
    fn classify_attachment_by_extension() {
        let img = |p: &str| classify_attachment(Path::new(p)).unwrap();
        assert_eq!(img("a.png"), AttachKind::Image);
        assert_eq!(img("a.JPG"), AttachKind::Image);
        assert_eq!(img("a.webp"), AttachKind::Image);
        assert_eq!(
            classify_attachment(Path::new("a.txt")).unwrap(),
            AttachKind::Text
        );
        assert_eq!(
            classify_attachment(Path::new("a.mp4")).unwrap(),
            AttachKind::Video
        );
        assert_eq!(
            classify_attachment(Path::new("a.wav")).unwrap(),
            AttachKind::Audio
        );
        assert!(classify_attachment(Path::new("a.xyz")).is_err());
        assert!(classify_attachment(Path::new("noext")).is_err());
    }

    #[test]
    fn resolve_image_attachment_rules() {
        // Empty → None.
        assert!(resolve_image_attachment(&[]).unwrap().is_none());
        // A recognized-but-unwired kind errors.
        assert!(resolve_image_attachment(&[PathBuf::from("notes.txt")]).is_err());
        // Two images error (not yet supported).
        assert!(
            resolve_image_attachment(&[PathBuf::from("a.png"), PathBuf::from("b.jpg")]).is_err()
        );
        // A single image that doesn't exist errors with "file not found".
        let err = resolve_image_attachment(&[PathBuf::from("/no/such/img.png")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("file not found"), "got: {err}");
    }

    #[test]
    fn load_params_from_config_preserves_cli_load_policy() {
        let config = HipfireConfig {
            max_seq: 8192,
            kv_cache: "asym3".to_string(),
            flash_mode: "auto".to_string(),
            dflash_draft: "off".to_string(),
            cask_sidecar: Some("/models/qwen3.5-27b.triattn.hfq".to_string()),
            ..Default::default()
        };

        let params = load_params_from_config(&config);

        assert_eq!(params.max_seq, 8192);
        assert_eq!(params.kv_cache.as_deref(), Some("asym3"));
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode.as_deref(), Some("off"));
        assert_eq!(
            params.cask_sidecar.as_deref(),
            Some("/models/qwen3.5-27b.triattn.hfq")
        );
    }

    #[test]
    fn load_params_from_config_omits_auto_and_empty_sidecar() {
        let config = HipfireConfig {
            max_seq: 4096,
            kv_cache: "auto".to_string(),
            flash_mode: "auto".to_string(),
            dflash_draft: "auto".to_string(),
            cask_sidecar: Some(String::new()),
            ..Default::default()
        };

        let params = load_params_from_config(&config);

        assert_eq!(params.max_seq, 4096);
        assert_eq!(params.kv_cache, None);
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode, None);
        assert_eq!(params.cask_sidecar, None);
    }

    #[test]
    fn generate_request_from_prompt_preserves_structured_boundary() {
        let req = generate_request_from_prompt(
            "req-1".to_string(),
            "hello",
            GenerationSamplingPolicy::greedy(8),
            Some("worker-a".to_string()),
        );

        assert_eq!(req.prompt, "hello");
        assert!(!req.prompt.contains("<|im_start|>"));
        assert_eq!(req.worker_key_id.as_deref(), Some("worker-a"));
        let messages = req.messages.as_ref().expect("structured messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(serde_json::to_value(&messages[0]).unwrap()["role"], "user");
        assert_eq!(messages[0].content, "hello");
    }

    #[test]
    fn thinking_controls_follow_config_default_off() {
        let config = HipfireConfig::default();
        let (thinking_mode, assistant_prefix, max_think_tokens) =
            thinking_controls_from_config(&config);
        assert_eq!(thinking_mode.as_deref(), Some("chat"));
        assert_eq!(assistant_prefix.as_deref(), Some("closed_think"));
        assert_eq!(max_think_tokens, Some(1));

        let req = generate_request_from_prompt(
            "req-1".to_string(),
            "hello",
            GenerationSamplingPolicy::greedy(8),
            None,
        )
        .with_thinking_controls(None, thinking_mode, assistant_prefix, max_think_tokens);

        assert_eq!(req.thinking_mode.as_deref(), Some("chat"));
        assert_eq!(req.assistant_prefix.as_deref(), Some("closed_think"));
        assert_eq!(req.max_think_tokens, Some(1));
    }

    #[test]
    fn thinking_controls_enable_open_think_when_config_on() {
        let config = HipfireConfig {
            thinking: "on".to_string(),
            ..Default::default()
        };

        let (thinking_mode, assistant_prefix, max_think_tokens) =
            thinking_controls_from_config(&config);

        assert_eq!(thinking_mode.as_deref(), Some("thinking"));
        assert_eq!(assistant_prefix.as_deref(), Some("open_think"));
        assert_eq!(max_think_tokens, None);
    }
}
