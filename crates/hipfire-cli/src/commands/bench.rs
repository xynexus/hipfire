use std::{future::Future, time::Instant};

use clap::Args;
use hipfire_config::{HipfireConfig, LoadedConfig};
use hipfire_daemon_adapter::{find_daemon_bin_or_error, DaemonEngine};
use hipfire_generate::{DoneEvent, GenerateTextRequest, GenerationSamplingPolicy};
use hipfire_model::ModelLoadParams;
use serde_json::json;
use uuid::Uuid;

use crate::model::find_model;

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  hipfire bench Qwen3.5-30B-A3B\n  hipfire bench --pp-tokens 512 --tg-tokens 128 --repetitions 5\n  hipfire bench Qwen3.5-30B-A3B --json\n"
)]
pub struct BenchArgs {
    /// Model name, shorthand, alias, or path. Falls back to default_model.
    pub model: Option<String>,
    /// Target prompt/prefill token count. The daemon reports the actual count.
    #[arg(long, default_value_t = 512)]
    pub pp_tokens: usize,
    /// Generated token count for the decode-throughput sample.
    #[arg(long, default_value_t = 128)]
    pub tg_tokens: u32,
    /// Number of measured repetitions, matching llama-bench's default.
    #[arg(short = 'r', long = "repetitions", default_value_t = 5)]
    pub repetitions: usize,
    /// Skip warmup runs before measuring.
    #[arg(long)]
    pub no_warmup: bool,
    /// Print JSON instead of a compact text report.
    #[arg(long)]
    pub json: bool,
}

pub async fn run(args: BenchArgs, loaded: LoadedConfig) -> anyhow::Result<()> {
    if args.repetitions == 0 {
        anyhow::bail!("--repetitions must be at least 1");
    }
    let model = args
        .model
        .as_deref()
        .or(loaded.config.default_model.as_deref())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model specified and no `default_model` configured; \
                 pass <model> or set `default_model` in {}",
                hipfire_config::config_path().display()
            )
        })?
        .to_string();
    let model_path = find_model(&model, &loaded.config)
        .ok_or_else(|| anyhow::anyhow!("model not found: {model}"))?;
    let config: HipfireConfig = loaded.resolve_for_model(&model).config;
    if let Some(url) = running_server_url(&loaded).await {
        return run_server_bench(args, model, model_path.display().to_string(), url).await;
    }
    let mut load_params = ModelLoadParams::from_hipfire_config(&config);
    let needed_seq = args
        .pp_tokens
        .saturating_add(args.tg_tokens as usize)
        .saturating_add(512)
        .min(u32::MAX as usize) as u32;
    load_params.max_seq = load_params.max_seq.max(needed_seq);

    let bin = find_daemon_bin_or_error()?;
    // Spawns a private daemon on purpose: a benchmark must measure the binary it
    // was pointed at, and a shared daemon would have another model resident.
    let mut engine = DaemonEngine::spawn(&bin).await?;

    progress(format!("loading {}", model_path.display()));
    let load_start = Instant::now();
    let loaded = engine
        .load(&model_path.to_string_lossy(), load_params)
        .await?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    progress(format!("loaded in {load_ms:.1} ms"));
    if !args.no_warmup {
        progress(format!("warmup: pp{}", args.pp_tokens));
        let _ = with_heartbeat(
            format!("warmup pp{}", args.pp_tokens),
            engine.bench_prefill(args.pp_tokens),
        )
        .await?;
        with_heartbeat("warmup reset".to_string(), engine.reset()).await?;
        progress("warmup: tg1");
        let _ = with_heartbeat(
            "warmup tg1".to_string(),
            generate_tg_sample(&mut engine, &loaded.worker_key_id, 1),
        )
        .await?;
        with_heartbeat("warmup reset".to_string(), engine.reset()).await?;
    }
    let mut samples = Vec::with_capacity(args.repetitions);
    for index in 0..args.repetitions {
        let sample_index = index + 1;
        progress(format!(
            "sample {sample_index}/{}: pp{}",
            args.repetitions, args.pp_tokens
        ));
        let prefill = with_heartbeat(
            format!(
                "sample {sample_index}/{} pp{}",
                args.repetitions, args.pp_tokens
            ),
            engine.bench_prefill(args.pp_tokens),
        )
        .await?;
        progress(format!(
            "sample {sample_index}/{}: pp{} done ({:.2} t/s)",
            args.repetitions, prefill.tokens, prefill.tok_s
        ));
        with_heartbeat(
            format!("sample {sample_index}/{} reset", args.repetitions),
            engine.reset(),
        )
        .await?;
        progress(format!(
            "sample {sample_index}/{}: tg{}",
            args.repetitions, args.tg_tokens
        ));
        let mut sample = with_heartbeat(
            format!(
                "sample {sample_index}/{} tg{}",
                args.repetitions, args.tg_tokens
            ),
            generate_tg_sample(&mut engine, &loaded.worker_key_id, args.tg_tokens),
        )
        .await?;
        if let Some(tg_tok_s) = sample.tg_tok_s {
            progress(format!(
                "sample {sample_index}/{}: tg{} done ({tg_tok_s:.2} t/s)",
                args.repetitions, sample.tg_tokens
            ));
        } else {
            progress(format!(
                "sample {sample_index}/{}: tg{} done",
                args.repetitions, sample.tg_tokens
            ));
        }
        sample.pp_tokens = prefill.tokens;
        sample.pp_ms = Some(prefill.ms);
        sample.pp_tok_s = Some(prefill.tok_s);
        samples.push(sample);
        with_heartbeat(
            format!("sample {sample_index}/{} reset", args.repetitions),
            engine.reset(),
        )
        .await?;
    }
    let report = BenchReport {
        model,
        path: model_path.display().to_string(),
        server_url: None,
        load_ms: Some(load_ms),
        pp_target: args.pp_tokens,
        tg_target: args.tg_tokens,
        repetitions: args.repetitions,
        warmup: !args.no_warmup,
        samples,
    };

    if args.json {
        print_json_report(&report);
    } else {
        print_text_report(&report);
    }
    progress("complete");

    Ok(())
}

async fn running_server_url(loaded: &LoadedConfig) -> Option<String> {
    let host = if loaded.config.host == "0.0.0.0" {
        "127.0.0.1"
    } else {
        loaded.config.host.as_str()
    };
    let url = format!("http://{}:{}", host, loaded.config.port);
    let ok = reqwest::Client::new()
        .get(format!("{url}/health"))
        .send()
        .await
        .ok()
        .is_some_and(|response| response.status().is_success());
    ok.then_some(url)
}

async fn run_server_bench(
    args: BenchArgs,
    model: String,
    model_path: String,
    server_url: String,
) -> anyhow::Result<()> {
    progress(format!("using running server at {server_url}"));
    warn_server_reset(&server_url);
    let client = reqwest::Client::new();
    if !args.no_warmup {
        progress("warmup: reset server state");
        with_heartbeat(
            "warmup server reset".to_string(),
            server_reset(&client, &server_url),
        )
        .await?;
        progress(format!("warmup: pp{}+tg1", args.pp_tokens));
        let _ = with_heartbeat(
            format!("warmup pp{}+tg1", args.pp_tokens),
            server_bench_sample(&client, &server_url, &model, args.pp_tokens, 1),
        )
        .await?;
    }
    let mut samples = Vec::with_capacity(args.repetitions);
    for index in 0..args.repetitions {
        let sample_index = index + 1;
        progress(format!(
            "sample {sample_index}/{}: reset server state",
            args.repetitions
        ));
        with_heartbeat(
            format!("sample {sample_index}/{} server reset", args.repetitions),
            server_reset(&client, &server_url),
        )
        .await?;
        progress(format!(
            "sample {sample_index}/{}: pp{}+tg{}",
            args.repetitions, args.pp_tokens, args.tg_tokens
        ));
        let sample = with_heartbeat(
            format!(
                "sample {sample_index}/{} pp{}+tg{}",
                args.repetitions, args.pp_tokens, args.tg_tokens
            ),
            server_bench_sample(&client, &server_url, &model, args.pp_tokens, args.tg_tokens),
        )
        .await?;
        match (sample.pp_tok_s, sample.tg_tok_s) {
            (Some(pp), Some(tg)) => progress(format!(
                "sample {sample_index}/{}: done (pp {:.2} t/s, tg {:.2} t/s)",
                args.repetitions, pp, tg
            )),
            _ => progress(format!("sample {sample_index}/{}: done", args.repetitions)),
        }
        samples.push(sample);
    }
    let report = BenchReport {
        model,
        path: model_path,
        server_url: Some(server_url),
        load_ms: None,
        pp_target: args.pp_tokens,
        tg_target: args.tg_tokens,
        repetitions: args.repetitions,
        warmup: !args.no_warmup,
        samples,
    };
    if args.json {
        print_json_report(&report);
    } else {
        print_text_report(&report);
    }
    progress("complete");
    Ok(())
}

fn progress(message: impl AsRef<str>) {
    eprintln!("[hipfire bench] {}", message.as_ref());
}

async fn with_heartbeat<T, F>(label: String, future: F) -> anyhow::Result<T>
where
    F: Future<Output = anyhow::Result<T>>,
{
    let started = Instant::now();
    tokio::pin!(future);
    loop {
        tokio::select! {
            result = &mut future => return result,
            _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                progress(format!("{label}: still running ({}s)", started.elapsed().as_secs()));
            }
        }
    }
}

fn synthetic_prefill_prompt(target_tokens: usize) -> String {
    let mut prompt = String::from(
        "Benchmark prompt. Continue with a short neutral answer after reading this payload.\n\n",
    );
    for i in 0..target_tokens {
        if i > 0 {
            prompt.push(' ');
        }
        prompt.push_str("the");
    }
    prompt
}

fn synthetic_generation_prompt() -> String {
    "Benchmark decode prompt. Continue with neutral filler text.".to_string()
}

#[derive(Clone, Debug)]
struct BenchSample {
    pp_tokens: usize,
    pp_ms: Option<f64>,
    pp_tok_s: Option<f64>,
    tg_tokens: u64,
    tg_tok_s: Option<f64>,
    ttft_ms: Option<f64>,
    generate_wall_ms: f64,
    output_bytes: usize,
}

#[derive(Debug)]
struct BenchReport {
    model: String,
    path: String,
    server_url: Option<String>,
    load_ms: Option<f64>,
    pp_target: usize,
    tg_target: u32,
    repetitions: usize,
    warmup: bool,
    samples: Vec<BenchSample>,
}

async fn generate_tg_sample(
    engine: &mut DaemonEngine,
    worker_key_id: &str,
    tg_tokens: u32,
) -> anyhow::Result<BenchSample> {
    let req = GenerateTextRequest::from_prompt(
        Uuid::new_v4().to_string(),
        synthetic_generation_prompt(),
        GenerationSamplingPolicy::greedy(tg_tokens),
    )
    .with_worker_key_id(Some(worker_key_id.to_string()));
    let generate_start = Instant::now();
    let (text, done) = engine.generate(req).await?;
    let wall_ms = generate_start.elapsed().as_secs_f64() * 1000.0;
    Ok(BenchSample {
        pp_tokens: done.prefill_tokens.unwrap_or(0) as usize,
        pp_ms: done.prefill_ms,
        pp_tok_s: prefill_tok_s(&done),
        tg_tokens: done.tokens as u64,
        tg_tok_s: decode_tok_s(&done),
        ttft_ms: done.ttft_ms,
        generate_wall_ms: wall_ms,
        output_bytes: text.len(),
    })
}

async fn server_bench_sample(
    client: &reqwest::Client,
    server_url: &str,
    model: &str,
    pp_tokens: usize,
    tg_tokens: u32,
) -> anyhow::Result<BenchSample> {
    let prompt = synthetic_prefill_prompt(pp_tokens);
    let started = Instant::now();
    let response: serde_json::Value = client
        .post(format!("{server_url}/v1/chat/completions"))
        .json(&json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": tg_tokens,
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let timings = response.get("timings").cloned().unwrap_or_default();
    let text = response
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.pointer("/message/content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    Ok(BenchSample {
        pp_tokens: timings
            .get("prefill_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize,
        pp_ms: timings.get("prefill_ms").and_then(|v| v.as_f64()),
        pp_tok_s: timings.get("prefill_tok_s").and_then(|v| v.as_f64()),
        tg_tokens: timings.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        tg_tok_s: timings
            .get("decode_tok_s")
            .or_else(|| timings.get("tok_s"))
            .and_then(|v| v.as_f64()),
        ttft_ms: timings.get("ttft_ms").and_then(|v| v.as_f64()),
        generate_wall_ms: wall_ms,
        output_bytes: text.len(),
    })
}

async fn server_reset(client: &reqwest::Client, server_url: &str) -> anyhow::Result<()> {
    let mut request = client.post(format!("{server_url}/admin/runtime/reset"));
    if let Some(secret) = hipfire_config::read_admin_secret() {
        request = request.bearer_auth(secret);
    }
    let response = request.send().await?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("server reset failed with {status}: {body}");
    }
    Ok(())
}

fn warn_server_reset(server_url: &str) {
    eprintln!(
        "\x1b[1;31mWARNING: hipfire bench is attached to a running server at {server_url}.\x1b[0m"
    );
    eprintln!(
        "\x1b[1;31mWARNING: benchmark repetitions will reset the server daemon state and clear active KV/session state.\x1b[0m"
    );
}

fn prefill_tok_s(done: &DoneEvent) -> Option<f64> {
    done.prefill_tok_s.or_else(|| {
        let tokens = done.prefill_tokens?;
        let ms = done.prefill_ms?;
        (ms > 0.0).then(|| tokens as f64 * 1000.0 / ms)
    })
}

fn decode_tok_s(done: &DoneEvent) -> Option<f64> {
    done.decode_tok_s.or(done.tok_s)
}

fn values(samples: &[BenchSample], f: impl Fn(&BenchSample) -> Option<f64>) -> Vec<f64> {
    samples.iter().filter_map(f).collect()
}

fn avg(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn stdev(v: &[f64]) -> Option<f64> {
    if v.len() <= 1 {
        return Some(0.0);
    }
    let mean = avg(v)?;
    let var = v
        .iter()
        .map(|x| {
            let delta = x - mean;
            delta * delta
        })
        .sum::<f64>()
        / (v.len() - 1) as f64;
    Some(var.sqrt())
}

fn avg_or_zero(v: &[f64]) -> f64 {
    avg(v).unwrap_or(0.0)
}

fn stdev_or_zero(v: &[f64]) -> f64 {
    stdev(v).unwrap_or(0.0)
}

fn print_text_report(report: &BenchReport) {
    println!("model: {}", report.model);
    println!("path: {}", report.path);
    if let Some(url) = &report.server_url {
        println!("server_url: {url}");
        println!("load_ms: n/a (server-owned daemon)");
    } else if let Some(load_ms) = report.load_ms {
        println!("load_ms: {:.1}", load_ms);
    }
    println!("repetitions: {}", report.repetitions);
    println!("warmup: {}", report.warmup);
    let pp = values(&report.samples, |s| s.pp_tok_s);
    let tg = values(&report.samples, |s| s.tg_tok_s);
    println!(
        "pp{} t/s: {:.2} ± {:.2}",
        report.pp_target,
        avg_or_zero(&pp),
        stdev_or_zero(&pp)
    );
    println!(
        "tg{} t/s: {:.2} ± {:.2}",
        report.tg_target,
        avg_or_zero(&tg),
        stdev_or_zero(&tg)
    );
    let ttft = values(&report.samples, |s| s.ttft_ms);
    if let Some(ttft_ms) = avg(&ttft) {
        println!("ttft_ms: {:.1}", ttft_ms);
    }
    let prefill_ms = values(&report.samples, |s| s.pp_ms);
    if let Some(ms) = avg(&prefill_ms) {
        println!("prefill_ms: {:.1}", ms);
    }
    let wall = values(&report.samples, |s| Some(s.generate_wall_ms));
    println!("generate_wall_ms: {:.1}", avg_or_zero(&wall));
    let actual_pp = values(&report.samples, |s| Some(s.pp_tokens as f64));
    let actual_tg = values(&report.samples, |s| Some(s.tg_tokens as f64));
    println!("actual_prefill_tokens: {:.1}", avg_or_zero(&actual_pp));
    println!("actual_decode_tokens: {:.1}", avg_or_zero(&actual_tg));
}

fn print_json_report(report: &BenchReport) {
    let pp = values(&report.samples, |s| s.pp_tok_s);
    let tg = values(&report.samples, |s| s.tg_tok_s);
    let payload = json!({
        "model": &report.model,
        "path": &report.path,
        "server_url": &report.server_url,
        "load_ms": report.load_ms,
        "pp_target": report.pp_target,
        "tg_target": report.tg_target,
        "repetitions": report.repetitions,
        "warmup": report.warmup,
        "pp_tok_s_avg": avg(&pp),
        "pp_tok_s_stdev": stdev(&pp),
        "tg_tok_s_avg": avg(&tg),
        "tg_tok_s_stdev": stdev(&tg),
        "samples": report.samples.iter().map(|s| json!({
            "prefill_tokens": s.pp_tokens,
            "prefill_ms": s.pp_ms,
            "prefill_tok_s": s.pp_tok_s,
            "decode_tokens": s.tg_tokens,
            "decode_tok_s": s.tg_tok_s,
            "ttft_ms": s.ttft_ms,
            "generate_wall_ms": s.generate_wall_ms,
            "output_bytes": s.output_bytes,
        })).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_prompt_scales_with_requested_prefill_size() {
        let small = synthetic_prefill_prompt(8);
        let large = synthetic_prefill_prompt(16);
        assert!(large.len() > small.len());
        assert!(small.contains("Benchmark prompt."));
    }

    #[test]
    fn prefill_rate_falls_back_to_tokens_over_ms() {
        let done = DoneEvent {
            id: "bench".to_string(),
            tokens: 128,
            tok_s: None,
            prefill_tokens: Some(512),
            prefill_ms: Some(256.0),
            prefill_tok_s: None,
            decode_tok_s: Some(12.0),
            ttft_ms: Some(300.0),
            finish_reason: None,
            response_id: None,
            extra: Default::default(),
        };
        assert_eq!(prefill_tok_s(&done), Some(2000.0));
        assert_eq!(decode_tok_s(&done), Some(12.0));
    }
}
