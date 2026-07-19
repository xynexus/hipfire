// SPDX-License-Identifier: Apache-2.0
//! CLI for the hipfire-steer driver, driven through a `hipfire-daemon`
//! subprocess: load a model + the +/- prompt sets, run capture → derive →
//! apply-sweep → score (all through the daemon's correct inference, templating,
//! and KLD), and print the Pareto front.
//!
//! ```text
//! cargo run --release -p hipfire-steer-harness -- \
//!     --hfq ~/.hipfire/models/medgemma-4b-it.q8f16.hfq \
//!     --data-dir crates/hipfire-steer/data/medical \
//!     --limit 16 --eval-limit 16 --mode ablate --strengths 0.5,1.0,1.5
//! ```

use std::error::Error;
use std::path::{Path, PathBuf};

use hipfire_steer::driver::{load_prompts, run_driver, DriverConfig, ModelHarness, Prompt};
use hipfire_steer::SteerMode;
use hipfire_steer_harness::{DaemonHarness, HttpHarness};

const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

struct Args {
    /// Required in the default (spawn-own-daemon) mode; optional in server mode
    /// where it only supplies the default `--model` stem.
    hfq: Option<String>,
    /// When set, run as a THIN CLIENT of a live server at this URL (talking to its
    /// HTTP `/steer` + `/v1/chat/completions` routes) instead of spawning a private
    /// daemon that would take the GPU flock and rival the serving process.
    server_url: Option<String>,
    /// Model geometry — the server's `/steer/capture` needs it and a thin client
    /// can't read it from a load response it never made. Required in server mode.
    num_layers: Option<usize>,
    hidden: Option<usize>,
    /// Chat/generate model id for `/v1/chat/completions`; defaults to the `--hfq`
    /// file stem.
    model: Option<String>,
    data_dir: PathBuf,
    limit: usize,
    eval_limit: usize,
    strengths: Vec<f32>,
    modes: Vec<SteerMode>,
    max_new_tokens: usize,
    max_seq: usize,
    orthogonalize: bool,
    /// When set, load this `.lora` and show base vs applied vs scale-0 refusals.
    apply_lora: Option<PathBuf>,
    /// When set, stack two copies of this `.lora` at +/- scale to show they sum.
    stack_demo: Option<PathBuf>,
    /// Just load `--hfq` and report bad-eval refusals + any auto-loaded adapters.
    eval_refusals: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut hfq = None;
    let mut server_url = None;
    let mut num_layers = None;
    let mut hidden = None;
    let mut model = None;
    let mut data_dir = PathBuf::from("crates/hipfire-steer/data/medical");
    let mut limit = 16usize;
    let mut eval_limit = 16usize;
    let mut strengths = vec![1.0f32];
    let mut modes = vec![SteerMode::Ablate];
    let mut max_new_tokens = 64usize;
    let mut max_seq = 2048usize;
    let mut orthogonalize = true;
    let mut apply_lora = None;
    let mut stack_demo = None;
    let mut eval_refusals = false;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut next = || it.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--hfq" => hfq = Some(next()?),
            "--server-url" => server_url = Some(next()?),
            "--num-layers" => num_layers = Some(next()?.parse().map_err(|_| "bad --num-layers")?),
            "--hidden" => hidden = Some(next()?.parse().map_err(|_| "bad --hidden")?),
            "--model" => model = Some(next()?),
            "--data-dir" => data_dir = PathBuf::from(next()?),
            "--limit" => limit = next()?.parse().map_err(|_| "bad --limit")?,
            "--eval-limit" => eval_limit = next()?.parse().map_err(|_| "bad --eval-limit")?,
            "--max-new-tokens" => {
                max_new_tokens = next()?.parse().map_err(|_| "bad --max-new-tokens")?
            }
            "--max-seq" => max_seq = next()?.parse().map_err(|_| "bad --max-seq")?,
            "--no-orthogonalize" => orthogonalize = false,
            "--apply-lora" => apply_lora = Some(PathBuf::from(next()?)),
            "--stack-demo" => stack_demo = Some(PathBuf::from(next()?)),
            "--eval-refusals" => eval_refusals = true,
            "--strengths" => {
                strengths = next()?
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().map_err(|_| "bad --strengths"))
                    .collect::<Result<_, _>>()?;
            }
            "--mode" => {
                modes = match next()?.as_str() {
                    "steer" => vec![SteerMode::Steer],
                    "ablate" => vec![SteerMode::Ablate],
                    "both" => vec![SteerMode::Steer, SteerMode::Ablate],
                    other => {
                        return Err(format!("--mode: expected steer|ablate|both, got {other}"))
                    }
                };
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: hipfire-steer --hfq <model.hfq> [--data-dir DIR] [--limit N] \
                     [--eval-limit N] [--strengths a,b,c] [--mode steer|ablate|both] \
                     [--max-new-tokens N] [--max-seq N] [--no-orthogonalize]\n\
                     thin-client (talk to a live server, no private daemon):\n\
                     \x20 --server-url URL --num-layers N --hidden H [--model NAME]\n\
                     runtime demos: [--apply-lora PATH] [--stack-demo PATH] [--eval-refusals]\n\
                     (adapter export / merge / convert live in `hipfire-coexistence lora ...`)"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    // In the default mode `--hfq` is required (it names the model to load); in
    // server mode the server already holds the model, so `--hfq` is optional and
    // only feeds the `--model` default.
    if server_url.is_none() && hfq.is_none() {
        return Err("--hfq is required (or use --server-url for thin-client mode)".to_string());
    }
    Ok(Args {
        hfq,
        server_url,
        num_layers,
        hidden,
        model,
        data_dir,
        limit,
        eval_limit,
        strengths,
        modes,
        max_new_tokens,
        max_seq,
        orthogonalize,
        apply_lora,
        stack_demo,
        eval_refusals,
    })
}

fn load_set(dir: &Path, name: &str, limit: usize) -> Result<Vec<Prompt>, Box<dyn Error>> {
    let path = dir.join(name);
    let mut prompts = load_prompts(&path, SYSTEM_PROMPT)
        .map_err(|e| format!("loading {}: {e}", path.display()))?;
    prompts.truncate(limit);
    Ok(prompts)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let good_prompts = load_set(&args.data_dir, "good_prompts.txt", args.limit)?;
    let bad_prompts = load_set(&args.data_dir, "bad_prompts.txt", args.limit)?;
    let good_eval = load_set(&args.data_dir, "good_eval.txt", args.eval_limit)?;
    let bad_eval = load_set(&args.data_dir, "bad_eval.txt", args.eval_limit)?;
    eprintln!(
        "prompts: {} good / {} bad (direction), {} good / {} bad (eval)",
        good_prompts.len(),
        bad_prompts.len(),
        good_eval.len(),
        bad_eval.len()
    );

    // Thin-client mode: drive a live server over HTTP (no private daemon, no rival
    // GPU flock). The lora/eval demos are daemon-only; server mode is the driver.
    if let Some(url) = args.server_url.clone() {
        let model = args
            .model
            .clone()
            .or_else(|| {
                args.hfq.as_deref().and_then(|h| {
                    Path::new(h)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                })
            })
            .ok_or("server mode needs --model (or --hfq to default its stem)")?;
        let num_layers = args.num_layers.ok_or(
            "server mode needs --num-layers (the server can't report geometry to a thin client)",
        )?;
        let hidden = args.hidden.ok_or(
            "server mode needs --hidden (the server can't report geometry to a thin client)",
        )?;
        eprintln!("thin client → {url} (model {model}, {num_layers} layers, hidden {hidden}) ...");
        let mut harness = HttpHarness::connect(
            url,
            model,
            num_layers,
            hidden,
            SYSTEM_PROMPT.to_string(),
            args.max_new_tokens,
        )?;
        return run_and_report(
            &args,
            good_prompts,
            bad_prompts,
            good_eval,
            bad_eval,
            &mut harness,
        );
    }

    let hfq = args.hfq.clone().ok_or("--hfq is required")?;
    let daemon_bin = hipfire_daemon_adapter::find_daemon_bin_or_error()?;
    eprintln!("loading {} via daemon {} ...", hfq, daemon_bin.display());
    let tmp = std::env::temp_dir().join(format!("hipfire-steer-{}", std::process::id()));
    let mut harness = DaemonHarness::connect(
        &daemon_bin,
        Path::new(&hfq),
        args.max_seq,
        args.max_new_tokens,
        SYSTEM_PROMPT.to_string(),
        tmp,
    )?;

    // Apply mode: load a `.lora` and compare base / applied / scale-0 refusals.
    if let Some(path) = args.apply_lora.as_ref() {
        return apply_lora(&mut harness, &bad_eval, path);
    }

    // Stack demo: load two copies of one adapter at +s / -s to show they sum.
    if let Some(path) = args.stack_demo.as_ref() {
        let s = args.strengths.first().copied().unwrap_or(0.2);
        return stack_demo(&mut harness, &bad_eval, path, s);
    }

    // Eval-refusals: just report bad-eval refusals + any auto-loaded adapters
    // (e.g. a `--merge-lora` model auto-applies its adapter at load).
    if args.eval_refusals {
        return eval_refusals(&mut harness, &bad_eval);
    }

    run_and_report(
        &args,
        good_prompts,
        bad_prompts,
        good_eval,
        bad_eval,
        &mut harness,
    )
}

/// Build the driver config, run the sweep, and print the Pareto report. Shared by
/// the daemon and thin-client (HTTP) paths — both reach the model through the
/// `ModelHarness` trait, so the driver code is identical.
fn run_and_report(
    args: &Args,
    good_prompts: Vec<Prompt>,
    bad_prompts: Vec<Prompt>,
    good_eval: Vec<Prompt>,
    bad_eval: Vec<Prompt>,
    harness: &mut dyn ModelHarness,
) -> Result<(), Box<dyn Error>> {
    let cfg = DriverConfig {
        good_prompts,
        bad_prompts,
        good_eval,
        bad_eval,
        modes: args.modes.clone(),
        strengths: args.strengths.clone(),
        layer_range: 0..harness.num_layers(),
        orthogonalize: args.orthogonalize,
        markers: DriverConfig::default_markers(),
    };

    eprintln!("running driver ({} layers) ...", harness.num_layers());
    let report = run_driver(&cfg, harness)?;

    println!("\n=== steer driver report ===");
    println!(
        "base refusals: {}/{}",
        report.base_refusals, report.n_bad_eval
    );
    println!("  (* = Pareto-optimal on refusals↓ + KLD↓)");
    for (i, t) in report.trials.iter().enumerate() {
        let star = if report.pareto.contains(&i) { "*" } else { " " };
        println!(
            "{star} {:?} strength={:.2}  refusals={:>3}/{:<3}  kld={:.4}",
            t.mode, t.strength, t.refusals, report.n_bad_eval, t.kl_divergence
        );
    }
    Ok(())
}

/// Load a `.lora` adapter into the live daemon and report refusal counts on the
/// bad-eval set for three states: base (no adapter), adapter applied at its baked
/// scale, and the same adapter dialed to scale 0 (≡ base) — proving load + the GPU
/// stack apply + the live intensity knob end to end.
fn apply_lora(
    harness: &mut DaemonHarness,
    bad_eval: &[Prompt],
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    use hipfire_steer::driver::count_refusals;
    let markers = DriverConfig::default_markers();
    let n = bad_eval.len();

    let base = harness
        .generate(bad_eval)
        .map_err(|e| format!("generate base: {e}"))?;
    let base_ref = count_refusals(&base, &markers);

    harness
        .lora_load(path, None, None)
        .map_err(|e| format!("lora_load: {e}"))?;
    let loaded = harness.lora_list().map_err(|e| format!("lora_list: {e}"))?;
    eprintln!("loaded adapters: {loaded:?}");

    let applied = harness
        .generate(bad_eval)
        .map_err(|e| format!("generate applied: {e}"))?;
    let applied_ref = count_refusals(&applied, &markers);

    let id = loaded
        .first()
        .map(|(id, _)| id.clone())
        .ok_or("apply-lora: no adapter loaded")?;
    harness
        .lora_set_scale(&id, 0.0)
        .map_err(|e| format!("lora_set_scale: {e}"))?;
    let off = harness
        .generate(bad_eval)
        .map_err(|e| format!("generate scale0: {e}"))?;
    let off_ref = count_refusals(&off, &markers);

    println!("\n=== lora apply report ===");
    println!("base (no adapter):          refusals {base_ref}/{n}");
    println!("adapter applied (default):  refusals {applied_ref}/{n}");
    println!("adapter scale=0 (≡ base):   refusals {off_ref}/{n}");
    Ok(())
}

/// Load `--hfq` and report bad-eval refusals + any adapters the daemon auto-loaded
/// at model load (a merged model applies its bundled adapter automatically).
fn eval_refusals(harness: &mut DaemonHarness, bad_eval: &[Prompt]) -> Result<(), Box<dyn Error>> {
    use hipfire_steer::driver::count_refusals;
    let markers = DriverConfig::default_markers();
    let loaded = harness.lora_list().unwrap_or_default();
    let resp = harness
        .generate(bad_eval)
        .map_err(|e| format!("generate: {e}"))?;
    let refusals = count_refusals(&resp, &markers);
    println!("\n=== eval refusals ===");
    println!("auto-loaded adapters: {loaded:?}");
    println!("refusals: {refusals}/{}", bad_eval.len());
    Ok(())
}

/// Stack two copies of one adapter to show the GPU apply sums them: load "a" at
/// `+s` (refusals drop), then load "b" (same directions) at `-s` — the deltas
/// cancel (`+s·δ − s·δ = 0`), so refusals return to base. Exercises the
/// multi-adapter stack path on hardware.
fn stack_demo(
    harness: &mut DaemonHarness,
    bad_eval: &[Prompt],
    path: &Path,
    s: f32,
) -> Result<(), Box<dyn Error>> {
    use hipfire_steer::driver::count_refusals;
    let markers = DriverConfig::default_markers();
    let n = bad_eval.len();

    harness
        .lora_load(path, Some(s), Some("a"))
        .map_err(|e| format!("lora_load a: {e}"))?;
    let one = harness
        .generate(bad_eval)
        .map_err(|e| format!("generate a: {e}"))?;
    let one_ref = count_refusals(&one, &markers);
    eprintln!(
        "after load a@{s:.2}: {:?}",
        harness.lora_list().unwrap_or_default()
    );

    harness
        .lora_load(path, Some(-s), Some("b"))
        .map_err(|e| format!("lora_load b: {e}"))?;
    let both = harness
        .generate(bad_eval)
        .map_err(|e| format!("generate a+b: {e}"))?;
    let both_ref = count_refusals(&both, &markers);
    eprintln!(
        "after load b@{:.2}: {:?}",
        -s,
        harness.lora_list().unwrap_or_default()
    );

    println!("\n=== lora stack demo ===");
    println!("a @ {s:+.2} (single):        refusals {one_ref}/{n}");
    println!(
        "a @ {s:+.2} + b @ {:+.2} (sum≈0): refusals {both_ref}/{n}",
        -s
    );
    Ok(())
}
