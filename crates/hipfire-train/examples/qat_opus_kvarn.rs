//! Light QAT (recovery-FT) prototype for an Opus weight-quantized model, with the
//! KVarN KV-cache-quant path in the loop — the deploy-faithful combination.
//!
//! TIER-PARAMETERIC as of 2026-08-28: `HIPFIRE_QAT_TIER=oq3|oq4|oq8` selects the
//! Opus tier the student is fake-quantized to, so the recoverable share of each
//! tier's deploy loss is measurable on ONE footing. Default `oq4` — OQ+ W4 is
//! the deployed Opus tier; `oq3` reproduces this file's original behaviour
//! (it was `qat_w3_kvarn.rs`, hardcoded to W3).
//!
//! Student = weights fake-quantized to the chosen tier (FROZEN) + trainable
//! LoRA(q/v) + RMSNorm; its attention forward optionally runs post-RoPE K/V through
//! KVarN-4bit + CASK merge (STE). Teacher = clean fp32. We KL-distill the student
//! toward the clean teacher and report the KL gap recovered, measured on an IN-SAMPLE
//! batch AND a HELD-OUT batch (calib≠eval, the rigor the multi-window run taught).
//!
//! The KVarN path (student-only; teacher stays clean) is gated by HIPFIRE_QAT_KVNOISE
//! (default 1). Whatever we TRAIN with, we EVAL with — so the held-out KL reflects the
//! deployed W3+KVarN condition. Set it to 0 to ablate (train/eval clean-KV).
//!
//! Run:
//!   source ./scripts/rocm-env.sh
//!   hipfire lock acquire "qat-w3-kvarn"
//!   cargo run -p hipfire-train --release --example qat_w3_kvarn [model_dir]
//!   hipfire lock release

use hipfire_model::tokenizer::Tokenizer;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_train::loader::{load_llama_fp32, LlamaWeightsF32};
use hipfire_train::model::{
    flatten_recovery_grads, model_distill_backward, model_forward, LlamaModel,
};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::optim::AdamW;
use hipfire_train::oqplus_quant::{
    oq3_simquant, oq4_mixed_simquant, oq8_simquant, oqplus_simquant,
};
use std::path::Path;

/// Default model. Must be a `model_type: llama` snapshot holding **safetensors** —
/// `load_llama_fp32` cannot read `.gguf`, and a snapshot dir that carries only a
/// `.gguf` dies with a bare "no .safetensors files found" before touching the GPU.
/// Note the `snapshots/main` pin: Llama-3.2-1B has a second, weightless snapshot
/// dir, so globbing `snapshots/*/` picks the wrong one.
const DEFAULT_DIR: &str = "/srv/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/main";
const SEQ: usize = 16;
const N_TRAIN: usize = 4; // sequences per optimizer step
const N_EVAL: usize = 4; // held-out sequences, synthetic path (disjoint tokens)
/// Distinct train batches cycled over on the corpus path, and held-out sequences
/// drawn there.
///
/// With ONE fixed batch, this loop trains 97 tensors on `N_TRAIN * SEQ` = 64 tokens
/// and overfits them flat: the first measured run drove in-sample KL 2.2430 -> 0.6114
/// (72.7% "recovered") while HELD-OUT went 2.5152 -> 2.9003, i.e. 15% WORSE. Cycling
/// a pool costs one extra teacher precompute and nothing per step, and is the
/// difference between measuring recovery and measuring memorisation.
const POOL_BATCHES: usize = 32;
const N_EVAL_CORPUS: usize = 32;
const RANK: usize = 16;
const ALPHA: f32 = 32.0;
const LR: f32 = 1e-3; // peak LR; override with HIPFIRE_QAT_LR
const STEPS: usize = 120;
/// Linear-warmup steps before the cosine decay.
///
/// A FLAT LR diverges at low-damage tiers, which is easy to mistake for "QAT does
/// not help". Measured on W8 KV-clean over real text, where the student starts
/// essentially undamaged: batch KL 0.0033 -> 0.529 -> 0.298 -> 0.365 -> 3.2295 by
/// step 80. There is nothing to recover at W8, so a flat 1e-3 on the trainable
/// RMSNorms just walks the model away from the teacher. Same root cause as the
/// DSpark drafter's non-convergence.
const WARMUP: usize = 10;

/// Which Opus tier the student is fake-quantized to.
///
/// All three share the FWHT-256 rotation and the symmetric clip-searched scale;
/// they differ only in code width, so a tier sweep isolates the width and
/// nothing else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tier {
    Oq3,
    Oq4,
    Oq8,
    /// Magnitude-tiered ("compact") W4: bulk int4 with `n_out` of each 256-group
    /// promoted to int8 on one shared scale. Effective width `4 + n_out/64`, so
    /// `oq4.25` is `n_out = 16`. This is the grid the mixed packers actually
    /// deploy; plain `Oq4` is uniform int4 and carries strictly more damage.
    Oq4Mixed(usize),
}

impl Tier {
    /// `HIPFIRE_QAT_TIER`, defaulting to the deployed tier (oq4 / OQ+ W4).
    fn from_env() -> Self {
        match std::env::var("HIPFIRE_QAT_TIER")
            .unwrap_or_else(|_| "oq4".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "oq3" | "w3" => Tier::Oq3,
            "oq8" | "w8" => Tier::Oq8,
            "oq4" | "w4" | "oqplus" | "oq+" => Tier::Oq4,
            // `oq4.<frac>`: mixed-precision widths carry a decimal place, per the
            // repo's artifact-naming rule. n_out = frac * 64 (int8 costs 4 extra
            // bits over int4, spread across the 256-group).
            other if other.starts_with("oq4.") => {
                let bits: f32 = other[2..]
                    .parse()
                    .unwrap_or_else(|_| panic!("HIPFIRE_QAT_TIER: bad width in {other:?}"));
                let n_out = ((bits - 4.0) * 64.0).round() as i64;
                assert!(
                    (1..256).contains(&n_out),
                    "HIPFIRE_QAT_TIER: {other:?} implies n_out {n_out}, want 1..256 \
                     (oq4.25 = 16, oq4.5 = 32)"
                );
                Tier::Oq4Mixed(n_out as usize)
            }
            other => {
                panic!("HIPFIRE_QAT_TIER: unknown tier {other:?} (want oq3|oq4|oq8|oq4.<frac>)")
            }
        }
    }

    fn simquant(self, w: &[f32]) -> Vec<f32> {
        match self {
            Tier::Oq3 => oq3_simquant(w),
            Tier::Oq4 => oqplus_simquant(w),
            Tier::Oq8 => oq8_simquant(w),
            Tier::Oq4Mixed(n_out) => oq4_mixed_simquant(w, n_out),
        }
    }

    /// Short deploy label, e.g. "W4".
    fn width(self) -> String {
        match self {
            Tier::Oq3 => "W3".to_string(),
            Tier::Oq4 => "W4".to_string(),
            Tier::Oq8 => "W8".to_string(),
            Tier::Oq4Mixed(n) => {
                // n/64 is a dyadic fraction, so 4 places is always exact; trim the
                // trailing zeros so `oq4.25` reports `W4.25`, not `W4.2500`.
                let w = format!("{:.4}", 4.0 + n as f32 / 64.0);
                w.trim_end_matches('0').trim_end_matches('.').to_string()
            }
        }
    }

    fn describe(self) -> String {
        match self {
            Tier::Oq3 => "Oq3 (W3, sym-int3 + FWHT-256)".to_string(),
            Tier::Oq4 => "OQ+ (W4, sym-int4 + FWHT-256, clip-searched)".to_string(),
            Tier::Oq8 => "Oq8 (W8, sym-int8 + FWHT-256)".to_string(),
            Tier::Oq4Mixed(n) => format!(
                "OQ+ mixed (bulk int4 + top-{n}/256 int8 on one shared scale, \
                 FWHT-256, joint clip-search) — the grid the compact packers deploy"
            ),
        }
    }
}

/// Linear warmup to `peak`, then cosine decay to ~0 over the remaining steps.
fn lr_at(step: usize, peak: f32) -> f32 {
    if step < WARMUP {
        peak * (step + 1) as f32 / WARMUP as f32
    } else {
        let t = (step - WARMUP) as f32 / (STEPS - WARMUP).max(1) as f32;
        0.5 * peak * (1.0 + (std::f32::consts::PI * t).cos())
    }
}

/// Opus sim-quant the 7 linears per layer (base weights, frozen thereafter).
fn quantize_linears(
    gpu: &mut Gpu,
    w: &mut LlamaWeightsF32,
    tier: Tier,
) -> Result<(), Box<dyn std::error::Error>> {
    for l in w.layers.iter_mut() {
        for t in [
            &mut l.q_proj,
            &mut l.k_proj,
            &mut l.v_proj,
            &mut l.o_proj,
            &mut l.gate_proj,
            &mut l.up_proj,
            &mut l.down_proj,
        ] {
            let host = gpu.download_f32(t)?;
            let q = tier.simquant(&host);
            *t = gpu.upload_f32(&q, &t.shape.clone())?;
        }
    }
    Ok(())
}

/// Mean KL(teacher‖student) over a batch (forward-only use of the distill op; grads
/// discarded). Runs under whatever KVarN env is currently set.
fn eval_kl(
    gpu: &mut Gpu,
    student: &LlamaModel,
    batch: &[Vec<u32>],
    teacher_p: &[GpuTensor],
    pos: &[f32],
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut total = 0.0f32;
    for (si, toks) in batch.iter().enumerate() {
        let acts = model_forward(gpu, student, toks, pos)?;
        let (kl, _g, _d) = model_distill_backward(gpu, student, &acts, &teacher_p[si])?;
        total += kl;
    }
    Ok(total / (batch.len() * SEQ) as f32)
}

/// Real-text batches from `HIPFIRE_QAT_CORPUS`, tokenized with the model's own
/// `tokenizer.json`. `None` when the env var is unset — the caller then falls back
/// to synthetic ids, so Stage A1's published numbers do not move silently.
///
/// Worth using before trusting a number for a deploy decision: quantization damage
/// concentrates on the activations *real text* produces, and a recoverable share
/// measured on uniform-random token ids need not transfer. Train and held-out come
/// from **opposite halves of the file** — this repo already retracted a
/// "budget = -13.6%" result that came from a calib corpus and a KLD reference being
/// the same file read from offset 0.
fn corpus_batches(
    model_dir: &Path,
    vocab: usize,
) -> Result<Option<(Vec<Vec<u32>>, Vec<Vec<u32>>)>, Box<dyn std::error::Error>> {
    // Path to a plain-text corpus to train and evaluate on; unset means synthetic ids.
    let Ok(path) = std::env::var("HIPFIRE_QAT_CORPUS") else {
        return Ok(None);
    };
    let tok = Tokenizer::from_tokenizer_json(&model_dir.join("tokenizer.json"))?
        .ok_or("HIPFIRE_QAT_CORPUS is set but the model dir holds no tokenizer.json")?;
    let raw = std::fs::read_to_string(&path)?;
    // Encode only the two slices we need. Tokenizing a multi-MB corpus to pull ~128
    // tokens is pure waste, and the calib corpora here are 5-20 MB.
    const SLICE: usize = 64 * 1024;
    let bound = |mut i: usize| {
        i = i.min(raw.len());
        while !raw.is_char_boundary(i) {
            i += 1;
        }
        i
    };
    let mid = bound(raw.len() / 2);
    let encode = |t: &str| -> Vec<u32> {
        tok.encode(t)
            .into_iter()
            .filter(|&i| (i as usize) < vocab)
            .collect()
    };
    let take = |ids: &[u32], n: usize, what: &str| -> Result<Vec<Vec<u32>>, String> {
        if ids.len() < n * SEQ {
            return Err(format!(
                "{path}: {what} slice yields {} tokens, need {}",
                ids.len(),
                n * SEQ
            ));
        }
        Ok((0..n)
            .map(|s| ids[s * SEQ..(s + 1) * SEQ].to_vec())
            .collect())
    };
    let train_ids = encode(&raw[..bound(SLICE)]);
    let eval_ids = encode(&raw[mid..bound(mid + SLICE)]);
    let n_train = POOL_BATCHES * N_TRAIN;
    println!(
        "batches: REAL text from {path} — train @byte 0, held-out @byte {mid} \
         ({} / {} tok encoded; disjoint halves); pool {n_train} train seq \
         ({POOL_BATCHES} batches cycled), {N_EVAL_CORPUS} held-out seq",
        train_ids.len(),
        eval_ids.len()
    );
    Ok(Some((
        take(&train_ids, n_train, "train")?,
        take(&eval_ids, N_EVAL_CORPUS, "held-out")?,
    )))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DIR.to_string());
    let dir = Path::new(&dir);
    if !dir.exists() {
        return Err(format!("model dir not found: {}", dir.display()).into());
    }
    // `load_llama_fp32` is safetensors-only; say so here rather than let the loader
    // report a bare "no .safetensors files found" from three frames down.
    let has_safetensors = std::fs::read_dir(dir)?
        .flatten()
        .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"));
    if !has_safetensors {
        return Err(format!(
            "no .safetensors in {} — this example needs an fp32-loadable llama \
             snapshot (a .gguf-only or weightless snapshot dir will not do)",
            dir.display()
        )
        .into());
    }
    // Run the student with the KVarN-4bit + CASK KV path; 0 ablates to clean KV.
    let kvnoise = std::env::var("HIPFIRE_QAT_KVNOISE")
        .map(|v| v != "0")
        .unwrap_or(true);
    // Stage A2: activation tier, a16|a8|a4. Resolve it NOW so a typo panics here
    // rather than 5 minutes into the teacher precompute.
    // Activation sim-quant tier for QAT: a16 (default, no-op) | a8 | a4.
    let act_tier = std::env::var("HIPFIRE_QAT_ACT").unwrap_or_else(|_| "a16".into());
    let act_tiers = hipfire_train::a4_quant::act_tiers_from_env();
    // Teacher must run CLEAN — force KV-noise AND activation-quant off during its
    // precompute. Both gates re-read the env on every block, so unsetting them
    // here and restoring after the teacher softmaxes are frozen is sufficient.
    std::env::remove_var("HIPFIRE_KVNOISE");
    std::env::remove_var("HIPFIRE_QAT_ACT");

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}  model: {}", gpu.arch, dir.display());

    let (cfg, w_teacher) = load_llama_fp32(&mut gpu, dir)?;
    let (_, mut w_student) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;
    let tier = Tier::from_env();
    println!("quantizing student to {}...", tier.describe());
    quantize_linears(&mut gpu, &mut w_student, tier)?;

    let teacher = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_teacher, SEQ, RANK, ALPHA)?;
    let student = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_student, SEQ, RANK, ALPHA)?;

    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let mkbatch = |n: usize, salt: usize| -> Vec<Vec<u32>> {
        (0..n)
            .map(|s| {
                (0..SEQ)
                    .map(|t| (((t + 1) * 2654435761 + (s + salt) * 40503) % vocab) as u32)
                    .collect()
            })
            .collect()
    };
    let (train_pool, eval_batch) = match corpus_batches(dir, vocab)? {
        Some(b) => b,
        None => {
            println!(
                "batches: SYNTHETIC random token ids — set HIPFIRE_QAT_CORPUS=<text file> \
                 for real text before trusting these for a deploy decision"
            );
            // Disjoint salt ⇒ held-out tokens.
            (mkbatch(N_TRAIN, 0), mkbatch(N_EVAL, 1000))
        }
    };

    // Frozen CLEAN teacher distributions (KV-noise off) for both batches.
    let teacher_dist =
        |gpu: &mut Gpu, batch: &[Vec<u32>]| -> Result<Vec<GpuTensor>, Box<dyn std::error::Error>> {
            let mut out = Vec::with_capacity(batch.len());
            for toks in batch {
                let at = model_forward(gpu, &teacher, toks, &pos)?;
                let p = gpu.zeros(&[SEQ * vocab], DType::F32)?;
                softmax_forward(gpu, &at.logits, &p, SEQ, vocab)?;
                out.push(p);
            }
            Ok(out)
        };
    let teacher_p_pool = teacher_dist(&mut gpu, &train_pool)?;
    let teacher_p_eval = teacher_dist(&mut gpu, &eval_batch)?;

    // Student runs the requested weight tier + activation tier + (optionally)
    // KVarN. Whatever we train with, we eval with.
    if act_tiers.is_noop() {
        println!(
            "student activations: A16 (unquantized; set HIPFIRE_QAT_ACT=a8|a4, or a \
             per-site policy like a4,act=a8,ctx=a8)"
        );
    } else {
        std::env::set_var("HIPFIRE_QAT_ACT", &act_tier);
        println!(
            "student activations: {} (per-group symmetric, GROUP=256, absmax) — forward-only STE",
            act_tiers.label()
        );
    }
    if kvnoise {
        std::env::set_var("HIPFIRE_KVNOISE", "1");
        let bits = std::env::var("HIPFIRE_KVNOISE_BITS").unwrap_or_else(|_| "4".into());
        let hot = std::env::var("HIPFIRE_KVNOISE_HOT").unwrap_or_else(|_| "4".into());
        let fold = std::env::var("HIPFIRE_KVNOISE_FOLD").unwrap_or_else(|_| "4".into());
        println!(
            "student: {} {} weights + KVarN-{bits}bit + CASK (hot={hot}, fold={fold}) on K&V",
            tier.describe(),
            tier.width()
        );
    } else {
        println!(
            "student: {} weights, KV clean (ablation: HIPFIRE_QAT_KVNOISE=0)",
            tier.describe()
        );
    }

    // Peak learning rate for the warmup+cosine schedule; default 1e-3 is too hot.
    let peak_lr = std::env::var("HIPFIRE_QAT_LR")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(LR);
    // Cheap check that the schedule is the shape we think it is, before 20 minutes
    // of GPU says otherwise: warmup reaches the peak, cosine lands on zero.
    assert!(
        (lr_at(WARMUP - 1, LR) - LR).abs() < 1e-9 && lr_at(STEPS, LR) < LR * 1e-6,
        "lr schedule broken: warmup end {} peak {LR}, final {}",
        lr_at(WARMUP - 1, LR),
        lr_at(STEPS, LR)
    );
    let sizes = student.recovery_param_sizes();
    let mut opt = AdamW::new(&mut gpu, &sizes, peak_lr, 0.9, 0.999, 1e-8, 0.0)?;
    println!(
        "recovery-FT: {} trainable tensors (LoRA q/v + RMSNorm); base {} frozen; \
         LR {peak_lr:.1e} peak, {WARMUP}-step warmup then cosine\n",
        sizes.len(),
        tier.width()
    );

    let batches = train_pool.len() / N_TRAIN;
    println!(
        "train: {} seq in {batches} batch(es) of {N_TRAIN}, cycled over {STEPS} steps; \
         held-out {} seq",
        train_pool.len(),
        eval_batch.len()
    );

    // Held-out gap BEFORE recovery (LoRA=0): the raw W_tier(+KVarN) deploy loss.
    let eval_before = eval_kl(&mut gpu, &student, &eval_batch, &teacher_p_eval, &pos)?;
    // In-sample reference is pinned to pool slot 0 so before/after compare the SAME
    // tokens — with a cycled pool the last step's batch is a different one.
    let train_start = eval_kl(
        &mut gpu,
        &student,
        &train_pool[..N_TRAIN],
        &teacher_p_pool[..N_TRAIN],
        &pos,
    )?;

    // Track the BEST held-out point, not just the last one. With a fixed step budget
    // the final iterate can be well past the optimum -- flat-LR W8 ended 3000x worse
    // than it started -- and "what light QAT can reach" is the honest question. This
    // is early stopping measured, not applied: nothing is rolled back.
    let mut best = (eval_before, 0usize);
    for step in 0..=STEPS {
        opt.set_lr(lr_at(step, peak_lr));
        let base = (step % batches) * N_TRAIN;
        let mut total = 0.0f32;
        for k in 0..N_TRAIN {
            let acts = model_forward(&mut gpu, &student, &train_pool[base + k], &pos)?;
            let (kl, grads, d_final) =
                model_distill_backward(&mut gpu, &student, &acts, &teacher_p_pool[base + k])?;
            total += kl;
            if step < STEPS {
                let params = student.recovery_params();
                let gflat = flatten_recovery_grads(&grads, &d_final);
                opt.step(&mut gpu, &params, &gflat)?;
            }
        }
        if step % 20 == 0 || step == STEPS {
            let held = eval_kl(&mut gpu, &student, &eval_batch, &teacher_p_eval, &pos)?;
            if held < best.0 {
                best = (held, step);
            }
            println!(
                "step {step:4}: batch KL = {:.4}  held-out {held:.4}  (slot {}, lr {:.2e})",
                total / (N_TRAIN * SEQ) as f32,
                step % batches,
                lr_at(step, peak_lr)
            );
        }
    }

    // Held-out gap AFTER recovery, and in-sample on the same slot-0 tokens.
    let eval_after = eval_kl(&mut gpu, &student, &eval_batch, &teacher_p_eval, &pos)?;
    let train_last = eval_kl(
        &mut gpu,
        &student,
        &train_pool[..N_TRAIN],
        &teacher_p_pool[..N_TRAIN],
        &pos,
    )?;

    let pct = |a: f32, b: f32| if a > 1e-6 { 100.0 * (a - b) / a } else { 0.0 };
    println!(
        "\n  ── {} {}{} recovery-FT ──",
        tier.width(),
        act_tiers.label(),
        if kvnoise { "+KVarN" } else { "" }
    );
    println!(
        "  in-sample KL: {train_start:.4} → {train_last:.4}  ({:.1}% recovered)",
        pct(train_start, train_last)
    );
    println!(
        "  HELD-OUT  KL: {eval_before:.4} → {eval_after:.4}  ({:.1}% recovered)",
        pct(eval_before, eval_after)
    );
    println!(
        "  BEST held-out: {:.4} at step {}  ({:.1}% recovered)",
        best.0,
        best.1,
        pct(eval_before, best.0)
    );
    println!("\n  (base {} weights frozen; only LoRA(q/v)+norm trained — measures the LIGHT-QAT\n   recoverable share of the {}{} deploy loss. Held-out is the honest number.)",
        tier.width(),
        tier.width(),
        if kvnoise { "+KVarN" } else { "" });
    Ok(())
}
