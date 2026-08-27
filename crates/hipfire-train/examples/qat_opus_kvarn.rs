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

use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_train::loader::{load_llama_fp32, LlamaWeightsF32};
use hipfire_train::model::{
    flatten_recovery_grads, model_distill_backward, model_forward, LlamaModel,
};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::optim::AdamW;
use hipfire_train::oqplus_quant::{oq3_simquant, oq8_simquant, oqplus_simquant};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const N_TRAIN: usize = 4;
const N_EVAL: usize = 4; // held-out sequences (disjoint tokens)
const RANK: usize = 16;
const ALPHA: f32 = 32.0;
const LR: f32 = 1e-3;
const STEPS: usize = 120;

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
            other => panic!("HIPFIRE_QAT_TIER: unknown tier {other:?} (want oq3|oq4|oq8)"),
        }
    }

    fn simquant(self, w: &[f32]) -> Vec<f32> {
        match self {
            Tier::Oq3 => oq3_simquant(w),
            Tier::Oq4 => oqplus_simquant(w),
            Tier::Oq8 => oq8_simquant(w),
        }
    }

    /// Short deploy label, e.g. "W4".
    fn width(self) -> &'static str {
        match self {
            Tier::Oq3 => "W3",
            Tier::Oq4 => "W4",
            Tier::Oq8 => "W8",
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Tier::Oq3 => "Oq3 (W3, sym-int3 + FWHT-256)",
            Tier::Oq4 => "OQ+ (W4, sym-int4 + FWHT-256, clip-searched)",
            Tier::Oq8 => "Oq8 (W8, sym-int8 + FWHT-256)",
        }
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DIR.to_string());
    let dir = Path::new(&dir);
    if !dir.exists() {
        return Err(format!("model dir not found: {}", dir.display()).into());
    }
    let kvnoise = std::env::var("HIPFIRE_QAT_KVNOISE")
        .map(|v| v != "0")
        .unwrap_or(true);
    // Teacher must run CLEAN — force KV-noise off during its precompute.
    std::env::remove_var("HIPFIRE_KVNOISE");

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
    let train_batch = mkbatch(N_TRAIN, 0);
    let eval_batch = mkbatch(N_EVAL, 1000); // disjoint salt ⇒ held-out tokens

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
    let teacher_p_train = teacher_dist(&mut gpu, &train_batch)?;
    let teacher_p_eval = teacher_dist(&mut gpu, &eval_batch)?;

    // Student runs W3 + (optionally) KVarN. Whatever we train with, we eval with.
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

    let sizes = student.recovery_param_sizes();
    let mut opt = AdamW::new(&mut gpu, &sizes, LR, 0.9, 0.999, 1e-8, 0.0)?;
    println!(
        "recovery-FT: {} trainable tensors (LoRA q/v + RMSNorm); base {} frozen\n",
        sizes.len(),
        tier.width()
    );

    // Held-out gap BEFORE recovery (LoRA=0): the raw W3(+KVarN) deploy loss.
    let eval_before = eval_kl(&mut gpu, &student, &eval_batch, &teacher_p_eval, &pos)?;

    let (mut train_start, mut train_last) = (0.0f32, 0.0f32);
    for step in 0..=STEPS {
        let mut total = 0.0f32;
        for (si, toks) in train_batch.iter().enumerate() {
            let acts = model_forward(&mut gpu, &student, toks, &pos)?;
            let (kl, grads, d_final) =
                model_distill_backward(&mut gpu, &student, &acts, &teacher_p_train[si])?;
            total += kl;
            if step < STEPS {
                let params = student.recovery_params();
                let gflat = flatten_recovery_grads(&grads, &d_final);
                opt.step(&mut gpu, &params, &gflat)?;
            }
        }
        train_last = total / (N_TRAIN * SEQ) as f32;
        if step == 0 {
            train_start = train_last;
        }
        if step % 20 == 0 || step == STEPS {
            println!("step {step:4}: train KL = {train_last:.4} nats/tok");
        }
    }

    // Held-out gap AFTER recovery.
    let eval_after = eval_kl(&mut gpu, &student, &eval_batch, &teacher_p_eval, &pos)?;

    let pct = |a: f32, b: f32| if a > 1e-6 { 100.0 * (a - b) / a } else { 0.0 };
    println!(
        "\n  ── {}{} recovery-FT ──",
        tier.width(),
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
    println!("\n  (base {} weights frozen; only LoRA(q/v)+norm trained — measures the LIGHT-QAT\n   recoverable share of the {}{} deploy loss. Held-out is the honest number.)",
        tier.width(),
        tier.width(),
        if kvnoise { "+KVarN" } else { "" });
    Ok(())
}
