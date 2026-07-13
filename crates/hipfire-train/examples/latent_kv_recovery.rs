//! Latent-KV recovery-FT probe (Tier-1 mechanism test).
//!
//! Question: every static / equivariant rank-r basis on the *frozen* model was
//! rejected on 0.8B/4B/9B (docs/plans/2026-07-11-latent-kv-large-model-
//! confirmation.md). The only remaining lever is per-model adaptation. This asks:
//! can LoRA-adapting the model *recover* the KL gap a FIXED calibrated rank-r
//! latent-KV bottleneck introduces?
//!
//! Student = clean fp32 weights + trainable LoRA(q/v) + RMSNorm, with post-RoPE
//! K/V projected onto a per-(layer,kv-head) rank-r subspace calibrated (top-r
//! eigenvectors of the K/V covariance) on the train batch, projection OFF.
//! Teacher = clean fp32, no projection. KL-distill student→teacher; report the KL
//! gap recovered IN-SAMPLE and HELD-OUT (calib≠eval).
//!
//! Batches are REAL tokenized text (calib≠eval = disjoint documents): pass
//! whitespace-separated token-ID windows, one per line, via
//! HIPFIRE_LATENTKV_TRAIN / HIPFIRE_LATENTKV_EVAL. All windows must share length.
//!
//! Supra-50M is small (head_dim 64, so rank 32 = 2x KV reduction). Per the
//! confirmation study small models show quirks that don't generalize, so this is
//! a mechanism probe, NOT admission evidence.
//!
//! Run:
//!   source ./scripts/rocm-env.sh
//!   hipfire lock acquire "latent-kv-recovery"
//!   HIPFIRE_LATENTKV_TRAIN=train.ids HIPFIRE_LATENTKV_EVAL=eval.ids \
//!     cargo run -p hipfire-train --release --example latent_kv_recovery [model_dir]
//!   hipfire lock release

use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_train::latent_kv::{self, CovAccum};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{
    flatten_recovery_grads, free_model_acts, model_distill_backward, model_forward, LlamaModel,
};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::optim::AdamW;
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Read whitespace-separated token-ID windows (one per line) into a batch.
fn load_ids(path: &str) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let batch: Vec<Vec<u32>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<u32>().unwrap())
                .collect()
        })
        .collect();
    if batch.is_empty() {
        return Err(format!("no token windows in {path}").into());
    }
    let seq = batch[0].len();
    if batch.iter().any(|s| s.len() != seq) {
        return Err(format!("token windows in {path} are not all length {seq}").into());
    }
    Ok(batch)
}

/// Free the owned gradient tensors returned by `model_distill_backward` (no Drop
/// on GpuTensor, so a long training loop OOMs without this).
fn free_grads(
    gpu: &mut Gpu,
    grads: Vec<hipfire_train::block::BlockLoraGrad>,
    d_final: GpuTensor,
) -> Result<(), Box<dyn std::error::Error>> {
    for g in grads {
        gpu.free_tensor(g.daq)?;
        gpu.free_tensor(g.dbq)?;
        gpu.free_tensor(g.dav)?;
        gpu.free_tensor(g.dbv)?;
        gpu.free_tensor(g.dnorm1)?;
        gpu.free_tensor(g.dnorm2)?;
    }
    gpu.free_tensor(d_final)?;
    Ok(())
}

/// Mean KL(teacher‖student) over a batch under the current latent-KV store.
fn eval_kl(
    gpu: &mut Gpu,
    student: &LlamaModel,
    batch: &[Vec<u32>],
    teacher_p: &[GpuTensor],
    pos: &[f32],
    seq: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut total = 0.0f32;
    for (si, toks) in batch.iter().enumerate() {
        let acts = model_forward(gpu, student, toks, pos)?;
        let (kl, grads, d_final) = model_distill_backward(gpu, student, &acts, &teacher_p[si])?;
        total += kl;
        free_model_acts(gpu, acts)?;
        free_grads(gpu, grads, d_final)?;
    }
    Ok(total / (batch.len() * seq) as f32)
}

/// Fit rank-`rank` K/V subspace projectors from a projection-OFF forward pass.
#[allow(clippy::too_many_arguments)]
fn calibrate(
    gpu: &mut Gpu,
    student: &LlamaModel,
    batch: &[Vec<u32>],
    pos: &[f32],
    seq: usize,
    n_layers: usize,
    n_kv: usize,
    head_dim: usize,
    rank: usize,
) -> Result<Vec<latent_kv::LayerProjectors>, Box<dyn std::error::Error>> {
    let mut acc = CovAccum::new(n_layers, n_kv, head_dim);
    for toks in batch {
        let acts = model_forward(gpu, student, toks, pos)?;
        for (l, b) in acts.layer_acts.iter().enumerate() {
            let kh = gpu.download_f32(&b.k_r)?;
            let vh = gpu.download_f32(&b.v)?;
            acc.add_layer(l, &kh, &vh, seq);
        }
        free_model_acts(gpu, acts)?;
    }
    Ok(acc.finish(rank))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DIR.to_string());
    let dir = Path::new(&dir);
    if !dir.exists() {
        return Err(format!("model dir not found: {}", dir.display()).into());
    }
    let latent_rank: usize = std::env::var("HIPFIRE_LATENTKV_RANK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let lr = env_f32("HIPFIRE_LATENTKV_LR", 2e-4);
    let steps = env_usize("HIPFIRE_LATENTKV_STEPS", 150);
    let lora_rank = env_usize("HIPFIRE_LATENTKV_LORA_RANK", 16);
    // Keep alpha/rank constant (baseline 32/16 = 2.0) so a rank sweep isolates
    // capacity, not effective LoRA scaling.
    let alpha = env_f32("HIPFIRE_LATENTKV_ALPHA", 2.0 * lora_rank as f32);
    let train_path = std::env::var("HIPFIRE_LATENTKV_TRAIN")
        .map_err(|_| "set HIPFIRE_LATENTKV_TRAIN to a token-ID windows file")?;
    let eval_path = std::env::var("HIPFIRE_LATENTKV_EVAL")
        .map_err(|_| "set HIPFIRE_LATENTKV_EVAL to a token-ID windows file")?;

    let train_batch = load_ids(&train_path)?;
    let eval_batch = load_ids(&eval_path)?;
    let seq = train_batch[0].len();
    if eval_batch[0].len() != seq {
        return Err("train and eval windows must share length".into());
    }
    let (n_train, n_eval) = (train_batch.len(), eval_batch.len());

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}  model: {}", gpu.arch, dir.display());

    let (cfg, w_teacher) = load_llama_fp32(&mut gpu, dir)?;
    let (_, w_student) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;
    let (n_layers, n_kv, head_dim) = (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim);
    println!(
        "real-text batches: {n_train} train / {n_eval} held-out windows x seq {seq}  (disjoint docs)"
    );
    println!(
        "latent-KV rank {latent_rank} of head_dim {head_dim}  ({:.2}x KV reduction), {n_layers} layers x {n_kv} kv-heads",
        head_dim as f32 / latent_rank as f32
    );

    let teacher = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_teacher, seq, lora_rank, alpha)?;
    let student = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_student, seq, lora_rank, alpha)?;
    println!("LoRA(q/v) rank {lora_rank}, alpha {alpha:.0}");

    let pos: Vec<f32> = (0..seq).map(|t| t as f32).collect();

    // Frozen CLEAN teacher distributions (no projection) for both batches.
    latent_kv::clear_projectors();
    let teacher_dist =
        |gpu: &mut Gpu, batch: &[Vec<u32>]| -> Result<Vec<GpuTensor>, Box<dyn std::error::Error>> {
            let mut out = Vec::with_capacity(batch.len());
            for toks in batch {
                let at = model_forward(gpu, &teacher, toks, &pos)?;
                let p = gpu.zeros(&[seq * vocab], DType::F32)?;
                softmax_forward(gpu, &at.logits, &p, seq, vocab)?;
                out.push(p);
                free_model_acts(gpu, at)?;
            }
            Ok(out)
        };
    let teacher_p_train = teacher_dist(&mut gpu, &train_batch)?;
    let teacher_p_eval = teacher_dist(&mut gpu, &eval_batch)?;

    // Calibrate the fixed rank-r K/V subspaces (projection OFF, LoRA=0 ⇒ clean),
    // on the TRAIN batch, then install them. Eval stays held-out.
    let projectors = calibrate(
        &mut gpu,
        &student,
        &train_batch,
        &pos,
        seq,
        n_layers,
        n_kv,
        head_dim,
        latent_rank,
    )?;
    latent_kv::set_projectors(projectors);
    println!("calibrated + installed rank-{latent_rank} latent-KV projectors\n");

    let sizes = student.recovery_param_sizes();
    let mut opt = AdamW::new(&mut gpu, &sizes, lr, 0.9, 0.999, 1e-8, 0.0)?;
    println!(
        "recovery-FT: {} trainable tensors (LoRA q/v + RMSNorm); base frozen; lr {lr:.1e}, {steps} steps\n",
        sizes.len()
    );

    // Held-out gap BEFORE recovery (LoRA=0): the raw latent-KV deploy loss.
    let eval_before = eval_kl(&mut gpu, &student, &eval_batch, &teacher_p_eval, &pos, seq)?;

    let (mut train_start, mut train_last) = (0.0f32, 0.0f32);
    for step in 0..=steps {
        let mut total = 0.0f32;
        for (si, toks) in train_batch.iter().enumerate() {
            let acts = model_forward(&mut gpu, &student, toks, &pos)?;
            let (kl, grads, d_final) =
                model_distill_backward(&mut gpu, &student, &acts, &teacher_p_train[si])?;
            total += kl;
            if step < steps {
                let params = student.recovery_params();
                let gflat = flatten_recovery_grads(&grads, &d_final);
                opt.step(&mut gpu, &params, &gflat)?;
            }
            free_model_acts(&mut gpu, acts)?;
            free_grads(&mut gpu, grads, d_final)?;
        }
        train_last = total / (n_train * seq) as f32;
        if step == 0 {
            train_start = train_last;
        }
        if step % 20 == 0 || step == steps {
            println!("step {step:4}: train KL = {train_last:.4} nats/tok");
        }
    }

    let eval_after = eval_kl(&mut gpu, &student, &eval_batch, &teacher_p_eval, &pos, seq)?;

    let pct = |a: f32, b: f32| if a > 1e-6 { 100.0 * (a - b) / a } else { 0.0 };
    println!("\n  ── latent-KV rank-{latent_rank} recovery-FT (real text) ──");
    println!(
        "  in-sample KL: {train_start:.4} → {train_last:.4}  ({:.1}% recovered)",
        pct(train_start, train_last)
    );
    println!(
        "  HELD-OUT  KL: {eval_before:.4} → {eval_after:.4}  ({:.1}% recovered)",
        pct(eval_before, eval_after)
    );
    println!("\n  (base weights frozen; only LoRA(q/v)+norm trained. Held-out = disjoint docs, the\n   honest number. Supra-50M is a mechanism probe — small-model quirks apply, not admission.)");
    Ok(())
}
