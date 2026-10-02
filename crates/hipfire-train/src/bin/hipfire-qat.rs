#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
// Inherited from this file's life as a GPU-probe example; it is now the
// `hipfire-qat` binary but the body is unchanged.

//! Light QAT for qwen3.5: block-local RMSNorm recovery against the **served
//! artifact's own weights**.
//!
//! Both norms in a block feed only linears, so both are recoverable without ever
//! running the (non-differentiable) DeltaNet/attention mixer:
//!
//!   post_attention_layernorm (γ_mlp)
//!     input  = captured `qwen35.premlp.L{i}`   (post-mixer, pre-FFN residual)
//!     target = bf16  down(swiglu(gate(norm(x)), up(norm(x))))
//!     student = the SAME expression with the artifact's quantized weights
//!
//!   input_layernorm (γ_attn)
//!     input  = captured `qwen35.pertoken.L{i-1}` (= this block's input; layer 0
//!              has no predecessor capture and is skipped)
//!     target = bf16  [q|k|v] (full-attn) or in_proj_qkv (linear-attn)
//!     student = the same projections, quantized
//!
//! γ_attn is trained on the PROJECTION outputs rather than the block output,
//! because the mixer between them has no backward here. That is a surrogate: it
//! minimises the error the mixer is fed, not the error it emits. Whether that
//! surrogate helps end-to-end is an empirical question — score the patched
//! artifact and compare.
//!
//! **The student weights are dequantized from the target artifact**, not
//! re-simulated. γ compensates a specific quantization error, so a simulated
//! error of a different format recovers the wrong correction. This is what makes
//! the result shippable rather than a mechanism probe.
//!
//! Capture (bf16 teacher, per-token decode path, files ACCUMULATE rows). Prompt
//! length is no longer constrained — the second-chunk corruption that once
//! forced it under 2048 tokens was the FP16 DeltaNet recurrence, fixed upstream:
//!   rm -f /tmp/residcap/qwen35.*
//!   HIPFIRE_FORWARD_LOWERED=0 HIPFIRE_DUMP_HIDDEN=/tmp/residcap/qwen35 \
//!   HIPFIRE_DUMP_HIDDEN_ALL=1 HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1 HIPFIRE_MAX_GEN=512 \
//!   cargo run -p hipfire-runtime --release --example infer_qwen35 -- \
//!       ~/.hipfire/models/Qwen3.5-0.8B--bf16.hfq --guards off "<text>"
//!
//! Run it in the foreground (the CLI takes the GPU lock for you):
//!   hipfire qat <quantized.hfq> <bf16.hfq> <tuned_norms.json>
//!
//! or hand it to the service, which queues it and hands it the GPU:
//!   hipfire qat --detach <quantized.hfq> <bf16.hfq> <tuned_norms.json>
//!   hipfire jobs watch <id>
//!
//! Then rebuild the artifact with the tuned norms folded in:
//!   hipfire-quantize --input <bf16.hfq> --output <recovered.hfq> \
//!       --format ... --norm-patch <tuned_norms.json>
//!
//! ## What this is worth, measured
//!
//! On Qwen3.5-0.8B-e4--qtip3g (3.78 bpw), against a control built by the same
//! recipe, reproduced across two independent captures:
//!
//!   layers only      kld 0.189131 -> 0.18915   ppl 17.911 -> 17.82   p99 -1.9%
//!   + final norm     kld 0.189131 -> 0.194610  ppl 17.911 -> 17.99
//!   final norm only,
//!   student input    kld 0.189131 -> 0.262497  ppl 17.911 -> 19.48
//!
//! Layer norms: mean KLD flat, perplexity ~0.6% better, p99 KLD ~2% better.
//! Small, reproducible, real.
//!
//! The final norm is a TRAP, and an instructive one. It is the only norm whose
//! error reaches the logits with no mixer in between, so it can be trained on
//! the KL that is actually scored — and it posts by far the best local number
//! (KL -59% against the bf16 hidden state, -68% against the model's own). Both
//! deploy WORSE, the better-measuring one much worse. Fed the teacher's clean
//! hidden state it solves a problem the deployed model does not have; fed the
//! model's own (drifted to cos 0.96 of the teacher by layer 23) it is asked to
//! absorb 24 layers of accumulated error with `dim` parameters, and overfits —
//! in-sample KL -68%, held-out KLD +39%. Hence `HIPFIRE_RECOVER_HEAD` defaults
//! off. Local win does not imply deployed win; always score the artifact.
//!
//! Env: HIPFIRE_CAP_DIR (default /tmp/residcap), HIPFIRE_RECOVER_LR (3e-4),
//!      HIPFIRE_RECOVER_STEPS (200), HIPFIRE_RECOVER_ATTN (1 = also recover
//!      input_layernorm; 0 = MLP norms only), HIPFIRE_RECOVER_LAYERS (cap the
//!      layer sweep, for LR/step probes), HIPFIRE_RECOVER_ROWS (1024),
//!      HIPFIRE_RECOVER_GAMMA (teacher|student — `student` + STEPS=0 measures a
//!      patched artifact's own block-local error), HIPFIRE_CAP_DIR_STUDENT (a
//!      capture from the QUANTIZED artifact; defaults to the teacher's).

use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_train::hfq_patch::{bf16_bits_to_f32, effective_metadata, parse_hfq, HfqEntry};
use hipfire_train::ops::linear::{linear_backward_x, linear_forward};
use hipfire_train::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use hipfire_train::ops::swiglu::{swiglu_backward, swiglu_forward};
use hipfire_train::optim::AdamW;
use std::collections::HashMap;

fn envs(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}
fn envf(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
fn envu(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Decode one tensor of a parsed `.hfq` to plain, un-rotated f32.
///
/// Handles the lossless bf16 recodings (the index reports the STORED code, e.g.
/// `Bf16Huff`=50 with the packed byte length — the runtime rewrites those to
/// logical BF16 at load, this parser does not) and the quantized body formats an
/// artifact actually ships. Anything else is refused by name rather than decoded
/// as something else.
fn read_f32(bytes: &[u8], e: &HfqEntry) -> Result<Vec<f32>, String> {
    let n: usize = e.shape.iter().map(|&d| d as usize).product();
    let data = &bytes[e.data_offset..e.data_offset + e.data_size];
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };
    match e.quant_type {
        2 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        16 => Ok(bf16(data)),
        // Lossless bf16 recodings (Bf16Lut3=49, Bf16Huff=50): expand, then read.
        49 | 50 => {
            let logical = hipfire_runtime::hfq::decode_bf16_packed(e.quant_type, data, n)
                .ok_or_else(|| {
                    format!("{}: bf16 recoding {} decode failed", e.name, e.quant_type)
                })?;
            Ok(bf16(&logical))
        }
        3 => Ok(hipfire_runtime::quant::dequant_q8f16(data, n)),
        // HFQ4G256: 136 B/group — [f32 scale][f32 min][128 nibbles], plain
        // affine (value = min + q*scale), no rotation. The tied head ships in
        // this format on an `-e4` artifact.
        6 => {
            const G: usize = 256;
            const BLK: usize = 136;
            let mut out = vec![0.0f32; n];
            for b in 0..n.div_ceil(G) {
                let off = b * BLK;
                if off + BLK > data.len() {
                    break;
                }
                let rd =
                    |o: usize| f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                let (scale, min_val) = (rd(off), rd(off + 4));
                for i in 0..128 {
                    let byte = data[off + 8 + i];
                    for (half, q) in [(0usize, byte & 0x0f), (1, byte >> 4)] {
                        let idx = b * G + 2 * i + half;
                        if idx < n {
                            out[idx] = min_val + q as f32 * scale;
                        }
                    }
                }
            }
            Ok(out)
        }
        31 => Ok(hipfire_train::qtip_quant::dequant_qtip3g256(data, n)),
        34 => Ok(hipfire_runtime::quant::dequant_oq4g256(data, n)),
        35 => Ok(hipfire_runtime::quant::dequant_oq8g256(data, n)),
        36 => {
            let rows = e.shape[0] as usize;
            let cols = n / rows.max(1);
            Ok(hipfire_runtime::quant::dequant_oqplus_compact(
                data, rows, cols,
            ))
        }
        other => Err(format!(
            "{}: quant_type {other} has no host dequantizer here — the student base \
             must be decodable, not guessed",
            e.name
        )),
    }
}

/// Read a captured residual file `{dir}/qwen35.{tag}.L{i}` as rows x dim f32.
///
/// `max_rows` strides the capture down. Each step round-trips the block output
/// through host memory to form the MSE gradient, so wall-clock is linear in
/// rows; γ has only `dim` free parameters, so a few times `dim` rows is already
/// far more constraint than it can use.
fn read_cap(
    dir: &str,
    tag: &str,
    i: usize,
    dim: usize,
    max_rows: usize,
) -> Result<(Vec<f32>, usize), String> {
    let p = format!("{dir}/qwen35.{tag}.L{i}");
    let raw = std::fs::read(&p).map_err(|e| format!("{p}: {e}"))?;
    if raw.len() % (dim * 4) != 0 {
        return Err(format!("{p}: {} bytes not a multiple of dim*4", raw.len()));
    }
    let all_rows = raw.len() / (dim * 4);
    // A defensive filter, kept after its cause was fixed upstream.
    //
    // Rows past the first prefill chunk used to come back non-finite for
    // full-attention layers (clean to position 2047, then ~every row bad on a
    // 2805-token prefill). That was NOT the dump, as first supposed: it was the
    // FP16 DeltaNet recurrence, which was not chunk-invariant, diverging in the
    // second chunk (origin/master 63deba175, docs/bugs/2026-09-02-fp16-deltanet-
    // recurrence-is-not-chunk-invariant.md). Re-measured on the same prompt
    // after that fix: 0 bad rows out of 2809, every layer, both tags.
    //
    // The filter stays because a capture is cheap to take and expensive to
    // silently train on. Each recovery uses exactly ONE capture, so dropping
    // cannot desynchronise an input from its target.
    let good: Vec<usize> = (0..all_rows)
        .filter(|r| {
            let off = r * dim * 4;
            raw[off..off + dim * 4]
                .chunks_exact(4)
                .all(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite())
        })
        .collect();
    if good.is_empty() {
        return Err(format!("{p}: every row is non-finite"));
    }
    let stride = good.len().div_ceil(max_rows.max(1)).max(1);
    let taken: Vec<usize> = good.iter().copied().step_by(stride).collect();
    let rows = taken.len();
    let mut out = Vec::with_capacity(rows * dim);
    for r in taken {
        let off = r * dim * 4;
        for c in raw[off..off + dim * 4].chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
    }
    if good.len() < all_rows {
        eprintln!(
            "      {tag}.L{i}: dropped {} / {all_rows} non-finite capture rows",
            all_rows - good.len()
        );
    }
    Ok((out, rows))
}

/// Zero a resident tensor in place: `t += -1 * t`. There is no f32 memset on
/// `Gpu` outside the `deltanet` feature, and re-allocating a zeroed tensor each
/// step is a synchronous hipMalloc/hipFree pair.
fn zero(gpu: &mut Gpu, t: &GpuTensor) -> Result<(), Box<dyn std::error::Error>> {
    gpu.scaled_add_inplace_cpu_scalar_f32(t, t, -1.0)?;
    Ok(())
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

struct Artifact {
    bytes: Vec<u8>,
    entries: Vec<HfqEntry>,
    meta: String,
}
impl Artifact {
    fn open(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let (entries, header) = parse_hfq(&bytes).map_err(|e| format!("{path}: parse_hfq: {e}"))?;
        // Config lives in the tail block on modern containers, not the header.
        let meta = effective_metadata(&bytes, &header);
        Ok(Self {
            bytes,
            entries,
            meta,
        })
    }
    fn get(&self, name: &str) -> Option<&HfqEntry> {
        self.entries.iter().find(|e| e.name == name)
    }
    fn read(&self, name: &str) -> Result<Vec<f32>, String> {
        let e = self.get(name).ok_or_else(|| format!("missing {name}"))?;
        read_f32(&self.bytes, e)
    }
}

/// One trainable-γ recovery: student linears vs teacher linears off a shared
/// input, MSE on the concatenated projection outputs, AdamW on γ only.
///
/// `outs` is one `(m, teacher_w, student_w)` per projection, each `[m x dim]`
/// row-major. Returns `(start_mse, final_mse, tuned_gamma)`.
fn recover_gamma(
    gpu: &mut Gpu,
    x_in_h: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
    gamma_h: &[f32],
    gamma_teacher_h: &[f32],
    outs: &[(usize, Vec<f32>, Vec<f32>)],
    lr: f32,
    steps: usize,
) -> Result<(f32, f32, Vec<f32>), Box<dyn std::error::Error>> {
    let x_in = gpu.upload_f32(x_in_h, &[rows * dim])?;
    let gamma = gpu.upload_f32(gamma_h, &[dim])?;
    // See `recover_mlp_gamma`: the teacher's norm is pinned so verification of a
    // patched artifact does not move the target along with the student.
    let gamma_t = gpu.upload_f32(gamma_teacher_h, &[dim])?;
    let mut opt = AdamW::new(gpu, &[dim], lr, 0.9, 0.999, 1e-8, 0.0)?;
    let yn = gpu.zeros(&[rows * dim], DType::F32)?;
    let rinv = gpu.zeros(&[rows], DType::F32)?;
    let d_x_unused = gpu.zeros(&[rows * dim], DType::F32)?;
    let d_yn = gpu.zeros(&[rows * dim], DType::F32)?;
    let d_gamma = gpu.zeros(&[dim], DType::F32)?;

    let mut w_t: Vec<GpuTensor> = Vec::new();
    let mut w_s: Vec<GpuTensor> = Vec::new();
    let mut y_buf: Vec<GpuTensor> = Vec::new();
    for (m, t, s) in outs {
        w_t.push(gpu.upload_f32(t, &[*m, dim])?);
        w_s.push(gpu.upload_f32(s, &[*m, dim])?);
        y_buf.push(gpu.zeros(&[rows * m], DType::F32)?);
    }

    // Teacher targets: computed here from the bf16 weights at the ORIGINAL γ, so
    // they never depend on a residual-difference capture. They stay RESIDENT —
    // the loss gradient is just `pred - target`, so forming it on the GPU keeps
    // the whole step on-device. Round-tripping activations through host memory
    // every step made this ~0.5 s/step and put a full sweep out of reach.
    let mut targets: Vec<GpuTensor> = Vec::new();
    rmsnorm_forward(gpu, &x_in, &gamma_t, &yn, &rinv, rows, dim, eps)?;
    for (i, (m, _, _)) in outs.iter().enumerate() {
        let t = gpu.zeros(&[rows * m], DType::F32)?;
        linear_forward(gpu, &yn, &w_t[i], &t, rows, dim, *m)?;
        targets.push(t);
        let _ = i;
    }

    // One step. `y_buf[i]` holds pred on entry and `pred - target` on exit, so
    // the residual needed for both the loss and the gradient exists once.
    // AdamW normalises by the gradient's second moment, so dropping the
    // constant `2/N` of the MSE derivative changes nothing but the epsilon.
    let residuals = |gpu: &mut Gpu,
                     yn: &GpuTensor,
                     y_buf: &[GpuTensor],
                     w_s: &[GpuTensor],
                     targets: &[GpuTensor]|
     -> Result<(), Box<dyn std::error::Error>> {
        for (i, (m, _, _)) in outs.iter().enumerate() {
            linear_forward(gpu, yn, &w_s[i], &y_buf[i], rows, dim, *m)?;
            gpu.scaled_add_inplace_cpu_scalar_f32(&y_buf[i], &targets[i], -1.0)?;
        }
        Ok(())
    };
    let resid_mse =
        |gpu: &mut Gpu, y_buf: &[GpuTensor]| -> Result<f32, Box<dyn std::error::Error>> {
            let (mut num, mut den) = (0.0f64, 0usize);
            for b in y_buf {
                let v = gpu.download_f32(b)?;
                num += v.iter().map(|x| (x * x) as f64).sum::<f64>();
                den += v.len();
            }
            Ok((num / den.max(1) as f64) as f32)
        };

    rmsnorm_forward(gpu, &x_in, &gamma, &yn, &rinv, rows, dim, eps)?;
    residuals(gpu, &yn, &y_buf, &w_s, &targets)?;
    let start = resid_mse(gpu, &y_buf)?;
    let mut last = start;

    for step in 0..steps {
        rmsnorm_forward(gpu, &x_in, &gamma, &yn, &rinv, rows, dim, eps)?;
        residuals(gpu, &yn, &y_buf, &w_s, &targets)?;
        if step + 1 == steps {
            last = resid_mse(gpu, &y_buf)?;
        }
        // Both accumulate, so both must start at zero — but a fresh
        // `gpu.zeros` per step is a synchronous hipMalloc/hipFree pair and was
        // most of the step time. Allocate once, zero in place (`d += -1·d`).
        zero(gpu, &d_yn)?;
        for (i, (m, _, _)) in outs.iter().enumerate() {
            // accumulate into d_yn across projections
            linear_backward_x(gpu, &y_buf[i], &w_s[i], &d_yn, rows, dim, *m, true)?;
        }
        zero(gpu, &d_gamma)?;
        rmsnorm_backward(
            gpu,
            &d_yn,
            &x_in,
            &gamma,
            &rinv,
            &d_x_unused,
            &d_gamma,
            rows,
            dim,
        )?;
        opt.step(gpu, &[&gamma], &[&d_gamma])?;
    }

    let tuned = gpu.download_f32(&gamma)?;
    for t in w_t.into_iter().chain(w_s).chain(y_buf).chain(targets) {
        gpu.free_tensor(t)?;
    }
    for t in [x_in, gamma, gamma_t, yn, rinv, d_x_unused, d_yn, d_gamma] {
        gpu.free_tensor(t)?;
    }
    Ok((start, last, tuned))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let q_path = args
        .next()
        .ok_or("usage: hipfire-qat <quantized.hfq> <bf16.hfq> <tuned_norms.json>")?;
    let bf_path = args.next().ok_or("missing <bf16.hfq>")?;
    let out_json = args.next().ok_or("missing <tuned_norms.json>")?;

    let cap_dir = envs("HIPFIRE_CAP_DIR", "/tmp/residcap");
    let lr = envf("HIPFIRE_RECOVER_LR", 3e-4);
    let steps = envu("HIPFIRE_RECOVER_STEPS", 200);
    let do_attn = envs("HIPFIRE_RECOVER_ATTN", "1") == "1";
    // Cap the layer sweep so LR/step probes cost seconds, not an hour.
    let max_layers = envu("HIPFIRE_RECOVER_LAYERS", usize::MAX);
    let max_rows = envu("HIPFIRE_RECOVER_ROWS", 1024);
    // Where the INITIAL γ comes from. `teacher` is right for recovery (student
    // and teacher share the norm before any patch). `student` is how you VERIFY
    // a patched artifact: run with 0 steps and the reported start MSE is that
    // artifact's real block-local error, so a fold that worked shows up as a
    // lower start than the unpatched control.
    let gamma_from_student = envs("HIPFIRE_RECOVER_GAMMA", "teacher") == "student";
    // OFF by default: it MEASURES well and DEPLOYS badly. See the module docs.
    let do_head = envs("HIPFIRE_RECOVER_HEAD", "0") == "1";
    // The head GEMM is [rows x dim] x [dim x vocab] with vocab ~248k, so rows
    // costs far more here than in a block. It only has to constrain `dim`
    // parameters.
    let head_rows = envu("HIPFIRE_RECOVER_HEAD_ROWS", 128);
    // Residuals captured from the QUANTIZED model. Recovery against the bf16
    // teacher's hidden states fits a correction for an input the deployed model
    // never sees: by the final norm the quantized residual has drifted to cos
    // 0.96 of the teacher's. Point this at a capture from the artifact being
    // recovered and the student is fed its OWN input while the target stays the
    // teacher's — which is the error deployment actually has. Defaults to the
    // teacher capture, i.e. the old clean-input behaviour.
    let cap_dir_student = envs("HIPFIRE_CAP_DIR_STUDENT", &cap_dir);

    let student = Artifact::open(&q_path)?;
    let teacher = Artifact::open(&bf_path)?;

    let mj: serde_json::Value = serde_json::from_str(&teacher.meta)?;
    let tc = mj
        .get("config")
        .and_then(|c| c.get("text_config"))
        .or_else(|| mj.get("config"))
        .unwrap_or(&mj);
    let getu = |k: &str| -> usize {
        tc.get(k)
            .or_else(|| mj.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize
    };
    let pick = |a: &str, b: &str| if getu(a) > 0 { getu(a) } else { getu(b) };
    let dim = pick("dim", "hidden_size");
    let inter = pick("hidden_dim", "intermediate_size");
    let n_layers = pick("n_layers", "num_hidden_layers");
    let eps = tc
        .get("norm_eps")
        .or_else(|| tc.get("rms_norm_eps"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let model_type = tc
        .get("model_type")
        .or_else(|| mj.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("qwen3_5");
    // qwen3.5 stores RMSNorm weights under the GemmaRMSNorm (1+w) convention:
    // train the EFFECTIVE scale, write back effective-1 so the runtime's own
    // +1 lands on the tuned value.
    let unit_offset = hipfire_train::loader::uses_unit_offset_norm(model_type);
    println!(
        "model_type={model_type} dim={dim} inter={inter} layers={n_layers} eps={eps} \
         unit_offset={unit_offset}  lr={lr} steps={steps} attn={do_attn}"
    );

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}\n", gpu.arch);

    let prefix = |i: usize, leaf: &str| -> String {
        for base in ["model.language_model.layers", "model.layers", "layers"] {
            let cand = format!("{base}.{i}.{leaf}");
            if teacher.get(&cand).is_some() {
                return cand;
            }
        }
        format!("model.language_model.layers.{i}.{leaf}")
    };
    // A non-finite decode poisons the whole layer's loss and looks like a
    // training divergence, so name the tensor and side instead.
    let finite = |v: &[f32], what: &str, name: &str| -> Result<(), String> {
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        if bad > 0 {
            let lo = v
                .iter()
                .cloned()
                .filter(|x| x.is_finite())
                .fold(f32::MAX, f32::min);
            let hi = v
                .iter()
                .cloned()
                .filter(|x| x.is_finite())
                .fold(f32::MIN, f32::max);
            return Err(format!(
                "{what} {name}: {bad}/{} non-finite after decode (finite range {lo:.3e}..{hi:.3e})",
                v.len()
            ));
        }
        Ok(())
    };
    // HIPFIRE_RECOVER_LR_FOLD=1: add the artifact's own rank-r quantization-error
    // sidecars (`<base>.lr_u/.lr_v`, from HIPFIRE_LOWRANK_R) into the student
    // weight before measuring. Nothing serves these on qwen3.5 today
    // (docs/bugs/2026-09-03-lowrank-sidecars-are-ignored-by-every-arch-but-
    // minimax.md), so this measures the CEILING a low-rank residual could reach
    // if it were wired — offline, before writing any serving code for it.
    let lr_fold = envs("HIPFIRE_RECOVER_LR_FOLD", "0") == "1";
    let fold_lr = |name: &str, w: &mut [f32], m: usize, k: usize| -> Result<bool, String> {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        let (un, vn) = (format!("{stem}.lr_u.weight"), format!("{stem}.lr_v.weight"));
        let (Some(ue), Some(ve)) = (student.get(&un), student.get(&vn)) else {
            return Ok(false);
        };
        let (u, v) = (student.read(&un)?, student.read(&vn)?);
        let r = ue.shape[1] as usize;
        if ue.shape[0] as usize != m || ve.shape[0] as usize != r || ve.shape[1] as usize != k {
            return Err(format!(
                "{stem}: lr shapes disagree with [{m},{k}] rank {r}"
            ));
        }
        // W += lr_u [m,r] @ lr_v [r,k], but the factors live in the FWHT-ROTATED
        // frame: the quantizer factorises `E_rot = W_rot - dequant(Q(W_rot))`.
        // `read_f32` hands back the UN-rotated weight, so the correction has to
        // be rotated back before it is added. Adding it raw makes the weight
        // measurably worse (cos 0.9866 -> 0.9860, block MSE up 2.5x) — which is
        // how this basis mismatch was found.
        use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
        use rayon::prelude::*;
        const G: usize = 256;
        if k % G != 0 {
            return Err(format!("{stem}: k={k} is not a multiple of {G}"));
        }
        let s1 = gen_fwht_signs(42, G);
        let s2 = gen_fwht_signs(1042, G);
        w.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
            let mut corr = vec![0.0f32; k];
            for t in 0..r {
                let a = u[i * r + t];
                if a == 0.0 {
                    continue;
                }
                for (o, &b) in corr.iter_mut().zip(&v[t * k..(t + 1) * k]) {
                    *o += a * b;
                }
            }
            for g in corr.chunks_mut(G) {
                cpu_fwht_256(g, &s2, &s1);
            }
            for (o, c) in row.iter_mut().zip(&corr) {
                *o += *c;
            }
        });
        Ok(true)
    };
    let load_pair = |name: &str| -> Result<(usize, Vec<f32>, Vec<f32>), String> {
        let e = teacher.get(name).ok_or_else(|| format!("missing {name}"))?;
        let m = e.shape[0] as usize;
        let (t, mut st) = (teacher.read(name)?, student.read(name)?);
        if lr_fold {
            let k = st.len() / m.max(1);
            fold_lr(name, &mut st, m, k)?;
        }
        finite(&t, "teacher", name)?;
        finite(&st, "student", name)?;
        Ok((m, t, st))
    };

    let mut tuned: HashMap<String, Vec<f32>> = HashMap::new();
    let (mut s_mlp, mut f_mlp, mut n_mlp) = (0.0f64, 0.0f64, 0usize);
    let (mut s_attn, mut f_attn, mut n_attn) = (0.0f64, 0.0f64, 0usize);

    for i in 0..n_layers.min(max_layers) {
        // HIPFIRE_RECOVER_DUMP_NORMS=1: compare the STORED norm in the student
        // artifact against the bf16 teacher's. Norms pass through unquantized, so
        // an untouched build must match the teacher exactly; anything else means
        // the build is rewriting them and the recovery started from the wrong γ.
        if std::env::var("HIPFIRE_RECOVER_DUMP_NORMS").as_deref() == Ok("1") {
            for leaf in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                let n = prefix(i, leaf);
                if let (Ok(t), Ok(st)) = (teacher.read(&n), student.read(&n)) {
                    let d = t
                        .iter()
                        .zip(&st)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
                    println!(
                        "N L{i:2} {leaf:<32} |teacher-student|max={d:.4e}  \
                         mean_t={:.5} mean_s={:.5}",
                        mean(&t),
                        mean(&st)
                    );
                }
            }
        }

        // ---- post_attention_layernorm: the dense FFN ----
        let gate_n = prefix(i, "mlp.gate_proj.weight");
        if teacher.get(&gate_n).is_some() {
            let norm_n = prefix(i, "post_attention_layernorm.weight");
            let (xin, rows) = read_cap(&cap_dir, "premlp", i, dim, max_rows)?;
            finite(&xin, "capture", &format!("premlp.L{i}"))?;
            let mut g_teacher = teacher.read(&norm_n)?;
            let mut g0 = if gamma_from_student {
                student.read(&norm_n)?
            } else {
                g_teacher.clone()
            };
            if unit_offset {
                for v in g0.iter_mut() {
                    *v += 1.0;
                }
                for v in g_teacher.iter_mut() {
                    *v += 1.0;
                }
            }
            let (_, wg_t, wg_s) = load_pair(&gate_n)?;
            let (_, wu_t, wu_s) = load_pair(&prefix(i, "mlp.up_proj.weight"))?;
            let (_, wd_t, wd_s) = load_pair(&prefix(i, "mlp.down_proj.weight"))?;
            // Weight-space fidelity of the dequantized student — if the decoder
            // were wrong (codebook, bit layout, missing inverse rotation) this
            // is ~0 and everything downstream is meaningless.
            let wcos = cos(&wg_t, &wg_s);
            let (start, last, tv) = recover_mlp_gamma(
                &mut gpu,
                &xin,
                rows,
                dim,
                inter,
                eps,
                &g0,
                &g_teacher,
                (&wg_s, &wu_s, &wd_s),
                (&wg_t, &wu_t, &wd_t),
                lr,
                steps,
            )?;
            let rec = 100.0 * (start - last) / start.max(1e-12);
            println!(
                "L{i:2} mlp : MSE {start:.3e} -> {last:.3e} ({rec:5.1}%)  w_cos={wcos:.4} rows={rows}"
            );
            tuned.insert(norm_n, store_back(tv, unit_offset));
            s_mlp += start as f64;
            f_mlp += last as f64;
            n_mlp += 1;
        }

        // ---- input_layernorm: the attention/LA input projections ----
        // This block's input is the PREVIOUS block's output; layer 0 has no
        // predecessor capture, so it is skipped rather than approximated.
        if !do_attn || i == 0 {
            continue;
        }
        let projs: Vec<String> = if teacher.get(&prefix(i, "self_attn.q_proj.weight")).is_some() {
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                .iter()
                .map(|p| prefix(i, &format!("{p}.weight")))
                .collect()
        } else if teacher
            .get(&prefix(i, "linear_attn.in_proj_qkv.weight"))
            .is_some()
        {
            // in_proj_z/a/b feed sigmoid/softplus gates whose raw-projection MSE
            // is on a different scale; the qkv projection is the dominant term
            // and the one a shared input norm can actually trade against.
            vec![prefix(i, "linear_attn.in_proj_qkv.weight")]
        } else {
            continue;
        };
        let norm_n = prefix(i, "input_layernorm.weight");
        let (xin, rows) = read_cap(&cap_dir, "pertoken", i - 1, dim, max_rows)?;
        let mut g_teacher = teacher.read(&norm_n)?;
        let mut g0 = if gamma_from_student {
            student.read(&norm_n)?
        } else {
            g_teacher.clone()
        };
        if unit_offset {
            for v in g0.iter_mut() {
                *v += 1.0;
            }
            for v in g_teacher.iter_mut() {
                *v += 1.0;
            }
        }
        let outs: Vec<(usize, Vec<f32>, Vec<f32>)> = projs
            .iter()
            .map(|n| load_pair(n))
            .collect::<Result<_, _>>()?;
        let (start, last, tv) = recover_gamma(
            &mut gpu, &xin, rows, dim, eps, &g0, &g_teacher, &outs, lr, steps,
        )?;
        let rec = 100.0 * (start - last) / start.max(1e-12);
        println!("L{i:2} attn: MSE {start:.3e} -> {last:.3e} ({rec:5.1}%)  rows={rows}");
        tuned.insert(norm_n, store_back(tv, unit_offset));
        s_attn += start as f64;
        f_attn += last as f64;
        n_attn += 1;
    }

    // ---- final norm, against the bf16 LOGITS ----
    //
    // Every other norm here is recovered against a block-local surrogate, and
    // the measured cost of that is real: block MSE improves 1-10% and mean KLD
    // does not move. This one is different. `model.norm` feeds the lm_head and
    // nothing else, so its error reaches the logits with no mixer in between —
    // and the loss can be the KL that is actually being scored, not a proxy for
    // it. Its input is the LAST block's `pertoken` capture, already on disk.
    //
    // On a tied model the head is the embedding table, which on an `-e4`
    // artifact is 4-bit: the single largest unshared quant error in the model,
    // and the one sitting closest to the output.
    if do_head {
        let norm_n = ["model.language_model.norm.weight", "model.norm.weight"]
            .iter()
            .find(|n| teacher.get(n).is_some())
            .map(|n| n.to_string());
        let head_n = [
            "model.language_model.lm_head.weight",
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ]
        .iter()
        .find(|n| teacher.get(n).is_some() && student.get(n).is_some())
        .map(|n| n.to_string());
        match (norm_n, head_n) {
            (Some(norm_n), Some(head_n)) => {
                let vocab = teacher.get(&head_n).unwrap().shape[0] as usize;
                let (xin_t, rows) = read_cap(&cap_dir, "pertoken", n_layers - 1, dim, head_rows)?;
                let (xin_s, rows_s) =
                    read_cap(&cap_dir_student, "pertoken", n_layers - 1, dim, head_rows)?;
                if rows != rows_s {
                    return Err(format!(
                        "teacher/student captures disagree on row count ({rows} vs {rows_s}); \
                         they must come from the same prompt"
                    )
                    .into());
                }
                finite(&xin_t, "capture", &format!("pertoken.L{}", n_layers - 1))?;
                finite(
                    &xin_s,
                    "student capture",
                    &format!("pertoken.L{}", n_layers - 1),
                )?;
                let mut g_teacher = teacher.read(&norm_n)?;
                let mut g0 = if gamma_from_student {
                    student.read(&norm_n)?
                } else {
                    g_teacher.clone()
                };
                // qwen3.5's FINAL norm goes through the same `load_norm_weight`
                // as the per-layer ones (`load_norm_weight_raw` is dead code),
                // so it carries the unit offset too.
                if unit_offset {
                    for v in g0.iter_mut() {
                        *v += 1.0;
                    }
                    for v in g_teacher.iter_mut() {
                        *v += 1.0;
                    }
                }
                let (w_t, w_s) = (teacher.read(&head_n)?, student.read(&head_n)?);
                finite(&w_t, "teacher", &head_n)?;
                finite(&w_s, "student", &head_n)?;
                println!(
                    "head: {head_n} vocab={vocab} rows={rows} w_cos={:.4}",
                    cos(&w_t, &w_s)
                );
                let (start, last, tv) = recover_head_gamma(
                    &mut gpu, &xin_s, &xin_t, rows, dim, vocab, eps, &g0, &g_teacher, &w_s, &w_t,
                    lr, steps,
                )?;
                let rec = 100.0 * (start - last) / start.max(1e-12);
                println!("final norm: KL {start:.6e} -> {last:.6e} ({rec:5.1}%)");
                tuned.insert(norm_n, store_back(tv, unit_offset));
            }
            _ => eprintln!("head recovery skipped: final norm or head tensor not found"),
        }
    }

    let pct = |a: f64, b: f64| 100.0 * (a - b) / a.max(1e-12);
    println!(
        "\n{n_mlp} mlp norms:  sum MSE {s_mlp:.4e} -> {f_mlp:.4e} ({:.1}% block-local)",
        pct(s_mlp, f_mlp)
    );
    if n_attn > 0 {
        println!(
            "{n_attn} attn norms: sum MSE {s_attn:.4e} -> {f_attn:.4e} ({:.1}% block-local)",
            pct(s_attn, f_attn)
        );
    }
    std::fs::write(&out_json, serde_json::to_string(&tuned)?)?;
    println!("wrote {} tuned norms -> {out_json}", tuned.len());
    Ok(())
}

/// Undo the unit offset before storing, so the runtime's own `1 + w` reproduces
/// the trained effective scale.
fn store_back(mut v: Vec<f32>, unit_offset: bool) -> Vec<f32> {
    if unit_offset {
        for x in v.iter_mut() {
            *x -= 1.0;
        }
    }
    v
}

/// MLP variant of [`recover_gamma`]: the FFN is not a bag of parallel linears,
/// so it gets its own forward/backward (norm -> gate/up -> swiglu -> down).
fn recover_mlp_gamma(
    gpu: &mut Gpu,
    x_in_h: &[f32],
    rows: usize,
    dim: usize,
    inter: usize,
    eps: f32,
    gamma_h: &[f32],
    gamma_teacher_h: &[f32],
    student: (&[f32], &[f32], &[f32]),
    teacher: (&[f32], &[f32], &[f32]),
    lr: f32,
    steps: usize,
) -> Result<(f32, f32, Vec<f32>), Box<dyn std::error::Error>> {
    let x_in = gpu.upload_f32(x_in_h, &[rows * dim])?;
    let gamma = gpu.upload_f32(gamma_h, &[dim])?;
    // The teacher is the bf16 model with ITS OWN norm, and it does not move.
    // During recovery this equals the initial `gamma`, so the target is the same
    // either way; they diverge only when verifying an already-patched artifact,
    // where reusing one tensor would move the target along with the student and
    // silently compare two different objectives.
    let gamma_t = gpu.upload_f32(gamma_teacher_h, &[dim])?;
    let mut opt = AdamW::new(gpu, &[dim], lr, 0.9, 0.999, 1e-8, 0.0)?;
    let yn = gpu.zeros(&[rows * dim], DType::F32)?;
    let rinv = gpu.zeros(&[rows], DType::F32)?;
    let g = gpu.zeros(&[rows * inter], DType::F32)?;
    let u = gpu.zeros(&[rows * inter], DType::F32)?;
    let act = gpu.zeros(&[rows * inter], DType::F32)?;
    let mlp = gpu.zeros(&[rows * dim], DType::F32)?;
    let d_act = gpu.zeros(&[rows * inter], DType::F32)?;
    let d_g = gpu.zeros(&[rows * inter], DType::F32)?;
    let d_u = gpu.zeros(&[rows * inter], DType::F32)?;
    let d_yn = gpu.zeros(&[rows * dim], DType::F32)?;
    let d_x_unused = gpu.zeros(&[rows * dim], DType::F32)?;
    let d_gamma = gpu.zeros(&[dim], DType::F32)?;

    let up = |gpu: &mut Gpu, w: &[f32], m: usize, k: usize| gpu.upload_f32(w, &[m, k]);
    let (wg_t, wu_t, wd_t) = (
        up(gpu, teacher.0, inter, dim)?,
        up(gpu, teacher.1, inter, dim)?,
        up(gpu, teacher.2, dim, inter)?,
    );
    let (wg_s, wu_s, wd_s) = (
        up(gpu, student.0, inter, dim)?,
        up(gpu, student.1, inter, dim)?,
        up(gpu, student.2, dim, inter)?,
    );

    macro_rules! ffn {
        ($gpu:expr, $g:expr, $wg:expr, $wu:expr, $wd:expr) => {{
            rmsnorm_forward($gpu, &x_in, $g, &yn, &rinv, rows, dim, eps)?;
            linear_forward($gpu, &yn, $wg, &g, rows, dim, inter)?;
            linear_forward($gpu, &yn, $wu, &u, rows, dim, inter)?;
            swiglu_forward($gpu, &g, &u, &act, rows * inter)?;
            linear_forward($gpu, &act, $wd, &mlp, rows, inter, dim)?;
        }};
    }

    // Teacher FFN output stays resident; `mlp` becomes `pred - target` in place,
    // which is both the loss and (up to the constant 2/N that AdamW normalises
    // away) the gradient. Keeps the step on-device.
    let target = gpu.zeros(&[rows * dim], DType::F32)?;
    ffn!(gpu, &gamma_t, &wg_t, &wu_t, &wd_t);
    gpu.copy_d2d(&mlp, &target, rows * dim * 4)?;
    let resid = |gpu: &mut Gpu| -> Result<f32, Box<dyn std::error::Error>> {
        let v = gpu.download_f32(&mlp)?;
        Ok(v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32)
    };

    macro_rules! student_resid {
        ($gpu:expr) => {{
            ffn!($gpu, &gamma, &wg_s, &wu_s, &wd_s);
            $gpu.scaled_add_inplace_cpu_scalar_f32(&mlp, &target, -1.0)?;
        }};
    }
    student_resid!(gpu);
    let start = resid(gpu)?;
    let mut last = start;
    for step in 0..steps {
        student_resid!(gpu);
        if step + 1 == steps {
            last = resid(gpu)?;
        }
        linear_backward_x(gpu, &mlp, &wd_s, &d_act, rows, inter, dim, false)?;
        swiglu_backward(gpu, &d_act, &g, &u, &d_g, &d_u, rows * inter)?;
        linear_backward_x(gpu, &d_g, &wg_s, &d_yn, rows, dim, inter, false)?;
        linear_backward_x(gpu, &d_u, &wu_s, &d_yn, rows, dim, inter, true)?;
        // rmsnorm_backward atomically accumulates into d_gamma, so it must be
        // zeroed each step; a per-step `gpu.zeros` is a synchronous alloc pair.
        zero(gpu, &d_gamma)?;
        rmsnorm_backward(
            gpu,
            &d_yn,
            &x_in,
            &gamma,
            &rinv,
            &d_x_unused,
            &d_gamma,
            rows,
            dim,
        )?;
        opt.step(gpu, &[&gamma], &[&d_gamma])?;
    }
    let tv = gpu.download_f32(&gamma)?;
    let dmax = tv
        .iter()
        .zip(gamma_h)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("      |dgamma|max={dmax:.3e}");
    for t in [
        wg_t, wu_t, wd_t, wg_s, wu_s, wd_s, x_in, gamma, yn, rinv, g, u, act, mlp, d_act, d_g, d_u,
        d_yn, d_x_unused, target, d_gamma, gamma_t,
    ] {
        gpu.free_tensor(t)?;
    }
    Ok((start, last, tv))
}

/// Final-norm recovery against the teacher's LOGITS, minimising the KL that is
/// actually scored rather than a block-local MSE proxy.
///
/// `x_in` is the last block's output (the final norm's input). Forward is
/// `rmsnorm(x, γ) → head → logits`; the teacher runs the same with its own
/// pinned γ and bf16 head, softmaxed once into `teacher_p`. `distill_kl` returns
/// the loss and `d_logits` together, which backprops through the head into γ.
#[allow(clippy::too_many_arguments)]
fn recover_head_gamma(
    gpu: &mut Gpu,
    x_in_student_h: &[f32],
    x_in_teacher_h: &[f32],
    rows: usize,
    dim: usize,
    vocab: usize,
    eps: f32,
    gamma_h: &[f32],
    gamma_teacher_h: &[f32],
    head_student_h: &[f32],
    head_teacher_h: &[f32],
    lr: f32,
    steps: usize,
) -> Result<(f32, f32, Vec<f32>), Box<dyn std::error::Error>> {
    use hipfire_train::ops::distill::distill_kl;
    use hipfire_train::ops::softmax::softmax_forward;

    let x_in = gpu.upload_f32(x_in_student_h, &[rows * dim])?;
    let x_in_t = gpu.upload_f32(x_in_teacher_h, &[rows * dim])?;
    let gamma = gpu.upload_f32(gamma_h, &[dim])?;
    let gamma_t = gpu.upload_f32(gamma_teacher_h, &[dim])?;
    let mut opt = AdamW::new(gpu, &[dim], lr, 0.9, 0.999, 1e-8, 0.0)?;
    let yn = gpu.zeros(&[rows * dim], DType::F32)?;
    let rinv = gpu.zeros(&[rows], DType::F32)?;
    let logits = gpu.zeros(&[rows * vocab], DType::F32)?;
    let teacher_p = gpu.zeros(&[rows * vocab], DType::F32)?;
    let d_logits = gpu.zeros(&[rows * vocab], DType::F32)?;
    let loss = gpu.zeros(&[rows], DType::F32)?;
    let d_yn = gpu.zeros(&[rows * dim], DType::F32)?;
    let d_x_unused = gpu.zeros(&[rows * dim], DType::F32)?;
    let d_gamma = gpu.zeros(&[dim], DType::F32)?;

    let w_t = gpu.upload_f32(head_teacher_h, &[vocab, dim])?;
    // Teacher probabilities: computed once, at the teacher's own γ.
    rmsnorm_forward(gpu, &x_in_t, &gamma_t, &yn, &rinv, rows, dim, eps)?;
    linear_forward(gpu, &yn, &w_t, &logits, rows, dim, vocab)?;
    softmax_forward(gpu, &logits, &teacher_p, rows, vocab)?;
    gpu.free_tensor(w_t)?;

    let w_s = gpu.upload_f32(head_student_h, &[vocab, dim])?;
    let mean_loss = |gpu: &mut Gpu, loss: &GpuTensor| -> Result<f32, Box<dyn std::error::Error>> {
        let v = gpu.download_f32(loss)?;
        Ok(v.iter().sum::<f32>() / v.len() as f32)
    };

    let step_once = |gpu: &mut Gpu| -> Result<f32, Box<dyn std::error::Error>> {
        rmsnorm_forward(gpu, &x_in, &gamma, &yn, &rinv, rows, dim, eps)?;
        linear_forward(gpu, &yn, &w_s, &logits, rows, dim, vocab)?;
        distill_kl(gpu, &logits, &teacher_p, &loss, &d_logits, rows, vocab)?;
        mean_loss(gpu, &loss)
    };

    let start = step_once(gpu)?;
    let mut last = start;
    for step in 0..steps {
        let l = step_once(gpu)?;
        if step + 1 == steps {
            last = l;
        }
        linear_backward_x(gpu, &d_logits, &w_s, &d_yn, rows, dim, vocab, false)?;
        zero(gpu, &d_gamma)?;
        rmsnorm_backward(
            gpu,
            &d_yn,
            &x_in,
            &gamma,
            &rinv,
            &d_x_unused,
            &d_gamma,
            rows,
            dim,
        )?;
        opt.step(gpu, &[&gamma], &[&d_gamma])?;
    }
    if steps > 0 {
        last = step_once(gpu)?;
    }

    let tuned = gpu.download_f32(&gamma)?;
    for t in [
        x_in, x_in_t, gamma, gamma_t, yn, rinv, logits, teacher_p, d_logits, loss, d_yn,
        d_x_unused, d_gamma, w_s,
    ] {
        gpu.free_tensor(t)?;
    }
    Ok((start, last, tuned))
}
