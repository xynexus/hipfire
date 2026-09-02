#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

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
//! Capture (bf16 teacher, per-token decode path, files ACCUMULATE rows):
//!   rm -f /tmp/residcap/qwen35.*
//!   HIPFIRE_FORWARD_LOWERED=0 HIPFIRE_DUMP_HIDDEN=/tmp/residcap/qwen35 \
//!   HIPFIRE_DUMP_HIDDEN_ALL=1 HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1 HIPFIRE_MAX_GEN=512 \
//!   cargo run -p hipfire-runtime --release --example infer_qwen35 -- \
//!       ~/.hipfire/models/Qwen3.5-0.8B--bf16.hfq --guards off "<text>"
//!
//! Run:
//!   hipfire lock acquire "qwen35-norm-recovery" --watch-pid $$
//!   cargo run -p hipfire-train --release --example qwen35_norm_recovery -- \
//!       <quantized.hfq> <bf16.hfq> <tuned_norms.json>
//!   hipfire lock release
//!
//! Then rebuild the artifact with the tuned norms folded in:
//!   hipfire-quantize --input <bf16.hfq> --output <recovered.hfq> \
//!       --format ... --norm-patch <tuned_norms.json>
//!
//! Env: HIPFIRE_CAP_DIR (default /tmp/residcap), HIPFIRE_RECOVER_LR (3e-4),
//!      HIPFIRE_RECOVER_STEPS (200), HIPFIRE_RECOVER_ATTN (1 = also recover
//!      input_layernorm; 0 = MLP norms only), HIPFIRE_RECOVER_LAYERS (cap the
//!      layer sweep, for LR/step probes), HIPFIRE_RECOVER_ROWS (1024).

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
    // Rows past the first prefill chunk come back non-finite for full-attention
    // layers (measured: clean to position 2047, then ~every row bad, on both
    // tags, on a 2805-token prefill). Whatever the dump is reading there, it is
    // not the residual stream — the same run generated coherent text. Drop those
    // rows rather than poison a whole layer's loss with them. Each recovery uses
    // exactly ONE capture, so dropping cannot desynchronise an input from its
    // target.
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
    outs: &[(usize, Vec<f32>, Vec<f32>)],
    lr: f32,
    steps: usize,
) -> Result<(f32, f32, Vec<f32>), Box<dyn std::error::Error>> {
    let x_in = gpu.upload_f32(x_in_h, &[rows * dim])?;
    let gamma = gpu.upload_f32(gamma_h, &[dim])?;
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
    rmsnorm_forward(gpu, &x_in, &gamma, &yn, &rinv, rows, dim, eps)?;
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
    for t in [x_in, gamma, yn, rinv, d_x_unused, d_yn, d_gamma] {
        gpu.free_tensor(t)?;
    }
    Ok((start, last, tuned))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let q_path = args
        .next()
        .ok_or("usage: qwen35_norm_recovery <quantized.hfq> <bf16.hfq> <tuned_norms.json>")?;
    let bf_path = args.next().ok_or("missing <bf16.hfq>")?;
    let out_json = args.next().ok_or("missing <tuned_norms.json>")?;

    let cap_dir = envs("HIPFIRE_CAP_DIR", "/tmp/residcap");
    let lr = envf("HIPFIRE_RECOVER_LR", 3e-4);
    let steps = envu("HIPFIRE_RECOVER_STEPS", 200);
    let do_attn = envs("HIPFIRE_RECOVER_ATTN", "1") == "1";
    // Cap the layer sweep so LR/step probes cost seconds, not an hour.
    let max_layers = envu("HIPFIRE_RECOVER_LAYERS", usize::MAX);
    let max_rows = envu("HIPFIRE_RECOVER_ROWS", 1024);

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
    let load_pair = |name: &str| -> Result<(usize, Vec<f32>, Vec<f32>), String> {
        let e = teacher.get(name).ok_or_else(|| format!("missing {name}"))?;
        let m = e.shape[0] as usize;
        let (t, st) = (teacher.read(name)?, student.read(name)?);
        finite(&t, "teacher", name)?;
        finite(&st, "student", name)?;
        Ok((m, t, st))
    };

    let mut tuned: HashMap<String, Vec<f32>> = HashMap::new();
    let (mut s_mlp, mut f_mlp, mut n_mlp) = (0.0f64, 0.0f64, 0usize);
    let (mut s_attn, mut f_attn, mut n_attn) = (0.0f64, 0.0f64, 0usize);

    for i in 0..n_layers.min(max_layers) {
        // ---- post_attention_layernorm: the dense FFN ----
        let gate_n = prefix(i, "mlp.gate_proj.weight");
        if teacher.get(&gate_n).is_some() {
            let norm_n = prefix(i, "post_attention_layernorm.weight");
            let (xin, rows) = read_cap(&cap_dir, "premlp", i, dim, max_rows)?;
            finite(&xin, "capture", &format!("premlp.L{i}"))?;
            let mut g0 = teacher.read(&norm_n)?;
            if unit_offset {
                for v in g0.iter_mut() {
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
        let mut g0 = teacher.read(&norm_n)?;
        if unit_offset {
            for v in g0.iter_mut() {
                *v += 1.0;
            }
        }
        let outs: Vec<(usize, Vec<f32>, Vec<f32>)> = projs
            .iter()
            .map(|n| load_pair(n))
            .collect::<Result<_, _>>()?;
        let (start, last, tv) =
            recover_gamma(&mut gpu, &xin, rows, dim, eps, &g0, &outs, lr, steps)?;
        let rec = 100.0 * (start - last) / start.max(1e-12);
        println!("L{i:2} attn: MSE {start:.3e} -> {last:.3e} ({rec:5.1}%)  rows={rows}");
        tuned.insert(norm_n, store_back(tv, unit_offset));
        s_attn += start as f64;
        f_attn += last as f64;
        n_attn += 1;
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
    student: (&[f32], &[f32], &[f32]),
    teacher: (&[f32], &[f32], &[f32]),
    lr: f32,
    steps: usize,
) -> Result<(f32, f32, Vec<f32>), Box<dyn std::error::Error>> {
    let x_in = gpu.upload_f32(x_in_h, &[rows * dim])?;
    let gamma = gpu.upload_f32(gamma_h, &[dim])?;
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
        ($gpu:expr, $wg:expr, $wu:expr, $wd:expr) => {{
            rmsnorm_forward($gpu, &x_in, &gamma, &yn, &rinv, rows, dim, eps)?;
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
    ffn!(gpu, &wg_t, &wu_t, &wd_t);
    gpu.copy_d2d(&mlp, &target, rows * dim * 4)?;
    let resid = |gpu: &mut Gpu| -> Result<f32, Box<dyn std::error::Error>> {
        let v = gpu.download_f32(&mlp)?;
        Ok(v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32)
    };

    macro_rules! student_resid {
        ($gpu:expr) => {{
            ffn!($gpu, &wg_s, &wu_s, &wd_s);
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
        d_yn, d_x_unused, target, d_gamma,
    ] {
        gpu.free_tensor(t)?;
    }
    Ok((start, last, tv))
}
