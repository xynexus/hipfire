// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The text trunk on the GPU, composed from the verified blocks.
//!
//! Structurally identical to [`crate::trunk`], which is differenced against the
//! pinned upstream implementation end-to-end. Each block here has its own
//! GPU-vs-CPU parity; this is the composition of them.
//!
//! Two things stay on the host on purpose:
//!
//! * **The n-gram lookup.** The table is 102 GB in the shipped model — 41% of the
//!   parameters — and cannot be resident. Rows are gathered host-side and the
//!   `[embed_dim]` result is uploaded, which is also the shape the on-disk store
//!   ([`crate::ngram_store`]) will serve.
//! * **The embedding row**, for the same reason at smaller scale, and because it is
//!   a row copy either way.

use crate::attn_gpu::{qsa_decode_step, QsaCache, QsaScratch, QsaWeights};
use crate::config::{LayerType, Qwen4ExpConfig};
use crate::gdn::{gdn_decode_step, GdnScratch, GdnState, GdnWeights};
use crate::hc_gpu::{hc_read, hc_write, HcScratch, HcWeights};
use crate::moe_gpu::{moe_forward, MoeScratch, MoeWeights};
use crate::ngram::NgramHasher;
use crate::ple_gpu::{ple_step, PleScratch, PleWeights};

use hipfire_rdna::{DType, Gpu, GpuTensor, HipError, HipResult};

pub enum TokenMixer {
    Gdn(GdnWeights),
    Qsa(QsaWeights),
}

pub struct LayerWeights {
    pub attn_hc: HcWeights,
    pub mlp_hc: HcWeights,
    pub mixer: TokenMixer,
    pub moe: MoeWeights,
    pub ple: Option<PleWeights>,
}

pub struct TrunkWeights {
    pub layers: Vec<LayerWeights>,
    pub mixer: HcWeights,
    pub lm_head: GpuTensor,
}

/// Per-sequence state: one recurrent state per Gated DeltaNet layer, one KV cache
/// per sparse-attention layer, one conv ring per PLE layer.
pub struct TrunkState {
    gdn: Vec<Option<GdnState>>,
    qsa: Vec<Option<QsaCache>>,
    ple: Vec<Option<PleScratch>>,
}

pub struct TrunkScratch {
    hc: HcScratch,
    gdn: GdnScratch,
    qsa: QsaScratch,
    moe: MoeScratch,
    mixed: GpuTensor,
    block_out: GpuTensor,
    wide: GpuTensor,
    ple_out: GpuTensor,
    collapsed: GpuTensor,
    logits: GpuTensor,
}

/// Yields one tensor at a time as f32.
///
/// Deliberately owned rather than borrowed: the shipped model is ~360 GB and its
/// n-gram table alone is 102 GB, so nothing may assume the whole weight set is
/// resident in host memory at once. Each tensor is read, uploaded, and dropped.
pub trait TensorReader {
    fn read(&self, name: &str) -> Result<Vec<f32>, String>;
}

/// A reader error is a LOAD FAILURE, not a crash.
///
/// These helpers used to `panic!` on an unreadable tensor. That is fine in an
/// example and wrong everywhere else: the daemon loads artifacts in-process, so an
/// unsupported quant format in one model would take down a server holding others.
/// The message is already specific — it names the tensor and the format — so it
/// only needed to be returned rather than thrown.
fn read_err(name: &str, e: String) -> HipError {
    HipError::new(0, &format!("qwen4_exp load `{name}`: {e}"))
}

fn up2(
    gpu: &mut Gpu,
    w: &dyn TensorReader,
    name: &str,
    rows: usize,
    cols: usize,
) -> HipResult<GpuTensor> {
    let v = w.read(name).map_err(|e| read_err(name, e))?;
    gpu.upload_f32(&v, &[rows, cols])
}

/// Assemble the stacked expert tensor the MoE block wants.
///
/// Two layouts exist and both are legitimate. A safetensors source stacks the
/// experts (`mlp.experts.gate_up_proj`, `[n_exp, ...]`), which is how the reference
/// holds them; the QUANTIZER splits them per expert
/// (`mlp.experts.<e>.gate_up_proj.weight`) so each can carry its own scales. A
/// loader that knows only the stacked name fails on every artifact it will
/// actually be given.
///
/// Reads one expert at a time, so peak host memory is one expert, not the set.
fn stack_experts(
    gpu: &mut Gpu,
    w: &dyn TensorReader,
    mp: &str,
    which: &str,
    n_exp: usize,
    shape: &[usize],
) -> HipResult<crate::moe_gpu::ExpertStack> {
    // `shape` is [n_exp, rows, cols]; one expert's region is rows*cols f32, which
    // is self-contained, so a per-expert view is a single offset.
    let rows = shape[1];
    let cols = shape[2];
    let wrap = |buf: GpuTensor| crate::moe_gpu::ExpertStack {
        buf,
        dtype: DType::F32,
        rows,
        cols,
        stride: rows * cols,
    };
    if let Ok(v) = w.read(&format!("{mp}.experts.{which}")) {
        return Ok(wrap(gpu.upload_f32(&v, shape)?));
    }
    let per: usize = shape[1..].iter().product();
    let mut all = Vec::with_capacity(n_exp * per);
    for e in 0..n_exp {
        let name = format!("{mp}.experts.{e}.{which}.weight");
        let v = w.read(&name).map_err(|err| {
            read_err(
                &name,
                format!("{err} (also tried the stacked `{mp}.experts.{which}`)"),
            )
        })?;
        if v.len() != per {
            return Err(read_err(
                &name,
                format!(
                    "expert {e} `{which}` has {} elements, expected {per}",
                    v.len()
                ),
            ));
        }
        all.extend_from_slice(&v);
    }
    Ok(wrap(gpu.upload_f32(&all, shape)?))
}

fn up1(gpu: &mut Gpu, w: &dyn TensorReader, name: &str) -> HipResult<GpuTensor> {
    let v = w.read(name).map_err(|e| read_err(name, e))?;
    let n = v.len();
    gpu.upload_f32(&v, &[n])
}

impl TrunkWeights {
    /// Upload from checkpoint-named weights.
    pub fn upload(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, w: &dyn TensorReader) -> HipResult<Self> {
        let p = "model.language_model";
        let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
        let width = hc * hidden;
        let lr = cfg.gated_residual.lowrank;
        let m = &cfg.moe;
        let d = &cfg.deltanet;
        let ix = &cfg.indexer;

        let hcw = |gpu: &mut Gpu, base: &str, inject: bool| -> HipResult<HcWeights> {
            Ok(HcWeights {
                hc_norm: up1(gpu, w, &format!("{base}.hc_norm.weight"))?,
                mix_down: up2(
                    gpu,
                    w,
                    &format!("{base}.input_mix_weight_down.weight"),
                    lr,
                    width,
                )?,
                mix_up: up2(
                    gpu,
                    w,
                    &format!("{base}.input_mix_weight_up.weight"),
                    width,
                    lr,
                )?,
                block_inject: if inject {
                    Some(up2(
                        gpu,
                        w,
                        &format!("{base}.block_inject_weight.weight"),
                        hc,
                        width,
                    )?)
                } else {
                    None
                },
            })
        };

        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let lp = format!("{p}.layers.{l}");
            let mixer = if cfg.layer_types[l] == LayerType::LinearAttention {
                let la = format!("{lp}.linear_attn");
                TokenMixer::Gdn(GdnWeights {
                    in_proj_qkv: up2(
                        gpu,
                        w,
                        &format!("{la}.in_proj_qkv.weight"),
                        d.qkv_dim(),
                        hidden,
                    )?,
                    in_proj_z: up2(gpu, w, &format!("{la}.in_proj_z.weight"), d.z_dim(), hidden)?,
                    in_proj_a: up2(
                        gpu,
                        w,
                        &format!("{la}.in_proj_a.weight"),
                        d.value_heads,
                        hidden,
                    )?,
                    in_proj_b: up2(
                        gpu,
                        w,
                        &format!("{la}.in_proj_b.weight"),
                        d.value_heads,
                        hidden,
                    )?,
                    conv_weight: up2(
                        gpu,
                        w,
                        &format!("{la}.conv1d.weight"),
                        d.qkv_dim(),
                        d.conv_kernel,
                    )?,
                    a_log: up1(gpu, w, &format!("{la}.A_log"))?,
                    dt_bias: up1(gpu, w, &format!("{la}.dt_bias"))?,
                    norm_weight: up1(gpu, w, &format!("{la}.norm.weight"))?,
                    out_proj: up2(gpu, w, &format!("{la}.out_proj.weight"), hidden, d.z_dim())?,
                })
            } else {
                let sa = format!("{lp}.self_attn");
                let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
                TokenMixer::Qsa(QsaWeights {
                    q_proj: up2(gpu, w, &format!("{sa}.q_proj.weight"), nh * hd * 2, hidden)?,
                    k_proj: up2(gpu, w, &format!("{sa}.k_proj.weight"), nkv * hd, hidden)?,
                    v_proj: up2(gpu, w, &format!("{sa}.v_proj.weight"), nkv * hd, hidden)?,
                    o_proj: up2(gpu, w, &format!("{sa}.o_proj.weight"), hidden, nh * hd)?,
                    q_norm: up1(gpu, w, &format!("{sa}.q_norm.weight"))?,
                    k_norm: up1(gpu, w, &format!("{sa}.k_norm.weight"))?,
                    ix_qk_proj: up2(
                        gpu,
                        w,
                        &format!("{sa}.indexer.index_qk_proj.weight"),
                        (ix.n_heads + ix.kv_heads) * ix.head_dim,
                        hidden,
                    )?,
                    ix_q_norm: up1(gpu, w, &format!("{sa}.indexer.q_layernorm.weight"))?,
                    ix_k_norm: up1(gpu, w, &format!("{sa}.indexer.k_layernorm.weight"))?,
                })
            };
            let mp = format!("{lp}.mlp");
            let moe = MoeWeights {
                router: up2(gpu, w, &format!("{mp}.gate.weight"), m.num_experts, hidden)?,
                gate_up: stack_experts(
                    gpu,
                    w,
                    &mp,
                    "gate_up_proj",
                    m.num_experts,
                    &[m.num_experts, 2 * m.intermediate, hidden],
                )?,
                down: stack_experts(
                    gpu,
                    w,
                    &mp,
                    "down_proj",
                    m.num_experts,
                    &[m.num_experts, hidden, m.intermediate],
                )?,
                shared_gate: up2(
                    gpu,
                    w,
                    &format!("{mp}.shared_expert.gate_proj.weight"),
                    m.shared_intermediate,
                    hidden,
                )?,
                shared_up: up2(
                    gpu,
                    w,
                    &format!("{mp}.shared_expert.up_proj.weight"),
                    m.shared_intermediate,
                    hidden,
                )?,
                shared_down: up2(
                    gpu,
                    w,
                    &format!("{mp}.shared_expert.down_proj.weight"),
                    hidden,
                    m.shared_intermediate,
                )?,
                shared_expert_gate: up2(
                    gpu,
                    w,
                    &format!("{mp}.shared_expert_gate.weight"),
                    1,
                    hidden,
                )?,
            };
            let ple = match cfg.ngram.as_ref().filter(|n| n.layer_idx == l) {
                Some(n) => {
                    let pl = format!("{lp}.ple");
                    Some(PleWeights {
                        key_proj: up2(
                            gpu,
                            w,
                            &format!("{pl}.key_proj.weight"),
                            width,
                            n.embed_dim,
                        )?,
                        value_proj: up2(
                            gpu,
                            w,
                            &format!("{pl}.value_proj.weight"),
                            hidden,
                            n.embed_dim,
                        )?,
                        norm_key: up1(gpu, w, &format!("{pl}.norm_key.weight"))?,
                        norm_query: up1(gpu, w, &format!("{pl}.norm_query.weight"))?,
                        norm_conv: up1(gpu, w, &format!("{pl}.norm_conv.weight"))?,
                        conv_weight: up2(
                            gpu,
                            w,
                            &format!("{pl}.conv1d.weight"),
                            width,
                            n.conv_kernel,
                        )?,
                    })
                }
                None => None,
            };
            layers.push(LayerWeights {
                attn_hc: hcw(gpu, &format!("{lp}.attn_hyper_connection"), true)?,
                mlp_hc: hcw(gpu, &format!("{lp}.mlp_hyper_connection"), true)?,
                mixer,
                moe,
                ple,
            });
        }

        Ok(Self {
            layers,
            mixer: hcw(gpu, &format!("{p}.hyper_connection_mixer"), false)?,
            lm_head: up2(gpu, w, "lm_head.weight", cfg.vocab, hidden)?,
        })
    }
}

impl TrunkState {
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, max_seq: usize) -> HipResult<Self> {
        let mut gdn = Vec::with_capacity(cfg.layers);
        let mut qsa = Vec::with_capacity(cfg.layers);
        let mut ple = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let linear = cfg.layer_types[l] == LayerType::LinearAttention;
            gdn.push(if linear {
                Some(GdnState::zeros(gpu, cfg)?)
            } else {
                None
            });
            qsa.push(if linear {
                None
            } else {
                Some(QsaCache::new(gpu, cfg, max_seq)?)
            });
            ple.push(match cfg.ngram.as_ref().filter(|n| n.layer_idx == l) {
                Some(_) => Some(PleScratch::new(gpu, cfg)?),
                None => None,
            });
        }
        Ok(Self { gdn, qsa, ple })
    }
}

impl TrunkScratch {
    /// The last-position logits buffer, `[vocab]` f32, left populated by
    /// [`decode_step_into`]. Exposed so the serving seam can sample on the GPU
    /// instead of paying a `vocab`-wide download every token.
    pub fn logits(&self) -> &GpuTensor {
        &self.logits
    }

    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig, max_seq: usize) -> HipResult<Self> {
        let width = cfg.gated_residual.count * cfg.hidden;
        Ok(Self {
            hc: HcScratch::new(gpu, cfg)?,
            gdn: GdnScratch::new(gpu, cfg)?,
            qsa: QsaScratch::new(gpu, cfg, max_seq)?,
            moe: MoeScratch::new(gpu, cfg)?,
            mixed: gpu.zeros(&[cfg.hidden], DType::F32)?,
            block_out: gpu.zeros(&[cfg.hidden], DType::F32)?,
            wide: gpu.zeros(&[width], DType::F32)?,
            ple_out: gpu.zeros(&[width], DType::F32)?,
            collapsed: gpu.zeros(&[cfg.hidden], DType::F32)?,
            logits: gpu.zeros(&[cfg.vocab], DType::F32)?,
        })
    }
}

/// One token in, `[vocab]` logits out. Advances every layer's state.
///
/// `history` is the token stream INCLUDING this token; the n-gram window needs it
/// and it is EOS-segment aware, so a bare "previous two tokens" is not enough.
#[allow(clippy::too_many_arguments)]
/// One decode step, leaving the last-position logits in `s.logits` ON THE GPU.
///
/// This is the serving entry point. [`decode_step`] wraps it and downloads, which
/// is what the parity examples and tests want but costs a `vocab`-wide transfer
/// per token — 993 KB at the shipped 248320 vocab, every step, for a sampler that
/// may only need an argmax.
pub fn decode_step_into(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &TrunkWeights,
    st: &mut TrunkState,
    s: &mut TrunkScratch,
    embed_table: &[f32],
    ngram_rows: Option<&dyn crate::ngram_rows::NgramRows>,
    history: &[u32],
    pos: usize,
    eos: u32,
) -> HipResult<()> {
    let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
    let tok = history[history.len() - 1] as usize;

    // Seed every stream with a COPY of the embedding — not zero-padding.
    let e = &embed_table[tok * hidden..(tok + 1) * hidden];
    let seeded: Vec<f32> = (0..hc).flat_map(|_| e.iter().copied()).collect();
    let wide = gpu.upload_f32(&seeded, &[hc * hidden])?;
    gpu.memcpy_dtod_auto(&s.wide.buf, &wide.buf, hc * hidden * 4)?;

    let visible: Vec<usize> = (0..=pos).collect();

    for l in 0..cfg.layers {
        let lw = &w.layers[l];

        // PLE is additive on the WIDE stream, before the residual read.
        if let (Some(pw), Some(ps), Some(n), Some(src)) = (
            lw.ple.as_ref(),
            st.ple[l].as_mut(),
            cfg.ngram.as_ref(),
            ngram_rows,
        ) {
            let hasher = NgramHasher::from_config(n, cfg.vocab as u64, eos);
            let hd = n.head_dim();
            let ctx_len = n.ngram_size - 1;
            let mut hist: Vec<u32> = vec![eos; ctx_len];
            hist.extend_from_slice(history);
            let i = hist.len() - 1;
            let preds: Vec<Option<u32>> = hist[..i].iter().map(|&v| Some(v)).collect();
            let rows = hasher.rows(hist[i], &preds);
            // `heads_per_ngram` rows, whether they come from a resident slice or a
            // ranged read of the shard tensors — the trunk does not know which.
            let emb = src.gather(&rows, hd).map_err(|e| HipError::new(0, &e))?;
            let g_emb = gpu.upload_f32(&emb, &[n.embed_dim])?;
            ple_step(gpu, cfg, pw, ps, &s.wide, &g_emb, &s.ple_out)?;
            gpu.add_inplace_f32(&s.wide, &s.ple_out)?;
        }

        // Token mixer, then MoE — same residual shape both times.
        hc_read(gpu, cfg, &lw.attn_hc, &mut s.hc, &s.wide, &s.mixed)?;
        match &lw.mixer {
            TokenMixer::Gdn(gw) => {
                let gs = st.gdn[l].as_mut().expect("gdn state on a linear layer");
                gdn_decode_step(gpu, cfg, gw, &mut s.gdn, gs, &s.mixed, &s.block_out)?;
            }
            TokenMixer::Qsa(qw) => {
                let cache = st.qsa[l].as_mut().expect("kv cache on a sparse-attn layer");
                qsa_decode_step(
                    gpu,
                    cfg,
                    qw,
                    &mut s.qsa,
                    cache,
                    &s.mixed,
                    pos,
                    &visible,
                    &s.block_out,
                )?;
            }
        }
        hc_write(gpu, cfg, &s.hc, &s.wide, &s.block_out)?;

        hc_read(gpu, cfg, &lw.mlp_hc, &mut s.hc, &s.wide, &s.mixed)?;
        moe_forward(gpu, cfg, &lw.moe, &mut s.moe, &s.mixed, &s.block_out)?;
        hc_write(gpu, cfg, &s.hc, &s.wide, &s.block_out)?;
    }

    // The mixer's own norm is the LAST normalisation — there is no `model.norm`.
    hc_read(gpu, cfg, &w.mixer, &mut s.hc, &s.wide, &s.collapsed)?;
    gpu.gemv_f32(&w.lm_head, &s.collapsed, &s.logits)
}

/// [`decode_step_into`] followed by a download of the logits.
#[allow(clippy::too_many_arguments)]
pub fn decode_step(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &TrunkWeights,
    st: &mut TrunkState,
    s: &mut TrunkScratch,
    embed_table: &[f32],
    ngram_rows: Option<&dyn crate::ngram_rows::NgramRows>,
    history: &[u32],
    pos: usize,
    eos: u32,
) -> HipResult<Vec<f32>> {
    decode_step_into(
        gpu,
        cfg,
        w,
        st,
        s,
        embed_table,
        ngram_rows,
        history,
        pos,
        eos,
    )?;
    gpu.download_f32(&s.logits)
}

// ── GPU teardown and per-sequence reset ─────────────────────────────────────
//
// Exhaustive destructures throughout, so a field added later fails to compile
// until someone decides whether it needs freeing (see the note in `hc_gpu.rs`).
// An `unload` that silently leaks has no test that would catch it, and this model
// is 360 GB.

impl TokenMixer {
    pub fn free(self, gpu: &mut Gpu) {
        match self {
            TokenMixer::Gdn(w) => w.free(gpu),
            TokenMixer::Qsa(w) => w.free(gpu),
        }
    }
}

impl LayerWeights {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            attn_hc,
            mlp_hc,
            mixer,
            moe,
            ple,
        } = self;
        attn_hc.free(gpu);
        mlp_hc.free(gpu);
        mixer.free(gpu);
        moe.free(gpu);
        if let Some(p) = ple {
            p.free(gpu);
        }
    }
}

impl TrunkWeights {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            layers,
            mixer,
            lm_head,
        } = self;
        for l in layers {
            l.free(gpu);
        }
        mixer.free(gpu);
        let _ = gpu.free_tensor(lm_head);
    }
}

impl TrunkState {
    /// Drop everything carried between sequences.
    ///
    /// All three halves matter and they fail differently: a stale GDN recurrent
    /// state or PLE conv ring silently conditions the next prompt on the last
    /// one, while a stale KV `len` makes attention read positions the new
    /// sequence never wrote.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        for g in self.gdn.iter().flatten() {
            g.reset(gpu)?;
        }
        for q in self.qsa.iter_mut().flatten() {
            q.reset();
        }
        for p in self.ple.iter().flatten() {
            p.reset(gpu)?;
        }
        Ok(())
    }

    pub fn free(self, gpu: &mut Gpu) {
        let Self { gdn, qsa, ple } = self;
        for g in gdn.into_iter().flatten() {
            g.free(gpu);
        }
        for q in qsa.into_iter().flatten() {
            q.free(gpu);
        }
        for p in ple.into_iter().flatten() {
            p.free(gpu);
        }
    }
}

impl TrunkScratch {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            hc,
            gdn,
            qsa,
            moe,
            mixed,
            block_out,
            wide,
            ple_out,
            collapsed,
            logits,
        } = self;
        hc.free(gpu);
        gdn.free(gpu);
        qsa.free(gpu);
        moe.free(gpu);
        for t in [mixed, block_out, wide, ple_out, collapsed, logits] {
            let _ = gpu.free_tensor(t);
        }
    }
}
