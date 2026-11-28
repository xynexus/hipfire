// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP speculation inside `decode_step`, differenced against plain decode.
//!
//! Speculation must be INVISIBLE: every logit row the decode loop sees comes
//! from the trunk over the true history, whatever the drafter proposed. So the
//! check is a greedy run with no head, then the same run three more times:
//!
//! * a SCRIPTED head that drafts the true continuation — every verify hits,
//!   so the "row 1 already ran" path carries the whole sequence;
//! * a scripted head that is always WRONG — every verify misses, so every step
//!   restores the row-0 checkpoint (GDN recurrence + conv ring, PLE ring, QSA
//!   length) and re-verifies;
//! * a real (synthetic, random-weight) head — the natural mix, and the head's
//!   own GPU forward and cache bookkeeping.
//!
//! Each must reproduce the reference tokens, with logits inside the batched
//! forward's numeric bound (its GEMMs run W8A8 / W4A16 where decode's GEMVs do
//! not). The hit/miss counters are checked too: a pass where the hit run never
//! hit, or the miss run never missed, would prove nothing.
//!
//!     mtp_serve_fixture <fixture.hfq> [kv_cache]

use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_arch_qwen4exp::mtp_gpu::MtpWeightsGpu;
use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
use hipfire_arch_qwen4exp::trunk_gpu::TensorReader;
use hipfire_rdna::Gpu;
use hipfire_runtime::arch::{ServingBackend, SimpleAr};
use hipfire_runtime::hfq::HfqFile;
use std::collections::HashMap;
use std::path::Path;

const STEPS: usize = 12;

fn fail(msg: String) -> ! {
    eprintln!("mtp_serve_fixture: FAIL — {msg}");
    std::process::exit(1);
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}

/// Generate `STEPS` tokens: greedily when `feed` is `None`, otherwise
/// TEACHER-FORCED with `feed`. Every speculative arm is fed the reference's
/// tokens, so a near-tie argmax flip under the batched forward's ~1% numeric
/// noise cannot cascade into a different sequence and pose as a rollback bug.
fn run(
    m: &mut Qwen4ExpBackend,
    gpu: &mut Gpu,
    prompt: &[u32],
    feed: Option<&[u32]>,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    m.prefill(gpu, prompt)
        .unwrap_or_else(|e| fail(format!("prefill: {e}")));
    let (mut toks, mut logits) = (Vec::new(), Vec::new());
    for step in 0..STEPS {
        let l = gpu.download_f32(m.logits()).unwrap();
        let t = argmax(&l);
        logits.push(l);
        toks.push(t);
        let next = feed.map_or(t, |f| f[step]);
        m.decode_step(gpu, next, prompt.len() + step)
            .unwrap_or_else(|e| fail(format!("decode_step {step}: {e}")));
    }
    (toks, logits)
}

/// Top-1 minus top-2 logit.
fn margin(v: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in v {
        if x > a {
            b = a;
            a = x;
        } else if x > b {
            b = x;
        }
    }
    a - b
}

struct Src(HashMap<String, Vec<f32>>);
impl TensorReader for Src {
    fn read(&self, name: &str) -> Result<Vec<f32>, String> {
        self.0
            .get(name)
            .cloned()
            .ok_or_else(|| format!("missing weight `{name}`"))
    }
}

/// A random-weight MTP head at `cfg`'s geometry — same names and shapes the
/// shipped sidecar carries (see `parity_mtp_gpu_vs_cpu`).
fn synthetic_head(cfg: &Qwen4ExpConfig) -> Src {
    let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
    let width = hc * hidden;
    let lr = cfg.gated_residual.lowrank;
    let (m, ix) = (&cfg.moe, &cfg.indexer);
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut put = |name: String, len: usize, scale: f32| {
        let mut s = name.bytes().fold(2_166_136_261u32, |h, b| {
            (h ^ b as u32).wrapping_mul(16_777_619)
        });
        let v = (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s % 2000) as f32 / 1000.0 - 1.0) * scale
            })
            .collect();
        w.insert(name, v);
    };
    let lin = |k: usize| 1.0 / (k as f32).sqrt();
    put("mtp.pre_fc_norm_hidden.weight".into(), width, 0.1);
    put("mtp.pre_fc_norm_embedding.weight".into(), hidden, 0.1);
    put("mtp.fc_hidden.weight".into(), hidden * hidden, lin(hidden));
    put(
        "mtp.fc_embedding.weight".into(),
        hidden * hidden,
        lin(hidden),
    );
    let lp = "mtp.layers.0";
    for (base, inject) in [
        (format!("{lp}.attn_hyper_connection"), true),
        (format!("{lp}.mlp_hyper_connection"), true),
        ("mtp.hyper_connection_mixer".to_string(), false),
    ] {
        put(format!("{base}.hc_norm.weight"), width, 0.1);
        put(
            format!("{base}.input_mix_weight_down.weight"),
            lr * width,
            lin(width),
        );
        put(
            format!("{base}.input_mix_weight_up.weight"),
            width * lr,
            lin(lr),
        );
        if inject {
            put(
                format!("{base}.block_inject_weight.weight"),
                hc * width,
                lin(width),
            );
        }
    }
    let sa = format!("{lp}.self_attn");
    let (q, kv) = (cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
    put(format!("{sa}.q_proj.weight"), q * 2 * hidden, lin(hidden));
    put(format!("{sa}.k_proj.weight"), kv * hidden, lin(hidden));
    put(format!("{sa}.v_proj.weight"), kv * hidden, lin(hidden));
    put(format!("{sa}.o_proj.weight"), hidden * q, lin(q));
    put(format!("{sa}.q_norm.weight"), cfg.head_dim, 1.0);
    put(format!("{sa}.k_norm.weight"), cfg.head_dim, 1.0);
    put(
        format!("{sa}.indexer.index_qk_proj.weight"),
        (ix.n_heads + ix.kv_heads) * ix.head_dim * hidden,
        lin(hidden),
    );
    put(format!("{sa}.indexer.q_layernorm.weight"), ix.head_dim, 1.0);
    put(format!("{sa}.indexer.k_layernorm.weight"), ix.head_dim, 1.0);
    let mp = format!("{lp}.mlp");
    put(
        format!("{mp}.gate.weight"),
        m.num_experts * hidden,
        lin(hidden),
    );
    put(
        format!("{mp}.experts.gate_up_proj"),
        m.num_experts * 2 * m.intermediate * hidden,
        lin(hidden),
    );
    put(
        format!("{mp}.experts.down_proj"),
        m.num_experts * hidden * m.intermediate,
        lin(m.intermediate),
    );
    put(
        format!("{mp}.shared_expert.gate_proj.weight"),
        m.shared_intermediate * hidden,
        lin(hidden),
    );
    put(
        format!("{mp}.shared_expert.up_proj.weight"),
        m.shared_intermediate * hidden,
        lin(hidden),
    );
    put(
        format!("{mp}.shared_expert.down_proj.weight"),
        hidden * m.shared_intermediate,
        lin(m.shared_intermediate),
    );
    put(
        format!("{mp}.shared_expert_gate.weight"),
        hidden,
        lin(hidden),
    );
    Src(w)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| fail("usage: mtp_serve_fixture <fixture.hfq>".into()));
    let mut hfq = HfqFile::open(Path::new(&path)).unwrap_or_else(|e| fail(format!("open: {e}")));
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("mtp_serve_fixture: no GPU ({e:?}) — skipped");
            return;
        }
    };
    // KVarN (the family default) quantises K in 128-token blocks, so the prompt
    // ends a few tokens short of a block boundary and the speculative steps
    // CROSS it: a draft that closes block 0 and is then rejected must leave the
    // block re-quantised from the real token, not the draft.
    // Optional 2nd arg: the `kv_cache` setting (default: the family default).
    let kv = hipfire_arch_qwen4exp::attn_gpu::QsaKvMode::from_setting(
        &std::env::args().nth(2).unwrap_or_else(|| "auto".into()),
    );
    let mut m = Qwen4ExpBackend::load_with_kv(&mut gpu, &mut hfq, 256, kv)
        .unwrap_or_else(|e| fail(format!("load: {e}")));
    println!("  kv: {kv:?}");
    let cfg = m.config().clone();
    let vocab = cfg.vocab as u32;
    let prompt: Vec<u32> = (0..122u32).map(|i| (i * 37 + 3) % vocab).collect();

    let (ref_toks, ref_logits) = run(&mut m, &mut gpu, &prompt, None);
    println!("  reference: {STEPS} greedy tokens {ref_toks:?}");

    let head = MtpWeightsGpu::upload(&mut gpu, &cfg, &synthetic_head(&cfg))
        .unwrap_or_else(|e| fail(format!("head upload: {e:?}")));
    m.attach_mtp_weights(&mut gpu, head)
        .unwrap_or_else(|e| fail(format!("attach: {e}")));

    // Absolute-position scripts. Position L+s holds ref_toks[s] (the token the
    // trunk produced from position L+s-1).
    let l = prompt.len();
    let mut truth = vec![0u32; l + STEPS + 2];
    for (s, &t) in ref_toks.iter().enumerate() {
        truth[l + s] = t;
    }
    let wrong: Vec<u32> = truth.iter().map(|t| (t + 1) % vocab).collect();

    let scale = ref_logits
        .iter()
        .flatten()
        .fold(0.0f32, |a, v| a.max(v.abs()))
        .max(1e-6);
    let check = |name: &str, script: Option<Vec<u32>>, m: &mut Qwen4ExpBackend, gpu: &mut Gpu| {
        m.set_draft_script(script);
        let before = m.mtp_stats().unwrap();
        let (toks, logits) = run(m, gpu, &prompt, Some(&ref_toks));
        let after = m.mtp_stats().unwrap();
        let (v, h, x) = (
            after.verifies - before.verifies,
            after.hits - before.hits,
            after.misses - before.misses,
        );
        let worst = logits
            .iter()
            .zip(&ref_logits)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
            .fold(0.0f32, f32::max);
        println!(
            "  {name:9} argmax agree {}/{STEPS}, max|d| {worst:.3e} ({:.2}% of scale), \
             {v} verifies / {h} hits / {x} misses",
            toks.iter().zip(&ref_toks).filter(|(a, b)| a == b).count(),
            100.0 * worst / scale
        );
        // An argmax may differ only where the reference itself was a near
        // tie, inside the numeric bound checked below.
        for (step, (&t, &r)) in toks.iter().zip(&ref_toks).enumerate() {
            let mg = margin(&ref_logits[step]);
            if t != r && mg > 0.03 * scale {
                fail(format!(
                    "{name}: step {step} argmax {t} != reference {r}, reference margin {mg:.3e}"
                ));
            }
        }
        // 3%: the batched 2-row forward's own numerics (W8A8/W4A16 GEMMs where
        // decode runs GEMVs), compounded over 12 steps. The mi640 compact-G128
        // fixture measured 1.7-2.0% — EQUALLY in all-hit, all-miss and head
        // runs, which is what clears the hit and rollback paths themselves.
        if worst > 0.03 * scale {
            fail(format!(
                "{name}: logits off by {worst:.3e} (> 3% of {scale:.3})"
            ));
        }
        (v, h, x)
    };

    let (_, h, x) = check("all-hit", Some(truth), &mut m, &mut gpu);
    if h == 0 || x > 0 {
        fail(format!(
            "all-hit run had {h} hits and {x} misses — the hit path was not exercised"
        ));
    }
    let (_, h, x) = check("all-miss", Some(wrong), &mut m, &mut gpu);
    if x == 0 || h > 0 {
        fail(format!(
            "all-miss run had {h} hits and {x} misses — rollback was not exercised"
        ));
    }
    check("head", None, &mut m, &mut gpu);

    Box::new(m).unload(&mut gpu);
    println!("mtp_serve_fixture: OK");
}
