// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! IS THE SHAPE-INFERRED MTP COMPOSITION RIGHT? — the cheap experiment.
//!
//! `mtp.rs` is the one part of this port with no reference to difference
//! against: upstream sets `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and
//! DROPS the head's weights on load, so its composition is inferred from tensor
//! shapes. Everything downstream — a GPU port, a spec-decode loop — is wasted if
//! that inference is wrong, and none of it would FAIL: speculative decoding is
//! lossless, so a wrong drafter yields correct output that simply never gets
//! accepted. The only signal is acceptance, so measure acceptance first.
//!
//! The test, per position `t`:
//!
//!   trunk    wide residual h_t  (what `mtp.pre_fc_norm_hidden` is shaped for)
//!            + embedding of the NEXT token, x_{t+1}   (see `weights.rs`)
//!   MTP      -> collapsed hidden, the same shape the trunk's mixer emits
//!   score    cosine( mtp_out(t), trunk_collapsed(t+1) )
//!
//! Both feed the SAME `lm_head`, so if the head predicts x_{t+2} the way the
//! trunk does at t+1, those two vectors point the same way. Cosine avoids
//! downloading a 248320-wide `lm_head`, and it is the measure that caught
//! medgemma's scrambled vision projector (see `dequant_oq8g256`).
//!
//! REFERENCE BANDS, decided before running so the result cannot be rationalised:
//!
//!   cos > 0.7   composition is doing real work; a GPU port is justified
//!   0.3 - 0.7   partially right — likely one unpinned choice wrong
//!   < 0.3       no better than the control; do NOT build on it
//!
//! The control is the same cosine against a RANDOM other position's collapsed
//! state, which is what "no signal" actually looks like on these distributions —
//! hidden states of one model share enough structure that even unrelated
//! positions are not at cosine 0.
//!
//! # Two phases, deliberately
//!
//! The head is 9.7 GiB as f32 and the paged trunk is ~20 GiB resident, and on a
//! UMA APU the page cache competes with both — holding them at once OOM-killed
//! this probe twice. So phase 1 runs the trunk and DUMPS its states (a few MB),
//! then exits; phase 2 loads only the head. Peak memory is the larger phase, not
//! the sum.
//!
//!     mtp_probe dump  <base.hfq> <states.bin> [n_tokens]
//!     mtp_probe score <mtp.hfq>  <states.bin>

use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
use hipfire_arch_qwen4exp::trunk::WeightSource;
use hipfire_runtime::arch::{Architecture, SimpleAr};
use hipfire_runtime::hfq::HfqFile;
use std::collections::HashMap;
use std::path::Path;

/// Dequantize one stored tensor to f32.
///
/// Inlined rather than reaching for `hipfire-train`'s `DequantHfq`: an arch
/// crate pulling in the training crate for a probe is the wrong dependency
/// direction. Only the four types an `oq8`/`bf16` MTP sidecar actually contains
/// are handled, and anything else is a loud panic rather than a wrong number.
fn to_f32(qt: u8, bytes: &[u8], n: usize, name: &str) -> Vec<f32> {
    match qt {
        3 => hipfire_runtime::quant::dequant_q8f16(bytes, n),
        35 => hipfire_runtime::quant::dequant_oq8g256(bytes, n),
        54 => hipfire_runtime::quant::dequant_oq8g128(bytes, n),
        1 => bytes
            .chunks_exact(2)
            .map(|c| {
                let h = u16::from_le_bytes([c[0], c[1]]);
                let (sgn, exp, man) = ((h >> 15) & 1, (h >> 10) & 0x1f, h & 0x3ff);
                let bits = match exp {
                    0 if man == 0 => (sgn as u32) << 31,
                    // Subnormal f16: renormalise into f32's wider exponent.
                    0 => {
                        let mut e = -1i32;
                        let mut m = man as u32;
                        while m & 0x400 == 0 {
                            m <<= 1;
                            e -= 1;
                        }
                        ((sgn as u32) << 31)
                            | (((127 - 15 + e + 1) as u32) << 23)
                            | ((m & 0x3ff) << 13)
                    }
                    0x1f => ((sgn as u32) << 31) | (0xff << 23) | ((man as u32) << 13),
                    _ => {
                        ((sgn as u32) << 31)
                            | (((exp as i32 - 15 + 127) as u32) << 23)
                            | ((man as u32) << 13)
                    }
                };
                f32::from_bits(bits)
            })
            .take(n)
            .collect(),
        16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .take(n)
            .collect(),
        2 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .take(n)
            .collect(),
        other => {
            panic!("mtp probe: {name} has quant_type {other}, which this probe does not decode")
        }
    }
}

/// Every `mtp.*` tensor, dequantized to f32 once.
struct MtpSrc(HashMap<String, Vec<f32>>);

impl WeightSource for MtpSrc {
    fn get(&self, name: &str) -> &[f32] {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("mtp probe: missing weight `{name}`"))
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn write_states(path: &str, n: usize, hidden: usize, hc: usize, v: &[Vec<f32>]) {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create states"));
    for x in [n, hidden, hc] {
        f.write_all(&(x as u64).to_le_bytes()).expect("write hdr");
    }
    for row in v {
        f.write_all(&(row.len() as u64).to_le_bytes())
            .expect("write len");
        for &y in row {
            f.write_all(&y.to_le_bytes()).expect("write f32");
        }
    }
}

fn read_states(path: &str) -> (usize, usize, usize, Vec<Vec<f32>>) {
    let b = std::fs::read(path).expect("read states");
    let g = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) as usize;
    let (n, hidden, hc) = (g(0), g(8), g(16));
    let mut off = 24;
    let mut rows = Vec::new();
    while off < b.len() {
        let len = g(off);
        off += 8;
        let mut row = Vec::with_capacity(len);
        for i in 0..len {
            row.push(f32::from_le_bytes(
                b[off + i * 4..off + i * 4 + 4].try_into().unwrap(),
            ));
        }
        off += len * 4;
        rows.push(row);
    }
    (n, hidden, hc, rows)
}

fn phase_dump(base: &str, out: &str, n_tok: usize) {
    let mut hfq = HfqFile::open(Path::new(base)).expect("open base artifact");
    let mut gpu = match hipfire_rdna::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("mtp_probe: no GPU ({e:?}) — skipped");
            return;
        }
    };
    let mut m = Qwen4ExpBackend::load(&mut gpu, &mut hfq, 256).expect("load base");
    let cfg = m.config().clone();
    let (hidden, hc) = (cfg.hidden, cfg.gated_residual.count);
    // GENERATE, don't feed fixed ids. A synthetic prompt (9707 + i%977) drives
    // the trunk into degenerate repetition — the first run of this probe showed
    // `198, 271, 487, 220` cycling — which both inflates the do-nothing baseline
    // (consecutive tokens repeat, so reusing the last state "predicts" well) and
    // asks the drafter to work on a distribution the model never sees. Feeding
    // the trunk its OWN argmax makes the sequence self-consistent, which is the
    // distribution a real drafter would face.
    let mut prompt: Vec<u32> = vec![9707];

    // Decode, not prefill: this is the state a drafter would actually see.
    let mut rows: Vec<Vec<f32>> = Vec::new();
    m.prefill(&mut gpu, &prompt[..1]).expect("prefill");
    for i in 0..n_tok {
        let tok = prompt[i];
        if i > 0 {
            m.decode_step(&mut gpu, tok, i).expect("decode");
        }
        let (wide_t, coll_t) = m.trunk_states();
        rows.push(gpu.download_f32(wide_t).expect("download wide"));
        rows.push(gpu.download_f32(coll_t).expect("download collapsed"));
        rows.push(m.embed_row(tok).to_vec());
        // The token the TRUNK would emit here. Token-level agreement against
        // this is what speculative decoding actually accepts on; cosine between
        // hidden states is only a proxy, and a weak one in 2560 dimensions.
        let lg = gpu.download_f32(m.trunk_logits()).expect("download logits");
        let (am, _) =
            lg.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                });
        rows.push(vec![am as f32]);
        // The trunk's own next token continues the sequence.
        prompt.push(am as u32);
    }
    assert_eq!(rows[0].len(), hc * hidden, "wide residual width");
    assert_eq!(rows[1].len(), hidden, "collapsed width");
    write_states(out, n_tok, hidden, hc, &rows);
    println!("wrote {} positions to {out}", n_tok);
}

fn phase_score(mtp_path: &str, states: &str, base: Option<&str>) {
    let (n_tok, hidden, hc, rows) = read_states(states);
    let wide = |t: usize| &rows[t * 4];
    let collapsed = |t: usize| &rows[t * 4 + 1];
    let embed = |t: usize| &rows[t * 4 + 2];
    let trunk_tok = |t: usize| rows[t * 4 + 3][0] as usize;

    let mtp_hfq = HfqFile::open(Path::new(mtp_path)).expect("open mtp sidecar");
    let names: Vec<String> = mtp_hfq
        .tensors()
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| n.starts_with("mtp."))
        .collect();
    if names.is_empty() {
        eprintln!("mtp_probe: {mtp_path} contains no `mtp.` tensors");
        std::process::exit(2);
    }
    let mut map = HashMap::new();
    let mut bytes = 0usize;
    for n in &names {
        let (info, raw) = mtp_hfq
            .tensor_data(n)
            .unwrap_or_else(|| panic!("mtp probe: {n} has no data"));
        let count: usize = info.shape.iter().map(|&d| d as usize).product();
        let v = to_f32(info.quant_type, &raw, count, n);
        assert_eq!(v.len(), count, "{n}: decoded {} != {count}", v.len());
        bytes += v.len() * 4;
        map.insert(n.clone(), v);
    }
    println!(
        "mtp head: {} tensors, {:.2} GiB as f32",
        names.len(),
        bytes as f64 / (1u64 << 30) as f64
    );

    // RESTACK. The source ships routed experts as one tensor
    // (`experts.gate_up_proj`, [n_experts, 2*mi, hidden]) and the CPU forward
    // slices it per expert — but `hipfire-quantize` SPLITS stacked MoE experts
    // into `experts.<e>.<proj>.weight` on the way into an artifact. So the
    // on-disk layout and the layout `mtp::weights_from` reads are not the same,
    // and the mismatch surfaces as a missing-weight panic rather than a wrong
    // answer. Concatenate them back in expert order.
    for proj in ["gate_up_proj", "down_proj"] {
        let stacked = format!("mtp.layers.0.mlp.experts.{proj}");
        if map.contains_key(&stacked) {
            continue;
        }
        // REMOVE each part as it is consumed. Concatenating while the originals
        // stay in the map holds the head TWICE — ~19 GiB instead of 9.7 — which
        // OOM-killed this phase on a box whose page cache is already full from
        // reading the 223 GB archive.
        let mut joined: Vec<f32> = Vec::new();
        let mut e = 0usize;
        while let Some(part) = map.remove(&format!("mtp.layers.0.mlp.experts.{e}.{proj}.weight")) {
            joined.extend_from_slice(&part);
            e += 1;
        }
        if e == 0 {
            panic!("mtp probe: neither stacked nor per-expert `{proj}` present");
        }
        joined.shrink_to_fit();
        println!(
            "  restacked {e} experts -> {stacked} ({} f32)",
            joined.len()
        );
        map.insert(stacked, joined);
    }
    let src = MtpSrc(map);

    // The config is rebuilt from the sidecar's own metadata so this phase needs
    // nothing from the base artifact.
    // The sidecar carries the same config blob as the base (same source), so
    // this phase needs nothing from the 158 GB artifact.
    let cfg: Qwen4ExpConfig = hipfire_arch_qwen4exp::arch::Qwen4Exp::config_from_hfq(&mtp_hfq)
        .expect("mtp sidecar carries no usable config");
    assert_eq!(cfg.hidden, hidden, "hidden mismatch vs dumped states");
    assert_eq!(cfg.gated_residual.count, hc, "hc mismatch vs dumped states");
    let w = hipfire_arch_qwen4exp::mtp::weights_from(&cfg, &src);

    // SWEEP the conventions the shapes do NOT pin, instead of assuming one.
    //
    // The module docs name the unpinned choice: "how `fc_embedding`'s narrow
    // output reaches the wide stream — broadcast-added to every stream is the
    // natural reading and the one used here, but nothing in the shapes rules
    // out, say, adding it to stream 0 only." The INDEX convention is the other
    // free variable: whether the head consumes emb(x_t) or emb(x_{t+1}), and
    // whether its output should match the trunk at t, t+1 or t+2.
    //
    // All of it is CPU over dumped states, so the whole grid costs one GPU run.
    let scored = n_tok - 2; // leave room for a +2 target
    let ifreq = hipfire_arch_qwen4exp::rope::inv_freq(cfg.rotary_dim(), cfg.rope_theta);
    let (cos, sin) = hipfire_arch_qwen4exp::rope::cos_sin(&(0..scored).collect::<Vec<_>>(), &ifreq);

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let mut best = (f32::MIN, String::new());

    // THE BASELINE THAT MATTERS: the head's output inherits the trunk's
    // residual — `mtp::forward` ends with the same mixer collapse the trunk
    // uses, over a `wide` built from the trunk's own `wide(t)`. So a high
    // cosine with collapsed(t) mostly says "the residual survived the layer",
    // not "the head predicts". Against collapsed(t+1) the question is whether
    // the head moves the residual TOWARD the next state further than doing
    // nothing does. `identity` is doing nothing: the trunk's own state at t.
    let ident_next = mean(
        &(0..scored)
            .map(|t| cosine(collapsed(t), collapsed(t + 1)))
            .collect::<Vec<_>>(),
    );
    let ident_same = mean(
        &(0..scored)
            .map(|t| cosine(collapsed(t), collapsed(t)))
            .collect::<Vec<_>>(),
    );
    println!("\n  do-nothing baseline (no MTP at all):");
    println!("    cos(collapsed_t, collapsed_t)   = {ident_same:+.4}  (trivially 1)");
    println!("    cos(collapsed_t, collapsed_t+1) = {ident_next:+.4}  <- BEAT THIS to be useful");
    println!("\n  fusion            emb   target   signal   control   sep     vs-baseline");
    for (fname, fusion) in [
        (
            "broadcast-all",
            hipfire_arch_qwen4exp::mtp::Fusion::BroadcastAllStreams,
        ),
        (
            "stream0-only ",
            hipfire_arch_qwen4exp::mtp::Fusion::Stream0Only,
        ),
    ] {
        for emb_off in [0usize, 1] {
            let mut wide_in: Vec<f32> = Vec::with_capacity(scored * hc * hidden);
            let mut emb_in: Vec<f32> = Vec::with_capacity(scored * hidden);
            for t in 0..scored {
                wide_in.extend_from_slice(wide(t));
                emb_in.extend_from_slice(embed(t + emb_off));
            }
            let out = hipfire_arch_qwen4exp::mtp::forward(
                &cfg, &w, &wide_in, &emb_in, scored, &cos, &sin, fusion,
            );
            for tgt_off in [0usize, 1, 2] {
                let (mut sc, mut ct) = (Vec::new(), Vec::new());
                for t in 0..scored {
                    let o = &out[t * hidden..(t + 1) * hidden];
                    sc.push(cosine(o, collapsed(t + tgt_off)));
                    ct.push(cosine(o, collapsed((t + n_tok / 2) % n_tok)));
                }
                let (ms, mc) = (mean(&sc), mean(&ct));
                let sep = ms - mc;
                // Only a t+1 target can show DRAFTING value; a t+0 target is the
                // residual-inheritance artifact described above, so it is reported
                // but never allowed to win.
                let vs_base = if tgt_off == 1 {
                    ms - ident_next
                } else {
                    f32::NAN
                };
                println!(
                "  {fname}     t+{emb_off}   t+{tgt_off}     {ms:+.4}  {mc:+.4}  {sep:+.4}  {vs_base:+.4}"
            );
                if tgt_off == 1 && vs_base > best.0 {
                    best = (
                        vs_base,
                        format!(
                            "fusion={}, emb=t+{emb_off}, target=t+{tgt_off}",
                            fname.trim()
                        ),
                    );
                }
            }
        }
    }
    println!(
        "\n  best DRAFTING gain over do-nothing: {:+.4}  at {}",
        best.0, best.1
    );
    if best.0 <= 0.0 {
        println!(
            "  ^ the head does NOT move the residual toward the next state any\n\
             \x20   better than not running it. That is not a drafter."
        );
    }

    // Re-score the winning convention so the verdict below reads it.
    let (mut scores, mut controls) = (Vec::new(), Vec::new());
    {
        let emb_off = if best.1.contains("emb=t+1") { 1 } else { 0 };
        let tgt_off = best
            .1
            .rsplit("target=t+")
            .next()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        let mut wide_in: Vec<f32> = Vec::new();
        let mut emb_in: Vec<f32> = Vec::new();
        for t in 0..scored {
            wide_in.extend_from_slice(wide(t));
            emb_in.extend_from_slice(embed(t + emb_off));
        }
        let fusion = if best.1.contains("stream0") {
            hipfire_arch_qwen4exp::mtp::Fusion::Stream0Only
        } else {
            hipfire_arch_qwen4exp::mtp::Fusion::BroadcastAllStreams
        };
        let out = hipfire_arch_qwen4exp::mtp::forward(
            &cfg, &w, &wide_in, &emb_in, scored, &cos, &sin, fusion,
        );
        for t in 0..scored {
            let o = &out[t * hidden..(t + 1) * hidden];
            scores.push(cosine(o, collapsed(t + tgt_off)));
            controls.push(cosine(o, collapsed((t + n_tok / 2) % n_tok)));
        }
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let (m_sig, m_ctl) = (mean(&scores), mean(&controls));
    println!("\npositions scored: {}", scores.len());
    for (t, (sc, c)) in scores.iter().zip(&controls).enumerate().take(8) {
        println!("  t={t:<3} signal {sc:+.4}   control {c:+.4}");
    }
    println!("\n  mean signal  = {m_sig:+.4}");
    println!("  mean control = {m_ctl:+.4}");
    println!("  separation   = {:+.4}", m_sig - m_ctl);
    // The verdict reads the BASELINE-RELATIVE gain, not the raw cosine: raw
    // cosine is inflated by residual inheritance and would pass a head that
    // does nothing.
    let verdict = if best.0 > 0.15 {
        "USEFUL AS A DRAFTER — clearly beats doing nothing; GPU port justified"
    } else if best.0 > 0.02 {
        "MARGINAL — beats doing nothing, but barely; resolve the fusion first"
    } else {
        "NOT A DRAFTER YET — no gain over the trunk's own residual"
    };
    println!("\nverdict (cosine proxy): {verdict}");

    // ── THE METRIC THAT DECIDES IT ────────────────────────────────────────
    //
    // Cosine between hidden states is a PROXY. Speculative decoding accepts on
    // TOKENS: the draft is kept only where argmax(lm_head . draft) equals what
    // the trunk emits. In 2560 dimensions two states can sit at modest cosine
    // and still argmax to the same row out of 248320, so the proxy can be
    // pessimistic — which is why a negative cosine result must not be the last
    // word before abandoning a drafter.
    let Some(base) = base else {
        println!("\n(no base.hfq given — token-level acceptance not measured)");
        return;
    };
    let base_hfq = HfqFile::open(Path::new(base)).expect("open base artifact");
    let (hi, hraw) = base_hfq
        .tensor_data("lm_head.weight")
        .expect("base artifact has no lm_head.weight");
    let vocab = hi.shape[0] as usize;
    println!("\nlm_head [{vocab}, {hidden}] qt {} -> f32", hi.quant_type);
    let head = to_f32(hi.quant_type, &hraw, vocab * hidden, "lm_head.weight");

    // Re-run the winning convention and take argmax over the real vocabulary.
    let fusion = if best.1.contains("stream0") {
        hipfire_arch_qwen4exp::mtp::Fusion::Stream0Only
    } else {
        hipfire_arch_qwen4exp::mtp::Fusion::BroadcastAllStreams
    };
    let emb_off = if best.1.contains("emb=t+1") { 1 } else { 0 };
    let mut wide_in: Vec<f32> = Vec::new();
    let mut emb_in: Vec<f32> = Vec::new();
    for t in 0..scored {
        wide_in.extend_from_slice(wide(t));
        emb_in.extend_from_slice(embed(t + emb_off));
    }
    let out = hipfire_arch_qwen4exp::mtp::forward(
        &cfg, &w, &wide_in, &emb_in, scored, &cos, &sin, fusion,
    );

    let argmax_of = |h: &[f32]| -> usize {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for r in 0..vocab {
            let row = &head[r * hidden..(r + 1) * hidden];
            let dot: f32 = row.iter().zip(h).map(|(a, b)| a * b).sum();
            if dot > best_v {
                best_v = dot;
                best_i = r;
            }
        }
        best_i
    };

    // Two baselines, both necessary:
    //   trunk_tok(t+1) is what the drafter must match.
    //   do-nothing = feeding the trunk's OWN state at t through lm_head; if the
    //   head cannot beat that, it is not adding prediction, just latency.
    let (mut hit, mut base_hit) = (0usize, 0usize);
    println!("\n  t    mtp_tok   do-nothing   trunk_tok   accept");
    for t in 0..scored {
        let o = &out[t * hidden..(t + 1) * hidden];
        let want = trunk_tok(t + 1);
        let got = argmax_of(o);
        let nothing = argmax_of(collapsed(t));
        if got == want {
            hit += 1;
        }
        if nothing == want {
            base_hit += 1;
        }
        if t < 10 {
            println!(
                "  {t:<3}  {got:<9} {nothing:<12} {want:<11} {}",
                if got == want { "YES" } else { "no" }
            );
        }
    }
    let acc = hit as f32 / scored as f32;
    let base_acc = base_hit as f32 / scored as f32;
    println!(
        "\n  MTP acceptance        = {hit}/{scored}  ({:.1}%)",
        100.0 * acc
    );
    println!(
        "  do-nothing acceptance = {base_hit}/{scored}  ({:.1}%)",
        100.0 * base_acc
    );
    println!(
        "\nTOKEN VERDICT: {}",
        if acc > base_acc && acc >= 0.25 {
            "DRAFTS USEFULLY — port it"
        } else if acc > base_acc {
            "beats do-nothing but acceptance is low"
        } else {
            "does not beat do-nothing on TOKENS either"
        }
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_default();
    match mode.as_str() {
        "dump" => {
            let base = args.next().expect("dump <base.hfq> <states.bin> [n]");
            let out = args.next().expect("dump <base.hfq> <states.bin> [n]");
            let n = args.next().and_then(|v| v.parse().ok()).unwrap_or(16);
            phase_dump(&base, &out, n);
        }
        "score" => {
            let mtp = args
                .next()
                .expect("score <mtp.hfq> <states.bin> [base.hfq]");
            let st = args
                .next()
                .expect("score <mtp.hfq> <states.bin> [base.hfq]");
            let base = args.next(); // optional: enables TOKEN-level acceptance
            phase_score(&mtp, &st, base.as_deref());
        }
        _ => {
            eprintln!("usage: mtp_probe dump <base.hfq> <states.bin> [n]");
            eprintln!("       mtp_probe score <mtp.hfq> <states.bin>");
            std::process::exit(2);
        }
    }
}
