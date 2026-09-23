// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! CALIBRATE qwen4_exp WITHOUT a streamed calibration adapter.
//!
//! `oq4.25++` requires `--hessian`, and `hipfire-coexistence calibrate` refuses
//! this family outright:
//!
//!     InvalidSourcePlan("no native calibration adapter is registered for
//!                        architecture 26")
//!
//! Writing one means reimplementing the family's forward for layer-by-layer
//! streaming — the five that exist run 1057-2410 lines. That machinery is for
//! models too big to hold resident. This one is NOT: it loads paged in ~6 s and
//! decodes at 3.25 tok/s, so the far shorter path is to arm the capture tap over
//! the forward that already works.
//!
//! `weight_gemv` carries that tap, and says so: "the single chokepoint that
//! makes activation capture work for every arch that routes its linears through
//! `weight_gemv`". qwen4_exp routes all of them through it. The only missing
//! piece was a buffer-pointer -> canonical-name map, which is
//! `TrunkWeights::capture_map`.
//!
//! ROUTED EXPERTS TAKE THE IMATRIX PATH. There are `num_experts` per layer and a
//! full [K,K] Hessian each would not fit — the same split every other family
//! uses (`CalibCollector::with_imatrix_only`).
//!
//! THE CORPUS MUST BE REAL TEXT. The first version self-generated its own
//! tokens by greedy argmax, which collapses into a repeating cycle: XtX then
//! has an effective rank of a handful no matter how many tokens run, LDLQ's
//! Cholesky only factorizes at 100x the requested lambda, and H+lambda*I is
//! effectively lambda*I -- the OBS solution is plain RTN with the AWQ scales
//! rebased against a degenerate Hessian. Teacher-force real tokens instead.
//!
//!     calibrate_qwen4exp <model.hfq> <corpus.txt> <out.calib.hfq> [n_tokens]

use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::arch::SimpleAr;
use hipfire_runtime::calibration::CalibCollector;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: calibrate_qwen4exp <model.hfq> <corpus.txt> <out.calib.hfq> [n]");
    let corpus = args.next().expect("corpus path");
    let out = args.next().expect("output .calib.hfq");
    // A full-rank XtX for a [K,K] Hessian needs more than K independent tokens.
    // The largest captured K here is 8192 (`HIPFIRE_CALIB_MAX_K`), so the default
    // clears it with room to spare.
    let n_tok: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(16384);

    let mut hfq = HfqFile::open(Path::new(&model)).expect("open model");
    let mut gpu = match hipfire_rdna::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("calibrate_qwen4exp: no GPU ({e:?}) — skipped");
            return;
        }
    };
    let t0 = Instant::now();
    let max_seq = (n_tok + 8).next_power_of_two().max(256);
    let mut m = Qwen4ExpBackend::load(&mut gpu, &mut hfq, max_seq).expect("load");
    let cfg = m.config().clone();
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Arm the tap: buffer pointer -> canonical artifact name. The names must
    // match the ARTIFACT's, because that is what `--hessian` looks up.
    // `max_k` caps the [K,K] device accumulator, whose cost is quadratic in K.
    // Uncapped by default: the capture set (no GDN — see `capture_map`) fits in
    // ~22 GB, and the K=10240 hyper-connection `mix_down` Hessians it includes
    // ARE consumed by LDLQ. Lower it only if a larger capture set OOMs.
    let max_k: usize = std::env::var("HIPFIRE_CALIB_MAX_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let map = m.weights().capture_map(&cfg, Some(max_k));
    println!(
        "capture targets: {} dense linears (max_k={max_k})",
        map.len()
    );
    for (ptr, name) in &map {
        gpu.capture_names.insert(*ptr, name.clone());
    }
    let collector = Arc::new(CalibCollector::with_imatrix_only(vec![
        "experts.".to_string()
    ]));
    gpu.active_capture = Some(collector.clone());

    // Drive the forward over REAL corpus tokens, teacher-forced. Calibration
    // wants the activation distribution the model sees on real text; letting it
    // generate its own continuation by argmax instead gives a repeating cycle,
    // and a Hessian built from a cycle is rank-deficient by construction.
    let text = std::fs::read_to_string(&corpus).expect("read corpus");
    let toks: Vec<u32> = match Tokenizer::from_hfq_metadata(&hfq.metadata_json) {
        Ok(tk) => tk.encode(&text),
        Err(e) => {
            // No tokenizer in the artifact: fall back to whitespace-separated ids.
            let ids: Vec<u32> = text
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            assert!(
                !ids.is_empty(),
                "corpus has no tokenizer ({e:?}) and does not parse as token ids"
            );
            ids
        }
    };
    assert!(
        toks.len() >= 2,
        "corpus tokenized to {} tokens; calibration needs real text",
        toks.len()
    );
    // STRIDE, do not take a prefix. A multilingual corpus is concatenated by
    // language, so the first n tokens are one language and the captured
    // statistics are a lie about the mix. Walk contiguous chunks spread evenly
    // across the whole file instead -- contiguous so activations stay coherent
    // within a chunk, spread so every section is represented.
    const CHUNK: usize = 512;
    let n_use = n_tok.min(toks.len());
    if n_use < n_tok {
        eprintln!("corpus holds only {n_use} tokens (asked for {n_tok}) -- Hessians for K>{n_use} stay rank-deficient");
    }
    let toks: Vec<u32> = if n_use >= toks.len() {
        toks
    } else {
        let chunks = n_use.div_ceil(CHUNK);
        let stride = (toks.len() - CHUNK) / chunks.max(1);
        (0..chunks)
            .flat_map(|c| {
                let start = c * stride;
                toks[start..(start + CHUNK).min(toks.len())].to_vec()
            })
            .take(n_use)
            .collect()
    };
    let toks = &toks[..];

    let distinct = {
        let mut v = toks.to_vec();
        v.sort_unstable();
        v.dedup();
        v.len()
    };

    let t1 = Instant::now();
    m.prefill(&mut gpu, &toks[..1]).expect("prefill");
    for i in 1..toks.len() {
        m.decode_step(&mut gpu, toks[i - 1], i).expect("decode");
    }
    println!(
        "captured over {n_use} real tokens ({distinct} distinct) in {:.1}s",
        t1.elapsed().as_secs_f32()
    );

    // Disarm before writing so the write itself is not captured.
    gpu.active_capture = None;
    println!("accumulators: {}", collector.len());
    if collector.is_empty() {
        eprintln!(
            "calibrate_qwen4exp: NOTHING was captured. The tap fires only for\n\
             weights whose buffer pointer is in `capture_names` AND whose dtype\n\
             routes through the weight_gemv wrapper — check both before trusting\n\
             an empty package."
        );
        std::process::exit(2);
    }

    let meta = serde_json::json!({
        "producer": "calibrate_qwen4exp",
        "arch_id": 26,
        "tokens": n_use,
        "imatrix_only": ["experts."],
    })
    .to_string();
    let t2 = Instant::now();
    // `write_streaming`'s return is NOT the package size — printing it as MB
    // reported "0.0 MB" for a 21 GB package, which reads as a failed write.
    // Stat the file instead.
    collector
        .write_streaming(&mut gpu, Path::new(&out), 26, &meta, &[])
        .expect("write calib package");
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "wrote {out} ({:.2} GiB) in {:.1}s",
        bytes as f64 / (1u64 << 30) as f64,
        t2.elapsed().as_secs_f32()
    );
    if bytes == 0 {
        eprintln!("calibrate_qwen4exp: package is EMPTY — refusing to call this a success");
        std::process::exit(2);
    }
}
