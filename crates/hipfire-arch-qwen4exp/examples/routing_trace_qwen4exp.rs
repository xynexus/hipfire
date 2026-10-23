// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Teacher-forced routing trace for qwen4_exp: which routed experts every token
//! picks, per layer, over a set of grouped documents — the input to offline
//! expert-scheduling / cache-policy simulation without re-running the model.
//!
//!     routing_trace_qwen4exp <model.hfq> <docs.jsonl> <out_dir> [n_tokens] [top_n]
//!
//! `docs.jsonl` lines are `{"group": .., "id": .., "text": ..}`. Each document is
//! prefilled from a reset trunk and truncated to `n_tokens` (default 512);
//! shorter ones are skipped. Output, appended per document so a partial run is
//! usable (format and dependency rules: `README.md` beside the results):
//!   `routing.bin`  u16 `[doc][token][layer][k]`      experts the GPU routed to
//!   `alt_ids.bin`  u16 `[doc][token][layer][top_n]`  top_n experts by router prob
//!   `alt_p.bin`    f16 `[doc][token][layer][top_n]`  their softmax-over-all-experts prob
//!   `tokens.bin`   u32 `[doc][token]`                 input token ids
//!   `docs.jsonl`   one `{"group", "id"}` line per document
//!   `meta.json`    shape
//!
//! Not self-locking: run under `hipfire lock run`.

use hipfire_arch_qwen4exp::moe_gpu::{RoutingTrace, ROUTING_TRACE};
use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::arch::SimpleAr;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::quant::f32_to_f16;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        a.len() >= 3,
        "usage: routing_trace_qwen4exp <model.hfq> <docs.jsonl> <out_dir> [n_tokens] [top_n]"
    );
    let n_tok: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(512);
    let top_n: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(32);
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out).expect("out_dir");

    let mut hfq = HfqFile::open(Path::new(&a[0])).expect("open model");
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu");
    let t0 = Instant::now();
    let mut m = Qwen4ExpBackend::load(&mut gpu, &mut hfq, n_tok.next_power_of_two().max(256))
        .expect("load");
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");

    let (layers, k, experts, norm) = {
        let c = m.config();
        (
            c.layers,
            c.moe.experts_per_tok,
            c.moe.num_experts,
            c.moe.norm_topk_prob,
        )
    };
    assert!(
        top_n >= k && top_n <= experts,
        "top_n must be in [k, experts]"
    );
    std::fs::write(
        out.join("meta.json"),
        serde_json::json!({"layers": layers, "k": k, "experts": experts, "n_tok": n_tok,
                           "top_n": top_n, "norm_topk_prob": norm})
        .to_string(),
    )
    .expect("meta");
    let create = |n: &str| File::create(out.join(n)).expect(n);
    let (mut routing, mut alt_ids, mut alt_p, mut tokens, mut idx) = (
        create("routing.bin"),
        create("alt_ids.bin"),
        create("alt_p.bin"),
        create("tokens.bin"),
        create("docs.jsonl"),
    );
    let mut mismatched = 0usize;

    for line in std::fs::read_to_string(&a[1]).expect("docs").lines() {
        let d: serde_json::Value = serde_json::from_str(line).expect("doc json");
        let ids = tok.encode(d["text"].as_str().expect("text"));
        if ids.len() < n_tok {
            println!("skip {} ({} tokens)", d["id"], ids.len());
            continue;
        }
        let calls = n_tok * layers;
        *ROUTING_TRACE.lock().unwrap() = Some(RoutingTrace {
            ids: Vec::with_capacity(calls * k),
            logits: Vec::with_capacity(calls * experts),
        });
        let t = Instant::now();
        m.prefill(&mut gpu, &ids[..n_tok]).expect("prefill");
        let tr = ROUTING_TRACE.lock().unwrap().take().unwrap();
        assert_eq!(tr.ids.len(), calls * k, "one record per layer per token");
        assert_eq!(tr.logits.len(), calls * experts);

        let mut aid = Vec::with_capacity(calls * top_n * 2);
        let mut ap = Vec::with_capacity(calls * top_n * 2);
        let mut order: Vec<usize> = (0..experts).collect();
        for c in 0..calls {
            let lg = &tr.logits[c * experts..(c + 1) * experts];
            let max = lg.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let z: f32 = lg.iter().map(|&v| (v - max).exp()).sum();
            // Descending logit, ties to the lower index.
            order.sort_by(|&x, &y| lg[y].total_cmp(&lg[x]).then(x.cmp(&y)));
            for &e in &order[..top_n] {
                aid.extend_from_slice(&(e as u16).to_le_bytes());
                ap.extend_from_slice(&f32_to_f16((lg[e] - max).exp() / z).to_le_bytes());
            }
            // The GPU's selection must be the host's top-k as a set; ties at
            // the k-th logit are the only legitimate disagreement.
            let mut gpu_k: Vec<u16> = tr.ids[c * k..(c + 1) * k].to_vec();
            let mut host_k: Vec<u16> = order[..k].iter().map(|&e| e as u16).collect();
            gpu_k.sort_unstable();
            host_k.sort_unstable();
            mismatched += (gpu_k != host_k) as usize;
        }
        let routed: Vec<u8> = tr.ids.iter().flat_map(|e| e.to_le_bytes()).collect();
        let toks: Vec<u8> = ids[..n_tok].iter().flat_map(|t| t.to_le_bytes()).collect();
        routing.write_all(&routed).expect("routing");
        alt_ids.write_all(&aid).expect("alt_ids");
        alt_p.write_all(&ap).expect("alt_p");
        tokens.write_all(&toks).expect("tokens");
        writeln!(
            idx,
            "{}",
            serde_json::json!({"group": d["group"], "id": d["id"]})
        )
        .expect("idx");
        println!(
            "{} {}: {:.1} tok/s, host/GPU top-k mismatches so far {mismatched}",
            d["group"],
            d["id"],
            n_tok as f32 / t.elapsed().as_secs_f32()
        );
    }
}
