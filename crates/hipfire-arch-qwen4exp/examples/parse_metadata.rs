// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Parse a qwen4_exp config out of an HFQ artifact's metadata, read on stdin.
//!
//! Lets a shell gate check the SERVED artifact rather than the source config —
//! the quantizer is what decides whether the config survives into the `.hfq`, and
//! nothing else in the test suite would notice if it stopped.
//!
//!     hipfire inspect model.hfq --json \
//!       | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["metadata"]))' \
//!       | cargo run -p hipfire-arch-qwen4exp --example parse_metadata

use std::io::Read;

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).expect("read stdin");
    match hipfire_arch_qwen4exp::config::Qwen4ExpConfig::from_metadata_json(&s) {
        Ok(c) => {
            let sparse = c.sparse_attention_layers().count();
            println!(
                "ok: {} layers ({} sparse-attn, {} linear), hidden {}, vocab {}, \
                 {} experts top-{}, hc {}, ngram {}",
                c.layers,
                sparse,
                c.layers - sparse,
                c.hidden,
                c.vocab,
                c.moe.num_experts,
                c.moe.experts_per_tok,
                c.gated_residual.count,
                if c.ngram.is_some() { "yes" } else { "NO" },
            );
        }
        Err(e) => {
            eprintln!("FAILED to parse qwen4_exp config from metadata: {e}");
            std::process::exit(1);
        }
    }
}
