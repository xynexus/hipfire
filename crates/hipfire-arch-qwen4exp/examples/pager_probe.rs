// SPDX-License-Identifier: Apache-2.0
// hipfire — what the weight pager would need from a qwen4_exp artifact.
use hipfire_runtime::hfq::HfqFile;
use std::collections::BTreeMap;
use std::path::Path;

fn main() {
    let p = std::env::args()
        .nth(1)
        .expect("usage: pager_probe <model.hfq>");
    let hfq = HfqFile::open(Path::new(&p)).expect("open");
    let mods = hfq.modules();
    println!("module records: {}", mods.len());
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for m in mods {
        *kinds.entry(format!("{:?}", m.kind)).or_default() += 1;
    }
    for (k, n) in &kinds {
        println!("  {k}: {n}");
    }
    // Which quant types the routed experts actually use — the pager repacks only
    // Oq4G256 / Oq8G256 / OqPlusCompact on page-in.
    let mut qt: BTreeMap<u8, usize> = BTreeMap::new();
    for t in hfq.tensors() {
        if t.name.contains(".mlp.experts.") {
            *qt.entry(t.quant_type).or_default() += 1;
        }
    }
    println!("routed-expert tensors by quant_type:");
    for (q, n) in &qt {
        println!("  qt {q}: {n}");
    }
}
