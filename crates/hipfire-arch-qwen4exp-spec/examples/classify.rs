// SPDX-License-Identifier: Apache-2.0
// hipfire — classify a checkpoint's tensor names through this arch's Ingest policy.
//
// Reads newline-separated tensor names on stdin and prints, per name, the role /
// importance / precision class the quantizer will see. The point is to run it over
// a REAL checkpoint's full name list before converting anything: the classification
// is pure string matching, so every mistake it can make is visible here, for free,
// without touching a single tensor byte.
//
//   python3 -c "import json;[print(k) for k in json.load(open('shapes.json'))]" \
//     | cargo run -p hipfire-arch-qwen4exp-spec --example classify -- --summary
use hipfire_arch_api::Ingest;
use hipfire_arch_qwen4exp_spec::Qwen4ExpSpec;
use std::collections::BTreeMap;
use std::io::Read;

fn main() {
    let summary = std::env::args().any(|a| a == "--summary");
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .expect("read stdin");
    let spec = Qwen4ExpSpec;
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    for name in buf.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let key = format!(
            "{:?} / imp {:3} / {:?}",
            spec.role(name),
            spec.importance(name),
            spec.precision_class(name)
        );
        if summary {
            *tally.entry(key).or_default() += 1;
        } else {
            println!("{key}  {name}");
        }
    }
    for (k, n) in &tally {
        println!("{n:5}  {k}");
    }
}
