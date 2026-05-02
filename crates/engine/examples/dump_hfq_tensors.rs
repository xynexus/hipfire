//! Dump every tensor name + shape + quant_type from an .hfq/.mq* file.
//! Read-only inspection — no GPU init.
//!
//! Usage: cargo run --release --example dump_hfq_tensors -- <path.mq4> [filter-substring]

use engine::hfq::HfqFile;
use std::path::Path;

fn quant_label(qt: u8) -> &'static str {
    // Authoritative mapping is `enum QuantType` in hipfire-quantize/src/main.rs.
    // Keep this in sync when new variants land.
    match qt {
        0 => "Q4_F16_G64",
        1 => "F16",
        2 => "F32",
        3 => "Q8_F16",
        4 => "Q4K",
        5 => "Q8HFQ",
        6 => "HFQ4G256",
        7 => "HFQ4G128",
        8 => "HFQ6G256",
        9 => "HFQ2G256",
        10 => "HFQ2G128",
        11 => "HFQ3G256",
        12 => "HFQ3G128",
        13 => "MQ4G256",
        14 => "MQ8G256",
        15 => "MQ6G256",
        16 => "BF16",
        17 => "MQ3G256",
        18 => "MQ2G256",
        19 => "MG4G256",
        _ => "?",
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: dump_hfq_tensors <path> [filter]");
    let filter = args.next();

    let hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    println!("arch_id={}, n_tensors={}", hfq.arch_id, hfq.tensors().len());

    let mut shown = 0usize;
    for t in hfq.tensors() {
        if let Some(f) = &filter {
            if !t.name.contains(f) {
                continue;
            }
        }
        println!(
            "{:>10}  {:<70}  {:?}",
            quant_label(t.quant_type),
            t.name,
            t.shape
        );
        shown += 1;
    }
    if filter.is_some() {
        println!("({} matched)", shown);
    }
}
