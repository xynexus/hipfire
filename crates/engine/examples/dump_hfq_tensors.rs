//! Dump every tensor name + shape + quant_type from an .hfq/.mq* file.
//! Read-only inspection — no GPU init.
//!
//! Usage: cargo run --release --example dump_hfq_tensors -- <path.mq4> [filter-substring]

use engine::hfq::HfqFile;
use std::path::Path;

fn quant_label(qt: u8) -> &'static str {
    match qt {
        0 => "Q4_F16_G64",
        1 => "F16",
        2 => "F32",
        3 => "Q8_0",
        4 => "HFQ4G256",
        5 => "HFQ4G128",
        6 => "HFQ6G256",
        7 => "MQ3",
        8 => "MQ2",
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
