// Dump the first few BF16 values of named tensors from an .hfq (debug tool).
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let hf = HfqFile::open(Path::new(&args[1])).expect("open");
    for name in &args[2..] {
        match hf.tensor_data_vec(name) {
            Some((info, data)) => {
                let vals: Vec<f32> = data
                    .chunks_exact(2)
                    .take(6)
                    .map(|c| bf16(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                println!(
                    "{name} qt={} shape={:?} first6={:?}",
                    info.quant_type, info.shape, vals
                );
            }
            None => println!("{name} MISSING"),
        }
    }
}
