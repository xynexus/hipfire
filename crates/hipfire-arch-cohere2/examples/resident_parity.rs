// SPDX-License-Identifier: Apache-2.0
//! Compare two arch-25 resident artifacts over the same teacher-forced tokens.

use hipfire_arch_cohere2::arch::COHERE2_SERVING_FACTORY;
use hipfire_rdna::Gpu;
use hipfire_runtime::arch::{ServingFactory, ServingFactoryOptions};
use hipfire_runtime::hfq::HfqFile;

fn score(gpu: &mut Gpu, path: &str, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    let mut hfq = HfqFile::open(std::path::Path::new(path))
        .map_err(|error| format!("open {path}: {error}"))?;
    let options = ServingFactoryOptions {
        max_seq: 32,
        kv_mode: "fp32",
        triattn: None,
        cask_budget: 0,
        cask_beta: 0,
        physical_cap: None,
    };
    let mut loaded = COHERE2_SERVING_FACTORY.load(&mut hfq, gpu, &options)?;
    let mut rows = Vec::new();
    loaded
        .backend
        .kld_forward()
        .expect("Cohere2 exposes KLD forward")
        .forward_chunk_scored(gpu, tokens, 0, &mut |_, logits, _| {
            rows.push(logits.to_vec())
        })?;
    loaded.backend.unload(gpu);
    Ok(rows)
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let reference = args
        .next()
        .ok_or("usage: resident_parity REFERENCE.hfq CANDIDATE.hfq")?;
    let candidate = args
        .next()
        .ok_or("usage: resident_parity REFERENCE.hfq CANDIDATE.hfq")?;
    if args.next().is_some() {
        return Err("usage: resident_parity REFERENCE.hfq CANDIDATE.hfq".into());
    }
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    let tokens = [104, 101, 108, 108, 111];
    let reference_rows = score(&mut gpu, &reference, &tokens)?;
    let candidate_rows = score(&mut gpu, &candidate, &tokens)?;
    if reference_rows.len() != candidate_rows.len() || reference_rows.is_empty() {
        return Err("Cohere2 parity produced mismatched or empty score rows".into());
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut count = 0usize;
    let mut top1_matches = 0usize;
    for (reference, candidate) in reference_rows.iter().zip(&candidate_rows) {
        if reference.len() != candidate.len()
            || reference
                .iter()
                .chain(candidate)
                .any(|value| !value.is_finite())
        {
            return Err("Cohere2 parity encountered invalid logits".into());
        }
        for (&reference, &candidate) in reference.iter().zip(candidate) {
            let delta = (reference - candidate).abs();
            max_abs = max_abs.max(delta);
            sum_abs += delta as f64;
            count += 1;
        }
        let top1 = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap()
        };
        top1_matches += usize::from(top1(reference) == top1(candidate));
    }
    eprintln!(
        "Cohere2 resident parity: rows={} max_abs={max_abs:.8} mean_abs={:.8} top1={top1_matches}/{}",
        reference_rows.len(),
        sum_abs / count as f64,
        reference_rows.len(),
    );
    Ok(())
}
