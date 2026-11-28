//! Greedy decode through the per-token decode path, dumping `token margin` per
//! step (margin = top-1 minus top-2 logit), for A/B'ing changes that act on
//! every decode step. The batched PPL scorer narrows state once per LAUNCH, so
//! it cannot see that cadence; this can. It is how f16 DeltaNet state was
//! rejected (continuous-batching plan, P0b): diverged at step 16 while reruns
//! stayed bit-identical over 512.
//!
//!   greedy_qwen4exp <model.hfq> <corpus.txt> <prompt_tokens> <steps> \
//!       <kv_cache> > stream.txt
//!
//! Note the sparse-attention indexer only selects past 2051 visible tokens;
//! below that QSA is dense, so a short run cannot exercise selection.
//!
//! Compare two streams: the first index where tokens differ, and that step's
//! margin in the reference (a flip at a near-tie is not the same failure as one
//! at a wide margin).
use hipfire_arch_qwen4exp::attn_gpu::QsaKvMode;
use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::arch::SimpleAr;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn top2(l: &[f32]) -> (u32, f32) {
    let (mut a, mut va, mut vb) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > va {
            (vb, va, a) = (va, v, i);
        } else if v > vb {
            vb = v;
        }
    }
    (a as u32, va - vb)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(
        a.len() == 6,
        "usage: greedy_qwen4exp <model> <corpus> <prompt_tokens> <steps> <kv_cache>"
    );
    let (n_prompt, steps): (usize, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap());
    let mut hfq = HfqFile::open(Path::new(&a[1])).expect("open model");
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu");
    let kv = QsaKvMode::from_setting(&a[5]);
    eprintln!("kv {kv:?}");
    let max_seq = (n_prompt + steps + 8).next_power_of_two();
    let mut m = Qwen4ExpBackend::load_with_kv(&mut gpu, &mut hfq, max_seq, kv).expect("load");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let ids = tok.encode(&std::fs::read_to_string(&a[2]).expect("corpus"));
    let prompt = &ids[..n_prompt.min(ids.len())];

    m.prefill(&mut gpu, prompt).expect("prefill");
    let (mut t, mut margin) = top2(&gpu.download_f32(m.logits()).expect("logits"));
    for i in 0..steps {
        println!("{t} {margin:.4}");
        m.decode_step(&mut gpu, t, prompt.len() + i)
            .expect("decode");
        (t, margin) = top2(&gpu.download_f32(m.logits()).expect("logits"));
    }
}
