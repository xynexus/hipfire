#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// hipfire — golden-output runner for tiny gating fixtures.
//
// Loads a tiny arch-5/6 .hfq (from `hipfire-quantize --emit-fixture`), runs the
// forward, and emits a deterministic golden: the per-position argmax token
// sequence + a hash of the logits. Run twice / across builds and diff — a
// drift in the argmax line is the tripwire (escalate to the 35B golden).
//
// Two modes (--mode):
//   tf  (teacher-forced, default) — feed a FIXED raw token-ID stream at every
//        position. Stable, but the inputs never depend on the model's own
//        output, so it CANNOT surface an attractor. This is the prefill-shaped
//        golden the existing gate (tests/fixture-golden-gate.sh) pins.
//   ar  (free-running greedy) — feed a short fixed prompt, then feed the model's
//        own argmax back as the next token for the rest of `--len`, growing the
//        KV cache. Exercises the real autoregressive decode loop and can reach
//        an attractor (per docs/plans/2026-06-20-tiny-golden-tripwire.md, D1).
//
#![allow(clippy::needless_range_loop)]

// Usage:
//   fixture_golden <model.hfq> [--mode tf|ar] [--len 32] [--warmup 2]
//                  [--prompt-len 4] [--seed 1]

use hipfire_arch_api::{
    ARCH_ID_DEEPSEEK4_FLASH, ARCH_ID_DOTS_OCR, ARCH_ID_GEMMA3_TEXT, ARCH_ID_GEMMA3_VL,
    ARCH_ID_GEMMA4, ARCH_ID_LFM2_MOE, ARCH_ID_LLAMA_MISTRAL, ARCH_ID_MAMBA2, ARCH_ID_MINIMAX_M2,
    ARCH_ID_NEMOTRON_H, ARCH_ID_QWEN2, ARCH_ID_QWEN35_DENSE, ARCH_ID_QWEN3_QWEN2_LEGACY,
    ARCH_ID_ZAYA,
};
use hipfire_arch_deepseek4::{forward as deepseek4_forward, DeepseekV4, DeepseekV4State};
use hipfire_arch_dots_ocr::dots_ocr::DotsOcrConfig;
use hipfire_arch_gemma3::{forward_step as gemma3_forward_step, Gemma3, Gemma3State};
use hipfire_arch_gemma3_vl::{load_vl, Gemma3VlBackend};
use hipfire_arch_gemma4::{forward_step, logits as gemma4_logits, lower_dense_forward, Gemma4};
use hipfire_arch_lfm2moe::{
    decode_step as lfm2_decode_step, Lfm2MoeConfig, Lfm2MoeState, Lfm2MoeWeights,
};
use hipfire_arch_llama::{Llama, LlamaBackend};
use hipfire_arch_minimax::{
    forward as minimax_forward, MiniMaxConfig, MiniMaxState, MiniMaxWeights,
};
use hipfire_arch_nemotron::{model::NemotronModel, NemotronHConfig};
use hipfire_arch_qwen2::{qwen2::forward_step as qwen2_forward_step, Qwen2};
use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_arch_zaya::{arch::ZayaModel, ZayaConfig};
use hipfire_runtime::arch::{Architecture, SimpleAr};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use std::path::Path;

/// splitmix64 — same generator as the fixture emitter, for a reproducible
/// fixed token stream independent of any tokenizer.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn update_hash_and_argmax(logits: &[f32], pos: usize, warmup: usize, hash: &mut u64) -> u32 {
    let mut best = (f32::NEG_INFINITY, 0u32);
    for (i, &v) in logits.iter().enumerate() {
        if pos >= warmup {
            for b in v.to_bits().to_le_bytes() {
                *hash ^= b as u64;
                *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        if v > best.0 {
            best = (v, i as u32);
        }
    }
    best.1
}

fn qwen35_vl_synthetic_patches(config: &qwen35_vl::VisionConfig) -> Vec<f32> {
    let grid_h = config.spatial_merge_size;
    let grid_w = config.spatial_merge_size;
    let patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
    let n = grid_h * grid_w * patch_dim;
    (0..n)
        .map(|i| {
            let x = ((i as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                & 0xffff) as f32
                / 65_535.0;
            x * 0.2 - 0.1
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: fixture_golden <model.hfq> [--len N] [--warmup N] [--seed N]");

    let mut len: usize = 32;
    let mut warmup: usize = 2;
    let mut seed: u64 = 1;
    let mut mode = String::from("tf");
    let mut prompt_len: usize = 4;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--len" => len = val.parse().unwrap(),
            "--warmup" => warmup = val.parse().unwrap(),
            "--seed" => seed = val.parse().unwrap(),
            "--mode" => mode = val,
            "--prompt-len" => prompt_len = val.parse().unwrap(),
            _ => panic!("unknown flag: {flag}"),
        }
    }
    assert!(len > warmup + 1, "len must exceed warmup");
    let ar = match mode.as_str() {
        "tf" => false,
        "ar" => true,
        other => panic!("unknown --mode {other:?} (expected tf|ar)"),
    };
    if ar {
        assert!(
            prompt_len >= 1 && prompt_len < len,
            "need 1 <= prompt-len < len"
        );
    }

    // Fixed token stream in a small range valid for any fixture vocab. In `tf`
    // mode this is the full forced input; in `ar` mode only the first
    // `prompt-len` entries are used (as the prompt) and the rest are replaced by
    // the model's own argmax, fed back each step.
    let mut st = seed ^ 0x5DEE_CE66_D8A1_0001;
    let tokens: Vec<u32> = (0..len).map(|_| (splitmix(&mut st) % 100) as u32).collect();

    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    if hfq.arch_id == ARCH_ID_QWEN35_DENSE && qwen35_vl::vision_config_from_hfq(&hfq).is_some() {
        run_qwen35_vl_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        );
        return;
    }
    match hfq.arch_id {
        ARCH_ID_LLAMA_MISTRAL | ARCH_ID_QWEN3_QWEN2_LEGACY => run_llama_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_QWEN2 => run_qwen2_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_DOTS_OCR => run_dots_ocr_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_DEEPSEEK4_FLASH => run_deepseek4_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_GEMMA3_TEXT => run_gemma3_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_GEMMA3_VL => run_gemma3_vl_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_GEMMA4 => run_gemma4_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_LFM2_MOE => run_lfm2_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_MINIMAX_M2 => run_minimax_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_NEMOTRON_H => run_nemotron_h_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_MAMBA2 => run_mamba2_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        ARCH_ID_ZAYA => run_zaya_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
        _ => run_qwen35_golden(
            model_path, hfq, len, warmup, seed, mode, ar, prompt_len, tokens,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_qwen35_vl_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let vision_config = qwen35_vl::vision_config_from_hfq(&hfq).expect("vision config");

    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let vision_weights =
        qwen35_vl::load_vision_weights(&hfq, &vision_config, &mut gpu).expect("vision weights");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    let kv_max = len + 16;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_max,
    )
    .unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();
    let visual_tokens = qwen35_vl::vision_forward(
        &mut gpu,
        &vision_weights,
        &vision_config,
        &qwen35_vl_synthetic_patches(&vision_config),
        vision_config.spatial_merge_size,
        vision_config.spatial_merge_size,
    )
    .expect("vision forward");
    assert_eq!(
        visual_tokens.len(),
        config.dim,
        "fixture golden expects one synthetic Qwen3.5-VL visual token"
    );

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        if pos == 0 {
            qwen35::forward_scratch_embed(
                &mut gpu,
                &weights,
                &config,
                &visual_tokens,
                pos,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
            )
            .expect("forward embed");
        } else {
            qwen35::forward_scratch(
                &mut gpu,
                &weights,
                &config,
                tok,
                pos,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
            )
            .expect("forward");
        }

        let logits = gpu.download_f32(&scratch.logits).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_qwen35_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = qwen35::config_from_hfq(&hfq).expect("config");

    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    let kv_max = len + 16;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_max,
    )
    .unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();

    // FNV-1a over the logit bits — sensitive (byte-exact) golden; the argmax
    // line is the robust golden. Print both.
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    // In `ar` mode the input after the prompt is the previous step's argmax.
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        // tf: always the forced stream. ar: prompt prefix, then own argmax.
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            tok,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("forward");

        let logits = gpu.download_f32(&scratch.logits).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_deepseek4_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = DeepseekV4::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = DeepseekV4::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
    let mut state = DeepseekV4State::new(&config).expect("state");

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        let logits = deepseek4_forward::decode_step(
            &config, &weights, &mut state, &mut gpu, tok, pos as u32,
        )
        .expect("forward");
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_gemma4_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = Gemma4::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = Gemma4::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
    let mut state = Gemma4::new_state(&mut gpu, &config).expect("state");
    let lowered = lower_dense_forward(&config, &state);

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        forward_step(&mut gpu, &weights, &config, &mut state, &lowered, tok).expect("forward");
        let logits = gemma4_logits(&gpu, &state).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_gemma3_vl_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let loaded = load_vl(&mut hfq, &mut gpu).expect("load_vl");
    let state = Gemma3State::new_with_max_seq(
        &mut gpu,
        &loaded.text_cfg,
        len + 16,
        hipfire_runtime::kv::KvQuantMode::Unquantized,
        4,
    )
    .expect("state");
    let mut backend = Gemma3VlBackend::new(
        loaded.text_cfg,
        loaded.vl_cfg,
        loaded.weights,
        state,
        loaded.vision_tier,
        loaded.vision_source_id,
    );

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        backend.decode_step(&mut gpu, tok, pos).expect("forward");
        let logits = gpu.download_f32(backend.logits()).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_lfm2_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = Lfm2MoeWeights::load(&mut hfq, &config, &mut gpu).expect("load_weights");
    let kv_max = len + 16;
    let mut state =
        Lfm2MoeState::new_with_physical_cap(&mut gpu, &config, kv_max, kv_max).expect("state");

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        let logits = lfm2_decode_step(&config, &weights, &mut state, &mut gpu, tok, pos as u32)
            .expect("forward");
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_minimax_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = MiniMaxConfig::from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = MiniMaxWeights::load(&mut hfq, &config, &mut gpu, None).expect("load_weights");
    let mut state = MiniMaxState::new_with_max_seq(&mut gpu, &config, len + 16).expect("state");

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        let logits =
            minimax_forward::decode_step(&config, &weights, &mut state, &mut gpu, tok, pos as u32)
                .expect("forward");
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_llama_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = Llama::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = Llama::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
    let kv_max = len + 16;
    let kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_max,
    )
    .expect("kv_cache");
    let scratch = Llama::new_state(&mut gpu, &config).expect("state");
    let mut backend = LlamaBackend::new(hfq.arch_id, config, weights, scratch, kv_cache);

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        backend.decode_step(&mut gpu, tok, pos).expect("forward");
        let logits = gpu.download_f32(backend.logits()).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_nemotron_h_golden(
    model_path: String,
    hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
    let cfg_json = meta
        .get("config")
        .expect("nemotron_h: metadata_json missing 'config'");
    let config = NemotronHConfig::from_json(cfg_json).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let max_seq = len + 16;
    let mut model = NemotronModel::from_hfq(&mut gpu, &hfq, config, max_seq).expect("load_weights");

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        model.decode_step(&mut gpu, tok, pos).expect("forward");
        let logits = gpu.download_f32(model.logits()).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_mamba2_golden(
    model_path: String,
    hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
    let cfg_json = meta
        .get("config")
        .expect("mamba2: metadata_json missing 'config'");
    let config = NemotronHConfig::from_mamba2_json(cfg_json).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let max_seq = len + 16;
    let mut model = NemotronModel::from_hfq(&mut gpu, &hfq, config, max_seq).expect("load_weights");

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        model.decode_step(&mut gpu, tok, pos).expect("forward");
        let logits = gpu.download_f32(model.logits()).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_zaya_golden(
    model_path: String,
    hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
    let cfg_json = meta
        .get("config")
        .expect("zaya: metadata_json missing 'config'");
    let config = ZayaConfig::from_json(cfg_json).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let max_seq = len + 16;
    let mut model = ZayaModel::from_hfq(&mut gpu, &hfq, config, max_seq).expect("load_weights");

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        model.decode_step(&mut gpu, tok, pos).expect("forward");
        let logits = gpu.download_f32(model.logits()).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_qwen2_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = Qwen2::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = Qwen2::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
    let mut state = Qwen2::new_state(&mut gpu, &config).expect("state");

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        qwen2_forward_step(&mut gpu, &weights, &config, &mut state, tok).expect("forward");
        let logits = gpu.download_f32(&state.logits).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_dots_ocr_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = DotsOcrConfig::from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = Qwen2::load_weights(&mut hfq, &config.text, &mut gpu).expect("load_weights");
    let mut state = Qwen2::new_state(&mut gpu, &config.text).expect("state");

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        qwen2_forward_step(&mut gpu, &weights, &config.text, &mut state, tok).expect("forward");
        let logits = gpu.download_f32(&state.logits).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}

#[allow(clippy::too_many_arguments)]
fn run_gemma3_golden(
    model_path: String,
    mut hfq: HfqFile,
    len: usize,
    warmup: usize,
    seed: u64,
    mode: String,
    ar: bool,
    prompt_len: usize,
    tokens: Vec<u32>,
) {
    let config = Gemma3::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = Gemma3::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
    let mut state = Gemma3::new_state(&mut gpu, &config).expect("state");

    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut argmax_seq: Vec<u32> = Vec::new();
    let mut last_argmax: u32 = 0;

    for pos in 0..len {
        let tok = if ar && pos >= prompt_len {
            last_argmax
        } else {
            tokens[pos]
        };

        gemma3_forward_step(&mut gpu, &weights, &config, &mut state, tok).expect("forward");
        let logits = gpu.download_f32(&state.logits).unwrap();
        last_argmax = update_hash_and_argmax(&logits, pos, warmup, &mut hash);
        if pos >= warmup {
            argmax_seq.push(last_argmax);
        }
    }

    println!("model:     {model_path}");
    println!("mode:      {mode}");
    println!("tokens:    len={len} warmup={warmup} seed={seed}");
    println!("argmax:    {argmax_seq:?}");
    println!("logit_hash: {hash:#018x}");
}
