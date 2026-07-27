// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU smoke for heterogeneous HFQM CASK eviction.
//!
//! Exercises a sliding layer plus two differently-shaped full-context layers,
//! including both half-split and interleaved RoPE center layouts. Two eviction
//! cycles prove that post-compaction writes use the translated physical slot.

use hipfire_rdna::Gpu;
use hipfire_runtime::layered_kv::{KvStorageKind, LayerKvSpec, LayeredKvArena, LayeredKvPlan};
use hipfire_runtime::triattn::{
    BandCenter, LayeredEvictionCtx, TriAttnArtifact, TriAttnAttentionKind, TriAttnContextPolicy,
    TriAttnLayerRecord, TriAttnPackageMetadata, TriAttnRopeConvention, TRIATTN_ARTIFACT_KIND,
    TRIATTN_HFQM_SCHEMA,
};

fn record(
    layer: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    convention: TriAttnRopeConvention,
    context: TriAttnContextPolicy,
) -> TriAttnLayerRecord {
    let sliding = context == TriAttnContextPolicy::Sliding;
    TriAttnLayerRecord {
        physical_layer: layer,
        attention_kind: if sliding {
            TriAttnAttentionKind::Sliding
        } else {
            TriAttnAttentionKind::Full
        },
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim: head_dim,
        rope_theta: 10_000.0,
        rope_convention: convention,
        context_policy: context,
        sliding_window: sliding.then_some(4),
        kv_producer: None,
        center_tensor: format!("triattn.layer.{layer}.centers"),
        center_offset: 0,
        center_count: (q_heads * head_dim / 2) as u64,
        sample_count: 8,
    }
}

fn centers(count: usize, seed: f32) -> Vec<BandCenter> {
    (0..count)
        .map(|index| BandCenter {
            eq_re: seed + index as f32 * 0.001,
            eq_im: seed * 0.5 - index as f32 * 0.0005,
            e_abs_q: 0.5 + seed.abs(),
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("initialize GPU");
    let plan = LayeredKvPlan::build(
        32,
        vec![
            LayerKvSpec::owned(4, 2, 4, KvStorageKind::SlidingWindow { window: 4 }),
            LayerKvSpec::owned(4, 2, 4, KvStorageKind::Full),
            LayerKvSpec::owned(2, 1, 8, KvStorageKind::Full),
        ],
    )
    .expect("build heterogeneous plan");
    let mut arena =
        LayeredKvArena::new_fp32_capped(&mut gpu, plan, 6).expect("allocate capped layered KV");
    let records = vec![
        record(
            0,
            4,
            2,
            4,
            TriAttnRopeConvention::Interleaved,
            TriAttnContextPolicy::Sliding,
        ),
        record(
            1,
            4,
            2,
            4,
            TriAttnRopeConvention::HalfSplit,
            TriAttnContextPolicy::Full,
        ),
        record(
            2,
            2,
            1,
            8,
            TriAttnRopeConvention::Interleaved,
            TriAttnContextPolicy::Full,
        ),
    ];
    let artifact = TriAttnArtifact {
        metadata: TriAttnPackageMetadata {
            artifact_kind: TRIATTN_ARTIFACT_KIND.into(),
            package_schema: TRIATTN_HFQM_SCHEMA.into(),
            model_arch_id: 24,
            model_layers: 3,
            model_fingerprint: "synthetic-model".into(),
            corpus_fingerprint: "synthetic-corpus".into(),
            adapter: "layered-cask-gpu-smoke".into(),
            engine: "hipfire".into(),
            layers: records.clone(),
        },
        centers: records
            .iter()
            .enumerate()
            .map(|(index, record)| centers(record.center_count as usize, 0.1 * (index + 1) as f32))
            .collect(),
    };
    artifact.validate().expect("validate synthetic CASK");
    let eviction =
        LayeredEvictionCtx::new(&mut gpu, &artifact, &arena, 4, 2).expect("build eviction");

    for logical_pos in 0..8 {
        for layer in 0..3 {
            let width = arena.plan().layers()[layer].kv_width();
            let k = (0..width)
                .map(|dim| logical_pos as f32 * 0.1 + layer as f32 + dim as f32 * 0.01)
                .collect::<Vec<_>>();
            let v = k.iter().map(|value| -*value).collect::<Vec<_>>();
            arena
                .store_f32(&gpu, layer, logical_pos, &k, &v)
                .expect("store layered KV row");
        }
        arena.advance(logical_pos).expect("advance logical cursor");
        let result = eviction
            .maybe_evict(&mut gpu, &mut arena)
            .expect("run layered eviction");
        if logical_pos == 5 || logical_pos == 7 {
            let result = result.expect("expected eviction at cap");
            assert_eq!(result.new_physical, 4);
            assert_eq!(result.retain_mask.len(), 4);
            assert_eq!(arena.full_physical_len().unwrap(), 4);
        } else {
            assert!(result.is_none());
        }
    }
    assert_eq!(eviction.eviction_count.get(), 2);
    assert_eq!(arena.view(1, 8).unwrap().physical_position, 4);
    assert_eq!(arena.view(2, 8).unwrap().physical_position, 4);
    eprintln!("layered CASK GPU smoke passed: 2 heterogeneous eviction cycles");

    eviction.free_gpu(&mut gpu);
    arena.free_gpu(&mut gpu);
}
