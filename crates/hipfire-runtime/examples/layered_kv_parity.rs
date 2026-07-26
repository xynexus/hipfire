// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Locked GPU parity probe for mixed local/global/shared layered F32 KV.

use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::layered_kv::{
    KvStorageKind, LayerKvSpec, LayeredAttentionScratch, LayeredKvArena, LayeredKvPlan,
};

fn assert_row(gpu: &Gpu, tensor: &GpuTensor, pos: usize, width: usize, expected: &[f32]) {
    let host = gpu.download_f32(tensor).expect("download cache");
    let start = pos * width;
    assert_eq!(&host[start..start + width], expected);
}

fn assert_swa_ring(gpu: &Gpu, tensor: &GpuTensor, slot: usize, window: usize, expected: &[f32]) {
    let host = gpu.download_f32(tensor).expect("download SWA ring");
    for (channel, value) in expected.iter().enumerate() {
        assert_eq!(host[channel * window + slot], *value);
    }
}

fn main() {
    const MAX_SEQ: usize = 16;
    const WINDOW: usize = 8;
    let plan = LayeredKvPlan::build(
        MAX_SEQ,
        vec![
            LayerKvSpec::owned(4, 2, 256, KvStorageKind::SlidingWindow { window: WINDOW }),
            LayerKvSpec::owned(8, 1, 512, KvStorageKind::Full),
            LayerKvSpec::owned(4, 2, 256, KvStorageKind::SlidingWindow { window: WINDOW })
                .shared(0),
            LayerKvSpec::owned(8, 1, 512, KvStorageKind::Full).shared(1),
        ],
    )
    .expect("mixed layered KV plan");
    assert_eq!(plan.physical_owned_layers(), 2);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let mut arena = LayeredKvArena::new_fp32(&mut gpu, plan.clone()).expect("allocate arena");
    let scratch = LayeredAttentionScratch::new(&mut gpu, &plan).expect("allocate scratch");
    assert_eq!(scratch.view(&plan, 0).unwrap().q.numel(), 4 * 256);
    assert_eq!(scratch.view(&plan, 1).unwrap().q.numel(), 8 * 512);

    let width = 512;
    let mut local_k = vec![vec![0.0f32; width]; WINDOW];
    let mut local_v = vec![vec![0.0f32; width]; WINDOW];
    let mut global_k = vec![vec![0.0f32; width]; MAX_SEQ];
    let mut global_v = vec![vec![0.0f32; width]; MAX_SEQ];

    for pos in 0..=WINDOW + 1 {
        let lk = vec![pos as f32 + 0.25; width];
        let lv = vec![pos as f32 + 0.50; width];
        let gk = vec![pos as f32 + 10.25; width];
        let gv = vec![pos as f32 + 10.50; width];
        arena
            .store_f32(&gpu, 0, pos, &lk, &lv)
            .expect("store local KV");
        arena
            .store_f32(&gpu, 1, pos, &gk, &gv)
            .expect("store global KV");
        local_k[pos % WINDOW] = lk;
        local_v[pos % WINDOW] = lv;
        global_k[pos] = gk;
        global_v[pos] = gv;
        arena.advance(pos).expect("contiguous growth");
    }

    for absolute in [WINDOW - 1, WINDOW, WINDOW + 1] {
        let local = arena.view(0, absolute).unwrap();
        assert_eq!(local.physical_position, absolute % WINDOW);
        assert_swa_ring(
            &gpu,
            local.k,
            local.physical_position,
            WINDOW,
            &local_k[local.physical_position],
        );
        assert_swa_ring(
            &gpu,
            local.v,
            local.physical_position,
            WINDOW,
            &local_v[local.physical_position],
        );

        let global = arena.view(1, absolute).unwrap();
        assert_eq!(global.physical_position, absolute);
        assert_row(&gpu, global.k, absolute, width, &global_k[absolute]);
        assert_row(&gpu, global.v, absolute, width, &global_v[absolute]);
    }

    let local_shared = arena.view(2, WINDOW + 1).unwrap();
    let global_shared = arena.view(3, WINDOW + 1).unwrap();
    assert_eq!(local_shared.producer_layer, 0);
    assert_eq!(global_shared.producer_layer, 1);
    assert_eq!(
        local_shared.k.buf.as_ptr(),
        arena.view(0, WINDOW + 1).unwrap().k.buf.as_ptr()
    );
    assert_eq!(
        global_shared.v.buf.as_ptr(),
        arena.view(1, WINDOW + 1).unwrap().v.buf.as_ptr()
    );

    arena.reset();
    assert_eq!(arena.next_pos(), 0);
    let second_k = vec![101.25f32; width];
    let second_v = vec![101.50f32; width];
    arena
        .store_f32(&gpu, 0, 0, &second_k, &second_v)
        .expect("second request local write");
    arena.advance(0).expect("second request growth");
    let second = arena.view(0, 0).unwrap();
    assert_swa_ring(&gpu, second.k, 0, WINDOW, &second_k);
    assert_swa_ring(&gpu, second.v, 0, WINDOW, &second_v);

    println!(
        "layered_kv_parity: PASS (owned={} logical={} bytes={})",
        plan.physical_owned_layers(),
        plan.layers().len(),
        plan.allocation_bytes()
    );
    scratch.free_gpu(&mut gpu);
    arena.free_gpu(&mut gpu);
    gpu.drain_pool();
}
