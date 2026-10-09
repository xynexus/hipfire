// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! `rotate_quantize_awq_indexed_batched` against the two-kernel chain it fuses,
//! `rotate_x_mq_awq_indexed_batched` + `quantize_act_oq8` (group 256): the int8
//! activation and its scales must match BYTE FOR BYTE -- per-expert AWQ, experts
//! without a sidecar, and a layer with no sidecar table at all. Also times both.
//!
//!   cargo run --release -p hipfire-rdna --example parity_rotate_quantize_indexed

use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::time::Instant;

const N_EXP: usize = 16;
const K_TOP: usize = 8;

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x9E37_79B9u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut fail = 0;
    for &(n, k) in &[(1usize, 2048usize), (7, 2048), (64, 512), (1024, 2048)] {
        let x: Vec<f32> = (0..n * k).map(|_| rnd() * 8.0 - 4.0).collect();
        let x_t = gpu.upload_f32(&x, &[x.len()]).unwrap();
        // Every third expert has no sidecar (null pointer).
        let awqs: Vec<Option<GpuTensor>> = (0..N_EXP)
            .map(|e| {
                (e % 3 != 2).then(|| {
                    let a: Vec<f32> = (0..k).map(|_| 0.25 + rnd() * 2.0).collect();
                    gpu.upload_f32(&a, &[k]).unwrap()
                })
            })
            .collect();
        let tab: Vec<u8> = awqs
            .iter()
            .flat_map(|a| {
                a.as_ref()
                    .map_or(0u64, |t| t.buf.as_ptr() as u64)
                    .to_le_bytes()
            })
            .collect();
        let tab_t = gpu.upload_raw(&tab, &[tab.len()]).unwrap();
        let topk: Vec<u8> = (0..n * K_TOP)
            .flat_map(|i| (((i * 7 + i / K_TOP) % N_EXP) as i32).to_le_bytes())
            .collect();
        let topk_t = gpu.upload_raw(&topk, &[topk.len()]).unwrap();
        let rows = n * K_TOP;
        let rot = gpu.alloc_tensor(&[rows * k], DType::F32).unwrap();
        let q_ref = gpu.alloc_tensor(&[rows * k], DType::Raw).unwrap();
        let s_ref = gpu.alloc_tensor(&[rows * k / 256], DType::F32).unwrap();
        let q_new = gpu.alloc_tensor(&[rows * k], DType::Raw).unwrap();
        let s_new = gpu.alloc_tensor(&[rows * k / 256], DType::F32).unwrap();
        for table in [Some(&tab_t), None] {
            let chain = |gpu: &mut Gpu| {
                gpu.rotate_x_mq_awq_indexed_batched(&x_t, table, &topk_t, &rot, k, K_TOP, n)
                    .unwrap();
                gpu.quantize_act_oq8(&rot, &q_ref, &s_ref, rows, k, 256)
                    .unwrap();
            };
            let fused = |gpu: &mut Gpu| {
                gpu.rotate_quantize_awq_indexed_batched(
                    &x_t, table, &topk_t, &q_new, &s_new, k, K_TOP, n,
                )
                .unwrap();
            };
            let mut ms = [0.0f64; 2];
            for (i, f) in [&chain as &dyn Fn(&mut Gpu), &fused]
                .into_iter()
                .enumerate()
            {
                f(&mut gpu);
                gpu.device_synchronize().unwrap();
                let t0 = Instant::now();
                for _ in 0..20 {
                    f(&mut gpu);
                }
                gpu.device_synchronize().unwrap();
                ms[i] = t0.elapsed().as_secs_f64() * 1e3 / 20.0;
            }
            let same_q = gpu.download_raw(&q_ref, rows * k).unwrap()
                == gpu.download_raw(&q_new, rows * k).unwrap();
            let same_s = gpu.download_raw(&s_ref, rows * k / 256 * 4).unwrap()
                == gpu.download_raw(&s_new, rows * k / 256 * 4).unwrap();
            let ok = same_q && same_s;
            fail += usize::from(!ok);
            println!(
                "  n={n:<5} K={k:<5} awq table {:<5}  chain {:.3} ms  fused {:.3} ms  {}",
                table.is_some(),
                ms[0],
                ms[1],
                if ok { "IDENTICAL" } else { "MISMATCH" }
            );
        }
    }
    if fail > 0 {
        eprintln!("parity_rotate_quantize_indexed: {fail} case(s) differ");
        std::process::exit(1);
    }
    println!("parity_rotate_quantize_indexed: PASS");
}
