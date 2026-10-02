// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Timing of the routed batched KVarN attention on the 27B's shape (24 q / 4 kv
//! heads, head_dim 256), zero-filled caches — the measurement behind
//! KVARN_ROUTED_CHUNK and the GQA-grouped kernel choice.
//!
//!   cargo run --release -p hipfire-rdna --example bench_kvarn_routed -- <ctx> <rows> <mode> [sessions]
//!   mode 0 = the dispatch, 1 = chunked per query head, 2 = GQA-grouped; CHUNK=<n> overrides 256 (modes 1-2)
use hipfire_rdna::Gpu;
use std::time::Instant;
const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const G: usize = 128;
fn main() {
    let mut gpu = Gpu::init().unwrap();
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (ctx, rows, mode) = (args[0], args[1], args[2]);
    let rec_bytes = (HD * G).div_ceil(2) + HD * 4 + G * 2;
    let nb = ctx.div_ceil(G);
    let z = |gpu: &mut Gpu, n: usize| {
        gpu.upload_raw(
            &vec![0u8; n.next_multiple_of(4)],
            &[n.next_multiple_of(4) / 4],
        )
        .unwrap()
    };
    let ns = args.get(3).copied().unwrap_or(1);
    let bufs: Vec<_> = (0..ns)
        .map(|_| {
            (
                z(&mut gpu, nb * NKV * rec_bytes),
                z(&mut gpu, G * NKV * HD * 4),
                z(&mut gpu, ctx * NKV * (HD / 32) * 34),
            )
        })
        .collect();
    let p = |t: &hipfire_rdna::GpuTensor| (t.buf.as_ptr() as usize as u64).to_le_bytes();
    let recp = gpu
        .upload_raw(
            &bufs.iter().flat_map(|b| p(&b.0)).collect::<Vec<_>>(),
            &[2 * ns],
        )
        .unwrap();
    let winp = gpu
        .upload_raw(
            &bufs.iter().flat_map(|b| p(&b.1)).collect::<Vec<_>>(),
            &[2 * ns],
        )
        .unwrap();
    let vp = gpu
        .upload_raw(
            &bufs.iter().flat_map(|b| p(&b.2)).collect::<Vec<_>>(),
            &[2 * ns],
        )
        .unwrap();
    let q = z(&mut gpu, rows * NH * HD * 4);
    let out = z(&mut gpu, rows * NH * HD * 4);
    let rsi = gpu
        .upload_raw(
            &(0..rows)
                .flat_map(|r| ((r % ns) as i32).to_le_bytes())
                .collect::<Vec<_>>(),
            &[rows],
        )
        .unwrap();
    let pos: Vec<u8> = (0..rows)
        .flat_map(|r| ((ctx - rows + r) as i32).to_le_bytes())
        .collect();
    let posd = gpu.upload_raw(&pos, &[rows]).unwrap();
    let run = |gpu: &mut Gpu| {
        if mode == 0 {
            gpu.attention_kvarn_routed_batched(
                false, &q, &recp, &winp, &vp, &out, &rsi, &posd, 1, 0, NH, NKV, HD, ctx, ctx, rows,
                4, 0,
            )
            .unwrap();
        } else {
            gpu.attention_kvarn_routed_batched_chunked(
                false,
                &q,
                &recp,
                &winp,
                &vp,
                &out,
                &rsi,
                &posd,
                1,
                0,
                NH,
                NKV,
                HD,
                ctx,
                rows,
                4,
                0,
                std::env::var("CHUNK")
                    .map(|c| c.parse().unwrap())
                    .unwrap_or(256),
                mode == 2,
            )
            .unwrap();
        }
    };
    run(&mut gpu);
    gpu.device_synchronize().unwrap();
    let t = Instant::now();
    let n = 5;
    for _ in 0..n {
        run(&mut gpu);
    }
    gpu.device_synchronize().unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("ctx={ctx} rows={rows} sessions={ns} mode={mode}: {ms:.2} ms/layer  (x16 layers = {:.0} ms)", ms * 16.0);
}
