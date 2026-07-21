// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — issue #18 determinism repro (INSTRUMENTATION, do not commit).
//
// Runs the batched FullAttention dispatch used by the DFlash verify forward
// against fixed synthetic Q/K/V/positions and checks bit-identity of the
// output across many in-process repeats. Print a hash so separate process
// invocations can be compared too.

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

const NH: usize = 16;
const NKV: usize = 2;
const HD: usize = 128;
const MAXS: usize = 1024;
const KVDIM: usize = NKV * HD;
const B: usize = 12; // verify batch (n > 1)

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let mut gpu = Gpu::init().expect("gpu init");

    let q = lcg(0xa5a5, B * NH * HD);
    let mut kflat = vec![0f32; MAXS * KVDIM];
    let mut vflat = vec![0f32; MAXS * KVDIM];
    for t in 0..MAXS {
        let k = lcg(0x1000 + t as u64, KVDIM);
        let v = lcg(0x9000 + t as u64, KVDIM);
        kflat[t * KVDIM..(t + 1) * KVDIM].copy_from_slice(&k);
        vflat[t * KVDIM..(t + 1) * KVDIM].copy_from_slice(&v);
    }
    // Verify-shaped positions: a contiguous block starting at start_pos.
    let start_pos = 400usize;
    let positions: Vec<i32> = (0..B).map(|i| (start_pos + i) as i32).collect();
    let max_ctx_len = start_pos + B;

    let d_q = gpu.upload_f32(&q, &[B * NH * HD]).unwrap();
    let d_k = gpu.upload_f32(&kflat, &[MAXS * KVDIM]).unwrap();
    let d_v = gpu.upload_f32(&vflat, &[MAXS * KVDIM]).unwrap();
    let d_out = gpu.zeros(&[B * NH * HD], DType::F32).unwrap();
    let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let d_pos = gpu.alloc_tensor(&[B], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&d_pos.buf, &pos_bytes).unwrap();

    // Tree-verify bias over a [B x B] block at block_start = start_pos.
    let mut bias = vec![0f32; B * B];
    for r in 0..B {
        for c in 0..B {
            bias[r * B + c] = if c <= r { 0.0 } else { -1e30 };
        }
    }
    let d_bias = gpu.upload_f32(&bias, &[B * B]).unwrap();

    for (label, tree) in [("causal", false), ("tree", true)] {
        let mut hashes = Vec::new();
        for _ in 0..reps {
            gpu.attention_f32_batched_masked(
                &d_q,
                &d_k,
                &d_v,
                &d_out,
                &d_pos,
                NH,
                NKV,
                HD,
                MAXS,
                max_ctx_len,
                B,
                if tree { Some(&d_bias) } else { None },
                if tree { start_pos } else { 0 },
                if tree { B } else { 0 },
            )
            .unwrap();
            gpu.device_synchronize().unwrap();
            let o = gpu.download_f32(&d_out).unwrap();
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(o.as_ptr() as *const u8, o.len() * 4) };
            hashes.push(fnv(bytes));
        }
        let stable = hashes.iter().all(|h| *h == hashes[0]);
        let uniq: std::collections::BTreeSet<_> = hashes.iter().collect();
        println!(
            "f32_batched_masked[{label}] reps={reps} stable={stable} uniq={} hash={:016x}",
            uniq.len(),
            hashes[0]
        );
    }
}
