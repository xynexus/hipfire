// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! How close to peak can we get PULLING from RAM, with no work attached?
//!
//! `bench_dram_bw` measures ops that read AND write (`copy_d2d`, `add_inplace`).
//! That is the right roofline input for a kernel that stores results, but it is
//! not the ceiling for a weight stream, which is almost pure READ — a GEMM
//! touches ~89 MiB of weights and writes a few hundred KiB. Writes cost extra on
//! the way out (write-allocate, partial-line read-modify-write), so mixing them
//! in understates what a read-only stream can do.
//!
//! This is the read-only number: stream a buffer far larger than the 32 MB MALL,
//! accumulate in registers, and store one word per block so nothing is optimized
//! away. Sweeps the knobs that actually matter for a stream — load width, loads
//! in flight per thread, block size, and how many waves are resident — plus a
//! non-temporal variant that bypasses the cache hierarchy.
//!
//! # What peak is on this box
//!
//! gfx1151 (Strix Halo, Ryzen AI MAX+ 395) is LPDDR5X on a 256-bit bus, so peak
//! is `MT/s x 32 B`. Three numbers are easy to confuse:
//!
//! | | MT/s | mclk max | peak |
//! |---|---|---|---|
//! | what the modules are RATED for | 8532 | — | 273 GB/s |
//! | **configured now** (post BIOS change, 2026-08-20) | **8000** | 1000 MHz | **256 GB/s** |
//! | configured before | ~7500 | 937 MHz | 240 GB/s |
//!
//! Read the truth from `dmidecode -t memory`: "Speed" is the module RATING,
//! **"Configured Memory Speed" is what it actually runs at**. `pp_dpm_mclk`
//! corroborates (max x 8 x 32 B = peak) — and note the driver never marks an
//! ACTIVE mclk state on this APU (SMU-managed UMC), so the table's max is the
//! only signal; you cannot confirm it by reading the clock under load.
//!
//! # Findings on gfx1151 (2 GiB buffer)
//!
//! * **250.3 GB/s = 97.8 % of the 256 GB/s** now configured, via `bw_chunk_u1`,
//!   block=1024, very wide grid. Before the BIOS change: 235.4 = 98.1 % of 240.
//!   The kernel tracked the clock almost exactly and stayed pinned at the wall,
//!   which is the useful result — nothing is left in the kernel, and more needs
//!   the modules configured at their rated 8532 MT/s.
//! * **Unrolling HURTS.** One 128-bit load per iteration wins; u2/u4/u8 give
//!   219/216/203 GB/s on the grid-stride form. "More loads in flight" is the
//!   usual advice and it is wrong here.
//! * The optimum is a WIDE, SHALLOW launch — ~1-2 `dwordx4` per thread.
//! * **Non-temporal loads are neutral**, worth knowing given they measurably
//!   regressed a different hipfire kernel (34eb024, -13 %).
//! * **Buffer size is not a factor**: flat from 64 MiB to 4 GiB, so address
//!   translation is not the limiter.
//! * Read-only beats the read+write roofline in `bench_dram_bw` (201-211 GB/s)
//!   by ~11 %, which is the point of having this one.
//!
//!   cargo run --release -p hipfire-rdna --example bench_read_bw

use hipfire_rdna::{DType, Gpu};
use std::ffi::c_void;
use std::time::Instant;

/// Peak to report against. Defaults to 256 GB/s (LPDDR5X-8000 x 256-bit), which is
/// what this box runs at after the 2026-08-20 BIOS change. Override with
/// `HIPFIRE_DRAM_PEAK_GBPS=273` if the modules are ever configured at their rated
/// 8532 MT/s, or `=240` to compare against the pre-change cap.
fn peak_gbps() -> f64 {
    std::env::var("HIPFIRE_DRAM_PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(256.0)
}

/// Pure-read streaming kernels. Every variant walks the buffer with a grid-stride
/// loop so consecutive lanes always touch consecutive addresses — the access
/// pattern is identical across variants and only the load shape changes.
///
/// The accumulator exists solely to keep the loads alive; the per-block store at
/// the end is 4 bytes against gigabytes read.
const SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int32_t __attribute__((ext_vector_type(4))) i32x4;
typedef int32_t __attribute__((ext_vector_type(2))) i32x2;

// UNROLL independent 128-bit loads per iteration. More loads in flight is the
// main lever on a latency-bound stream: one dwordx4 per thread per iteration
// cannot cover DRAM latency no matter how many waves are resident.
#define READ_X4(NAME, UNROLL)                                                  \
extern "C" __global__ void NAME(const i32x4* __restrict__ src,                 \
                                unsigned long long n4, int* __restrict__ out) { \
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; \
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;    \
    i32x4 acc = {0, 0, 0, 0};                                                  \
    while (i + (UNROLL - 1) * stride < n4) {                                   \
        i32x4 v[UNROLL];                                                       \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < UNROLL; ++u) v[u] = src[i + u * stride];           \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < UNROLL; ++u) acc += v[u];                          \
        i += UNROLL * stride;                                                  \
    }                                                                          \
    for (; i < n4; i += stride) acc += src[i];                                 \
    int s = acc.x + acc.y + acc.z + acc.w;                                     \
    if (threadIdx.x == 0) out[blockIdx.x] = s;                                 \
}

READ_X4(bw_x4_u1, 1)
READ_X4(bw_x4_u2, 2)
READ_X4(bw_x4_u4, 4)
READ_X4(bw_x4_u8, 8)
READ_X4(bw_x4_u16, 16)

// 64-bit loads, to confirm the wide load is worth it rather than assuming so.
extern "C" __global__ void bw_x2_u8(const i32x2* __restrict__ src,
                                    unsigned long long n2, int* __restrict__ out) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    i32x2 acc = {0, 0};
    while (i + 7 * stride < n2) {
        i32x2 v[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = src[i + u * stride];
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc += v[u];
        i += 8 * stride;
    }
    for (; i < n2; i += stride) acc += src[i];
    int s = acc.x + acc.y;
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// Non-temporal: tells the hardware this data will not be reused, so it need not
// be retained. A pure stream never revisits a line, so in principle this is free
// capacity for everything else — but nontemporal loads have MEASURABLY REGRESSED
// a hipfire kernel before (commit 34eb024, -13%), so it gets measured, not
// assumed.
// Block-contiguous. Each block owns one CONTIGUOUS slab and walks it in tiles of
// blockDim*UNROLL elements. Two differences from the grid-stride form above, both
// aimed at the memory controller rather than the core:
//   - a block's successive loads advance by blockDim*16 bytes (a few KB), not by
//     the whole grid stride (268 MB at these sizes), so a block stays inside a
//     DRAM page/row far longer;
//   - the UNROLL loads of one iteration cover one contiguous blockDim*UNROLL*16
//     byte run, so they are in flight together AND adjacent, instead of being
//     scattered a grid-stride apart the way a naively unrolled grid-stride loop
//     puts them.
#define READ_CHUNK(NAME, UNROLL)                                               \
extern "C" __global__ void NAME(const i32x4* __restrict__ src,                 \
                                unsigned long long n4, int* __restrict__ out) { \
    const unsigned long long slab = (n4 + gridDim.x - 1) / gridDim.x;          \
    const unsigned long long base = (unsigned long long)blockIdx.x * slab;     \
    const unsigned long long end  = (base + slab < n4) ? (base + slab) : n4;   \
    const unsigned long long tile = (unsigned long long)blockDim.x * UNROLL;   \
    i32x4 acc = {0, 0, 0, 0};                                                  \
    unsigned long long t = base;                                               \
    for (; t + tile <= end; t += tile) {                                       \
        i32x4 v[UNROLL];                                                       \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < UNROLL; ++u)                                       \
            v[u] = src[t + (unsigned long long)u * blockDim.x + threadIdx.x];  \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < UNROLL; ++u) acc += v[u];                          \
    }                                                                          \
    for (unsigned long long i = t + threadIdx.x; i < end; i += blockDim.x)     \
        acc += src[i];                                                         \
    int s = acc.x + acc.y + acc.z + acc.w;                                     \
    if (threadIdx.x == 0) out[blockIdx.x] = s;                                 \
}

READ_CHUNK(bw_chunk_u1, 1)
READ_CHUNK(bw_chunk_u2, 2)
READ_CHUNK(bw_chunk_u4, 4)
READ_CHUNK(bw_chunk_u8, 8)

extern "C" __global__ void bw_x4_u8_nt(const i32x4* __restrict__ src,
                                       unsigned long long n4, int* __restrict__ out) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    i32x4 acc = {0, 0, 0, 0};
    while (i + 7 * stride < n4) {
        i32x4 v[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = __builtin_nontemporal_load(&src[i + u * stride]);
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc += v[u];
        i += 8 * stride;
    }
    for (; i < n4; i += stride) acc += src[i];
    int s = acc.x + acc.y + acc.z + acc.w;
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}
"#;

struct Blob(Vec<u8>);
impl Blob {
    fn new() -> Self {
        Blob(Vec::new())
    }
    fn ptr(&mut self, p: *const c_void) -> &mut Self {
        while self.0.len() % 8 != 0 {
            self.0.push(0);
        }
        self.0.extend_from_slice(&(p as u64).to_ne_bytes());
        self
    }
    fn u64v(&mut self, v: u64) -> &mut Self {
        while self.0.len() % 8 != 0 {
            self.0.push(0);
        }
        self.0.extend_from_slice(&v.to_ne_bytes());
        self
    }
}

fn size_sweep(gpu: &mut Gpu) {
    // Same kernel, same elements-per-thread, only the footprint changes. If
    // bandwidth falls as the buffer grows, the limiter is address translation
    // (2 GiB is 512 K pages at 4 KiB) rather than the DRAM channels.
    println!("buffer-size sweep — bw_x4_u1, grid sized for ~8 x4 per thread\n");
    let out = gpu.alloc_tensor(&[262144], DType::F32).expect("out");
    gpu.ensure_kernel_public("bench_read_bw", SRC, "bw_x4_u1")
        .expect("compile");
    for mib in [64usize, 128, 256, 512, 1024, 2048, 4096] {
        let bytes = mib * 1024 * 1024;
        let src = match gpu.alloc_tensor(&[bytes / 4], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                println!("  {mib:>5} MiB   alloc failed: {e:?}");
                continue;
            }
        };
        gpu.fill_f32(&src, 1.0).expect("fill");
        gpu.device_synchronize().expect("sync");
        let n4 = (bytes / 16) as u64;
        let block = 512u32;
        let blocks = ((n4 / 8) as u32 / block).max(1);
        let run = |g: &Gpu| {
            let mut kb = Blob::new();
            kb.ptr(src.buf.as_ptr() as *const c_void)
                .u64v(n4)
                .ptr(out.buf.as_ptr() as *const c_void);
            g.launch_kernel_blob("bw_x4_u1", [blocks, 1, 1], [block, 1, 1], 0, &mut kb.0)
                .expect("launch");
        };
        run(gpu);
        gpu.device_synchronize().expect("sync");
        let mut best = 0.0f64;
        for _ in 0..10 {
            let t = Instant::now();
            run(gpu);
            gpu.device_synchronize().expect("sync");
            best = best.max(bytes as f64 / t.elapsed().as_secs_f64() / 1e9);
        }
        println!(
            "  {mib:>5} MiB   blocks={blocks:<6} {best:7.1} GB/s  {:5.1}% of peak",
            100.0 * best / peak_gbps()
        );
        let _ = gpu.free_tensor(src);
    }
    let _ = gpu.free_tensor(out);
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    if std::env::args().any(|a| a == "--wide") {
        // The plateau showed up at very wide, very shallow launches — many waves
        // each issuing a couple of loads. Push that axis to where each thread
        // does exactly one load and there is no loop at all.
        let bytes: usize = 2 * 1024 * 1024 * 1024;
        let src = gpu.alloc_tensor(&[bytes / 4], DType::F32).expect("src");
        gpu.fill_f32(&src, 1.0).expect("fill");
        let out = gpu.alloc_tensor(&[1 << 20], DType::F32).expect("out");
        gpu.device_synchronize().expect("sync");
        let n4 = (bytes / 16) as u64;
        for name in ["bw_x4_u1", "bw_chunk_u1"] {
            gpu.ensure_kernel_public("bench_read_bw", SRC, name)
                .expect("compile");
            println!("{name}  (n4={n4}):");
            for &block in &[256u32, 512, 1024] {
                for &blocks in &[65536u32, 131072, 262144, 524288] {
                    let per = n4 as f64 / (blocks as f64 * block as f64);
                    let run = |g: &Gpu| {
                        let mut kb = Blob::new();
                        kb.ptr(src.buf.as_ptr() as *const c_void)
                            .u64v(n4)
                            .ptr(out.buf.as_ptr() as *const c_void);
                        g.launch_kernel_blob(name, [blocks, 1, 1], [block, 1, 1], 0, &mut kb.0)
                            .expect("launch");
                    };
                    run(&gpu);
                    gpu.device_synchronize().expect("sync");
                    let mut best = 0.0f64;
                    for _ in 0..10 {
                        let t = Instant::now();
                        run(&gpu);
                        gpu.device_synchronize().expect("sync");
                        best = best.max(bytes as f64 / t.elapsed().as_secs_f64() / 1e9);
                    }
                    println!(
                        "   block={block:<5} blocks={blocks:<7} {per:4.1} x4/thread  \
                         {best:7.1} GB/s  {:5.1}%",
                        100.0 * best / peak_gbps()
                    );
                }
            }
        }
        return;
    }
    if std::env::args().any(|a| a == "--size-sweep") {
        size_sweep(&mut gpu);
        return;
    }
    // Far past the 32 MB MALL, and big enough that one pass is ~10 ms at peak —
    // long enough that launch overhead and the timer are both irrelevant.
    let bytes: usize = 2 * 1024 * 1024 * 1024;
    let n_i32 = bytes / 4;
    let src = gpu.alloc_tensor(&[n_i32], DType::F32).expect("src");
    // Touch every page so the measurement is not timing first-touch faults.
    gpu.fill_f32(&src, 1.0).expect("fill");
    let out = gpu.alloc_tensor(&[65536], DType::F32).expect("out");
    gpu.device_synchronize().expect("sync");

    for name in [
        "bw_x4_u1",
        "bw_x4_u8_nt",
        "bw_chunk_u1",
        "bw_chunk_u2",
        "bw_chunk_u4",
        "bw_chunk_u8",
    ] {
        gpu.ensure_kernel_public("bench_read_bw", SRC, name)
            .expect("compile");
    }

    println!(
        "pure-read DRAM bandwidth  arch={}  buf={} MiB",
        gpu.arch,
        bytes >> 20
    );
    println!("  peak for this part: {:.0} GB/s\n", peak_gbps());

    let mut best = (0.0f64, String::new());
    for (name, elem_bytes) in [
        ("bw_x4_u1", 16usize),
        ("bw_chunk_u1", 16),
        ("bw_chunk_u2", 16),
        ("bw_chunk_u4", 16),
        ("bw_chunk_u8", 16),
        ("bw_x4_u8_nt", 16),
    ] {
        let n_elems = (bytes / elem_bytes) as u64;
        println!("{name}:");
        for &block in &[256u32, 512, 1024] {
            // Sweep resident waves: too few cannot cover latency, too many just
            // adds tail. Expressed in blocks so the shape is readable.
            for &blocks in &[2048u32, 8192, 32768, 131072] {
                let run = |g: &Gpu| {
                    let mut kb = Blob::new();
                    kb.ptr(src.buf.as_ptr() as *const c_void)
                        .u64v(n_elems)
                        .ptr(out.buf.as_ptr() as *const c_void);
                    g.launch_kernel_blob(name, [blocks, 1, 1], [block, 1, 1], 0, &mut kb.0)
                        .expect("launch");
                };
                run(&gpu);
                gpu.device_synchronize().expect("sync");
                let mut best_gbps = 0.0f64;
                for _ in 0..5 {
                    let t = Instant::now();
                    run(&gpu);
                    gpu.device_synchronize().expect("sync");
                    let gbps = bytes as f64 / t.elapsed().as_secs_f64() / 1e9;
                    best_gbps = best_gbps.max(gbps);
                }
                let pct = 100.0 * best_gbps / peak_gbps();
                println!("   block={block:<4} blocks={blocks:<6} {best_gbps:7.1} GB/s  {pct:5.1}% of peak");
                if best_gbps > best.0 {
                    best = (best_gbps, format!("{name} block={block} blocks={blocks}"));
                }
            }
        }
        println!();
    }
    println!(
        "BEST: {:.1} GB/s ({:.1}% of {:.0}) via {}",
        best.0,
        100.0 * best.0 / peak_gbps(),
        peak_gbps(),
        best.1
    );
}
