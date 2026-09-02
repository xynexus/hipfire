// SPDX-License-Identifier: Apache-2.0
// hipfire — is the KVarN K write path batch-invariant?
//
//! Writing N rows in ONE `kvarn_attend` call must leave the cache byte-identical
//! to writing the same N rows one at a time. Speculative decode writes `b` rows
//! per call; plain AR decode writes one. If those disagree, the same committed
//! tokens end up stored differently depending on how they were produced, and
//! speculative output cannot match AR output — which is exactly what
//! `docs/bugs/2026-09-01-spec-decode-not-output-equivalent-to-ar.md` measures.
//!
//! This isolates the question from the model entirely: identical synthetic K/V,
//! two caches, two batching schedules, compare the stored bytes.
//!
//!   kvarn_write_batch_parity [--n 40] [--batch 17] [--bits 2|4|8] [--heads 4]
//!                            [--head-dim 128]
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::kv::KvCache;

fn arg(name: &str, default: usize) -> usize {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic, non-degenerate K/V so quantisation actually has work to do.
fn synth(rows: usize, kv_dim: usize, salt: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(rows * kv_dim);
    let mut s = 0x9e3779b9u32 ^ salt;
    for _ in 0..rows * kv_dim {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        // spread over a few orders of magnitude: var-norm records care
        v.push(((s >> 8) as f32 / 8388608.0 - 1.0) * 4.0);
    }
    v
}

fn main() {
    let n = arg("--n", 40);
    let batch = arg("--batch", 17);
    let bits = arg("--bits", 4);
    let n_kv_heads = arg("--heads", 4);
    let head_dim = arg("--head-dim", 128);
    let n_heads = n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;
    let cap = 512usize;

    let mut gpu = Gpu::init().expect("gpu init");
    println!("kvarn write parity: n={n} batch={batch} bits={bits} heads={n_kv_heads} head_dim={head_dim}");

    let k_all = synth(n, kv_dim, 1);
    let v_all = synth(n, kv_dim, 2);
    let q_all = synth(n, n_heads * head_dim, 3);

    // Two caches, identical geometry. `is_kv_layer` of one true layer keeps the
    // buffers real rather than placeholders.
    let mk = |g: &mut Gpu| {
        KvCache::new_gpu_kvarn_capped_filtered(g, &[true], n_kv_heads, head_dim, cap, cap, bits)
            .expect("kvarn cache")
    };
    let cache_a = mk(&mut gpu);
    let cache_b = mk(&mut gpu);

    let up = |g: &mut Gpu, d: &[f32]| -> GpuTensor { g.upload_f32(d, &[d.len()]).expect("upload") };
    let out = gpu
        .alloc_tensor(&[batch.max(1) * n_heads * head_dim], DType::F32)
        .unwrap();
    let partials = gpu
        .alloc_tensor(&[batch.max(1) * n_heads * head_dim * 4 + 4096], DType::F32)
        .unwrap();
    let tiles = gpu.alloc_tensor(&[kv_dim * 128], DType::F32).unwrap();

    // One call per schedule step; `sched` gives (start_pos, rows).
    // Also capture the ATTENTION OUTPUT per absolute row. The stored bytes test
    // the WRITE path; the outputs test the READ path on the same cache. If the
    // bytes match and the outputs do not, the divergence is in the multi-row
    // read, not in what was stored.
    let run = |g: &mut Gpu, cache: &KvCache, sched: &[(usize, usize)]| -> Vec<Vec<f32>> {
        let mut rows_out: Vec<Vec<f32>> = vec![Vec::new(); n];
        for &(start, rows) in sched {
            let ks = &k_all[start * kv_dim..(start + rows) * kv_dim];
            let vs = &v_all[start * kv_dim..(start + rows) * kv_dim];
            let qs = &q_all[start * n_heads * head_dim..(start + rows) * n_heads * head_dim];
            let fa_k = up(g, ks);
            let fa_v = up(g, vs);
            let fa_q = up(g, qs);
            let posv: Vec<f32> = (0..rows)
                .map(|i| f32::from_bits((start + i) as u32))
                .collect();
            let pos = up(g, &posv);
            g.kvarn_attend(
                &cache.k_gpu[0],
                &cache.k_window[0],
                &cache.v_gpu[0],
                &fa_q,
                &fa_k,
                &fa_v,
                &pos,
                &out,
                &partials,
                &tiles,
                rows,
                start,
                n_heads,
                n_kv_heads,
                head_dim,
                cap,
                None,
                0,
                0,
                bits,
            )
            .expect("kvarn_attend");
            let got = g.download_f32(&out).expect("out");
            let w = n_heads * head_dim;
            for i in 0..rows {
                if (i + 1) * w <= got.len() {
                    rows_out[start + i] = got[i * w..(i + 1) * w].to_vec();
                }
            }
            for t in [fa_k, fa_v, fa_q, pos] {
                let _ = g.free_tensor(t);
            }
        }
        rows_out
    };

    // A: batched. B: strictly one row at a time.
    let mut sched_a = Vec::new();
    let mut p = 0;
    while p < n {
        let rows = batch.min(n - p);
        sched_a.push((p, rows));
        p += rows;
    }
    let sched_b: Vec<(usize, usize)> = (0..n).map(|i| (i, 1)).collect();

    let out_a = run(&mut gpu, &cache_a, &sched_a);
    let out_b = run(&mut gpu, &cache_b, &sched_b);

    let dl = |g: &Gpu, t: &GpuTensor| -> Vec<u8> {
        let mut b = vec![0u8; t.buf.size()];
        g.hip.memcpy_dtoh(&mut b, &t.buf).expect("dtoh");
        b
    };
    let (ra, rb) = (dl(&gpu, &cache_a.k_gpu[0]), dl(&gpu, &cache_b.k_gpu[0]));
    let (wa, wb) = (
        dl(&gpu, &cache_a.k_window[0]),
        dl(&gpu, &cache_b.k_window[0]),
    );

    let diff = |a: &[u8], b: &[u8]| -> (usize, usize) {
        let d = a.iter().zip(b).filter(|(x, y)| x != y).count();
        (d, a.len())
    };
    let (rd, rl) = diff(&ra, &rb);
    let (wd, wl) = diff(&wa, &wb);
    // NON-VACUITY GUARD. Comparing two buffers that were never written is a
    // guaranteed pass that proves nothing -- this investigation has been misled
    // by exactly that three times. A block only seals every GROUP=128 rows, so
    // `--n` below 128 leaves `records` untouched and all-zero.
    let nz = |b: &[u8]| b.iter().filter(|x| **x != 0).count();
    let (rnz, wnz) = (nz(&ra), nz(&wa));
    println!("  records: {rd}/{rl} bytes differ   ({rnz} non-zero bytes written)");
    println!("  window : {wd}/{wl} bytes differ   ({wnz} non-zero bytes written)");
    if rnz == 0 {
        println!(
            "INCONCLUSIVE — the record buffer is untouched: no 128-row block sealed. \
             Re-run with --n above 128 so the flush path actually executes."
        );
        std::process::exit(2);
    }
    if wnz == 0 {
        println!("INCONCLUSIVE — the window buffer is untouched; nothing was staged.");
        std::process::exit(2);
    }
    // Read-path comparison, per row.
    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    let mut diff_rows = 0usize;
    for r in 0..n {
        if out_a[r].len() != out_b[r].len() || out_a[r].is_empty() {
            continue;
        }
        let d = out_a[r]
            .iter()
            .zip(&out_b[r])
            .map(|(x, y)| (x - y).abs() / (x.abs().max(y.abs()).max(1e-6)))
            .fold(0.0f32, f32::max);
        if d > 0.0 {
            diff_rows += 1;
        }
        if d > worst {
            worst = d;
            worst_row = r;
        }
    }
    println!("  attend : {diff_rows}/{n} rows differ, worst |rel| {worst:.3e} at row {worst_row}");

    if rd == 0 && wd == 0 && diff_rows == 0 {
        println!("PASS — the KVarN K write AND read paths are batch-invariant at these sizes");
    } else if rd == 0 && wd == 0 {
        println!(
            "SPLIT — stored bytes are identical but the ATTENTION OUTPUT differs. \
             The write path is batch-invariant; the multi-row READ is not."
        );
        std::process::exit(1);
    } else {
        println!(
            "FAIL — batching changes the stored K. Speculation writes b rows per call \
             and AR writes one, so the same committed tokens are stored differently."
        );
        std::process::exit(1);
    }
}
