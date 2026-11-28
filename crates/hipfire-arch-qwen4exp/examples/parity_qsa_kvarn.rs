// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! KVarN sparse-attention cache: write N positions, gather them back.
//!
//! At the shipped geometry (2 KV heads x 256), writes `N` random K/V rows into a
//! KVarN-4 `QsaCache` and an f32 one, then gathers every position from both
//! through the SAME `QsaCache::gather` the attention uses, and checks:
//!
//! * open block (the last `N % 128` positions, still in the f32 window):
//!   K bit-identical to the f32 cache;
//! * closed blocks (quantised records): per-row cosine >= 0.98 to the source row;
//! * V (Q8_0 everywhere): within half a Q8 step plus the f16 scale's rounding;
//! * a NEGATIVE CONTROL — the same rows compared one position off must fail the
//!   cosine bar, so a slot-indexing bug cannot hide inside 4-bit quantisation
//!   noise;
//! * a scattered slot subset (the shape QSA's selection produces) gathers the
//!   same bytes as the full sweep.
//!
//!     parity_qsa_kvarn [N]

use hipfire_arch_qwen4exp::attn_gpu::{QsaCache, QsaKvMode};
use hipfire_arch_qwen4exp::config::Qwen4ExpConfig;
use hipfire_rdna::{DType, Gpu};

fn fail(msg: String) -> ! {
    eprintln!("parity_qsa_kvarn: FAIL — {msg}");
    std::process::exit(1);
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    d / (na * nb).max(1e-20)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let cfg_json: serde_json::Value =
        serde_json::from_str(include_str!("../tests/qwen3_8_flash_next.config.json"))
            .expect("config json");
    let cfg = Qwen4ExpConfig::from_json(&cfg_json).expect("config");
    let (nkv, hd) = (cfg.n_kv_heads, cfg.head_dim);
    let kv_dim = nkv * hd;
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_qsa_kvarn: no GPU ({e:?}) — skipped");
            return;
        }
    };
    let max_seq = n.next_power_of_two().max(256);
    let mut kvarn =
        QsaCache::with_kv(&mut gpu, &cfg, max_seq, QsaKvMode::Kvarn { bits: 4 }).unwrap();
    let mut f32c = QsaCache::with_kv(&mut gpu, &cfg, max_seq, QsaKvMode::F32).unwrap();

    // Deterministic rows with a per-channel scale spread, like real K.
    let mut seed = 0x9e37_79b9u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed % 20001) as f32 / 10000.0 - 1.0
    };
    let mut ks: Vec<f32> = Vec::with_capacity(n * kv_dim);
    let mut vs = Vec::with_capacity(n * kv_dim);
    for _ in 0..n {
        for c in 0..kv_dim {
            ks.push(rnd() * (1.0 + (c % 17) as f32));
            vs.push(rnd());
        }
    }
    let positions: Vec<f32> = (0..n).map(|p| f32::from_bits(p as u32)).collect();
    let pos_dev = gpu.upload_f32(&positions, &[n]).unwrap();
    for p in 0..n {
        let k = gpu
            .upload_f32(&ks[p * kv_dim..(p + 1) * kv_dim], &[kv_dim])
            .unwrap();
        let v = gpu
            .upload_f32(&vs[p * kv_dim..(p + 1) * kv_dim], &[kv_dim])
            .unwrap();
        for c in [&mut kvarn, &mut f32c] {
            c.write(&mut gpu, &cfg, &k, &v, p, &pos_dev.sub_offset(p, 1))
                .unwrap();
            c.len = p + 1;
        }
        gpu.free_tensor(k).unwrap();
        gpu.free_tensor(v).unwrap();
    }

    let out = |gpu: &mut Gpu, c: &QsaCache, slots: &[usize]| -> (Vec<f32>, Vec<f32>) {
        let sv: Vec<f32> = slots.iter().map(|&p| f32::from_bits(p as u32)).collect();
        let st = gpu.upload_f32(&sv, &[sv.len()]).unwrap();
        let ok = gpu.zeros(&[nkv * slots.len() * hd], DType::F32).unwrap();
        let ov = gpu.zeros(&[nkv * slots.len() * hd], DType::F32).unwrap();
        c.gather(gpu, &cfg, &st, slots.len(), &ok, &ov).unwrap();
        let r = (
            gpu.download_f32(&ok).unwrap(),
            gpu.download_f32(&ov).unwrap(),
        );
        for t in [st, ok, ov] {
            gpu.free_tensor(t).unwrap();
        }
        r
    };
    // A rotated cache holds R·k (R orthonormal, per 256 channels); compare
    // against the source rows in THAT frame, rotated by the same kernel.
    let rotated = kvarn.k_rotated();
    if rotated {
        let src_t = gpu.upload_f32(&ks, &[ks.len()]).unwrap();
        let dst_t = gpu.zeros(&[ks.len()], DType::F32).unwrap();
        gpu.rotate_x_mq_batched(&src_t, &dst_t, kv_dim, n).unwrap();
        ks = gpu.download_f32(&dst_t).unwrap();
        gpu.free_tensor(src_t).unwrap();
        gpu.free_tensor(dst_t).unwrap();
    }
    println!("  K stored rotated: {rotated}");
    let all: Vec<usize> = (0..n).collect();
    let (qk, qv) = out(&mut gpu, &kvarn, &all);
    let (fk, _) = out(&mut gpu, &f32c, &all);
    // [nkv, n_slots, hd] -> row (head, slot)
    let at = |buf: &[f32], h: usize, s: usize, ns: usize| -> Vec<f32> {
        buf[(h * ns + s) * hd..(h * ns + s + 1) * hd].to_vec()
    };
    let src = |rows: &[f32], h: usize, p: usize| -> Vec<f32> {
        rows[p * kv_dim + h * hd..p * kv_dim + (h + 1) * hd].to_vec()
    };

    let closed = (n / 128) * 128;
    let (mut worst_cos, mut worst_neighbour, mut v_viol) = (1.0f32, -1.0f32, 0usize);
    for h in 0..nkv {
        for p in 0..n {
            let got = at(&qk, h, p, n);
            if p >= closed {
                let want = if rotated {
                    src(&ks, h, p)
                } else {
                    at(&fk, h, p, n)
                };
                if got != want {
                    fail(format!("open-block K at pos {p} head {h} is not exact"));
                }
            } else {
                worst_cos = worst_cos.min(cosine(&got, &src(&ks, h, p)));
                if p + 1 < closed {
                    worst_neighbour = worst_neighbour.max(cosine(&got, &src(&ks, h, p + 1)));
                }
            }
            let gv = at(&qv, h, p, n);
            let want = src(&vs, h, p);
            for blk in 0..hd / 32 {
                let amax = want[blk * 32..(blk + 1) * 32]
                    .iter()
                    .fold(0.0f32, |a, v| a.max(v.abs()));
                let step = amax / 127.0;
                for i in blk * 32..(blk + 1) * 32 {
                    // Half a step of rounding, plus the f16 SCALE's own
                    // rounding (Q8_0 stores it as f16): |q| <= 127 times a
                    // 2^-11 relative error on the step.
                    let bound = 0.5 * step + 127.0 * step * f32::EPSILON.max(2f32.powi(-11)) + 1e-7;
                    if (gv[i] - want[i]).abs() > bound {
                        v_viol += 1;
                    }
                }
            }
        }
    }
    println!(
        "  {n} positions ({closed} quantised, {} in the window): closed-block K cosine >= {worst_cos:.4}, \
         neighbour control <= {worst_neighbour:.4}, V outside its Q8 bound: {v_viol}",
        n - closed
    );
    if closed == 0 {
        fail("no block closed — the quantised path was never exercised".into());
    }
    if worst_cos < 0.98 {
        fail(format!("closed-block K cosine {worst_cos:.4} < 0.98"));
    }
    if worst_neighbour > 0.5 {
        fail(format!(
            "negative control passed ({worst_neighbour:.4}) — the check cannot see an off-by-one"
        ));
    }
    if v_viol > 0 {
        fail(format!("{v_viol} V elements outside the Q8 bound"));
    }

    // A scattered subset — QSA's shape — must gather exactly what the sweep did.
    let subset: Vec<usize> = (0..n).rev().step_by(7).collect();
    let (sk, sv) = out(&mut gpu, &kvarn, &subset);
    for h in 0..nkv {
        for (i, &p) in subset.iter().enumerate() {
            if at(&sk, h, i, subset.len()) != at(&qk, h, p, n)
                || at(&sv, h, i, subset.len()) != at(&qv, h, p, n)
            {
                fail(format!(
                    "scattered gather differs from the sweep at pos {p} head {h}"
                ));
            }
        }
    }
    println!(
        "  scattered subset of {} slots matches the sweep exactly",
        subset.len()
    );
    kvarn.free(&mut gpu);
    f32c.free(&mut gpu);
    println!("parity_qsa_kvarn: OK");
}
