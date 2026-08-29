// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.
// SPDX-License-Identifier: Apache-2.0
// hipfire — GPU/CPU parity for every qwen4_exp kernel.
//
// Differences each family-specific kernel against the independent CPU reference in
// `hipfire_arch_qwen4exp::hc`. The two implementations share no code, which is the
// whole point: an earlier adversarial review of this port found a comparison where
// both arms ran the SAME routing code, so a bug was common-mode and cancelled
// exactly out of the metric. A kernel checked against a reimplementation of itself
// proves nothing.
//
//   cargo run --release -p hipfire-rdna --example parity_qwen4exp_kernels
//
// Exit code is nonzero on failure so a gate can call it directly.

use hipfire_arch_qwen4exp::hc::{gated_rmsnorm_sigmoid, grouped_rmsnorm, GatedResidual};
use hipfire_arch_qwen4exp::ple::dilated_conv_silu_step;
use hipfire_arch_qwen4exp::qsa::{
    pool_block, score_block, select as qsa_select, topk_by_threshold, QsaParams,
};
use hipfire_rdna::Gpu;

/// Deterministic xorshift so a failure is reproducible without a fixture file.
fn seeded(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("parity_qwen4exp_kernels: no GPU ({e}); skipping");
            return;
        }
    };

    // The real geometry (hidden 2560, 4 streams) plus a ragged width that is not a
    // multiple of the 256-wide block, since the tail guard is the easiest thing to
    // get wrong in the launch.
    let cases: &[(usize, usize)] = &[(2560, 4), (256, 4), (300, 4), (1, 4), (2560, 2)];
    let mut worst = 0.0f32;
    let mut failed = false;

    for &(hidden, hc) in cases {
        let width = hidden * hc;
        let streams = seeded(width, 17);
        let norm_w = seeded(width, 23);
        // The kernel consumes ALREADY-normalised streams and an ALREADY-sigmoid'd
        // gate; both come from upstream ops. Build them on the CPU so this isolates
        // the collapse itself.
        let normed = grouped_rmsnorm(&streams, &norm_w, hidden, 1e-6);
        let mix: Vec<f32> = seeded(width, 31)
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();

        // CPU reference: mean over streams of the per-channel-gated normed streams.
        let mut want = vec![0.0f32; hidden];
        for s in 0..hc {
            for d in 0..hidden {
                want[d] += mix[s * hidden + d] * normed[s * hidden + d];
            }
        }
        for v in want.iter_mut() {
            *v /= hc as f32;
        }

        let g_mix = gpu.upload_f32(&mix, &[hc, hidden]).expect("upload mix");
        let g_str = gpu
            .upload_f32(&normed, &[hc, hidden])
            .expect("upload streams");
        let g_out = gpu
            .zeros(&[hidden], hipfire_rdna::DType::F32)
            .expect("alloc out");
        gpu.hc_input_map_perchannel(&g_mix, &g_str, &g_out, hidden as i32, hc as i32)
            .expect("launch");
        let got = gpu.download_f32(&g_out).expect("download");

        let max_abs = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(max_abs);
        // f32 fma vs a plain CPU multiply-add: the only divergence allowed is the
        // last ulp or two, so the bar is tight rather than a tolerance to hide in.
        let ok = max_abs < 1e-6;
        failed |= !ok;
        println!(
            "  hidden={hidden:<5} hc={hc}  max|Δ| = {max_abs:.3e}  {}",
            if ok { "ok" } else { "FAIL" }
        );
    }

    // ── grouped RMSNorm (M11) ────────────────────────────────────────────────
    for &(hidden, hc) in cases {
        let width = hidden * hc;
        let x = seeded(width, 41);
        let w = seeded(width, 43);
        let want = grouped_rmsnorm(&x, &w, hidden, 1e-6);

        let g_x = gpu.upload_f32(&x, &[hc, hidden]).unwrap();
        let g_w = gpu.upload_f32(&w, &[hc, hidden]).unwrap();
        let g_o = gpu.zeros(&[hc, hidden], hipfire_rdna::DType::F32).unwrap();
        gpu.hc_grouped_rmsnorm(&g_x, &g_w, &g_o, hidden as i32, hc as i32, 1e-6)
            .expect("launch rmsnorm");
        let got = gpu.download_f32(&g_o).unwrap();

        let max_abs = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(max_abs);
        // Tree reduction on GPU vs sequential on CPU, so the bar is looser than the
        // elementwise kernel's — but still far tighter than any real bug.
        let ok = max_abs < 1e-5;
        failed |= !ok;
        println!(
            "  rmsnorm hidden={hidden:<5} hc={hc}  max|Δ| = {max_abs:.3e}  {}",
            if ok { "ok" } else { "FAIL" }
        );
    }

    // ── whole read path, GPU vs the CPU oracle (M12) ─────────────────────────
    // norm -> down -> silu(/hc) -> up -> sigmoid -> per-channel mean collapse,
    // and the inject gates. Every stage on GPU; the oracle shares no code.
    {
        let (hidden, hc, lowrank) = (2560usize, 4usize, 320usize);
        let width = hidden * hc;
        let streams = seeded(width, 51);
        let norm_w = seeded(width, 53);
        let down = seeded(lowrank * width, 55);
        let up = seeded(width * lowrank, 57);
        let inject_w = seeded(hc * width, 59);

        let oracle = GatedResidual {
            hc_norm: &norm_w,
            mix_down: &down,
            mix_up: &up,
            block_inject: Some(&inject_w),
            hc_count: hc,
            hidden,
            lowrank,
            eps: 1e-6,
        }
        .read(&streams);

        let g_str = gpu.upload_f32(&streams, &[hc, hidden]).unwrap();
        let g_nw = gpu.upload_f32(&norm_w, &[hc, hidden]).unwrap();
        let g_normed = gpu.zeros(&[hc, hidden], hipfire_rdna::DType::F32).unwrap();
        gpu.hc_grouped_rmsnorm(&g_str, &g_nw, &g_normed, hidden as i32, hc as i32, 1e-6)
            .unwrap();

        let g_down = gpu.upload_f32(&down, &[lowrank, width]).unwrap();
        let g_t = gpu.zeros(&[lowrank], hipfire_rdna::DType::F32).unwrap();
        gpu.gemv_f32(&g_down, &g_normed, &g_t).unwrap();
        let g_ta = gpu.zeros(&[lowrank], hipfire_rdna::DType::F32).unwrap();
        gpu.hc_scaled_silu(&g_t, &g_ta, lowrank as i32, 1.0 / hc as f32)
            .unwrap();

        let g_up = gpu.upload_f32(&up, &[width, lowrank]).unwrap();
        let g_m = gpu.zeros(&[width], hipfire_rdna::DType::F32).unwrap();
        gpu.gemv_f32(&g_up, &g_ta, &g_m).unwrap();
        let g_mix = gpu.zeros(&[width], hipfire_rdna::DType::F32).unwrap();
        gpu.hc_sigmoid(&g_m, &g_mix, width as i32).unwrap();

        let g_out = gpu.zeros(&[hidden], hipfire_rdna::DType::F32).unwrap();
        gpu.hc_input_map_perchannel(&g_mix, &g_normed, &g_out, hidden as i32, hc as i32)
            .unwrap();
        let got = gpu.download_f32(&g_out).unwrap();

        let max_abs = oracle
            .mixed_input
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(max_abs);
        // Two 10240-wide reductions compound here, so the bar is looser than a
        // single kernel's — still orders below any real defect.
        let ok = max_abs < 1e-4;
        failed |= !ok;
        println!(
            "  read path  hidden={hidden} hc={hc} r={lowrank}  max|Δ| = {max_abs:.3e}  {}",
            if ok { "ok" } else { "FAIL" }
        );

        // Inject gates: gemv then 2*sigmoid(v / hc).
        let g_iw = gpu.upload_f32(&inject_w, &[hc, width]).unwrap();
        let g_i = gpu.zeros(&[hc], hipfire_rdna::DType::F32).unwrap();
        gpu.gemv_f32(&g_iw, &g_normed, &g_i).unwrap();
        let want_i = oracle.inject.clone().unwrap();
        // The oracle applies `/hc` INSIDE the sigmoid and doubles the result, and
        // `hc_sigmoid` takes neither argument — so the scaling is applied here,
        // which keeps this check pointed at the GEMV rather than at an activation
        // already covered above.
        let raw = gpu.download_f32(&g_i).unwrap();
        let gpu_inject: Vec<f32> = raw
            .iter()
            .map(|v| 2.0 / (1.0 + (-(v / hc as f32)).exp()))
            .collect();
        let mi = want_i
            .iter()
            .zip(&gpu_inject)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(mi);
        let ok = mi < 1e-4;
        failed |= !ok;
        println!(
            "  inject gates                          max|Δ| = {mi:.3e}  {}",
            if ok { "ok" } else { "FAIL" }
        );
    }

    // ── residual write-back (M13) ────────────────────────────────────────────
    {
        let (hidden, hc) = (2560usize, 4usize);
        let mut streams = seeded(hidden * hc, 61);
        let block_out = seeded(hidden, 63);
        let inject = vec![0.0f32, 1.0, 0.5, 1.75];
        let g_s = gpu.upload_f32(&streams, &[hc, hidden]).unwrap();
        let g_b = gpu.upload_f32(&block_out, &[hidden]).unwrap();
        let g_i = gpu.upload_f32(&inject, &[hc]).unwrap();
        gpu.hc_residual_inject(&g_s, &g_b, &g_i, hidden as i32, hc as i32)
            .unwrap();
        let got = gpu.download_f32(&g_s).unwrap();

        let dummy = vec![0.0f32; hidden * hc];
        GatedResidual {
            hc_norm: &dummy,
            mix_down: &[],
            mix_up: &[],
            block_inject: None,
            hc_count: hc,
            hidden,
            lowrank: 0,
            eps: 1e-6,
        }
        .write(&mut streams, &block_out, &inject);

        let max_abs = streams
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(max_abs);
        let ok = max_abs < 1e-6;
        failed |= !ok;
        println!(
            "  write-back hidden={hidden} hc={hc}         max|Δ| = {max_abs:.3e}  {}",
            if ok { "ok" } else { "FAIL" }
        );
    }

    // ── PLE dilated short conv (M19) ─────────────────────────────────────────
    // Multiple steps, so the ring bookkeeping is exercised rather than just the
    // tap arithmetic. Real geometry: hc-wide channels, kernel 4, dilation 3.
    {
        let (channels, kernel, dilation) = (10240usize, 4usize, 3usize);
        let state_len = (kernel - 1) * dilation;
        let w = seeded(channels * kernel, 71);
        let mut cpu_state = vec![0.0f32; channels * state_len];
        let g_state = gpu.upload_f32(&cpu_state, &[channels, state_len]).unwrap();
        let g_w = gpu.upload_f32(&w, &[channels, kernel]).unwrap();
        let g_out = gpu.zeros(&[channels], hipfire_rdna::DType::F32).unwrap();

        let mut step_worst = 0.0f32;
        for t in 0..6 {
            let x = seeded(channels, 73 + t);
            let want = dilated_conv_silu_step(
                &mut cpu_state,
                &x,
                &w,
                channels,
                state_len,
                kernel,
                dilation,
            );
            let g_x = gpu.upload_f32(&x, &[channels]).unwrap();
            gpu.ple_dilated_conv_silu(
                &g_state,
                &g_x,
                &g_w,
                &g_out,
                channels as i32,
                state_len as i32,
                kernel as i32,
                dilation as i32,
            )
            .unwrap();
            let got = gpu.download_f32(&g_out).unwrap();
            let m = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            step_worst = step_worst.max(m);
        }
        // The GPU state must have tracked the CPU's over all six steps, which is
        // what proves the in-place ring advance is right rather than just the taps.
        let gpu_state = gpu.download_f32(&g_state).unwrap();
        let state_delta = cpu_state
            .iter()
            .zip(&gpu_state)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(step_worst).max(state_delta);
        let ok = step_worst < 1e-6 && state_delta < 1e-9;
        failed |= !ok;
        println!(
            "  ple conv  ch={channels} k={kernel} d={dilation}  max|Δ| = {step_worst:.3e}  \
             state Δ = {state_delta:.3e}  {}",
            if ok { "ok" } else { "FAIL" }
        );
    }

    // ── GDN gated RMSNorm with a SIGMOID gate (M20 delta) ────────────────────
    {
        let (n_heads, head_dim) = (48usize, 128usize); // the real DeltaNet V geometry
        let n = n_heads * head_dim;
        let x = seeded(n, 81);
        let z = seeded(n, 83);
        let w = seeded(head_dim, 85);
        let want = gated_rmsnorm_sigmoid(&x, &z, &w, n_heads, head_dim, 1e-6);

        let g_x = gpu.upload_f32(&x, &[n_heads, head_dim]).unwrap();
        let g_z = gpu.upload_f32(&z, &[n_heads, head_dim]).unwrap();
        let g_w = gpu.upload_f32(&w, &[head_dim]).unwrap();
        let g_o = gpu
            .zeros(&[n_heads, head_dim], hipfire_rdna::DType::F32)
            .unwrap();
        gpu.gated_norm_sigmoid_f32(
            &g_x,
            &g_z,
            &g_w,
            &g_o,
            n_heads as i32,
            head_dim as i32,
            1,
            1e-6,
        )
        .expect("launch gated_norm_sigmoid");
        let got = gpu.download_f32(&g_o).unwrap();
        let max_abs = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(max_abs);
        let ok = max_abs < 1e-6;
        failed |= !ok;
        println!(
            "  gdn gated-norm nh={n_heads} hd={head_dim}  max|Δ| = {max_abs:.3e}  {}",
            if ok { "ok" } else { "FAIL" }
        );
    }

    // ── QSA block scoring (M21b core) ────────────────────────────────────────
    {
        let p = QsaParams {
            n_heads: 4,
            head_dim: 128,
            budget: 2048,
            compress_ratio: 4,
        };
        let n_pos = 4096usize;
        let n_blocks = n_pos / p.compress_ratio;
        let q = seeded(p.n_heads * p.head_dim, 91);
        let keys = seeded(n_pos * p.head_dim, 93);
        let visible: Vec<i32> = (0..n_pos as i32).collect();

        let g_q = gpu.upload_f32(&q, &[p.n_heads, p.head_dim]).unwrap();
        let g_k = gpu.upload_f32(&keys, &[n_pos, p.head_dim]).unwrap();
        // Positions ride as i32; upload through an f32 tensor of the same width.
        let vis_as_f32: Vec<f32> = visible.iter().map(|v| f32::from_bits(*v as u32)).collect();
        let g_v = gpu.upload_f32(&vis_as_f32, &[n_pos]).unwrap();
        let g_s = gpu.zeros(&[n_blocks], hipfire_rdna::DType::F32).unwrap();
        gpu.qsa_block_score(
            &g_q,
            &g_k,
            &g_v,
            &g_s,
            p.n_heads as i32,
            p.head_dim as i32,
            p.compress_ratio as i32,
            n_blocks as i32,
        )
        .expect("launch qsa_block_score");
        let got = gpu.download_f32(&g_s).unwrap();

        let mut max_abs = 0.0f32;
        for b in 0..n_blocks {
            let pooled = pool_block(&keys, p.head_dim, b * p.compress_ratio, p.compress_ratio);
            let want = score_block(&q, &pooled, &p);
            max_abs = max_abs.max((want - got[b]).abs());
        }
        worst = worst.max(max_abs);
        // 128-wide dot reduced across 64 lanes vs sequential on CPU.
        let ok = max_abs < 1e-4;
        failed |= !ok;
        println!(
            "  qsa score  blocks={n_blocks} nh={} hd={}  max|Δ| = {max_abs:.3e}  {}",
            p.n_heads,
            p.head_dim,
            if ok { "ok" } else { "FAIL" }
        );
    }

    // Negative control: the per-head ReLU must be INSIDE the sum. Build a query
    // where one head scores strongly positive and another strongly negative: a
    // sum-then-ReLU would cancel them to ~0, ReLU-then-sum keeps the positive.
    {
        let p = QsaParams {
            n_heads: 2,
            head_dim: 128,
            budget: 8,
            compress_ratio: 4,
        };
        let n_pos = 4usize;
        let mut keys = vec![0.0f32; n_pos * p.head_dim];
        for t in 0..n_pos {
            keys[t * p.head_dim] = 1.0; // pooled key = e0
        }
        let mut q = vec![0.0f32; p.n_heads * p.head_dim];
        q[0] = 5.0; // head 0 dot = +5
        q[p.head_dim] = -500.0; // head 1 dot = -500
        let g_q = gpu.upload_f32(&q, &[p.n_heads, p.head_dim]).unwrap();
        let g_k = gpu.upload_f32(&keys, &[n_pos, p.head_dim]).unwrap();
        let vis: Vec<f32> = (0..n_pos as i32)
            .map(|v| f32::from_bits(v as u32))
            .collect();
        let g_v = gpu.upload_f32(&vis, &[n_pos]).unwrap();
        let g_s = gpu.zeros(&[1], hipfire_rdna::DType::F32).unwrap();
        gpu.qsa_block_score(
            &g_q,
            &g_k,
            &g_v,
            &g_s,
            p.n_heads as i32,
            p.head_dim as i32,
            p.compress_ratio as i32,
            1,
        )
        .unwrap();
        let got = gpu.download_f32(&g_s).unwrap()[0];
        let want = 5.0f32 / (p.head_dim as f32).sqrt();
        let ok = (got - want).abs() < 1e-4;
        println!(
            "  relu-before-sum control: got {got:.5} (want {want:.5}; sum-then-relu = 0)  {}",
            if ok { "ok" } else { "FAIL" }
        );
        failed |= !ok;
    }

    // Negative control: the gate must be SIGMOID, not the silu the Qwen3.5/3.8
    // sibling uses. At z = 0 they are maximally distinguishable: sigmoid gives
    // 0.5 * normed, silu gives 0.
    {
        let (n_heads, head_dim) = (4usize, 32usize);
        let n = n_heads * head_dim;
        let x = vec![1.0f32; n];
        let z = vec![0.0f32; n];
        let w = vec![1.0f32; head_dim];
        let g_x = gpu.upload_f32(&x, &[n_heads, head_dim]).unwrap();
        let g_z = gpu.upload_f32(&z, &[n_heads, head_dim]).unwrap();
        let g_w = gpu.upload_f32(&w, &[head_dim]).unwrap();
        let g_o = gpu
            .zeros(&[n_heads, head_dim], hipfire_rdna::DType::F32)
            .unwrap();
        gpu.gated_norm_sigmoid_f32(
            &g_x,
            &g_z,
            &g_w,
            &g_o,
            n_heads as i32,
            head_dim as i32,
            1,
            1e-6,
        )
        .unwrap();
        let got = gpu.download_f32(&g_o).unwrap();
        let is_sigmoid = got.iter().all(|v| (v - 0.5).abs() < 1e-4);
        println!(
            "  sigmoid-not-silu control: got {:.4} (silu would be 0.0)  {}",
            got[0],
            if is_sigmoid { "ok" } else { "FAIL" }
        );
        failed |= !is_sigmoid;
    }

    // Negative control: an UNDILATED conv would read t-1/t-2/t-3 instead of
    // t-9/t-6/t-3. Put a spike at t-3 only (a tap under BOTH) and at t-1 (a tap
    // only under the undilated reading); the dilated kernel must ignore t-1.
    {
        let (channels, kernel, dilation) = (4usize, 4usize, 3usize);
        let state_len = (kernel - 1) * dilation;
        let w = vec![1.0f32; channels * kernel];
        let x = vec![0.0f32; channels];
        let mut st = vec![0.0f32; channels * state_len];
        // slot state_len-1 is t-1, which the DILATED taps never read.
        for c in 0..channels {
            st[c * state_len + state_len - 1] = 100.0;
        }
        let g_s = gpu.upload_f32(&st, &[channels, state_len]).unwrap();
        let g_w = gpu.upload_f32(&w, &[channels, kernel]).unwrap();
        let g_x = gpu.upload_f32(&x, &[channels]).unwrap();
        let g_o = gpu.zeros(&[channels], hipfire_rdna::DType::F32).unwrap();
        gpu.ple_dilated_conv_silu(
            &g_s,
            &g_x,
            &g_w,
            &g_o,
            channels as i32,
            state_len as i32,
            kernel as i32,
            dilation as i32,
        )
        .unwrap();
        let got = gpu.download_f32(&g_o).unwrap();
        let ignored = got.iter().all(|v| v.abs() < 1e-6);
        println!(
            "  dilated-not-adjacent control: out[0]={:.4} (undilated would be ~100)  {}",
            got[0],
            if ignored { "ok" } else { "FAIL" }
        );
        failed |= !ignored;
    }

    // Negative control: a FLAT norm over the whole width is the obvious wrong
    // implementation. Give stream 0 a much larger magnitude than the rest; under
    // grouped norm every stream still normalises to ~unit RMS, under a flat norm
    // they would not.
    {
        let (hidden, hc) = (256usize, 4usize);
        let mut x = vec![1.0f32; hidden * hc];
        for v in x[0..hidden].iter_mut() {
            *v = 100.0;
        }
        let w = vec![0.0f32; hidden * hc];
        let g_x = gpu.upload_f32(&x, &[hc, hidden]).unwrap();
        let g_w = gpu.upload_f32(&w, &[hc, hidden]).unwrap();
        let g_o = gpu.zeros(&[hc, hidden], hipfire_rdna::DType::F32).unwrap();
        gpu.hc_grouped_rmsnorm(&g_x, &g_w, &g_o, hidden as i32, hc as i32, 1e-6)
            .unwrap();
        let got = gpu.download_f32(&g_o).unwrap();
        let all_unit = got.iter().all(|v| (v - 1.0).abs() < 1e-3);
        println!(
            "  grouped-not-flat control: stream0[0]={:.4} stream1[0]={:.4}  {}",
            got[0],
            got[hidden],
            if all_unit { "ok" } else { "FAIL" }
        );
        failed |= !all_unit;
    }

    // Negative control: the kernel must NOT be computing a sum. If it were, a
    // uniform gate of 1.0 would give hc times the stream mean instead of once.
    {
        let (hidden, hc) = (512usize, 4usize);
        let ones = vec![1.0f32; hidden * hc];
        let streams = vec![2.0f32; hidden * hc];
        let g_mix = gpu.upload_f32(&ones, &[hc, hidden]).unwrap();
        let g_str = gpu.upload_f32(&streams, &[hc, hidden]).unwrap();
        let g_out = gpu.zeros(&[hidden], hipfire_rdna::DType::F32).unwrap();
        gpu.hc_input_map_perchannel(&g_mix, &g_str, &g_out, hidden as i32, hc as i32)
            .unwrap();
        let got = gpu.download_f32(&g_out).unwrap();
        let mean_ok = got.iter().all(|v| (v - 2.0).abs() < 1e-6);
        let looks_like_sum = got.iter().any(|v| (v - 8.0).abs() < 1e-3);
        println!(
            "  mean-not-sum control: got {:.3} (mean=2.0, a sum would be 8.0)  {}",
            got[0],
            if mean_ok && !looks_like_sum {
                "ok"
            } else {
                "FAIL"
            }
        );
        failed |= !mean_ok || looks_like_sum;
    }

    // ── QSA top-k mask (M21d) ────────────────────────────────────────────────
    // Differenced against the CPU threshold select, which is itself differenced
    // against a full sort. Sizes span the real case (top-512 of 65536 blocks at
    // native context) and the tie-heavy case where a threshold method and a sort
    // diverge if the band rule differs.
    {
        let cases: &[(usize, usize, u32)] = &[
            (1024, 128, 131),
            (4096, 512, 133),
            (65536, 512, 135), // native context: 262144 tokens / 4 per block
        ];
        let mut all_ok = true;
        for &(n, k, sd) in cases {
            let scores = seeded(n, sd);
            let want = topk_by_threshold(&scores, k);
            let g_s = gpu.upload_f32(&scores, &[n]).unwrap();
            let g_m = gpu.zeros(&[n], hipfire_rdna::DType::F32).unwrap();
            gpu.qsa_topk_mask(&g_s, &g_m, n as i32, k as i32)
                .expect("launch topk");
            let raw = gpu.download_f32(&g_m).unwrap();
            let got: Vec<usize> = raw
                .iter()
                .enumerate()
                .filter(|(_, v)| v.to_bits() != 0)
                .map(|(i, _)| i)
                .collect();
            let ok = got == want;
            all_ok &= ok;
            println!(
                "  qsa topk   n={n:<6} k={k:<4} selected={:<5}  {}",
                got.len(),
                if ok { "ok" } else { "FAIL" }
            );
        }
        failed |= !all_ok;
    }

    // Negative control: the tie band is `max{s <= t}`, NOT `== t`. With only a few
    // distinct scores the threshold never lands ON a value, so a kernel testing
    // `== t` selects nothing from the band and returns FEWER than k — silently.
    {
        let (n, k) = (2048usize, 700usize);
        let scores: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let want = topk_by_threshold(&scores, k);
        let g_s = gpu.upload_f32(&scores, &[n]).unwrap();
        let g_m = gpu.zeros(&[n], hipfire_rdna::DType::F32).unwrap();
        gpu.qsa_topk_mask(&g_s, &g_m, n as i32, k as i32).unwrap();
        let raw = gpu.download_f32(&g_m).unwrap();
        let got: Vec<usize> = raw
            .iter()
            .enumerate()
            .filter(|(_, v)| v.to_bits() != 0)
            .map(|(i, _)| i)
            .collect();
        let exact = got == want;
        println!(
            "  tie-band control: selected {} of k={k} (a `== t` band would give 512)  {}",
            got.len(),
            if exact { "ok" } else { "FAIL" }
        );
        failed |= !exact;
    }

    // ── QSA selection, whole pipeline on GPU (M21e) ──────────────────────────
    // score -> top-k mask -> expand to a token mask, differenced against the CPU
    // `select` oracle. This is the pipeline the attention kernel will consume.
    {
        let p = QsaParams {
            n_heads: 4,
            head_dim: 128,
            budget: 64,
            compress_ratio: 4,
        };
        let mut all_ok = true;
        // Spans the dense regime (at/below `dense_below`, where selection MUST be
        // everything) and well past it, where it must exclude.
        for &n_pos in &[16usize, 64, p.dense_below(), p.dense_below() + 1, 512, 4096] {
            let q = seeded(p.n_heads * p.head_dim, 141);
            let keys = seeded(n_pos * p.head_dim, 143);
            let visible: Vec<usize> = (0..n_pos).collect();
            let want = qsa_select(&q, &keys, &visible, &p);

            let n_blocks = n_pos / p.compress_ratio;
            let g_q = gpu.upload_f32(&q, &[p.n_heads, p.head_dim]).unwrap();
            let g_k = gpu.upload_f32(&keys, &[n_pos, p.head_dim]).unwrap();
            let vis_f: Vec<f32> = (0..n_pos as i32)
                .map(|v| f32::from_bits(v as u32))
                .collect();
            let g_v = gpu.upload_f32(&vis_f, &[n_pos]).unwrap();
            let g_tok = gpu.zeros(&[n_pos], hipfire_rdna::DType::F32).unwrap();

            let got: Vec<usize> = if n_blocks == 0 {
                // No complete block: everything is ragged tail.
                gpu.qsa_expand_mask(
                    &g_tok,
                    &g_v,
                    &g_tok,
                    n_pos as i32,
                    0,
                    p.compress_ratio as i32,
                )
                .unwrap();
                gpu.download_f32(&g_tok)
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.to_bits() != 0)
                    .map(|(i, _)| i)
                    .collect()
            } else {
                let g_s = gpu.zeros(&[n_blocks], hipfire_rdna::DType::F32).unwrap();
                gpu.qsa_block_score(
                    &g_q,
                    &g_k,
                    &g_v,
                    &g_s,
                    p.n_heads as i32,
                    p.head_dim as i32,
                    p.compress_ratio as i32,
                    n_blocks as i32,
                )
                .unwrap();
                let g_bm = gpu.zeros(&[n_blocks], hipfire_rdna::DType::F32).unwrap();
                gpu.qsa_topk_mask(
                    &g_s,
                    &g_bm,
                    n_blocks as i32,
                    p.block_topk().min(n_blocks) as i32,
                )
                .unwrap();
                gpu.qsa_expand_mask(
                    &g_bm,
                    &g_v,
                    &g_tok,
                    n_pos as i32,
                    n_blocks as i32,
                    p.compress_ratio as i32,
                )
                .unwrap();
                gpu.download_f32(&g_tok)
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.to_bits() != 0)
                    .map(|(i, _)| i)
                    .collect()
            };

            let ok = got == want;
            all_ok &= ok;
            let dense = n_pos <= p.dense_below();
            println!(
                "  qsa select n={n_pos:<5} kept={:<5} of {n_pos:<5} dense_expected={dense:<5}  {}",
                got.len(),
                if ok { "ok" } else { "FAIL" }
            );
            // The free exact oracle: at or below the budget nothing may be dropped.
            if dense && got.len() != n_pos {
                println!("    FAIL: below the budget the selection must be everything");
                all_ok = false;
            }
        }
        failed |= !all_ok;
    }

    // ── QSA sparse attention, whole decode path (M21f) ──────────────────────
    // score -> top-k -> slot list -> gather -> `attention_cold_slots`. No new
    // attention kernel: that one already attends an arbitrary compacted slot set.
    //
    // The oracle is exact and free. At or below `dense_below` the selection cannot
    // exclude anything, so the gathered slots must be byte-identical to the dense
    // slot set in the same order => the two attention outputs must agree TO THE
    // BIT. Past it they must differ, or the mask is doing nothing.
    {
        let p = QsaParams {
            n_heads: 8,
            head_dim: 128,
            budget: 64,
            compress_ratio: 4,
        };
        let (nkv, hd) = (2usize, p.head_dim);
        let mut all_ok = true;
        for &n_pos in &[64usize, p.dense_below(), p.dense_below() + 1, 1024] {
            let dense = n_pos <= p.dense_below();
            let q = seeded(p.n_heads * hd, 151);
            let keys = seeded(n_pos * hd, 153); // scoring keys (pooled)
            let ck = seeded(nkv * n_pos * hd, 157); // the real KV cache
            let cv = seeded(nkv * n_pos * hd, 159);
            let visible: Vec<usize> = (0..n_pos).collect();
            let want = qsa_select(&q, &keys, &visible, &p);
            let n_blocks = n_pos / p.compress_ratio;
            let scale = 1.0f32 / (hd as f32).sqrt();

            let g_q = gpu.upload_f32(&q, &[p.n_heads, hd]).unwrap();
            let g_k = gpu.upload_f32(&keys, &[n_pos, hd]).unwrap();
            let g_ck = gpu.upload_f32(&ck, &[nkv, n_pos, hd]).unwrap();
            let g_cv = gpu.upload_f32(&cv, &[nkv, n_pos, hd]).unwrap();
            let vis_f: Vec<f32> = (0..n_pos as i32)
                .map(|v| f32::from_bits(v as u32))
                .collect();
            let g_v = gpu.upload_f32(&vis_f, &[n_pos]).unwrap();

            // sparse arm: score -> mask -> indices
            let g_s = gpu.zeros(&[n_blocks], hipfire_rdna::DType::F32).unwrap();
            gpu.qsa_block_score(
                &g_q,
                &g_k,
                &g_v,
                &g_s,
                p.n_heads as i32,
                hd as i32,
                p.compress_ratio as i32,
                n_blocks as i32,
            )
            .unwrap();
            let g_bm = gpu.zeros(&[n_blocks], hipfire_rdna::DType::F32).unwrap();
            gpu.qsa_topk_mask(
                &g_s,
                &g_bm,
                n_blocks as i32,
                p.block_topk().min(n_blocks) as i32,
            )
            .unwrap();
            let g_idx = gpu.zeros(&[n_pos], hipfire_rdna::DType::F32).unwrap();
            let g_cnt = gpu.zeros(&[1], hipfire_rdna::DType::F32).unwrap();
            gpu.qsa_select_indices(
                &g_bm,
                &g_v,
                &g_idx,
                &g_cnt,
                n_pos as i32,
                n_blocks as i32,
                p.compress_ratio as i32,
            )
            .unwrap();
            let n_sel = gpu.download_f32(&g_cnt).unwrap()[0].to_bits() as usize;
            let got: Vec<usize> = gpu.download_f32(&g_idx).unwrap()[..n_sel]
                .iter()
                .map(|v| v.to_bits() as usize)
                .collect();
            let idx_ok = got == want;
            all_ok &= idx_ok;

            // gather + attend
            let g_sk = gpu
                .zeros(&[nkv, n_sel, hd], hipfire_rdna::DType::F32)
                .unwrap();
            let g_sv = gpu
                .zeros(&[nkv, n_sel, hd], hipfire_rdna::DType::F32)
                .unwrap();
            for (src, dst) in [(&g_ck, &g_sk), (&g_cv, &g_sv)] {
                gpu.qsa_gather_kv(
                    src,
                    &g_idx,
                    dst,
                    nkv as i32,
                    n_sel as i32,
                    hd as i32,
                    n_pos as i32,
                )
                .unwrap();
            }
            let o1 = gpu
                .zeros(&[p.n_heads * hd], hipfire_rdna::DType::F32)
                .unwrap();
            let m1 = gpu.zeros(&[p.n_heads], hipfire_rdna::DType::F32).unwrap();
            let l1 = gpu.zeros(&[p.n_heads], hipfire_rdna::DType::F32).unwrap();
            gpu.attention_cold_slots(
                &g_q, &g_sk, &g_sv, &o1, &m1, &l1, p.n_heads, nkv, n_sel, scale, 0, 0, 0, None, hd,
            )
            .unwrap();
            // dense arm: the full cache, unfiltered
            let o2 = gpu
                .zeros(&[p.n_heads * hd], hipfire_rdna::DType::F32)
                .unwrap();
            let m2 = gpu.zeros(&[p.n_heads], hipfire_rdna::DType::F32).unwrap();
            let l2 = gpu.zeros(&[p.n_heads], hipfire_rdna::DType::F32).unwrap();
            gpu.attention_cold_slots(
                &g_q, &g_ck, &g_cv, &o2, &m2, &l2, p.n_heads, nkv, n_pos, scale, 0, 0, 0, None, hd,
            )
            .unwrap();
            let (a, b) = (
                gpu.download_f32(&o1).unwrap(),
                gpu.download_f32(&o2).unwrap(),
            );
            let bitwise_equal = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());

            let attn_ok = bitwise_equal == dense;
            all_ok &= attn_ok;
            println!(
                "  qsa attend n={n_pos:<5} slots={n_sel:<4} idx={:<4} dense={dense:<5} bit_eq_dense={bitwise_equal:<5}  {}",
                if idx_ok { "ok" } else { "FAIL" },
                if attn_ok { "ok" } else { "FAIL" }
            );
            if !attn_ok {
                println!(
                    "    FAIL: {}",
                    if dense {
                        "below the budget sparse MUST equal dense bit-for-bit"
                    } else {
                        "past the budget sparse must DIFFER from dense (mask is inert)"
                    }
                );
            }
        }
        failed |= !all_ok;
    }

    // ── GDN layer step, end to end (M20b) ────────────────────────────────────
    // Not a parity check — there is no independent CPU recurrence to difference
    // against, and qwen35's GDN kernels already carry their own parity tests. What
    // this proves is that the ASSEMBLY runs at the real geometry, advances state,
    // and produces finite output: the failure modes of a layer driver are shape
    // mismatches and stale state, not arithmetic.
    {
        use hipfire_arch_qwen4exp::gdn::{gdn_decode_step, GdnScratch, GdnState, GdnWeights};
        use hipfire_arch_qwen4exp::Qwen4ExpConfig;

        let cfg = Qwen4ExpConfig::from_slice(include_bytes!(
            "../../hipfire-arch-qwen4exp/tests/qwen3_8_flash_next.config.json"
        ))
        .expect("shipped config");
        let d = cfg.deltanet;
        // `gemv_f32` reads `shape[1]`, so weight matrices must be uploaded 2-D.
        let mat = |g: &mut hipfire_rdna::Gpu, o: usize, i: usize, sd: u32| {
            g.upload_f32(&seeded(o * i, sd), &[o, i]).unwrap()
        };
        let vec1 = |g: &mut hipfire_rdna::Gpu, n: usize, sd: u32| {
            g.upload_f32(&seeded(n, sd), &[n]).unwrap()
        };
        let w = GdnWeights {
            in_proj_qkv: mat(&mut gpu, d.qkv_dim(), cfg.hidden, 101),
            in_proj_z: mat(&mut gpu, d.z_dim(), cfg.hidden, 103),
            in_proj_a: mat(&mut gpu, d.value_heads, cfg.hidden, 105),
            in_proj_b: mat(&mut gpu, d.value_heads, cfg.hidden, 107),
            conv_weight: vec1(&mut gpu, d.qkv_dim() * d.conv_kernel, 109),
            a_log: vec1(&mut gpu, d.value_heads, 111),
            dt_bias: vec1(&mut gpu, d.value_heads, 113),
            norm_weight: vec1(&mut gpu, d.value_head_dim, 115),
            out_proj: mat(&mut gpu, cfg.hidden, d.z_dim(), 117),
        };
        let mut st = GdnState::zeros(&mut gpu, &cfg).unwrap();
        let mut sc = GdnScratch::new(&mut gpu, &cfg).unwrap();
        let y = gpu.zeros(&[cfg.hidden], hipfire_rdna::DType::F32).unwrap();

        let mut ok = true;
        let mut prev: Option<Vec<f32>> = None;
        for t in 0..4 {
            let x = vec1(&mut gpu, cfg.hidden, 121 + t);
            gdn_decode_step(&mut gpu, &cfg, &w, &mut sc, &mut st, &x, &y).expect("gdn step");
            let out = gpu.download_f32(&y).unwrap();
            if !out.iter().all(|v| v.is_finite()) {
                ok = false;
            }
            // The recurrence must actually carry state: feeding the SAME input at
            // t=3 as t=0 must not reproduce t=0's output.
            if t == 0 {
                prev = Some(out.clone());
            }
        }
        // Replay step 0's input against the now-warm state.
        let x0 = vec1(&mut gpu, cfg.hidden, 121);
        gdn_decode_step(&mut gpu, &cfg, &w, &mut sc, &mut st, &x0, &y).unwrap();
        let replay = gpu.download_f32(&y).unwrap();
        let first = prev.unwrap();
        let moved = first
            .iter()
            .zip(&replay)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let stateful = moved > 1e-6;
        if !stateful {
            ok = false;
        }
        println!(
            "  gdn layer  hidden={} qkv={} v_heads={}  finite={} stateful={} (Δ={moved:.3e})  {}",
            cfg.hidden,
            d.qkv_dim(),
            d.value_heads,
            first.iter().all(|v| v.is_finite()),
            stateful,
            if ok { "ok" } else { "FAIL" }
        );
        failed |= !ok;
    }

    if failed {
        eprintln!("parity_qwen4exp_kernels: FAILED (worst {worst:.3e})");
        std::process::exit(1);
    }
    println!("parity_qwen4exp_kernels: OK (worst max|Δ| = {worst:.3e})");
}
