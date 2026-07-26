//! End-to-end **Qwen3.5-A3B routed-expert FFN** on the NPU, W4A8 op4++, at exact
//! shape (hidden=2048, moe_intermediate=768, SwiGLU), composed from the proven
//! projection GEMMs: fused **gate_up** (K=2048→N=1536, fullk kg8) → **SiLU-mul**
//! → **down** (K=768→N=2048, whole-scaled kg3_n2304, view first 2048).
//!
//! Milestone M2 / path-a (`docs/plans/2026-07-17-npu-w4a8-op4pp-moe-qwen35.md`):
//! prove the composed expert FFN is numerically correct end-to-end on silicon
//! before pursuing a single-dispatch fused kernel. The NPU path
//! (`run_f32` → silu_mul → `run_f32`) is checked against the same chain through
//! the CPU op4++ oracle (`reference_f32` → silu_mul → `reference_f32`); both use
//! int8 activations, so agreement should be near-bit-exact.
//!
//! SiLU-mul is host-side here (cheap vs the GEMMs); moving it + the inter-GEMM
//! handoff on-device (shared dma-buf) is the perf follow-on that yields a clean
//! device-time PP/TG number. Hold `hipfire lock` while running.
//!
//! Usage: `npu_expert_ffn_w4_parity [GATE_UP_CACHE DOWN_CACHE] [--iters N]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_xdna::NpuOpusGemmMp;

    const HID: usize = 2048; // hidden = K(gate_up) = N(down, viewed)
    const MI: usize = 768; // moe_intermediate = N(each gate/up half) = K(down)
    const GU_N: usize = 1536; // gate || up
    const DOWN_N: usize = 2304; // down cache N (view first HID)

    let home = std::env::var("HOME")?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let positional = args
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .collect::<Vec<_>>();
    let gu_cache = positional
        .first()
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            format!("{home}/.hipfire/npu/embgemma_aie2p_fullk_submit_w4-scaled_m256_kg8_n1536")
        });
    let down_cache = positional.get(1).map(|s| s.to_string()).unwrap_or_else(|| {
        format!("{home}/.hipfire/npu/embgemma_aie2p_whole8_w4-scaled_m256_kg3_n2304")
    });
    let iterations = args
        .iter()
        .position(|a| a == "--iters")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10);

    let gu_payload = w4_payload(HID, GU_N, 3);
    let down_payload = w4_payload(MI, DOWN_N, 23);
    let mut gate_up =
        NpuOpusGemmMp::load_fullk_only(&gu_cache, 8, 33, HID, GU_N, &gu_payload, None)?;
    let mut down =
        NpuOpusGemmMp::load_whole_scaled_only(&down_cache, 33, MI, DOWN_N, &down_payload, None)?;
    let m = gate_up.rows_per_dispatch();
    assert_eq!(
        m,
        down.rows_per_dispatch(),
        "gate_up and down disagree on M"
    );

    let x = (0..m * HID)
        .map(|i| ((i as f32 * 0.0011).sin() * 1.5) + ((i % 13) as f32 - 6.0) * 0.05)
        .collect::<Vec<_>>();

    // SiLU-mul: inter[r,j] = silu(gu[r,j]) * gu[r, MI+j], silu(z)=z·sigmoid(z).
    let silu_mul = |gu: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; m * MI];
        for r in 0..m {
            for j in 0..MI {
                let g = gu[r * GU_N + j];
                let u = gu[r * GU_N + MI + j];
                out[r * MI + j] = (g / (1.0 + (-g).exp())) * u;
            }
        }
        out
    };
    // First HID columns of a [m, DOWN_N] down output = the real expert output.
    let view = |buf: &[f32]| -> Vec<f32> {
        (0..m)
            .flat_map(|r| buf[r * DOWN_N..r * DOWN_N + HID].to_vec())
            .collect()
    };

    // CPU op4++ reference chain (int8 activations, same as the NPU GEMMs).
    let gu_ref = gate_up.reference_f32(m, &x)?;
    let inter_ref = silu_mul(&gu_ref);
    let down_ref = down.reference_f32(m, &inter_ref)?;
    let reference = view(&down_ref);

    // NPU chain.
    let mut gu_out = vec![0.0f32; m * GU_N];
    let mut down_out = vec![0.0f32; m * DOWN_N];
    let mut run =
        |gu_out: &mut [f32], down_out: &mut [f32]| -> Result<(), hipfire_xdna::XdnaError> {
            gate_up.run_f32(m, &x, gu_out)?;
            let inter = silu_mul(gu_out);
            down.run_f32(m, &inter, down_out)
        };
    run(&mut gu_out, &mut down_out)?;
    let got = view(&down_out);

    let (cosine, max_abs, db) = metrics(&got, &reference);
    let pass = cosine >= 0.999 && db > 30.0;
    if !pass {
        for idx in [0usize, 1, HID - 1, HID, m * HID - 1] {
            eprintln!("ffn[{idx}] got={:.6} ref={:.6}", got[idx], reference[idx]);
        }
        return Err(format!("expert FFN parity FAILED: cosine={cosine:.8} db={db:.2}").into());
    }

    for _ in 0..2 {
        run(&mut gu_out, &mut down_out)?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        run(&mut gu_out, &mut down_out)?;
    }
    let ffn_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "npu_expert_ffn_w4 (Qwen3.5-A3B) M={m} hidden={HID} moe_inter={MI} SwiGLU: cosine={cosine:.8} max_abs={max_abs:.6} SQNR={db:.2} dB  ffn_ms(host-prep-incl)={ffn_ms:.4} iters={iterations} -> PASS"
    );
    println!("  note: ffn_ms includes host activation-prep + readback for BOTH GEMMs + host silu; device-resident chaining is the PP/TG follow-on.");
    Ok(())
}

/// op4++ weights in the qt=33 W4 on-disk layout (130-byte blocks: fp16 scale +
/// 128 nibbles, low = inner 2j, high = inner 2j+1), block order col-major.
#[cfg(target_os = "linux")]
fn w4_payload(k: usize, n: usize, seed: usize) -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_f16;
    const GROUP: usize = 256;
    let groups = k / GROUP;
    let mut payload = vec![0u8; n * groups * 130];
    for col in 0..n {
        for g in 0..groups {
            let block = &mut payload[(col * groups + g) * 130..(col * groups + g + 1) * 130];
            let scale = 0.012 * (1.0 + ((col + 3 * g + seed) % 7) as f32 * 0.03);
            block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for j in 0..128 {
                let val = |inner: usize| -> u8 {
                    let mixed = (inner as u64).wrapping_mul(0x9e37_79b1)
                        ^ (col as u64).wrapping_mul(0x85eb_ca77)
                        ^ (g as u64).wrapping_mul(0xc2b2_ae3d)
                        ^ (seed as u64).wrapping_mul(0x27d4_eb2f);
                    (((mixed % 15) as i8 - 7) as u8) & 0x0f
                };
                block[2 + j] = val(g * GROUP + 2 * j) | (val(g * GROUP + 2 * j + 1) << 4);
            }
        }
    }
    payload
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32, f64) {
    let mut dot = 0.0;
    let (mut gn, mut en) = (0.0f64, 0.0f64);
    let mut max_abs = 0.0f32;
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for (&g, &e) in got.iter().zip(expected) {
        max_abs = max_abs.max((g - e).abs());
        dot += g as f64 * e as f64;
        gn += (g as f64).powi(2);
        en += (e as f64).powi(2);
        sig += (e as f64).powi(2);
        noise += ((e - g) as f64).powi(2);
    }
    (
        dot / (gn.sqrt() * en.sqrt()),
        max_abs,
        10.0 * (sig / noise.max(1e-30)).log10(),
    )
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("resident AIE2P expert FFN verification is Linux-only");
}
