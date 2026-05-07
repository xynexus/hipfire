//! PR3 of the per-shape hipGraph cache: bit-exact correctness test for
//! the four fused families wired by PR3:
//!
//!   - `gemv_hfq4g256_residual` (wo / w_down projections)
//!   - `fused_qkv_hfq4g256` (FA layer preamble)
//!   - `fused_qkvza_hfq4g256` (LA layer preamble)
//!   - `fused_gate_up_hfq4g256` (MLP preamble)
//!
//! For each family, the test runs the same call sequence twice:
//!   1. cache disabled (sequential launch every call) → record y_seq[N]
//!   2. cache enabled  (capture-on-second, replay-on-third+) → y_graph[N]
//!
//! Both arrays are compared byte-exact. Any mismatch fails the test.
//!
//! Schedule mirrors the plain GEMV test: 50 calls with the same buffer
//! triple (steady-state replay path), then 50 calls with rotating buffers
//! (eviction + recapture path).
//!
//! Run with:
//!   cargo run --release -p rdna-compute --example test_gemv_graph_cache_correctness_pr3

use rdna_compute::{DType, Gpu};

const K: usize = 1024;
const N_CALLS: usize = 100;
const N_BUFFER_SLOTS: usize = 8;

fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let scale_target = 1.0e-3f32;
    let zp_max = 1.0f32;
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            let scale = scale_target * (0.5 + (next() & 0xFFFF) as f32 / 65535.0 * 1.5);
            let zp = ((next() & 0xFFFF) as f32 / 65535.0) * 2.0 * zp_max - zp_max;
            out[gp..gp + 4].copy_from_slice(&scale.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn build_x_pool(gpu: &mut Gpu, k: usize) -> Vec<rdna_compute::GpuTensor> {
    let mut x_pool = Vec::with_capacity(N_BUFFER_SLOTS);
    for slot in 0..N_BUFFER_SLOTS {
        let x_data: Vec<f32> = (0..k)
            .map(|i| {
                let phase = (slot as f32) * 0.137 + (i as f32) * 0.0123;
                phase.sin() * 0.5
            })
            .collect();
        x_pool.push(gpu.upload_f32(&x_data, &[k]).expect("upload x"));
    }
    x_pool
}

/// Build an (xi, yi) call schedule. Phase A: 50 calls with (0, 0).
/// Phase B: 50 calls with rotating slots.
fn build_schedule() -> Vec<(usize, usize)> {
    let mut rng_state: u64 = 0xDEAD_BEEF_C0DE_FACE;
    (0..N_CALLS)
        .map(|i| {
            if i < N_CALLS / 2 {
                (0, 0)
            } else {
                let xi = (lcg_next(&mut rng_state) >> 33) as usize % N_BUFFER_SLOTS;
                let yi = (lcg_next(&mut rng_state) >> 33) as usize % N_BUFFER_SLOTS;
                (xi, yi)
            }
        })
        .collect()
}

fn assert_bit_exact(
    name: &str,
    seq: &[Vec<f32>],
    graph: &[Vec<f32>],
) -> bool {
    let mut total_mismatch: usize = 0;
    let mut first: Option<(usize, usize, f32, f32)> = None;
    for (call_idx, (a, b)) in seq.iter().zip(graph.iter()).enumerate() {
        for row in 0..a.len() {
            if a[row].to_bits() != b[row].to_bits() {
                total_mismatch += 1;
                if first.is_none() {
                    first = Some((call_idx, row, a[row], b[row]));
                }
            }
        }
    }
    if total_mismatch > 0 {
        eprintln!("FAIL [{name}]: {total_mismatch} mismatched (call,row) pairs");
        if let Some((c, r, sv, gv)) = first {
            eprintln!(
                "  first mismatch: call={c} row={r}  seq={sv} ({:08x})  graph={gv} ({:08x})",
                sv.to_bits(), gv.to_bits(),
            );
        }
        return false;
    }
    true
}

/// Family 1: gemv_hfq4g256_residual.
/// Test wo-projection shape: M=1024, K=1024.
fn test_residual(gpu: &mut Gpu) -> bool {
    eprintln!("\n--- Family 1/4: gemv_hfq4g256_residual ---");
    const M: usize = 1024;

    let groups_per_row = K / 256;
    let a_bytes = synth_hfq4g256_weights(M, groups_per_row, 0xC0DE_FACE_0001u64);
    let a = gpu.upload_raw(&a_bytes, &[M, K]).expect("upload A");

    let x_pool = build_x_pool(gpu, K);
    // Initial y values per slot (residual reads + writes y).
    let mut y_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    let mut y_init_bytes: Vec<Vec<u8>> = Vec::with_capacity(N_BUFFER_SLOTS);
    for slot in 0..N_BUFFER_SLOTS {
        let y_data: Vec<f32> = (0..M)
            .map(|i| ((slot * 7 + i) as f32) * 1e-4)
            .collect();
        y_pool.push(gpu.upload_f32(&y_data, &[M]).expect("upload y"));
        let mut bytes = vec![0u8; M * 4];
        for (i, v) in y_data.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        y_init_bytes.push(bytes);
    }

    let schedule = build_schedule();

    // Run 1: cache disabled.
    gpu.disable_gemv_graph_cache_for_test();
    let mut y_seq: Vec<Vec<f32>> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        let y = &y_pool[*yi];
        // Reset y to its initial value so each call's residual is independent.
        gpu.hip.memcpy_htod(&y.buf, &y_init_bytes[*yi]).expect("reset y");
        gpu.gemv_hfq4g256_residual(&a, &x_pool[*xi], y, M, K)
            .expect("seq residual");
        let out = gpu.download_f32(y).expect("dl y_seq");
        y_seq.push(out);
    }
    eprintln!("  Run 1 (disabled): {} calls", y_seq.len());

    // Run 2: cache enabled.
    gpu.enable_gemv_graph_cache_for_test().expect("enable");
    let mut y_graph: Vec<Vec<f32>> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        let y = &y_pool[*yi];
        gpu.hip.memcpy_htod(&y.buf, &y_init_bytes[*yi]).expect("reset y");
        gpu.gemv_hfq4g256_residual(&a, &x_pool[*xi], y, M, K)
            .expect("graph residual");
        let out = gpu.download_f32(y).expect("dl y_graph");
        y_graph.push(out);
    }
    eprintln!("  Run 2 (enabled):  {} calls", y_graph.len());
    if let Some(s) = gpu.gemv_graph_stats() {
        eprintln!("  cache stats: {s}");
    }
    assert_bit_exact("residual", &y_seq, &y_graph)
}

/// Family 2: fused_qkv_hfq4g256.
/// Test FA-preamble shape: q_m=1024, k_m=256, v_m=256, K=1024.
fn test_fused_qkv(gpu: &mut Gpu) -> bool {
    eprintln!("\n--- Family 2/4: fused_qkv_hfq4g256 ---");
    const Q_M: usize = 1024;
    const K_M: usize = 256;
    const V_M: usize = 256;

    let groups = K / 256;
    let aq_bytes = synth_hfq4g256_weights(Q_M, groups, 0xC0DE_FACE_0002u64);
    let ak_bytes = synth_hfq4g256_weights(K_M, groups, 0xC0DE_FACE_0003u64);
    let av_bytes = synth_hfq4g256_weights(V_M, groups, 0xC0DE_FACE_0004u64);
    let aq = gpu.upload_raw(&aq_bytes, &[Q_M, K]).expect("a_q");
    let ak = gpu.upload_raw(&ak_bytes, &[K_M, K]).expect("a_k");
    let av = gpu.upload_raw(&av_bytes, &[V_M, K]).expect("a_v");

    let x_pool = build_x_pool(gpu, K);
    // Three independent y pools for q/k/v.
    let mut yq_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    let mut yk_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    let mut yv_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    for _ in 0..N_BUFFER_SLOTS {
        yq_pool.push(gpu.zeros(&[Q_M], DType::F32).expect("yq"));
        yk_pool.push(gpu.zeros(&[K_M], DType::F32).expect("yk"));
        yv_pool.push(gpu.zeros(&[V_M], DType::F32).expect("yv"));
    }
    let schedule = build_schedule();

    let zero_q = vec![0u8; Q_M * 4];
    let zero_k = vec![0u8; K_M * 4];
    let zero_v = vec![0u8; V_M * 4];

    // Run 1: disabled.
    gpu.disable_gemv_graph_cache_for_test();
    let mut seq: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        let yq = &yq_pool[*yi];
        let yk = &yk_pool[*yi];
        let yv = &yv_pool[*yi];
        gpu.hip.memcpy_htod(&yq.buf, &zero_q).expect("zero yq");
        gpu.hip.memcpy_htod(&yk.buf, &zero_k).expect("zero yk");
        gpu.hip.memcpy_htod(&yv.buf, &zero_v).expect("zero yv");
        gpu.fused_qkv_hfq4g256(&aq, &ak, &av, &x_pool[*xi], yq, yk, yv, Q_M, K_M, V_M, K)
            .expect("seq fused_qkv");
        seq.push((
            gpu.download_f32(yq).unwrap(),
            gpu.download_f32(yk).unwrap(),
            gpu.download_f32(yv).unwrap(),
        ));
    }
    eprintln!("  Run 1 (disabled): {} calls", seq.len());

    // Run 2: enabled.
    gpu.enable_gemv_graph_cache_for_test().expect("enable");
    let mut graph: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        let yq = &yq_pool[*yi];
        let yk = &yk_pool[*yi];
        let yv = &yv_pool[*yi];
        gpu.hip.memcpy_htod(&yq.buf, &zero_q).expect("zero yq");
        gpu.hip.memcpy_htod(&yk.buf, &zero_k).expect("zero yk");
        gpu.hip.memcpy_htod(&yv.buf, &zero_v).expect("zero yv");
        gpu.fused_qkv_hfq4g256(&aq, &ak, &av, &x_pool[*xi], yq, yk, yv, Q_M, K_M, V_M, K)
            .expect("graph fused_qkv");
        graph.push((
            gpu.download_f32(yq).unwrap(),
            gpu.download_f32(yk).unwrap(),
            gpu.download_f32(yv).unwrap(),
        ));
    }
    eprintln!("  Run 2 (enabled):  {} calls", graph.len());
    if let Some(s) = gpu.gemv_graph_stats() {
        eprintln!("  cache stats: {s}");
    }

    let q_ok = assert_bit_exact("fused_qkv:q",
        &seq.iter().map(|t| t.0.clone()).collect::<Vec<_>>(),
        &graph.iter().map(|t| t.0.clone()).collect::<Vec<_>>());
    let k_ok = assert_bit_exact("fused_qkv:k",
        &seq.iter().map(|t| t.1.clone()).collect::<Vec<_>>(),
        &graph.iter().map(|t| t.1.clone()).collect::<Vec<_>>());
    let v_ok = assert_bit_exact("fused_qkv:v",
        &seq.iter().map(|t| t.2.clone()).collect::<Vec<_>>(),
        &graph.iter().map(|t| t.2.clone()).collect::<Vec<_>>());
    q_ok && k_ok && v_ok
}

/// Family 3: fused_qkvza_hfq4g256.
/// Test LA-preamble shape: qkv_m=768, z_m=256, beta_m=32, alpha_m=32, K=1024.
fn test_fused_qkvza(gpu: &mut Gpu) -> bool {
    eprintln!("\n--- Family 3/4: fused_qkvza_hfq4g256 ---");
    const QKV_M: usize = 768;
    const Z_M: usize = 256;
    const BETA_M: usize = 32;
    const ALPHA_M: usize = 32;

    let groups = K / 256;
    let a_qkv = gpu.upload_raw(
        &synth_hfq4g256_weights(QKV_M, groups, 0xC0DE_FACE_0010u64),
        &[QKV_M, K]).expect("aqkv");
    let a_z = gpu.upload_raw(
        &synth_hfq4g256_weights(Z_M, groups, 0xC0DE_FACE_0011u64),
        &[Z_M, K]).expect("az");
    let a_beta = gpu.upload_raw(
        &synth_hfq4g256_weights(BETA_M, groups, 0xC0DE_FACE_0012u64),
        &[BETA_M, K]).expect("ab");
    let a_alpha = gpu.upload_raw(
        &synth_hfq4g256_weights(ALPHA_M, groups, 0xC0DE_FACE_0013u64),
        &[ALPHA_M, K]).expect("aa");

    let x_pool = build_x_pool(gpu, K);
    let mut yq_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    let mut yz_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    let mut yb_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    let mut ya_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    for _ in 0..N_BUFFER_SLOTS {
        yq_pool.push(gpu.zeros(&[QKV_M], DType::F32).expect("yq"));
        yz_pool.push(gpu.zeros(&[Z_M], DType::F32).expect("yz"));
        yb_pool.push(gpu.zeros(&[BETA_M], DType::F32).expect("yb"));
        ya_pool.push(gpu.zeros(&[ALPHA_M], DType::F32).expect("ya"));
    }
    let schedule = build_schedule();

    let zq = vec![0u8; QKV_M * 4];
    let zz = vec![0u8; Z_M * 4];
    let zb = vec![0u8; BETA_M * 4];
    let za = vec![0u8; ALPHA_M * 4];

    gpu.disable_gemv_graph_cache_for_test();
    let mut seq: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        let yq = &yq_pool[*yi]; let yz = &yz_pool[*yi];
        let yb = &yb_pool[*yi]; let ya = &ya_pool[*yi];
        gpu.hip.memcpy_htod(&yq.buf, &zq).unwrap();
        gpu.hip.memcpy_htod(&yz.buf, &zz).unwrap();
        gpu.hip.memcpy_htod(&yb.buf, &zb).unwrap();
        gpu.hip.memcpy_htod(&ya.buf, &za).unwrap();
        gpu.fused_qkvza_hfq4g256(
            &a_qkv, &a_z, &a_beta, &a_alpha,
            &x_pool[*xi],
            yq, yz, yb, ya,
            QKV_M, Z_M, BETA_M, ALPHA_M, K,
        ).expect("seq fused_qkvza");
        seq.push((
            gpu.download_f32(yq).unwrap(),
            gpu.download_f32(yz).unwrap(),
            gpu.download_f32(yb).unwrap(),
            gpu.download_f32(ya).unwrap(),
        ));
    }
    eprintln!("  Run 1 (disabled): {} calls", seq.len());

    gpu.enable_gemv_graph_cache_for_test().expect("enable");
    let mut graph: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        let yq = &yq_pool[*yi]; let yz = &yz_pool[*yi];
        let yb = &yb_pool[*yi]; let ya = &ya_pool[*yi];
        gpu.hip.memcpy_htod(&yq.buf, &zq).unwrap();
        gpu.hip.memcpy_htod(&yz.buf, &zz).unwrap();
        gpu.hip.memcpy_htod(&yb.buf, &zb).unwrap();
        gpu.hip.memcpy_htod(&ya.buf, &za).unwrap();
        gpu.fused_qkvza_hfq4g256(
            &a_qkv, &a_z, &a_beta, &a_alpha,
            &x_pool[*xi],
            yq, yz, yb, ya,
            QKV_M, Z_M, BETA_M, ALPHA_M, K,
        ).expect("graph fused_qkvza");
        graph.push((
            gpu.download_f32(yq).unwrap(),
            gpu.download_f32(yz).unwrap(),
            gpu.download_f32(yb).unwrap(),
            gpu.download_f32(ya).unwrap(),
        ));
    }
    eprintln!("  Run 2 (enabled):  {} calls", graph.len());
    if let Some(s) = gpu.gemv_graph_stats() {
        eprintln!("  cache stats: {s}");
    }

    let oq = assert_bit_exact("fused_qkvza:qkv",
        &seq.iter().map(|t| t.0.clone()).collect::<Vec<_>>(),
        &graph.iter().map(|t| t.0.clone()).collect::<Vec<_>>());
    let oz = assert_bit_exact("fused_qkvza:z",
        &seq.iter().map(|t| t.1.clone()).collect::<Vec<_>>(),
        &graph.iter().map(|t| t.1.clone()).collect::<Vec<_>>());
    let ob = assert_bit_exact("fused_qkvza:beta",
        &seq.iter().map(|t| t.2.clone()).collect::<Vec<_>>(),
        &graph.iter().map(|t| t.2.clone()).collect::<Vec<_>>());
    let oa = assert_bit_exact("fused_qkvza:alpha",
        &seq.iter().map(|t| t.3.clone()).collect::<Vec<_>>(),
        &graph.iter().map(|t| t.3.clone()).collect::<Vec<_>>());
    oq && oz && ob && oa
}

/// Family 4: fused_gate_up_hfq4g256.
/// Test MLP-preamble shape: gate_m=2816, up_m=2816, K=1024.
fn test_fused_gate_up(gpu: &mut Gpu) -> bool {
    eprintln!("\n--- Family 4/4: fused_gate_up_hfq4g256 ---");
    const GATE_M: usize = 2816;
    const UP_M: usize = 2816;

    let groups = K / 256;
    let a_gate = gpu.upload_raw(
        &synth_hfq4g256_weights(GATE_M, groups, 0xC0DE_FACE_0020u64),
        &[GATE_M, K]).expect("a_gate");
    let a_up = gpu.upload_raw(
        &synth_hfq4g256_weights(UP_M, groups, 0xC0DE_FACE_0021u64),
        &[UP_M, K]).expect("a_up");

    let x_pool = build_x_pool(gpu, K);
    let mut yg_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    let mut yu_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    for _ in 0..N_BUFFER_SLOTS {
        yg_pool.push(gpu.zeros(&[GATE_M], DType::F32).expect("yg"));
        yu_pool.push(gpu.zeros(&[UP_M], DType::F32).expect("yu"));
    }
    let schedule = build_schedule();
    let zg = vec![0u8; GATE_M * 4];
    let zu = vec![0u8; UP_M * 4];

    gpu.disable_gemv_graph_cache_for_test();
    let mut seq: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        let yg = &yg_pool[*yi]; let yu = &yu_pool[*yi];
        gpu.hip.memcpy_htod(&yg.buf, &zg).unwrap();
        gpu.hip.memcpy_htod(&yu.buf, &zu).unwrap();
        gpu.fused_gate_up_hfq4g256(&a_gate, &a_up, &x_pool[*xi], yg, yu, GATE_M, UP_M, K)
            .expect("seq fused_gate_up");
        seq.push((gpu.download_f32(yg).unwrap(), gpu.download_f32(yu).unwrap()));
    }
    eprintln!("  Run 1 (disabled): {} calls", seq.len());

    gpu.enable_gemv_graph_cache_for_test().expect("enable");
    let mut graph: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        let yg = &yg_pool[*yi]; let yu = &yu_pool[*yi];
        gpu.hip.memcpy_htod(&yg.buf, &zg).unwrap();
        gpu.hip.memcpy_htod(&yu.buf, &zu).unwrap();
        gpu.fused_gate_up_hfq4g256(&a_gate, &a_up, &x_pool[*xi], yg, yu, GATE_M, UP_M, K)
            .expect("graph fused_gate_up");
        graph.push((gpu.download_f32(yg).unwrap(), gpu.download_f32(yu).unwrap()));
    }
    eprintln!("  Run 2 (enabled):  {} calls", graph.len());
    if let Some(s) = gpu.gemv_graph_stats() {
        eprintln!("  cache stats: {s}");
    }

    let og = assert_bit_exact("fused_gate_up:gate",
        &seq.iter().map(|t| t.0.clone()).collect::<Vec<_>>(),
        &graph.iter().map(|t| t.0.clone()).collect::<Vec<_>>());
    let ou = assert_bit_exact("fused_gate_up:up",
        &seq.iter().map(|t| t.1.clone()).collect::<Vec<_>>(),
        &graph.iter().map(|t| t.1.clone()).collect::<Vec<_>>());
    og && ou
}

fn main() {
    eprintln!("=== test_gemv_graph_cache_correctness_pr3 ===");
    eprintln!("K={K}  N_CALLS={N_CALLS}  N_BUFFER_SLOTS={N_BUFFER_SLOTS}");
    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("GPU: {} (device {})", gpu.arch, gpu.device_id);

    // Sanity: cache should exist after enable_for_test, but final stat
    // assertion happens per-family (each family disables/enables fresh).
    let r1 = test_residual(&mut gpu);
    let r2 = test_fused_qkv(&mut gpu);
    let r3 = test_fused_qkvza(&mut gpu);
    let r4 = test_fused_gate_up(&mut gpu);

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("  gemv_hfq4g256_residual:  {}", if r1 { "PASS" } else { "FAIL" });
    eprintln!("  fused_qkv_hfq4g256:      {}", if r2 { "PASS" } else { "FAIL" });
    eprintln!("  fused_qkvza_hfq4g256:    {}", if r3 { "PASS" } else { "FAIL" });
    eprintln!("  fused_gate_up_hfq4g256:  {}", if r4 { "PASS" } else { "FAIL" });
    if !(r1 && r2 && r3 && r4) {
        std::process::exit(1);
    }

    // Final: confirm the cache actually engaged (ran captures + replays).
    // After the last family (fused_gate_up), `enable_for_test` was
    // called fresh and ran 100 calls. We expect at least 1 capture and
    // at least 1 hit.
    let stats = gpu.gemv_graph_stats().expect("stats");
    if stats.captures == 0 {
        eprintln!("FAIL: cache produced 0 captures across all families");
        std::process::exit(1);
    }
    if stats.hits == 0 {
        eprintln!("FAIL: cache produced 0 hits across all families");
        std::process::exit(1);
    }

    println!("PASS  4 families bit-exact across cache disabled vs enabled");
    println!("      final stats: {stats}");
}
