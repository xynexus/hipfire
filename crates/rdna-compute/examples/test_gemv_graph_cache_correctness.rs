//! PR2 of the per-shape hipGraph cache: bit-exact correctness test for
//! plain `gemv_hfq4g256` capture/replay.
//!
//! Generates 100 calls with N distinct (a, x, y) buffer triples chosen
//! from a small pool — each call uses a different y so the cache MUST
//! rewrite the y_ptr slot in the cached kernarg blob between replays.
//! The same M/K shape is used across all 100 calls so the cache hits
//! the same shape entry every time.
//!
//! The test runs the same call sequence twice:
//!   1. cache disabled (sequential launch every call) → record y_seq[100]
//!   2. cache enabled  (capture-on-second, replay-on-third+) → y_graph[100]
//!
//! Both arrays are compared byte-exact. Any mismatch fails the test.
//!
//! The test also prints cache stats at the end (hits / misses / captures /
//! hit_rate) so the operator can verify the capture/replay routing
//! actually triggered (>1 hit, == 1 capture, == 1 miss for a single shape).
//!
//! Run with:
//!   cargo run --release -p rdna-compute --example test_gemv_graph_cache_correctness

use rdna_compute::{DType, Gpu};

const M: usize = 1024;
const K: usize = 1024;
const N_CALLS: usize = 100;
const N_BUFFER_SLOTS: usize = 8; // pool of y buffers; each call picks one

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

fn main() {
    eprintln!("=== test_gemv_graph_cache_correctness ===");
    eprintln!("M={M} K={K} N_CALLS={N_CALLS} N_BUFFER_SLOTS={N_BUFFER_SLOTS}");

    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("GPU: {} (device {})", gpu.arch, gpu.device_id);

    // Build A weights (HFQ4-G256 layout, 1 row × 4 groups × 136 bytes).
    let groups_per_row = K / 256;
    let a_bytes = synth_hfq4g256_weights(M, groups_per_row, 0xC0DE_FACEu64);
    let a_tensor = gpu.upload_raw(&a_bytes, &[M, K]).expect("upload A");

    // Build a pool of distinct X tensors. Each call picks one — same shape
    // (K floats), different content → different output, exercising the
    // x_ptr slot rewrite path on replay.
    let mut x_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    for slot in 0..N_BUFFER_SLOTS {
        let x_data: Vec<f32> = (0..K)
            .map(|i| {
                let phase = (slot as f32) * 0.137 + (i as f32) * 0.0123;
                phase.sin() * 0.5
            })
            .collect();
        x_pool.push(gpu.upload_f32(&x_data, &[K]).expect("upload x"));
    }

    // Build a pool of distinct Y tensors. Same reason — exercises the
    // y_ptr slot rewrite. Y is initialized to known sentinel bytes so we
    // catch dispatch-skipped-write bugs.
    let mut y_pool: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_BUFFER_SLOTS);
    for _ in 0..N_BUFFER_SLOTS {
        y_pool.push(gpu.zeros(&[M], DType::F32).expect("alloc y"));
    }

    // Generate the call schedule: 100 calls, each picking (x_idx, y_idx).
    // Deterministic RNG so the two runs (disabled / enabled) issue the
    // same sequence of (x, y) selections.
    let mut rng_state: u64 = 0xDEAD_BEEF_C0DE_FACE;
    let schedule: Vec<(usize, usize)> = (0..N_CALLS)
        .map(|_| {
            let xi = (lcg_next(&mut rng_state) >> 33) as usize % N_BUFFER_SLOTS;
            let yi = (lcg_next(&mut rng_state) >> 33) as usize % N_BUFFER_SLOTS;
            (xi, yi)
        })
        .collect();

    // ── Run 1: cache DISABLED (sequential launch each call) ────────────
    gpu.disable_gemv_graph_cache_for_test();
    let mut y_seq_outputs: Vec<Vec<f32>> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        // Re-zero the y buffer so each call's output is independent of
        // any prior accumulation/garbage. (gemv_hfq4g256 sets y, doesn't
        // accumulate — but be defensive.)
        let y = &y_pool[*yi];
        let zero = vec![0u8; M * 4];
        gpu.hip.memcpy_htod(&y.buf, &zero).expect("zero y");

        gpu.gemv_hfq4g256(&a_tensor, &x_pool[*xi], y, M, K)
            .expect("seq gemv");
        let y_out = gpu.download_f32(y).expect("download y_seq");
        y_seq_outputs.push(y_out);
    }
    eprintln!("Run 1 (cache disabled): {} sequential calls done", y_seq_outputs.len());

    // ── Run 2: cache ENABLED (capture/replay) ─────────────────────────
    gpu.enable_gemv_graph_cache_for_test()
        .expect("enable_gemv_graph_cache_for_test");
    let mut y_graph_outputs: Vec<Vec<f32>> = Vec::with_capacity(N_CALLS);
    for (xi, yi) in &schedule {
        let y = &y_pool[*yi];
        let zero = vec![0u8; M * 4];
        gpu.hip.memcpy_htod(&y.buf, &zero).expect("zero y");

        gpu.gemv_hfq4g256(&a_tensor, &x_pool[*xi], y, M, K)
            .expect("graph gemv");
        let y_out = gpu.download_f32(y).expect("download y_graph");
        y_graph_outputs.push(y_out);
    }
    eprintln!("Run 2 (cache enabled): {} calls done", y_graph_outputs.len());

    let stats = gpu.gemv_graph_stats().expect("stats present");
    eprintln!("Cache stats: {stats}");

    // ── bit-exact compare ─────────────────────────────────────────────
    let mut total_mismatch: usize = 0;
    let mut first_mismatch_call: Option<usize> = None;
    let mut first_mismatch_row: Option<usize> = None;
    let mut first_mismatch_seq: f32 = 0.0;
    let mut first_mismatch_graph: f32 = 0.0;
    for (call_idx, (y_seq, y_graph)) in y_seq_outputs.iter().zip(y_graph_outputs.iter()).enumerate() {
        for row in 0..M {
            if y_seq[row].to_bits() != y_graph[row].to_bits() {
                total_mismatch += 1;
                if first_mismatch_call.is_none() {
                    first_mismatch_call = Some(call_idx);
                    first_mismatch_row = Some(row);
                    first_mismatch_seq = y_seq[row];
                    first_mismatch_graph = y_graph[row];
                }
            }
        }
    }

    if total_mismatch > 0 {
        eprintln!(
            "FAIL: {total_mismatch} mismatched (call,row) pairs out of {} ({} per-call avg)",
            N_CALLS * M,
            total_mismatch as f64 / N_CALLS as f64
        );
        if let (Some(c), Some(r)) = (first_mismatch_call, first_mismatch_row) {
            eprintln!(
                "  first mismatch: call={c} row={r}  seq={first_mismatch_seq} ({:08x})  graph={first_mismatch_graph} ({:08x})",
                first_mismatch_seq.to_bits(), first_mismatch_graph.to_bits(),
            );
        }
        std::process::exit(1);
    }

    // Sanity: cache must have produced at least one capture and at least
    // one replay. If not, the test was a no-op (bit-equality is trivial
    // when we just ran sequential both times).
    if stats.captures == 0 {
        eprintln!("FAIL: cache produced 0 captures — routing did not engage");
        std::process::exit(1);
    }
    if stats.hits == 0 {
        eprintln!("FAIL: cache produced 0 hits — replay path never executed");
        std::process::exit(1);
    }

    println!("PASS  100 calls bit-exact across cache disabled vs enabled");
    println!("      hits={} misses={} captures={} hit_rate={:.1}%",
        stats.hits, stats.misses, stats.captures,
        (stats.hits as f64) / ((stats.hits + stats.misses) as f64) * 100.0);
}
