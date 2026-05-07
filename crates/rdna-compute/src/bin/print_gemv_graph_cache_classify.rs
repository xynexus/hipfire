//! Smoke-test the GEMV graph cache classifier without touching the GPU.
//!
//! Exercises every kernel family + variant listed in the PRD section 1
//! (`docs/plans/gemv-graph-cache.prd`) plus a couple of negative cases,
//! and prints the resulting `GemvShape` (or "unrecognized" for negatives).
//!
//! Run with:
//!     cargo run -p rdna-compute --bin print_gemv_graph_cache_classify

use rdna_compute::GemvGraphCache;

fn show(label: &str, func_name: &str, m_tuple: &[u32], k: u32) {
    print!("{label:38} func={func_name:38} m={m_tuple:?} k={k:5} -> ");
    match GemvGraphCache::classify(func_name, m_tuple, k) {
        Some(shape) => println!(
            "GemvShape {{ kernel: {:?}, m_tuple: {}, k: {} }}",
            shape.kernel, shape.m_tuple, shape.k
        ),
        None => println!("unrecognized"),
    }
}

fn main() {
    println!("== Qwen3.5 0.8B decode shape inventory (PRD section 1) ==\n");

    // FA layers: fused_qkv_hfq4g256
    // (q=1024, k=256, v=256), K=1024, 6 calls/token (6 FA layers)
    show(
        "FA fused_qkv (default)",
        "fused_qkv_hfq4g256",
        &[1024, 256, 256],
        1024,
    );
    show(
        "FA fused_qkv (wave64)",
        "fused_qkv_hfq4g256_wave64",
        &[1024, 256, 256],
        1024,
    );
    show(
        "FA fused_qkv (wave64+dp4a)",
        "fused_qkv_hfq4g256_wave64_dp4a",
        &[1024, 256, 256],
        1024,
    );

    // LA layers: fused_qkvza_hfq4g256
    // (qkv=768, z=256, β=32, α=32), K=1024, 18 calls/token (18 LA layers)
    show(
        "LA fused_qkvza (default)",
        "fused_qkvza_hfq4g256",
        &[768, 256, 32, 32],
        1024,
    );
    show(
        "LA fused_qkvza (wave64)",
        "fused_qkvza_hfq4g256_wave64",
        &[768, 256, 32, 32],
        1024,
    );

    // All layers: fused_gate_up_hfq4g256
    // (gate=2816, up=2816), K=1024, 24 calls/token
    show(
        "MLP fused_gate_up (default)",
        "fused_gate_up_hfq4g256",
        &[2816, 2816],
        1024,
    );
    show(
        "MLP fused_gate_up (wave64+dp4a)",
        "fused_gate_up_hfq4g256_wave64_dp4a",
        &[2816, 2816],
        1024,
    );

    // wo / wo-FA: gemv_hfq4g256, M=1024
    show("wo gemv (default)", "gemv_hfq4g256", &[1024], 256);
    show("wo gemv (wide)", "gemv_hfq4g256_wide", &[1024], 256);
    show(
        "wo gemv (multirow R=2)",
        "gemv_hfq4g256_multirow_r2",
        &[1024],
        256,
    );
    show(
        "wo gemv (multirow R=4)",
        "gemv_hfq4g256_multirow_r4",
        &[1024],
        1024,
    );
    show(
        "wo gemv (multirow R=8)",
        "gemv_hfq4g256_multirow_r8",
        &[1024],
        1024,
    );

    // w_down: gemv_hfq4g256, M=1024, K=2816
    show("w_down gemv", "gemv_hfq4g256", &[1024], 2816);

    // lm_head: gemv_hfq4g256, M=vocab, K=1024
    show("lm_head gemv", "gemv_hfq4g256", &[151936], 1024);

    println!("\n== Qwen3.5 9B (sample shapes) ==\n");

    // 9B: dim=3584, hidden≈18944, head_dim=128, n_heads=28, n_kv=4
    show(
        "9B fused_qkv",
        "fused_qkv_hfq4g256",
        &[3584, 512, 512],
        3584,
    );
    show(
        "9B fused_gate_up",
        "fused_gate_up_hfq4g256",
        &[18944, 18944],
        3584,
    );
    show("9B w_down gemv", "gemv_hfq4g256", &[3584], 18944);

    println!("\n== Negative cases (must classify as unrecognized) ==\n");

    show("non-GEMV kernel: rmsnorm", "rmsnorm", &[1024], 1024);
    show("non-GEMV kernel: rope", "rope", &[1024], 128);
    show(
        "non-GEMV kernel: flash_attn_fwd",
        "flash_attn_fwd",
        &[1024],
        128,
    );
    show("legacy q4k kernel", "fused_qkv_q4k", &[1024, 256, 256], 1024);
    show("empty function name", "", &[1024], 1024);

    // Wrong-arity fused_qkv (only 1 m): should fall through (None)
    show(
        "fused_qkv wrong arity",
        "fused_qkv_hfq4g256",
        &[1024],
        1024,
    );

    println!("\n(PR1 classifier OK if every line above prints either a\n GemvShape struct or 'unrecognized'.)");
}
