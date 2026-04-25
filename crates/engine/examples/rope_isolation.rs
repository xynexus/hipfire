//! Isolate rope_partial_interleaved_f32 determinism.
//! Fill buffer with known values, call rope at pos=0 (identity), verify output==input.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::Gpu;

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    let n_heads_q: usize = 32;
    let n_heads_k: usize = 8;
    let head_dim: usize = 128;
    let n_rot: usize = 64;
    let freq_base: f32 = 1_000_000.0;

    let mut gpu = Gpu::init().expect("gpu init");

    // Deterministic input: q[i] = i as f32, k[i] = -(i as f32)
    let q_host: Vec<f32> = (0..n_heads_q * head_dim).map(|i| i as f32).collect();
    let k_host: Vec<f32> = (0..n_heads_k * head_dim).map(|i| -(i as f32)).collect();

    let q = gpu.upload_f32(&q_host, &[n_heads_q, head_dim]).unwrap();
    let k = gpu.upload_f32(&k_host, &[n_heads_k, head_dim]).unwrap();

    // pos_buf = 0
    let pos_buf = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &0i32.to_ne_bytes()).unwrap();

    // Pre-hash
    let q_pre = gpu.download_f32(&q).unwrap();
    let k_pre = gpu.download_f32(&k).unwrap();
    let q_pre_bytes: Vec<u8> = q_pre.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let k_pre_bytes: Vec<u8> = k_pre.iter().flat_map(|v| v.to_ne_bytes()).collect();
    println!("  pre  q=0x{:016x} k=0x{:016x}", fnv1a(&q_pre_bytes), fnv1a(&k_pre_bytes));

    // Call rope 5 times at pos=0 on the same buffer. Each call should be identity.
    for iter in 0..5 {
        gpu.rope_partial_interleaved_f32(&q, &k, &pos_buf,
            n_heads_q, n_heads_k, head_dim, n_rot, freq_base).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let q_post = gpu.download_f32(&q).unwrap();
        let k_post = gpu.download_f32(&k).unwrap();
        let q_post_bytes: Vec<u8> = q_post.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let k_post_bytes: Vec<u8> = k_post.iter().flat_map(|v| v.to_ne_bytes()).collect();
        // Diff counts
        let q_diffs = q_pre.iter().zip(q_post.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let k_diffs = k_pre.iter().zip(k_post.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("  iter={} q=0x{:016x} k=0x{:016x} q_diffs={}/{} k_diffs={}/{}",
            iter,
            fnv1a(&q_post_bytes), fnv1a(&k_post_bytes),
            q_diffs, q_post.len(), k_diffs, k_post.len());
        // Show first few differing indices
        if q_diffs > 0 && iter == 0 {
            let first: Vec<_> = q_pre.iter().zip(q_post.iter()).enumerate()
                .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                .take(8).collect();
            for (idx, (a, b)) in first {
                let head = idx / head_dim;
                let dim = idx % head_dim;
                println!("    q[h={} d={}]: pre={} ({:08x}) post={} ({:08x})",
                    head, dim, a, a.to_bits(), b, b.to_bits());
            }
        }
    }
}
