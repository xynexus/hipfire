//! Prove the artifact -> NPU path on a REAL oq4++ weight tensor.
//!
//! Everything measured in `docs/npu/wire-in-r6-prefill-offload.md` used synthetic
//! weights. This closes that gap: it reads an actual `.hfq` produced by
//! `hipfire-quantize --format oq4++`, expands one projection through
//! `OpusPackedMatrix` (the same path the resident executors use at upload), runs it on
//! the NPU through `NpuGemm`, applies the group scales on the host, and checks the
//! result against a CPU reference computed from the same expanded weights.
//!
//! This is step 3 ("upload path") of `docs/npu/decoder-layer-npu-scope.md`, isolated
//! from the daemon: if this passes, real oq4++ weights are consumable by the NPU GEMM
//! and only the routing remains.
//!
//! Run: cargo run --release -p hipfire-runtime --example npu_artifact_gemv -- \
//!        <model.hfq> <tensor-name> <r6-cache-dir> MT KCHUNK GROUPS NB [rows]
fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_xdna::{NpuGemm, OpusPackedMatrix};

        let a: Vec<String> = std::env::args().collect();
        if a.len() < 8 {
            eprintln!(
                "usage: npu_artifact_gemv <model.hfq> <tensor> <cache-dir> MT KCHUNK GROUPS NB [rows]"
            );
            std::process::exit(2);
        }
        let (path, tname, dir) = (&a[1], &a[2], &a[3]);
        let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        let (mt, kc, g, nb) = (p(4, 1), p(5, 32), p(6, 128), p(7, 16));

        let hfq = HfqFile::open(std::path::Path::new(path)).expect("open hfq");
        let (info, payload) = hfq
            .tensor_data_vec(tname)
            .unwrap_or_else(|| panic!("tensor not found: {tname}"));
        // HFQ stores weights [out, in]; the GEMM contracts over K = in.
        let (n_out, k_in) = (info.shape[0] as usize, info.shape[1] as usize);
        println!(
            "{tname}: shape [{n_out}, {k_in}] quant_type={} payload={} MB",
            info.quant_type,
            payload.len() / (1 << 20)
        );

        // The AWQ sidecar rides alongside as `<stem>.awq_scale.weight` for `oq4+`/`oq4++`.
        let awq_name = tname.replace(".weight", ".awq_scale.weight");
        let awq = hfq.tensor_data_vec(&awq_name).map(|(_, bytes)| {
            bytes
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect::<Vec<f32>>()
        });
        println!(
            "  awq sidecar: {}",
            if awq.is_some() { "present" } else { "none" }
        );

        let matrix = OpusPackedMatrix::from_payload(info.quant_type, k_in, n_out, &payload, awq)
            .expect("OpusPackedMatrix::from_payload");
        println!(
            "  encoding={:?} groups={} k={} n={}",
            matrix.encoding(),
            matrix.group_count(),
            matrix.k(),
            matrix.n()
        );

        // Expand every group to dense signed bytes — the same expansion the resident
        // executors do once at upload. For OQ4 these land in -8..=7, which is exactly
        // NpuGemm's `w_int4` contract (one int4 value per byte).
        let gcount = matrix.group_count();
        let gk = k_in / gcount; // rows of K per group
        let mut w_dense = vec![0i8; k_in * n_out]; // row-major [K, N]
        let mut out_of_range = 0usize;
        for gi in 0..gcount {
            let dense = matrix.group_dense_i8(gi); // [gk, n_out] for this K-slice
            for (i, &v) in dense.iter().enumerate() {
                if !(-8..=7).contains(&v) {
                    out_of_range += 1;
                }
                let (kk, nn) = (i / n_out, i % n_out);
                w_dense[(gi * gk + kk) * n_out + nn] = v;
            }
        }
        if out_of_range > 0 {
            eprintln!(
                "  NOTE: {out_of_range} weights outside int4 range — this tensor is not \
                 4-bit-resident; the NPU W4A8 kernel would truncate. Aborting."
            );
            std::process::exit(3);
        }
        println!(
            "  expanded {} weights, all within int4 range",
            w_dense.len()
        );

        let x = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let i = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let mut gemm = NpuGemm::load_rounds(&x, &i, mt, 4, kc, g, nb, 1).expect("load kernel");
        let (bm, bn, bk) = (gemm.block_m(), gemm.block_n(), gemm.block_k());
        if k_in % bk != 0 || n_out % bn != 0 {
            eprintln!(
                "  shape does not tile on this kernel (block {bm}x{bn}x{bk}); \
                 pick a cache whose block_n divides N={n_out} and block_k divides K={k_in}"
            );
            std::process::exit(4);
        }

        // One token of int8 activations, padded to the kernel's M block.
        let act: Vec<i8> = (0..k_in)
            .map(|i| (((i as u32).wrapping_mul(2654435761) >> 17) as i32 % 127 - 63) as i8)
            .collect();
        let mut a_pad = vec![0i8; bm * k_in];
        a_pad[..k_in].copy_from_slice(&act);
        let mut c = vec![0i32; bm * n_out];

        let weights = gemm.upload_weights(k_in, n_out, &w_dense).expect("upload");
        gemm.run_resident(bm, k_in, n_out, &a_pad, &weights, &mut c)
            .expect("run");

        // CPU reference over the SAME expanded weights: pure int32 dot products.
        let mut bad = 0usize;
        let mut first = None;
        for nn in 0..n_out {
            let want: i32 = (0..k_in)
                .map(|kk| act[kk] as i32 * w_dense[kk * n_out + nn] as i32)
                .sum();
            if c[nn] != want {
                bad += 1;
                first.get_or_insert((nn, c[nn], want));
            }
        }
        if bad == 0 {
            println!(
                "  PASS: NPU int32 output matches CPU reference on all {n_out} columns \
                 (real oq4++ weights, {} K-chunks x {} N-slabs)",
                k_in / bk,
                n_out / bn
            );
        } else {
            let (nn, got, want) = first.unwrap();
            eprintln!("  FAIL: {bad}/{n_out} mismatch; first at col {nn}: got {got}, want {want}");
            std::process::exit(1);
        }
    }
}

#[cfg(target_os = "linux")]
fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 0x1f, h & 0x3ff);
    let bits = match e {
        0 if m == 0 => (s as u32) << 31,
        0 => {
            let mut e2 = -1i32;
            let mut m2 = m as u32;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            ((s as u32) << 31) | (((127 - 15 + e2 + 1) as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
        0x1f => ((s as u32) << 31) | (0xff << 23) | ((m as u32) << 13),
        _ => ((s as u32) << 31) | (((e as i32 - 15 + 127) as u32) << 23) | ((m as u32) << 13),
    };
    f32::from_bits(bits)
}
