//! DFlash native driver, step 2: one op native, BIT-FOR-BIT vs Python.
//!
//! Dispatches the int8 projection (`int_matmul`) natively on the exact
//! quantized operands the Python harness used, and diffs the int32 result
//! against Python's. The GEMM is integer, so a mismatch is a buffer-layout or
//! argument-order bug — never precision. This proves the contract before any
//! dispatch chaining.
//!
//! Inputs come from `dflash_body_npu.py --dump-op DIR` (op_A_int8.npy,
//! op_B_int8.npy, op_C_int32.npy, op_meta.json) plus the manifest for the
//! resolved xclbin/insts.
//!
//! Usage: `dflash_op_parity MANIFEST.json OPDIR`   (hold the hipfire lock)

#[cfg(target_os = "linux")]
#[path = "common/npy.rs"]
mod npy;

#[cfg(target_os = "linux")]
fn main() {
    use hipfire_xdna::NpuKernel;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dflash_op_parity MANIFEST.json OPDIR");
        std::process::exit(2);
    }
    let (manifest_path, opdir) = (&args[1], &args[2]);

    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{opdir}/op_meta.json")).expect("op_meta"))
            .expect("parse op_meta");
    let (m_dim, k_dim, n_dim) = (
        meta["M"].as_u64().unwrap() as usize,
        meta["K"].as_u64().unwrap() as usize,
        meta["N"].as_u64().unwrap() as usize,
    );

    // Find the manifest entry for this exact shape; its CompileTime args keyed
    // the JIT cache dir, so shape-matching is what identifies the artifact.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(manifest_path).expect("read manifest"))
            .expect("parse manifest");
    let kernels = manifest["kernels"].as_object().expect("kernels");
    let (kname, spec) = kernels
        .iter()
        .find(|(_, s)| {
            let ca = &s["compile_args"];
            ca["M"].as_u64() == Some(m_dim as u64)
                && ca["K"].as_u64() == Some(k_dim as u64)
                && ca["N"].as_u64() == Some(n_dim as u64)
                && ca["dtype_in_str"].as_str() == Some("i8")
        })
        .unwrap_or_else(|| panic!("no int_matmul kernel for M={m_dim} K={k_dim} N={n_dim}"));
    println!("kernel: {kname}\n  {}", spec["xclbin"].as_str().unwrap());

    let a = npy::read(&format!("{opdir}/op_A_int8.npy")).expect("A");
    let b = npy::read(&format!("{opdir}/op_B_int8.npy")).expect("B");
    let c_ref = npy::read(&format!("{opdir}/op_C_int32.npy")).expect("C");
    let (a_i8, b_i8, c_ref_i32) = (a.as_i8(), b.as_i8(), c_ref.as_i32());
    assert_eq!(a_i8.len(), m_dim * k_dim, "A shape");
    assert_eq!(b_i8.len(), n_dim * k_dim, "B shape");
    assert_eq!(c_ref_i32.len(), m_dim * n_dim, "C shape");

    let xclbin = std::fs::read(spec["xclbin"].as_str().unwrap()).expect("xclbin");
    let insts = std::fs::read(spec["insts"].as_str().unwrap()).expect("insts");
    let kernel = NpuKernel::load(&xclbin, &insts).expect("load");

    // Argument order is A (resident weight), B (activation), C (output) — the
    // manifest's recorded order, matching run_iters(int_matmul, A_t, B_t, C_t).
    let mut abuf = kernel.alloc_arg(m_dim * k_dim).expect("alloc A");
    let mut bbuf = kernel.alloc_arg(n_dim * k_dim).expect("alloc B");
    let mut cbuf = kernel.alloc_arg(m_dim * n_dim * 4).expect("alloc C");
    abuf.as_mut_slice().copy_from_slice(unsafe {
        std::slice::from_raw_parts(a_i8.as_ptr() as *const u8, a_i8.len())
    });
    bbuf.as_mut_slice().copy_from_slice(unsafe {
        std::slice::from_raw_parts(b_i8.as_ptr() as *const u8, b_i8.len())
    });
    cbuf.as_mut_slice().fill(0);

    kernel.dispatch(&[&abuf, &bbuf, &cbuf]).expect("dispatch");
    kernel.sync_output(&cbuf).expect("sync C");

    let c_native = unsafe {
        std::slice::from_raw_parts(cbuf.as_slice().as_ptr() as *const i32, m_dim * n_dim)
    };

    let mut mismatches = 0usize;
    let mut first = None;
    let mut max_abs_diff = 0i64;
    for i in 0..m_dim * n_dim {
        if c_native[i] != c_ref_i32[i] {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, c_native[i], c_ref_i32[i]));
            }
            max_abs_diff = max_abs_diff.max((c_native[i] as i64 - c_ref_i32[i] as i64).abs());
        }
    }

    println!(
        "  M={m_dim} K={k_dim} N={n_dim}  elems={}  mismatches={mismatches}",
        m_dim * n_dim
    );
    if mismatches == 0 {
        println!("=== STEP 2: BIT-EXACT vs Python ===");
    } else {
        let (i, got, want) = first.unwrap();
        println!(
            "  first mismatch at [{i}]: native={got} python={want}  max_abs_diff={max_abs_diff}"
        );
        println!("=== STEP 2: MISMATCH (layout/arg-order bug, not precision) ===");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Linux-only");
}
