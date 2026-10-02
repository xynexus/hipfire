//! `KvCache::grow_kvarn` parity: every buffer of a grown KVarN cache starts with
//! the original's exact bytes and is zero past them.
//!
//! The grow relies on KVarN being position-major (V by token, K by block then
//! head), so a smaller cache is a byte prefix of a larger one. If a layout change
//! breaks that, positions land in the wrong place after a session grows — this
//! is the check that would notice.
//!
//!   cargo run --release -p hipfire-runtime --example parity_kvarn_grow

use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::kv::KvCache;

fn bytes_of(t: &GpuTensor) -> usize {
    t.numel() * t.dtype.size()
}

fn main() {
    if std::env::args().any(|a| a == "paged") {
        return paged_main();
    }
    let mut gpu = Gpu::init().unwrap();
    // One placeholder layer in the middle, as a qwen3.5 DeltaNet layer would be.
    let mask = [true, false, true, true];
    let (heads, head_dim, max_seq, bits) = (4, 128, 4096, 4);
    let mut small = KvCache::new_gpu_kvarn_capped_filtered(
        &mut gpu, &mask, heads, head_dim, max_seq, 256, bits,
    )
    .unwrap();
    small.compact_offset = 7;

    // Distinct pseudo-random content per buffer.
    let mut seed = 0x9e37_79b9_u32;
    let mut originals = Vec::new();
    for t in small
        .k_gpu
        .iter()
        .chain(&small.v_gpu)
        .chain(&small.k_window)
    {
        let data: Vec<u8> = (0..bytes_of(t))
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as u8
            })
            .collect();
        gpu.hip.memcpy_htod(&t.buf, &data).unwrap();
        originals.push(data);
    }

    let grown = small.grow_kvarn(&mut gpu, 1024, &mask).unwrap();
    assert_eq!(grown.physical_cap, 1024);
    assert_eq!(grown.compact_offset, 7);

    let mut checked = 0;
    for (t, want) in grown
        .k_gpu
        .iter()
        .chain(&grown.v_gpu)
        .chain(&grown.k_window)
        .zip(&originals)
    {
        let got = gpu.download_raw(t, bytes_of(t)).unwrap();
        assert!(
            got.len() >= want.len(),
            "grown buffer is smaller than the original"
        );
        assert_eq!(&got[..want.len()], &want[..], "prefix differs after grow");
        assert!(
            got[want.len()..].iter().all(|&b| b == 0),
            "grown tail is not zero"
        );
        checked += 1;
    }
    // K/V on real layers must actually have grown; placeholders and windows not.
    for (i, &is_kv) in mask.iter().enumerate() {
        let (k0, k1) = (bytes_of(&small.k_gpu[i]), bytes_of(&grown.k_gpu[i]));
        let (v0, v1) = (bytes_of(&small.v_gpu[i]), bytes_of(&grown.v_gpu[i]));
        if is_kv {
            assert!(k1 > k0 && v1 > v0, "layer {i} did not grow");
        } else {
            assert_eq!((k0, v0), (k1, v1), "placeholder layer {i} changed size");
        }
    }
    println!("parity_kvarn_grow: OK ({checked} buffers, 256 -> 1024 positions)");
}

/// Paged variant: grow in place, and a fork that shares the sealed prefix.
/// Run with `-- paged` (needs HIP virtual memory management).
#[allow(dead_code)]
fn paged_main() {
    let mut gpu = Gpu::init().unwrap();
    let mask = [true, false, true, true];
    let (heads, head_dim, max_seq, bits) = (4, 128, 8192, 4);
    let Some(mut src) =
        KvCache::new_gpu_kvarn_paged(&mut gpu, &mask, heads, head_dim, max_seq, 512, bits).unwrap()
    else {
        println!("parity_kvarn_grow paged: SKIPPED (no virtual memory management)");
        return;
    };
    let ptr_before = src.k_gpu[0].buf.as_ptr();
    let mut seed = 0x1234_5678_u32;
    let mut fill = |gpu: &mut Gpu, t: &GpuTensor| -> Vec<u8> {
        let data: Vec<u8> = (0..bytes_of(t))
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed as u8) | 1 // never zero, so "zeroed" is unambiguous
            })
            .collect();
        gpu.hip.memcpy_htod(&t.buf, &data).unwrap();
        data
    };
    let k0 = fill(&mut gpu, &src.k_gpu[0]);
    let v0 = fill(&mut gpu, &src.v_gpu[0]);
    let w0 = fill(&mut gpu, &src.k_window[0]);

    // Grow in place: same address, old bytes intact, new bytes zero.
    src.grow_paged(&mut gpu, 2048).unwrap();
    assert_eq!(
        src.k_gpu[0].buf.as_ptr(),
        ptr_before,
        "grow moved the buffer"
    );
    let k_after = gpu
        .download_raw(&src.k_gpu[0], bytes_of(&src.k_gpu[0]))
        .unwrap();
    assert_eq!(&k_after[..k0.len()], &k0[..], "grow lost data");
    assert!(
        k_after[k0.len()..].iter().all(|&b| b == 0),
        "grown tail not zeroed"
    );

    // Fork at 300 sealed positions (2 full K blocks + 44 open rows).
    let sealed = 300;
    let fork = src.fork_paged(&mut gpu, sealed, 1024).unwrap();
    let rec = KvCache::kvarn_k_record_bytes_bits(head_dim, bits);
    let v_bpp = heads * (head_dim / 32) * 34;
    let (k_sealed, v_sealed) = ((sealed / 128) * heads * rec, sealed * v_bpp);
    let kf = gpu
        .download_raw(&fork.k_gpu[0], bytes_of(&fork.k_gpu[0]))
        .unwrap();
    let vf = gpu
        .download_raw(&fork.v_gpu[0], bytes_of(&fork.v_gpu[0]))
        .unwrap();
    assert_eq!(
        &kf[..k_sealed],
        &k0[..k_sealed],
        "fork K sealed prefix differs"
    );
    assert!(
        kf[k_sealed..].iter().all(|&b| b == 0),
        "fork K past the seal not zero"
    );
    assert_eq!(
        &vf[..v_sealed],
        &v0[..v_sealed],
        "fork V sealed prefix differs"
    );
    assert!(
        vf[v_sealed..].iter().all(|&b| b == 0),
        "fork V past the seal not zero"
    );
    let wf = gpu.download_raw(&fork.k_window[0], w0.len()).unwrap();
    assert_eq!(wf, w0, "window not copied");

    // Sharing is physical: a byte in the first (shared) V page, changed in the
    // source, reads back changed in the fork. (Never done in serving — shared bytes
    // are sealed — but it is what proves they are one copy.)
    let page = fork.pages.as_ref().unwrap()[0]
        .as_ref()
        .unwrap()
        .v
        .page_size();
    assert!(v_sealed >= page, "test prefix must cover a whole page");
    gpu.hip
        .memcpy_htod_offset(&src.v_gpu[0].buf, 0, &[0u8])
        .unwrap();
    let first = gpu.download_raw(&fork.v_gpu[0], 1).unwrap();
    assert_eq!(first[0], 0, "first V page is not shared");
    // And the fork's writes past the shared pages stay private.
    gpu.hip
        .memcpy_htod_offset(&fork.v_gpu[0].buf, v_sealed, &[0xAB])
        .unwrap();
    let src_byte = gpu.download_raw(&src.v_gpu[0], v_sealed + 1).unwrap()[v_sealed];
    assert_eq!(
        src_byte, v0[v_sealed],
        "a fork write past its seal reached the source"
    );

    fork.free_gpu(&mut gpu);
    // Source still intact after the fork's pages are unmapped.
    let v_again = gpu.download_raw(&src.v_gpu[0], v_sealed).unwrap();
    assert_eq!(
        &v_again[1..],
        &v0[1..v_sealed],
        "freeing the fork disturbed the source"
    );
    src.free_gpu(&mut gpu);
    println!(
        "parity_kvarn_grow paged: OK (grow in place; fork shares {page}-byte pages below the seal)"
    );
}
