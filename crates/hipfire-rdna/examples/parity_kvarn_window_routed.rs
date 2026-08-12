// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Validation of the routed batched KVarN K-window append
//! (`kv_cache_write_kvarn_window_routed_batched`) — the write half that lets the
//! fused grouped-MoE session batch run on KVarN KV instead of hard-requiring Q8.
//!
//! Three sessions with different base positions (including one whose slots sit
//! against the block boundary), rows routed to them in SHUFFLED order via a
//! session-major pointer table. Checks two things a wrong kernel would get
//! silently wrong:
//!
//!   1. every written slot holds exactly that row's K, in the token-major
//!      `[group, kv_dim]` layout `attention_kvarn_routed_batched` reads;
//!   2. every UNwritten slot still holds its sentinel — i.e. no row scattered
//!      into another session's window or another session's slot.
//!
//! (2) is the one that matters: a routing bug that writes the right values to the
//! wrong session produces plausible attention output, not a crash.
//!
//!   cargo run --release -p hipfire-rdna --example parity_kvarn_window_routed

use hipfire_rdna::{Gpu, GpuTensor};

const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const GROUP: usize = 128;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s as f32 + 0.5) / 2_147_483_648.0 * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let kv_dim = N_KV_HEADS * HEAD_DIM;

    // Base absolute positions chosen so the slots differ per session and one
    // session sits hard against the block boundary (125,126,127) without
    // crossing it — crossing is forbidden by the kernel's caller contract.
    let bases = [0usize, 125, 200];
    let per_session = 3usize;
    const SENTINEL: f32 = -12345.0;

    // One f32 window per session, [GROUP, kv_dim], pre-filled with a sentinel.
    let windows: Vec<GpuTensor> = (0..bases.len())
        .map(|_| {
            let bytes: Vec<u8> = std::iter::repeat(SENTINEL)
                .take(GROUP * kv_dim)
                .flat_map(|v: f32| v.to_le_bytes())
                .collect();
            gpu.upload_raw(&bytes, &[GROUP * kv_dim]).unwrap()
        })
        .collect();

    // Rows in shuffled session order, each carrying its own K payload.
    let order: Vec<usize> = vec![2, 0, 1, 1, 2, 0, 0, 1, 2];
    let n_rows = order.len();
    assert_eq!(n_rows, bases.len() * per_session);

    let mut seen = vec![0usize; bases.len()];
    let mut row_sessions = vec![0i32; n_rows];
    let mut positions = vec![0i32; n_rows];
    let mut k_batch = vec![0.0f32; n_rows * kv_dim];
    // expected[session][slot] = Some(row) for written slots
    let mut expected: Vec<Vec<Option<usize>>> = vec![vec![None; GROUP]; bases.len()];

    for (row, &sess) in order.iter().enumerate() {
        let idx = seen[sess];
        seen[sess] += 1;
        let pos = bases[sess] + idx;
        row_sessions[row] = sess as i32;
        positions[row] = pos as i32;
        let payload = lcg((row as u32 + 1) * 7919, kv_dim);
        k_batch[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(&payload);
        let slot = pos % GROUP;
        assert!(
            expected[sess][slot].is_none(),
            "test bug: session {sess} slot {slot} written twice"
        );
        expected[sess][slot] = Some(row);
    }

    let up_f32 = |g: &mut Gpu, v: &[f32], shape: usize| {
        g.upload_raw(
            &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            &[shape],
        )
        .unwrap()
    };
    let up_i32 = |g: &mut Gpu, v: &[i32], shape: usize| {
        g.upload_raw(
            &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            &[shape],
        )
        .unwrap()
    };

    let kd = up_f32(&mut gpu, &k_batch, n_rows * kv_dim);
    let rsid = up_i32(&mut gpu, &row_sessions, n_rows);
    let posd = up_i32(&mut gpu, &positions, n_rows);

    // Session-major pointer table, ptr_layer_stride=1, layer 0.
    let win_ptr_bytes: Vec<u8> = windows
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as usize as u64).to_le_bytes())
        .collect();
    let winp = gpu
        .upload_raw(&win_ptr_bytes, &[windows.len() * 2])
        .unwrap(); // u64 = 2 u32 elems

    // Write in SEGMENTS, not one launch — this is how the flush loop drives it,
    // and it exercises `row_offset`. Segment boundaries are deliberately uneven
    // and the buffers passed are always the FULL batch, so an off-by-one in the
    // absolute row indexing shows up as a wrong-slot or wrong-session write
    // rather than silently working.
    let segments: [(usize, usize); 3] = [(0, 2), (2, 5), (7, n_rows - 7)];
    for (off, cnt) in segments {
        gpu.kv_cache_write_kvarn_window_routed_batched(
            &winp, &kd, &rsid, &posd, 1, 0, N_KV_HEADS, HEAD_DIM, GROUP, off, cnt,
        )
        .unwrap();
    }
    gpu.device_synchronize().unwrap();

    let mut bad_written = 0usize;
    let mut bad_sentinel = 0usize;
    let mut max_abs = 0.0f32;

    for (sess, win) in windows.iter().enumerate() {
        let got = gpu.download_f32(win).unwrap();
        for slot in 0..GROUP {
            let base = slot * kv_dim;
            match expected[sess][slot] {
                Some(row) => {
                    for c in 0..kv_dim {
                        let want = k_batch[row * kv_dim + c];
                        let diff = (got[base + c] - want).abs();
                        if diff > 0.0 {
                            bad_written += 1;
                        }
                        max_abs = max_abs.max(diff);
                    }
                }
                None => {
                    for c in 0..kv_dim {
                        if got[base + c] != SENTINEL {
                            bad_sentinel += 1;
                        }
                    }
                }
            }
        }
    }

    let written_slots: usize = expected
        .iter()
        .map(|s| s.iter().filter(|e| e.is_some()).count())
        .sum();
    println!("sessions={} rows={n_rows} written_slots={written_slots}", bases.len());
    println!("  max |delta| on written slots : {max_abs:e}  (mismatched elems {bad_written})");
    println!("  clobbered sentinel elems     : {bad_sentinel}");

    if bad_written == 0 && bad_sentinel == 0 {
        println!("PASS: routed KVarN window append is exact and touches nothing else");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
