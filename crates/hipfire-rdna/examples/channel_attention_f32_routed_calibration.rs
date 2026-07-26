// SPDX-License-Identifier: Apache-2.0
//! Calibration-shaped channel for session-routed F32 attention.
//!
//! This matches the Qwen3.5-397B production geometry at the most expensive
//! context tile: 64 independent sessions, 32 positions per session, 32 query
//! heads, 2 KV heads, head dimension 256, and context 2,048. Every session has
//! its own K/V allocation and pointer-table entries. Zero inputs make the
//! expected result exact while retaining the real launch and memory geometry.

use hipfire_rdna::{DType, Gpu};

const SESSIONS: usize = 64;
const TIME_TILE: usize = 32;
const ROWS: usize = SESSIONS * TIME_TILE;
const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const MAX_CONTEXT: usize = 2048;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;

fn i32_bytes(values: &[i32]) -> &[u8] {
    // The device table is consumed as `int*`; its tensor dtype is cosmetic.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }
}

fn u64_bytes(values: &[u64]) -> &[u8] {
    // The device table is consumed as `uint64_t*`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 8) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    println!(
        "arch={} sessions={SESSIONS} time_tile={TIME_TILE} rows={ROWS} context={MAX_CONTEXT}",
        gpu.arch
    );

    let max_abs = {
        let k_sessions = (0..SESSIONS)
            .map(|_| gpu.zeros_owned(&[MAX_CONTEXT * KV_DIM], DType::F32))
            .collect::<Result<Vec<_>, _>>()?;
        let v_sessions = (0..SESSIONS)
            .map(|_| gpu.zeros_owned(&[MAX_CONTEXT * KV_DIM], DType::F32))
            .collect::<Result<Vec<_>, _>>()?;
        let k_ptrs = k_sessions
            .iter()
            .map(|tensor| tensor.buf.as_ptr() as usize as u64)
            .collect::<Vec<_>>();
        let v_ptrs = v_sessions
            .iter()
            .map(|tensor| tensor.buf.as_ptr() as usize as u64)
            .collect::<Vec<_>>();
        let k_table = gpu.alloc_owned(&[SESSIONS * 8], DType::Raw)?;
        let v_table = gpu.alloc_owned(&[SESSIONS * 8], DType::Raw)?;
        gpu.hip.memcpy_htod(&k_table.buf, u64_bytes(&k_ptrs))?;
        gpu.hip.memcpy_htod(&v_table.buf, u64_bytes(&v_ptrs))?;

        // Calibration rows are ordered by time round and then session.
        let first_position = MAX_CONTEXT - TIME_TILE;
        let mut row_sessions = Vec::with_capacity(ROWS);
        let mut positions = Vec::with_capacity(ROWS);
        for position in first_position..MAX_CONTEXT {
            for session in 0..SESSIONS {
                row_sessions.push(session as i32);
                positions.push(position as i32);
            }
        }
        let row_session_table = gpu.alloc_owned(&[ROWS], DType::F32)?;
        let position_table = gpu.alloc_owned(&[ROWS], DType::F32)?;
        gpu.hip
            .memcpy_htod(&row_session_table.buf, i32_bytes(&row_sessions))?;
        gpu.hip
            .memcpy_htod(&position_table.buf, i32_bytes(&positions))?;

        let q = gpu.zeros_owned(&[ROWS * N_HEADS * HEAD_DIM], DType::F32)?;
        let output = gpu.zeros_owned(&[ROWS * N_HEADS * HEAD_DIM], DType::F32)?;
        gpu.attention_f32_routed_batched(
            &q,
            &k_table,
            &v_table,
            &output,
            &row_session_table,
            &position_table,
            1,
            0,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            MAX_CONTEXT,
            MAX_CONTEXT,
            ROWS,
        )?;
        gpu.device_synchronize()?;
        gpu.download_f32(&output)?
            .into_iter()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
    };
    gpu.reclaim_pending();

    println!("max_abs={max_abs:.9}");
    if !max_abs.is_finite() || max_abs != 0.0 {
        return Err(format!("calibration-shaped routed attention mismatch: {max_abs}").into());
    }
    println!("OK — calibration-shaped routed attention completed");
    Ok(())
}
