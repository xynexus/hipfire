// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Rotary tables for the text trunk.
//!
//! The reference builds these with **interleaved mRoPE**: it computes frequencies
//! for three position grids (T, H, W) and interleaves them as `THWTHW...`.
//!
//! For a TEXT-ONLY sequence all three grids carry the same position ids, so the
//! three frequency sets are identical and the interleave is a no-op — mRoPE
//! reduces exactly to plain RoPE. That is why this module is enough for the text
//! trunk, and why vision, where the grids genuinely differ, needs the interleave
//! (and is deliberately not implemented here rather than implemented untested).

/// `inv_freq[i] = 1 / theta^(2i / rotary_dim)`, length `rotary_dim / 2`.
pub fn inv_freq(rotary_dim: usize, theta: f32) -> Vec<f32> {
    (0..rotary_dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32))
        .collect()
}

/// `cos`/`sin` for each position, each `rotary_dim` wide.
///
/// The half-width frequency vector is concatenated with itself before the cosine,
/// matching `emb = cat((freqs, freqs))` — which is what makes the `rotate_half`
/// form of the rotation correct.
pub fn cos_sin(positions: &[usize], inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = Vec::with_capacity(positions.len() * half * 2);
    let mut sin = Vec::with_capacity(positions.len() * half * 2);
    for &p in positions {
        let f: Vec<f32> = inv_freq.iter().map(|w| w * p as f32).collect();
        for _ in 0..2 {
            cos.extend(f.iter().map(|v| v.cos()));
            sin.extend(f.iter().map(|v| v.sin()));
        }
    }
    (cos, sin)
}
