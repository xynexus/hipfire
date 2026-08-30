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
//! reduces exactly to plain RoPE. [`cos_sin`] is that case.
//!
//! [`cos_sin_mrope`] is the general one, for vision, where the grids genuinely
//! differ.

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

/// Interleaved mRoPE `cos`/`sin` for three position grids.
///
/// `positions` is `[T, H, W]`, each `n` long. `section` is the config's
/// `mrope_section` (`[11, 11, 10]` in the shipped checkpoint).
///
/// The reference reorganises the frequency layout from CHUNKED `[TTT…HHH…WWW]`
/// to INTERLEAVED `[THWTHW…TT]`, which is not a permutation of the chunked form —
/// it is a per-index CHOICE OF GRID, and the tail matters:
///
/// * index `j` takes grid `j % 3`…
/// * …but only while `j < section[j % 3] * 3`; past its section's reach an index
///   falls back to **T**.
///
/// So with `[11, 11, 10]` the H entries stop after index 32 and the W entries
/// after 29, and every index above that is T — the `…TT` tail in the reference's
/// own docstring. Reading the pattern as a plain round-robin gets the last few
/// frequencies wrong on every token, which is the kind of error that degrades
/// long-range position sense without ever looking like a failure.
///
/// Mirrors `Qwen4ExpTextRotaryEmbedding.apply_interleaved_mrope`
/// (`modeling_qwen4_exp.py:146`), pinned under `third_party/`.
pub fn cos_sin_mrope(
    positions: &[[usize; 3]],
    inv_freq: &[f32],
    section: &[usize; 3],
) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = Vec::with_capacity(positions.len() * half * 2);
    let mut sin = Vec::with_capacity(positions.len() * half * 2);
    for p in positions {
        let f: Vec<f32> = inv_freq
            .iter()
            .enumerate()
            .map(|(j, w)| {
                let g = j % 3;
                // Grid `g` owns this index only inside its section's reach.
                let grid = if g != 0 && j < section[g] * 3 { g } else { 0 };
                w * p[grid] as f32
            })
            .collect();
        for _ in 0..2 {
            cos.extend(f.iter().map(|v| v.cos()));
            sin.extend(f.iter().map(|v| v.sin()));
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod mrope_tests {
    use super::*;

    /// With all three grids equal, the interleave cannot matter: mRoPE must
    /// reduce EXACTLY to plain RoPE. This is what makes the text trunk's use of
    /// `cos_sin` correct, so it is worth asserting rather than assuming.
    #[test]
    fn identical_grids_reduce_to_plain_rope() {
        let inv = inv_freq(64, 10000.0);
        let pos = [3usize, 17, 42];
        let (c0, s0) = cos_sin(&pos, &inv);
        let tri: Vec<[usize; 3]> = pos.iter().map(|&p| [p, p, p]).collect();
        let (c1, s1) = cos_sin_mrope(&tri, &inv, &[11, 11, 10]);
        assert_eq!(c0, c1);
        assert_eq!(s0, s1);
    }

    /// The grid each index reads, spelled out against the reference's slicing.
    /// `freqs_t[..., offset:section[dim]*3:3] = freqs[dim][..., same]` means index
    /// `j` belongs to grid `j % 3` while `j < section[j % 3] * 3`, and to T after.
    #[test]
    fn index_ownership_matches_the_reference_slicing() {
        let section = [4usize, 2, 1];
        // Distinct positions per grid make the chosen grid readable from the value.
        let positions = [[1usize, 10, 100]];
        let inv: Vec<f32> = vec![1.0; 12];
        let (_, sin) = cos_sin_mrope(&positions, &inv, &section);
        // sin(1*p) for the grid that owns each index.
        let owner = |j: usize| -> usize {
            let g = j % 3;
            if g != 0 && j < section[g] * 3 {
                g
            } else {
                0
            }
        };
        let expect: Vec<f32> = (0..12)
            .map(|j| (positions[0][owner(j)] as f32).sin())
            .collect();
        assert_eq!(&sin[..12], &expect[..]);
        // H reaches index 4 (1,4 < 6) but not 7; W reaches 2 but not 5.
        assert_eq!(owner(1), 1, "H owns 1");
        assert_eq!(owner(4), 1, "H owns 4");
        assert_eq!(owner(7), 0, "H's section ends: 7 falls back to T");
        assert_eq!(owner(2), 2, "W owns 2");
        assert_eq!(owner(5), 0, "W's section ends: 5 falls back to T");
        assert_eq!(owner(3), 0, "every multiple of 3 is always T");
    }

    /// The shipped `[11, 11, 10]` over a 128-wide rotary half: the tail really is
    /// all T, which is the half of the layout a round-robin reading gets wrong.
    #[test]
    fn shipped_section_leaves_a_t_only_tail() {
        let section = [11usize, 11, 10];
        let owner = |j: usize| -> usize {
            let g = j % 3;
            if g != 0 && j < section[g] * 3 {
                g
            } else {
                0
            }
        };
        assert_eq!(owner(31), 1, "H still owns 31 (< 33)");
        assert_eq!(owner(29), 2, "W still owns 29 (< 30)");
        assert_eq!(
            owner(32),
            0,
            "32 % 3 == 2 but 32 >= 30, so W has ended -> T"
        );
        for j in 33..64 {
            assert_eq!(owner(j), 0, "index {j} is past both sections");
        }
    }
}
