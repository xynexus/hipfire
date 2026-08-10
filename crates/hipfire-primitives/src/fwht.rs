// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — per-256 signed FWHT (Walsh-Hadamard) + the engine-matching sign table.

/// In-place per-256 signed FWHT: signs1 pre, Hadamard butterfly, 1/16·signs2 post
/// (orthonormal, 1/√256 = 1/16 normalization). Inverse = call with signs swapped.
pub fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    signed_fwht(x, signs1, signs2);
}

/// In-place signed FWHT over a power-of-two slice.
///
/// Applies `signs1` before the Hadamard butterfly and `signs2` after the
/// orthonormal `1/sqrt(n)` scale. For `n == 256`, this is the same transform as
/// [`cpu_fwht_256`].
pub fn signed_fwht(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FWHT length must be a power of two");
    assert_eq!(signs1.len(), n);
    assert_eq!(signs2.len(), n);
    for i in 0..n {
        x[i] *= signs1[i];
    }
    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for i in 0..n {
        x[i] *= scale * signs2[i];
    }
}

/// Generate FWHT sign table (matches the engine's `gen_fwht_signs`).
pub fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_is_orthonormal_involution() {
        // With identity signs, applying the transform twice returns the input
        // (orthonormal Hadamard is its own inverse up to the sign pre/post).
        let s1 = vec![1.0f32; 256];
        let s2 = vec![1.0f32; 256];
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let orig = x.clone();
        cpu_fwht_256(&mut x, &s1, &s2);
        cpu_fwht_256(&mut x, &s1, &s2);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-4, "involution drift: {a} vs {b}");
        }
    }

    /// The transform destroys per-channel saliency, exactly and by construction.
    ///
    /// Every entry of a signed Hadamard has the same magnitude (`1/√n`), so for
    /// any per-channel weighting `s`, the rotated weighting's diagonal is
    /// `[R·diag(s)·Rᵀ]ᵢᵢ = Σⱼ Rᵢⱼ²·sⱼ = mean(s)` — the SAME value at every
    /// position. Anything that ranks positions INSIDE a rotated group by a
    /// per-input-channel saliency is therefore multiplying every candidate by
    /// one constant, which cannot reorder them.
    ///
    /// This is why the plan's E1 ("reweight the gain in `mixed_overlay_indices`
    /// by GuidedQuant saliency") is a no-op: the mixed overlay selects in the
    /// rotated domain. Saliency has to act BEFORE the rotation (which is what
    /// the `+` AWQ pass does) or account for the off-diagonal mass of
    /// `R·diag(s)·Rᵀ` (which is what `++` LDLQ does).
    #[test]
    fn rotation_flattens_any_per_channel_saliency() {
        let n = 256;
        let signs1 = gen_fwht_signs(42, n);
        let signs2 = gen_fwht_signs(1042, n);
        // Heavy-tailed saliency — the case a reweight would most want to exploit.
        let saliency: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 + 0.5).powi(3)).collect();

        // Column j of R, obtained by transforming the j-th basis vector, gives
        // [R·diag(s)·Rᵀ]ᵢᵢ = Σⱼ Rᵢⱼ²·sⱼ without materialising R.
        let mut diagonal = vec![0.0f32; n];
        for (j, &s) in saliency.iter().enumerate() {
            let mut basis = vec![0.0f32; n];
            basis[j] = 1.0;
            cpu_fwht_256(&mut basis, &signs1, &signs2);
            for (d, r) in diagonal.iter_mut().zip(basis.iter()) {
                *d += r * r * s;
            }
        }

        let mean = saliency.iter().sum::<f32>() / n as f32;
        let (lo, hi) = diagonal
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        assert!(
            (hi - lo) / mean < 1e-4,
            "saliency survived the rotation: spread {} over mean {mean}",
            hi - lo
        );
        assert!(
            (lo - mean).abs() / mean < 1e-4,
            "rotated weight {lo} is not mean(s) {mean}"
        );
    }
}
