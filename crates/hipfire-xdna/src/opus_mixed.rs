//! Compact mixed-precision Opus GEMM over paired AIE2P W4A8 and sparse kernels.
//!
//! The current qt=36 layout stores each rotated 256-weight group as int4 bulk
//! plus three sparse int8 replacements. The NPU evaluates it exactly as two integer GEMMs:
//! `A·Q4 + A·(Q8-Q4)`. The first term uses the packed W4 kernel; the correction
//! uses a dedicated sparse3 kernel fed directly from the three compact
//! `(index, residual)` pairs per output column.
//!
//! Plain, `+`, and `++` artifacts share this runtime. `+`/`++` optionally pass
//! the tensor's AWQ sidecar; `++` changes only the offline packed values.
#![cfg(target_os = "linux")]

use hipfire_primitives::{
    conv::f16_to_f32,
    fwht::{cpu_fwht_256, gen_fwht_signs},
};

use crate::{NpuGemmMp, NpuSparse3Mp, XdnaError};

const GROUP: usize = 256;

struct OpusMixedGroup {
    packed_w4: Vec<u8>,
    sparse_residual_chunks: Vec<Vec<u8>>,
    scales: Vec<f32>,
    q4: Vec<i8>,
    residual: Vec<i8>,
}

struct PreparedActivations {
    groups: Vec<Vec<i8>>,
    scales: Vec<Vec<f32>>,
    padded_rows: usize,
}

/// Host-packed compact mixed-precision Opus matrix reusable with a resident
/// executor of the same output width.
pub struct OpusMixedPackedMatrix {
    groups: Vec<OpusMixedGroup>,
    awq_scale: Option<Vec<f32>>,
    k: usize,
    n: usize,
}

impl OpusMixedPackedMatrix {
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

/// Resident paired W4/sparse3 kernels shared by mixed Opus matrices with one `N`.
pub struct NpuOpusMixedExecutor {
    w4: NpuGemmMp,
    residual_sparse3: NpuSparse3Mp,
    n: usize,
    rows_per_dispatch: usize,
}

impl NpuOpusMixedExecutor {
    /// Load paired W4/sparse3 caches for matrices with output width `N`.
    pub fn load_cached(w4_cache: &str, sparse3_cache: &str, n: usize) -> Result<Self, XdnaError> {
        if n == 0 {
            return Err(invalid("want non-zero N"));
        }
        let w4 = NpuGemmMp::load_cached(w4_cache)?;
        let residual_sparse3 = NpuSparse3Mp::load_cached(sparse3_cache)?;
        if w4.weight_bits() != 4 {
            return Err(invalid("base cache must be W4"));
        }
        if w4.k() != GROUP
            || residual_sparse3.k() != GROUP
            || w4.n() != n
            || residual_sparse3.n() != n
        {
            return Err(invalid(format!(
                "cache shapes must both be K=256 N={n}; got W4 {}x{} sparse3 {}x{}",
                w4.k(),
                w4.n(),
                residual_sparse3.k(),
                residual_sparse3.n()
            )));
        }
        let rows_per_dispatch = lcm(w4.rows_per_dispatch(), residual_sparse3.rows_per_dispatch());
        Ok(Self {
            w4,
            residual_sparse3,
            n,
            rows_per_dispatch,
        })
    }

    /// Decode and prepack a compact row-major `[N,K]` mixed Opus matrix.
    pub fn pack_matrix(
        &self,
        k: usize,
        n: usize,
        compact: &[u8],
        awq_scale: Option<Vec<f32>>,
    ) -> Result<OpusMixedPackedMatrix, XdnaError> {
        if k == 0 || n == 0 || k % GROUP != 0 || n != self.n {
            return Err(invalid(format!(
                "want non-zero K%256=0 and executor N={}, got K={k} N={n}",
                self.n
            )));
        }
        if awq_scale.as_ref().is_some_and(|scale| scale.len() != k) {
            return Err(invalid(format!("AWQ scale length must be K={k}")));
        }
        let decoded = decode_compact(compact, k, n)?;
        let groups = decoded
            .into_iter()
            .map(|decoded| OpusMixedGroup {
                packed_w4: self.w4.prepack_weights(GROUP, n, &decoded.q4),
                sparse_residual_chunks: decoded.sparse_residual_chunks,
                scales: decoded.scales,
                q4: decoded.q4,
                residual: decoded.residual,
            })
            .collect();
        Ok(OpusMixedPackedMatrix {
            groups,
            awq_scale,
            k,
            n,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn rows_per_dispatch(&self) -> usize {
        self.rows_per_dispatch
    }

    /// Run `C[M,N] = X[M,K]·Wᵀ` with activation AWQ, FWHT-256, and int8 quantization.
    pub fn run_f32(
        &mut self,
        matrix: &OpusMixedPackedMatrix,
        m: usize,
        x: &[f32],
        c: &mut [f32],
    ) -> Result<(), XdnaError> {
        if matrix.n != self.n {
            return Err(invalid(format!(
                "matrix N={} does not match executor N={}",
                matrix.n, self.n
            )));
        }
        validate_run_shapes(m, matrix.k, matrix.n, x, c)?;
        let prepared = prepare_activations(
            m,
            matrix.k,
            x,
            matrix.awq_scale.as_deref(),
            self.rows_per_dispatch,
        );
        c.fill(0.0);
        let mut base = vec![0i32; prepared.padded_rows * matrix.n];
        let mut residual = vec![0i32; prepared.padded_rows * matrix.n];
        for (group_idx, group) in matrix.groups.iter().enumerate() {
            self.w4.load_weights(&group.packed_w4);
            self.w4.run(
                prepared.padded_rows,
                GROUP,
                matrix.n,
                &prepared.groups[group_idx],
                &mut base,
            )?;
            accumulate_scaled(
                m,
                matrix.n,
                &base,
                &prepared.scales[group_idx],
                &group.scales,
                c,
            );
            for sparse_chunk in &group.sparse_residual_chunks {
                self.residual_sparse3.load_weights(sparse_chunk);
                self.residual_sparse3.run(
                    prepared.padded_rows,
                    GROUP,
                    matrix.n,
                    &prepared.groups[group_idx],
                    &mut residual,
                )?;
                accumulate_scaled(
                    m,
                    matrix.n,
                    &residual,
                    &prepared.scales[group_idx],
                    &group.scales,
                    c,
                );
            }
        }
        Ok(())
    }

    /// CPU oracle for the exact integer/scaling contract used by [`Self::run_f32`].
    pub fn reference_f32(
        &self,
        matrix: &OpusMixedPackedMatrix,
        m: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, XdnaError> {
        let mut output = vec![0.0f32; m * matrix.n];
        validate_run_shapes(m, matrix.k, matrix.n, x, &output)?;
        let prepared = prepare_activations(
            m,
            matrix.k,
            x,
            matrix.awq_scale.as_deref(),
            self.rows_per_dispatch,
        );
        for (group_idx, group) in matrix.groups.iter().enumerate() {
            for row in 0..m {
                for col in 0..matrix.n {
                    let dot: i32 = (0..GROUP)
                        .map(|inner| {
                            let activation = prepared.groups[group_idx][row * GROUP + inner] as i32;
                            let index = inner * matrix.n + col;
                            activation * (group.q4[index] as i32 + group.residual[index] as i32)
                        })
                        .sum();
                    output[row * matrix.n + col] +=
                        dot as f32 * prepared.scales[group_idx][row] * group.scales[col];
                }
            }
        }
        Ok(output)
    }
}

/// Convenience wrapper for one matrix. Full models should share
/// [`NpuOpusMixedExecutor`] instances across their packed projections.
pub struct NpuOpusMixedGemmMp {
    executor: NpuOpusMixedExecutor,
    matrix: OpusMixedPackedMatrix,
}

impl NpuOpusMixedGemmMp {
    pub fn load_cached(
        w4_cache: &str,
        sparse3_cache: &str,
        k: usize,
        n: usize,
        compact: &[u8],
        awq_scale: Option<Vec<f32>>,
    ) -> Result<Self, XdnaError> {
        let executor = NpuOpusMixedExecutor::load_cached(w4_cache, sparse3_cache, n)?;
        let matrix = executor.pack_matrix(k, n, compact, awq_scale)?;
        Ok(Self { executor, matrix })
    }

    pub fn k(&self) -> usize {
        self.matrix.k()
    }

    pub fn n(&self) -> usize {
        self.matrix.n()
    }

    pub fn rows_per_dispatch(&self) -> usize {
        self.executor.rows_per_dispatch()
    }

    pub fn run_f32(&mut self, m: usize, x: &[f32], c: &mut [f32]) -> Result<(), XdnaError> {
        self.executor.run_f32(&self.matrix, m, x, c)
    }

    pub fn reference_f32(&self, m: usize, x: &[f32]) -> Result<Vec<f32>, XdnaError> {
        self.executor.reference_f32(&self.matrix, m, x)
    }
}

struct DecodedGroup {
    q4: Vec<i8>,
    residual: Vec<i8>,
    sparse_residual_chunks: Vec<Vec<u8>>,
    scales: Vec<f32>,
}

fn decode_compact(compact: &[u8], k: usize, n: usize) -> Result<Vec<DecodedGroup>, XdnaError> {
    let group_count = k / GROUP;
    let blocks = n * group_count;
    if blocks == 0 || compact.len() % blocks != 0 {
        return Err(invalid(format!(
            "{} bytes not divisible by {blocks} blocks",
            compact.len()
        )));
    }
    let block_bytes = compact.len() / blocks;
    if block_bytes < 132 || (block_bytes - 130) % 2 != 0 {
        return Err(invalid(format!(
            "block size {block_bytes} is not 130+2*N_out"
        )));
    }
    let outlier_count = (block_bytes - 130) / 2;
    let sparse_chunk_count = outlier_count.div_ceil(3);
    let mut groups: Vec<DecodedGroup> = (0..group_count)
        .map(|_| DecodedGroup {
            q4: vec![0; GROUP * n],
            residual: vec![0; GROUP * n],
            sparse_residual_chunks: vec![vec![0; n * 6]; sparse_chunk_count],
            scales: vec![0.0; n],
        })
        .collect();
    for col in 0..n {
        for (group_idx, group) in groups.iter_mut().enumerate() {
            let offset = (col * group_count + group_idx) * block_bytes;
            group.scales[col] =
                f16_to_f32(u16::from_le_bytes([compact[offset], compact[offset + 1]]));
            for packed_idx in 0..128 {
                let packed = compact[offset + 2 + packed_idx];
                for (lane, nibble) in [(0, packed & 0x0f), (1, packed >> 4)] {
                    let inner = 2 * packed_idx + lane;
                    let value = sext4(nibble);
                    group.q4[inner * n + col] = value;
                }
            }
            let mut seen_outlier = [false; GROUP];
            for outlier_idx in 0..outlier_count {
                let table = offset + 130 + 2 * outlier_idx;
                let inner = compact[table] as usize;
                if seen_outlier[inner] {
                    return Err(invalid(format!(
                        "duplicate sparse overlay index {inner} in column {col} group {group_idx}"
                    )));
                }
                seen_outlier[inner] = true;
                let replacement = compact[table + 1] as i8;
                let index = inner * n + col;
                let delta = replacement as i16 - group.q4[index] as i16;
                if !(-128..=127).contains(&delta) {
                    return Err(invalid(format!(
                        "outlier residual {delta} does not fit int8"
                    )));
                }
                group.residual[index] = delta as i8;
                let sparse_chunk = outlier_idx / 3;
                let sparse_lane = outlier_idx % 3;
                let sparse_offset = col * 6 + 2 * sparse_lane;
                group.sparse_residual_chunks[sparse_chunk][sparse_offset] = inner as u8;
                group.sparse_residual_chunks[sparse_chunk][sparse_offset + 1] = delta as i8 as u8;
            }
        }
    }
    Ok(groups)
}

fn prepare_activations(
    m: usize,
    k: usize,
    x: &[f32],
    awq_scale: Option<&[f32]>,
    rows_per_dispatch: usize,
) -> PreparedActivations {
    let group_count = k / GROUP;
    let padded_rows = m.div_ceil(rows_per_dispatch) * rows_per_dispatch;
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);
    let mut groups = vec![vec![0i8; padded_rows * GROUP]; group_count];
    let mut scales = vec![vec![1.0f32; padded_rows]; group_count];
    for row in 0..m {
        for group_idx in 0..group_count {
            let mut rotated = [0.0f32; GROUP];
            for inner in 0..GROUP {
                let column = group_idx * GROUP + inner;
                rotated[inner] = awq_scale.map_or(x[row * k + column], |scale| {
                    x[row * k + column] / scale[column]
                });
            }
            cpu_fwht_256(&mut rotated, &signs1, &signs2);
            let scale = rotated
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()))
                / 127.0;
            scales[group_idx][row] = if scale > 0.0 { scale } else { 1.0 };
            for inner in 0..GROUP {
                groups[group_idx][row * GROUP + inner] = (rotated[inner] / scales[group_idx][row])
                    .round()
                    .clamp(-127.0, 127.0)
                    as i8;
            }
        }
    }
    PreparedActivations {
        groups,
        scales,
        padded_rows,
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_scaled(
    m: usize,
    n: usize,
    values: &[i32],
    activation_scales: &[f32],
    weight_scales: &[f32],
    output: &mut [f32],
) {
    for row in 0..m {
        for col in 0..n {
            let index = row * n + col;
            output[index] += values[index] as f32 * activation_scales[row] * weight_scales[col];
        }
    }
}

fn validate_run_shapes(
    m: usize,
    k: usize,
    n: usize,
    x: &[f32],
    c: &[f32],
) -> Result<(), XdnaError> {
    if m == 0 || x.len() != m * k || c.len() != m * n {
        return Err(invalid(format!(
            "run wants X={} elements and C={} elements, got X={} C={}",
            m * k,
            m * n,
            x.len(),
            c.len()
        )));
    }
    Ok(())
}

fn sext4(nibble: u8) -> i8 {
    let value = (nibble & 0x0f) as i8;
    if value > 7 {
        value - 16
    } else {
        value
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpusMixed(message.into())
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn lcm(left: usize, right: usize) -> usize {
    left / gcd(left, right) * right
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_primitives::conv::f32_to_f16;

    #[test]
    fn compact_decode_splits_bulk_and_sparse_residual() {
        let mut block = vec![0u8; 136];
        block[..2].copy_from_slice(&f32_to_f16(0.25).to_le_bytes());
        block[2] = 0x7f;
        block[130..136].copy_from_slice(&[0, 20, 1, (-30i8) as u8, 255, 100]);
        let decoded = decode_compact(&block, 256, 1).unwrap();
        assert_eq!(decoded[0].q4[0], -1);
        assert_eq!(decoded[0].q4[1], 7);
        assert_eq!(decoded[0].residual[0], 21);
        assert_eq!(decoded[0].residual[1], -37);
        assert_eq!(decoded[0].residual[255], 100);
        assert_eq!(decoded[0].sparse_residual_chunks.len(), 1);
        assert_eq!(
            decoded[0].sparse_residual_chunks[0],
            vec![0, 21, 1, (-37i8) as u8, 255, 100]
        );
        assert!((decoded[0].scales[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn compact_decode_chunks_variable_sparse_overlays() {
        let mut block = vec![0u8; 140];
        block[..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        block[130..140].copy_from_slice(&[0, 20, 1, 30, 2, 40, 3, 50, 4, 60]);
        let decoded = decode_compact(&block, 256, 1).unwrap();
        assert_eq!(decoded[0].sparse_residual_chunks.len(), 2);
        assert_eq!(
            decoded[0].sparse_residual_chunks[0],
            vec![0, 20, 1, 30, 2, 40]
        );
        assert_eq!(
            decoded[0].sparse_residual_chunks[1],
            vec![3, 50, 4, 60, 0, 0]
        );
        assert_eq!(decoded[0].residual[4], 60);
    }

    #[test]
    fn compact_decode_rejects_duplicate_sparse_indices() {
        let mut block = vec![0u8; 134];
        block[130..134].copy_from_slice(&[7, 20, 7, 30]);
        assert!(decode_compact(&block, 256, 1).is_err());
    }

    #[test]
    fn activation_preparation_applies_awq_and_padding() {
        let x = vec![2.0f32; 256];
        let awq = vec![2.0f32; 256];
        let prepared = prepare_activations(1, 256, &x, Some(&awq), 128);
        assert_eq!(prepared.padded_rows, 128);
        assert_eq!(prepared.groups[0].len(), 128 * 256);
        assert!(prepared.scales[0][0].is_finite());
        assert!(prepared.groups[0][..256].iter().any(|value| *value != 0));
        assert!(prepared.groups[0][256..].iter().all(|value| *value == 0));
    }
}
