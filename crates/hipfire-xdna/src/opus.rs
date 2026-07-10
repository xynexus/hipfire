//! Format-generic Opus GEMM over AIE2P W4A8, W8A8, and sparse-overlay kernels.
//!
//! W4 (`qt=33/34`), compact mixed (`qt=36`), and W8 (`qt=35`) share one
//! activation preprocessing and output reconstruction contract. Mixed matrices
//! evaluate as `A·Q4 + A·(Q8-Q4)` with a variable number of sparse chunks.
//!
//! Plain, `+`, and `++` artifacts share this runtime. `+`/`++` optionally pass
//! the tensor's AWQ sidecar; `++` changes only the offline packed values.
#![cfg(target_os = "linux")]

use hipfire_primitives::{
    conv::f16_to_f32,
    fwht::{cpu_fwht_256, gen_fwht_signs},
};

use crate::{
    NpuFullKMode, NpuFullKResidentWeights, NpuGemmFullK, NpuGemmMp, NpuGemmResidentWeights,
    NpuGemmWholeArray, NpuSparse3Mp, NpuSparse3ResidentWeights, NpuWholeMode,
    NpuWholeResidentWeights, XdnaError,
};
use std::collections::HashMap;

const GROUP: usize = 256;

/// Runtime encoding for any grouped Opus projection matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpusMatrixEncoding {
    /// Pure signed-int4 groups (`qt=33` or canonical `qt=34`).
    W4,
    /// Int4 bulk plus a variable number of int8 replacements (`qt=36`).
    Mixed { overlays: usize },
    /// Pure signed-int8 groups (`qt=35`).
    W8,
}

impl OpusMatrixEncoding {
    /// Classify and validate one rank-two HFQ tensor from its quant type and
    /// byte geometry. `data_len` is the complete tensor payload length.
    pub fn classify(
        quant_type: u8,
        data_len: usize,
        k: usize,
        n: usize,
    ) -> Result<Self, XdnaError> {
        if k == 0 || n == 0 {
            return Err(invalid(format!(
                "Opus matrix wants non-zero K and N, got K={k} N={n}"
            )));
        }
        let blocks = n * k.div_ceil(GROUP);
        if data_len % blocks != 0 {
            return Err(invalid(format!(
                "{data_len} payload bytes not divisible by {blocks} Opus blocks"
            )));
        }
        let block_bytes = data_len / blocks;
        match quant_type {
            33 | 34 if block_bytes == 130 => Ok(Self::W4),
            35 if block_bytes == 258 => Ok(Self::W8),
            36 if block_bytes >= 132 && (block_bytes - 130) % 2 == 0 => Ok(Self::Mixed {
                overlays: (block_bytes - 130) / 2,
            }),
            33 | 34 => Err(invalid(format!(
                "W4 qt={quant_type} wants 130-byte blocks, got {block_bytes}"
            ))),
            35 => Err(invalid(format!(
                "W8 qt=35 wants 258-byte blocks, got {block_bytes}"
            ))),
            36 => Err(invalid(format!(
                "mixed qt=36 wants 130+2*N nonzero-overlay blocks, got {block_bytes}"
            ))),
            other => Err(invalid(format!(
                "unsupported Opus quant type {other}; expected 33/34/35/36"
            ))),
        }
    }
}

enum ResidentBaseWeights {
    W4(NpuGemmResidentWeights),
    W8(NpuGemmResidentWeights),
}

struct OpusGroup {
    resident_base: Option<ResidentBaseWeights>,
    resident_sparse: Vec<NpuSparse3ResidentWeights>,
    scales: Vec<f32>,
    base: Vec<i8>,
    residual: Vec<i8>,
}

struct PreparedActivations {
    groups: Vec<Vec<i8>>,
    scales: Vec<Vec<f32>>,
    padded_rows: usize,
}

/// Host-packed compact mixed-precision Opus matrix reusable with a resident
/// executor of the same output width.
pub struct OpusPackedMatrix {
    encoding: OpusMatrixEncoding,
    groups: Vec<OpusGroup>,
    awq_scale: Option<Vec<f32>>,
    k: usize,
    n: usize,
    fullk_weights: Option<NpuFullKResidentWeights>,
    whole_weights: Option<NpuWholeResidentWeights>,
}

impl OpusPackedMatrix {
    pub fn encoding(&self) -> OpusMatrixEncoding {
        self.encoding
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

/// Resident W4, W8, and sparse-overlay kernels shared by Opus matrices with one `N`.
pub struct NpuOpusExecutor {
    w4: Option<NpuGemmMp>,
    w8: Option<NpuGemmMp>,
    residual_sparse3: Option<NpuSparse3Mp>,
    n: usize,
    rows_per_dispatch: usize,
    fullk: HashMap<(NpuFullKMode, usize), NpuGemmFullK>,
    whole: HashMap<(NpuWholeMode, usize), NpuGemmWholeArray>,
}

impl NpuOpusExecutor {
    /// Load W4, W8, and sparse-overlay caches for matrices with output width `N`.
    pub fn load_cached(
        w4_cache: &str,
        w8_cache: &str,
        sparse3_cache: &str,
        n: usize,
    ) -> Result<Self, XdnaError> {
        if n == 0 {
            return Err(invalid("want non-zero N"));
        }
        let w4 = NpuGemmMp::load_cached(w4_cache)?;
        let w8 = NpuGemmMp::load_cached(w8_cache)?;
        let residual_sparse3 = NpuSparse3Mp::load_cached(sparse3_cache)?;
        if w4.weight_bits() != 4 {
            return Err(invalid("base cache must be W4"));
        }
        if w8.weight_bits() != 8 {
            return Err(invalid("dense cache must be W8"));
        }
        if w4.k() != GROUP
            || w8.k() != GROUP
            || residual_sparse3.k() != GROUP
            || w4.n() != n
            || w8.n() != n
            || residual_sparse3.n() != n
        {
            return Err(invalid(format!(
                "cache shapes must all be K=256 N={n}; got W4 {}x{} W8 {}x{} sparse3 {}x{}",
                w4.k(),
                w4.n(),
                w8.k(),
                w8.n(),
                residual_sparse3.k(),
                residual_sparse3.n()
            )));
        }
        let rows_per_dispatch = lcm(
            lcm(w4.rows_per_dispatch(), w8.rows_per_dispatch()),
            residual_sparse3.rows_per_dispatch(),
        );
        Ok(Self {
            w4: Some(w4),
            w8: Some(w8),
            residual_sparse3: Some(residual_sparse3),
            n,
            rows_per_dispatch,
            fullk: HashMap::new(),
            whole: HashMap::new(),
        })
    }

    /// Load only complete-projection caches. This avoids allocating legacy
    /// per-group W4/W8/sparse hardware contexts in a resident full-K model.
    pub fn load_fullk_cached(caches: &[(&str, usize)], n: usize) -> Result<Self, XdnaError> {
        if n == 0 || caches.is_empty() {
            return Err(invalid("full-K executor wants non-zero N and caches"));
        }
        let mut executor = Self {
            w4: None,
            w8: None,
            residual_sparse3: None,
            n,
            rows_per_dispatch: 0,
            fullk: HashMap::new(),
            whole: HashMap::new(),
        };
        for &(cache, cols) in caches {
            executor.enable_fullk_cache(cache, cols)?;
        }
        Ok(executor)
    }

    /// Load only AIE2P whole-array caches. W4 and W8 projections share this
    /// executor; mixed matrices still require a mixed/full-K fallback.
    pub fn load_whole_cached(caches: &[&str], n: usize) -> Result<Self, XdnaError> {
        if n == 0 || caches.is_empty() {
            return Err(invalid("whole-array executor wants non-zero N and caches"));
        }
        let mut executor = Self {
            w4: None,
            w8: None,
            residual_sparse3: None,
            n,
            rows_per_dispatch: 0,
            fullk: HashMap::new(),
            whole: HashMap::new(),
        };
        for &cache in caches {
            executor.enable_whole_cache(cache)?;
        }
        Ok(executor)
    }

    pub fn enable_whole_cache(&mut self, cache: &str) -> Result<(), XdnaError> {
        let whole = NpuGemmWholeArray::load_cached(cache)?;
        if whole.n() != self.n {
            return Err(invalid(format!(
                "whole-array cache N={} does not match executor N={}",
                whole.n(),
                self.n
            )));
        }
        self.whole.insert((whole.mode(), whole.k()), whole);
        Ok(())
    }

    /// Add a one-dispatch projection cache. Multiple K widths and formats may
    /// coexist for this executor's output width.
    pub fn enable_fullk_cache(&mut self, cache: &str, cols: usize) -> Result<(), XdnaError> {
        let fullk = NpuGemmFullK::load_cached(cache, cols)?;
        if fullk.n() != self.n {
            return Err(invalid(format!(
                "full-K cache N={} does not match executor N={}",
                fullk.n(),
                self.n
            )));
        }
        self.fullk.insert((fullk.mode(), fullk.k()), fullk);
        Ok(())
    }

    /// Decode, prepack, and upload a row-major `[N,K]` Opus matrix.
    pub fn pack_matrix(
        &self,
        quant_type: u8,
        k: usize,
        n: usize,
        payload: &[u8],
        awq_scale: Option<Vec<f32>>,
    ) -> Result<OpusPackedMatrix, XdnaError> {
        if k == 0 || n == 0 || n != self.n {
            return Err(invalid(format!(
                "want non-zero K and executor N={}, got K={k} N={n}",
                self.n
            )));
        }
        if awq_scale.as_ref().is_some_and(|scale| scale.len() != k) {
            return Err(invalid(format!("AWQ scale length must be K={k}")));
        }
        let encoding = OpusMatrixEncoding::classify(quant_type, payload.len(), k, n)?;
        let decoded = decode_opus_groups(encoding, payload, k, n)?;
        let mode = fullk_mode(encoding);
        let whole_mode = whole_mode(encoding);
        let padded_k = decoded.len() * GROUP;
        let whole_weights = if let Some(mode) = whole_mode {
            if let Some(whole) = self.whole.get(&(mode, padded_k)) {
                let base: Vec<&[i8]> = decoded.iter().map(|group| group.base.as_slice()).collect();
                let packed = whole.prepack_weights(&base)?;
                Some(whole.upload_resident_weights(&packed)?)
            } else {
                None
            }
        } else {
            None
        };
        let fullk_weights = if let Some(fullk) = self.fullk.get(&(mode, padded_k)) {
            let base: Vec<&[i8]> = decoded.iter().map(|group| group.base.as_slice()).collect();
            let residual: Vec<&[i8]> = if mode == NpuFullKMode::Mixed {
                decoded
                    .iter()
                    .map(|group| group.residual.as_slice())
                    .collect()
            } else {
                Vec::new()
            };
            let scales: Vec<&[f32]> = decoded
                .iter()
                .map(|group| group.scales.as_slice())
                .collect();
            let packed = if fullk.scaled_output() {
                fullk.prepack_weights_with_scales(&base, &residual, &scales)?
            } else {
                fullk.prepack_weights(&base, &residual)?
            };
            Some(fullk.upload_resident_weights(&packed)?)
        } else {
            None
        };
        let resident_only =
            self.w4.is_none() && self.w8.is_none() && self.residual_sparse3.is_none();
        if resident_only && fullk_weights.is_none() && whole_weights.is_none() {
            return Err(invalid(format!(
                "no {:?} full-K cache for padded K={padded_k} N={n}",
                mode
            )));
        }
        let mut groups = Vec::with_capacity(decoded.len());
        for decoded in decoded {
            let resident_base = if fullk_weights.is_some() || whole_weights.is_some() {
                None
            } else {
                Some(match encoding {
                    OpusMatrixEncoding::W4 | OpusMatrixEncoding::Mixed { .. } => {
                        let w4 = self
                            .w4
                            .as_ref()
                            .ok_or_else(|| invalid("missing W4 cache"))?;
                        let packed = w4.prepack_weights(GROUP, n, &decoded.base);
                        ResidentBaseWeights::W4(w4.upload_resident_weights(&packed)?)
                    }
                    OpusMatrixEncoding::W8 => {
                        let w8 = self
                            .w8
                            .as_ref()
                            .ok_or_else(|| invalid("missing W8 cache"))?;
                        let packed = w8.prepack_weights(GROUP, n, &decoded.base);
                        ResidentBaseWeights::W8(w8.upload_resident_weights(&packed)?)
                    }
                })
            };
            let mut resident_sparse = if fullk_weights.is_some() || whole_weights.is_some() {
                Vec::new()
            } else {
                Vec::with_capacity(decoded.sparse_residual_chunks.len())
            };
            if !decoded.sparse_residual_chunks.is_empty()
                && fullk_weights.is_none()
                && whole_weights.is_none()
            {
                let sparse_kernel = self
                    .residual_sparse3
                    .as_ref()
                    .ok_or_else(|| invalid("missing sparse residual cache"))?;
                for sparse in &decoded.sparse_residual_chunks {
                    resident_sparse.push(sparse_kernel.upload_resident_weights(sparse)?);
                }
            }
            groups.push(OpusGroup {
                resident_base,
                resident_sparse,
                scales: decoded.scales,
                base: decoded.base,
                residual: decoded.residual,
            });
        }
        Ok(OpusPackedMatrix {
            encoding,
            groups,
            awq_scale,
            k,
            n,
            fullk_weights,
            whole_weights,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn rows_per_dispatch(&self) -> usize {
        self.rows_per_dispatch
    }

    fn rows_per_dispatch_for(&self, encoding: OpusMatrixEncoding) -> usize {
        match encoding {
            OpusMatrixEncoding::W4 => self
                .w4
                .as_ref()
                .map_or(self.rows_per_dispatch, NpuGemmMp::rows_per_dispatch),
            OpusMatrixEncoding::Mixed { .. } => lcm(
                self.w4.as_ref().map_or(1, NpuGemmMp::rows_per_dispatch),
                self.residual_sparse3
                    .as_ref()
                    .map_or(1, NpuSparse3Mp::rows_per_dispatch),
            ),
            OpusMatrixEncoding::W8 => self
                .w8
                .as_ref()
                .map_or(self.rows_per_dispatch, NpuGemmMp::rows_per_dispatch),
        }
    }

    fn rows_per_dispatch_for_matrix(&self, matrix: &OpusPackedMatrix) -> usize {
        if matrix.whole_weights.is_some() {
            if let Some(mode) = whole_mode(matrix.encoding) {
                if let Some(whole) = self.whole.get(&(mode, matrix.groups.len() * GROUP)) {
                    return whole.rows();
                }
            }
        }
        if matrix.fullk_weights.is_some() {
            if let Some(fullk) = self
                .fullk
                .get(&(fullk_mode(matrix.encoding), matrix.groups.len() * GROUP))
            {
                return fullk.rows();
            }
        }
        self.rows_per_dispatch_for(matrix.encoding)
    }

    /// Run `C[M,N] = X[M,K]·Wᵀ` with activation AWQ, FWHT-256, and int8 quantization.
    pub fn run_f32(
        &mut self,
        matrix: &OpusPackedMatrix,
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
            self.rows_per_dispatch_for_matrix(matrix),
        );
        c.fill(0.0);
        let padded_k = matrix.groups.len() * GROUP;
        if let (Some(mode), Some(weights)) = (whole_mode(matrix.encoding), &matrix.whole_weights) {
            if let Some(whole) = self.whole.get_mut(&(mode, padded_k)) {
                let chunk_rows = whole.rows();
                if prepared.padded_rows % chunk_rows == 0 {
                    let mut activations = vec![0i8; chunk_rows * padded_k];
                    let mut partials = vec![0i32; matrix.groups.len() * chunk_rows * matrix.n];
                    for row0 in (0..prepared.padded_rows).step_by(chunk_rows) {
                        activations.fill(0);
                        for row in 0..chunk_rows {
                            let source_row = row0 + row;
                            for (group_idx, group) in prepared.groups.iter().enumerate() {
                                let source = source_row * GROUP;
                                let destination = row * padded_k + group_idx * GROUP;
                                activations[destination..destination + GROUP]
                                    .copy_from_slice(&group[source..source + GROUP]);
                            }
                        }
                        whole.run_resident(weights, &activations, &mut partials)?;
                        let valid_rows = (m - row0.min(m)).min(chunk_rows);
                        for (group_idx, group) in matrix.groups.iter().enumerate() {
                            for row in 0..valid_rows {
                                for col in 0..matrix.n {
                                    let partial =
                                        partials[(group_idx * chunk_rows + row) * matrix.n + col];
                                    c[(row0 + row) * matrix.n + col] += partial as f32
                                        * prepared.scales[group_idx][row0 + row]
                                        * group.scales[col];
                                }
                            }
                        }
                    }
                    return Ok(());
                }
            }
        }
        let fullk_key = (fullk_mode(matrix.encoding), padded_k);
        if let (Some(fullk), Some(weights)) =
            (self.fullk.get_mut(&fullk_key), &matrix.fullk_weights)
        {
            let chunk_rows = fullk.rows();
            if prepared.padded_rows % chunk_rows == 0 {
                let mut activations = vec![0i8; chunk_rows * padded_k];
                let mut partials = vec![0i32; matrix.groups.len() * chunk_rows * matrix.n];
                let mut activation_scales = vec![0.0f32; matrix.groups.len() * chunk_rows];
                let mut scaled_output = vec![0.0f32; chunk_rows * matrix.n];
                for row0 in (0..prepared.padded_rows).step_by(chunk_rows) {
                    activations.fill(0);
                    for row in 0..chunk_rows {
                        let source_row = row0 + row;
                        for (group_idx, group) in prepared.groups.iter().enumerate() {
                            let source = source_row * GROUP;
                            let destination = row * padded_k + group_idx * GROUP;
                            activations[destination..destination + GROUP]
                                .copy_from_slice(&group[source..source + GROUP]);
                        }
                    }
                    if fullk.scaled_output() {
                        for group_idx in 0..matrix.groups.len() {
                            activation_scales[group_idx * chunk_rows..(group_idx + 1) * chunk_rows]
                                .copy_from_slice(
                                    &prepared.scales[group_idx][row0..row0 + chunk_rows],
                                );
                        }
                        fullk.run_resident_scaled(
                            weights,
                            &activations,
                            &activation_scales,
                            &mut scaled_output,
                        )?;
                        let valid_rows = (m - row0.min(m)).min(chunk_rows);
                        for row in 0..valid_rows {
                            c[(row0 + row) * matrix.n..(row0 + row + 1) * matrix.n]
                                .copy_from_slice(
                                    &scaled_output[row * matrix.n..(row + 1) * matrix.n],
                                );
                        }
                        continue;
                    }
                    fullk.run_resident(weights, &activations, &mut partials)?;
                    let valid_rows = (m - row0.min(m)).min(chunk_rows);
                    for (group_idx, group) in matrix.groups.iter().enumerate() {
                        for row in 0..valid_rows {
                            for col in 0..matrix.n {
                                let partial =
                                    partials[(group_idx * chunk_rows + row) * matrix.n + col];
                                c[(row0 + row) * matrix.n + col] += partial as f32
                                    * prepared.scales[group_idx][row0 + row]
                                    * group.scales[col];
                            }
                        }
                    }
                }
                return Ok(());
            }
        }
        let output_elements = prepared.padded_rows * matrix.n;
        let mut base_outputs = vec![vec![0i32; output_elements]; matrix.groups.len()];
        let activation_groups: Vec<&[i8]> = prepared.groups.iter().map(Vec::as_slice).collect();
        let mut output_groups: Vec<&mut [i32]> =
            base_outputs.iter_mut().map(Vec::as_mut_slice).collect();
        match matrix.encoding {
            OpusMatrixEncoding::W4 | OpusMatrixEncoding::Mixed { .. } => {
                let weights: Vec<&NpuGemmResidentWeights> = matrix
                    .groups
                    .iter()
                    .map(|group| {
                        match group
                            .resident_base
                            .as_ref()
                            .expect("fallback W4 matrix has resident weights")
                        {
                            ResidentBaseWeights::W4(weights) => weights,
                            ResidentBaseWeights::W8(_) => unreachable!("W4 matrix has W8 weights"),
                        }
                    })
                    .collect();
                self.w4
                    .as_mut()
                    .ok_or_else(|| invalid("missing W4 fallback cache"))?
                    .run_resident_batch(
                        &weights,
                        prepared.padded_rows,
                        GROUP,
                        matrix.n,
                        &activation_groups,
                        &mut output_groups,
                    )?;
            }
            OpusMatrixEncoding::W8 => {
                let weights: Vec<&NpuGemmResidentWeights> = matrix
                    .groups
                    .iter()
                    .map(|group| {
                        match group
                            .resident_base
                            .as_ref()
                            .expect("fallback W8 matrix has resident weights")
                        {
                            ResidentBaseWeights::W8(weights) => weights,
                            ResidentBaseWeights::W4(_) => unreachable!("W8 matrix has W4 weights"),
                        }
                    })
                    .collect();
                self.w8
                    .as_mut()
                    .ok_or_else(|| invalid("missing W8 fallback cache"))?
                    .run_resident_batch(
                        &weights,
                        prepared.padded_rows,
                        GROUP,
                        matrix.n,
                        &activation_groups,
                        &mut output_groups,
                    )?;
            }
        }
        drop(output_groups);

        let mut residual = vec![0i32; prepared.padded_rows * matrix.n];
        for (group_idx, group) in matrix.groups.iter().enumerate() {
            accumulate_scaled(
                m,
                matrix.n,
                &base_outputs[group_idx],
                &prepared.scales[group_idx],
                &group.scales,
                c,
            );
            for sparse_weights in &group.resident_sparse {
                self.residual_sparse3
                    .as_mut()
                    .ok_or_else(|| invalid("missing sparse fallback cache"))?
                    .run_resident(
                        sparse_weights,
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
        matrix: &OpusPackedMatrix,
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
            self.rows_per_dispatch_for_matrix(matrix),
        );
        for (group_idx, group) in matrix.groups.iter().enumerate() {
            for row in 0..m {
                for col in 0..matrix.n {
                    let dot: i32 = (0..GROUP)
                        .map(|inner| {
                            let activation = prepared.groups[group_idx][row * GROUP + inner] as i32;
                            let index = inner * matrix.n + col;
                            activation * (group.base[index] as i32 + group.residual[index] as i32)
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
/// [`NpuOpusExecutor`] instances across their packed projections.
pub struct NpuOpusGemmMp {
    executor: NpuOpusExecutor,
    matrix: OpusPackedMatrix,
}

impl NpuOpusGemmMp {
    pub fn load_whole_only(
        whole_cache: &str,
        quant_type: u8,
        k: usize,
        n: usize,
        payload: &[u8],
        awq_scale: Option<Vec<f32>>,
    ) -> Result<Self, XdnaError> {
        let executor = NpuOpusExecutor::load_whole_cached(&[whole_cache], n)?;
        let matrix = executor.pack_matrix(quant_type, k, n, payload, awq_scale)?;
        Ok(Self { executor, matrix })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_fullk_only(
        fullk_cache: &str,
        fullk_cols: usize,
        quant_type: u8,
        k: usize,
        n: usize,
        payload: &[u8],
        awq_scale: Option<Vec<f32>>,
    ) -> Result<Self, XdnaError> {
        let executor = NpuOpusExecutor::load_fullk_cached(&[(fullk_cache, fullk_cols)], n)?;
        let matrix = executor.pack_matrix(quant_type, k, n, payload, awq_scale)?;
        Ok(Self { executor, matrix })
    }

    pub fn load_cached(
        w4_cache: &str,
        w8_cache: &str,
        sparse3_cache: &str,
        quant_type: u8,
        k: usize,
        n: usize,
        payload: &[u8],
        awq_scale: Option<Vec<f32>>,
    ) -> Result<Self, XdnaError> {
        let executor = NpuOpusExecutor::load_cached(w4_cache, w8_cache, sparse3_cache, n)?;
        let matrix = executor.pack_matrix(quant_type, k, n, payload, awq_scale)?;
        Ok(Self { executor, matrix })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_cached_fullk(
        w4_cache: &str,
        w8_cache: &str,
        sparse3_cache: &str,
        fullk_cache: &str,
        fullk_cols: usize,
        quant_type: u8,
        k: usize,
        n: usize,
        payload: &[u8],
        awq_scale: Option<Vec<f32>>,
    ) -> Result<Self, XdnaError> {
        let mut executor = NpuOpusExecutor::load_cached(w4_cache, w8_cache, sparse3_cache, n)?;
        executor.enable_fullk_cache(fullk_cache, fullk_cols)?;
        let matrix = executor.pack_matrix(quant_type, k, n, payload, awq_scale)?;
        Ok(Self { executor, matrix })
    }

    pub fn k(&self) -> usize {
        self.matrix.k()
    }

    pub fn n(&self) -> usize {
        self.matrix.n()
    }

    pub fn rows_per_dispatch(&self) -> usize {
        self.executor.rows_per_dispatch_for_matrix(&self.matrix)
    }

    pub fn run_f32(&mut self, m: usize, x: &[f32], c: &mut [f32]) -> Result<(), XdnaError> {
        self.executor.run_f32(&self.matrix, m, x, c)
    }

    pub fn reference_f32(&self, m: usize, x: &[f32]) -> Result<Vec<f32>, XdnaError> {
        self.executor.reference_f32(&self.matrix, m, x)
    }
}

struct DecodedGroup {
    base: Vec<i8>,
    residual: Vec<i8>,
    sparse_residual_chunks: Vec<Vec<u8>>,
    scales: Vec<f32>,
}

fn decode_opus_groups(
    encoding: OpusMatrixEncoding,
    payload: &[u8],
    k: usize,
    n: usize,
) -> Result<Vec<DecodedGroup>, XdnaError> {
    let group_count = k.div_ceil(GROUP);
    let blocks = n * group_count;
    if blocks == 0 || payload.len() % blocks != 0 {
        return Err(invalid(format!(
            "{} bytes not divisible by {blocks} blocks",
            payload.len()
        )));
    }
    let block_bytes = payload.len() / blocks;
    let outlier_count = match encoding {
        OpusMatrixEncoding::W4 if block_bytes == 130 => 0,
        OpusMatrixEncoding::Mixed { overlays }
            if block_bytes == 130 + overlays.saturating_mul(2) =>
        {
            overlays
        }
        OpusMatrixEncoding::W8 if block_bytes == 258 => 0,
        _ => {
            return Err(invalid(format!(
                "block size {block_bytes} does not match {encoding:?}"
            )))
        }
    };
    let sparse_chunk_count = outlier_count.div_ceil(3);
    let mut groups: Vec<DecodedGroup> = (0..group_count)
        .map(|_| DecodedGroup {
            base: vec![0; GROUP * n],
            residual: vec![0; GROUP * n],
            sparse_residual_chunks: vec![vec![0; n * 6]; sparse_chunk_count],
            scales: vec![0.0; n],
        })
        .collect();
    for col in 0..n {
        for (group_idx, group) in groups.iter_mut().enumerate() {
            let offset = (col * group_count + group_idx) * block_bytes;
            group.scales[col] =
                f16_to_f32(u16::from_le_bytes([payload[offset], payload[offset + 1]]));
            match encoding {
                OpusMatrixEncoding::W4 | OpusMatrixEncoding::Mixed { .. } => {
                    for packed_idx in 0..128 {
                        let packed = payload[offset + 2 + packed_idx];
                        for (lane, nibble) in [(0, packed & 0x0f), (1, packed >> 4)] {
                            let inner = 2 * packed_idx + lane;
                            group.base[inner * n + col] = sext4(nibble);
                        }
                    }
                }
                OpusMatrixEncoding::W8 => {
                    for inner in 0..GROUP {
                        group.base[inner * n + col] = payload[offset + 2 + inner] as i8;
                    }
                }
            }
            let mut seen_outlier = [false; GROUP];
            for outlier_idx in 0..outlier_count {
                let table = offset + 130 + 2 * outlier_idx;
                let inner = payload[table] as usize;
                if seen_outlier[inner] {
                    return Err(invalid(format!(
                        "duplicate sparse overlay index {inner} in column {col} group {group_idx}"
                    )));
                }
                seen_outlier[inner] = true;
                let replacement = payload[table + 1] as i8;
                let index = inner * n + col;
                let delta = replacement as i16 - group.base[index] as i16;
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
    let group_count = k.div_ceil(GROUP);
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
                if column < k {
                    rotated[inner] = awq_scale.map_or(x[row * k + column], |scale| {
                        x[row * k + column] / scale[column]
                    });
                }
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
    XdnaError::InvalidOpus(message.into())
}

fn fullk_mode(encoding: OpusMatrixEncoding) -> NpuFullKMode {
    match encoding {
        OpusMatrixEncoding::W4 => NpuFullKMode::W4,
        OpusMatrixEncoding::Mixed { .. } => NpuFullKMode::Mixed,
        OpusMatrixEncoding::W8 => NpuFullKMode::W8,
    }
}

fn whole_mode(encoding: OpusMatrixEncoding) -> Option<NpuWholeMode> {
    match encoding {
        OpusMatrixEncoding::W4 => Some(NpuWholeMode::W4),
        OpusMatrixEncoding::W8 => Some(NpuWholeMode::W8),
        OpusMatrixEncoding::Mixed { .. } => None,
    }
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
    fn generic_opus_encoding_classifies_w4_mixed_and_w8() {
        assert_eq!(
            OpusMatrixEncoding::classify(33, 130, 256, 1).unwrap(),
            OpusMatrixEncoding::W4
        );
        assert_eq!(
            OpusMatrixEncoding::classify(34, 130, 256, 1).unwrap(),
            OpusMatrixEncoding::W4
        );
        assert_eq!(
            OpusMatrixEncoding::classify(36, 140, 256, 1).unwrap(),
            OpusMatrixEncoding::Mixed { overlays: 5 }
        );
        assert_eq!(
            OpusMatrixEncoding::classify(35, 258, 256, 1).unwrap(),
            OpusMatrixEncoding::W8
        );
        assert!(OpusMatrixEncoding::classify(36, 130, 256, 1).is_err());
        assert!(OpusMatrixEncoding::classify(35, 257, 256, 1).is_err());
        assert!(OpusMatrixEncoding::classify(7, 130, 256, 1).is_err());
        assert_eq!(
            OpusMatrixEncoding::classify(34, 5 * 130, 1152, 1).unwrap(),
            OpusMatrixEncoding::W4
        );
    }

    #[test]
    fn generic_opus_decode_accepts_zero_and_variable_overlays() {
        let mut w4 = vec![0u8; 130];
        w4[..2].copy_from_slice(&f32_to_f16(0.25).to_le_bytes());
        w4[2] = 0x7f;
        let decoded = decode_opus_groups(OpusMatrixEncoding::W4, &w4, 256, 1).unwrap();
        assert_eq!(decoded[0].base[0], -1);
        assert_eq!(decoded[0].base[1], 7);
        assert!(decoded[0].sparse_residual_chunks.is_empty());

        let mut w8 = vec![0u8; 258];
        w8[..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        w8[2] = (-120i8) as u8;
        w8[257] = 99;
        let decoded = decode_opus_groups(OpusMatrixEncoding::W8, &w8, 256, 1).unwrap();
        assert_eq!(decoded[0].base[0], -120);
        assert_eq!(decoded[0].base[255], 99);
        assert!(decoded[0].sparse_residual_chunks.is_empty());
    }

    #[test]
    fn compact_decode_splits_bulk_and_sparse_residual() {
        let mut block = vec![0u8; 136];
        block[..2].copy_from_slice(&f32_to_f16(0.25).to_le_bytes());
        block[2] = 0x7f;
        block[130..136].copy_from_slice(&[0, 20, 1, (-30i8) as u8, 255, 100]);
        let decoded =
            decode_opus_groups(OpusMatrixEncoding::Mixed { overlays: 3 }, &block, 256, 1).unwrap();
        assert_eq!(decoded[0].base[0], -1);
        assert_eq!(decoded[0].base[1], 7);
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
        let decoded =
            decode_opus_groups(OpusMatrixEncoding::Mixed { overlays: 5 }, &block, 256, 1).unwrap();
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
        assert!(
            decode_opus_groups(OpusMatrixEncoding::Mixed { overlays: 2 }, &block, 256, 1,).is_err()
        );
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

    #[test]
    fn activation_preparation_zero_pads_a_tail_k_group_before_rotation() {
        let x = vec![1.0f32; 1152];
        let prepared = prepare_activations(1, 1152, &x, None, 1);
        assert_eq!(prepared.groups.len(), 5);
        assert_eq!(prepared.groups[4].len(), 256);
        assert!(prepared.groups[4].iter().any(|value| *value != 0));
    }
}
