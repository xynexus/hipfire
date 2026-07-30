//! One-dispatch full-K AIE2P GEMM primitive.
//!
//! Generated full-K caches stream every K=256 group and N=64 slab through one
//! AIE runtime sequence. The NPU returns exact int32 partials per K group; the
//! Opus layer applies each group's activation/weight scales afterwards. Mixed
//! caches accumulate a W4 base and dense W8 residual into one int32 partial on
//! AIE without changing their shared group scale.
#![cfg(target_os = "linux")]

use std::path::Path;

use crate::opus_hfp::{self, OpusHfpDescriptor, OpusHfpEncoding, OpusHfpLayout};
use crate::{DeviceBuffer, NpuKernel, XdnaError};

const GROUP_K: usize = 256;
const SLAB_N: usize = 64;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NpuFullKMode {
    W4,
    Mixed,
    W8,
}

impl NpuFullKMode {
    fn weight_bytes(self) -> usize {
        match self {
            Self::W4 => 8192,
            Self::Mixed | Self::W8 => 16384,
        }
    }

    fn weight_entries(self) -> usize {
        if self == Self::Mixed {
            2
        } else {
            1
        }
    }

    fn output_components(self) -> usize {
        1
    }
}

/// Device-resident packed weights for one complete projection.
pub struct NpuFullKResidentWeights {
    buffer: DeviceBuffer,
    scales: Vec<f32>,
}

/// Reusable activation/output buffers and one compiled full-K AIE schedule.
pub struct NpuGemmFullK {
    kernel: NpuKernel,
    mode: NpuFullKMode,
    cols: usize,
    rows: usize,
    groups: usize,
    n: usize,
    direct_output: bool,
    scaled_output: bool,
    combined_input: bool,
    input: DeviceBuffer,
    scale_input: Option<DeviceBuffer>,
    output: DeviceBuffer,
}

impl NpuGemmFullK {
    /// Load a generated cache whose basename includes `_mM_kgG_nN` and one of
    /// `w4`, `mixed`, or `w8`. `cols` is explicit because it is a physical AIE
    /// array property and older experimental cache names did not encode it.
    pub fn load_cached(dir: &str, cols: usize) -> Result<Self, XdnaError> {
        let base = Path::new(dir)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        let rows = parse_prefixed(base, "m").ok_or_else(|| bad_cache(base))?;
        let groups = parse_prefixed(base, "kg").ok_or_else(|| bad_cache(base))?;
        let n = parse_prefixed(base, "n").ok_or_else(|| bad_cache(base))?;
        let tokens: Vec<&str> = base.split('_').collect();
        let mode = if tokens.contains(&"mixed") {
            NpuFullKMode::Mixed
        } else if tokens.contains(&"w8") {
            NpuFullKMode::W8
        } else if tokens.contains(&"w4") || tokens.contains(&"w4-scaled") {
            NpuFullKMode::W4
        } else {
            return Err(bad_cache(base));
        };
        if cols == 0 || rows == 0 || rows % cols != 0 || groups == 0 || n == 0 || n % SLAB_N != 0 {
            return Err(bad_cache(base));
        }

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?;
        let instructions = std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?;
        let output_layout =
            std::fs::read_to_string(format!("{dir}/output-layout.txt")).unwrap_or_default();
        let direct_output = matches!(output_layout.trim(), "direct" | "scaled-f32-direct");
        let scaled_output = output_layout.trim().starts_with("scaled-f32");
        let combined_input = scaled_output
            && std::fs::read_to_string(format!("{dir}/input-layout.txt"))
                .is_ok_and(|layout| layout.trim() == "combined");
        let kernel = NpuKernel::load(&xclbin, &instructions)?;
        let input_bytes = if combined_input {
            let rows_per_core = rows / cols;
            cols * (n / SLAB_N)
                * groups
                * (rows_per_core * GROUP_K
                    + (padded_scale_rows(rows_per_core) + SLAB_N) * std::mem::size_of::<f32>())
        } else {
            rows * groups * GROUP_K
        };
        let output_elements = if scaled_output {
            rows * n
        } else {
            rows * groups * n * mode.output_components()
        };
        let input = kernel.alloc_arg(input_bytes)?;
        let scale_input = if scaled_output && !combined_input {
            let rows_per_core = rows / cols;
            Some(kernel.alloc_arg(
                cols * (n / SLAB_N)
                    * groups
                    * (padded_scale_rows(rows_per_core) + SLAB_N)
                    * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        };
        let output = kernel.alloc_arg(output_elements * std::mem::size_of::<i32>())?;
        Ok(Self {
            kernel,
            mode,
            cols,
            rows,
            groups,
            n,
            direct_output,
            scaled_output,
            combined_input,
            input,
            scale_input,
            output,
        })
    }

    pub fn mode(&self) -> NpuFullKMode {
        self.mode
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn k(&self) -> usize {
        self.groups * GROUP_K
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn scaled_output(&self) -> bool {
        self.scaled_output
    }

    fn packed_weight_entry_bytes(&self) -> usize {
        self.mode.weight_entries() * self.mode.weight_bytes()
    }

    fn device_weight_bytes(&self) -> usize {
        self.groups * (self.n / SLAB_N) * self.packed_weight_entry_bytes()
    }

    pub fn packed_weight_bytes(&self) -> usize {
        self.device_weight_bytes()
            + usize::from(self.scaled_output) * self.groups * self.n * std::mem::size_of::<f32>()
    }

    fn hfp_descriptor(&self, quant_type: u8) -> OpusHfpDescriptor {
        let encoding = match self.mode {
            NpuFullKMode::W4 => OpusHfpEncoding::W4,
            NpuFullKMode::Mixed => OpusHfpEncoding::MixedW4WithOverlays,
            NpuFullKMode::W8 => OpusHfpEncoding::W8,
        };
        let flags = u32::from(self.direct_output)
            | (u32::from(self.scaled_output) << 1)
            | (u32::from(self.combined_input) << 2);
        OpusHfpDescriptor {
            encoding,
            layout: OpusHfpLayout::FullKV1,
            quant_type: quant_type.into(),
            flags,
            m: self.rows as u32,
            k: self.k() as u32,
            n: self.n as u32,
            columns: self.cols as u32,
            groups: self.groups as u32,
            m_macros: (self.rows / self.cols) as u32,
            n_macros: (self.n / SLAB_N) as u32,
            outblocks: (self.groups * (self.n / SLAB_N)) as u32,
            tile_bytes: self.packed_weight_entry_bytes() as u32,
            data_bytes: self.device_weight_bytes() as u32,
            scale_offset: self.device_weight_bytes() as u32,
            scale_values: if self.scaled_output {
                (self.groups * self.n) as u32
            } else {
                0
            },
            payload_bytes: self.packed_weight_bytes() as u64,
            segment_bytes: [0; 4],
        }
    }

    /// Load the schedule-ready full-K weight stream from `.rdna2.hfp`, or
    /// perform the global slab ordering once and persist it. W4 base entries
    /// remain nibble-packed; only the AIE kernel decodes/swizzles their nibbles.
    pub(crate) fn prepack_weights_cached(
        &self,
        path: &Path,
        quant_type: u8,
        source_payload: &[u8],
        base: &[&[i8]],
        residual: &[&[i8]],
        scales: &[&[f32]],
    ) -> Result<Vec<u8>, XdnaError> {
        let descriptor = self.hfp_descriptor(quant_type);
        let source_sha = opus_hfp::source_sha256(&[source_payload]);
        if let Some(packed) = opus_hfp::read(path, descriptor, source_sha).map_err(invalid)? {
            return Ok(packed);
        }
        let packed = self.prepack_weights_with_scales(base, residual, scales)?;
        opus_hfp::write(path, descriptor, source_sha, &packed).map_err(invalid)?;
        Ok(packed)
    }

    pub fn upload_resident_weights(
        &self,
        packed: &[u8],
    ) -> Result<NpuFullKResidentWeights, XdnaError> {
        if packed.len() != self.packed_weight_bytes() {
            return Err(invalid(format!(
                "full-K weights want {} bytes, got {}",
                self.packed_weight_bytes(),
                packed.len()
            )));
        }
        let device_bytes = self.device_weight_bytes();
        let mut buffer = self.kernel.alloc_arg(device_bytes)?;
        buffer
            .as_mut_slice()
            .copy_from_slice(&packed[..device_bytes]);
        self.kernel.sync_to_device(&buffer)?;
        let scales = if self.scaled_output {
            as_f32(&packed[device_bytes..]).to_vec()
        } else {
            Vec::new()
        };
        Ok(NpuFullKResidentWeights { buffer, scales })
    }

    /// Pack row-major K-group matrices into the generated broadcast schedule.
    /// `base[group]` and `residual[group]` are `[256,N]`; residuals are required
    /// only for mixed mode and may contain arbitrary per-column int8 deltas.
    pub fn prepack_weights(
        &self,
        base: &[&[i8]],
        residual: &[&[i8]],
    ) -> Result<Vec<u8>, XdnaError> {
        self.prepack_weights_with_scales(base, residual, &[])
    }

    pub fn prepack_weights_with_scales(
        &self,
        base: &[&[i8]],
        residual: &[&[i8]],
        scales: &[&[f32]],
    ) -> Result<Vec<u8>, XdnaError> {
        if base.len() != self.groups || base.iter().any(|weights| weights.len() != GROUP_K * self.n)
        {
            return Err(invalid("full-K base group geometry mismatch"));
        }
        if self.mode == NpuFullKMode::Mixed {
            if residual.len() != self.groups
                || residual
                    .iter()
                    .any(|weights| weights.len() != GROUP_K * self.n)
            {
                return Err(invalid("full-K residual group geometry mismatch"));
            }
        } else if !residual.is_empty() {
            return Err(invalid("residual weights require mixed full-K mode"));
        }
        if self.scaled_output {
            if scales.len() != self.groups || scales.iter().any(|scale| scale.len() != self.n) {
                return Err(invalid("scaled full-K weight-scale geometry mismatch"));
            }
        } else if !scales.is_empty() {
            return Err(invalid("weight scales require a scaled full-K cache"));
        }

        let entry_bytes = self.mode.weight_bytes();
        let packed_entry_bytes = self.packed_weight_entry_bytes();
        let nb = self.n / SLAB_N;
        let device_bytes = self.device_weight_bytes();
        let mut packed = vec![0u8; self.packed_weight_bytes()];
        for group in 0..self.groups {
            for slab in 0..nb {
                let entry_index = if self.scaled_output {
                    slab * self.groups + group
                } else {
                    group * nb + slab
                };
                let entry_offset = entry_index * packed_entry_bytes;
                let base_out = &mut packed[entry_offset..entry_offset + entry_bytes];
                match self.mode {
                    NpuFullKMode::W4 | NpuFullKMode::Mixed => {
                        pack_w4_slab(base[group], self.n, slab, &mut base_out[..8192])?;
                    }
                    NpuFullKMode::W8 => {
                        pack_w8_slab(base[group], self.n, slab, base_out);
                    }
                }
                if self.mode == NpuFullKMode::Mixed {
                    let residual_offset = entry_offset + entry_bytes;
                    let residual_out = &mut packed[residual_offset..residual_offset + entry_bytes];
                    pack_w8_slab(residual[group], self.n, slab, residual_out);
                }
                if self.scaled_output {
                    let scale_offset = device_bytes
                        + (group * self.n + slab * SLAB_N) * std::mem::size_of::<f32>();
                    packed[scale_offset..scale_offset + SLAB_N * std::mem::size_of::<f32>()]
                        .copy_from_slice(as_bytes_f32(
                            &scales[group][slab * SLAB_N..(slab + 1) * SLAB_N],
                        ));
                }
            }
        }
        Ok(packed)
    }

    /// Run one complete projection. `activations` is row-major `[M,K]`; `partials`
    /// is group-major `[K/256,M,N]`. Mixed base/residual outputs have already
    /// been accumulated on AIE before the caller applies their group scale.
    pub fn run_resident(
        &mut self,
        weights: &NpuFullKResidentWeights,
        activations: &[i8],
        partials: &mut [i32],
    ) -> Result<(), XdnaError> {
        if self.scaled_output {
            return Err(invalid("scaled full-K cache requires run_resident_scaled"));
        }
        if activations.len() != self.rows * self.k() {
            return Err(invalid(format!(
                "full-K activations want {} elements, got {}",
                self.rows * self.k(),
                activations.len()
            )));
        }
        if partials.len() != self.groups * self.rows * self.n {
            return Err(invalid(format!(
                "full-K partials want {} elements, got {}",
                self.groups * self.rows * self.n,
                partials.len()
            )));
        }
        if weights.buffer.len() != self.device_weight_bytes() {
            return Err(invalid("resident full-K weight shape changed"));
        }

        let rows_per_core = self.rows / self.cols;
        for core in 0..self.cols {
            for group in 0..self.groups {
                let destination = (core * self.groups + group) * rows_per_core * GROUP_K;
                for local_row in 0..rows_per_core {
                    let row = core * rows_per_core + local_row;
                    let source = row * self.k() + group * GROUP_K;
                    self.input.as_mut_slice()[destination + local_row * GROUP_K
                        ..destination + (local_row + 1) * GROUP_K]
                        .copy_from_slice(as_bytes(&activations[source..source + GROUP_K]));
                }
            }
        }

        self.kernel.dispatch_synced(
            &[&self.input, &weights.buffer, &self.output],
            &[true, false, true],
        )?;
        let physical = as_i32(self.output.as_slice());
        if self.direct_output {
            partials.copy_from_slice(physical);
        } else {
            let rows_per_core = self.rows / self.cols;
            let nb = self.n / SLAB_N;
            let block_elements = rows_per_core * SLAB_N;
            for core in 0..self.cols {
                for group in 0..self.groups {
                    for slab in 0..nb {
                        for local_row in 0..rows_per_core {
                            let row = core * rows_per_core + local_row;
                            for local_col in 0..SLAB_N {
                                let col = slab * SLAB_N + local_col;
                                let source = ((core * self.groups + group) * nb + slab)
                                    * block_elements
                                    + local_row * SLAB_N
                                    + local_col;
                                partials[(group * self.rows + row) * self.n + col] =
                                    physical[source];
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Run a scaled full-K projection. Activation scales are group-major
    /// `[K/256,M]`; resident weight scale tails are applied on AIE and the
    /// returned output is already accumulated row-major `[M,N]` f32.
    pub fn run_resident_scaled(
        &mut self,
        weights: &NpuFullKResidentWeights,
        activations: &[i8],
        activation_scales: &[f32],
        output: &mut [f32],
    ) -> Result<(), XdnaError> {
        if !self.scaled_output {
            return Err(invalid(
                "run_resident_scaled requires a scaled full-K cache",
            ));
        }
        if activations.len() != self.rows * self.k()
            || activation_scales.len() != self.groups * self.rows
            || output.len() != self.rows * self.n
        {
            return Err(invalid("scaled full-K activation/output geometry mismatch"));
        }
        if weights.buffer.len() != self.device_weight_bytes()
            || weights.scales.len() != self.groups * self.n
        {
            return Err(invalid("resident scaled full-K weight shape changed"));
        }

        let rows_per_core = self.rows / self.cols;
        let scale_bytes = (padded_scale_rows(rows_per_core) + SLAB_N) * std::mem::size_of::<f32>();
        if self.combined_input {
            let activation_bytes = rows_per_core * GROUP_K;
            let entry_bytes = activation_bytes + scale_bytes;
            for core in 0..self.cols {
                for slab in 0..self.n / SLAB_N {
                    for group in 0..self.groups {
                        let entry =
                            ((core * (self.n / SLAB_N) + slab) * self.groups + group) * entry_bytes;
                        for local_row in 0..rows_per_core {
                            let row = core * rows_per_core + local_row;
                            let source = row * self.k() + group * GROUP_K;
                            let destination = entry + local_row * GROUP_K;
                            self.input.as_mut_slice()[destination..destination + GROUP_K]
                                .copy_from_slice(as_bytes(&activations[source..source + GROUP_K]));
                        }
                        copy_scale_payload(
                            &mut self.input.as_mut_slice()
                                [entry + activation_bytes..entry + activation_bytes + scale_bytes],
                            activation_scales,
                            &weights.scales,
                            group,
                            core,
                            slab,
                            self.rows,
                            self.n,
                            rows_per_core,
                        );
                    }
                }
            }
            self.kernel.dispatch_synced(
                &[&self.input, &weights.buffer, &self.output],
                &[true, false, true],
            )?;
        } else {
            for core in 0..self.cols {
                for group in 0..self.groups {
                    let entry = (core * self.groups + group) * rows_per_core * GROUP_K;
                    for local_row in 0..rows_per_core {
                        let row = core * rows_per_core + local_row;
                        let source = row * self.k() + group * GROUP_K;
                        let destination = entry + local_row * GROUP_K;
                        self.input.as_mut_slice()[destination..destination + GROUP_K]
                            .copy_from_slice(as_bytes(&activations[source..source + GROUP_K]));
                    }
                }
            }
            let scale_input = self
                .scale_input
                .as_mut()
                .ok_or_else(|| invalid("scaled full-K cache has no scale input"))?;
            for core in 0..self.cols {
                for slab in 0..self.n / SLAB_N {
                    for group in 0..self.groups {
                        let offset =
                            ((core * (self.n / SLAB_N) + slab) * self.groups + group) * scale_bytes;
                        copy_scale_payload(
                            &mut scale_input.as_mut_slice()[offset..offset + scale_bytes],
                            activation_scales,
                            &weights.scales,
                            group,
                            core,
                            slab,
                            self.rows,
                            self.n,
                            rows_per_core,
                        );
                    }
                }
            }
            self.kernel.dispatch_synced(
                &[&self.input, &weights.buffer, scale_input, &self.output],
                &[true, false, true, true],
            )?;
        }
        let physical = as_f32(self.output.as_slice());
        if self.direct_output {
            output.copy_from_slice(physical);
        } else {
            let nb = self.n / SLAB_N;
            for core in 0..self.cols {
                for slab in 0..nb {
                    for local_row in 0..rows_per_core {
                        let row = core * rows_per_core + local_row;
                        let source = ((core * nb + slab) * rows_per_core + local_row) * SLAB_N;
                        let destination = row * self.n + slab * SLAB_N;
                        output[destination..destination + SLAB_N]
                            .copy_from_slice(&physical[source..source + SLAB_N]);
                    }
                }
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
/// Activation-scale rows rounded up so the weight scales that follow start
/// 64-byte aligned for the kernel's `aie::load_v<16>`.
fn padded_scale_rows(rows_per_core: usize) -> usize {
    rows_per_core.div_ceil(16) * 16
}

fn copy_scale_payload(
    destination: &mut [u8],
    activation_scales: &[f32],
    weight_scales: &[f32],
    group: usize,
    core: usize,
    slab: usize,
    rows: usize,
    n: usize,
    rows_per_core: usize,
) {
    // The weight scales must start 64-byte aligned so the kernel's
    // `aie::load_v<16>` reads the right lanes, so the activation region is
    // padded up to a multiple of 16 floats. Mirror of `ROWS_PADDED` in
    // r6_scale_accum.cc / r6_gen_mp_fullk_scaled.py.
    let live_bytes = rows_per_core * std::mem::size_of::<f32>();
    let activation_bytes = padded_scale_rows(rows_per_core) * std::mem::size_of::<f32>();
    destination[..live_bytes].copy_from_slice(as_bytes_f32(
        &activation_scales
            [group * rows + core * rows_per_core..group * rows + (core + 1) * rows_per_core],
    ));
    destination[live_bytes..activation_bytes].fill(0);
    destination[activation_bytes..].copy_from_slice(as_bytes_f32(
        &weight_scales[group * n + slab * SLAB_N..group * n + (slab + 1) * SLAB_N],
    ));
}

fn parse_prefixed(name: &str, prefix: &str) -> Option<usize> {
    name.split('_').find_map(|token| {
        token
            .strip_prefix(prefix)
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|digits| digits.parse().ok())
    })
}

fn pack_w4_slab(weights: &[i8], n: usize, slab: usize, output: &mut [u8]) -> Result<(), XdnaError> {
    for nt in 0..4 {
        for k_tile in 0..16 {
            for kk in 0..16 {
                for nn in 0..16 {
                    let k = k_tile * 16 + kk;
                    let col = slab * SLAB_N + nt * 16 + nn;
                    let value = weights[k * n + col];
                    if !(-8..=7).contains(&value) {
                        return Err(invalid(format!("W4 value {value} outside -8..=7")));
                    }
                    let index = (nt * 16 + k_tile) * 256 + kk * 16 + nn;
                    let nibble = (value & 0x0f) as u8;
                    output[index / 2] |= if index % 2 == 0 { nibble } else { nibble << 4 };
                }
            }
        }
    }
    Ok(())
}

fn pack_w8_slab(weights: &[i8], n: usize, slab: usize, output: &mut [u8]) {
    for nt in 0..4 {
        for k_tile in 0..32 {
            for n_half in 0..2 {
                for kk in 0..8 {
                    for nn in 0..8 {
                        let k = k_tile * 8 + kk;
                        let col = slab * SLAB_N + nt * 16 + n_half * 8 + nn;
                        let index = ((nt * 32 + k_tile) * 2 + n_half) * 64 + kk * 8 + nn;
                        output[index] = weights[k * n + col] as u8;
                    }
                }
            }
        }
    }
}

fn as_bytes(values: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len()) }
}

fn as_i32(values: &[u8]) -> &[i32] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<i32>(),
            values.len() / std::mem::size_of::<i32>(),
        )
    }
}

fn as_f32(values: &[u8]) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<f32>(),
            values.len() / std::mem::size_of::<f32>(),
        )
    }
}

fn as_bytes_f32(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn bad_cache(name: &str) -> XdnaError {
    XdnaError::BadCacheName(name.to_string())
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fullk_cache_dimensions_without_confusing_prefixes() {
        let name = "embgemma_aie2p_fullk_submit_mixed_dense_m256_kg12_n3072";
        assert_eq!(parse_prefixed(name, "m"), Some(256));
        assert_eq!(parse_prefixed(name, "kg"), Some(12));
        assert_eq!(parse_prefixed(name, "n"), Some(3072));
        assert_eq!(parse_prefixed(name, "x"), None);
    }

    #[test]
    fn mode_geometry_accumulates_mixed_components_on_aie() {
        assert_eq!(NpuFullKMode::W4.weight_bytes(), 8192);
        assert_eq!(NpuFullKMode::W8.weight_bytes(), 16384);
        assert_eq!(NpuFullKMode::Mixed.weight_entries(), 2);
        assert_eq!(NpuFullKMode::Mixed.output_components(), 1);
    }
}
