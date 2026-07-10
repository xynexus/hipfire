//! One-dispatch full-K AIE2P GEMM primitive.
//!
//! Generated full-K caches stream every K=256 group and N=64 slab through one
//! AIE runtime sequence. The NPU returns exact int32 partials per K group; the
//! Opus layer applies each group's activation/weight scales afterwards. Mixed
//! caches accumulate a W4 base and dense W8 residual into one int32 partial on
//! AIE without changing their shared group scale.
#![cfg(target_os = "linux")]

use std::path::Path;

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
    input: DeviceBuffer,
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
        } else if tokens.contains(&"w4") {
            NpuFullKMode::W4
        } else {
            return Err(bad_cache(base));
        };
        if cols == 0 || rows == 0 || rows % cols != 0 || groups == 0 || n == 0 || n % SLAB_N != 0 {
            return Err(bad_cache(base));
        }

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?;
        let instructions = std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?;
        let direct_output = std::fs::read_to_string(format!("{dir}/output-layout.txt"))
            .is_ok_and(|layout| layout.trim() == "direct");
        let kernel = NpuKernel::load(&xclbin, &instructions)?;
        let input = kernel.alloc_arg(rows * groups * GROUP_K)?;
        let output = kernel
            .alloc_arg(rows * groups * n * mode.output_components() * std::mem::size_of::<i32>())?;
        Ok(Self {
            kernel,
            mode,
            cols,
            rows,
            groups,
            n,
            direct_output,
            input,
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

    pub fn packed_weight_bytes(&self) -> usize {
        self.groups * (self.n / SLAB_N) * self.mode.weight_entries() * self.mode.weight_bytes()
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
        let mut buffer = self.kernel.alloc_arg(packed.len())?;
        buffer.as_mut_slice().copy_from_slice(packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuFullKResidentWeights { buffer })
    }

    /// Pack row-major K-group matrices into the generated broadcast schedule.
    /// `base[group]` and `residual[group]` are `[256,N]`; residuals are required
    /// only for mixed mode and may contain arbitrary per-column int8 deltas.
    pub fn prepack_weights(
        &self,
        base: &[&[i8]],
        residual: &[&[i8]],
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

        let entry_bytes = self.mode.weight_bytes();
        let entries = self.mode.weight_entries();
        let nb = self.n / SLAB_N;
        let mut packed = vec![0u8; self.packed_weight_bytes()];
        for group in 0..self.groups {
            for slab in 0..nb {
                let entry = (group * nb + slab) * entries;
                let base_out = &mut packed[entry * entry_bytes..(entry + 1) * entry_bytes];
                match self.mode {
                    NpuFullKMode::W4 | NpuFullKMode::Mixed => {
                        pack_w4_slab(base[group], self.n, slab, &mut base_out[..8192])?;
                    }
                    NpuFullKMode::W8 => {
                        pack_w8_slab(base[group], self.n, slab, base_out);
                    }
                }
                if self.mode == NpuFullKMode::Mixed {
                    let residual_out =
                        &mut packed[(entry + 1) * entry_bytes..(entry + 2) * entry_bytes];
                    pack_w8_slab(residual[group], self.n, slab, residual_out);
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
        if weights.buffer.len() != self.packed_weight_bytes() {
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
