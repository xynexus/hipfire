//! Activation-once full-K AIE2P projection primitive.
#![cfg(target_os = "linux")]

use std::path::Path;

use crate::opus_hfp::{self, OpusHfpDescriptor, OpusHfpEncoding, OpusHfpLayout};
use crate::{DeviceBuffer, NpuKernel, XdnaError};

const COLS: usize = 8;
const DOCUMENT_ROWS: usize = 256;
const GROUP_K: usize = 256;
const GROUPS: usize = 3;
const N_BLOCK: usize = 32;
const A_SLOT: usize = 6_144;
const A_JOIN: usize = 4 * A_SLOT;
const A_DOCUMENT_BYTES: usize = 4 * 2 * GROUPS * A_JOIN;
const W_RECORD: usize = 8_320;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StagedOutputLayout {
    CanonicalRowMajor,
    StreamBlockDocument,
}

/// Device-resident offline records for one complete staged projection.
pub struct NpuStagedFullKResidentWeights {
    buffer: DeviceBuffer,
}

/// R121's activation-once full-K W8 schedule.
pub struct NpuGemmStagedFullK {
    kernel: NpuKernel,
    rows: usize,
    batch: usize,
    n: usize,
    n_blocks: usize,
    output_layout: StagedOutputLayout,
    input: DeviceBuffer,
    output: DeviceBuffer,
    primed: bool,
}

impl NpuGemmStagedFullK {
    /// Load an R121-compatible cache and validate its complete DMA contract.
    pub fn load_cached(dir: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{dir}/shape.txt")).map_err(XdnaError::Open)?;
        let rows = manifest_usize(&manifest, "m")?;
        if rows == 0 || !rows.is_multiple_of(DOCUMENT_ROWS) {
            return Err(invalid(format!(
                "staged full-K M={rows} must be a positive multiple of {DOCUMENT_ROWS}"
            )));
        }
        let batch = rows / DOCUMENT_ROWS;
        let n = manifest_usize(&manifest, "n")?;
        let n_blocks = manifest_usize(&manifest, "n32-output-blocks")?;
        for expected in [
            "mode=w8-scaled",
            "k=768",
            "activation-groups=3",
            "activation-stage-bytes-per-core=6336",
            "activation-stage-group-stride=2112",
            "activation-stage-group-alignment=64",
            "activation-dma-passes=1",
            "nmacro-materialized-replicas=0",
            "weight-layout=offline-rdna2-hfp-records",
            "immutable-tensor-reorder=none",
        ] {
            if !manifest.lines().any(|line| line == expected) {
                return Err(invalid(format!("staged full-K cache missing {expected}")));
            }
        }
        if manifest_usize(&manifest, "activation-physical-bytes")? != batch * A_DOCUMENT_BYTES {
            return Err(invalid("staged full-K activation byte geometry mismatch"));
        }
        if n == 0 || n != n_blocks * N_BLOCK {
            return Err(invalid(format!(
                "invalid staged full-K geometry N={n} blocks={n_blocks}"
            )));
        }
        let expected_op = if batch == 1 {
            if manifest_usize(&manifest, "output-task-repeat-count")? + 1 != n_blocks
                || manifest_usize(&manifest, "output-bd-outer-dimension")? != n_blocks
            {
                return Err(invalid("invalid R121 staged full-K output schedule"));
            }
            format!("op=embeddinggemma-r113-staged-fullk-n{n}-repeat-output")
        } else {
            if manifest_usize(&manifest, "logical-output-objects-per-stream")? != batch * n_blocks
                || manifest_usize(&manifest, "output-objects-per-stream")? != n_blocks
                || manifest_usize(&manifest, "output-documents-per-object")? != batch
                || manifest_usize(&manifest, "output-task-max-objects")? != 64
                || manifest_usize(&manifest, "output-task-count-per-stream")? != 1
            {
                return Err(invalid("invalid R129 staged full-K output schedule"));
            }
            for expected in [
                format!("batch={batch}"),
                "segment-rows=256".to_string(),
                format!("activation-stage-documents={batch}"),
                "weight-dma-passes=1".to_string(),
                "weight-batch-replicas=0".to_string(),
                "output-layout=stream-nblock-core-document-row-n32".to_string(),
            ] {
                if !manifest.lines().any(|line| line == expected) {
                    return Err(invalid(format!("staged full-K cache missing {expected}")));
                }
            }
            format!("op=embeddinggemma-r129-staged-fullk-weight-reuse-n{n}")
        };
        if !manifest.lines().any(|line| line == expected_op) {
            return Err(invalid(format!(
                "staged full-K cache missing {expected_op}"
            )));
        }
        let output_layout = if batch == 1 {
            StagedOutputLayout::CanonicalRowMajor
        } else {
            StagedOutputLayout::StreamBlockDocument
        };

        let kernel = NpuKernel::load(
            &std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        let input = kernel.alloc_arg(batch * A_DOCUMENT_BYTES)?;
        let mut output = kernel.alloc_arg(rows * n * std::mem::size_of::<f32>())?;
        // Touch every output page before the first S2MM. The admitted R121
        // verifier does this as part of its zero oracle; leaving the host-only
        // BO untouched can expose a partially populated mapping on first use.
        output.as_mut_slice().fill(0);
        Ok(Self {
            kernel,
            rows,
            batch,
            n,
            n_blocks,
            output_layout,
            input,
            output,
            primed: false,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn batch(&self) -> usize {
        self.batch
    }

    pub const fn activation_bytes(&self) -> usize {
        self.batch * A_DOCUMENT_BYTES
    }

    pub fn output_bytes(&self) -> usize {
        self.rows * self.n * std::mem::size_of::<f32>()
    }

    pub const fn k(&self) -> usize {
        GROUPS * GROUP_K
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    pub fn packed_weight_bytes(&self) -> usize {
        COLS * self.n_blocks * GROUPS * W_RECORD
    }

    pub fn recreate_hwctx(&mut self) -> Result<(), XdnaError> {
        self.kernel.recreate_hwctx()?;
        self.primed = false;
        Ok(())
    }

    fn hfp_descriptor(&self, quant_type: u8) -> Result<OpusHfpDescriptor, XdnaError> {
        if quant_type != 35 {
            return Err(invalid(format!(
                "staged full-K W8 cache requires OQ8 qt=35, got {quant_type}"
            )));
        }
        Ok(OpusHfpDescriptor {
            encoding: OpusHfpEncoding::W8,
            layout: OpusHfpLayout::StagedFullKV1,
            quant_type: quant_type.into(),
            flags: 1,
            // The immutable weight layout is independent of activation batch.
            // Retain the R121 descriptor so B1/B2 share one prepacked artifact.
            m: DOCUMENT_ROWS as u32,
            k: self.k() as u32,
            n: self.n as u32,
            columns: COLS as u32,
            groups: GROUPS as u32,
            m_macros: (DOCUMENT_ROWS / 32) as u32,
            n_macros: self.n_blocks as u32,
            outblocks: (GROUPS * self.n_blocks) as u32,
            tile_bytes: W_RECORD as u32,
            data_bytes: self.packed_weight_bytes() as u32,
            scale_offset: 8_192,
            scale_values: (COLS * GROUPS * self.n) as u32,
            payload_bytes: self.packed_weight_bytes() as u64,
            segment_bytes: [0; 4],
        })
    }

    /// Build or reuse the complete immutable R121 record order. This loader
    /// conversion is the only global tensor-block reorder in the path.
    pub fn prepack_weights_cached(
        &self,
        path: &Path,
        quant_type: u8,
        source_payload: &[u8],
        groups: &[&[i8]],
        scales: &[&[f32]],
    ) -> Result<Vec<u8>, XdnaError> {
        let descriptor = self.hfp_descriptor(quant_type)?;
        let source_sha = opus_hfp::source_sha256(&[source_payload]);
        if let Some(packed) = opus_hfp::read(path, descriptor, source_sha).map_err(invalid)? {
            return Ok(packed);
        }
        let packed = self.prepack_weights(groups, scales)?;
        opus_hfp::write(path, descriptor, source_sha, &packed).map_err(invalid)?;
        Ok(packed)
    }

    pub fn prepack_weights(
        &self,
        groups: &[&[i8]],
        scales: &[&[f32]],
    ) -> Result<Vec<u8>, XdnaError> {
        pack_dense_weight_records(self.n, groups, scales)
    }

    pub fn upload_resident_weights(
        &self,
        packed: &[u8],
    ) -> Result<NpuStagedFullKResidentWeights, XdnaError> {
        if packed.len() != self.packed_weight_bytes() {
            return Err(invalid(format!(
                "staged full-K weights want {} bytes, got {}",
                self.packed_weight_bytes(),
                packed.len()
            )));
        }
        let mut buffer = self.kernel.alloc_arg(packed.len())?;
        buffer.as_mut_slice().copy_from_slice(packed);
        self.kernel.sync_to_device(&buffer)?;
        Ok(NpuStagedFullKResidentWeights { buffer })
    }

    /// Execute canonical group-major int8 activations/scales and return
    /// canonical row-major f32 output. The activation DMA remains one pass.
    pub fn run_resident_scaled(
        &mut self,
        weights: &NpuStagedFullKResidentWeights,
        activation_groups: &[&[i8]],
        activation_scales: &[&[f32]],
        output: &mut [f32],
    ) -> Result<(), XdnaError> {
        if weights.buffer.len() != self.packed_weight_bytes() || output.len() != self.rows * self.n
        {
            return Err(invalid("staged full-K weight/output geometry mismatch"));
        }
        let packed = pack_compact_activations(self.rows, activation_groups, activation_scales)?;
        self.input.as_mut_slice().copy_from_slice(&packed);
        if !self.primed {
            self.kernel.dispatch_synced(
                &[&self.input, &weights.buffer, &self.output],
                &[true, false, false],
            )?;
            self.output.as_mut_slice().fill(0);
            self.kernel.sync_to_device(&self.output)?;
            self.primed = true;
        }
        self.kernel.dispatch_synced(
            &[&self.input, &weights.buffer, &self.output],
            &[true, false, false],
        )?;
        match self.output_layout {
            StagedOutputLayout::CanonicalRowMajor => {
                output.copy_from_slice(as_f32(self.output.as_slice()));
            }
            StagedOutputLayout::StreamBlockDocument => {
                unpack_stream_block_document(self.output.as_slice(), self.batch, self.n, output)?
            }
        }
        Ok(())
    }
}

fn pack_dense_weight_records(
    n: usize,
    groups: &[&[i8]],
    scales: &[&[f32]],
) -> Result<Vec<u8>, XdnaError> {
    if n == 0
        || !n.is_multiple_of(N_BLOCK)
        || groups.len() != GROUPS
        || groups.iter().any(|group| group.len() != GROUP_K * n)
        || scales.len() != GROUPS
        || scales.iter().any(|scale| scale.len() != n)
    {
        return Err(invalid("staged full-K dense weight geometry mismatch"));
    }
    let n_blocks = n / N_BLOCK;
    let mut packed = vec![0u8; COLS * n_blocks * GROUPS * W_RECORD];
    for physical_col in 0..COLS {
        for n_block in 0..n_blocks {
            for group in 0..GROUPS {
                let base = (physical_col * n_blocks * GROUPS + n_block * GROUPS + group) * W_RECORD;
                for slice in 0..2 {
                    for kt in 0..32 {
                        for kk in 0..8 {
                            for local_col in 0..16 {
                                let col = n_block * N_BLOCK + slice * 16 + local_col;
                                let target = base
                                    + slice * 4_096
                                    + kt * 128
                                    + (local_col / 8) * 64
                                    + kk * 8
                                    + local_col % 8;
                                packed[target] = groups[group][(kt * 8 + kk) * n + col] as u8;
                            }
                        }
                    }
                }
                let scale_start = base + 8_192;
                packed[scale_start..scale_start + N_BLOCK * std::mem::size_of::<f32>()]
                    .copy_from_slice(as_bytes_f32(
                        &scales[group][n_block * N_BLOCK..(n_block + 1) * N_BLOCK],
                    ));
            }
        }
    }
    Ok(packed)
}

fn pack_compact_activations(
    rows: usize,
    groups: &[&[i8]],
    scales: &[&[f32]],
) -> Result<Vec<u8>, XdnaError> {
    if rows == 0
        || !rows.is_multiple_of(DOCUMENT_ROWS)
        || groups.len() != GROUPS
        || groups.iter().any(|group| group.len() != rows * GROUP_K)
        || scales.len() != GROUPS
        || scales.iter().any(|scale| scale.len() != rows)
    {
        return Err(invalid(
            "staged full-K compact activation geometry mismatch",
        ));
    }
    let mut packed = vec![0u8; rows / DOCUMENT_ROWS * A_DOCUMENT_BYTES];
    for token in 0..rows {
        let document = token / DOCUMENT_ROWS;
        let document_token = token % DOCUMENT_ROWS;
        let half = document_token / 128;
        let within_half = document_token % 128;
        let core_row = within_half / 32;
        let within_row = within_half % 32;
        let local_col = within_row / 8;
        let local_row = within_row % 8;
        for group in 0..GROUPS {
            let record = (core_row * 2 + half) * GROUPS + group;
            let base = document * A_DOCUMENT_BYTES + record * A_JOIN + local_col * A_SLOT;
            for inner in 0..GROUP_K {
                let target = base + (inner / 8) * 64 + local_row * 8 + inner % 8;
                packed[target] = groups[group][token * GROUP_K + inner] as u8;
            }
            packed[base + 2_048 + local_row * 4..base + 2_052 + local_row * 4]
                .copy_from_slice(&scales[group][token].to_le_bytes());
        }
    }
    Ok(packed)
}

fn unpack_stream_block_document(
    physical: &[u8],
    batch: usize,
    n: usize,
    canonical: &mut [f32],
) -> Result<(), XdnaError> {
    if batch < 2
        || n == 0
        || !n.is_multiple_of(N_BLOCK)
        || physical.len() != batch * DOCUMENT_ROWS * n * std::mem::size_of::<f32>()
        || canonical.len() != batch * DOCUMENT_ROWS * n
    {
        return Err(invalid("staged full-K physical output geometry mismatch"));
    }
    let physical = as_f32(physical);
    let n_blocks = n / N_BLOCK;
    for stream in 0..8 {
        let core_row = stream / 2;
        let half = stream % 2;
        let token_base = half * 128 + core_row * 32;
        for n_block in 0..n_blocks {
            for document in 0..batch {
                for local_col in 0..4 {
                    for local_row in 0..8 {
                        let row = document * DOCUMENT_ROWS + token_base + local_col * 8 + local_row;
                        let source = (((stream * n_blocks + n_block) * 4 + local_col) * batch
                            + document)
                            * 8
                            + local_row;
                        let source = source * N_BLOCK;
                        let destination = row * n + n_block * N_BLOCK;
                        canonical[destination..destination + N_BLOCK]
                            .copy_from_slice(&physical[source..source + N_BLOCK]);
                    }
                }
            }
        }
    }
    Ok(())
}

fn manifest_usize(manifest: &str, key: &str) -> Result<usize, XdnaError> {
    manifest
        .lines()
        .find_map(|line| {
            line.strip_prefix(key)
                .and_then(|value| value.strip_prefix('='))
        })
        .ok_or_else(|| invalid(format!("staged full-K cache missing {key}")))?
        .parse()
        .map_err(|_| invalid(format!("staged full-K cache has invalid {key}")))
}

fn as_bytes_f32(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
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

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_weight_records_match_r121_order_and_column_replication() {
        const N: usize = 32;
        let groups = (0..3)
            .map(|group| {
                (0..256 * N)
                    .map(|index| ((group * 11 + index * 7) % 251) as u8 as i8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scales = (0..3)
            .map(|group| {
                (0..N)
                    .map(|col| group as f32 + col as f32 / 100.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let group_refs = groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let scale_refs = scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let packed = pack_dense_weight_records(N, &group_refs, &scale_refs).unwrap();

        assert_eq!(packed.len(), 8 * 3 * 8_320);
        for kt in 0..32 {
            for kk in 0..8 {
                for local_col in 0..16 {
                    let target = kt * 128 + (local_col / 8) * 64 + kk * 8 + local_col % 8;
                    let source = (kt * 8 + kk) * N + local_col;
                    assert_eq!(packed[target], groups[0][source] as u8);
                }
            }
        }
        assert_eq!(
            &packed[8_192..8_320],
            as_bytes_f32(&scales[0]),
            "scale tail must remain adjacent to its N32 weight record"
        );
        assert_eq!(
            &packed[..3 * 8_320],
            &packed[3 * 8_320..6 * 8_320],
            "every physical column receives the same offline records"
        );
    }

    #[test]
    fn compact_activation_records_match_r113_token_owners() {
        let groups = (0..3)
            .map(|group| {
                (0..256 * 256)
                    .map(|index| ((group * 13 + index) % 127) as i8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scales = (0..3)
            .map(|group| {
                (0..256)
                    .map(|row| group as f32 + row as f32 / 1000.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let group_refs = groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let scale_refs = scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let packed = pack_compact_activations(DOCUMENT_ROWS, &group_refs, &scale_refs).unwrap();

        assert_eq!(packed.len(), 589_824);
        let token = 173usize;
        let half = token / 128;
        let within_half = token % 128;
        let core_row = within_half / 32;
        let within_row = within_half % 32;
        let local_col = within_row / 8;
        let local_row = within_row % 8;
        let group = 2usize;
        let record = (core_row * 2 + half) * 3 + group;
        let base = record * 24_576 + local_col * 6_144;
        for inner in 0..256 {
            let offset = base + (inner / 8) * 64 + local_row * 8 + inner % 8;
            assert_eq!(packed[offset], groups[group][token * 256 + inner] as u8);
        }
        assert_eq!(
            &packed[base + 2_048 + local_row * 4..base + 2_052 + local_row * 4],
            &scales[group][token].to_le_bytes()
        );
    }

    #[test]
    fn compact_activation_records_keep_batched_documents_disjoint() {
        const ROWS: usize = 2 * DOCUMENT_ROWS;
        let groups = (0..GROUPS)
            .map(|group| {
                (0..ROWS * GROUP_K)
                    .map(|index| ((group * 17 + index * 3) % 127) as i8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scales = (0..GROUPS)
            .map(|group| {
                (0..ROWS)
                    .map(|row| group as f32 + row as f32 / 1000.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let group_refs = groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let scale_refs = scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let packed = pack_compact_activations(ROWS, &group_refs, &scale_refs).unwrap();

        assert_eq!(packed.len(), 2 * A_DOCUMENT_BYTES);
        let token = DOCUMENT_ROWS + 173;
        let local_token = token - DOCUMENT_ROWS;
        let half = local_token / 128;
        let within_half = local_token % 128;
        let core_row = within_half / 32;
        let within_row = within_half % 32;
        let local_col = within_row / 8;
        let local_row = within_row % 8;
        let group = 2;
        let record = (core_row * 2 + half) * GROUPS + group;
        let base = A_DOCUMENT_BYTES + record * A_JOIN + local_col * A_SLOT;
        for inner in 0..GROUP_K {
            let offset = base + (inner / 8) * 64 + local_row * 8 + inner % 8;
            assert_eq!(packed[offset], groups[group][token * GROUP_K + inner] as u8);
        }
        assert_eq!(
            &packed[base + 2_048 + local_row * 4..base + 2_052 + local_row * 4],
            &scales[group][token].to_le_bytes()
        );
    }

    #[test]
    fn block_major_batch_output_unpacks_to_independent_rows() {
        const BATCH: usize = 2;
        const N: usize = 64;
        let mut physical = vec![0.0f32; BATCH * DOCUMENT_ROWS * N];
        for stream in 0..8 {
            for n_block in 0..N / N_BLOCK {
                for document in 0..BATCH {
                    for local_col in 0..4 {
                        for local_row in 0..8 {
                            for n_col in 0..N_BLOCK {
                                let source = (((stream * (N / N_BLOCK) + n_block) * 4 + local_col)
                                    * BATCH
                                    + document)
                                    * 8
                                    + local_row;
                                physical[source * N_BLOCK + n_col] = (document * 1_000_000
                                    + stream * 100_000
                                    + n_block * 10_000
                                    + local_col * 1_000
                                    + local_row * 100
                                    + n_col)
                                    as f32;
                            }
                        }
                    }
                }
            }
        }
        let mut canonical = vec![0.0f32; BATCH * DOCUMENT_ROWS * N];
        unpack_stream_block_document(as_bytes_f32(&physical), BATCH, N, &mut canonical).unwrap();

        for document in 0..BATCH {
            for row in 0..DOCUMENT_ROWS {
                let half = row / 128;
                let within_half = row % 128;
                let core_row = within_half / 32;
                let local_token = within_half % 32;
                let local_col = local_token / 8;
                let local_row = local_token % 8;
                let stream = core_row * 2 + half;
                for col in 0..N {
                    let n_block = col / N_BLOCK;
                    let n_col = col % N_BLOCK;
                    let expected = (document * 1_000_000
                        + stream * 100_000
                        + n_block * 10_000
                        + local_col * 1_000
                        + local_row * 100
                        + n_col) as f32;
                    assert_eq!(
                        canonical[(document * DOCUMENT_ROWS + row) * N + col],
                        expected
                    );
                }
            }
        }
    }
}
