//! Versioned, checksum-validated offline weight layouts for Opus NPU kernels.
//!
//! `.rdna2.hfp` identifies a model artifact whose tensor-block ordering has
//! already been converted for the target NPU schedule. The payload retains the
//! source weight precision: W4 stays nibble-packed and is decoded/swizzled by
//! the AIE kernel. No runtime kernel is allowed to redo the global block
//! ordering represented by this container.

use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"HFOPHFP2";
const VERSION: u32 = 2;
const HEADER_BYTES: usize = 192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum OpusHfpEncoding {
    W4 = 1,
    W8 = 2,
    MixedW4WithOverlays = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum OpusHfpLayout {
    /// `NpuGemmWholeScaled` block order. W4 data occupies the first 12 KiB
    /// of every 16 KiB block and f32 scales follow at `scale_offset`.
    WholeScaledV1 = 1,
    /// `NpuGemmFullK` slab order. Mixed matrices retain a nibble-packed W4
    /// base entry followed by the dense W8 overlay consumed by the AIE kernel.
    FullKV1 = 2,
    /// One destination context's already-converted projection segments,
    /// column-major across roles, followed by one padded parameter tile.
    ResidentContextBundleV1 = 3,
    /// Whole-scaled blocks grouped by adjacent destination-core pair. Each
    /// source block remains byte-identical; only complete block records move
    /// from `(column, block)` to `(column-pair, block, lane)` order offline.
    PairedWholeScaledV1 = 4,
    /// R121 activation-once full-K order. Each physical column owns complete
    /// `(N32 block, K256 group)` W8+scale records; activations are not
    /// replicated across N blocks.
    StagedFullKV1 = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpusHfpDescriptor {
    pub encoding: OpusHfpEncoding,
    pub layout: OpusHfpLayout,
    pub quant_type: u32,
    pub flags: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub columns: u32,
    pub groups: u32,
    pub m_macros: u32,
    pub n_macros: u32,
    pub outblocks: u32,
    pub tile_bytes: u32,
    pub data_bytes: u32,
    pub scale_offset: u32,
    pub scale_values: u32,
    pub payload_bytes: u64,
    /// Logical source-segment sizes. Single-matrix layouts leave these zero.
    pub segment_bytes: [u64; 4],
}

struct InspectedHfp {
    descriptor: OpusHfpDescriptor,
    payload: Vec<u8>,
    raw: Vec<u8>,
}

pub(crate) fn source_sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
    }
    digest.finalize().into()
}

fn payload_sha256(payload: &[u8]) -> [u8; 32] {
    Sha256::digest(payload).into()
}

fn inspect(path: &Path) -> Result<InspectedHfp, String> {
    let raw = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if raw.len() < HEADER_BYTES || &raw[0..8] != MAGIC {
        return Err(format!("{} has invalid Opus HFP header", path.display()));
    }
    let u32_at = |offset| u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
    if u32_at(8) != VERSION || u32_at(12) as usize != HEADER_BYTES {
        return Err(format!(
            "{} has unsupported Opus HFP version",
            path.display()
        ));
    }
    let encoding = match u32_at(16) {
        1 => OpusHfpEncoding::W4,
        2 => OpusHfpEncoding::W8,
        3 => OpusHfpEncoding::MixedW4WithOverlays,
        value => return Err(format!("{} has unknown encoding {value}", path.display())),
    };
    let layout = match u32_at(20) {
        1 => OpusHfpLayout::WholeScaledV1,
        2 => OpusHfpLayout::FullKV1,
        3 => OpusHfpLayout::ResidentContextBundleV1,
        4 => OpusHfpLayout::PairedWholeScaledV1,
        5 => OpusHfpLayout::StagedFullKV1,
        value => return Err(format!("{} has unknown layout {value}", path.display())),
    };
    let payload_bytes = u64::from_le_bytes(raw[80..88].try_into().unwrap());
    if raw.len() != HEADER_BYTES + payload_bytes as usize {
        return Err(format!(
            "{} has inconsistent payload length",
            path.display()
        ));
    }
    let payload = raw[HEADER_BYTES..].to_vec();
    if raw[128..160] != payload_sha256(&payload) {
        return Err(format!("{} payload SHA-256 mismatch", path.display()));
    }
    let mut segment_bytes = [0u64; 4];
    for (index, value) in segment_bytes.iter_mut().enumerate() {
        *value = u64::from_le_bytes(raw[160 + index * 8..168 + index * 8].try_into().unwrap());
    }
    Ok(InspectedHfp {
        descriptor: OpusHfpDescriptor {
            encoding,
            layout,
            quant_type: u32_at(24),
            flags: u32_at(28),
            m: u32_at(32),
            k: u32_at(36),
            n: u32_at(40),
            columns: u32_at(44),
            groups: u32_at(48),
            m_macros: u32_at(52),
            n_macros: u32_at(56),
            outblocks: u32_at(60),
            tile_bytes: u32_at(64),
            data_bytes: u32_at(68),
            scale_offset: u32_at(72),
            scale_values: u32_at(76),
            payload_bytes,
            segment_bytes,
        },
        payload,
        raw,
    })
}

/// Build or reuse a pair-major derivative of one whole-scaled artifact. This
/// is an offline/loader conversion: blocks are copied intact, and inference
/// kernels perform no tensor-block reordering.
pub(crate) fn paired_whole_scaled_cached(
    path: &Path,
    source_path: &Path,
) -> Result<Vec<u8>, String> {
    let source = inspect(source_path)?;
    let descriptor = source.descriptor;
    let columns = descriptor.columns as usize;
    let tile_bytes = descriptor.tile_bytes as usize;
    let blocks_per_column = descriptor.groups as usize * descriptor.outblocks as usize;
    if descriptor.layout != OpusHfpLayout::WholeScaledV1
        || columns == 0
        || columns % 2 != 0
        || tile_bytes == 0
        || blocks_per_column == 0
        || source.payload.len() != columns * blocks_per_column * tile_bytes
    {
        return Err("paired HFP requires an even-column whole-scaled source".to_string());
    }

    let mut payload = Vec::with_capacity(source.payload.len());
    let column_bytes = blocks_per_column * tile_bytes;
    for pair in 0..columns / 2 {
        for block in 0..blocks_per_column {
            for lane in 0..2 {
                let column = 2 * pair + lane;
                let start = column * column_bytes + block * tile_bytes;
                payload.extend_from_slice(&source.payload[start..start + tile_bytes]);
            }
        }
    }
    let paired_descriptor = OpusHfpDescriptor {
        layout: OpusHfpLayout::PairedWholeScaledV1,
        flags: 2,
        columns: descriptor.columns / 2,
        outblocks: descriptor.outblocks * 2,
        payload_bytes: payload.len() as u64,
        segment_bytes: [descriptor.payload_bytes, 0, 0, 0],
        ..descriptor
    };
    let source_sha = source_sha256(&[&source.raw]);
    if let Some(cached) = read(path, paired_descriptor, source_sha)? {
        return Ok(cached);
    }
    write(path, paired_descriptor, source_sha, &payload)?;
    Ok(payload)
}

/// Read an existing offline-layout artifact after validating its version,
/// payload length, and payload SHA-256. Resident executors use this path when
/// the loader has already created the architecture-specific artifact and no
/// source tensor is needed at inference startup.
pub(crate) fn read_existing(path: &Path) -> Result<(OpusHfpDescriptor, Vec<u8>), String> {
    if !path.to_string_lossy().ends_with(".rdna2.hfp") {
        return Err(format!(
            "Opus prepacked path must end in .rdna2.hfp: {}",
            path.display()
        ));
    }
    let inspected = inspect(path)?;
    Ok((inspected.descriptor, inspected.payload))
}

/// Persist one destination context's immutable projection streams and
/// parameters without changing block order inside any source role.
pub(crate) fn resident_context_bundle_cached(
    path: &Path,
    source_paths: &[&Path],
    parameters: &[u8],
) -> Result<Vec<u8>, String> {
    if source_paths.is_empty() || source_paths.len() > 3 {
        return Err("resident HFP bundle wants one to three source matrices".to_string());
    }
    let sources = source_paths
        .iter()
        .map(|path| inspect(path))
        .collect::<Result<Vec<_>, _>>()?;
    let first = sources[0].descriptor;
    if first.layout != OpusHfpLayout::WholeScaledV1
        || first.columns == 0
        || first.tile_bytes == 0
        || parameters.len() > first.tile_bytes as usize
    {
        return Err(
            "resident HFP bundle requires whole-scaled sources and one parameter tile".to_string(),
        );
    }
    if sources.iter().any(|source| {
        source.descriptor.layout != OpusHfpLayout::WholeScaledV1
            || source.descriptor.encoding != first.encoding
            || source.descriptor.quant_type != first.quant_type
            || source.descriptor.m != first.m
            || source.descriptor.columns != first.columns
            || source.descriptor.tile_bytes != first.tile_bytes
            || source.payload.len() % first.columns as usize != 0
    }) {
        return Err("resident HFP bundle source contracts differ".to_string());
    }

    let tile_bytes = first.tile_bytes as usize;
    let columns = first.columns as usize;
    let mut parameter_tile = vec![0u8; tile_bytes];
    parameter_tile[..parameters.len()].copy_from_slice(parameters);
    let column_bytes = sources
        .iter()
        .map(|source| source.payload.len() / columns)
        .sum::<usize>()
        + tile_bytes;
    let mut payload = Vec::with_capacity(columns * column_bytes);
    for column in 0..columns {
        for source in &sources {
            let bytes = source.payload.len() / columns;
            payload.extend_from_slice(&source.payload[column * bytes..(column + 1) * bytes]);
        }
        payload.extend_from_slice(&parameter_tile);
    }

    let mut segment_bytes = [0u64; 4];
    for (index, source) in sources.iter().enumerate() {
        segment_bytes[index] = source.payload.len() as u64;
    }
    segment_bytes[sources.len()] = parameters.len() as u64;
    let descriptor = OpusHfpDescriptor {
        encoding: first.encoding,
        layout: OpusHfpLayout::ResidentContextBundleV1,
        quant_type: first.quant_type,
        flags: (sources.len() + 1) as u32,
        m: first.m,
        k: 0,
        n: 0,
        columns: first.columns,
        groups: 0,
        m_macros: 0,
        n_macros: 0,
        outblocks: sources
            .iter()
            .map(|source| source.descriptor.groups * source.descriptor.outblocks)
            .sum::<u32>()
            + 1,
        tile_bytes: first.tile_bytes,
        data_bytes: 0,
        scale_offset: 0,
        scale_values: parameters.len() as u32,
        payload_bytes: payload.len() as u64,
        segment_bytes,
    };
    let mut source_parts = sources
        .iter()
        .map(|source| source.raw.as_slice())
        .collect::<Vec<_>>();
    source_parts.push(parameters);
    let source_sha = source_sha256(&source_parts);
    if let Some(cached) = read(path, descriptor, source_sha)? {
        return Ok(cached);
    }
    write(path, descriptor, source_sha, &payload)?;
    Ok(payload)
}

pub(crate) fn read(
    path: &Path,
    expected: OpusHfpDescriptor,
    expected_source_sha256: [u8; 32],
) -> Result<Option<Vec<u8>>, String> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("read {}: {error}", path.display())),
    };
    let expected_len = HEADER_BYTES
        .checked_add(expected.payload_bytes as usize)
        .ok_or_else(|| format!("{} payload length overflows usize", path.display()))?;
    if bytes.len() != expected_len {
        return Err(format!(
            "{} has {} bytes, expected {expected_len}",
            path.display(),
            bytes.len()
        ));
    }
    if &bytes[0..8] != MAGIC {
        return Err(format!("{} has invalid Opus HFP magic", path.display()));
    }
    let fields = [
        (8, VERSION, "version"),
        (12, HEADER_BYTES as u32, "header bytes"),
        (16, expected.encoding as u32, "encoding"),
        (20, expected.layout as u32, "layout"),
        (24, expected.quant_type, "quant type"),
        (28, expected.flags, "flags"),
        (32, expected.m, "M"),
        (36, expected.k, "K"),
        (40, expected.n, "N"),
        (44, expected.columns, "columns"),
        (48, expected.groups, "groups"),
        (52, expected.m_macros, "M macros"),
        (56, expected.n_macros, "N macros"),
        (60, expected.outblocks, "outblocks"),
        (64, expected.tile_bytes, "tile bytes"),
        (68, expected.data_bytes, "data bytes"),
        (72, expected.scale_offset, "scale offset"),
        (76, expected.scale_values, "scale values"),
    ];
    for (offset, wanted, label) in fields {
        let actual = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if actual != wanted {
            return Err(format!(
                "{} has Opus HFP {label} {actual}, expected {wanted}",
                path.display()
            ));
        }
    }
    let payload_len = u64::from_le_bytes(bytes[80..88].try_into().unwrap());
    if payload_len != expected.payload_bytes {
        return Err(format!(
            "{} has payload length {payload_len}, expected {}",
            path.display(),
            expected.payload_bytes
        ));
    }
    if bytes[96..128] != expected_source_sha256 {
        return Err(format!(
            "{} source SHA-256 does not match the current tensor",
            path.display()
        ));
    }
    let payload = &bytes[HEADER_BYTES..];
    if bytes[128..160] != payload_sha256(payload) {
        return Err(format!("{} payload SHA-256 mismatch", path.display()));
    }
    for (index, wanted) in expected.segment_bytes.iter().enumerate() {
        let actual =
            u64::from_le_bytes(bytes[160 + index * 8..168 + index * 8].try_into().unwrap());
        if actual != *wanted {
            return Err(format!(
                "{} has Opus HFP segment {index} bytes {actual}, expected {wanted}",
                path.display()
            ));
        }
    }
    Ok(Some(payload.to_vec()))
}

pub(crate) fn write(
    path: &Path,
    descriptor: OpusHfpDescriptor,
    source_sha256: [u8; 32],
    payload: &[u8],
) -> Result<(), String> {
    if !path.to_string_lossy().ends_with(".rdna2.hfp") {
        return Err(format!(
            "Opus prepacked path must end in .rdna2.hfp: {}",
            path.display()
        ));
    }
    if payload.len() as u64 != descriptor.payload_bytes {
        return Err(format!(
            "Opus prepacked payload has {} bytes, expected {}",
            payload.len(),
            descriptor.payload_bytes
        ));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }

    let mut header = [0u8; HEADER_BYTES];
    header[0..8].copy_from_slice(MAGIC);
    for (offset, value) in [
        (8, VERSION),
        (12, HEADER_BYTES as u32),
        (16, descriptor.encoding as u32),
        (20, descriptor.layout as u32),
        (24, descriptor.quant_type),
        (28, descriptor.flags),
        (32, descriptor.m),
        (36, descriptor.k),
        (40, descriptor.n),
        (44, descriptor.columns),
        (48, descriptor.groups),
        (52, descriptor.m_macros),
        (56, descriptor.n_macros),
        (60, descriptor.outblocks),
        (64, descriptor.tile_bytes),
        (68, descriptor.data_bytes),
        (72, descriptor.scale_offset),
        (76, descriptor.scale_values),
    ] {
        header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    header[80..88].copy_from_slice(&descriptor.payload_bytes.to_le_bytes());
    header[96..128].copy_from_slice(&source_sha256);
    header[128..160].copy_from_slice(&payload_sha256(payload));
    for (index, value) in descriptor.segment_bytes.iter().enumerate() {
        header[160 + index * 8..168 + index * 8].copy_from_slice(&value.to_le_bytes());
    }

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid Opus HFP filename: {}", path.display()))?;
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let result = (|| {
        let mut file = std::fs::File::create(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?;
        file.write_all(&header)
            .and_then(|()| file.write_all(payload))
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("write {}: {error}", temporary.display()))?;
        std::fs::rename(&temporary, path).map_err(|error| {
            format!(
                "rename {} to {}: {error}",
                temporary.display(),
                path.display()
            )
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> OpusHfpDescriptor {
        OpusHfpDescriptor {
            encoding: OpusHfpEncoding::W4,
            layout: OpusHfpLayout::WholeScaledV1,
            quant_type: 34,
            flags: 0,
            m: 256,
            k: 768,
            n: 1280,
            columns: 8,
            groups: 3,
            m_macros: 3,
            n_macros: 2,
            outblocks: 6,
            tile_bytes: 16_384,
            data_bytes: 12_288,
            scale_offset: 12_288,
            scale_values: 96,
            payload_bytes: 8 * 6 * 3 * 16_384,
            segment_bytes: [0; 4],
        }
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "hipfire-{label}-{}-{}.rdna2.hfp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn round_trip_preserves_packed_nibbles_and_metadata() {
        let path = temp_path("opus-hfp-roundtrip");
        let descriptor = descriptor();
        let source = source_sha256(&[b"oq4-source"]);
        let payload = (0..descriptor.payload_bytes)
            .map(|index| (index as u8).wrapping_mul(29))
            .collect::<Vec<_>>();
        write(&path, descriptor, source, &payload).unwrap();
        assert_eq!(read(&path, descriptor, source).unwrap().unwrap(), payload);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], MAGIC);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn existing_artifact_loader_validates_and_returns_descriptor_and_payload() {
        let path = temp_path("opus-hfp-existing");
        let descriptor = descriptor();
        let payload = vec![0x4d; descriptor.payload_bytes as usize];
        write(
            &path,
            descriptor,
            source_sha256(&[b"offline-loader-source"]),
            &payload,
        )
        .unwrap();
        let (loaded_descriptor, loaded_payload) = read_existing(&path).unwrap();
        assert_eq!(loaded_descriptor, descriptor);
        assert_eq!(loaded_payload, payload);

        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0x80;
        std::fs::write(&path, bytes).unwrap();
        let error = read_existing(&path).unwrap_err();
        assert!(error.contains("payload SHA-256"), "{error}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn geometry_mismatch_is_rejected() {
        let path = temp_path("opus-hfp-geometry");
        let descriptor = descriptor();
        let source = source_sha256(&[b"oq4-source"]);
        write(
            &path,
            descriptor,
            source,
            &vec![0x87; descriptor.payload_bytes as usize],
        )
        .unwrap();
        let mut wrong = descriptor;
        wrong.n = 768;
        let error = read(&path, wrong, source).unwrap_err();
        assert!(error.contains("Opus HFP N"), "{error}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn source_and_payload_corruption_are_rejected() {
        let path = temp_path("opus-hfp-corrupt");
        let descriptor = descriptor();
        let source = source_sha256(&[b"oq4-source"]);
        write(
            &path,
            descriptor,
            source,
            &vec![0x21; descriptor.payload_bytes as usize],
        )
        .unwrap();
        let error = read(&path, descriptor, source_sha256(&[b"other"])).unwrap_err();
        assert!(error.contains("source SHA-256"), "{error}");
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        std::fs::write(&path, bytes).unwrap();
        let error = read(&path, descriptor, source).unwrap_err();
        assert!(error.contains("payload SHA-256"), "{error}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn full_k_layout_is_part_of_the_validated_descriptor() {
        let path = temp_path("opus-hfp-full-k");
        let mut descriptor = descriptor();
        descriptor.encoding = OpusHfpEncoding::MixedW4WithOverlays;
        descriptor.layout = OpusHfpLayout::FullKV1;
        descriptor.quant_type = 36;
        descriptor.flags = 1;
        descriptor.tile_bytes = 32_768;
        descriptor.data_bytes = 32_768;
        descriptor.scale_offset = 0;
        descriptor.scale_values = 0;
        descriptor.payload_bytes = 3 * 12 * 32_768;
        let source = source_sha256(&[b"mixed-source"]);
        let payload = vec![0xa5; descriptor.payload_bytes as usize];
        write(&path, descriptor, source, &payload).unwrap();
        assert_eq!(read(&path, descriptor, source).unwrap().unwrap(), payload);

        let mut wrong = descriptor;
        wrong.layout = OpusHfpLayout::WholeScaledV1;
        let error = read(&path, wrong, source).unwrap_err();
        assert!(error.contains("Opus HFP layout"), "{error}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn paired_whole_scaled_reorders_only_complete_blocks() {
        let source_path = temp_path("opus-hfp-paired-source");
        let paired_path = temp_path("opus-hfp-paired-output");
        let mut source_descriptor = descriptor();
        source_descriptor.columns = 4;
        source_descriptor.groups = 1;
        source_descriptor.outblocks = 2;
        source_descriptor.tile_bytes = 16;
        source_descriptor.data_bytes = 8;
        source_descriptor.scale_offset = 8;
        source_descriptor.scale_values = 2;
        source_descriptor.payload_bytes = 4 * 2 * 16;
        let mut source = Vec::new();
        for column in 0..4u8 {
            for block in 0..2u8 {
                source.extend(std::iter::repeat_n(column * 16 + block, 16));
            }
        }
        write(
            &source_path,
            source_descriptor,
            source_sha256(&[b"paired-source"]),
            &source,
        )
        .unwrap();

        let paired = paired_whole_scaled_cached(&paired_path, &source_path).unwrap();
        let expected_tags = [0u8, 16, 1, 17, 32, 48, 33, 49];
        let actual_tags = paired
            .chunks_exact(16)
            .map(|block| block[0])
            .collect::<Vec<_>>();
        assert_eq!(actual_tags, expected_tags);
        for block in paired.chunks_exact(16) {
            assert!(block.iter().all(|&byte| byte == block[0]));
        }
        let (paired_descriptor, cached) = read_existing(&paired_path).unwrap();
        assert_eq!(paired_descriptor.layout, OpusHfpLayout::PairedWholeScaledV1);
        assert_eq!(paired_descriptor.columns, 2);
        assert_eq!(paired_descriptor.outblocks, 4);
        assert_eq!(paired_descriptor.segment_bytes, [128, 0, 0, 0]);
        assert_eq!(cached, paired);
        assert_eq!(
            paired_whole_scaled_cached(&paired_path, &source_path).unwrap(),
            paired
        );
        std::fs::remove_file(source_path).unwrap();
        std::fs::remove_file(paired_path).unwrap();
    }

    #[test]
    fn resident_context_bundle_preserves_each_roles_block_order() {
        let first_path = temp_path("opus-hfp-bundle-first");
        let second_path = temp_path("opus-hfp-bundle-second");
        let bundle_path = temp_path("opus-hfp-bundle-output");
        let mut first_descriptor = descriptor();
        first_descriptor.m = 256;
        first_descriptor.k = 256;
        first_descriptor.n = 32;
        first_descriptor.columns = 2;
        first_descriptor.groups = 1;
        first_descriptor.m_macros = 1;
        first_descriptor.n_macros = 1;
        first_descriptor.outblocks = 2;
        first_descriptor.tile_bytes = 16;
        first_descriptor.data_bytes = 8;
        first_descriptor.scale_offset = 8;
        first_descriptor.scale_values = 2;
        first_descriptor.payload_bytes = 64;
        let mut second_descriptor = first_descriptor;
        second_descriptor.n = 16;
        second_descriptor.outblocks = 1;
        second_descriptor.payload_bytes = 32;
        let first = (0..64).map(|value| value as u8).collect::<Vec<_>>();
        let second = (100..132).map(|value| value as u8).collect::<Vec<_>>();
        write(
            &first_path,
            first_descriptor,
            source_sha256(&[b"first"]),
            &first,
        )
        .unwrap();
        write(
            &second_path,
            second_descriptor,
            source_sha256(&[b"second"]),
            &second,
        )
        .unwrap();

        let parameters = [9u8, 8, 7];
        let payload = resident_context_bundle_cached(
            &bundle_path,
            &[first_path.as_path(), second_path.as_path()],
            &parameters,
        )
        .unwrap();
        let mut expected = Vec::new();
        for column in 0..2 {
            expected.extend_from_slice(&first[column * 32..(column + 1) * 32]);
            expected.extend_from_slice(&second[column * 16..(column + 1) * 16]);
            expected.extend_from_slice(&parameters);
            expected.extend_from_slice(&[0u8; 13]);
        }
        assert_eq!(payload, expected);
        let inspected = inspect(&bundle_path).unwrap();
        assert_eq!(
            inspected.descriptor.layout,
            OpusHfpLayout::ResidentContextBundleV1
        );
        assert_eq!(inspected.descriptor.flags, 3);
        assert_eq!(inspected.descriptor.outblocks, 4);
        assert_eq!(inspected.descriptor.segment_bytes, [64, 32, 3, 0]);
        assert_eq!(
            resident_context_bundle_cached(
                &bundle_path,
                &[first_path.as_path(), second_path.as_path()],
                &parameters,
            )
            .unwrap(),
            expected
        );
        for path in [first_path, second_path, bundle_path] {
            std::fs::remove_file(path).unwrap();
        }
    }
}
