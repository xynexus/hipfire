use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"HFR34P01";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 128;
const M: u32 = 256;
const K: u32 = 768;
const N: u32 = 1280;
const COLUMNS: u32 = 4;
const TILE_BYTES: u32 = 16_384;
const BLOCKS_PER_COLUMN: u32 = 125;
pub(crate) const R34_PAYLOAD_BYTES: usize = 8_192_000;

pub(crate) fn sha256_parts(parts: &[&[u8]]) -> [u8; 32] {
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

pub(crate) fn read(
    path: &Path,
    expected_source_sha256: [u8; 32],
) -> Result<Option<Vec<u8>>, String> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("read {}: {error}", path.display())),
    };
    if bytes.len() != HEADER_BYTES + R34_PAYLOAD_BYTES {
        return Err(format!(
            "{} has {} bytes, expected {}",
            path.display(),
            bytes.len(),
            HEADER_BYTES + R34_PAYLOAD_BYTES
        ));
    }
    if &bytes[0..8] != MAGIC {
        return Err(format!(
            "{} has invalid R34 prepacked magic",
            path.display()
        ));
    }
    for (offset, expected, label) in [
        (8, VERSION, "version"),
        (12, HEADER_BYTES as u32, "header bytes"),
        (16, M, "M"),
        (20, K, "K"),
        (24, N, "N"),
        (28, COLUMNS, "columns"),
        (32, TILE_BYTES, "tile bytes"),
        (36, BLOCKS_PER_COLUMN, "blocks per column"),
    ] {
        let actual = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if actual != expected {
            return Err(format!(
                "{} has R34 {label} {actual}, expected {expected}",
                path.display()
            ));
        }
    }
    let payload_len = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
    if payload_len != R34_PAYLOAD_BYTES as u64 {
        return Err(format!(
            "{} has payload length {payload_len}, expected {R34_PAYLOAD_BYTES}",
            path.display()
        ));
    }
    if bytes[48..80] != expected_source_sha256 {
        return Err(format!(
            "{} source SHA-256 does not match the current tensors",
            path.display()
        ));
    }
    let payload = &bytes[HEADER_BYTES..];
    if bytes[80..112] != payload_sha256(payload) {
        return Err(format!("{} payload SHA-256 mismatch", path.display()));
    }
    Ok(Some(payload.to_vec()))
}

pub(crate) fn write(path: &Path, source_sha256: [u8; 32], payload: &[u8]) -> Result<(), String> {
    if !path.to_string_lossy().ends_with(".rdna2.hfp") {
        return Err(format!(
            "R34 prepacked path must end in .rdna2.hfp: {}",
            path.display()
        ));
    }
    if payload.len() != R34_PAYLOAD_BYTES {
        return Err(format!(
            "R34 prepacked payload has {} bytes, expected {R34_PAYLOAD_BYTES}",
            payload.len()
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
        (16, M),
        (20, K),
        (24, N),
        (28, COLUMNS),
        (32, TILE_BYTES),
        (36, BLOCKS_PER_COLUMN),
    ] {
        header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    header[40..48].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    header[48..80].copy_from_slice(&source_sha256);
    header[80..112].copy_from_slice(&payload_sha256(payload));

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid R34 prepacked filename: {}", path.display()))?;
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
    fn round_trip_validates_source_and_payload_hashes() {
        let path = temp_path("r34-roundtrip");
        let source = sha256_parts(&[b"source-a", b"source-b"]);
        let payload = vec![0x5a; R34_PAYLOAD_BYTES];
        write(&path, source, &payload).unwrap();
        assert_eq!(read(&path, source).unwrap().unwrap(), payload);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn source_mismatch_is_rejected() {
        let path = temp_path("r34-source");
        let source = sha256_parts(&[b"source"]);
        write(&path, source, &vec![7; R34_PAYLOAD_BYTES]).unwrap();
        let error = read(&path, sha256_parts(&[b"different"])).unwrap_err();
        assert!(error.contains("source SHA-256"), "{error}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn payload_corruption_is_rejected() {
        let path = temp_path("r34-corrupt");
        let source = sha256_parts(&[b"source"]);
        write(&path, source, &vec![3; R34_PAYLOAD_BYTES]).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        std::fs::write(&path, bytes).unwrap();
        let error = read(&path, source).unwrap_err();
        assert!(error.contains("payload SHA-256"), "{error}");
        std::fs::remove_file(path).unwrap();
    }
}
