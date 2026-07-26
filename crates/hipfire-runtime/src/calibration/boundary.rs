// SPDX-License-Identifier: Apache-2.0
// hipfire — deterministic activation-boundary storage for layer streaming.

use super::contracts::CalibError;
use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const BOUNDARY_CHECKPOINT_SCHEMA_VERSION: u32 = 2;
const MANIFEST_FILE: &str = "calibration-boundary.json";
const BUFFER_FILES: [&str; 2] = ["boundary-a.f32", "boundary-b.f32"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundaryBackend {
    Ram,
    Mmap { directory: PathBuf },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryCheckpoint {
    pub schema_version: u32,
    pub rows: usize,
    pub width: usize,
    pub active_buffer: usize,
    /// Number of sequential transformer layers durably committed.
    pub completed_layers: usize,
    pub total_layers: usize,
    pub sample_fingerprint: String,
    /// Exact producer/execution identity required to resume an incomplete job.
    pub execution_fingerprint: String,
    pub kld_finalized: bool,
    pub artifact_complete: bool,
    pub buffer_files: [String; 2],
}

impl BoundaryCheckpoint {
    fn new(
        rows: usize,
        width: usize,
        total_layers: usize,
        sample_fingerprint: String,
        execution_fingerprint: String,
    ) -> Self {
        Self {
            schema_version: BOUNDARY_CHECKPOINT_SCHEMA_VERSION,
            rows,
            width,
            active_buffer: 0,
            completed_layers: 0,
            total_layers,
            sample_fingerprint,
            execution_fingerprint,
            kld_finalized: false,
            artifact_complete: false,
            buffer_files: BUFFER_FILES.map(str::to_string),
        }
    }
}

enum BoundaryBuffer {
    Ram(Vec<u8>),
    Mmap(MmapMut),
}

impl BoundaryBuffer {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Ram(bytes) => bytes,
            Self::Mmap(bytes) => bytes,
        }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Ram(bytes) => bytes,
            Self::Mmap(bytes) => bytes,
        }
    }

    fn flush(&mut self) -> Result<(), CalibError> {
        if let Self::Mmap(mmap) = self {
            mmap.flush()
                .map_err(|error| CalibError::Boundary(error.to_string()))?;
        }
        Ok(())
    }
}

/// Two exact-size F32 residual boundaries. Layer N reads `active` and writes
/// `next`; only a successful checkpoint flush swaps them.
pub struct BoundaryStore {
    checkpoint: BoundaryCheckpoint,
    buffers: [BoundaryBuffer; 2],
    manifest_path: Option<PathBuf>,
    bytes_per_buffer: usize,
}

impl BoundaryStore {
    /// Return whether an mmap boundary checkpoint has been committed in
    /// `directory`. Buffer files without a manifest are deliberately not a
    /// resumable checkpoint: [`Self::create`] will reject those stale files
    /// instead of guessing whether they are complete.
    pub fn mmap_checkpoint_exists(directory: &Path) -> Result<bool, CalibError> {
        directory
            .join(MANIFEST_FILE)
            .try_exists()
            .map_err(|error| CalibError::Checkpoint(error.to_string()))
    }

    pub fn create(
        backend: BoundaryBackend,
        rows: usize,
        width: usize,
        total_layers: usize,
        sample_fingerprint: impl Into<String>,
        execution_fingerprint: impl Into<String>,
    ) -> Result<Self, CalibError> {
        validate_geometry(rows, width, total_layers)?;
        let bytes_per_buffer = boundary_bytes(rows, width)?;
        let execution_fingerprint = execution_fingerprint.into();
        if execution_fingerprint.is_empty() {
            return Err(CalibError::Checkpoint(
                "boundary execution fingerprint must not be empty".into(),
            ));
        }
        let checkpoint = BoundaryCheckpoint::new(
            rows,
            width,
            total_layers,
            sample_fingerprint.into(),
            execution_fingerprint,
        );
        match backend {
            BoundaryBackend::Ram => Ok(Self {
                checkpoint,
                buffers: [
                    BoundaryBuffer::Ram(vec![0; bytes_per_buffer]),
                    BoundaryBuffer::Ram(vec![0; bytes_per_buffer]),
                ],
                manifest_path: None,
                bytes_per_buffer,
            }),
            BoundaryBackend::Mmap { directory } => {
                fs::create_dir_all(&directory)
                    .map_err(|error| CalibError::Boundary(error.to_string()))?;
                let manifest_path = directory.join(MANIFEST_FILE);
                if manifest_path.exists() {
                    return Err(CalibError::Boundary(format!(
                        "checkpoint already exists at {}",
                        manifest_path.display()
                    )));
                }
                let buffers = [
                    create_mmap(&directory.join(BUFFER_FILES[0]), bytes_per_buffer)?,
                    create_mmap(&directory.join(BUFFER_FILES[1]), bytes_per_buffer)?,
                ];
                let store = Self {
                    checkpoint,
                    buffers,
                    manifest_path: Some(manifest_path),
                    bytes_per_buffer,
                };
                store.persist_manifest()?;
                Ok(store)
            }
        }
    }

    pub fn resume_mmap(
        directory: &Path,
        expected_sample_fingerprint: &str,
        expected_execution_fingerprint: &str,
    ) -> Result<Self, CalibError> {
        let manifest_path = directory.join(MANIFEST_FILE);
        let bytes =
            fs::read(&manifest_path).map_err(|error| CalibError::Checkpoint(error.to_string()))?;
        let checkpoint: BoundaryCheckpoint = serde_json::from_slice(&bytes)
            .map_err(|error| CalibError::Checkpoint(error.to_string()))?;
        validate_checkpoint(
            &checkpoint,
            expected_sample_fingerprint,
            expected_execution_fingerprint,
        )?;
        let bytes_per_buffer = boundary_bytes(checkpoint.rows, checkpoint.width)?;
        let buffers = [
            open_mmap(
                &directory.join(&checkpoint.buffer_files[0]),
                bytes_per_buffer,
            )?,
            open_mmap(
                &directory.join(&checkpoint.buffer_files[1]),
                bytes_per_buffer,
            )?,
        ];
        Ok(Self {
            checkpoint,
            buffers,
            manifest_path: Some(manifest_path),
            bytes_per_buffer,
        })
    }

    /// Resume an existing mmap checkpoint, or create a fresh one when no
    /// manifest exists. The boolean reports whether durable state was resumed.
    pub fn resume_or_create_mmap(
        directory: &Path,
        rows: usize,
        width: usize,
        total_layers: usize,
        sample_fingerprint: impl Into<String>,
        execution_fingerprint: impl Into<String>,
    ) -> Result<(Self, bool), CalibError> {
        let sample_fingerprint = sample_fingerprint.into();
        let execution_fingerprint = execution_fingerprint.into();
        if Self::mmap_checkpoint_exists(directory)? {
            return Self::resume_mmap(directory, &sample_fingerprint, &execution_fingerprint)
                .map(|store| (store, true));
        }
        Self::create(
            BoundaryBackend::Mmap {
                directory: directory.to_path_buf(),
            },
            rows,
            width,
            total_layers,
            sample_fingerprint,
            execution_fingerprint,
        )
        .map(|store| (store, false))
    }

    pub fn checkpoint(&self) -> &BoundaryCheckpoint {
        &self.checkpoint
    }

    pub const fn bytes_per_buffer(&self) -> usize {
        self.bytes_per_buffer
    }

    pub const fn allocated_bytes(&self) -> usize {
        self.bytes_per_buffer * 2
    }

    pub fn read_active_rows(
        &self,
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>, CalibError> {
        read_rows(
            &self.buffers[self.checkpoint.active_buffer],
            self.checkpoint.rows,
            self.checkpoint.width,
            start_row,
            row_count,
        )
    }

    pub fn write_active_rows(
        &mut self,
        start_row: usize,
        values: &[f32],
    ) -> Result<(), CalibError> {
        write_rows(
            &mut self.buffers[self.checkpoint.active_buffer],
            self.checkpoint.rows,
            self.checkpoint.width,
            start_row,
            values,
        )
    }

    pub fn write_next_rows(&mut self, start_row: usize, values: &[f32]) -> Result<(), CalibError> {
        let next = 1 - self.checkpoint.active_buffer;
        write_rows(
            &mut self.buffers[next],
            self.checkpoint.rows,
            self.checkpoint.width,
            start_row,
            values,
        )
    }

    pub fn read_active_indexed(&self, row_indices: &[usize]) -> Result<Vec<f32>, CalibError> {
        let mut values = Vec::with_capacity(row_indices.len() * self.checkpoint.width);
        for &row in row_indices {
            values.extend(read_rows(
                &self.buffers[self.checkpoint.active_buffer],
                self.checkpoint.rows,
                self.checkpoint.width,
                row,
                1,
            )?);
        }
        Ok(values)
    }

    pub fn write_active_indexed(
        &mut self,
        row_indices: &[usize],
        values: &[f32],
    ) -> Result<(), CalibError> {
        write_indexed(
            &mut self.buffers[self.checkpoint.active_buffer],
            self.checkpoint.rows,
            self.checkpoint.width,
            row_indices,
            values,
        )
    }

    pub fn write_next_indexed(
        &mut self,
        row_indices: &[usize],
        values: &[f32],
    ) -> Result<(), CalibError> {
        let next = 1 - self.checkpoint.active_buffer;
        write_indexed(
            &mut self.buffers[next],
            self.checkpoint.rows,
            self.checkpoint.width,
            row_indices,
            values,
        )
    }

    /// Durably commit the next sequential layer and swap boundaries.
    pub fn commit_layer(&mut self, layer: usize) -> Result<(), CalibError> {
        if layer != self.checkpoint.completed_layers {
            return Err(CalibError::Checkpoint(format!(
                "cannot commit layer {layer}; next required layer is {}",
                self.checkpoint.completed_layers
            )));
        }
        if layer >= self.checkpoint.total_layers {
            return Err(CalibError::Checkpoint(format!(
                "layer {layer} exceeds total layer count {}",
                self.checkpoint.total_layers
            )));
        }
        let next = 1 - self.checkpoint.active_buffer;
        self.buffers[next].flush()?;
        self.checkpoint.active_buffer = next;
        self.checkpoint.completed_layers += 1;
        self.persist_manifest()
    }

    /// Mark KLD packing complete without claiming that the final artifact has
    /// been assembled. The output package is committed separately.
    pub fn finalize_kld(&mut self) -> Result<(), CalibError> {
        if self.checkpoint.completed_layers != self.checkpoint.total_layers {
            return Err(CalibError::Checkpoint(format!(
                "cannot finalize KLD after {} of {} layers",
                self.checkpoint.completed_layers, self.checkpoint.total_layers
            )));
        }
        self.buffers[self.checkpoint.active_buffer].flush()?;
        self.checkpoint.kld_finalized = true;
        self.persist_manifest()
    }

    pub fn finalize_artifact(&mut self) -> Result<(), CalibError> {
        if self.checkpoint.completed_layers != self.checkpoint.total_layers
            || !self.checkpoint.kld_finalized
        {
            return Err(CalibError::Checkpoint(
                "cannot finalize the artifact before all layers and KLD packing complete".into(),
            ));
        }
        self.checkpoint.artifact_complete = true;
        self.persist_manifest()
    }

    fn persist_manifest(&self) -> Result<(), CalibError> {
        let Some(path) = &self.manifest_path else {
            return Ok(());
        };
        let bytes = serde_json::to_vec_pretty(&self.checkpoint)
            .map_err(|error| CalibError::Checkpoint(error.to_string()))?;
        let tmp_path = path.with_extension("json.tmp");
        let mut file =
            File::create(&tmp_path).map_err(|error| CalibError::Checkpoint(error.to_string()))?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| CalibError::Checkpoint(error.to_string()))?;
        fs::rename(&tmp_path, path).map_err(|error| CalibError::Checkpoint(error.to_string()))?;
        Ok(())
    }
}

fn validate_geometry(rows: usize, width: usize, total_layers: usize) -> Result<(), CalibError> {
    if rows == 0 || width == 0 {
        return Err(CalibError::Boundary(
            "row count and hidden width must be nonzero".into(),
        ));
    }
    if total_layers == 0 {
        return Err(CalibError::Boundary(
            "total layer count must be nonzero".into(),
        ));
    }
    boundary_bytes(rows, width).map(|_| ())
}

fn validate_checkpoint(
    checkpoint: &BoundaryCheckpoint,
    expected_sample_fingerprint: &str,
    expected_execution_fingerprint: &str,
) -> Result<(), CalibError> {
    if checkpoint.schema_version != BOUNDARY_CHECKPOINT_SCHEMA_VERSION {
        return Err(CalibError::Checkpoint(format!(
            "unsupported schema version {}",
            checkpoint.schema_version
        )));
    }
    validate_geometry(checkpoint.rows, checkpoint.width, checkpoint.total_layers)?;
    if checkpoint.active_buffer > 1 {
        return Err(CalibError::Checkpoint(format!(
            "invalid active boundary {}",
            checkpoint.active_buffer
        )));
    }
    if checkpoint.completed_layers > checkpoint.total_layers {
        return Err(CalibError::Checkpoint(
            "completed layer count exceeds total layers".into(),
        ));
    }
    if checkpoint.sample_fingerprint != expected_sample_fingerprint {
        return Err(CalibError::Checkpoint(format!(
            "sample fingerprint {} does not match expected {expected_sample_fingerprint}",
            checkpoint.sample_fingerprint
        )));
    }
    if checkpoint.execution_fingerprint.is_empty() || expected_execution_fingerprint.is_empty() {
        return Err(CalibError::Checkpoint(
            "boundary execution fingerprint must not be empty".into(),
        ));
    }
    if checkpoint.execution_fingerprint != expected_execution_fingerprint {
        return Err(CalibError::Checkpoint(format!(
            "execution fingerprint {} does not match expected {expected_execution_fingerprint}",
            checkpoint.execution_fingerprint
        )));
    }
    if checkpoint.kld_finalized && checkpoint.completed_layers != checkpoint.total_layers {
        return Err(CalibError::Checkpoint(
            "KLD completion flag is inconsistent with layer state".into(),
        ));
    }
    if checkpoint.artifact_complete
        && (!checkpoint.kld_finalized || checkpoint.completed_layers != checkpoint.total_layers)
    {
        return Err(CalibError::Checkpoint(
            "artifact completion flag is inconsistent with layer/KLD state".into(),
        ));
    }
    Ok(())
}

fn boundary_bytes(rows: usize, width: usize) -> Result<usize, CalibError> {
    rows.checked_mul(width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| CalibError::Boundary("boundary byte size overflow".into()))
}

fn create_mmap(path: &Path, len: usize) -> Result<BoundaryBuffer, CalibError> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| CalibError::Boundary(error.to_string()))?;
    file.set_len(len as u64)
        .map_err(|error| CalibError::Boundary(error.to_string()))?;
    let mmap = unsafe { MmapOptions::new().len(len).map_mut(&file) }
        .map_err(|error| CalibError::Boundary(error.to_string()))?;
    Ok(BoundaryBuffer::Mmap(mmap))
}

fn open_mmap(path: &Path, expected_len: usize) -> Result<BoundaryBuffer, CalibError> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| CalibError::Boundary(error.to_string()))?;
    let actual_len = file
        .metadata()
        .map_err(|error| CalibError::Boundary(error.to_string()))?
        .len();
    if actual_len != expected_len as u64 {
        return Err(CalibError::Checkpoint(format!(
            "boundary {} is {actual_len} bytes; expected {expected_len}",
            path.display()
        )));
    }
    let mmap = unsafe { MmapOptions::new().len(expected_len).map_mut(&file) }
        .map_err(|error| CalibError::Boundary(error.to_string()))?;
    Ok(BoundaryBuffer::Mmap(mmap))
}

fn row_byte_range(
    total_rows: usize,
    width: usize,
    start_row: usize,
    row_count: usize,
) -> Result<std::ops::Range<usize>, CalibError> {
    let end_row = start_row
        .checked_add(row_count)
        .ok_or_else(|| CalibError::Boundary("row range overflow".into()))?;
    if end_row > total_rows {
        return Err(CalibError::Boundary(format!(
            "row range {start_row}..{end_row} exceeds {total_rows} rows"
        )));
    }
    let start = boundary_bytes(start_row, width)?;
    let end = boundary_bytes(end_row, width)?;
    Ok(start..end)
}

fn write_rows(
    buffer: &mut BoundaryBuffer,
    total_rows: usize,
    width: usize,
    start_row: usize,
    values: &[f32],
) -> Result<(), CalibError> {
    if values.len() % width != 0 {
        return Err(CalibError::Boundary(format!(
            "{} F32 values do not contain whole rows of width {width}",
            values.len()
        )));
    }
    let range = row_byte_range(total_rows, width, start_row, values.len() / width)?;
    for (dst, value) in buffer.bytes_mut()[range].chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn write_indexed(
    buffer: &mut BoundaryBuffer,
    total_rows: usize,
    width: usize,
    row_indices: &[usize],
    values: &[f32],
) -> Result<(), CalibError> {
    let expected = row_indices
        .len()
        .checked_mul(width)
        .ok_or_else(|| CalibError::Boundary("indexed write size overflow".into()))?;
    if values.len() != expected {
        return Err(CalibError::Boundary(format!(
            "indexed write has {} values; expected {expected}",
            values.len()
        )));
    }
    for (&row, row_values) in row_indices.iter().zip(values.chunks_exact(width)) {
        write_rows(buffer, total_rows, width, row, row_values)?;
    }
    Ok(())
}

fn read_rows(
    buffer: &BoundaryBuffer,
    total_rows: usize,
    width: usize,
    start_row: usize,
    row_count: usize,
) -> Result<Vec<f32>, CalibError> {
    let range = row_byte_range(total_rows, width, start_row, row_count)?;
    Ok(buffer.bytes()[range]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_checkpoint_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hipfire-calib-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn ram_boundary_swaps_exact_f32_bytes() {
        let mut store =
            BoundaryStore::create(BoundaryBackend::Ram, 3, 2, 2, "samples", "engine").unwrap();
        let initial = [1.0, -2.0, 3.5, 4.0, 5.0, 6.25];
        store.write_active_rows(0, &initial).unwrap();
        assert_eq!(store.read_active_rows(0, 3).unwrap(), initial);
        let next = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        store.write_next_rows(0, &next).unwrap();
        store.commit_layer(0).unwrap();
        assert_eq!(store.read_active_rows(0, 3).unwrap(), next);
        assert_eq!(store.allocated_bytes(), 3 * 2 * 4 * 2);
        assert!(!store.checkpoint().artifact_complete);
    }

    #[test]
    fn indexed_rows_preserve_scheduler_order_without_relaying_out_boundary() {
        let mut store =
            BoundaryStore::create(BoundaryBackend::Ram, 4, 2, 1, "samples", "engine").unwrap();
        store
            .write_active_rows(0, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();
        assert_eq!(
            store.read_active_indexed(&[3, 0, 2]).unwrap(),
            [6.0, 7.0, 0.0, 1.0, 4.0, 5.0]
        );
        store
            .write_next_indexed(&[3, 0, 2], &[16.0, 17.0, 10.0, 11.0, 14.0, 15.0])
            .unwrap();
        store.write_next_indexed(&[1], &[12.0, 13.0]).unwrap();
        store.commit_layer(0).unwrap();
        assert_eq!(
            store.read_active_rows(0, 4).unwrap(),
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        );
    }

    #[test]
    fn mmap_boundary_resumes_at_exact_committed_layer() {
        let dir = temp_checkpoint_dir("resume");
        {
            let mut store = BoundaryStore::create(
                BoundaryBackend::Mmap {
                    directory: dir.clone(),
                },
                2,
                3,
                2,
                "sample-fp",
                "engine-a",
            )
            .unwrap();
            store
                .write_next_rows(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
            store.commit_layer(0).unwrap();
        }
        {
            let mut resumed = BoundaryStore::resume_mmap(&dir, "sample-fp", "engine-a").unwrap();
            assert_eq!(resumed.checkpoint().completed_layers, 1);
            assert_eq!(
                resumed.read_active_rows(0, 2).unwrap(),
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            );
            assert!(resumed.commit_layer(0).is_err());
            resumed
                .write_next_rows(0, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
                .unwrap();
            resumed.commit_layer(1).unwrap();
            assert!(!resumed.checkpoint().artifact_complete);
            resumed.finalize_kld().unwrap();
            assert!(!resumed.checkpoint().artifact_complete);
        }
        assert!(BoundaryStore::mmap_checkpoint_exists(&dir).unwrap());
        let mut final_store = BoundaryStore::resume_mmap(&dir, "sample-fp", "engine-a").unwrap();
        assert!(final_store.checkpoint().kld_finalized);
        assert!(!final_store.checkpoint().artifact_complete);
        assert_eq!(
            final_store.read_active_rows(0, 2).unwrap(),
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        final_store.finalize_artifact().unwrap();
        assert!(final_store.checkpoint().artifact_complete);
        drop(final_store);
        fs::remove_dir_all(&dir).unwrap();
        assert!(!BoundaryStore::mmap_checkpoint_exists(&dir).unwrap());
    }

    #[test]
    fn checkpoint_rejects_wrong_samples_and_early_kld_completion() {
        let dir = temp_checkpoint_dir("reject");
        assert!(BoundaryStore::create(BoundaryBackend::Ram, 1, 1, 1, "right", "").is_err());
        let mut store = BoundaryStore::create(
            BoundaryBackend::Mmap {
                directory: dir.clone(),
            },
            1,
            1,
            1,
            "right",
            "engine-a",
        )
        .unwrap();
        assert!(store.finalize_kld().is_err());
        drop(store);
        assert!(BoundaryStore::resume_mmap(&dir, "wrong", "engine-a").is_err());
        assert!(BoundaryStore::resume_mmap(&dir, "right", "engine-b").is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn resume_or_create_starts_fresh_then_resumes() {
        let dir = temp_checkpoint_dir("resume-or-create");
        let (mut fresh, resumed) =
            BoundaryStore::resume_or_create_mmap(&dir, 1, 2, 1, "sample-fp", "engine-a").unwrap();
        assert!(!resumed);
        fresh.write_next_rows(0, &[3.0, 4.0]).unwrap();
        fresh.commit_layer(0).unwrap();
        drop(fresh);

        let (restored, resumed) =
            BoundaryStore::resume_or_create_mmap(&dir, 1, 2, 1, "sample-fp", "engine-a").unwrap();
        assert!(resumed);
        assert_eq!(restored.checkpoint().completed_layers, 1);
        assert_eq!(restored.read_active_rows(0, 1).unwrap(), [3.0, 4.0]);
        drop(restored);
        fs::remove_dir_all(&dir).unwrap();
    }
}
