// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! NPU module opt-in contracts.
//!
//! This crate intentionally does not own XDNA runtime dispatch. It owns the
//! typed policy boundary for deciding whether an NPU module has the artifacts
//! needed to be admitted by an architecture-specific caller. For live device
//! presence it consumes the `hipfire-xdna` device layer via
//! [`XdnaHardwareProbe`]; all admission/inventory policy stays pure by taking a
//! probe value rather than touching hardware directly.

use hipfire_cpu::{BackendSelection, DenseFfnBackend, DenseFfnBackendPreference, ModuleInvocation};
use hipfire_model::AcceleratorDeviceInfo;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const XDNA_SWIGLU_BACKEND: &str = "npu_xdna";
pub const NPU_ARTIFACTS_MISSING_FALLBACK: &str = "npu_artifacts_missing";
pub const XDNA_RUNTIME_MISSING_FALLBACK: &str = "xdna_runtime_missing";
pub const NPU_HARDWARE_ABSENT_FALLBACK: &str = "npu_hardware_absent";
pub const EMBEDDING_IMAGE_SCHEMA: &str = "hipfire.npu_embedding_image.v1";
pub const FULL_EMBEDDING_ENCODER_ABI: &str = "hipfire.full_embedding_encoder.v1";

#[derive(Debug, Clone, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct EmbeddingModelGeometry {
    /// Underlying HF architecture identity, for example `qwen3` or `embeddinggemma`.
    pub architecture: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
}

#[derive(Debug, Clone, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct EmbeddingImageCacheKey {
    pub npu_architecture: String,
    pub model_geometry: EmbeddingModelGeometry,
    pub quant_format: String,
    pub sequence_bucket: usize,
    pub dispatch_batch: usize,
}

impl EmbeddingImageCacheKey {
    pub fn directory_name(&self) -> Result<String, String> {
        for (label, value) in [
            ("NPU architecture", self.npu_architecture.as_str()),
            (
                "model architecture",
                self.model_geometry.architecture.as_str(),
            ),
            ("quant format", self.quant_format.as_str()),
        ] {
            if value.is_empty()
                || !value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'+'))
            {
                return Err(format!("invalid embedding image {label} {value:?}"));
            }
        }
        let g = &self.model_geometry;
        Ok(format!(
            "{}-{}-h{}-l{}-qh{}-kvh{}-d{}-i{}-{}-s{}-b{}",
            self.npu_architecture,
            g.architecture,
            g.hidden_size,
            g.num_hidden_layers,
            g.num_attention_heads,
            g.num_key_value_heads,
            g.head_dim,
            g.intermediate_size,
            self.quant_format,
            self.sequence_bucket,
            self.dispatch_batch,
        ))
    }
}

#[derive(Debug, Clone, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmbeddingImageManifest {
    pub schema: String,
    pub runtime_abi: String,
    pub key: EmbeddingImageCacheKey,
    pub xclbin: String,
    pub instructions: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResolvedEmbeddingImage {
    pub directory: PathBuf,
    pub xclbin: PathBuf,
    pub instructions: PathBuf,
}

/// Resolve one exact compiled embedding image. There is deliberately no nearest
/// geometry or filename fallback: a missing/incompatible NPU-only image must
/// fail before execution can be reported as NPU-backed.
pub fn resolve_embedding_image(
    cache_root: &Path,
    key: &EmbeddingImageCacheKey,
) -> Result<ResolvedEmbeddingImage, String> {
    let directory = cache_root.join("embedding").join(key.directory_name()?);
    let manifest_path = directory.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|error| {
        format!(
            "missing embedding NPU image manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    let manifest: EmbeddingImageManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| {
            format!(
                "invalid embedding NPU image manifest {}: {error}",
                manifest_path.display()
            )
        })?;
    if manifest.schema != EMBEDDING_IMAGE_SCHEMA {
        return Err(format!(
            "embedding NPU image {} has unsupported schema {:?}",
            directory.display(),
            manifest.schema
        ));
    }
    if manifest.runtime_abi != FULL_EMBEDDING_ENCODER_ABI {
        return Err(format!(
            "embedding NPU image {} has incompatible runtime ABI {:?}",
            directory.display(),
            manifest.runtime_abi
        ));
    }
    if &manifest.key != key {
        return Err(format!(
            "embedding NPU image {} is incompatible with the requested cache key",
            directory.display()
        ));
    }
    let file = |name: &str, role: &str| -> Result<PathBuf, String> {
        let relative = Path::new(name);
        if relative.is_absolute()
            || relative.components().count() != 1
            || relative
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(format!(
                "embedding NPU image has invalid {role} path {name:?}"
            ));
        }
        let path = directory.join(relative);
        if !path.is_file() {
            return Err(format!(
                "embedding NPU image is missing {role} {}",
                path.display()
            ));
        }
        Ok(path)
    };
    let xclbin = file(&manifest.xclbin, "xclbin")?;
    let instructions = file(&manifest.instructions, "instructions")?;
    Ok(ResolvedEmbeddingImage {
        directory,
        xclbin,
        instructions,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuModuleTarget {
    XdnaSwiglu,
}

impl NpuModuleTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::XdnaSwiglu => "xdna_swiglu",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdnaModuleArtifacts {
    pub xclbin: Option<String>,
    pub instr: Option<String>,
}

impl XdnaModuleArtifacts {
    pub fn new(xclbin: Option<String>, instr: Option<String>) -> Self {
        Self { xclbin, instr }
    }

    pub fn complete(&self) -> bool {
        self.xclbin.as_deref().is_some_and(|path| !path.is_empty())
            && self.instr.as_deref().is_some_and(|path| !path.is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdnaInventoryConfig {
    pub runtime_lib: Option<String>,
    pub swiglu_artifacts: XdnaModuleArtifacts,
}

impl XdnaInventoryConfig {
    pub fn new(runtime_lib: Option<String>, swiglu_artifacts: XdnaModuleArtifacts) -> Self {
        Self {
            runtime_lib,
            swiglu_artifacts,
        }
    }

    pub fn from_env() -> Self {
        Self {
            runtime_lib: std::env::var("HIPFIRE_XDNA1_LIB")
                .ok()
                .filter(|value| !value.trim().is_empty()),
            swiglu_artifacts: XdnaModuleArtifacts::new(
                std::env::var("HIPFIRE_QWEN35_XDNA1_XCLBIN")
                    .ok()
                    .filter(|value| !value.trim().is_empty()),
                std::env::var("HIPFIRE_QWEN35_XDNA1_INSTR")
                    .ok()
                    .filter(|value| !value.trim().is_empty()),
            ),
        }
    }

    pub fn explicitly_configured(&self) -> bool {
        self.runtime_lib
            .as_deref()
            .is_some_and(|path| !path.is_empty())
            || self
                .swiglu_artifacts
                .xclbin
                .as_deref()
                .is_some_and(|path| !path.is_empty())
            || self
                .swiglu_artifacts
                .instr
                .as_deref()
                .is_some_and(|path| !path.is_empty())
    }
}

/// Result of probing for live XDNA NPU hardware through `hipfire-xdna`.
///
/// This is plain data so inventory policy stays unit-testable without a device.
/// The real hardware access lives in [`XdnaHardwareProbe::detect`]; everything
/// downstream takes a probe value and remains pure.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XdnaHardwareProbe {
    /// Whether an XDNA NPU device responded.
    pub present: bool,
    /// Device ordinal when present.
    pub ordinal: Option<usize>,
    /// Device node path (when present) or the probe error (when absent).
    pub detail: Option<String>,
}

impl XdnaHardwareProbe {
    /// No NPU detected.
    pub fn absent() -> Self {
        Self::default()
    }

    /// NPU detected at `ordinal` (used in tests and synthetic inventories).
    pub fn present(ordinal: Option<usize>) -> Self {
        Self {
            present: true,
            ordinal,
            detail: None,
        }
    }

    /// Probe the live NPU via `hipfire-xdna`.
    ///
    /// Uses `resource_info` (which responds even when the NPU is idle, unlike
    /// the power/utilization sensors) to confirm the device is reachable. Never
    /// panics; absent or unsupported hardware yields `present == false` with the
    /// reason captured in [`XdnaHardwareProbe::detail`].
    pub fn detect() -> Self {
        match hipfire_xdna::XdnaDevice::open_default() {
            Ok(dev) => match dev.resource_info() {
                Ok(_) => Self {
                    present: true,
                    ordinal: Some(0),
                    detail: Some(dev.path().to_string()),
                },
                Err(err) => Self {
                    present: false,
                    ordinal: None,
                    detail: Some(err.to_string()),
                },
            },
            Err(err) => Self {
                present: false,
                ordinal: None,
                detail: Some(err.to_string()),
            },
        }
    }
}

/// Build the NPU accelerator inventory by merging operator config with a live
/// hardware [`XdnaHardwareProbe`].
///
/// A device entry is emitted when the hardware is detected **or** a runtime /
/// artifacts path was explicitly configured, so "configured but no NPU" stays
/// visible (with a hardware-absent reason) and "NPU present but unconfigured"
/// surfaces the missing-runtime reason. `available` requires all three:
/// hardware present, runtime configured, and artifacts complete.
pub fn xdna_inventory_devices(
    config: &XdnaInventoryConfig,
    probe: &XdnaHardwareProbe,
) -> Vec<AcceleratorDeviceInfo> {
    if !probe.present && !config.explicitly_configured() {
        return Vec::new();
    }

    let runtime_available = config
        .runtime_lib
        .as_deref()
        .is_some_and(|path| !path.trim().is_empty());
    let artifacts_available = config.swiglu_artifacts.complete();
    let available = probe.present && runtime_available && artifacts_available;
    let reason = if !probe.present {
        Some(NPU_HARDWARE_ABSENT_FALLBACK.to_string())
    } else if !runtime_available {
        Some(XDNA_RUNTIME_MISSING_FALLBACK.to_string())
    } else if !artifacts_available {
        Some(NPU_ARTIFACTS_MISSING_FALLBACK.to_string())
    } else {
        None
    };
    let runtime = runtime_available.then(|| "xdna1_ffi".to_string());

    vec![AcceleratorDeviceInfo::npu_xdna(
        "xdna:0",
        probe.ordinal.or(Some(0)),
        runtime,
        available,
        reason,
    )]
}

/// Convenience wrapper: read config from the environment and probe live hardware.
pub fn xdna_inventory_devices_from_env() -> Vec<AcceleratorDeviceInfo> {
    xdna_inventory_devices(
        &XdnaInventoryConfig::from_env(),
        &XdnaHardwareProbe::detect(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpuModuleAdmission {
    pub target: NpuModuleTarget,
    pub module_kind: &'static str,
    pub module_id: String,
    pub opt_in: bool,
    pub artifacts_available: bool,
    pub selection: BackendSelection,
}

impl NpuModuleAdmission {
    pub fn admitted(&self) -> bool {
        self.opt_in
            && self.artifacts_available
            && self.selection.selected_backend == DenseFfnBackend::NpuXdna
    }

    pub fn fallback_reason(&self) -> Option<&'static str> {
        self.selection.fallback_reason
    }
}

pub fn xdna_swiglu_admission(
    invocation: &ModuleInvocation,
    artifacts: &XdnaModuleArtifacts,
) -> NpuModuleAdmission {
    let opt_in =
        invocation.backend_selection().preferred_backend == DenseFfnBackendPreference::NpuOptIn;
    let artifacts_available = artifacts.complete();
    let (selected_backend, fallback_reason) = if opt_in && artifacts_available {
        (DenseFfnBackend::NpuXdna, None)
    } else if opt_in {
        (
            DenseFfnBackend::GpuProduction,
            Some(NPU_ARTIFACTS_MISSING_FALLBACK),
        )
    } else {
        (DenseFfnBackend::GpuProduction, None)
    };

    NpuModuleAdmission {
        target: NpuModuleTarget::XdnaSwiglu,
        module_kind: invocation.module_kind(),
        module_id: invocation.module_id().to_string(),
        opt_in,
        artifacts_available,
        selection: BackendSelection::new(
            invocation.backend_selection().preferred_backend,
            selected_backend,
            fallback_reason,
        ),
    }
}

pub fn npu_module_admission_json(admission: &NpuModuleAdmission) -> Value {
    json!({
        "target": admission.target.as_str(),
        "module_kind": admission.module_kind,
        "module_id": admission.module_id,
        "opt_in": admission.opt_in,
        "artifacts_available": admission.artifacts_available,
        "admitted": admission.admitted(),
        "preferred_backend": admission.selection.preferred_backend.as_str(),
        "selected_backend": admission.selection.selected_backend.as_str(),
        "fallback_reason": admission.selection.fallback_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_cpu::{dense_ffn_module_invocation_from_shape, DenseFfnBackendPreference};

    fn embedding_image_key() -> EmbeddingImageCacheKey {
        EmbeddingImageCacheKey {
            npu_architecture: "aie2p".into(),
            model_geometry: EmbeddingModelGeometry {
                architecture: "qwen3".into(),
                hidden_size: 1024,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                intermediate_size: 3072,
            },
            quant_format: "oq8+".into(),
            sequence_bucket: 512,
            dispatch_batch: 8,
        }
    }

    #[test]
    fn embedding_image_cache_key_covers_arch_geometry_quant_bucket_and_batch() {
        let key = embedding_image_key();
        assert_eq!(
            key.directory_name().unwrap(),
            "aie2p-qwen3-h1024-l28-qh16-kvh8-d128-i3072-oq8+-s512-b8"
        );
        let mut different = key.clone();
        different.dispatch_batch = 4;
        assert_ne!(
            key.directory_name().unwrap(),
            different.directory_name().unwrap()
        );
    }

    #[test]
    fn embedding_image_resolution_fails_closed_on_incompatible_manifest() {
        let root = std::env::temp_dir().join(format!(
            "hipfire-npu-image-resolution-{}",
            std::process::id()
        ));
        let key = embedding_image_key();
        let directory = root.join("embedding").join(key.directory_name().unwrap());
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("final.xclbin"), b"xclbin").unwrap();
        std::fs::write(directory.join("insts.bin"), b"instructions").unwrap();
        let mut wrong_key = key.clone();
        wrong_key.sequence_bucket = 256;
        let manifest = EmbeddingImageManifest {
            schema: EMBEDDING_IMAGE_SCHEMA.into(),
            runtime_abi: FULL_EMBEDDING_ENCODER_ABI.into(),
            key: wrong_key,
            xclbin: "final.xclbin".into(),
            instructions: "insts.bin".into(),
        };
        std::fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let error = resolve_embedding_image(&root, &key).unwrap_err();
        assert!(error.contains("incompatible"));

        std::fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec(&EmbeddingImageManifest {
                runtime_abi: "hipfire.full_embedding_encoder.v0".into(),
                key: key.clone(),
                ..manifest.clone()
            })
            .unwrap(),
        )
        .unwrap();
        let error = resolve_embedding_image(&root, &key).unwrap_err();
        assert!(error.contains("runtime ABI"));

        std::fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec(&EmbeddingImageManifest {
                key: key.clone(),
                ..manifest
            })
            .unwrap(),
        )
        .unwrap();
        let resolved = resolve_embedding_image(&root, &key).unwrap();
        assert_eq!(resolved.xclbin, directory.join("final.xclbin"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn xdna_artifacts_require_both_paths() {
        assert!(!XdnaModuleArtifacts::new(None, None).complete());
        assert!(!XdnaModuleArtifacts::new(Some("a.xclbin".to_string()), None).complete());
        assert!(XdnaModuleArtifacts::new(
            Some("a.xclbin".to_string()),
            Some("a.instr".to_string())
        )
        .complete());
    }

    #[test]
    fn xdna_inventory_stays_empty_without_hardware_or_config() {
        let devices = xdna_inventory_devices(
            &XdnaInventoryConfig::new(None, XdnaModuleArtifacts::new(None, None)),
            &XdnaHardwareProbe::absent(),
        );

        assert!(devices.is_empty());
    }

    #[test]
    fn xdna_inventory_reports_configured_runtime_and_artifacts() {
        let devices = xdna_inventory_devices(
            &XdnaInventoryConfig::new(
                Some("libxdna1.so".to_string()),
                XdnaModuleArtifacts::new(Some("a.xclbin".to_string()), Some("a.instr".to_string())),
            ),
            &XdnaHardwareProbe::present(Some(0)),
        );

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].kind, "npu");
        assert_eq!(devices[0].device_id, "xdna:0");
        assert_eq!(devices[0].arch.as_deref(), Some("xdna"));
        assert_eq!(devices[0].runtime.as_deref(), Some("xdna1_ffi"));
        assert!(devices[0].available);
        assert_eq!(devices[0].reason, None);
    }

    #[test]
    fn xdna_inventory_marks_partial_configuration_unavailable() {
        let devices = xdna_inventory_devices(
            &XdnaInventoryConfig::new(
                Some("libxdna1.so".to_string()),
                XdnaModuleArtifacts::new(Some("a.xclbin".to_string()), None),
            ),
            &XdnaHardwareProbe::present(Some(0)),
        );

        assert_eq!(devices.len(), 1);
        assert!(!devices[0].available);
        assert_eq!(
            devices[0].reason.as_deref(),
            Some(NPU_ARTIFACTS_MISSING_FALLBACK)
        );
    }

    #[test]
    fn xdna_inventory_marks_artifacts_without_runtime_unavailable() {
        let devices = xdna_inventory_devices(
            &XdnaInventoryConfig::new(
                None,
                XdnaModuleArtifacts::new(Some("a.xclbin".to_string()), Some("a.instr".to_string())),
            ),
            &XdnaHardwareProbe::present(Some(0)),
        );

        assert_eq!(devices.len(), 1);
        assert!(!devices[0].available);
        assert_eq!(
            devices[0].reason.as_deref(),
            Some(XDNA_RUNTIME_MISSING_FALLBACK)
        );
    }

    #[test]
    fn xdna_inventory_surfaces_detected_hardware_when_unconfigured() {
        // NPU present but operator configured nothing: the device is now visible
        // (not hidden as before) and flagged unavailable with the runtime reason.
        let devices = xdna_inventory_devices(
            &XdnaInventoryConfig::new(None, XdnaModuleArtifacts::new(None, None)),
            &XdnaHardwareProbe::present(Some(0)),
        );

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].kind, "npu");
        assert!(!devices[0].available);
        assert_eq!(
            devices[0].reason.as_deref(),
            Some(XDNA_RUNTIME_MISSING_FALLBACK)
        );
    }

    #[test]
    fn xdna_inventory_marks_configured_hardware_absent() {
        // Fully configured but no NPU detected: stay visible with a hardware
        // reason instead of falsely reporting available.
        let devices = xdna_inventory_devices(
            &XdnaInventoryConfig::new(
                Some("libxdna1.so".to_string()),
                XdnaModuleArtifacts::new(Some("a.xclbin".to_string()), Some("a.instr".to_string())),
            ),
            &XdnaHardwareProbe::absent(),
        );

        assert_eq!(devices.len(), 1);
        assert!(!devices[0].available);
        assert_eq!(
            devices[0].reason.as_deref(),
            Some(NPU_HARDWARE_ABSENT_FALLBACK)
        );
    }

    #[test]
    fn xdna_swiglu_admission_keeps_npu_opt_in_explicit() {
        let invocation = ModuleInvocation::from(dense_ffn_module_invocation_from_shape(
            3,
            4096,
            11008,
            DenseFfnBackendPreference::NpuOptIn,
            false,
        ));

        let missing = xdna_swiglu_admission(&invocation, &XdnaModuleArtifacts::new(None, None));
        assert!(!missing.admitted());
        assert_eq!(
            missing.selection.preferred_backend,
            DenseFfnBackendPreference::NpuOptIn
        );
        assert_eq!(
            missing.selection.selected_backend,
            DenseFfnBackend::GpuProduction
        );
        assert_eq!(
            missing.fallback_reason(),
            Some(NPU_ARTIFACTS_MISSING_FALLBACK)
        );

        let admitted = xdna_swiglu_admission(
            &invocation,
            &XdnaModuleArtifacts::new(Some("a.xclbin".to_string()), Some("a.instr".to_string())),
        );
        assert!(admitted.admitted());
        assert_eq!(
            admitted.selection.selected_backend,
            DenseFfnBackend::NpuXdna
        );
        assert_eq!(
            npu_module_admission_json(&admitted)["target"],
            "xdna_swiglu"
        );
    }

    #[test]
    fn xdna_swiglu_admission_leaves_gpu_path_as_production() {
        let invocation = ModuleInvocation::from(dense_ffn_module_invocation_from_shape(
            3,
            4096,
            11008,
            DenseFfnBackendPreference::GpuProduction,
            false,
        ));

        let admission = xdna_swiglu_admission(
            &invocation,
            &XdnaModuleArtifacts::new(Some("a.xclbin".to_string()), Some("a.instr".to_string())),
        );
        assert!(!admission.opt_in);
        assert!(!admission.admitted());
        assert_eq!(
            admission.selection.preferred_backend,
            DenseFfnBackendPreference::GpuProduction
        );
        assert_eq!(
            admission.selection.selected_backend,
            DenseFfnBackend::GpuProduction
        );
        assert_eq!(admission.fallback_reason(), None);
    }
}
