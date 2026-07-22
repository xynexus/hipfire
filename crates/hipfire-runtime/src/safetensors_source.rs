//! SafetensorsSource: load HuggingFace safetensors models directly.
//!
//! Supports ParoQuant, AWQ, and unquantized safetensors models.
//! Reads config.json for architecture detection and quantization config.
//! Mmaps .safetensors files and serves tensor data by name.

use hipfire_model::{
    ModelSource, QuantConfig, TensorInfo, TensorStorageLocation, ARCH_ID_GEMMA3_TEXT,
    ARCH_ID_GEMMA3_VL, ARCH_ID_MAMBA2, ARCH_ID_NEMOTRON_H,
};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read as _;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::fd::AsRawFd;

#[cfg(unix)]
fn release_mmap_range(mmap: &Mmap, offset: usize, len: usize) -> std::io::Result<()> {
    if len == 0 {
        return Ok(());
    }
    // SAFETY: callers invoke this only after the temporary tensor view and its
    // synchronous host-to-device upload are complete. No slice into this range
    // remains live. Later reads are permitted and fault the file bytes back in.
    unsafe { mmap.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, offset, len) }
}

struct SafetensorsFile {
    _file: File,
    mmap: Mmap,
    path: PathBuf,
}

pub struct SafetensorsSource {
    dir: PathBuf,
    files: Vec<SafetensorsFile>,
    tensors: Vec<TensorInfo>,
    tensor_map: HashMap<String, (usize, usize)>, // name -> (file_idx, tensor_idx)
    metadata_json_cached: String,
    arch_id: u32,
    quant_config: Option<QuantConfig>,
}

impl SafetensorsSource {
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        // Read config.json
        let config_path = dir.join("config.json");
        let mut config_str = String::new();
        File::open(&config_path)?.read_to_string(&mut config_str)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Derive arch_id from architectures field
        let arch_id = derive_arch_id(&config);

        // Parse quantization config
        let quant_config = parse_quant_config(&config);

        // Build metadata JSON in HFQ-compatible format
        let metadata_json_cached = build_metadata_json(&config, &config_str);

        // Find and open all .safetensors files
        let mut st_paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        st_paths.sort();

        if st_paths.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{}: no .safetensors files found", dir.display()),
            ));
        }

        let mut files = Vec::new();
        let mut tensors = Vec::new();
        let mut tensor_map = HashMap::new();

        for (file_idx, st_path) in st_paths.iter().enumerate() {
            let file = File::open(st_path)?;
            let mmap = unsafe { Mmap::map(&file)? };

            // Parse safetensors header
            if mmap.len() < 8 {
                return Err(invalid_safetensors(
                    st_path,
                    "file is shorter than header prefix",
                ));
            }
            let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header_size = 8usize.checked_add(header_len).ok_or_else(|| {
                invalid_safetensors(st_path, "header length overflows address space")
            })?;
            if header_size > mmap.len() {
                return Err(invalid_safetensors(st_path, "header extends beyond file"));
            }
            let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let raw: serde_json::Value = serde_json::from_str(header_json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            if let serde_json::Value::Object(map) = raw {
                for (name, meta) in map {
                    if name == "__metadata__" {
                        continue;
                    }
                    let dtype = meta["dtype"].as_str().unwrap_or("F16").to_string();
                    let shape: Vec<usize> = meta["shape"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect()
                        })
                        .unwrap_or_default();
                    let offsets = meta["data_offsets"].as_array().ok_or_else(|| {
                        invalid_safetensors(st_path, &format!("tensor {name} has no data_offsets"))
                    })?;
                    if offsets.len() != 2 {
                        return Err(invalid_safetensors(
                            st_path,
                            &format!("tensor {name} data_offsets must contain two values"),
                        ));
                    }
                    let start = offsets[0].as_u64().ok_or_else(|| {
                        invalid_safetensors(st_path, &format!("tensor {name} start is not u64"))
                    })? as usize;
                    let end = offsets[1].as_u64().ok_or_else(|| {
                        invalid_safetensors(st_path, &format!("tensor {name} end is not u64"))
                    })? as usize;
                    if end < start
                        || header_size
                            .checked_add(end)
                            .is_none_or(|end| end > mmap.len())
                    {
                        return Err(invalid_safetensors(
                            st_path,
                            &format!("tensor {name} byte range is outside the shard"),
                        ));
                    }
                    if tensor_map.contains_key(&name) {
                        return Err(invalid_safetensors(
                            st_path,
                            &format!("tensor {name} appears in more than one shard"),
                        ));
                    }

                    let tensor_idx = tensors.len();
                    let info = TensorInfo {
                        name: name.clone(),
                        dtype,
                        shape,
                        quant_type: 0xFF, // not an HFQ quant_type
                        data_offset: header_size + start,
                        data_size: end - start,
                    };
                    tensors.push(info);
                    tensor_map.insert(name, (file_idx, tensor_idx));
                }
            }

            files.push(SafetensorsFile {
                _file: file,
                mmap,
                path: st_path.clone(),
            });
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            files,
            tensors,
            tensor_map,
            metadata_json_cached,
            arch_id,
            quant_config,
        })
    }

    /// Raw bytes at an absolute byte range within a specific shard mmap.
    ///
    /// Used to back an in-memory [`crate::hfq::HfqFile`] over a safetensors
    /// directory: stacked routed-expert sub-ranges are not exposed as named
    /// tensors, so the HfqFile builder addresses them directly by
    /// `(shard, offset, len)` computed from the parent tensor's layout.
    pub fn shard_bytes(&self, shard_idx: usize, offset: usize, len: usize) -> Option<&[u8]> {
        let mmap = &self.files.get(shard_idx)?.mmap;
        mmap.get(offset..offset.checked_add(len)?)
    }

    /// Per-tensor physical layout as `(name, shard_idx, absolute_offset,
    /// byte_len, dtype, shape)`. Exposes the shard index the `ModelSource` API
    /// hides so an `HfqFile` can be built directly over the mmapped shards
    /// without a temporary bf16 `.hfq` roundtrip.
    pub fn tensor_layout(&self) -> Vec<(String, usize, usize, usize, String, Vec<usize>)> {
        self.tensor_map
            .iter()
            .map(|(name, &(file_idx, tensor_idx))| {
                let info = &self.tensors[tensor_idx];
                (
                    name.clone(),
                    file_idx,
                    info.data_offset,
                    info.data_size,
                    info.dtype.clone(),
                    info.shape.clone(),
                )
            })
            .collect()
    }
}

impl ModelSource for SafetensorsSource {
    fn metadata_json(&self) -> &str {
        &self.metadata_json_cached
    }

    fn arch_id(&self) -> u32 {
        self.arch_id
    }

    fn quant_config(&self) -> Option<&QuantConfig> {
        self.quant_config.as_ref()
    }

    fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
        let &(file_idx, tensor_idx) = self.tensor_map.get(name)?;
        let info = &self.tensors[tensor_idx];
        let mmap = &self.files[file_idx].mmap;
        Some((
            info,
            &mmap[info.data_offset..info.data_offset + info.data_size],
        ))
    }

    fn release_tensor_pages(&self, name: &str) {
        let Some(&(_file_idx, tensor_idx)) = self.tensor_map.get(name) else {
            return;
        };
        self.release_tensor_range_pages(name, 0, self.tensors[tensor_idx].data_size);
    }

    fn release_tensor_range_pages(&self, name: &str, byte_offset: usize, byte_len: usize) -> bool {
        let Some(&(file_idx, tensor_idx)) = self.tensor_map.get(name) else {
            return false;
        };
        let info = &self.tensors[tensor_idx];
        if byte_len == 0 || byte_offset >= info.data_size {
            return false;
        }
        let byte_len = byte_len.min(info.data_size - byte_offset);
        let data_offset = info.data_offset + byte_offset;
        #[cfg(unix)]
        {
            // MADV_DONTNEED removes this mapping's resident PTEs immediately;
            // posix_fadvise then gives the page cache the matching backing-file
            // hint. The mapping remains valid and refaults if a declared alias
            // later reads the same source range.
            let _ = release_mmap_range(&self.files[file_idx].mmap, data_offset, byte_len);
            unsafe {
                libc::posix_fadvise(
                    self.files[file_idx]._file.as_raw_fd(),
                    data_offset as libc::off_t,
                    byte_len as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
            true
        }
        #[cfg(not(unix))]
        {
            let _ = (file_idx, data_offset, byte_len);
            false
        }
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let &(_file_idx, tensor_idx) = self.tensor_map.get(name)?;
        Some(&self.tensors[tensor_idx])
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    fn path(&self) -> &Path {
        &self.dir
    }

    fn tensor_storage(&self, name: &str) -> Option<TensorStorageLocation> {
        let &(file_idx, tensor_idx) = self.tensor_map.get(name)?;
        let info = &self.tensors[tensor_idx];
        Some(TensorStorageLocation {
            path: self.files[file_idx].path.clone(),
            byte_offset: info.data_offset as u64,
            byte_len: info.data_size as u64,
        })
    }

    fn tokenizer_json_path(&self) -> Option<PathBuf> {
        let p = self.dir.join("tokenizer.json");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    fn chat_template(&self) -> Option<String> {
        let p = self.dir.join("tokenizer_config.json");
        let mut s = String::new();
        File::open(p).ok()?.read_to_string(&mut s).ok()?;
        let v: serde_json::Value = serde_json::from_str(&s).ok()?;
        v.get("chat_template")?.as_str().map(|s| s.to_string())
    }
}

fn invalid_safetensors(path: &Path, message: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("{}: {message}", path.display()),
    )
}

fn derive_arch_id(config: &serde_json::Value) -> u32 {
    let archs = config
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    // Check text_config for MoE indicators
    let text_config = config.get("text_config").unwrap_or(config);
    let has_experts = text_config
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        > 0;

    for arch in &archs {
        let arch_lower = arch.to_lowercase();
        if arch_lower.contains("gemma3forcausallm") {
            return ARCH_ID_GEMMA3_TEXT;
        }
        if arch_lower.contains("gemma3forconditionalgeneration") {
            return ARCH_ID_GEMMA3_VL;
        }
        if arch_lower.contains("qwen3_5")
            || arch_lower.contains("qwen3.5")
            || arch_lower.contains("qwen3_6")
            || arch_lower.contains("qwen3.6")
        {
            return if has_experts { 6 } else { 5 };
        }
        if arch_lower.contains("qwen3") || arch_lower.contains("qwen2") {
            return 1;
        }
        if arch_lower.contains("llama") || arch_lower.contains("mistral") {
            return 0;
        }
        // NemotronHForCausalLM (Mamba-2 + attn + MLP hybrid). Match the "H"
        // hybrid specifically so plain (llama-based) Nemotron isn't caught.
        if arch_lower.contains("nemotronh") {
            return ARCH_ID_NEMOTRON_H;
        }
        if arch_lower.contains("mamba2") {
            return ARCH_ID_MAMBA2;
        }
    }

    if config
        .get("ssm_cfg")
        .and_then(|v| v.get("layer"))
        .and_then(|v| v.as_str())
        .map(|s| s.eq_ignore_ascii_case("mamba2"))
        .unwrap_or(false)
    {
        return ARCH_ID_MAMBA2;
    }

    // Fallback: check model_type
    let model_type = config
        .get("model_type")
        .or_else(|| text_config.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match model_type {
        "qwen3_5" | "qwen3.5" | "qwen3_6" | "qwen3.6" => {
            if has_experts {
                6
            } else {
                5
            }
        }
        "qwen3" | "qwen2" => 1,
        "llama" | "mistral" => 0,
        "gemma3_text" => ARCH_ID_GEMMA3_TEXT,
        "gemma3" => ARCH_ID_GEMMA3_VL,
        "nemotron_h" => ARCH_ID_NEMOTRON_H,
        "mamba2" => ARCH_ID_MAMBA2,
        _ => {
            eprintln!(
                "warning: unknown model_type '{model_type}', defaulting to arch_id=5 (Qwen3.5)"
            );
            5
        }
    }
}

fn parse_quant_config(config: &serde_json::Value) -> Option<QuantConfig> {
    let qc = config.get("quantization_config")?;
    let method = qc.get("quant_method")?.as_str()?.to_string();
    let bits = qc.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as u8;
    let group_size = qc.get("group_size").and_then(|v| v.as_u64()).unwrap_or(128) as u32;
    let krot = qc.get("krot").and_then(|v| v.as_u64()).unwrap_or(0) as u8;

    let dynamic_excludes = qc
        .get("dynamic")
        .and_then(|d| d.as_object())
        .map(|obj| {
            obj.keys()
                .filter(|k| k.starts_with("-:"))
                .map(|k| k.strip_prefix("-:").unwrap_or(k).to_string())
                .collect()
        })
        .unwrap_or_default();

    Some(QuantConfig {
        method,
        bits,
        group_size,
        krot,
        dynamic_excludes,
    })
}

fn build_metadata_json(config: &serde_json::Value, raw_config: &str) -> String {
    // Build HFQ-compatible metadata: { "architecture": "...", "config": {...} }
    // The Qwen35 config parser expects metadata_json to contain a "config" key.
    let mut meta = serde_json::Map::new();

    // Determine architecture string
    let text_config = config.get("text_config").unwrap_or(config);
    let model_type = text_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    meta.insert(
        "architecture".to_string(),
        serde_json::Value::String(model_type.to_string()),
    );

    // Embed the full config.json as the "config" key
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw_config) {
        meta.insert("config".to_string(), parsed);
    }

    serde_json::to_string(&serde_json::Value::Object(meta)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hipfire-safetensors-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn write_shard(path: &Path, name: &str, bytes: &[u8]) {
        let mut header = serde_json::Map::new();
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": [bytes.len() / 2],
                "data_offsets": [0, bytes.len()],
            }),
        );
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(bytes).unwrap();
    }

    #[test]
    fn derives_gemma3_text_and_vl_architectures() {
        assert_eq!(
            derive_arch_id(&serde_json::json!({
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3_text"
            })),
            ARCH_ID_GEMMA3_TEXT
        );
        assert_eq!(
            derive_arch_id(&serde_json::json!({
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "text_config": { "model_type": "gemma3_text" }
            })),
            ARCH_ID_GEMMA3_VL
        );
    }

    #[test]
    fn multi_shard_source_reports_exact_backing_file_and_range() {
        let dir = temp_dir("multi-shard");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
        )
        .unwrap();
        write_shard(
            &dir.join("model-00001-of-00002.safetensors"),
            "model.embed_tokens.weight",
            &[1, 2, 3, 4],
        );
        write_shard(
            &dir.join("model-00002-of-00002.safetensors"),
            "model.layers.0.mlp.down_proj.weight",
            &[5, 6, 7, 8, 9, 10],
        );

        let source = SafetensorsSource::open(&dir).unwrap();
        assert_eq!(source.tensor_names().len(), 2);
        let embed = source.tensor_storage("model.embed_tokens.weight").unwrap();
        assert!(embed.path.ends_with("model-00001-of-00002.safetensors"));
        assert_eq!(embed.byte_len, 4);
        assert_eq!(
            source.tensor_data("model.embed_tokens.weight").unwrap().1,
            [1, 2, 3, 4]
        );
        let down = source
            .tensor_storage("model.layers.0.mlp.down_proj.weight")
            .unwrap();
        assert!(down.path.ends_with("model-00002-of-00002.safetensors"));
        assert_eq!(down.byte_len, 6);
        drop(source);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn released_mmap_range_refaults_the_original_tensor_bytes() {
        let dir = temp_dir("madvise-refault");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        let payload = vec![0x5au8; 128 * 1024];
        write_shard(&dir.join("model.safetensors"), "weight", &payload);

        let source = SafetensorsSource::open(&dir).unwrap();
        assert_eq!(source.tensor_data("weight").unwrap().1, payload);
        assert!(source.release_tensor_range_pages("weight", 0, 64 * 1024));
        assert!(source.release_tensor_range_pages("weight", 64 * 1024, 64 * 1024));
        assert!(!source.release_tensor_range_pages("weight", 128 * 1024, 1));
        assert_eq!(source.tensor_data("weight").unwrap().1, payload);

        drop(source);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn malformed_range_and_duplicate_tensor_are_rejected() {
        let malformed = temp_dir("malformed");
        fs::create_dir_all(&malformed).unwrap();
        fs::write(malformed.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        let header = serde_json::to_vec(&serde_json::json!({
            "bad": {"dtype":"F16", "shape":[2], "data_offsets":[0, 99]}
        }))
        .unwrap();
        let mut file = File::create(malformed.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&[0, 1, 2, 3]).unwrap();
        drop(file);
        assert!(SafetensorsSource::open(&malformed).is_err());
        fs::remove_dir_all(malformed).unwrap();

        let duplicate = temp_dir("duplicate");
        fs::create_dir_all(&duplicate).unwrap();
        fs::write(duplicate.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        write_shard(&duplicate.join("a.safetensors"), "same", &[1, 2]);
        write_shard(&duplicate.join("b.safetensors"), "same", &[3, 4]);
        assert!(SafetensorsSource::open(&duplicate).is_err());
        fs::remove_dir_all(duplicate).unwrap();
    }
}
