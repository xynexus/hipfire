// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Eval dataset resolution, fetching, and prompt-artifact materialization.
//!
//! Resolves the dataset manifest for the selected suites (GPQA / HumanEval /
//! lm-eval-micro / builtin barrage), fetches/caches them, parses their items,
//! and writes per-item prompt artifacts with provenance. Extracted verbatim
//! from the former `hipfire-eval/src/lib.rs` monolith (no behavior change).

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::*;

pub(crate) fn resolve_datasets(config: &EvalConfig) -> Result<Vec<DatasetManifestEntry>, String> {
    let mut entries = Vec::new();
    for suite in &config.suites {
        if matches!(
            *suite,
            SuiteId::LmEvalMicro | SuiteId::DeepSwe | SuiteId::SweBench
        ) {
            entries.push(builtin_dataset_entry(*suite));
            continue;
        }
        // Niah / SequentialNiah / Ruler ship as vendored in-repo fixtures —
        // always available, no fetch (resolved by `resolve_repo_path` at run time).
        if matches!(
            *suite,
            SuiteId::Niah | SuiteId::SequentialNiah | SuiteId::Ruler
        ) {
            entries.push(local_longctx_dataset_entry(*suite));
            continue;
        }
        let cache_path = config.dataset_cache.join(suite.as_str());
        if let Some(reason) = dataset_unavailable_reason(*suite, &cache_path) {
            if config.fetch_datasets {
                match fetch_dataset(*suite, &cache_path) {
                    Ok(fetched) => entries.push(DatasetManifestEntry {
                        suite: *suite,
                        source: fetched.source,
                        repo_id: suite.hf_repo_id().map(str::to_string),
                        revision: fetched.revision,
                        files: fetched.files,
                        digest: directory_hash(&cache_path),
                        license: suite.license().map(str::to_string),
                        cache_path: cache_path.display().to_string(),
                        selected_item_ids: selected_item_ids(*suite),
                        status: EvalStatus::Pass,
                        reason: None,
                    }),
                    Err(reason) => entries.push(dataset_skip(*suite, &cache_path, reason)),
                }
                continue;
            }

            let reason = if config.offline && !cache_path.exists() {
                "dataset not cached and --offline forbids fetch".to_string()
            } else if config.offline {
                format!("{reason}; --offline forbids fetch")
            } else {
                format!("{reason}; rerun with --fetch-datasets to opt in")
            };
            entries.push(dataset_skip(*suite, &cache_path, reason));
            continue;
        }

        if cache_path.exists() {
            entries.push(DatasetManifestEntry {
                suite: *suite,
                source: "local_cache".to_string(),
                repo_id: suite.hf_repo_id().map(str::to_string),
                revision: suite.hf_revision().map(str::to_string),
                files: list_files(&cache_path),
                digest: directory_hash(&cache_path),
                license: suite.license().map(str::to_string),
                cache_path: cache_path.display().to_string(),
                selected_item_ids: selected_item_ids(*suite),
                status: EvalStatus::Pass,
                reason: None,
            });
            continue;
        }
    }
    Ok(entries)
}

pub(crate) fn builtin_dataset_entry(suite: SuiteId) -> DatasetManifestEntry {
    let selected_item_ids = selected_item_ids(suite);
    let files = match suite {
        SuiteId::LmEvalMicro => vec!["builtin:lm_eval_micro:v1".to_string()],
        SuiteId::DeepSwe => vec!["builtin:deep_swe_micro:v1".to_string()],
        SuiteId::SweBench => vec!["builtin:swe_bench_micro:v1".to_string()],
        _ => Vec::new(),
    };
    let digest = match suite {
        SuiteId::LmEvalMicro => Some(stable_hash_bytes(
            lm_eval_micro_items()
                .iter()
                .flat_map(|item| {
                    item.item_id
                        .as_bytes()
                        .iter()
                        .copied()
                        .chain([0])
                        .chain(item.prompt.as_bytes().iter().copied())
                        .chain([0xff])
                })
                .collect::<Vec<_>>()
                .as_slice(),
        )),
        SuiteId::DeepSwe | SuiteId::SweBench => Some(stable_hash_bytes(
            builtin_barrage_items(suite)
                .iter()
                .flat_map(|item| {
                    item.item_id
                        .as_bytes()
                        .iter()
                        .copied()
                        .chain([0])
                        .chain(item.prompt.as_bytes().iter().copied())
                        .chain([0xff])
                })
                .collect::<Vec<_>>()
                .as_slice(),
        )),
        _ => None,
    };
    DatasetManifestEntry {
        suite,
        source: "builtin".to_string(),
        repo_id: None,
        revision: Some("hipfire-native-v1".to_string()),
        files,
        digest,
        license: Some("hipfire-native".to_string()),
        cache_path: format!("builtin:{}", suite.as_str()),
        selected_item_ids,
        status: EvalStatus::Pass,
        reason: None,
    }
}

/// Vendored local long-context fixtures (Niah / SequentialNiah): always
/// available in-repo, no fetch. Item ids resolve to `benchmarks/<subdir>/<name>.jsonl`.
pub(crate) fn local_longctx_dataset_entry(suite: SuiteId) -> DatasetManifestEntry {
    let subdir = match suite {
        SuiteId::SequentialNiah => "benchmarks/longctx/seqniah",
        SuiteId::Ruler => "benchmarks/longctx/ruler",
        _ => "benchmarks/longctx/niah",
    };
    let ids = selected_item_ids(suite);
    let files: Vec<String> = ids
        .iter()
        .map(|id| {
            let name = id.split_once(':').map(|(n, _)| n).unwrap_or(id);
            format!("{subdir}/{name}.jsonl")
        })
        .collect();
    DatasetManifestEntry {
        suite,
        source: "local_fixtures".to_string(),
        repo_id: None,
        revision: Some("hipfire-vendored-v1".to_string()),
        files,
        digest: None,
        license: Some("hipfire-vendored".to_string()),
        cache_path: subdir.to_string(),
        selected_item_ids: ids,
        status: EvalStatus::Pass,
        reason: None,
    }
}

pub(crate) fn dataset_unavailable_reason(suite: SuiteId, cache_path: &Path) -> Option<String> {
    match suite {
        SuiteId::Gpqa => {
            if !cache_path.exists() {
                return Some("dataset not cached".to_string());
            }
            if gpqa_csv_paths(cache_path).is_empty() {
                if cache_path.join("dataset.zip").exists() {
                    Some(
                        "GPQA cache contains encrypted dataset.zip but no extracted gpqa_*.csv files"
                            .to_string(),
                    )
                } else {
                    Some("GPQA cache has no gpqa_*.csv files".to_string())
                }
            } else {
                None
            }
        }
        SuiteId::HumanEval => {
            if !cache_path.exists() {
                return Some("dataset not cached".to_string());
            }
            if humaneval_jsonl_paths(cache_path).is_empty() {
                Some("HumanEval cache has no HumanEval*.jsonl files".to_string())
            } else {
                None
            }
        }
        _ => {
            if cache_path.exists() {
                None
            } else {
                Some("dataset not cached".to_string())
            }
        }
    }
}

pub(crate) fn dataset_skip(
    suite: SuiteId,
    cache_path: &Path,
    reason: String,
) -> DatasetManifestEntry {
    DatasetManifestEntry {
        suite,
        source: "unavailable".to_string(),
        repo_id: suite.hf_repo_id().map(str::to_string),
        revision: suite.hf_revision().map(str::to_string),
        files: Vec::new(),
        digest: None,
        license: suite.license().map(str::to_string),
        cache_path: cache_path.display().to_string(),
        selected_item_ids: selected_item_ids(suite),
        status: EvalStatus::Skip,
        reason: Some(reason),
    }
}

pub(crate) struct FetchedDataset {
    pub(crate) source: String,
    pub(crate) revision: Option<String>,
    pub(crate) files: Vec<String>,
}

pub(crate) fn fetch_dataset(suite: SuiteId, cache_path: &Path) -> Result<FetchedDataset, String> {
    if let Ok(root) = std::env::var("HIPFIRE_EVAL_DATASET_MIRROR") {
        let mirror_path = Path::new(&root).join(suite.as_str());
        if mirror_path.exists() {
            copy_dir_recursive(&mirror_path, cache_path).map_err(|e| {
                format!(
                    "copy dataset mirror {} to {}: {e}",
                    mirror_path.display(),
                    cache_path.display()
                )
            })?;
            return Ok(FetchedDataset {
                source: "local_mirror".to_string(),
                revision: suite.hf_revision().map(str::to_string),
                files: list_files(cache_path),
            });
        }
    }

    let repo_id = suite
        .hf_repo_id()
        .ok_or_else(|| format!("suite {} has no native HF fetch recipe yet", suite.as_str()))?;
    fs::create_dir_all(cache_path).map_err(|e| format!("create dataset cache: {e}"))?;
    let revision = suite.hf_revision();
    let script = format!(
        "from huggingface_hub import snapshot_download\nsnapshot_download(repo_id={repo_id:?}, repo_type='dataset', revision={revision:?}, local_dir={cache:?}, local_dir_use_symlinks=False)",
        repo_id = repo_id,
        revision = revision,
        cache = cache_path.display().to_string(),
    );
    let out = Command::new("python3")
        .args(["-c", &script])
        .output()
        .map_err(|e| format!("python3/huggingface_hub unavailable: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(FetchedDataset {
        source: "huggingface".to_string(),
        revision: revision.map(str::to_string),
        files: list_files(cache_path),
    })
}

pub(crate) fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

pub(crate) fn selected_item_ids(suite: SuiteId) -> Vec<String> {
    match suite {
        SuiteId::Gpqa => vec!["gpqa_diamond:0".to_string(), "gpqa_main:0".to_string()],
        SuiteId::LmEvalMicro => vec![
            "arc_easy:0".to_string(),
            "hellaswag:0".to_string(),
            "mmlu_stem:0".to_string(),
        ],
        SuiteId::HumanEval => vec!["HumanEval/0".to_string(), "HumanEval/53".to_string()],
        SuiteId::DeepSwe => vec!["deep_swe_verified:0".to_string()],
        SuiteId::SweBench => vec!["swe_bench_lite:0".to_string()],
        // Ruler: vendored generated slices (S-NIAH + variable-tracking).
        SuiteId::Ruler => vec![
            "ruler_niah_4k:0".to_string(),
            "ruler_niah_8k:0".to_string(),
            "ruler_vt_4k:0".to_string(),
            "ruler_vt_8k:0".to_string(),
        ],
        // NoLiMa: <needle_id>:<test_key>:<ctx>k:<book> over the HF components.
        SuiteId::NoLiMa => vec![
            "0401:T17_C02:4k:1".to_string(),
            "0401:T15_C02:4k:2".to_string(),
        ],
        // NeedleChain: <k-chain>:<ordering>:<row> over the HF parquet shards.
        SuiteId::NeedleChain => vec!["k5:forward:0".to_string(), "k10:forward:0".to_string()],
        // Niah / SequentialNiah: <fixture>:<row> over vendored local fixtures.
        SuiteId::Niah => vec![
            "niah_8k:0".to_string(),
            "niah_16k:0".to_string(),
            "niah_32k:0".to_string(),
            "niah_multi_16k:0".to_string(),
        ],
        SuiteId::SequentialNiah => vec!["seqniah_8k:0".to_string(), "seqniah_16k:0".to_string()],
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GpqaItem {
    pub(crate) item_id: String,
    pub(crate) dataset_file: String,
    pub(crate) prompt: String,
    pub(crate) correct_answer: String,
    pub(crate) answer_label: String,
    pub(crate) choices: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub(crate) struct HumanEvalItem {
    pub(crate) item_id: String,
    pub(crate) task_id: String,
    pub(crate) dataset_file: String,
    pub(crate) prompt: String,
    pub(crate) canonical_solution_hash: Option<String>,
    pub(crate) test_hash: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct LmEvalMicroItem {
    pub(crate) item_id: String,
    pub(crate) task: String,
    pub(crate) prompt: String,
    pub(crate) answer_label: String,
    pub(crate) answer_hash: String,
    pub(crate) choices_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct BuiltinBarrageItem {
    pub(crate) item_id: String,
    pub(crate) suite: SuiteId,
    pub(crate) task: String,
    pub(crate) prompt: String,
    pub(crate) answer_label: String,
    pub(crate) answer_hash: String,
    pub(crate) choices_count: usize,
    pub(crate) dataset_file: String,
    pub(crate) prompt_format: String,
    pub(crate) scoring_mode: String,
}

pub(crate) fn lm_eval_micro_items() -> Vec<LmEvalMicroItem> {
    [
        (
            "arc_easy:0",
            "arc_easy",
            "Question: Which object is designed to measure temperature?\n\nA. Barometer\nB. Thermometer\nC. Compass\nD. Stopwatch\n\nAnswer with only the letter A, B, C, or D.\n",
            "B",
            "Thermometer",
        ),
        (
            "hellaswag:0",
            "hellaswag",
            "Choose the most plausible continuation.\n\nA person opens an umbrella while walking outside because\n\nA. it has started raining.\nB. the oven is preheating.\nC. the book needs a bookmark.\nD. the train is underwater.\n\nAnswer with only the letter A, B, C, or D.\n",
            "A",
            "it has started raining.",
        ),
        (
            "mmlu_stem:0",
            "mmlu_stem",
            "Question: A triangle has angles 30 degrees and 60 degrees. What is the third angle?\n\nA. 30 degrees\nB. 60 degrees\nC. 90 degrees\nD. 120 degrees\n\nAnswer with only the letter A, B, C, or D.\n",
            "C",
            "90 degrees",
        ),
    ]
    .into_iter()
    .map(|(item_id, task, prompt, answer_label, answer)| LmEvalMicroItem {
        item_id: item_id.to_string(),
        task: task.to_string(),
        prompt: prompt.to_string(),
        answer_label: answer_label.to_string(),
        answer_hash: stable_hash_bytes(answer.as_bytes()),
        choices_count: 4,
    })
    .collect()
}

pub(crate) fn lm_eval_micro_materialized_items(
    item_ids: &[String],
) -> Result<Vec<LmEvalMicroItem>, String> {
    let items = lm_eval_micro_items();
    let mut out = Vec::new();
    for id in item_ids {
        let item = items
            .iter()
            .find(|item| &item.item_id == id)
            .cloned()
            .ok_or_else(|| format!("lm_eval_micro item {id} not found"))?;
        out.push(item);
    }
    Ok(out)
}

pub(crate) fn builtin_barrage_items(suite: SuiteId) -> Vec<BuiltinBarrageItem> {
    let rows = match suite {
        SuiteId::DeepSwe => vec![(
            "deep_swe_verified:0",
            "deep_swe_patch_reasoning",
            "A regression report says that `hipfire-eval --suite gpqa --offline` should never try to fetch Hugging Face data. The current parser accepts both `--fetch-datasets` and `--offline`, then later attempts a dataset download.\n\nWhich minimal patch best preserves the intended contract?\n\nA. Ignore `--offline` whenever `--fetch-datasets` is also present.\nB. Reject `--fetch-datasets` and `--offline` together during CLI parsing before any dataset resolution.\nC. Fetch the dataset first, then mark the row skipped if network fails.\nD. Remove the GPQA suite from all tiers.\n\nAnswer with only the letter A, B, C, or D.\n",
            "B",
            "Reject mutually exclusive fetch/offline flags during CLI parsing.",
            "deep_swe_micro_zero_shot_v1",
        )],
        SuiteId::SweBench => vec![(
            "swe_bench_lite:0",
            "swe_bench_bug_localization",
            "A failing test reports: `summary.md does not mention admission verdict reject after --fail-on-admission writes artifacts`. The code already builds `admission.json` correctly, but the Markdown summary only prints pass/fail/skip counts.\n\nWhich change most directly fixes the user-visible bug?\n\nA. Delete `admission.json` so the summary cannot disagree with it.\nB. Change the pass/fail/skip counters to include skipped rows twice.\nC. Add the admission verdict and findings section to `summary.md` using the same admission artifact built for JSON output.\nD. Make `--fail-on-admission` exit before writing artifacts.\n\nAnswer with only the letter A, B, C, or D.\n",
            "C",
            "Add the admission verdict and findings section to the Markdown summary.",
            "swe_bench_micro_zero_shot_v1",
        )],
        _ => Vec::new(),
    };
    rows.into_iter()
        .map(
            |(item_id, task, prompt, answer_label, answer, prompt_format)| BuiltinBarrageItem {
                item_id: item_id.to_string(),
                suite,
                task: task.to_string(),
                prompt: prompt.to_string(),
                answer_label: answer_label.to_string(),
                answer_hash: stable_hash_bytes(answer.as_bytes()),
                choices_count: 4,
                dataset_file: format!("builtin:{}:v1", suite.as_str()),
                prompt_format: prompt_format.to_string(),
                scoring_mode: "exact_letter".to_string(),
            },
        )
        .collect()
}

pub(crate) fn builtin_barrage_materialized_items(
    suite: SuiteId,
    item_ids: &[String],
) -> Result<Vec<BuiltinBarrageItem>, String> {
    let items = builtin_barrage_items(suite);
    let mut out = Vec::new();
    for id in item_ids {
        let item = items
            .iter()
            .find(|item| &item.item_id == id)
            .cloned()
            .ok_or_else(|| format!("{} item {id} not found", suite.as_str()))?;
        out.push(item);
    }
    Ok(out)
}

/// One long-context retrieval item (NIAH-family + NeedleChain), shared across the
/// suites scored by recovering expected substrings from the model's answer.
/// `prompt` is the full assembled text prompt (haystack + question); scoring
/// passes when at least `min_recovered` of `expected` appear in the output.
#[derive(Debug, Clone)]
pub(crate) struct LongCtxItem {
    pub(crate) item_id: String,
    pub(crate) suite: SuiteId,
    pub(crate) case_id: String,
    pub(crate) task: String,
    pub(crate) prompt: String,
    pub(crate) expected: Vec<String>,
    pub(crate) min_recovered: usize,
    pub(crate) context_tokens: usize,
    pub(crate) dataset_file: String,
}

/// Read the first non-empty JSON object from a `.jsonl` fixture (our vendored
/// NIAH fixtures carry one sample per file).
pub(crate) fn read_first_jsonl_object(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read fixture {}: {e}", path.display()))?;
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| format!("empty fixture {}", path.display()))?;
    serde_json::from_str(line).map_err(|e| format!("parse fixture {}: {e}", path.display()))
}

/// Assemble a NIAH prompt (single- or multi-needle) from a fixture object.
/// Single-needle fixtures carry `expected_answer_substring` (min_recovered = 1);
/// multi-needle fixtures carry `expected_answer_substrings` + `min_recovered`.
fn niah_item_from_value(
    suite: SuiteId,
    item_id: &str,
    task: &str,
    dataset_file: &str,
    v: &Value,
) -> Result<LongCtxItem, String> {
    let filler = v
        .get("filler_text")
        .and_then(Value::as_str)
        .ok_or("fixture missing filler_text")?;
    let question = v
        .get("question")
        .and_then(Value::as_str)
        .ok_or("fixture missing question")?;
    let context_tokens = v.get("context_tokens").and_then(Value::as_u64).unwrap_or(0) as usize;
    let (expected, min_recovered) = if let Some(list) = v
        .get("expected_answer_substrings")
        .and_then(Value::as_array)
    {
        let expected: Vec<String> = list
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        let min = v
            .get("min_recovered")
            .and_then(Value::as_u64)
            .unwrap_or(expected.len() as u64) as usize;
        (expected, min.max(1))
    } else if let Some(one) = v.get("expected_answer_substring").and_then(Value::as_str) {
        (vec![one.to_string()], 1)
    } else {
        return Err("fixture missing expected_answer_substring(s)".to_string());
    };
    if expected.is_empty() {
        return Err("fixture has no expected substrings".to_string());
    }
    Ok(LongCtxItem {
        item_id: item_id.to_string(),
        suite,
        case_id: format!("{}_native", suite.as_str()),
        task: task.to_string(),
        prompt: format!("{filler}\n\n{question}"),
        expected,
        min_recovered,
        context_tokens,
        dataset_file: dataset_file.to_string(),
    })
}

/// Materialize a NIAH-family suite from vendored local fixtures. Item ids are
/// `<fixture>:<row>` (one row/file), resolved to
/// `benchmarks/<subdir>/<fixture>.jsonl`.
fn niah_family_items(
    suite: SuiteId,
    subdir: &str,
    item_ids: &[String],
) -> Result<Vec<LongCtxItem>, String> {
    let mut out = Vec::new();
    for id in item_ids {
        let name = id.split_once(':').map(|(n, _)| n).unwrap_or(id.as_str());
        let rel = format!("benchmarks/{subdir}/{name}.jsonl");
        let path = crate::resolve_repo_path(&rel)
            .ok_or_else(|| format!("{} fixture not found: {rel}", suite.as_str()))?;
        let v = read_first_jsonl_object(&path)?;
        out.push(niah_item_from_value(suite, id, name, &rel, &v)?);
    }
    Ok(out)
}

/// Niah: single/multi needle-in-haystack from `benchmarks/longctx/niah/`.
pub(crate) fn niah_materialized_items(item_ids: &[String]) -> Result<Vec<LongCtxItem>, String> {
    niah_family_items(SuiteId::Niah, "longctx/niah", item_ids)
}

/// SequentialNiah: ordered multi-needle from `benchmarks/longctx/seqniah/`.
pub(crate) fn sequential_niah_materialized_items(
    item_ids: &[String],
) -> Result<Vec<LongCtxItem>, String> {
    niah_family_items(SuiteId::SequentialNiah, "longctx/seqniah", item_ids)
}

/// Ruler: vendored generated slices (S-NIAH + variable-tracking) from
/// `benchmarks/longctx/ruler/`, in the NIAH multi-needle schema.
pub(crate) fn ruler_materialized_items(item_ids: &[String]) -> Result<Vec<LongCtxItem>, String> {
    niah_family_items(SuiteId::Ruler, "longctx/ruler", item_ids)
}

/// Locate the parquet shard for a k-chain (e.g. `k5` → `data/k5-00000-of-00001.parquet`)
/// anywhere under the fetched cache dir.
fn find_needlechain_parquet(cache_path: &Path, kchain: &str) -> Result<PathBuf, String> {
    fn walk(dir: &Path, kchain: &str, depth: usize, out: &mut Option<PathBuf>) {
        if depth > 4 || out.is_some() {
            return;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, kchain, depth + 1, out);
            } else if let Some(name) = p.file_name().and_then(OsStr::to_str) {
                if name.starts_with(&format!("{kchain}-")) && name.ends_with(".parquet") {
                    *out = Some(p.clone());
                    return;
                }
            }
        }
    }
    let mut found = None;
    walk(cache_path, kchain, 0, &mut found);
    found.ok_or_else(|| {
        format!(
            "NeedleChain parquet {kchain}-*.parquet not found under {}",
            cache_path.display()
        )
    })
}

/// Field value from a parquet record row as `String`.
fn parquet_field_str(f: &parquet::record::Field) -> Option<String> {
    match f {
        parquet::record::Field::Str(s) => Some(s.clone()),
        _ => None,
    }
}

/// Field value from a parquet record row as `f64` (Double/Float/Int/Long).
fn parquet_field_f64(f: &parquet::record::Field) -> Option<f64> {
    use parquet::record::Field;
    match f {
        Field::Double(d) => Some(*d),
        Field::Float(x) => Some(*x as f64),
        Field::Long(l) => Some(*l as f64),
        Field::Int(i) => Some(*i as f64),
        _ => None,
    }
}

/// Read one row of a parquet file into a name→field map (record API, no arrow).
fn read_parquet_row(
    path: &Path,
    row_idx: usize,
) -> Result<std::collections::BTreeMap<String, parquet::record::Field>, String> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let file = fs::File::open(path).map_err(|e| format!("open parquet {}: {e}", path.display()))?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| format!("read parquet {}: {e}", path.display()))?;
    let mut iter = reader
        .get_row_iter(None)
        .map_err(|e| format!("row iter {}: {e}", path.display()))?;
    let row = iter
        .nth(row_idx)
        .ok_or_else(|| format!("parquet {} row {row_idx} out of range", path.display()))?
        .map_err(|e| format!("parquet {} row {row_idx}: {e}", path.display()))?;
    let mut map = std::collections::BTreeMap::new();
    for (name, field) in row.get_column_iter() {
        map.insert(name.clone(), field.clone());
    }
    Ok(map)
}

/// Accepted string forms of a numeric answer (plain, thousands-grouped, one
/// decimal) so substring-recall matches common model formatting.
fn numeric_answer_variants(v: f64) -> Vec<String> {
    let mut out = Vec::new();
    if v.fract() == 0.0 {
        let i = v as i64;
        out.push(i.to_string());
        // thousands separators: 17600 -> 17,600
        let digits = i.abs().to_string();
        let mut grouped = String::new();
        for (n, ch) in digits.chars().rev().enumerate() {
            if n > 0 && n % 3 == 0 {
                grouped.push(',');
            }
            grouped.push(ch);
        }
        let grouped: String = grouped.chars().rev().collect();
        out.push(if i < 0 {
            format!("-{grouped}")
        } else {
            grouped
        });
        out.push(format!("{v:.1}"));
    } else {
        out.push(format!("{v}"));
    }
    out.sort();
    out.dedup();
    out
}

/// NeedleChain: multi-hop numeric reasoning from the HF parquet shards
/// (`hyeonsss/needlechain`). Item id is `k<K>:<ordering>:<row>` where ordering ∈
/// {parallel,forward,backward,chaotic}. The `<ordering>_chain` text is the
/// haystack; the target is the `<ordering>_total_val` sum.
pub(crate) fn needlechain_materialized_items(
    cache_path: &Path,
    item_ids: &[String],
) -> Result<Vec<LongCtxItem>, String> {
    const ORDERINGS: &[&str] = &["parallel", "forward", "backward", "chaotic"];
    let mut out = Vec::new();
    for id in item_ids {
        let parts: Vec<&str> = id.split(':').collect();
        if parts.len() != 3 {
            return Err(format!(
                "bad NeedleChain item id: {id} (want k<K>:<ordering>:<row>)"
            ));
        }
        let (kchain, ordering, row_s) = (parts[0], parts[1], parts[2]);
        if !ORDERINGS.contains(&ordering) {
            return Err(format!("bad NeedleChain ordering {ordering} in {id}"));
        }
        let row_idx: usize = row_s
            .parse()
            .map_err(|_| format!("bad NeedleChain row index in {id}"))?;
        let path = find_needlechain_parquet(cache_path, kchain)?;
        let row = read_parquet_row(&path, row_idx)?;
        let chain = row
            .get(&format!("{ordering}_chain"))
            .and_then(parquet_field_str)
            .ok_or_else(|| format!("NeedleChain row missing {ordering}_chain"))?;
        let total = row
            .get(&format!("{ordering}_total_val"))
            .and_then(parquet_field_f64)
            .ok_or_else(|| format!("NeedleChain row missing {ordering}_total_val"))?;
        let names = row
            .get("names")
            .and_then(parquet_field_str)
            .unwrap_or_default();
        let question = format!(
            "Above is a set of statements about how much money each of these people \
             received or earns: {names}. Work through the statements and compute the \
             combined total amount across all of them. Answer with only the final number."
        );
        let prompt = format!("{chain}\n\n{question}");
        let context_tokens = prompt.len() / 4; // rough token estimate for KV sizing
        out.push(LongCtxItem {
            item_id: id.clone(),
            suite: SuiteId::NeedleChain,
            case_id: "needle_chain_native".to_string(),
            task: format!("needle_chain:{kchain}:{ordering}"),
            prompt,
            expected: numeric_answer_variants(total),
            min_recovered: 1,
            context_tokens,
            dataset_file: format!(
                "{}",
                path.file_name()
                    .and_then(OsStr::to_str)
                    .unwrap_or("needlechain.parquet")
            ),
        });
    }
    Ok(out)
}

/// Parse a `<ctx>k` shard token (e.g. `4k` → 4096) into a token budget.
fn parse_ctx_k(tok: &str) -> Option<usize> {
    let n: usize = tok.trim_end_matches('k').parse().ok()?;
    Some(n * 1024)
}

/// NoLiMa: assemble a needle-in-book test from the HF components
/// (`amodaresi/NoLiMa`: needle templates + haystack books). Item id is
/// `<needle_id>:<test_key>:<ctx>k:<book>`, e.g. `0401:T17_C02:4k:1`. The needle
/// (`{CHAR} lives next to {1}`) is inserted at mid-depth into a book truncated
/// to the context budget; the one-hop question asks which character has been to
/// `{2}` (the place `{1}` is located in). Expected answer = the character name.
pub(crate) fn nolima_materialized_items(
    cache_path: &Path,
    item_ids: &[String],
) -> Result<Vec<LongCtxItem>, String> {
    let needle_set_path = cache_path.join("needlesets/needle_set.json");
    let text = fs::read_to_string(&needle_set_path)
        .map_err(|e| format!("read {}: {e}", needle_set_path.display()))?;
    let needle_set: Value =
        serde_json::from_str(&text).map_err(|e| format!("parse needle_set.json: {e}"))?;
    let needles = needle_set
        .as_array()
        .ok_or("needle_set.json is not a JSON array")?;

    let mut out = Vec::new();
    for id in item_ids {
        let parts: Vec<&str> = id.split(':').collect();
        if parts.len() != 4 {
            return Err(format!(
                "bad NoLiMa item id: {id} (want <needle_id>:<test_key>:<ctx>k:<book>)"
            ));
        }
        let (needle_id, test_key, ctx_tok, book_tok) = (parts[0], parts[1], parts[2], parts[3]);
        let ctx_tokens =
            parse_ctx_k(ctx_tok).ok_or_else(|| format!("bad NoLiMa ctx {ctx_tok} in {id}"))?;

        let needle = needles
            .iter()
            .find(|n| n.get("id").and_then(Value::as_str) == Some(needle_id))
            .ok_or_else(|| format!("NoLiMa needle {needle_id} not found"))?;
        let task_template = needle
            .get("task_template")
            .and_then(Value::as_str)
            .ok_or("needle missing task_template")?;
        let needle_tmpl = needle
            .get("needle")
            .and_then(Value::as_str)
            .ok_or("needle missing needle template")?;
        let question_tmpl = needle
            .get("questions")
            .and_then(|q| q.get("onehop"))
            .and_then(Value::as_str)
            .ok_or("needle missing questions.onehop")?;
        let character_set: Vec<&str> = needle
            .get("character_set")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if character_set.is_empty() {
            return Err(format!("NoLiMa needle {needle_id} has empty character_set"));
        }
        let args: Vec<&str> = needle
            .get("tests")
            .and_then(|t| t.get(test_key))
            .and_then(|t| t.get("input_args"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .ok_or_else(|| format!("NoLiMa test {test_key} not found for needle {needle_id}"))?;
        if args.len() < 2 {
            return Err(format!("NoLiMa test {test_key} has too few input_args"));
        }
        // Character index encoded in the test key suffix (…_C0N).
        let char_idx = test_key
            .rsplit_once('C')
            .and_then(|(_, n)| n.trim_start_matches('0').parse::<usize>().ok())
            .unwrap_or(0);
        let character = character_set[char_idx % character_set.len()];

        let fill = |t: &str| -> String {
            let mut s = t.replace("{CHAR}", character).replace("{1}", args[0]);
            s = s.replace("{2}", args[1]);
            if let Some(a3) = args.get(2) {
                s = s.replace("{3}", a3);
            }
            s
        };
        let needle_sentence = fill(needle_tmpl);
        let question = fill(question_tmpl);

        // Truncate a haystack book to the context budget and insert the needle
        // at mid-depth (char-based to stay UTF-8 safe).
        let book_path = cache_path.join(format!("haystack/rand_shuffle/rand_book_{book_tok}.txt"));
        let book = fs::read_to_string(&book_path)
            .map_err(|e| format!("read NoLiMa book {}: {e}", book_path.display()))?;
        let budget_chars = ctx_tokens.saturating_mul(4);
        let chars: Vec<char> = book.chars().take(budget_chars).collect();
        let mid = chars.len() / 2;
        let head: String = chars[..mid].iter().collect();
        let tail: String = chars[mid..].iter().collect();
        let haystack = format!("{head}\n\n{needle_sentence}\n\n{tail}");
        let prompt = task_template
            .replace("{haystack}", &haystack)
            .replace("{question}", &question);

        out.push(LongCtxItem {
            item_id: id.clone(),
            suite: SuiteId::NoLiMa,
            case_id: "nolima_native".to_string(),
            task: format!("nolima:{needle_id}:{test_key}"),
            prompt,
            expected: vec![character.to_string()],
            min_recovered: 1,
            context_tokens: ctx_tokens,
            dataset_file: format!("needle_set.json#{needle_id}/{test_key}"),
        });
    }
    Ok(out)
}

pub(crate) fn gpqa_csv_paths(cache_path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_gpqa_csv_paths(cache_path, 0, &mut out);
    out.sort();
    out
}

pub(crate) fn collect_gpqa_csv_paths(path: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_gpqa_csv_paths(&p, depth + 1, out);
        } else if p
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| matches!(name, "gpqa_diamond.csv" | "gpqa_main.csv"))
        {
            out.push(p);
        }
    }
}

pub(crate) fn gpqa_materialized_items(
    cache_path: &Path,
    item_ids: &[String],
) -> Result<Vec<GpqaItem>, String> {
    let mut out = Vec::new();
    for id in item_ids {
        let Some((subset, row_idx)) = id.split_once(':') else {
            continue;
        };
        let row_idx: usize = row_idx
            .parse()
            .map_err(|_| format!("invalid GPQA item id row index: {id}"))?;
        let csv_path = gpqa_csv_paths(cache_path)
            .into_iter()
            .find(|p| p.file_stem().and_then(OsStr::to_str) == Some(subset))
            .ok_or_else(|| format!("GPQA subset CSV not found for {subset}"))?;
        out.push(read_gpqa_item(&csv_path, subset, row_idx)?);
    }
    Ok(out)
}

pub(crate) fn read_gpqa_item(
    path: &Path,
    subset: &str,
    row_idx: usize,
) -> Result<GpqaItem, String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| format!("open GPQA CSV {}: {e}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("read GPQA CSV headers: {e}"))?
        .clone();
    let find = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("GPQA CSV missing header {name:?}"))
    };
    let q_col = find("Question")?;
    let correct_col = find("Correct Answer")?;
    let i1_col = find("Incorrect Answer 1")?;
    let i2_col = find("Incorrect Answer 2")?;
    let i3_col = find("Incorrect Answer 3")?;
    let rec_col = headers.iter().position(|h| h == "Record ID");

    for (idx, row) in reader.records().enumerate() {
        let row = row.map_err(|e| format!("read GPQA CSV row: {e}"))?;
        if idx != row_idx {
            continue;
        }
        let question = row.get(q_col).unwrap_or("").trim().to_string();
        let correct_answer = row.get(correct_col).unwrap_or("").trim().to_string();
        let incorrect = [
            row.get(i1_col).unwrap_or("").trim().to_string(),
            row.get(i2_col).unwrap_or("").trim().to_string(),
            row.get(i3_col).unwrap_or("").trim().to_string(),
        ];
        if question.is_empty()
            || correct_answer.is_empty()
            || incorrect.iter().any(String::is_empty)
        {
            return Err(format!(
                "GPQA row {subset}:{row_idx} has empty question/choice"
            ));
        }
        let record_suffix = rec_col
            .and_then(|c| row.get(c))
            .filter(|s| !s.trim().is_empty())
            .map(|s| format!(":{s}"))
            .unwrap_or_default();
        let item_id = format!("{subset}:{row_idx}{record_suffix}");
        return Ok(build_gpqa_item(
            item_id,
            path.file_name()
                .and_then(OsStr::to_str)
                .unwrap_or(subset)
                .to_string(),
            question,
            correct_answer,
            incorrect,
        ));
    }
    Err(format!("GPQA row {subset}:{row_idx} not found"))
}

pub(crate) fn build_gpqa_item(
    item_id: String,
    dataset_file: String,
    question: String,
    correct_answer: String,
    incorrect: [String; 3],
) -> GpqaItem {
    let mut raw_choices = vec![
        (true, correct_answer.clone()),
        (false, incorrect[0].clone()),
        (false, incorrect[1].clone()),
        (false, incorrect[2].clone()),
    ];
    let rotate = (stable_hash_bytes(item_id.as_bytes())
        .bytes()
        .fold(0usize, |acc, b| acc.wrapping_add(b as usize)))
        % raw_choices.len();
    raw_choices.rotate_left(rotate);

    let labels = ["A", "B", "C", "D"];
    let mut choices = Vec::new();
    let mut answer_label = "A".to_string();
    for (idx, (is_correct, answer)) in raw_choices.into_iter().enumerate() {
        let label = labels[idx].to_string();
        if is_correct {
            answer_label = label.clone();
        }
        choices.push((label, answer));
    }

    let mut prompt = String::new();
    prompt.push_str("Answer the following graduate-level science multiple-choice question.\n");
    prompt.push_str("Return only the letter of the correct answer.\n\n");
    prompt.push_str("Question:\n");
    prompt.push_str(question.trim());
    prompt.push_str("\n\nChoices:\n");
    for (label, answer) in &choices {
        prompt.push_str(label);
        prompt.push_str(". ");
        prompt.push_str(answer.trim());
        prompt.push('\n');
    }
    prompt.push_str("\nAnswer:");

    GpqaItem {
        item_id,
        dataset_file,
        prompt,
        correct_answer,
        answer_label,
        choices,
    }
}

pub(crate) fn write_gpqa_prompt_artifact(
    dir: &Path,
    _config: &EvalConfig,
    datasets: &[DatasetManifestEntry],
) -> Result<Option<(String, usize)>, String> {
    let mut rows = Vec::new();
    for d in datasets {
        if d.suite != SuiteId::Gpqa || d.status != EvalStatus::Pass {
            continue;
        }
        match gpqa_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
            Ok(items) => {
                for item in items {
                    rows.push(with_dataset_provenance(
                        json!({
                            "schema": 1,
                            "suite": "gpqa",
                            "item_id": item.item_id,
                            "status": "pass",
                            "dataset_file": item.dataset_file,
                            "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                            "prompt_format": "gpqa_zero_shot_v1",
                            "answer_label": item.answer_label,
                            "answer_hash": stable_hash_bytes(item.correct_answer.as_bytes()),
                            "choices_count": item.choices.len(),
                        }),
                        d,
                    ));
                }
            }
            Err(reason) => {
                for id in &d.selected_item_ids {
                    rows.push(with_dataset_provenance(
                        json!({
                            "schema": 1,
                            "suite": "gpqa",
                            "item_id": id,
                            "status": "skip",
                            "reason": reason.clone(),
                        }),
                        d,
                    ));
                }
            }
        }
    }
    if rows.is_empty() {
        return Ok(None);
    }
    let rel = "artifacts/gpqa_prompts.jsonl";
    let path = dir.join("gpqa_prompts.jsonl");
    let mut f = File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
    for row in &rows {
        serde_json::to_writer(&mut f, row)
            .map_err(|e| format!("serialize GPQA prompt row: {e}"))?;
        f.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(Some((rel.to_string(), rows.len())))
}

pub(crate) fn write_barrage_prompt_artifact(
    dir: &Path,
    datasets: &[DatasetManifestEntry],
) -> Result<Option<(String, usize)>, String> {
    let rows = barrage_prompt_artifact_rows(datasets);
    if rows.is_empty() {
        return Ok(None);
    }
    let rel = "artifacts/barrage_prompts.jsonl";
    let path = dir.join("barrage_prompts.jsonl");
    let mut f = File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
    for row in &rows {
        serde_json::to_writer(&mut f, row)
            .map_err(|e| format!("serialize barrage prompt row: {e}"))?;
        f.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(Some((rel.to_string(), rows.len())))
}

pub(crate) fn with_dataset_provenance(mut row: Value, dataset: &DatasetManifestEntry) -> Value {
    if let Value::Object(ref mut object) = row {
        object.insert("dataset_source".to_string(), json!(dataset.source));
        object.insert("dataset_repo_id".to_string(), json!(dataset.repo_id));
        object.insert("dataset_revision".to_string(), json!(dataset.revision));
        object.insert("dataset_digest".to_string(), json!(dataset.digest));
        object.insert("dataset_license".to_string(), json!(dataset.license));
        object.insert("dataset_cache_path".to_string(), json!(dataset.cache_path));
    }
    row
}

pub(crate) fn barrage_prompt_artifact_rows(datasets: &[DatasetManifestEntry]) -> Vec<Value> {
    let mut rows = Vec::new();
    for d in datasets {
        match d.suite {
            SuiteId::Gpqa if d.status == EvalStatus::Pass => {
                match gpqa_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().map(|item| {
                            with_dataset_provenance(json!({
                                "schema": 1,
                                "suite": "gpqa",
                                "item_id": item.item_id,
                                "status": "pass",
                                "dataset_file": item.dataset_file,
                                "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                                "prompt_format": "gpqa_zero_shot_v1",
                                "answer_label": item.answer_label,
                                "answer_hash": stable_hash_bytes(item.correct_answer.as_bytes()),
                                "choices_count": item.choices.len(),
                            }), d)
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().map(|id| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "gpqa",
                                    "item_id": id,
                                    "status": "skip",
                                    "reason": reason,
                                }),
                                d,
                            )
                        }));
                    }
                }
            }
            SuiteId::LmEvalMicro if d.status == EvalStatus::Pass => {
                match lm_eval_micro_materialized_items(&d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().map(|item| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "lm_eval_micro",
                                    "item_id": item.item_id,
                                    "task": item.task,
                                    "status": "pass",
                                    "dataset_file": "builtin:lm_eval_micro:v1",
                                    "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                                    "prompt_format": "lm_eval_micro_zero_shot_v1",
                                    "answer_label": item.answer_label,
                                    "answer_hash": item.answer_hash,
                                    "choices_count": item.choices_count,
                                }),
                                d,
                            )
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().map(|id| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "lm_eval_micro",
                                    "item_id": id,
                                    "status": "skip",
                                    "reason": reason,
                                }),
                                d,
                            )
                        }));
                    }
                }
            }
            SuiteId::HumanEval if d.status == EvalStatus::Pass => {
                match humaneval_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().map(|item| {
                            let mut row = with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "humaneval",
                                    "item_id": item.item_id,
                                    "task_id": item.task_id,
                                    "status": "pass",
                                    "dataset_file": item.dataset_file,
                                    "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                                    "prompt_format": "humaneval_completion_v1",
                                    "scoring_mode": "execution_only",
                                }),
                                d,
                            );
                            if let Value::Object(ref mut object) = row {
                                if let Some(hash) = item.canonical_solution_hash {
                                    object
                                        .insert("canonical_solution_hash".to_string(), json!(hash));
                                }
                                if let Some(hash) = item.test_hash {
                                    object.insert("test_hash".to_string(), json!(hash));
                                }
                            }
                            row
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().map(|id| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "humaneval",
                                    "item_id": id,
                                    "status": "skip",
                                    "reason": reason,
                                }),
                                d,
                            )
                        }));
                    }
                }
            }
            SuiteId::DeepSwe | SuiteId::SweBench if d.status == EvalStatus::Pass => {
                match builtin_barrage_materialized_items(d.suite, &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().map(|item| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": item.suite.as_str(),
                                    "item_id": item.item_id,
                                    "task": item.task,
                                    "status": "pass",
                                    "dataset_file": item.dataset_file,
                                    "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                                    "prompt_format": item.prompt_format,
                                    "answer_label": item.answer_label,
                                    "answer_hash": item.answer_hash,
                                    "choices_count": item.choices_count,
                                    "scoring_mode": item.scoring_mode,
                                }),
                                d,
                            )
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().map(|id| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": d.suite.as_str(),
                                    "item_id": id,
                                    "status": "skip",
                                    "reason": reason,
                                }),
                                d,
                            )
                        }));
                    }
                }
            }
            _ => {}
        }
    }
    rows
}

pub(crate) fn humaneval_jsonl_paths(cache_path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_humaneval_jsonl_paths(cache_path, 0, &mut out);
    out.sort();
    out
}

pub(crate) fn collect_humaneval_jsonl_paths(path: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_humaneval_jsonl_paths(&p, depth + 1, out);
        } else if p.file_name().and_then(OsStr::to_str).is_some_and(|name| {
            let lower = name.to_ascii_lowercase();
            lower.ends_with(".jsonl") && lower.contains("humaneval")
        }) {
            out.push(p);
        }
    }
}

pub(crate) fn humaneval_materialized_items(
    cache_path: &Path,
    item_ids: &[String],
) -> Result<Vec<HumanEvalItem>, String> {
    let paths = humaneval_jsonl_paths(cache_path);
    if paths.is_empty() {
        return Err("HumanEval JSONL not found".to_string());
    }
    let mut out = Vec::new();
    for id in item_ids {
        let mut found = None;
        for path in &paths {
            if let Some(item) = read_humaneval_item_by_task_id(path, id)? {
                found = Some(item);
                break;
            }
            let row_idx = humaneval_item_row_index(id)?;
            if let Some(item) = read_humaneval_item_by_row(path, row_idx)? {
                found = Some(item);
                break;
            }
        }
        out.push(found.ok_or_else(|| format!("HumanEval row {id} not found"))?);
    }
    Ok(out)
}

pub(crate) fn humaneval_item_row_index(id: &str) -> Result<usize, String> {
    id.rsplit_once('/')
        .map(|(_, idx)| idx)
        .unwrap_or(id)
        .parse()
        .map_err(|_| format!("invalid HumanEval item id row index: {id}"))
}

pub(crate) fn read_humaneval_item_by_task_id(
    path: &Path,
    task_id: &str,
) -> Result<Option<HumanEvalItem>, String> {
    let body = fs::read_to_string(path)
        .map_err(|e| format!("read HumanEval JSONL {}: {e}", path.display()))?;
    for (idx, line) in body.lines().enumerate() {
        let value: Value = serde_json::from_str(line)
            .map_err(|e| format!("parse HumanEval JSONL row {idx}: {e}"))?;
        if value
            .get("task_id")
            .and_then(Value::as_str)
            .is_some_and(|candidate| candidate == task_id)
        {
            return parse_humaneval_item(path, idx, value).map(Some);
        }
    }
    Ok(None)
}

pub(crate) fn read_humaneval_item_by_row(
    path: &Path,
    row_idx: usize,
) -> Result<Option<HumanEvalItem>, String> {
    let body = fs::read_to_string(path)
        .map_err(|e| format!("read HumanEval JSONL {}: {e}", path.display()))?;
    for (idx, line) in body.lines().enumerate() {
        if idx != row_idx {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|e| format!("parse HumanEval JSONL row {row_idx}: {e}"))?;
        return parse_humaneval_item(path, row_idx, value).map(Some);
    }
    Ok(None)
}

pub(crate) fn parse_humaneval_item(
    path: &Path,
    row_idx: usize,
    value: Value,
) -> Result<HumanEvalItem, String> {
    let task_id = value
        .get("task_id")
        .and_then(Value::as_str)
        .unwrap_or("HumanEval/unknown")
        .to_string();
    let prompt = value
        .get("prompt")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("HumanEval row {row_idx} missing prompt"))?
        .to_string();
    if prompt.trim().is_empty() {
        return Err(format!("HumanEval row {row_idx} has empty prompt"));
    }
    let canonical_solution_hash = value
        .get("canonical_solution")
        .and_then(Value::as_str)
        .map(|s| stable_hash_bytes(s.as_bytes()));
    let test_hash = value
        .get("test")
        .and_then(Value::as_str)
        .map(|s| stable_hash_bytes(s.as_bytes()));
    Ok(HumanEvalItem {
        item_id: task_id.clone(),
        task_id,
        dataset_file: path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("HumanEval.jsonl")
            .to_string(),
        prompt,
        canonical_solution_hash,
        test_hash,
    })
}

#[cfg(test)]
mod longctx_tests {
    use super::*;

    #[test]
    fn niah_single_needle_materializes_from_fixture() {
        let items = niah_materialized_items(&["niah_8k:0".to_string()])
            .expect("niah_8k fixture should materialize");
        assert_eq!(items.len(), 1);
        let it = &items[0];
        assert_eq!(it.suite, SuiteId::Niah);
        assert_eq!(it.expected.len(), 1);
        assert_eq!(it.min_recovered, 1);
        // Prompt is haystack + question; the expected answer is embedded in it.
        assert!(it.prompt.contains(&it.expected[0]));
        assert!(it.prompt.contains('?'));
        assert!(it.context_tokens >= 4096);
    }

    #[test]
    fn niah_multi_needle_carries_threshold() {
        let items = niah_materialized_items(&["niah_multi_16k:0".to_string()])
            .expect("niah_multi_16k fixture should materialize");
        let it = &items[0];
        assert!(it.expected.len() >= 2);
        // Multi fixtures set an explicit recovery threshold below the full count.
        assert!(it.min_recovered >= 1 && it.min_recovered <= it.expected.len());
    }

    #[test]
    fn sequential_niah_materializes_ordered_chain() {
        let items = sequential_niah_materialized_items(&["seqniah_8k:0".to_string()])
            .expect("seqniah_8k fixture should materialize");
        let it = &items[0];
        assert_eq!(it.suite, SuiteId::SequentialNiah);
        // Sequential chain: multiple ordered steps, full-chain recovery required.
        assert!(it.expected.len() >= 3);
        assert_eq!(it.min_recovered, it.expected.len());
        for secret in &it.expected {
            assert!(it.prompt.contains(secret));
        }
    }

    #[test]
    fn ruler_materializes_niah_and_vt() {
        let niah = ruler_materialized_items(&["ruler_niah_4k:0".to_string()])
            .expect("ruler_niah_4k should materialize");
        assert_eq!(niah[0].suite, SuiteId::Ruler);
        assert_eq!(niah[0].expected.len(), 1);
        assert!(niah[0].prompt.contains(&niah[0].expected[0]));
        let vt = ruler_materialized_items(&["ruler_vt_4k:0".to_string()])
            .expect("ruler_vt_4k should materialize");
        // Variable tracking: multiple variables, full-chain recall required.
        assert!(vt[0].expected.len() >= 3);
        assert_eq!(vt[0].min_recovered, vt[0].expected.len());
    }

    #[test]
    fn niah_local_dataset_entry_is_available() {
        let entry = local_longctx_dataset_entry(SuiteId::Niah);
        assert!(matches!(entry.status, EvalStatus::Pass));
        assert!(!entry.selected_item_ids.is_empty());
        assert!(entry.files.iter().all(|f| f.ends_with(".jsonl")));
    }

    #[test]
    fn numeric_answer_variants_cover_common_formats() {
        let v = numeric_answer_variants(17600.0);
        assert!(v.contains(&"17600".to_string()));
        assert!(v.contains(&"17,600".to_string()));
        assert!(v.contains(&"17600.0".to_string()));
        // Non-integral keeps the plain form.
        assert!(numeric_answer_variants(12.5).contains(&"12.5".to_string()));
    }

    // Fetch-dependent parser check. Point HIPFIRE_TEST_NEEDLECHAIN_CACHE at a
    // dir containing data/k5-*.parquet (the fetched HF layout) and run:
    //   cargo test -p hipfire-eval needlechain_parses_real_parquet -- --ignored
    #[test]
    #[ignore]
    fn needlechain_parses_real_parquet() {
        let cache = std::env::var("HIPFIRE_TEST_NEEDLECHAIN_CACHE")
            .expect("set HIPFIRE_TEST_NEEDLECHAIN_CACHE to the fetched cache dir");
        let items =
            needlechain_materialized_items(Path::new(&cache), &["k5:forward:0".to_string()])
                .expect("materialize k5:forward:0");
        let it = &items[0];
        assert_eq!(it.suite, SuiteId::NeedleChain);
        assert!(!it.prompt.is_empty());
        assert!(!it.expected.is_empty());
        assert_eq!(it.min_recovered, 1);
        // The chain text and the compute-the-total question are both present.
        assert!(it.prompt.contains("total"));
    }

    // Fetch-dependent assembly check. Point HIPFIRE_TEST_NOLIMA_CACHE at a dir
    // containing needlesets/needle_set.json + haystack/rand_shuffle/rand_book_*.txt
    // and run:
    //   cargo test -p hipfire-eval nolima_assembles_from_components -- --ignored
    #[test]
    #[ignore]
    fn nolima_assembles_from_components() {
        let cache = std::env::var("HIPFIRE_TEST_NOLIMA_CACHE")
            .expect("set HIPFIRE_TEST_NOLIMA_CACHE to the fetched cache dir");
        let items =
            nolima_materialized_items(Path::new(&cache), &["0401:T17_C02:4k:1".to_string()])
                .expect("materialize NoLiMa item");
        let it = &items[0];
        assert_eq!(it.suite, SuiteId::NoLiMa);
        assert_eq!(it.expected.len(), 1);
        assert_eq!(it.context_tokens, 4096);
        // The expected character answer and the assembled question are present.
        assert!(it.prompt.contains(&it.expected[0]));
        assert!(it.prompt.contains("Question:"));
        assert!(it.prompt.len() > 4000);
    }
}
