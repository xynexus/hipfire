//! Unit tests for the `generate_batch_prefill` / `generate_batch_decode_step`
//! planning and validation helpers.
//!
//! This is a `#[cfg(test)]` *file module* of the daemon binary, not an
//! integration test under `tests/`: the helpers under test are private to the
//! bin crate, so `tests/` could not reach them. It lived inline in `main.rs`
//! until it grew to ~2.1k lines sitting between the free helpers and `main()`,
//! straddling every module boundary the daemon split needs to cut.
//!
//! `use super::*` resolves to the crate root (`main.rs`), so nothing about the
//! test bodies changed in the move.

use super::*;
use hipfire_arch_qwen35::qwen35::LayerType;
use hipfire_generate::{
    build_qwen35_fused_dense_prefill_batch_contract, compute_qwen35_prefix_hash,
    plan_generate_batch_prefill_qwen35, qwen35_decode_batch_scheduler_metadata,
    qwen35_fused_prefill_boundary_cuts, qwen35_grouped_moe_decode_auto_latency_gate_passed,
    qwen35_prefill_scratch_target_batch, select_qwen35_decode_batch_backend,
    select_qwen35_prefill_batch_backend, validate_qwen35_fused_grouped_moe_prefill_batch_preflight,
    GenerateBatchPrefillPlan, Qwen35DecodeBatchBackend, Qwen35FusedDensePrefillInputKind,
    Qwen35PrefillBatchBackend, Qwen35PreparedPrefillSession, Qwen35SemanticBoundaryCheckpoint,
};
use hipfire_model::ModelWorkerId;
use hipfire_state::{
    describe_sequence_state_descriptors, parsed_handle_may_target_loaded_state,
    qwen35_sequence_state_handle, ModelWorkerMemoryView, ModelWorkerRuntimeView,
    ParsedSequenceStateHandle, SequenceStateArenaBackend, SequenceStatePageDescriptor,
    SequenceStatePageKind, SequenceStatePrefixHash,
};
fn test_dense_qwen35_config() -> qwen35::Qwen35Config {
    qwen35::Qwen35Config {
        dim: 16,
        n_layers: 2,
        vocab_size: 32,
        norm_eps: 1e-6,
        eos_token: 0,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 8,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 0.25,
        attn_output_gate: true,
        is_vl_text: false,
        mrope_interleaved: false,
        mrope_section: [0, 0, 0],
        linear_num_key_heads: 2,
        linear_num_value_heads: 2,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        conv_kernel_dim: 4,
        hidden_dim: 32,
        num_experts: 0,
        num_experts_per_tok: 0,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        has_shared_expert: false,
        norm_topk_prob: false,
        layer_types: vec![LayerType::FullAttention, LayerType::LinearAttention],
        paged_experts: false,
        vram_budget_bytes: u64::MAX,
    }
}

fn test_grouped_moe_qwen35_config() -> qwen35::Qwen35Config {
    qwen35::Qwen35Config {
        num_experts: 4,
        num_experts_per_tok: 8,
        moe_intermediate_size: 16,
        shared_expert_intermediate_size: 16,
        has_shared_expert: true,
        norm_topk_prob: true,
        ..test_dense_qwen35_config()
    }
}

fn fp32_decode_state_signature() -> qwen35::DensePrefillSessionBatchStateSignature {
    qwen35::DensePrefillSessionBatchStateSignature {
        kv_physical_cap: 128,
        kv_compact_offset: 0,
        kv_quantized: false,
        kv_quant_q8: false,
        kv_quant_asym2: false,
        kv_quant_asym3: false,
        kv_quant_asym4: false,
        kv_quant_fwht: false,
        dn_quant: qwen35::StateQuant::FP32,
    }
}

fn q8_decode_state_signature() -> qwen35::DensePrefillSessionBatchStateSignature {
    qwen35::DensePrefillSessionBatchStateSignature {
        kv_quantized: true,
        kv_quant_q8: true,
        dn_quant: qwen35::StateQuant::Q8,
        ..fp32_decode_state_signature()
    }
}

fn u32_payload(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_payload(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn write_tiny_hfqm_kldref(path: &std::path::Path, arch_id: u32) {
    let metadata = serde_json::json!({
        "schema": 1,
        "artifact_kind": "hipfire.kldref",
        "package_schema": "hipfire.kldref.v1",
        "base_model_id": "tiny-qwen35",
        "source_model_sha256": "abc",
        "arch_id": arch_id,
        "n_vocab": 16,
        "n_ctx": 4,
        "n_chunk": 1,
        "scored_per_chunk": 2,
        "scoring_start": 2,
        "top_k": 1,
        "total_scored": 2,
        "slice": "tiny.txt",
        "slice_md5": "md5",
        "kv_mode": "Fp32",
        "kld_graph_prefill": true,
        "kld_graph_prefill_max_batch": "4",
        "hipfire_version": "test",
        "gpu_arch": "gfx-test"
    })
    .to_string();
    let tensors = vec![
        hipfire_runtime::hfq::HfqMemTensor {
            name: "kldref.tokens".to_string(),
            quant_type: 0,
            shape: vec![1, 4],
            group_size: 0,
            data: u32_payload(&[1, 2, 3, 4]),
        },
        hipfire_runtime::hfq::HfqMemTensor {
            name: "kldref.top_indices".to_string(),
            quant_type: 0,
            shape: vec![1, 2, 1],
            group_size: 0,
            data: u32_payload(&[3, 4]),
        },
        hipfire_runtime::hfq::HfqMemTensor {
            name: "kldref.top_log_probs".to_string(),
            quant_type: 0,
            shape: vec![1, 2, 1],
            group_size: 0,
            data: f32_payload(&[-0.1, -0.2]),
        },
        hipfire_runtime::hfq::HfqMemTensor {
            name: "kldref.residual_mass".to_string(),
            quant_type: 0,
            shape: vec![1, 2],
            group_size: 0,
            data: f32_payload(&[0.01, 0.02]),
        },
    ];
    hipfire_runtime::hfq::write_hfqm_package_mem(path, arch_id, &metadata, &tensors).unwrap();
}

#[test]
fn daemon_reads_hfqm_kldref_with_parent_arch_id() {
    let path = std::env::temp_dir().join(format!(
        "hipfire-daemon-kldref-{}-parent.hfq",
        std::process::id()
    ));
    write_tiny_hfqm_kldref(&path, 5);
    let archive = read_kld_ref_archive(&path).unwrap();
    assert_eq!(archive.meta.arch_id, 5);
    assert_eq!(archive.meta.config.kv_mode, "fp32");
    assert!(archive.meta.config.graph);
    assert_eq!(archive.meta.config.prefill_max_batch, Some(4));
    assert_eq!(archive.tokens, vec![1, 2, 3, 4]);
    assert_eq!(archive.top_indices, vec![3, 4]);
    assert_eq!(archive.top_log_probs, vec![-0.1, -0.2]);
    assert_eq!(archive.residual_mass, vec![0.01, 0.02]);
    let _ = std::fs::remove_file(path);
}

#[test]
fn daemon_rejects_hfqm_kldref_arch_zero() {
    let path = std::env::temp_dir().join(format!(
        "hipfire-daemon-kldref-{}-arch0.hfq",
        std::process::id()
    ));
    write_tiny_hfqm_kldref(&path, hipfire_runtime::hfq::HFQM_ARCH_NON_WEIGHT_PACKAGE);
    let err = read_kld_ref_archive(&path).unwrap_err();
    assert!(err.to_string().contains("arch_id is 0"));
    let _ = std::fs::remove_file(path);
}

#[test]
fn validates_minimal_prompt_envelope() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-1",
        "batch_id": "batch-1",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0
            },
            "params": {
                "max_tokens": 8,
                "temperature": 0.0
            }
        }]
    });

    let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
    assert_eq!(envelope.id, "probe-1");
    assert_eq!(envelope.batch_id, "batch-1");
    assert_eq!(envelope.session_count, 1);
    assert!(!envelope.sessions[0].semantic_boundary_checkpoints);
}

#[test]
fn validates_suffix_token_envelope_with_model_identity() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "batch_id": "batch-2",
        "model": "qwen3.5:9b",
        "sessions": [{
            "id": "req-1",
            "suffix_tokens": [1, 2, 3],
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 0
            }
        }]
    });

    let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
    assert_eq!(envelope.id, "0");
    assert_eq!(envelope.batch_id, "batch-2");
    assert_eq!(envelope.session_count, 1);
}

#[test]
fn validates_runtime_state_handle_for_attachable_prefix() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "batch_id": "batch-attach",
        "model": "qwen3.5:9b",
        "sessions": [{
            "id": "req-1",
            "suffix_tokens": [4, 5],
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 12,
                "cached_prefix_tokens": 12,
                "runtime_state_handle": "qwen35-checkpoint:req-0"
            }
        }]
    });

    let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
    assert_eq!(
        envelope.sessions[0]
            .state_handle
            .runtime_state_handle
            .as_deref(),
        Some("qwen35-checkpoint:req-0")
    );
}

#[test]
fn validates_runtime_prefix_hash_for_attachable_prefix() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "batch_id": "batch-1",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "suffix_tokens": [4, 5],
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 3,
                "cached_prefix_tokens": 3,
                "runtime_state_handle": "qwen35-checkpoint:req-0",
                "prefix_hash": {
                    "algorithm": "xxh128",
                    "value": "0123456789abcdef0123456789abcdef",
                    "prefix_len": 3
                }
            }
        }]
    });

    let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
    let hash = envelope.sessions[0]
        .state_handle
        .prefix_hash
        .as_ref()
        .expect("prefix hash parsed");
    assert_eq!(hash.algorithm, "xxh128");
    assert_eq!(hash.prefix_len, 3);
}

#[test]
fn rejects_malformed_runtime_prefix_hash() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "batch_id": "batch-1",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "suffix_tokens": [4, 5],
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 3,
                "cached_prefix_tokens": 3,
                "runtime_state_handle": "qwen35-checkpoint:req-0",
                "prefix_hash": {
                    "algorithm": "xxh128",
                    "value": "ABC",
                    "prefix_len": 3
                }
            }
        }]
    });

    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains("32 lowercase hex"));
}

#[test]
fn validates_prefix_hash_preflight_envelope() {
    let msg = serde_json::json!({
        "type": "prefix_hash_preflight",
        "id": "prefix-1",
        "worker_key_id": "worker-a",
        "boundary_policy": "semantic_chat_template",
        "session": {
            "id": "req-1",
            "prompt": "hello",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"}
            ],
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 0
            },
            "params": {
                "assistant_prefix": "open_think",
                "max_think_tokens": 16
            }
        }
    });

    let envelope = validate_prefix_hash_preflight(&msg).expect("valid preflight");
    assert_eq!(envelope.id, "prefix-1");
    assert_eq!(envelope.boundary_policy, "semantic_chat_template");
    assert_eq!(envelope.session.id, "req-1");
    assert_eq!(envelope.session.messages_history.as_ref().unwrap().len(), 2);
    assert_eq!(envelope.session.assistant_prefix, "open_think");
    assert_eq!(envelope.session.max_think_tokens, 16);
}

#[test]
fn validates_lfm2_prefix_hash_preflight_envelope() {
    let msg = serde_json::json!({
        "type": "prefix_hash_preflight",
        "id": "lfm2-prefix-1",
        "worker_key_id": "lfm2-worker",
        "boundary_policy": "semantic_chat_template",
        "session": {
            "id": "lfm2-req-1",
            "prompt": "hello",
            "messages": [
                {"role": "user", "content": "hello"}
            ],
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0
            },
            "params": {
                "assistant_prefix": "plain"
            }
        }
    });

    let envelope = validate_prefix_hash_preflight(&msg).expect("valid LFM2 preflight");
    assert_eq!(envelope.id, "lfm2-prefix-1");
    assert_eq!(envelope.boundary_policy, "semantic_chat_template");
    assert_eq!(envelope.session.id, "lfm2-req-1");
    assert_eq!(
        envelope.session.state_handle.state_kinds,
        vec!["attention_kv".to_string()]
    );
    assert_eq!(envelope.session.assistant_prefix, "plain");
}

#[test]
fn validates_generate_batch_decode_step_envelope() {
    let msg = serde_json::json!({
        "type": "generate_batch_decode_step",
        "id": "decode-1",
        "batch_id": "decode-batch-1",
        "worker_key_id": "worker-a",
        "cached_prefix_tokens": 12,
        "sessions": [
            {
                "id": "req-1",
                "session_id": "qwen35-checkpoint:batch:req-1:8",
                "logical_position": 8,
                "max_tokens_remaining": 4
            },
            {
                "id": "req-2",
                "session_id": "qwen35-checkpoint:batch:req-2:8",
                "logical_position": 8,
                "max_tokens_remaining": 3
            }
        ]
    });

    let envelope = validate_generate_batch_decode(&msg).expect("valid decode envelope");
    assert_eq!(envelope.id, "decode-1");
    assert_eq!(envelope.batch_id, "decode-batch-1");
    assert_eq!(envelope.session_count, 2);
    assert_eq!(envelope.cached_prefix_tokens, 12);
    assert_eq!(
        envelope.sessions[0].session_id,
        "qwen35-checkpoint:batch:req-1:8"
    );
    assert_eq!(envelope.sessions[1].max_tokens_remaining, 3);
}

#[test]
fn rejects_invalid_generate_batch_decode_step_envelope() {
    let missing_worker = serde_json::json!({
        "type": "generate_batch_decode_step",
        "batch_id": "decode-batch-1",
        "sessions": [{
            "id": "req-1",
            "session_id": "runtime-1",
            "logical_position": 8,
            "max_tokens_remaining": 1
        }]
    });
    let err = validate_generate_batch_decode(&missing_worker).unwrap_err();
    assert!(err.contains("worker_key_id or model"));

    let zero_remaining = serde_json::json!({
        "type": "generate_batch_decode_step",
        "batch_id": "decode-batch-1",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "session_id": "runtime-1",
            "logical_position": 8,
            "max_tokens_remaining": 0
        }]
    });
    let err = validate_generate_batch_decode(&zero_remaining).unwrap_err();
    assert!(err.contains("max_tokens_remaining"));

    let invalid_cached_prefix = serde_json::json!({
        "type": "generate_batch_decode_step",
        "batch_id": "decode-batch-1",
        "worker_key_id": "worker-a",
        "cached_prefix_tokens": -1,
        "sessions": [{
            "id": "req-1",
            "session_id": "runtime-1",
            "logical_position": 8,
            "max_tokens_remaining": 1
        }]
    });
    let err = validate_generate_batch_decode(&invalid_cached_prefix).unwrap_err();
    assert!(err.contains("cached_prefix_tokens"));
}

#[test]
fn rejects_prefix_hash_preflight_suffix_or_unknown_policy() {
    let suffix = serde_json::json!({
        "type": "prefix_hash_preflight",
        "id": "prefix-1",
        "worker_key_id": "worker-a",
        "session": {
            "id": "req-1",
            "suffix_tokens": [1, 2, 3],
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 0
            }
        }
    });
    let err = match validate_prefix_hash_preflight(&suffix) {
        Ok(_) => panic!("suffix preflight unexpectedly accepted"),
        Err(err) => err,
    };
    assert!(err.contains("must include prompt"));

    let bad_policy = serde_json::json!({
        "type": "prefix_hash_preflight",
        "id": "prefix-1",
        "worker_key_id": "worker-a",
        "boundary_policy": "every_token",
        "session": {
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 0
            }
        }
    });
    let err = match validate_prefix_hash_preflight(&bad_policy) {
        Ok(_) => panic!("bad boundary policy unexpectedly accepted"),
        Err(err) => err,
    };
    assert!(err.contains("semantic_chat_template"));
}

#[test]
fn qwen35_prefix_hash_is_domain_separated() {
    let kinds = vec!["attention_kv".to_string(), "deltanet_recurrent".to_string()];
    let reordered = vec!["deltanet_recurrent".to_string(), "attention_kv".to_string()];
    let base = compute_qwen35_prefix_hash(5, Some("q8"), &kinds, "plain", 0, &[1, 2, 3]);
    let same = compute_qwen35_prefix_hash(5, Some("q8"), &reordered, "plain", 0, &[1, 2, 3]);
    let different_tokens =
        compute_qwen35_prefix_hash(5, Some("q8"), &kinds, "plain", 0, &[1, 2, 4]);
    let different_think =
        compute_qwen35_prefix_hash(5, Some("q8"), &kinds, "open_think", 0, &[1, 2, 3]);

    assert_eq!(base, same);
    assert_eq!(base.algorithm, "xxh128");
    assert_eq!(base.value.len(), 32);
    assert_eq!(base.prefix_len, 3);
    assert_ne!(base, different_tokens);
    assert_ne!(base, different_think);
}

#[test]
fn lfm2_prefix_hash_uses_lfm2_kv_domain() {
    let kinds = vec!["attention_kv".to_string()];
    let lfm2 = compute_qwen35_prefix_hash(
        ARCH_ID_LFM2_MOE,
        Some("lfm2_q8_kv"),
        &kinds,
        "plain",
        0,
        &[1, 2, 3],
    );
    let lfm2_other_kv =
        compute_qwen35_prefix_hash(ARCH_ID_LFM2_MOE, Some("q8"), &kinds, "plain", 0, &[1, 2, 3]);
    let qwen_same_label =
        compute_qwen35_prefix_hash(5, Some("lfm2_q8_kv"), &kinds, "plain", 0, &[1, 2, 3]);

    assert_eq!(lfm2.algorithm, "xxh128");
    assert_eq!(lfm2.value.len(), 32);
    assert_eq!(lfm2.prefix_len, 3);
    assert_ne!(lfm2, lfm2_other_kv);
    assert_ne!(lfm2, qwen_same_label);
}

fn test_prefill_boundary_session(
    id: &str,
    tokens: &[u32],
    boundaries: &[usize],
) -> Qwen35PreparedPrefillSession {
    Qwen35PreparedPrefillSession {
        id: id.to_string(),
        tokens: tokens.to_vec(),
        cached_prefix_tokens: 0,
        replay_as_generated_suffix: false,
        state_kinds: vec!["attention_kv".to_string(), "deltanet_recurrent".to_string()],
        assistant_prefix: "plain".to_string(),
        max_think_tokens: 0,
        boundary_checkpoints: boundaries
            .iter()
            .enumerate()
            .map(
                |(boundary_index, &prefix_len)| Qwen35SemanticBoundaryCheckpoint {
                    checkpoint_id: None,
                    prefix_len,
                    hash: SequenceStatePrefixHash {
                        algorithm: "xxh128".to_string(),
                        value: format!("{prefix_len:032x}"),
                        prefix_len,
                    },
                    boundary: "message_end".to_string(),
                    boundary_index,
                },
            )
            .collect(),
    }
}

#[test]
fn fused_prefill_boundary_cuts_allow_single_session_serial_segments() {
    let synchronized = vec![
        test_prefill_boundary_session("req-a", &[1, 2, 3, 4], &[2]),
        test_prefill_boundary_session("req-b", &[5, 6, 7, 8], &[2]),
    ];
    assert_eq!(
        qwen35_fused_prefill_boundary_cuts(&synchronized).expect("valid cuts"),
        Some(vec![2, 4])
    );

    let uneven_tail = vec![
        test_prefill_boundary_session("req-a", &[1], &[]),
        test_prefill_boundary_session("req-b", &[5, 6, 7, 8], &[1]),
    ];
    assert_eq!(
        qwen35_fused_prefill_boundary_cuts(&uneven_tail).expect("valid mixed cuts"),
        Some(vec![1, 4])
    );
}

#[test]
fn fused_prefill_boundary_cuts_cover_multiple_boundaries_and_suffix_replay_fallback() {
    let multi_cut = vec![
        test_prefill_boundary_session("req-a", &[1, 2, 3, 4], &[1, 3]),
        test_prefill_boundary_session("req-b", &[5, 6, 7, 8], &[1, 3]),
    ];
    assert_eq!(
        qwen35_fused_prefill_boundary_cuts(&multi_cut).expect("valid multi-cut layout"),
        Some(vec![1, 3, 4])
    );

    let mut replay = test_prefill_boundary_session("req-a", &[1, 2, 3, 4], &[2]);
    replay.replay_as_generated_suffix = true;
    let err = qwen35_fused_prefill_boundary_cuts(&[replay]).unwrap_err();
    assert!(err.contains("only supported for full-prompt prefill"));
}

#[test]
fn model_worker_runtime_view_json_reports_state_page_descriptors() {
    let worker = ModelWorkerRuntimeView {
        worker_id: ModelWorkerId {
            value: "qwen35:worker-a".to_string(),
        },
        max_seq: 32768,
        physical_cap: 2048,
        max_resident_workers: 1,
        resident_workers: 1,
        state_arena_backend: SequenceStateArenaBackend::Qwen35Wrapped,
        state_arena_operations: SequenceStateArenaBackend::Qwen35Wrapped
            .supported_operations()
            .to_vec(),
        resident_sessions: 1,
        state_page_descriptors: vec![
            SequenceStatePageDescriptor {
                session_id: "checkpoint-a".to_string(),
                handle: SequenceStateHandle {
                    id: "checkpoint-a".to_string(),
                    kind: "qwen35_session".to_string(),
                    generation: 0,
                },
                kind: SequenceStatePageKind::Kv,
                label: "qwen35.kv_cache".to_string(),
                logical_position: 16,
                resident_bytes: 128,
                allocation_epoch: 0,
                owns_pages: false,
                shape: vec![2, 16, 4, 32],
                placement: "hip:arch5:device0".to_string(),
                role: "resident".to_string(),
            },
            SequenceStatePageDescriptor {
                session_id: "checkpoint-a".to_string(),
                handle: SequenceStateHandle {
                    id: "checkpoint-a".to_string(),
                    kind: "qwen35_session".to_string(),
                    generation: 0,
                },
                kind: SequenceStatePageKind::DeltaNet,
                label: "qwen35.deltanet_state".to_string(),
                logical_position: 16,
                resident_bytes: 96,
                allocation_epoch: 0,
                owns_pages: false,
                shape: vec![48, 48, 48],
                placement: "hip:arch5:device0".to_string(),
                role: "resident".to_string(),
            },
            SequenceStatePageDescriptor {
                session_id: "checkpoint-a".to_string(),
                handle: SequenceStateHandle {
                    id: "checkpoint-a".to_string(),
                    kind: "qwen35_session".to_string(),
                    generation: 0,
                },
                kind: SequenceStatePageKind::Logits,
                label: "qwen35.logits_snapshot".to_string(),
                logical_position: 16,
                resident_bytes: 64,
                allocation_epoch: 0,
                owns_pages: false,
                shape: vec![16],
                placement: "hip:arch5:device0".to_string(),
                role: "resident".to_string(),
            },
            SequenceStatePageDescriptor {
                session_id: "checkpoint-a".to_string(),
                handle: SequenceStateHandle {
                    id: "checkpoint-a".to_string(),
                    kind: "qwen35_session".to_string(),
                    generation: 0,
                },
                kind: SequenceStatePageKind::BackendPrivate,
                label: "qwen35.prefix_metadata".to_string(),
                logical_position: 16,
                resident_bytes: 32,
                allocation_epoch: 0,
                owns_pages: false,
                shape: vec![1],
                placement: "host".to_string(),
                role: "resident".to_string(),
            },
        ],
        memory: ModelWorkerMemoryView {
            model_file_bytes: 256,
            model_weight_bytes: 224,
            runtime_base_bytes: 64,
            runtime_session_bytes: 320,
            runtime_state_bytes: 384,
            total_resident_bytes: 608,
            evictable_state_bytes: 320,
        },
    };

    let json = model_worker_runtime_view_json(&worker);
    assert_eq!(json["state_arena_backend"], "qwen35_wrapped");
    assert_eq!(json["state_arena_owns_pages"], true);
    assert_eq!(json["state_allocator"]["page_ownership"], "backend_wrapped");
    assert_eq!(
        json["state_allocator"]["eviction_policy"],
        "manual_release_only"
    );
    assert_eq!(json["state_allocator"]["spill_target"], "disabled");
    assert_eq!(json["state_allocator"]["copy_on_write_attach"], false);
    assert_eq!(
        json["state_arena_operations"],
        serde_json::json!([
            "reserve_session_state",
            "attach_checkpoint",
            "fork_checkpoint",
            "release_state",
            "describe_state"
        ])
    );
    assert_eq!(json["max_seq"], 32768);
    assert_eq!(json["physical_cap"], 2048);
    assert_eq!(json["state_page_descriptor_entries"], 4);
    assert_eq!(json["state_page_descriptor_bytes"], 320);
    assert_eq!(json["model_file_bytes"], 256);
    assert_eq!(json["model_weight_bytes"], 224);
    assert_eq!(json["runtime_base_bytes"], 64);
    assert_eq!(json["runtime_session_bytes"], 320);
    assert_eq!(json["runtime_state_bytes"], 384);
    assert_eq!(json["total_resident_bytes"], 608);
    assert_eq!(json["evictable_state_bytes"], 320);
    assert_eq!(
        json["state_page_descriptors"][0]["state_kind"],
        "attention_kv"
    );
    assert_eq!(
        json["state_page_descriptors"][0]["page_kind"],
        "attention_kv"
    );
    assert_eq!(
        json["state_page_descriptors"][0]["handle"]["id"],
        "checkpoint-a"
    );
    assert_eq!(json["state_page_descriptors"][0]["allocation_epoch"], 0);
    assert_eq!(json["state_page_descriptors"][0]["owns_pages"], false);
    assert_eq!(
        json["state_page_descriptors"][0]["shape"],
        serde_json::json!([2, 16, 4, 32])
    );
    assert_eq!(
        json["state_page_descriptors"][1]["state_kind"],
        "deltanet_recurrent"
    );
    assert_eq!(
        json["state_page_descriptors"][1]["label"],
        "qwen35.deltanet_state"
    );
    assert_eq!(
        json["state_page_descriptors"][1]["shape"],
        serde_json::json!([48, 48, 48])
    );
    assert_eq!(json["state_page_descriptors"][2]["state_kind"], "logits");
    assert_eq!(
        json["state_page_descriptors"][2]["label"],
        "qwen35.logits_snapshot"
    );
    assert_eq!(
        json["state_page_descriptors"][2]["shape"],
        serde_json::json!([16])
    );
    assert_eq!(
        json["state_page_descriptors"][3]["state_kind"],
        "backend_private"
    );
    assert_eq!(json["state_page_descriptors"][3]["placement"], "host");
}

#[test]
fn model_worker_runtime_view_json_reports_unsupported_state_arena_without_pages() {
    let worker = ModelWorkerRuntimeView {
        worker_id: ModelWorkerId {
            value: "worker:arch9:pp1:q8".to_string(),
        },
        max_seq: 4096,
        physical_cap: 4096,
        max_resident_workers: 1,
        resident_workers: 1,
        state_arena_backend: SequenceStateArenaBackend::Unsupported,
        state_arena_operations: SequenceStateArenaBackend::Unsupported
            .supported_operations()
            .to_vec(),
        resident_sessions: 0,
        state_page_descriptors: Vec::new(),
        memory: ModelWorkerMemoryView {
            model_file_bytes: 512,
            model_weight_bytes: 384,
            runtime_base_bytes: 0,
            runtime_session_bytes: 0,
            runtime_state_bytes: 0,
            total_resident_bytes: 384,
            evictable_state_bytes: 0,
        },
    };

    let json = model_worker_runtime_view_json(&worker);
    assert_eq!(json["state_arena_backend"], "unsupported");
    assert_eq!(json["state_arena_owns_pages"], false);
    assert_eq!(json["state_arena_operations"], serde_json::json!([]));
    assert_eq!(json["resident_sessions"], 0);
    assert_eq!(json["state_page_descriptor_entries"], 0);
    assert_eq!(json["state_page_descriptor_bytes"], 0);
    assert_eq!(json["state_page_descriptors"], serde_json::json!([]));
    assert_eq!(json["runtime_session_bytes"], 0);
    assert_eq!(json["evictable_state_bytes"], 0);
}

#[test]
fn reserve_session_state_kinds_default_deduplicate_and_alias() {
    let defaults = parse_reserve_session_state_kinds(&serde_json::json!({})).unwrap();
    assert_eq!(
        defaults,
        vec![SequenceStatePageKind::Kv, SequenceStatePageKind::DeltaNet]
    );

    let parsed = parse_reserve_session_state_kinds(&serde_json::json!({
        "state_kinds": [
            "attention_kv",
            "deltanet_recurrent",
            "attention_kv",
            "architecture_specific"
        ]
    }))
    .unwrap();
    assert_eq!(
        parsed,
        vec![
            SequenceStatePageKind::Kv,
            SequenceStatePageKind::DeltaNet,
            SequenceStatePageKind::BackendPrivate
        ]
    );

    let err = parse_reserve_session_state_kinds(&serde_json::json!({
        "state_kinds": ["unknown_state"]
    }))
    .unwrap_err();
    assert!(err.contains("unsupported kind unknown_state"));
}

#[test]
fn generic_state_reservation_descriptors_are_owned_handles() {
    let handle = SequenceStateHandle {
        id: "reserve-a".to_string(),
        kind: "generic_reserved_state".to_string(),
        generation: 7,
    };
    let descriptors = generic_state_reservation_descriptors(
        "worker-a",
        &handle,
        &[
            SequenceStatePageKind::Kv,
            SequenceStatePageKind::DeltaNet,
            SequenceStatePageKind::Logits,
        ],
        2048,
        10,
    );

    assert_eq!(descriptors.len(), 3);
    assert_eq!(descriptors[0].handle.id, "reserve-a");
    assert_eq!(descriptors[0].handle.kind, "generic_reserved_state");
    assert_eq!(descriptors[0].handle.generation, 7);
    assert_eq!(descriptors[0].session_id, "reserve-a");
    assert_eq!(descriptors[0].resident_bytes, 4);
    assert_eq!(descriptors[0].allocation_epoch, 7);
    assert!(descriptors[0].owns_pages);
    assert_eq!(descriptors[1].resident_bytes, 3);
    assert_eq!(descriptors[2].resident_bytes, 3);
    assert_eq!(descriptors[0].shape, vec![2048]);
    assert_eq!(descriptors[2].shape, vec![1]);
    assert_eq!(descriptors[0].placement, "host:reserved:worker-a");
    assert_eq!(descriptors[0].role, "reserved");

    let json = sequence_state_page_descriptor_json(&descriptors[0]);
    assert_eq!(json["handle"]["kind"], "generic_reserved_state");
    assert_eq!(json["handle"]["generation"], 7);
    assert_eq!(json["state_kind"], "attention_kv");
    assert_eq!(json["resident_bytes"], 4);
    assert_eq!(json["allocation_epoch"], 7);
    assert_eq!(json["owns_pages"], true);
}

fn test_state_descriptor(
    id: &str,
    kind: &str,
    generation: u64,
    page_kind: SequenceStatePageKind,
    bytes: usize,
) -> SequenceStatePageDescriptor {
    SequenceStatePageDescriptor {
        session_id: id.to_string(),
        handle: SequenceStateHandle {
            id: id.to_string(),
            kind: kind.to_string(),
            generation,
        },
        kind: page_kind,
        label: format!("{kind}.{}", page_kind.as_str()),
        logical_position: 16,
        resident_bytes: bytes,
        allocation_epoch: generation,
        owns_pages: generation != 0,
        shape: vec![1],
        placement: "hip:arch5:device0".to_string(),
        role: "resident".to_string(),
    }
}

#[test]
fn sequence_state_descriptor_lookup_binds_qwen35_epoch_handles() {
    let descriptors = vec![
        test_state_descriptor(
            "qwen35-checkpoint:batch:req:16",
            "qwen35_checkpoint",
            41,
            SequenceStatePageKind::Kv,
            128,
        ),
        test_state_descriptor(
            "qwen35-checkpoint:batch:req:16",
            "qwen35_checkpoint",
            41,
            SequenceStatePageKind::DeltaNet,
            96,
        ),
        test_state_descriptor(
            "qwen35-checkpoint:batch:req:16",
            "qwen35_checkpoint",
            42,
            SequenceStatePageKind::Logits,
            64,
        ),
    ];
    let handle = ParsedSequenceStateHandle {
        id: "qwen35-checkpoint:batch:req:16".to_string(),
        kind: Some("qwen35_checkpoint".to_string()),
        generation: Some(41),
    };
    let matched = describe_sequence_state_descriptors(descriptors.clone(), &handle).expect("match");
    assert_eq!(matched.len(), 2);
    assert!(matched.iter().all(|descriptor| {
        descriptor.handle.kind == "qwen35_checkpoint" && descriptor.allocation_epoch == 41
    }));

    let stale = ParsedSequenceStateHandle {
        generation: Some(40),
        ..handle
    };
    assert!(describe_sequence_state_descriptors(descriptors, &stale).is_none());
}

#[test]
fn parsed_state_handle_kind_routes_generic_qwen35_and_lfm2_surfaces() {
    let generic = parse_sequence_state_handle(&serde_json::json!({
        "id": "reserve-a",
        "kind": "generic_reserved_state",
        "generation": 7
    }))
    .unwrap();
    assert!(parsed_handle_may_target_generic(&generic));
    assert!(!parsed_handle_may_target_loaded_state(&generic));

    let qwen35 = parse_sequence_state_handle(&serde_json::json!({
        "id": "qwen35-checkpoint:batch:req:16",
        "kind": "qwen35_checkpoint",
        "allocation_epoch": 41
    }))
    .unwrap();
    assert!(!parsed_handle_may_target_generic(&qwen35));
    assert!(parsed_handle_may_target_loaded_state(&qwen35));
    assert_eq!(qwen35.generation, Some(41));

    let lfm2 = parse_sequence_state_handle(&serde_json::json!({
        "id": "lfm2-a",
        "kind": "lfm2_session",
        "allocation_epoch": 12
    }))
    .unwrap();
    assert!(!parsed_handle_may_target_generic(&lfm2));
    assert!(parsed_handle_may_target_loaded_state(&lfm2));
    assert_eq!(lfm2.generation, Some(12));

    let legacy = parse_sequence_state_handle(&serde_json::json!("session-a")).unwrap();
    assert!(parsed_handle_may_target_generic(&legacy));
    assert!(parsed_handle_may_target_loaded_state(&legacy));
}

#[test]
fn qwen35_checkpoint_handles_report_owned_epoch_identity() {
    let session_handle = qwen35_sequence_state_handle("session-a", 11);
    assert_eq!(session_handle.id, "session-a");
    assert_eq!(session_handle.kind, "qwen35_session");
    assert_eq!(session_handle.generation, 11);

    let checkpoint_handle = qwen35_sequence_state_handle("qwen35-checkpoint:batch:req:16", 12);
    assert_eq!(checkpoint_handle.kind, "qwen35_checkpoint");
    assert_eq!(checkpoint_handle.generation, 12);
}

#[test]
fn sequence_state_handle_id_accepts_string_or_handle_object() {
    assert_eq!(
        sequence_state_handle_id(&serde_json::json!("reserve-a")),
        Some("reserve-a")
    );
    assert_eq!(
        sequence_state_handle_id(&serde_json::json!({
            "id": "reserve-b",
            "kind": "generic_reserved_state",
            "generation": 9
        })),
        Some("reserve-b")
    );
    assert_eq!(
        sequence_state_handle_parts(&serde_json::json!({
            "id": "reserve-b",
            "kind": "generic_reserved_state",
            "generation": 9
        })),
        Some(("reserve-b", Some(9)))
    );
    assert_eq!(sequence_state_handle_id(&serde_json::json!("")), None);
    assert_eq!(
        sequence_state_handle_id(&serde_json::json!({"kind": "missing_id"})),
        None
    );
}

#[test]
fn generic_state_arena_rejects_stale_generation_handles() {
    let mut arena = GenericSequenceStateArena::new();
    let first = arena.reserve(
        "worker-a",
        "reserve-a".to_string(),
        &[SequenceStatePageKind::Kv],
        128,
        64,
        0,
    );
    assert_eq!(first.handle.generation, 1);
    assert!(arena.describe("reserve-a", Some(1)).is_some());
    assert!(arena.describe("reserve-a", Some(2)).is_none());
    assert_eq!(
        arena.release(vec![("reserve-a".to_string(), Some(2))]),
        (0, 0)
    );
    assert!(arena.describe("reserve-a", Some(1)).is_some());
    assert_eq!(
        arena.release(vec![("reserve-a".to_string(), Some(1))]),
        (1, 64)
    );

    let second = arena.reserve(
        "worker-a",
        "reserve-a".to_string(),
        &[SequenceStatePageKind::DeltaNet],
        128,
        32,
        0,
    );
    assert_eq!(second.handle.generation, 2);
    assert!(arena.describe("reserve-a", Some(1)).is_none());
    assert!(arena.describe("reserve-a", Some(2)).is_some());
}

#[test]
fn generic_state_arena_purges_ttl_and_releases_by_worker() {
    let mut arena = GenericSequenceStateArena::new();
    let expiring = arena.reserve(
        "worker-a",
        "reserve-expiring".to_string(),
        &[SequenceStatePageKind::Kv],
        128,
        16,
        1,
    );
    let persistent = arena.reserve(
        "worker-a",
        "reserve-persistent".to_string(),
        &[SequenceStatePageKind::DeltaNet],
        128,
        32,
        0,
    );
    let other_worker = arena.reserve(
        "worker-b",
        "reserve-other".to_string(),
        &[SequenceStatePageKind::Logits],
        128,
        64,
        0,
    );
    assert_eq!(arena.outstanding_bytes_for_worker("worker-a"), 48);
    assert_eq!(arena.outstanding_bytes_for_worker("worker-b"), 64);

    std::thread::sleep(std::time::Duration::from_millis(5));
    arena.purge_expired();
    assert!(arena
        .describe("reserve-expiring", Some(expiring.handle.generation))
        .is_none());
    assert!(arena
        .describe("reserve-persistent", Some(persistent.handle.generation))
        .is_some());
    assert_eq!(arena.outstanding_bytes_for_worker("worker-a"), 32);

    arena.release_worker("worker-a");
    assert!(arena
        .describe("reserve-persistent", Some(persistent.handle.generation))
        .is_none());
    assert!(arena
        .describe("reserve-other", Some(other_worker.handle.generation))
        .is_some());
    assert_eq!(arena.outstanding_bytes_for_worker("worker-a"), 0);
    assert_eq!(arena.outstanding_bytes_for_worker("worker-b"), 64);
}

#[test]
fn rejects_missing_worker_identity() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "batch_id": "batch-1",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0
            }
        }]
    });

    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains("worker_key_id or model"));
}

#[test]
fn accepts_empty_suffix_for_attached_checkpoint_reuse() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "batch_id": "batch-1",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "suffix_tokens": [],
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 4,
                "cached_prefix_tokens": 4,
                "runtime_state_handle": "qwen35-checkpoint:batch-0:req-0:4"
            }
        }]
    });

    let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
    assert_eq!(
        envelope.sessions[0].suffix_tokens.as_ref().unwrap().len(),
        0
    );
    assert_eq!(envelope.sessions[0].state_handle.cached_prefix_tokens, 4);
}

#[test]
fn rejects_duplicate_session_ids() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-1",
        "batch_id": "batch-dup",
        "worker_key_id": "worker-a",
        "sessions": [
            {
                "id": "req-1",
                "prompt": "first",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                }
            },
            {
                "id": "req-1",
                "prompt": "second",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                },
                "params": { "max_tokens": 8 }
            }
        ]
    });

    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains("duplicate session id"));
}

#[test]
fn rejects_missing_state_handle_fields() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-2",
        "batch_id": "batch-missing-state",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello"
        }]
    });

    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains(".state_handle must be an object"));
}

#[test]
fn rejects_invalid_state_handle_field_values() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-3",
        "batch_id": "batch-invalid-state",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv", "bad_kind"],
                "logical_position": 0
            }
        }]
    });
    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains("unsupported kind"));
}

#[test]
fn rejects_state_handle_missing_state_kinds() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-kinds",
        "batch_id": "batch-missing-state-kinds",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "logical_position": 0
            }
        }]
    });
    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains("state_kinds"));
}

#[test]
fn rejects_state_handle_negative_cached_prefix_tokens() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-cached",
        "batch_id": "batch-cached-prefix-negative",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0,
                "cached_prefix_tokens": -1
            }
        }]
    });
    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains("cached_prefix_tokens"));
}

#[test]
fn rejects_empty_runtime_state_handle() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-runtime-handle",
        "batch_id": "batch-runtime-handle",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "suffix_tokens": [1],
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 1,
                "cached_prefix_tokens": 1,
                "runtime_state_handle": ""
            }
        }]
    });
    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains("runtime_state_handle"));
}

#[test]
fn rejects_state_handle_missing_logical_position() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-logic",
        "batch_id": "batch-missing-logical-position",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"]
            }
        }]
    });
    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains("logical_position"));
}

#[test]
fn rejects_unsupported_params_shape() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-4",
        "batch_id": "batch-params",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0
            },
            "params": ["max_tokens", 32]
        }]
    });
    let err = validate_generate_batch_prefill(&msg).unwrap_err();
    assert!(err.contains(".params must be an object"));
}

#[test]
fn validates_semantic_boundary_checkpoint_param() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-boundary",
        "batch_id": "batch-boundary",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0
            },
            "params": {
                "semantic_boundary_checkpoints": true
            }
        }]
    });
    let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
    assert!(envelope.sessions[0].semantic_boundary_checkpoints);

    let bad = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-boundary-bad",
        "batch_id": "batch-boundary-bad",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0
            },
            "params": {
                "semantic_boundary_checkpoints": "yes"
            }
        }]
    });
    let err = validate_generate_batch_prefill(&bad).unwrap_err();
    assert!(err.contains(".params.semantic_boundary_checkpoints must be a boolean"));
}

#[test]
fn rejects_prompt_suffix_contract_violations() {
    let msg_both = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-5",
        "batch_id": "batch-contract",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "suffix_tokens": [1],
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0
            }
        }]
    });
    let err_both = validate_generate_batch_prefill(&msg_both).unwrap_err();
    assert!(err_both.contains("exactly one"));

    let msg_none = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "probe-6",
        "batch_id": "batch-contract-2",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0
            }
        }]
    });
    let err_none = validate_generate_batch_prefill(&msg_none).unwrap_err();
    assert!(err_none.contains("exactly one"));
}

#[test]
fn plans_dense_qwen35_multi_session_as_fused_candidate() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "prefill",
        "batch_id": "batch",
        "worker_key_id": "worker-a",
        "sessions": [
            {
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                }
            },
            {
                "id": "req-2",
                "prompt": "world",
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                }
            }
        ]
    });

    let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
    assert_eq!(
        plan_generate_batch_prefill_qwen35(5, envelope.session_count),
        GenerateBatchPrefillPlan::FusedDenseQwen35Candidate
    );
}

#[test]
fn keeps_singletons_serial_and_marks_moe_as_grouped_candidate() {
    let singleton = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "prefill",
        "batch_id": "batch",
        "worker_key_id": "worker-a",
        "sessions": [{
            "id": "req-1",
            "prompt": "hello",
            "state_handle": {
                "state_kinds": ["attention_kv", "deltanet_recurrent"],
                "logical_position": 0
            }
        }]
    });
    let singleton = validate_generate_batch_prefill(&singleton).expect("valid envelope");
    assert_eq!(
        plan_generate_batch_prefill_qwen35(5, singleton.session_count),
        GenerateBatchPrefillPlan::SerialExact
    );

    let moe = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "prefill",
        "batch_id": "batch",
        "worker_key_id": "worker-a",
        "sessions": [
            {
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                }
            },
            {
                "id": "req-2",
                "prompt": "world",
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                }
            }
        ]
    });
    let moe = validate_generate_batch_prefill(&moe).expect("valid envelope");
    assert_eq!(
        plan_generate_batch_prefill_qwen35(6, moe.session_count),
        GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate
    );
}

#[test]
fn dummy_state_counter_increments_across_prefill_and_generate() {
    let msg = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": "prefill",
        "batch_id": "batch",
        "worker_key_id": "dummy-worker",
        "sessions": [{
            "id": "req-1",
            "prompt": "one two three",
            "state_handle": {
                "state_kinds": ["attention_kv"],
                "logical_position": 0
            }
        }]
    });
    let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
    let mut dummy = DummyModelState::default();
    let consumed = dummy.consume_prefill_session(&envelope.sessions[0]);
    assert_eq!(consumed, 3);
    assert_eq!(dummy.sessions.get("req-1").copied(), Some(3));

    let counter = dummy.sessions.entry("req-1".to_string()).or_insert(0);
    let emitted = *counter;
    *counter += 1;
    assert_eq!(emitted, 3);
    assert_eq!(dummy.sessions.get("req-1").copied(), Some(4));
}

#[test]
fn dummy_release_sessions_removes_state() {
    let mut dummy = DummyModelState::default();
    dummy.sessions.insert("keep".to_string(), 1);
    dummy.sessions.insert("drop".to_string(), 2);

    let released = dummy.release_sessions(&["drop".to_string(), "missing".to_string()]);
    assert_eq!(released, 1);
    assert_eq!(dummy.session_count(), 1);
    assert!(dummy.sessions.contains_key("keep"));
    assert!(!dummy.sessions.contains_key("drop"));
}

#[test]
fn selects_fused_backend_by_default_for_fused_candidate_plans() {
    let fused_grouped_ok = || Ok::<(), String>(());
    assert_eq!(
        select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
            None,
            fused_grouped_ok(),
        )
        .unwrap(),
        Qwen35PrefillBatchBackend::FusedDense
    );
    assert_eq!(
        select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
            None,
            fused_grouped_ok(),
        )
        .unwrap(),
        Qwen35PrefillBatchBackend::FusedGroupedMoe
    );
    assert_eq!(
        select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::SerialExact,
            Some("auto"),
            fused_grouped_ok(),
        )
        .unwrap(),
        Qwen35PrefillBatchBackend::SerialReference
    );
}

#[test]
fn selects_dense_layer_chunked_decode_backend_only_for_dense_batches() {
    assert_eq!(
        select_qwen35_decode_batch_backend("auto", 5, 2).unwrap(),
        Qwen35DecodeBatchBackend::SerialReference
    );
    assert_eq!(
        select_qwen35_decode_batch_backend("auto", 6, 8).unwrap(),
        Qwen35DecodeBatchBackend::SerialReference
    );
    assert_eq!(
        select_qwen35_decode_batch_backend("fused", 5, 2).unwrap(),
        Qwen35DecodeBatchBackend::FusedDenseLayerChunked
    );
    assert_eq!(
        select_qwen35_decode_batch_backend("fused", 5, 1).unwrap(),
        Qwen35DecodeBatchBackend::FusedDenseLayerChunked
    );
    let moe_err = select_qwen35_decode_batch_backend("fused", 6, 2).unwrap_err();
    assert!(moe_err.contains("not dense Qwen35"));
    assert_eq!(
        select_qwen35_decode_batch_backend("fused_grouped_moe", 6, 2).unwrap(),
        Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked
    );
    for batch_size in [2, 4, 8] {
        assert_eq!(
            select_qwen35_decode_batch_backend("fused_grouped_moe", 6, batch_size).unwrap(),
            Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked,
            "explicit grouped-MoE decode should admit B={batch_size}"
        );
    }
    let singleton_err = select_qwen35_decode_batch_backend("fused_grouped_moe", 6, 1).unwrap_err();
    assert!(singleton_err.contains("at least two sessions"));
    let grouped_dense_err =
        select_qwen35_decode_batch_backend("fused_grouped_moe", 5, 2).unwrap_err();
    assert!(grouped_dense_err.contains("not Qwen35 grouped-MoE"));
}

#[test]
fn decode_batch_runtime_surface_rejects_spec_decode_and_eviction_state() {
    validate_qwen35_decode_batch_runtime_surface(5, 1, false, false).unwrap();
    validate_qwen35_decode_batch_runtime_surface(6, 1, false, false).unwrap();

    let pp_err = validate_qwen35_decode_batch_runtime_surface(5, 2, false, false).unwrap_err();
    assert!(pp_err.contains("single-GPU qwen35/qwen35-moe"));

    let arch_err = validate_qwen35_decode_batch_runtime_surface(9, 1, false, false).unwrap_err();
    assert!(arch_err.contains("single-GPU qwen35/qwen35-moe"));

    let dflash_err = validate_qwen35_decode_batch_runtime_surface(5, 1, true, false).unwrap_err();
    assert_eq!(
        dflash_err,
        "generate_batch_decode_step is not supported on DFlash-loaded models"
    );

    let eviction_err = validate_qwen35_decode_batch_runtime_surface(5, 1, false, true).unwrap_err();
    assert_eq!(
        eviction_err,
        "generate_batch_decode_step is not supported with active eviction state"
    );
}

#[test]
fn decode_batch_scheduler_metadata_reports_backend_state_and_fallback() {
    let small_auto_grouped = qwen35_decode_batch_scheduler_metadata(
        "auto",
        6,
        Qwen35DecodeBatchBackend::SerialReference,
        2,
        3,
    );
    assert_eq!(small_auto_grouped.selected_backend, "serial_reference");
    assert_eq!(
        small_auto_grouped.fallback_reason,
        "auto_grouped_moe_serial_small_batch_latency_gate"
    );
    assert_eq!(small_auto_grouped.cached_prefix_tokens, 3);

    let auto_grouped = qwen35_decode_batch_scheduler_metadata(
        "auto",
        6,
        Qwen35DecodeBatchBackend::SerialReference,
        8,
        7,
    );
    assert_eq!(auto_grouped.selected_backend, "serial_reference");
    assert_eq!(auto_grouped.batch_size, 8);
    assert_eq!(
        auto_grouped.compatible_state_kinds,
        vec!["attention_kv", "deltanet_recurrent"]
    );
    assert_eq!(auto_grouped.cached_prefix_tokens, 7);
    assert_eq!(
        auto_grouped.fallback_reason,
        "auto_grouped_moe_serial_pending_latency_gate"
    );

    let explicit_grouped = qwen35_decode_batch_scheduler_metadata(
        "fused_grouped_moe",
        6,
        Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked,
        4,
        0,
    );
    assert_eq!(
        explicit_grouped.selected_backend,
        "fused_grouped_moe_layer_chunked"
    );
    assert_eq!(explicit_grouped.fallback_reason, "none");

    let explicit_serial = qwen35_decode_batch_scheduler_metadata(
        "serial",
        5,
        Qwen35DecodeBatchBackend::SerialReference,
        2,
        0,
    );
    assert_eq!(
        explicit_serial.fallback_reason,
        "requested_serial_reference"
    );
}

#[test]
fn grouped_moe_decode_auto_latency_gate_requires_b4_or_larger() {
    assert!(!qwen35_grouped_moe_decode_auto_latency_gate_passed(2));
    assert!(qwen35_grouped_moe_decode_auto_latency_gate_passed(4));
    assert!(qwen35_grouped_moe_decode_auto_latency_gate_passed(8));
}

#[test]
fn fused_dense_decode_accepts_only_fp32_uncompacted_state_signatures() {
    let config = test_dense_qwen35_config();
    let fp32 = fp32_decode_state_signature();
    validate_qwen35_fused_dense_decode_session_signatures(&config, &[fp32, fp32], 2)
        .expect("fp32 dense decode state should be admitted");

    let mut compacted = fp32;
    compacted.kv_compact_offset = 8;
    let err =
        validate_qwen35_fused_dense_decode_session_signatures(&config, &[compacted, compacted], 2)
            .unwrap_err();
    assert!(err.contains("compacted KV offset"));

    // Plain Q8 KV (Q8_0, inline per-block scale) is fused, not rejected: the
    // per-layer KV write + attention branch on `kv_q8`. Admitted alongside FP32.
    let mut q8_kv = fp32;
    q8_kv.kv_quantized = true;
    q8_kv.kv_quant_q8 = true;
    validate_qwen35_fused_dense_decode_session_signatures(&config, &[q8_kv, q8_kv], 2)
        .expect("plain Q8 KV dense decode state should be admitted");

    // Asym/FWHT KV (any quantized-but-not-plain-Q8 state) stays on
    // serial_reference and is refused by the fused-prefix contract.
    let mut asym_kv = fp32;
    asym_kv.kv_quantized = true;
    asym_kv.kv_quant_asym2 = true;
    let err =
        validate_qwen35_fused_dense_decode_session_signatures(&config, &[asym_kv, asym_kv], 2)
            .unwrap_err();
    assert!(err.contains("unsupported KV quantization"));

    let mut q8_dn = fp32;
    q8_dn.dn_quant = qwen35::StateQuant::Q8;
    let err = validate_qwen35_fused_dense_decode_session_signatures(&config, &[q8_dn, q8_dn], 2)
        .unwrap_err();
    assert!(err.contains("Q8 DeltaNet state"));
}

#[test]
fn fused_dense_decode_batch_max_can_force_smaller_chunks() {
    unsafe {
        std::env::set_var("HIPFIRE_QWEN35_DECODE_BATCH_MAX", "1");
    }
    assert_eq!(qwen35_decode_batch_max_chunk_size(4), 1);
    unsafe {
        std::env::set_var("HIPFIRE_QWEN35_DECODE_BATCH_MAX", "3");
    }
    assert_eq!(qwen35_decode_batch_max_chunk_size(8), 3);
    unsafe {
        std::env::remove_var("HIPFIRE_QWEN35_DECODE_BATCH_MAX");
    }
    assert_eq!(qwen35_decode_batch_max_chunk_size(4), 4);
}

#[test]
fn native_decode_chunk_ranges_support_bounded_chunks() {
    assert_eq!(
        qwen35_decode_native_chunk_ranges(4, 1).unwrap(),
        vec![(0, 1), (1, 2), (2, 3), (3, 4)]
    );
    assert_eq!(
        qwen35_decode_native_chunk_ranges(1, 1).unwrap(),
        vec![(0, 1)]
    );
    assert_eq!(
        qwen35_decode_native_chunk_ranges(8, 4).unwrap(),
        vec![(0, 4), (4, 8)]
    );
    assert_eq!(
        qwen35_decode_native_chunk_ranges(5, 4).unwrap(),
        vec![(0, 4), (4, 5)]
    );
    assert_eq!(
        qwen35_decode_native_chunk_ranges(4, 2).unwrap(),
        vec![(0, 2), (2, 4)]
    );
    assert_eq!(
        qwen35_decode_native_chunk_ranges(8, 2).unwrap(),
        vec![(0, 2), (2, 4), (4, 6), (6, 8)]
    );
}

#[test]
fn grouped_moe_decode_contract_admits_q8_state_batches_and_rejects_fallback_cases() {
    let config = test_grouped_moe_qwen35_config();
    for batch_size in [2, 4, 8] {
        let signatures = vec![q8_decode_state_signature(); batch_size];
        validate_qwen35_grouped_moe_decode_session_signatures(
            &config,
            &signatures,
            batch_size,
            "gfx1151",
        )
        .unwrap_or_else(|err| panic!("B={batch_size} should admit grouped-MoE decode: {err}"));
    }

    let fp32 = vec![fp32_decode_state_signature(); 2];
    let err = validate_qwen35_grouped_moe_decode_session_signatures(&config, &fp32, 2, "gfx1151")
        .unwrap_err();
    assert!(err.contains("must use Q8 KV state"));

    let dense = test_dense_qwen35_config();
    let q8 = vec![q8_decode_state_signature(); 2];
    let err = validate_qwen35_grouped_moe_decode_session_signatures(&dense, &q8, 2, "gfx1151")
        .unwrap_err();
    assert!(err.contains("requires Qwen35 MoE/A3B weights"));

    let err = validate_qwen35_grouped_moe_decode_session_signatures(&config, &q8, 2, "gfx906")
        .unwrap_err();
    assert!(err.contains("requires an RDNA grouped-MoE target"));
}

#[test]
fn auto_grouped_moe_decode_stays_serial_when_native_route_is_unsupported() {
    let config = test_grouped_moe_qwen35_config();
    let signatures = vec![q8_decode_state_signature(); 8];
    let unsupported =
        validate_qwen35_grouped_moe_decode_session_signatures(&config, &signatures, 8, "gfx906")
            .unwrap_err();
    assert!(unsupported.contains("requires an RDNA grouped-MoE target"));

    let backend = select_qwen35_decode_batch_backend("auto", 6, 8).unwrap();
    assert_eq!(backend, Qwen35DecodeBatchBackend::SerialReference);
    let metadata = qwen35_decode_batch_scheduler_metadata("auto", 6, backend, 8, 11);
    assert_eq!(metadata.selected_backend, "serial_reference");
    assert_eq!(
        metadata.fallback_reason,
        "auto_grouped_moe_serial_pending_latency_gate"
    );
    assert_eq!(metadata.cached_prefix_tokens, 11);
}

#[test]
fn admits_fused_backend_only_for_dense_fused_candidate_plan() {
    let fused_grouped_ok = || Ok::<(), String>(());
    assert_eq!(
        select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
            Some("fused"),
            fused_grouped_ok(),
        )
        .unwrap(),
        Qwen35PrefillBatchBackend::FusedDense
    );
    let err = select_qwen35_prefill_batch_backend(
        GenerateBatchPrefillPlan::SerialExact,
        Some("fused"),
        fused_grouped_ok(),
    )
    .unwrap_err();
    assert!(err.contains("not fused-eligible"));
    let moe_err = select_qwen35_prefill_batch_backend(
        GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
        Some("fused"),
        fused_grouped_ok(),
    )
    .unwrap();
    assert_eq!(moe_err, Qwen35PrefillBatchBackend::FusedGroupedMoe);
}

#[test]
fn admits_explicit_grouped_moe_backend_only_for_grouped_candidate_plan() {
    let fused_grouped_ok = || Ok::<(), String>(());
    assert_eq!(
        select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
            Some("fused_moe"),
            fused_grouped_ok(),
        )
        .unwrap(),
        Qwen35PrefillBatchBackend::FusedGroupedMoe
    );
    assert_eq!(
        select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
            Some("grouped_moe"),
            fused_grouped_ok(),
        )
        .unwrap(),
        Qwen35PrefillBatchBackend::FusedGroupedMoe
    );
    let dense_err = select_qwen35_prefill_batch_backend(
        GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
        Some("fused_moe"),
        fused_grouped_ok(),
    )
    .unwrap_err();
    assert!(dense_err.contains("not grouped-MoE eligible"));
    let serial_err = select_qwen35_prefill_batch_backend(
        GenerateBatchPrefillPlan::SerialExact,
        Some("grouped_moe"),
        fused_grouped_ok(),
    )
    .unwrap_err();
    assert!(serial_err.contains("not grouped-MoE eligible"));
}

#[test]
fn auto_grouped_moe_falls_back_to_serial_when_fused_capability_fails() {
    let unsupported = Err::<(), String>(
        "grouped MoE session fused prefix currently requires K_TOP=8, got 10".to_string(),
    );
    assert_eq!(
        select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
            None,
            unsupported.clone(),
        )
        .unwrap(),
        Qwen35PrefillBatchBackend::SerialReference
    );
    let err = select_qwen35_prefill_batch_backend(
        GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
        Some("fused_moe"),
        unsupported,
    )
    .unwrap_err();
    assert!(err.contains("requires K_TOP=8"));
}

#[test]
fn paged_prefill_scratch_defaults_to_live_rows() {
    assert_eq!(
        qwen35_prefill_scratch_target_batch(true, 16, None, qwen35::PREFILL_MAX_BATCH),
        16
    );
    assert_eq!(
        qwen35_prefill_scratch_target_batch(true, 1, None, qwen35::PREFILL_MAX_BATCH),
        2
    );
    assert_eq!(
        qwen35_prefill_scratch_target_batch(true, 16, Some("64"), qwen35::PREFILL_MAX_BATCH),
        64
    );
    assert_eq!(
        qwen35_prefill_scratch_target_batch(false, 16, None, qwen35::PREFILL_MAX_BATCH),
        qwen35::PREFILL_MAX_BATCH
    );
}

fn test_prepared_prefill_session(
    id: &str,
    tokens: Vec<u32>,
    cached_prefix_tokens: usize,
    replay_as_generated_suffix: bool,
) -> Qwen35PreparedPrefillSession {
    Qwen35PreparedPrefillSession {
        id: id.to_string(),
        tokens,
        cached_prefix_tokens,
        replay_as_generated_suffix,
        state_kinds: vec!["attention_kv".to_string(), "deltanet_recurrent".to_string()],
        assistant_prefix: "plain".to_string(),
        max_think_tokens: 0,
        boundary_checkpoints: Vec::new(),
    }
}

#[test]
fn fused_dense_preflight_rejects_non_fused_candidate_plan() {
    let prepared = vec![test_prepared_prefill_session(
        "req-1",
        vec![1, 2, 3],
        0,
        false,
    )];

    let err = validate_qwen35_fused_dense_prefill_batch_preflight(
        &prepared,
        GenerateBatchPrefillPlan::SerialExact,
    )
    .unwrap_err();
    assert!(err.contains("requires plan=fused_dense_qwen35_candidate"));
}

#[test]
fn fused_dense_preflight_rejects_empty_session_token_slices() {
    let prepared = vec![
        test_prepared_prefill_session("req-1", Vec::new(), 0, true),
        test_prepared_prefill_session("req-2", vec![4], 0, true),
    ];

    let err = validate_qwen35_fused_dense_prefill_batch_preflight(
        &prepared,
        GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
    )
    .unwrap_err();
    assert!(err.contains("requires non-empty session token slices"));
}

#[test]
fn fused_dense_preflight_rejects_single_session_batches() {
    let prepared = vec![test_prepared_prefill_session(
        "req-1",
        vec![1, 2, 3],
        0,
        false,
    )];

    let err = validate_qwen35_fused_dense_prefill_batch_preflight(
        &prepared,
        GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
    )
    .unwrap_err();
    assert!(err.contains("requires at least two sessions"));
}

#[test]
fn fused_dense_preflight_rejects_mixed_prompt_and_suffix_batches() {
    let prepared = vec![
        test_prepared_prefill_session("req-1", vec![1, 2, 3], 0, false),
        test_prepared_prefill_session("req-2", vec![4], 16, true),
    ];

    let err = validate_qwen35_fused_dense_prefill_batch_preflight(
        &prepared,
        GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
    )
    .unwrap_err();
    assert!(err.contains("cannot mix full-prompt prefill and generated-suffix replay"));
}

#[test]
fn fused_grouped_moe_preflight_uses_grouped_candidate_plan() {
    let prepared = vec![
        test_prepared_prefill_session("req-1", vec![1, 2, 3], 0, false),
        test_prepared_prefill_session("req-2", vec![4], 0, false),
    ];

    validate_qwen35_fused_grouped_moe_prefill_batch_preflight(
        &prepared,
        GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
    )
    .expect("valid grouped-MoE preflight");
    let err = validate_qwen35_fused_grouped_moe_prefill_batch_preflight(
        &prepared,
        GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
    )
    .unwrap_err();
    assert!(err.contains("requires plan=grouped_moe_qwen35_candidate"));
}

#[test]
fn builds_dense_fused_worker_contract_for_prompt_batch() {
    let prepared = vec![
        test_prepared_prefill_session("req-1", vec![1, 2, 3], 0, false),
        test_prepared_prefill_session("req-2", vec![4, 5], 0, false),
    ];

    let contract = build_qwen35_fused_dense_prefill_batch_contract(
        &prepared,
        GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
    )
    .expect("valid fused dense contract");
    assert_eq!(contract.total_tokens, 5);
    assert_eq!(
        contract.input_kind,
        Qwen35FusedDensePrefillInputKind::FullPrompt
    );
    assert_eq!(contract.sessions[0].id, "req-1");
    assert_eq!(contract.sessions[1].tokens, &[4, 5]);
}

#[test]
fn builds_dense_fused_worker_contract_for_suffix_batch() {
    let prepared = vec![
        test_prepared_prefill_session("req-1", vec![11], 8, true),
        test_prepared_prefill_session("req-2", vec![12, 13], 8, true),
    ];

    let contract = build_qwen35_fused_dense_prefill_batch_contract(
        &prepared,
        GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
    )
    .expect("valid fused dense suffix contract");
    assert_eq!(contract.total_tokens, 3);
    assert_eq!(
        contract.input_kind,
        Qwen35FusedDensePrefillInputKind::GeneratedSuffixReplay
    );
    assert_eq!(contract.sessions[0].cached_prefix_tokens, 8);
    assert!(contract.sessions[1].replay_as_generated_suffix);
}
