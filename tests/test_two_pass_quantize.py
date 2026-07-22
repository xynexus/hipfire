import copy
import importlib.util
import json
import struct
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "two_pass_quantize.py"
SPEC = importlib.util.spec_from_file_location("two_pass_quantize", SCRIPT)
assert SPEC and SPEC.loader
two_pass = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(two_pass)


def write_safetensors_index(path: Path, tensors: dict[str, tuple[str, list[int], int]]) -> None:
    offset = 0
    header = {}
    for name, (dtype, shape, byte_len) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_len],
        }
        offset += byte_len
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def test_default_quant_format_is_mixed_oq425_double_plus():
    assert two_pass.DEFAULT_QUANT_FORMAT == "oq4.25++"
    assert two_pass.DEFAULT_LAYER_PREFETCH_BYTES == 16 * 1024**3
    assert two_pass.DEFAULT_MIN_EXPERT_ACTIVATIONS == 2048
    assert two_pass.DEFAULT_EXPERT_CAPTURE_TARGET == 4096
    assert two_pass.DEFAULT_EXPERT_CAPTURE_TILE_ROWS == 256
    assert two_pass.DEFAULT_REQUIRED_EXPERT_FRACTION == 1.0
    assert two_pass.DEFAULT_SAMPLING_SEED == 1
    assert two_pass.DEFAULT_EXPERT_COVERAGE_POLICY == "preserve-undercovered"


def test_pass_two_storage_preflight_counts_grouped_preserved_experts(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    write_safetensors_index(
        model / "model.safetensors",
        {
            # Four grouped experts. Gate/up and down are deliberately different
            # shapes so the estimator must derive each expert's actual payload.
            "model.layers.0.mlp.experts.gate_up_proj.weight": ("BF16", [4, 8, 256], 4 * 8 * 256 * 2),
            "model.layers.0.mlp.experts.down_proj.weight": ("BF16", [4, 256, 4], 4 * 256 * 4 * 2),
            "model.layers.0.self_attn.q_proj.weight": ("BF16", [256, 256], 256 * 256 * 2),
            "model.layers.0.input_layernorm.weight": ("BF16", [256], 256 * 2),
        },
    )
    calibration = {
        "metadata": {
            "preserve_high_precision": [
                {"layer": 0, "expert": 1},
                {"layer": 0, "expert": 3},
            ]
        }
    }

    preflight = two_pass.pass_two_storage_preflight(
        model=model,
        output=tmp_path / "out" / "Tiny-MoE.oq4.25++.hfq",
        quant_format="oq4.25++",
        calibration=calibration,
        available_bytes=1,
    )

    # Each preserved expert owns 8*256 + 256*4 BF16 values.
    assert preflight["preserve_high_precision"]["requested_experts"] == 2
    assert preflight["preserve_high_precision"]["matched_experts"] == 2
    assert preflight["preserve_high_precision"]["output_bytes"] == 2 * (8 * 256 + 256 * 4) * 2
    assert preflight["estimate"]["completed_artifact_estimate_bytes"] > 0
    assert preflight["estimate"]["required_free_bytes"] > preflight["estimate"]["completed_artifact_estimate_bytes"]
    assert preflight["filesystem"]["available_bytes"] == 1
    assert preflight["filesystem"]["sufficient"] is False
    with pytest.raises(RuntimeError, match="insufficient output storage"):
        two_pass.require_pass_two_storage(preflight)


def test_pass_two_storage_preflight_rejects_unmatched_preserved_expert(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    write_safetensors_index(
        model / "model.safetensors",
        {"model.layers.0.self_attn.q_proj.weight": ("BF16", [256, 256], 256 * 256 * 2)},
    )
    calibration = {
        "metadata": {"preserve_high_precision": [{"layer": 0, "expert": 7}]}
    }

    with pytest.raises(RuntimeError, match="no routed-expert source tensors"):
        two_pass.pass_two_storage_preflight(
            model=model,
            output=tmp_path / "Tiny.oq4.25++.hfq",
            quant_format="oq4.25++",
            calibration=calibration,
            available_bytes=10**12,
        )


def test_pass_two_storage_preflight_accepts_presplit_w1_w2_w3_experts(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    tensors = {
        "model.layers.2.mlp.experts.5.w1.weight": ("F16", [8, 256], 8 * 256 * 2),
        "model.layers.2.mlp.experts.5.w2.weight": ("F16", [256, 8], 256 * 8 * 2),
        "model.layers.2.mlp.experts.5.w3.weight": ("F16", [8, 256], 8 * 256 * 2),
    }
    write_safetensors_index(model / "model.safetensors", tensors)

    preflight = two_pass.pass_two_storage_preflight(
        model=model,
        output=tmp_path / "Tiny.oq4.25++.hfq",
        quant_format="oq4.25++",
        calibration={
            "metadata": {"preserve_high_precision": [{"layer": 2, "expert": 5}]}
        },
        available_bytes=10**12,
    )

    assert preflight["preserve_high_precision"]["matched_experts"] == 1
    assert preflight["preserve_high_precision"]["output_bytes"] == 3 * 8 * 256 * 2


def test_pass_two_storage_preflight_uses_q8_ceiling_for_nonexpert_weights(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    write_safetensors_index(
        model / "model.safetensors",
        {
            "model.layers.0.self_attn.q_proj.weight": ("BF16", [256, 256], 256 * 256 * 2),
            "model.layers.0.input_layernorm.weight": ("BF16", [256], 256 * 2),
        },
    )

    preflight = two_pass.pass_two_storage_preflight(
        model=model,
        output=tmp_path / "Tiny.oq4.25++.hfq",
        quant_format="oq4.25++",
        calibration={"metadata": {"preserve_high_precision": []}},
        available_bytes=10**12,
    )

    expected_q8 = (256 * 256 // 32) * 34
    assert preflight["estimate"]["nonexpert_weight_ceiling"] == "q8f16"
    assert preflight["estimate"]["mixed_payload_bytes"] == expected_q8 + 256 * 2


def test_build_commands_use_one_layer_streamed_teacher_pass_then_quantize(tmp_path):
    model = Path("/srv/huggingface/models--Qwen--Qwen3.5-397B-A17B")
    calib = tmp_path / "Qwen3.5-397B-A17B.calib.hfq"
    output = tmp_path / "Qwen3.5-397B-A17B.oq4++.hfq"

    collect_cmd, quant_cmd = two_pass.build_commands(
        coexistence="target/release/hipfire-coexistence",
        quantizer="target/release/hipfire-quantize",
        model=model,
        calib=calib,
        output=output,
        quant_format="oq4++",
        corpus="benchmarks/prompts/calibration.txt",
        n_sequences=128,
        ctx_len=2048,
        batch_size=64,
        time_tile=32,
        max_rows=2048,
        layer_prefetch_bytes=16 * 1024**3,
        kldref_topk=64,
        min_expert_activations=2048,
        expert_capture_target=4096,
        expert_capture_tile_rows=256,
        required_expert_fraction=1.0,
        sampling_seed=1,
        expert_coverage_policy="preserve-undercovered",
        quant_args=["--awq", "--ldlq"],
    )

    assert collect_cmd[:2] == ["target/release/hipfire-coexistence", "calibrate"]
    assert "scripts/depreciated/collect_hessian.py" not in collect_cmd
    assert "--kldref" in collect_cmd
    assert "--resume" in collect_cmd
    assert collect_cmd[collect_cmd.index("--expert-coverage-policy") + 1] == "preserve-undercovered"
    assert collect_cmd[collect_cmd.index("--kldref-topk") + 1] == "64"
    assert collect_cmd[collect_cmd.index("--sequence-batch") + 1] == "64"
    assert collect_cmd[collect_cmd.index("--time-tile") + 1] == "32"
    assert collect_cmd[collect_cmd.index("--max-rows") + 1] == "2048"
    assert collect_cmd[collect_cmd.index("--layer-prefetch-bytes") + 1] == str(16 * 1024**3)
    assert collect_cmd[collect_cmd.index("--min-expert-activations") + 1] == "2048"
    assert collect_cmd[collect_cmd.index("--expert-capture-target") + 1] == "4096"
    assert collect_cmd[collect_cmd.index("--expert-capture-tile-rows") + 1] == "256"
    assert collect_cmd[collect_cmd.index("--required-expert-fraction") + 1] == "1.0"
    assert collect_cmd[collect_cmd.index("--sampling-seed") + 1] == "1"
    assert quant_cmd == [
        "target/release/hipfire-quantize",
        "--input",
        str(model),
        "--output",
        str(output),
        "--format",
        "oq4++",
        "--hessian",
        str(calib),
        "--awq",
        "--ldlq",
    ]
    collect_exec, quant_exec = two_pass.scope_gpu_commands("target/release/hipfire", collect_cmd, quant_cmd)
    assert collect_exec == collect_cmd
    assert quant_exec[:6] == [
        "target/release/hipfire",
        "lock",
        "run",
        "two-pass-quantization",
        "--",
        "target/release/hipfire-quantize",
    ]


def test_recipe_fingerprint_changes_with_inputs_but_not_dict_order(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("one\ntwo\n")
    first = two_pass.recipe_manifest(
        model=Path("/models/snapshot"),
        calib=tmp_path / "model.calib.hfq",
        output=tmp_path / "model.oq4++.hfq",
        quant_format="oq4++",
        corpus=corpus,
        n_sequences=128,
        ctx_len=2048,
        batch_size=16,
        time_tile=32,
        max_rows=512,
        layer_prefetch_bytes=16 * 1024**3,
        kldref_topk=64,
        min_expert_activations=2048,
        expert_capture_target=4096,
        expert_capture_tile_rows=256,
        required_expert_fraction=1.0,
        sampling_seed=1,
        expert_coverage_policy="preserve-undercovered",
        quant_args=["--awq", "--ldlq"],
    )
    second = two_pass.recipe_manifest(
        model=Path("/models/snapshot"),
        calib=tmp_path / "model.calib.hfq",
        output=tmp_path / "model.oq4++.hfq",
        quant_format="oq4++",
        corpus=corpus,
        n_sequences=128,
        ctx_len=2048,
        batch_size=16,
        time_tile=32,
        max_rows=512,
        layer_prefetch_bytes=16 * 1024**3,
        kldref_topk=64,
        min_expert_activations=2048,
        expert_capture_target=4096,
        expert_capture_tile_rows=256,
        required_expert_fraction=1.0,
        sampling_seed=1,
        expert_coverage_policy="preserve-undercovered",
        quant_args=["--awq", "--ldlq"],
    )
    assert first == second
    assert first["recipe_fingerprint"].startswith("sha256:")
    corpus.write_text("changed\n")
    changed = two_pass.recipe_manifest(
        model=Path("/models/snapshot"),
        calib=tmp_path / "model.calib.hfq",
        output=tmp_path / "model.oq4++.hfq",
        quant_format="oq4++",
        corpus=corpus,
        n_sequences=128,
        ctx_len=2048,
        batch_size=16,
        time_tile=64,
        max_rows=1024,
        layer_prefetch_bytes=8 * 1024**3,
        kldref_topk=64,
        min_expert_activations=1024,
        expert_capture_target=8192,
        expert_capture_tile_rows=128,
        required_expert_fraction=0.75,
        sampling_seed=7,
        expert_coverage_policy="strict",
        quant_args=["--awq", "--ldlq"],
    )
    assert changed["recipe_fingerprint"] != first["recipe_fingerprint"]
    assert first["time_tile"] == 32
    assert first["max_rows"] == 512
    assert first["layer_prefetch_bytes"] == 16 * 1024**3
    assert first["min_expert_activations"] == 2048
    assert first["expert_capture_target"] == 4096
    assert first["expert_capture_tile_rows"] == 256
    assert first["required_expert_fraction"] == 1.0
    assert first["sampling_seed"] == 1
    assert first["expert_coverage_policy"] == "preserve_undercovered"

    changed_quality_policy = two_pass.recipe_manifest(
        model=Path("/models/snapshot"),
        calib=tmp_path / "model.calib.hfq",
        output=tmp_path / "model.oq4++.hfq",
        quant_format="oq4++",
        corpus=corpus,
        n_sequences=128,
        ctx_len=2048,
        batch_size=16,
        time_tile=64,
        max_rows=1024,
        layer_prefetch_bytes=8 * 1024**3,
        kldref_topk=64,
        min_expert_activations=1024,
        expert_capture_target=4096,
        expert_capture_tile_rows=128,
        required_expert_fraction=0.75,
        sampling_seed=7,
        expert_coverage_policy="preserve-undercovered",
        quant_args=["--awq", "--ldlq"],
    )
    assert changed_quality_policy["recipe_fingerprint"] != changed["recipe_fingerprint"]


def test_manifest_consumes_native_read_ledger_and_artifact_fingerprints(tmp_path):
    path = tmp_path / "two-pass.json"
    recipe = {"recipe_fingerprint": "sha256:recipe"}
    calibration = {
        "artifact_fingerprint": "fnv64:calib",
        "metadata": {
            "engine_build": "executable:sha256-engine",
            "run_fingerprint": "fnv1a64:run",
            "source_manifest": {"fingerprint": "fnv1a64:source"},
            "job": {"samples": {"fingerprint": "fnv1a64:samples"}},
            "read_ledger": {
                "planned_logical": ["a", "b"],
                "consumed_logical": ["a", "b"],
                "duplicate_logical": [],
                "missing_logical": [],
            },
        },
    }
    quantized = {
        "artifact_fingerprint": "fnv64:model",
        "metadata": {
            "quantization_hash": {"value": "0123456789abcdef"},
            "calibration": {"xxh64": "feedface"},
        },
    }
    calibration_audit = {
        "schema": "hipfire.calibration_audit.v1",
        "valid": True,
        "artifact_fingerprint": "fnv64:calib",
        "index_only": True,
        "payload_values_checked": False,
        "errors": [],
    }
    storage_preflight = {
        "schema": "hipfire.pass_two_storage_preflight.v1",
        "filesystem": {"sufficient": True},
    }

    manifest = two_pass.update_manifest(
        path,
        recipe=recipe,
        phase="complete",
        calibration=calibration,
        calibration_audit=calibration_audit,
        storage_preflight=storage_preflight,
        quantized=quantized,
    )

    restored = json.loads(path.read_text())
    assert restored == manifest
    assert restored["source_reads"] == calibration["metadata"]["read_ledger"]
    assert restored["calibration_audit"] == calibration_audit
    assert restored["pass_two_storage_preflight"] == storage_preflight
    assert restored["fingerprints"] == {
        "calibration_artifact": "fnv64:calib",
        "calibration_engine_build": "executable:sha256-engine",
        "calibration_run": "fnv1a64:run",
        "source": "fnv1a64:source",
        "samples": "fnv1a64:samples",
        "quantized_artifact": "fnv64:model",
        "quantized_payload": "0123456789abcdef",
    }


def test_interrupted_manifest_resume_preserves_completed_calibration(tmp_path):
    path = tmp_path / "two-pass.json"
    recipe = {"recipe_fingerprint": "sha256:recipe"}
    calibration = {
        "artifact_fingerprint": "fnv64:calib",
        "metadata": {
            "run_fingerprint": "run",
            "read_ledger": {"missing_logical": [], "duplicate_logical": []},
        },
    }
    two_pass.update_manifest(
        path,
        recipe=recipe,
        phase="calibration_complete",
        calibration=calibration,
        phase_timings={"calibration_seconds": 12.5},
    )

    resumed = two_pass.update_manifest(path, recipe=recipe, phase="quantization_running")

    assert resumed["calibration"] == calibration
    assert resumed["source_reads"] == calibration["metadata"]["read_ledger"]
    assert resumed["status"] == "quantization_running"
    assert resumed["phase_timings"] == {"calibration_seconds": 12.5}

    completed = two_pass.update_manifest(
        path,
        recipe=recipe,
        phase="complete",
        phase_timings={"quantization_seconds": 7.25},
    )
    assert completed["phase_timings"] == {
        "calibration_seconds": 12.5,
        "quantization_seconds": 7.25,
    }


def test_calibration_sigkill_records_attempt_time_and_can_resume(tmp_path):
    path = tmp_path / "two-pass.json"
    calib = tmp_path / "Tiny.calib.hfq"
    recipe = {"recipe_fingerprint": "sha256:recipe"}
    execution = {
        "mode": "segmented",
        "process_segment_layers": 2,
        "release_seconds": 0,
        "segments": [
            {
                "started_after_layer": 0,
                "pause_after_layer": 2,
                "completed_layers": 2,
                "artifact_complete": False,
            }
        ],
    }
    two_pass.update_manifest(
        path,
        recipe=recipe,
        phase="calibration_running",
        calibration_execution=execution,
        phase_timings={"calibration_seconds": 12.5},
    )

    def terminate(_command, *, check):
        assert check is True
        raise subprocess.CalledProcessError(-9, ["hipfire-coexistence", "calibrate"])

    def record_failure(phase, elapsed_seconds, failure):
        previous = json.loads(path.read_text())
        two_pass.update_manifest(
            path,
            recipe=recipe,
            phase=phase,
            failure=failure,
            phase_timings=two_pass.accumulate_attempt_timing(
                previous,
                phase_name="calibration",
                elapsed_seconds=elapsed_seconds,
            ),
        )

    with pytest.raises(subprocess.CalledProcessError):
        two_pass.run_calibration_attempt(
            ["hipfire-coexistence", "calibrate", "--output", str(calib), "--resume"],
            calib=calib,
            total_layers=4,
            segment_layers=0,
            runner=terminate,
            on_failure=record_failure,
        )

    interrupted = json.loads(path.read_text())
    assert interrupted["status"] == "calibration_interrupted"
    assert interrupted["calibration_execution"] == execution
    assert interrupted["failure"]["kind"] == "signal"
    assert interrupted["failure"]["returncode"] == -9
    assert interrupted["failure"]["signal"] == 9
    assert interrupted["phase_timings"]["calibration_seconds"] >= 12.5
    assert interrupted["phase_timings"]["last_calibration_attempt_seconds"] >= 0

    resumed = two_pass.update_manifest(path, recipe=recipe, phase="calibration_running")
    assert resumed["status"] == "calibration_running"
    assert resumed["calibration_execution"] == execution
    assert "failure" not in resumed


def test_quantization_sigterm_records_interrupted_manifest_and_can_resume(tmp_path):
    path = tmp_path / "two-pass.json"
    recipe = {"recipe_fingerprint": "sha256:recipe"}
    calibration = {
        "artifact_fingerprint": "fnv64:calib",
        "metadata": {
            "run_fingerprint": "run",
            "read_ledger": {"missing_logical": [], "duplicate_logical": []},
        },
    }
    two_pass.update_manifest(
        path,
        recipe=recipe,
        phase="quantization_running",
        calibration=calibration,
    )

    def terminate(_command, *, check):
        assert check is True
        raise subprocess.CalledProcessError(143, ["hipfire", "lock", "acquire"])

    def record_failure(phase, elapsed_seconds, failure):
        two_pass.update_manifest(
            path,
            recipe=recipe,
            phase=phase,
            failure=failure,
            phase_timings={"last_quantization_attempt_seconds": elapsed_seconds},
        )

    with pytest.raises(subprocess.CalledProcessError):
        two_pass.run_quantization_pass(
            ["hipfire", "lock", "acquire"],
            runner=terminate,
            on_failure=record_failure,
        )

    interrupted = json.loads(path.read_text())
    assert interrupted["status"] == "quantization_interrupted"
    assert interrupted["calibration"] == calibration
    assert interrupted["failure"]["kind"] == "signal"
    assert interrupted["failure"]["returncode"] == 143
    assert interrupted["failure"]["signal"] == 15
    assert interrupted["phase_timings"]["last_quantization_attempt_seconds"] >= 0

    resumed = two_pass.update_manifest(path, recipe=recipe, phase="quantization_running")
    assert resumed["status"] == "quantization_running"
    assert resumed["calibration"] == calibration
    assert "failure" not in resumed


def test_quantization_non_signal_failure_is_not_labeled_interrupted():
    recorded = []

    def fail(_command, *, check):
        assert check is True
        raise subprocess.CalledProcessError(2, ["hipfire-quantize"])

    with pytest.raises(subprocess.CalledProcessError):
        two_pass.run_quantization_pass(
            ["hipfire-quantize"],
            runner=fail,
            on_failure=lambda phase, elapsed, failure: recorded.append(
                (phase, elapsed, failure)
            ),
        )

    assert len(recorded) == 1
    phase, elapsed, failure = recorded[0]
    assert phase == "quantization_failed"
    assert elapsed >= 0
    assert failure["kind"] == "process_error"
    assert failure["returncode"] == 2
    assert "signal" not in failure

    phase, failure = two_pass._quantization_failure(KeyboardInterrupt())
    assert phase == "quantization_interrupted"
    assert failure["kind"] == "signal"
    assert failure["signal"] == 2


def reusable_calibration_contract():
    expected = {
        "engine_build": "executable:sha256-engine",
        "run_fingerprint": "fnv64:run",
        "model": {"family": "qwen3.5", "adapter_version": "qwen3.5-stream-v1", "arch_id": 6},
        "corpus": {
            "sequences": 128,
            "context": 2048,
            "rows": 262144,
            "sample_fingerprint": "fnv64:samples",
            "corpus_fingerprint": "sha256:corpus",
        },
        "microbatch": {"sequence_batch": 64, "time_tile": 32, "max_rows": 2048},
        "source_plan": {
            "source_fingerprint": "fnv64:source",
            "tokenizer_fingerprint": "sha256:tokenizer",
            "shards": [{"file": "model-00001.safetensors", "bytes": 123}],
        },
        "expert_capture": {
            "minimum_rows": 2048,
            "target_rows": 4096,
            "tile_rows": 256,
            "sampling": {"kind": "deterministic_first", "seed": 1},
            "required_fraction": 1.0,
            "coverage_policy": "preserve-undercovered",
        },
        "kldref": {"enabled": True, "top_k": 64},
    }
    inspection = {
        "metadata": {
            "artifact_kind": "calibration",
            "engine_build": "executable:sha256-engine",
            "run_fingerprint": "fnv64:run",
            "family": "qwen3.5",
            "adapter_version": "qwen3.5-stream-v1",
            "arch_id": 6,
            "source_manifest": {
                "fingerprint": "fnv64:source",
                "shards": [{"file": "model-00001.safetensors", "bytes": 123}],
            },
            "job": {
                "source_fingerprint": "fnv64:source",
                "tokenizer_fingerprint": "sha256:tokenizer",
                "corpus_fingerprint": "sha256:corpus",
                "samples": {
                    "samples": [{"id": f"sample-{index}", "tokens": [1] * 2048} for index in range(128)],
                    "context_len": 2048,
                    "sampling_seed": 1,
                    "fingerprint": "fnv64:samples",
                },
                "options": {
                    "sequence_batch": 64,
                    "time_tile": 32,
                    "max_rows": 2048,
                    "boundary_precision": "f32",
                    "expert_quota": {
                        "min_rows": 2048,
                        "target_rows": 4096,
                        "tile_rows": 256,
                        "sampling": {"kind": "deterministic_first", "seed": 1},
                    },
                    "required_expert_fraction": 1.0,
                    "expert_coverage_policy": "preserve-undercovered",
                    "kldref": True,
                    "kldref_top_k": 64,
                },
            },
            "microbatch_geometry": {"sequence_batch": 64, "time_tile": 32, "row_budget": 2048},
            "read_ledger": {"missing_logical": [], "duplicate_logical": []},
        }
    }
    return expected, inspection


def test_skip_calibration_recipe_validation_binds_native_identity_and_policy():
    expected, inspection = reusable_calibration_contract()
    two_pass.validate_calibration_inspection(inspection)
    two_pass.validate_reusable_calibration(inspection, expected)

    rebuilt_plan = copy.deepcopy(expected)
    rebuilt_plan["engine_build"] = "executable:new-producer"
    two_pass.validate_reusable_calibration(inspection, rebuilt_plan)

    mutations = [
        ("run fingerprint", lambda value: value["metadata"].update(run_fingerprint="other")),
        ("source", lambda value: value["metadata"]["source_manifest"].update(fingerprint="other")),
        ("tokenizer", lambda value: value["metadata"]["job"].update(tokenizer_fingerprint="other")),
        ("corpus", lambda value: value["metadata"]["job"].update(corpus_fingerprint="other")),
        ("sample", lambda value: value["metadata"]["job"]["samples"].update(fingerprint="other")),
        ("geometry", lambda value: value["metadata"]["microbatch_geometry"].update(row_budget=1024)),
        (
            "minimum_rows",
            lambda value: value["metadata"]["job"]["options"]["expert_quota"].update(min_rows=1024),
        ),
        (
            "coverage_policy",
            lambda value: value["metadata"]["job"]["options"].update(expert_coverage_policy="strict"),
        ),
        ("KLDREF", lambda value: value["metadata"]["job"]["options"].update(kldref_top_k=32)),
    ]
    for label, mutate in mutations:
        changed = copy.deepcopy(inspection)
        mutate(changed)
        with pytest.raises(RuntimeError, match=label):
            two_pass.validate_reusable_calibration(changed, expected)


def test_skip_calibration_validation_command_is_native_dry_run():
    collect = [
        "target/release/hipfire-coexistence",
        "calibrate",
        "--model",
        "/model",
        "--resume",
    ]
    assert two_pass.calibration_validation_command(collect) == [*collect, "--dry-run"]


def test_segmented_calibration_resumes_from_durable_checkpoint_and_finalizes(tmp_path):
    calib = tmp_path / "Tiny.calib.hfq"
    boundary = two_pass.calibration_boundary_checkpoint(calib)
    boundary.parent.mkdir(parents=True)
    boundary.write_text(
        json.dumps(
            {
                "completed_layers": 3,
                "total_layers": 10,
                "artifact_complete": False,
            }
        )
    )
    calls = []
    progress = []

    def runner(command, *, check):
        assert check is True
        calls.append(command)
        checkpoint = json.loads(boundary.read_text())
        if "--pause-after-layers" in command:
            checkpoint["completed_layers"] = int(command[command.index("--pause-after-layers") + 1])
        else:
            checkpoint["completed_layers"] = checkpoint["total_layers"]
            checkpoint["artifact_complete"] = True
        boundary.write_text(json.dumps(checkpoint))

    execution = two_pass.run_calibration_pass(
        ["hipfire-coexistence", "calibrate", "--output", str(calib), "--resume"],
        calib=calib,
        total_layers=10,
        segment_layers=4,
        runner=runner,
        release_seconds=0,
        progress=lambda value: progress.append(copy.deepcopy(value)),
    )

    assert calls == [
        [
            "hipfire-coexistence",
            "calibrate",
            "--output",
            str(calib),
            "--resume",
            "--pause-after-layers",
            "7",
        ],
        ["hipfire-coexistence", "calibrate", "--output", str(calib), "--resume"],
    ]
    assert execution["mode"] == "segmented"
    assert execution["process_segment_layers"] == 4
    assert [segment["completed_layers"] for segment in execution["segments"]] == [7, 10]
    assert execution["artifact_complete"] is True
    assert [state["segments"][-1]["completed_layers"] for state in progress] == [7, 10]


def test_segmented_calibration_rejects_a_successful_process_without_progress(tmp_path):
    calib = tmp_path / "Tiny.calib.hfq"
    boundary = two_pass.calibration_boundary_checkpoint(calib)
    boundary.parent.mkdir(parents=True)
    boundary.write_text(
        json.dumps(
            {
                "completed_layers": 1,
                "total_layers": 4,
                "artifact_complete": False,
            }
        )
    )

    with pytest.raises(RuntimeError, match="did not advance"):
        two_pass.run_calibration_pass(
            ["hipfire-coexistence", "calibrate", "--output", str(calib), "--resume"],
            calib=calib,
            total_layers=4,
            segment_layers=2,
            runner=lambda _command, *, check: None,
            release_seconds=0,
        )


def test_calibration_segmentation_is_execution_provenance_not_recipe_identity(tmp_path):
    path = tmp_path / "two-pass.json"
    recipe = {"recipe_fingerprint": "sha256:recipe"}
    execution = {
        "mode": "segmented",
        "process_segment_layers": 4,
        "release_seconds": 5,
    }

    manifest = two_pass.update_manifest(
        path,
        recipe=recipe,
        phase="calibration_running",
        calibration_execution=execution,
    )

    assert manifest["recipe_fingerprint"] == "sha256:recipe"
    assert manifest["calibration_execution"] == execution

    resumed = two_pass.update_manifest(
        path,
        recipe=recipe,
        phase="calibration_running",
        calibration_execution={
            "mode": "segmented",
            "process_segment_layers": 4,
            "release_seconds": 5,
            "segments": [
                {
                    "started_after_layer": 0,
                    "pause_after_layer": 4,
                    "completed_layers": 4,
                    "artifact_complete": False,
                }
            ],
        },
    )
    restarted = two_pass.update_manifest(
        path,
        recipe=recipe,
        phase="calibration_running",
        calibration_execution=execution,
    )

    assert resumed["calibration_execution"]["segments"][0]["completed_layers"] == 4
    assert restarted["calibration_execution"]["segments"] == resumed["calibration_execution"]["segments"]


def test_calibration_artifact_audit_command_uses_family_neutral_native_gate():
    assert two_pass.calibration_audit_command(
        "target/release/hipfire-coexistence",
        Path("/artifacts/model.calib.hfq"),
    ) == [
        "target/release/hipfire-coexistence",
        "artifact",
        "audit-calibration",
        "--input",
        "/artifacts/model.calib.hfq",
    ]

    inspection = {"artifact_fingerprint": "fnv64:calib"}
    audit = {
        "schema": "hipfire.calibration_audit.v1",
        "valid": True,
        "artifact_fingerprint": "fnv64:calib",
        "index_only": True,
        "payload_values_checked": False,
        "errors": [],
    }
    two_pass.validate_calibration_audit(audit, inspection)
    changed = copy.deepcopy(audit)
    changed["artifact_fingerprint"] = "fnv64:other"
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        two_pass.validate_calibration_audit(changed, inspection)
