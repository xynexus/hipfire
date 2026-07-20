import importlib.util
import json
import copy
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "two_pass_quantize.py"
SPEC = importlib.util.spec_from_file_location("two_pass_quantize", SCRIPT)
assert SPEC and SPEC.loader
two_pass = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(two_pass)


def test_default_quant_format_is_mixed_oq425_double_plus():
    assert two_pass.DEFAULT_QUANT_FORMAT == "oq4.25++"
    assert two_pass.DEFAULT_LAYER_PREFETCH_BYTES == 16 * 1024**3
    assert two_pass.DEFAULT_MIN_EXPERT_ACTIVATIONS == 2048
    assert two_pass.DEFAULT_EXPERT_CAPTURE_TARGET == 4096
    assert two_pass.DEFAULT_EXPERT_CAPTURE_TILE_ROWS == 256
    assert two_pass.DEFAULT_REQUIRED_EXPERT_FRACTION == 1.0
    assert two_pass.DEFAULT_SAMPLING_SEED == 1
    assert two_pass.DEFAULT_EXPERT_COVERAGE_POLICY == "preserve-undercovered"


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
    assert "scripts/collect_hessian.py" not in collect_cmd
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

    manifest = two_pass.update_manifest(
        path,
        recipe=recipe,
        phase="complete",
        calibration=calibration,
        calibration_audit=calibration_audit,
        quantized=quantized,
    )

    restored = json.loads(path.read_text())
    assert restored == manifest
    assert restored["source_reads"] == calibration["metadata"]["read_ledger"]
    assert restored["calibration_audit"] == calibration_audit
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
    )

    resumed = two_pass.update_manifest(path, recipe=recipe, phase="quantization_running")

    assert resumed["calibration"] == calibration
    assert resumed["source_reads"] == calibration["metadata"]["read_ledger"]
    assert resumed["status"] == "quantization_running"


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
