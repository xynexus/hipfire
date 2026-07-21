#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "astrea_expert_sweep.py"
ASTREA_SCRIPT = ROOT / "scripts" / "astrea.py"


def load_module():
    spec = importlib.util.spec_from_file_location("astrea_expert_sweep", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_astrea():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("astrea_expert_sweep_cli", ASTREA_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def common_inputs(tmp_path):
    calibration = tmp_path / "calibration.txt"
    evaluation = tmp_path / "heldout.txt"
    calibration.write_text("calibration rows\n", encoding="utf-8")
    evaluation.write_text("held-out rows\n", encoding="utf-8")
    return {
        "model": write_safetensors_fixture(tmp_path / "model"),
        "artifact_stem": "Example-16B-A2B",
        "calibration_dataset": calibration,
        "evaluation_dataset": evaluation,
        "reference_model": write_hfq_fixture(tmp_path / "Example-16B-A2B.bf16.hfq"),
        "output_dir": tmp_path / "sweep",
        "evaluation_command_template": (
            "hipfire eval --model {candidate} --reference {reference_model} "
            "--dataset {evaluation_dataset} --out {evaluation_output}"
        ),
        "engine": {"fingerprint_id": "sha256:engine"},
        "command": ["python3", "scripts/astrea.py", "expert-sweep-plan"],
    }


def write_safetensors_fixture(root: Path, payload: bytes = b"weights") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "example_moe", "num_hidden_layers": 1}), encoding="utf-8"
    )
    shard = root / "model-00001-of-00001.safetensors"
    header = json.dumps(
        {"model.layers.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
        separators=(",", ":"),
    ).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": shard.name}}), encoding="utf-8"
    )
    return root


def write_hfq_fixture(path: Path, control: bytes = b'{"quantization_hash":"fixture"}\0') -> Path:
    data_offset = 32 + len(control)
    header = struct.pack("<4sIIIQQ", b"HFQM", 2, 6, 0, 32, data_offset)
    path.write_bytes(header + control + b"payload")
    return path


def stability_report(plan, variant, relative_l2_error):
    reference = max(plan["variants"], key=lambda item: item["expert_capture_target"])
    return {
        "schema": "hipfire.calibration_expert_statistic_stability.v1",
        "reference": reference["calibration_artifact"],
        "candidate": variant["calibration_artifact"],
        "provenance_complete": True,
        "fallback_set_match": True,
        "capture_target_order_valid": True,
        "reference_capture_target": reference["expert_capture_target"],
        "candidate_capture_target": variant["expert_capture_target"],
        "compared_expert_tensors": 4,
        "compared_values": 16,
        "non_finite_values": 0,
        "relative_l2_error": relative_l2_error,
        "valid": True,
    }


def test_floor_sweep_freezes_one_axis_and_heldout_commands(tmp_path):
    sweep = load_module()
    plan = sweep.build_plan(
        **common_inputs(tmp_path),
        axis="minimum",
        minimum_rows=[512, 1024, 2048, 4096],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )

    assert plan["schema"] == sweep.EXPERT_SWEEP_PLAN_SCHEMA
    assert plan["status"] == "planned_heldout_untouched"
    assert plan["axis"] == "minimum_expert_activations"
    assert plan["recipe"]["quant_format"] == "oq4.25++"
    assert plan["recipe"]["quant_args"] == ["--awq", "--ldlq"]
    assert plan["datasets"]["calibration"]["sha256"] != plan["datasets"]["evaluation"]["sha256"]
    assert [variant["minimum_expert_activations"] for variant in plan["variants"]] == [
        512,
        1024,
        2048,
        4096,
    ]
    assert {variant["expert_capture_target"] for variant in plan["variants"]} == {4096}

    for variant in plan["variants"]:
        run = variant["two_pass_command"]
        assert run[:2] == ["python3", "scripts/two_pass_quantize.py"]
        assert run[run.index("--min-expert-activations") + 1] == str(variant["minimum_expert_activations"])
        assert run[run.index("--expert-capture-target") + 1] == "4096"
        assert run[-3:] == ["--", "--awq", "--ldlq"]
        assert variant["calibration_artifact"].endswith(f".{variant['id']}.calib.hfq")
        assert variant["quantized_artifact"].endswith(f".{variant['id']}.oq4.25++.hfq")
        evaluation = variant["evaluation_command"]
        assert evaluation[:5] == [
            "target/release/hipfire",
            "lock",
            "run",
            f"expert-calibration-sweep-{variant['id']}",
            "--",
        ]
        assert str(common_inputs(tmp_path)["evaluation_dataset"].resolve()) in evaluation
        assert variant["quantized_artifact"] in evaluation
        assert variant["evaluation_output"] in evaluation

    assert plan["comparison_contract"]["required_metrics"] == [
        "mean_kld",
        "ppl",
        "low_traffic_expert_sensitivity",
        "artifact_size_bytes",
        "calibration_seconds",
        "reduction_launches",
    ]
    assert plan["plan_fingerprint"].startswith("sha256:")


def test_capture_sweep_holds_selected_minimum_fixed(tmp_path):
    sweep = load_module()
    plan = sweep.build_plan(
        **common_inputs(tmp_path),
        axis="capture",
        minimum_rows=None,
        capture_targets=[2048, 4096, 8192],
        selected_minimum=2048,
        fixed_capture_target=None,
    )

    assert plan["axis"] == "expert_capture_target"
    assert {variant["minimum_expert_activations"] for variant in plan["variants"]} == {2048}
    assert plan["comparison_contract"]["required_metrics"] == [
        "mean_kld",
        "ppl",
        "statistic_stability",
        "artifact_size_bytes",
        "calibration_seconds",
        "reduction_launches",
    ]
    assert [variant["expert_capture_target"] for variant in plan["variants"]] == [2048, 4096, 8192]
    assert plan["selection_contract"]["selected_minimum"] == 2048
    assert plan["selection_contract"]["selection_evidence_required"] is True


def test_analyze_floor_sweep_reports_low_traffic_cohorts_and_costs(tmp_path):
    sweep = load_module()
    plan = sweep.build_plan(
        **common_inputs(tmp_path),
        axis="minimum",
        minimum_rows=[512, 1024],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )
    records = [
        {
            "id": "min512-cap4096",
            "mean_kld": 0.12,
            "ppl": 11.0,
            "artifact_size_bytes": 1000,
            "calibration_seconds": 20.0,
            "reduction_launches": 80,
            "preserve_high_precision": [{"layer": 0, "expert": 3}],
        },
        {
            "id": "min1024-cap4096",
            "mean_kld": 0.10,
            "ppl": 10.0,
            "artifact_size_bytes": 1200,
            "calibration_seconds": 21.0,
            "reduction_launches": 80,
            "preserve_high_precision": [
                {"layer": 0, "expert": 3},
                {"layer": 0, "expert": 7},
            ],
        },
    ]
    results = sweep.build_results(plan, records)
    analysis = sweep.analyze_results(plan, results)

    assert results["results_fingerprint"].startswith("sha256:")
    assert analysis["schema"] == sweep.EXPERT_SWEEP_ANALYSIS_SCHEMA
    assert analysis["status"] == "complete_selection_required"
    assert analysis["reference_variant"] == "min1024-cap4096"
    assert analysis["required_metrics_complete"] is True
    assert analysis["variants"][0]["quality_vs_reference"] == {
        "mean_kld_delta": pytest.approx(0.02),
        "ppl_ratio": pytest.approx(1.1),
    }
    cohort = analysis["low_traffic_cohorts"][0]
    assert cohort["newly_low_bit_experts"] == [{"layer": 0, "expert": 7}]
    assert cohort["expert_count"] == 1
    assert cohort["mean_kld_penalty"] == pytest.approx(0.02)
    assert cohort["mean_kld_penalty_per_expert"] == pytest.approx(0.02)
    assert cohort["ppl_ratio"] == pytest.approx(1.1)


def test_sweep_analysis_rejects_missing_metrics_nonmonotonic_fallback_and_drift(tmp_path):
    sweep = load_module()
    plan = sweep.build_plan(
        **common_inputs(tmp_path),
        axis="minimum",
        minimum_rows=[512, 1024],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )
    records = [
        {
            "id": "min512-cap4096",
            "mean_kld": 0.12,
            "ppl": 11.0,
            "artifact_size_bytes": 1000,
            "calibration_seconds": 20.0,
            "reduction_launches": 80,
            "preserve_high_precision": [{"layer": 0, "expert": 7}],
        },
        {
            "id": "min1024-cap4096",
            "mean_kld": 0.10,
            "ppl": 10.0,
            "artifact_size_bytes": 1200,
            "calibration_seconds": 21.0,
            "reduction_launches": 80,
            "preserve_high_precision": [{"layer": 0, "expert": 3}],
        },
    ]

    with pytest.raises(ValueError, match="not monotonic"):
        sweep.analyze_results(plan, sweep.build_results(plan, records))

    missing = [dict(record) for record in records]
    missing[0].pop("mean_kld")
    with pytest.raises(ValueError, match="mean_kld"):
        sweep.build_results(plan, missing)

    valid = [dict(record) for record in records]
    valid[0]["preserve_high_precision"] = []
    valid[1]["preserve_high_precision"] = [{"layer": 0, "expert": 3}]
    results = sweep.build_results(plan, valid)
    results["variants"][0]["ppl"] = 999.0
    with pytest.raises(ValueError, match="results fingerprint"):
        sweep.analyze_results(plan, results)


def test_capture_sweep_analysis_requires_statistic_stability(tmp_path):
    sweep = load_module()
    plan = sweep.build_plan(
        **common_inputs(tmp_path),
        axis="capture",
        minimum_rows=None,
        capture_targets=[2048, 4096],
        selected_minimum=2048,
        fixed_capture_target=None,
    )
    records = [
        {
            "id": variant["id"],
            "mean_kld": 0.1,
            "ppl": 10.0,
            "artifact_size_bytes": 1000,
            "calibration_seconds": 20.0,
            "reduction_launches": 80,
            "preserve_high_precision": [{"layer": 0, "expert": 3}],
        }
        for variant in plan["variants"]
    ]

    with pytest.raises(ValueError, match="statistic_stability"):
        sweep.build_results(plan, records)

    for index, record in enumerate(records):
        record["statistic_stability_report"] = stability_report(
            plan,
            plan["variants"][index],
            0.01 / (index + 1),
        )
    analysis = sweep.analyze_results(plan, sweep.build_results(plan, records))
    assert analysis["axis"] == "expert_capture_target"
    assert analysis["low_traffic_cohorts"] == []
    assert analysis["variants"][0]["statistic_stability"] == pytest.approx(0.01)
    assert analysis["variants"][0]["statistic_stability_report_fingerprint"].startswith("sha256:")

    records[0]["statistic_stability_report"]["candidate"] = "wrong.calib.hfq"
    with pytest.raises(ValueError, match="stability candidate"):
        sweep.build_results(plan, records)


def test_sweep_rejects_contaminated_or_nonisolated_experiments(tmp_path):
    sweep = load_module()
    inputs = common_inputs(tmp_path)
    inputs["evaluation_dataset"].write_text(inputs["calibration_dataset"].read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct content"):
        sweep.build_plan(
            **inputs,
            axis="minimum",
            minimum_rows=[512, 1024],
            capture_targets=None,
            selected_minimum=None,
            fixed_capture_target=4096,
        )

    inputs = common_inputs(tmp_path)
    with pytest.raises(ValueError, match="below the selected minimum"):
        sweep.build_plan(
            **inputs,
            axis="capture",
            minimum_rows=None,
            capture_targets=[1024, 2048],
            selected_minimum=2048,
            fixed_capture_target=None,
        )

    with pytest.raises(ValueError, match="fixed capture target"):
        sweep.build_plan(
            **inputs,
            axis="minimum",
            minimum_rows=[512, 4096],
            capture_targets=None,
            selected_minimum=None,
            fixed_capture_target=2048,
        )


def test_plan_fingerprint_is_stable_and_covers_expert_policy(tmp_path):
    sweep = load_module()
    kwargs = common_inputs(tmp_path)
    first = sweep.build_plan(
        **kwargs,
        axis="minimum",
        minimum_rows=[512, 1024],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )
    second = sweep.build_plan(
        **kwargs,
        axis="minimum",
        minimum_rows=[512, 1024],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )
    changed = sweep.build_plan(
        **kwargs,
        axis="minimum",
        minimum_rows=[512, 2048],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )

    assert first == second
    assert first["plan_fingerprint"] != changed["plan_fingerprint"]
    json.dumps(first)

    timestamp_changed = dict(kwargs)
    timestamp_changed["engine"] = {
        "fingerprint_id": "sha256:engine",
        "captured_at_utc": "2099-01-01T00:00:00Z",
    }
    timestamp_plan = sweep.build_plan(
        **timestamp_changed,
        axis="minimum",
        minimum_rows=[512, 1024],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )
    assert first["plan_fingerprint"] == timestamp_plan["plan_fingerprint"]


def test_astrea_cli_writes_frozen_expert_sweep_plan(tmp_path):
    astrea = load_astrea()
    inputs = common_inputs(tmp_path)
    output = tmp_path / "plan.json"
    code, stdout, stderr = astrea.main_for_test(
        [
            "expert-sweep-plan",
            "--model",
            str(inputs["model"]),
            "--artifact-stem",
            inputs["artifact_stem"],
            "--calibration-dataset",
            str(inputs["calibration_dataset"]),
            "--evaluation-dataset",
            str(inputs["evaluation_dataset"]),
            "--reference-model",
            str(inputs["reference_model"]),
            "--output-dir",
            str(inputs["output_dir"]),
            "--evaluation-command-template",
            inputs["evaluation_command_template"],
            "--axis",
            "minimum",
            "--minimum-rows",
            "512",
            "--minimum-rows",
            "1024",
            "--fixed-capture-target",
            "4096",
            "--out",
            str(output),
        ]
    )

    assert code == 0, stderr
    assert stdout == ""
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["schema"] == "hipfire.astrea.expert_calibration_sweep_plan.v1"
    assert [variant["minimum_expert_activations"] for variant in plan["variants"]] == [512, 1024]
    assert plan["engine"]["schema"] == astrea.ENGINE_SCHEMA
    assert "crates/hipfire-runtime/src/calibration/contracts.rs" in plan["engine"]["source_hashes"]
    assert "crates/hipfire-coexistence/src/calibrate.rs" in plan["engine"]["source_hashes"]
    assert "scripts/astrea_expert_sweep.py" in plan["engine"]["source_hashes"]
    assert plan["command_argv"][:3] == ["python3", "scripts/astrea.py", "expert-sweep-plan"]


def test_verify_plan_rejects_dataset_engine_and_payload_drift(tmp_path):
    sweep = load_module()
    inputs = common_inputs(tmp_path)
    inputs["model"] = tmp_path / "model"
    inputs["reference_model"] = tmp_path / "reference.hfq"
    write_safetensors_fixture(inputs["model"])
    write_hfq_fixture(inputs["reference_model"])
    plan = sweep.build_plan(
        **inputs,
        axis="minimum",
        minimum_rows=[512, 1024],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )

    verified = sweep.verify_plan(plan, current_engine={"fingerprint_id": "sha256:engine"})
    assert verified["schema"] == sweep.EXPERT_SWEEP_VERIFY_SCHEMA
    assert verified["status"] == "verified_not_run"
    assert verified["variant_ids"] == ["min512-cap4096", "min1024-cap4096"]

    inputs["evaluation_dataset"].write_text("changed held-out rows\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation dataset hash drift"):
        sweep.verify_plan(plan, current_engine={"fingerprint_id": "sha256:engine"})

    inputs["evaluation_dataset"].write_text("held-out rows\n", encoding="utf-8")
    with pytest.raises(ValueError, match="engine fingerprint drift"):
        sweep.verify_plan(plan, current_engine={"fingerprint_id": "sha256:different"})

    tampered = json.loads(json.dumps(plan))
    tampered["variants"][0]["minimum_expert_activations"] = 1
    with pytest.raises(ValueError, match="plan fingerprint mismatch"):
        sweep.verify_plan(tampered, current_engine={"fingerprint_id": "sha256:engine"})


def test_astrea_cli_verifies_plan_before_execution(tmp_path):
    astrea = load_astrea()
    sweep = load_module()
    inputs = common_inputs(tmp_path)
    inputs["model"] = tmp_path / "model"
    inputs["reference_model"] = tmp_path / "reference.hfq"
    write_safetensors_fixture(inputs["model"])
    write_hfq_fixture(inputs["reference_model"])
    current_engine = astrea.engine_fingerprint()
    inputs["engine"] = current_engine
    plan = sweep.build_plan(
        **inputs,
        axis="capture",
        minimum_rows=None,
        capture_targets=[2048, 4096],
        selected_minimum=2048,
        fixed_capture_target=None,
    )
    plan_path = tmp_path / "capture-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    code, stdout, stderr = astrea.main_for_test(["expert-sweep-verify", "--plan", str(plan_path)])
    assert code == 0, stderr
    result = json.loads(stdout)
    assert result["status"] == "verified_not_run"
    assert result["plan_fingerprint"] == plan["plan_fingerprint"]


def test_astrea_cli_builds_and_analyzes_measured_expert_sweep_results(tmp_path):
    astrea = load_astrea()
    sweep = load_module()
    inputs = common_inputs(tmp_path)
    plan = sweep.build_plan(
        **inputs,
        axis="minimum",
        minimum_rows=[512, 1024],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    record_paths = []
    for index, variant in enumerate(plan["variants"]):
        record = {
            "id": variant["id"],
            "mean_kld": 0.2 - index * 0.05,
            "ppl": 12.0 - index,
            "artifact_size_bytes": 1000 + index,
            "calibration_seconds": 20.0,
            "reduction_launches": 80,
            "preserve_high_precision": [
                {"layer": 0, "expert": expert} for expert in range(index + 1)
            ],
        }
        path = tmp_path / f"{variant['id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        record_paths.append(path)

    argv = ["expert-sweep-results", "--plan", str(plan_path)]
    for path in record_paths:
        argv.extend(["--record", str(path)])
    code, stdout, stderr = astrea.main_for_test(argv)
    assert code == 0, stderr
    results = json.loads(stdout)
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    code, stdout, stderr = astrea.main_for_test(
        ["expert-sweep-analyze", "--plan", str(plan_path), "--results", str(results_path)]
    )
    assert code == 0, stderr
    analysis = json.loads(stdout)
    assert analysis["status"] == "complete_selection_required"
    assert analysis["required_metrics_complete"] is True


def test_verify_plan_rejects_source_and_reference_identity_drift(tmp_path):
    sweep = load_module()
    inputs = common_inputs(tmp_path)
    inputs["model"] = write_safetensors_fixture(tmp_path / "model")
    inputs["reference_model"] = write_hfq_fixture(tmp_path / "reference.hfq")
    plan = sweep.build_plan(
        **inputs,
        axis="minimum",
        minimum_rows=[512, 1024],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )

    assert plan["model"]["identity"]["kind"] == "safetensors_manifest"
    assert plan["reference_model"]["identity"]["kind"] == "hfq_control_region"

    shard = inputs["model"] / "model-00001-of-00001.safetensors"
    original = shard.read_bytes()
    shard.write_bytes(original.replace(b"F16", b"BF1", 1))
    with pytest.raises(ValueError, match="source model identity drift"):
        sweep.verify_plan(plan, current_engine={"fingerprint_id": "sha256:engine"})

    shard.write_bytes(original)
    write_hfq_fixture(inputs["reference_model"], control=b'{"quantization_hash":"changed"}\0')
    with pytest.raises(ValueError, match="reference model identity drift"):
        sweep.verify_plan(plan, current_engine={"fingerprint_id": "sha256:engine"})


def test_verify_plan_rejects_command_binding_drift_even_with_refingerprinted_payload(tmp_path):
    sweep = load_module()
    inputs = common_inputs(tmp_path)
    inputs["model"] = write_safetensors_fixture(tmp_path / "model")
    inputs["reference_model"] = write_hfq_fixture(tmp_path / "reference.hfq")
    plan = sweep.build_plan(
        **inputs,
        axis="minimum",
        minimum_rows=[512],
        capture_targets=None,
        selected_minimum=None,
        fixed_capture_target=4096,
    )

    tampered = json.loads(json.dumps(plan))
    command = tampered["variants"][0]["two_pass_command"]
    command[command.index("--corpus") + 1] = str(inputs["evaluation_dataset"].resolve())
    body = {key: value for key, value in tampered.items() if key != "plan_fingerprint"}
    tampered["plan_fingerprint"] = sweep._plan_fingerprint(body)
    with pytest.raises(ValueError, match="calibration dataset disagrees"):
        sweep.verify_plan(tampered, current_engine={"fingerprint_id": "sha256:engine"})
