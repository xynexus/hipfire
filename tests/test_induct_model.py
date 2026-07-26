import importlib.util
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "induct_model.py"
SPEC = importlib.util.spec_from_file_location("induct_model", SCRIPT)
assert SPEC and SPEC.loader
induct = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(induct)

TWO_PASS_SCRIPT = Path(__file__).parents[1] / "scripts" / "two_pass_quantize.py"
TWO_PASS_SPEC = importlib.util.spec_from_file_location("induct_two_pass_quantize", TWO_PASS_SCRIPT)
assert TWO_PASS_SPEC and TWO_PASS_SPEC.loader
two_pass = importlib.util.module_from_spec(TWO_PASS_SPEC)
TWO_PASS_SPEC.loader.exec_module(two_pass)


def _configs(tmp_path):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "hidden_size": 4096,
                    "num_hidden_layers": 60,
                    # The target and DFlash draft have independent attention
                    # geometries; the published 397B pair is 32x2x256 vs
                    # 32x8x128 while sharing hidden/vocab/target-layer shape.
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "vocab_size": 248320,
                },
            }
        )
    )
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}})
    )
    (target / "model-00001-of-00001.safetensors").write_bytes(b"target")
    (draft / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlashDraftModel"],
                "hidden_size": 4096,
                "num_hidden_layers": 6,
                "num_target_layers": 60,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 248320,
                "dflash_config": {
                    "block_size": 16,
                    "mask_token_id": 248077,
                    "target_layer_ids": [1, 9, 17, 25, 33, 41, 49, 57],
                },
            }
        )
    )
    (draft / "model.safetensors").write_bytes(b"draft")
    return target, draft


def test_preflight_accepts_current_nested_target_and_zlab_dflash_shape(tmp_path):
    target, draft = _configs(tmp_path)

    result = induct.preflight_sources(target, draft)

    assert result["target"]["num_hidden_layers"] == 60
    assert result["draft"]["block_size"] == 16
    assert result["draft"]["target_layer_ids"] == [1, 9, 17, 25, 33, 41, 49, 57]
    assert result["compatibility"] == "compatible"


def test_preflight_and_layout_allow_a_cask_only_target(tmp_path):
    target, _draft = _configs(tmp_path)

    result = induct.preflight_sources(target, None)
    paths = induct.artifact_layout(tmp_path, "BLS-Mini-Code-1.0", "oq4.25++", [])

    assert result["draft"] is None
    assert result["compatibility"] == "not-applicable"
    assert paths["bundle"].name == "BLS-Mini-Code-1.0--triattn.oq4.25++.hfq"
    assert not any(key.startswith("dflash_") for key in paths)


def test_artifact_layout_matches_registry_sidecar_names(tmp_path):
    paths = induct.artifact_layout(
        tmp_path, "Qwen3.5-397B-A17B", "oq4++", ["oq4+"]
    )

    assert paths["model"].name == "Qwen3.5-397B-A17B--oq4++.hfq"
    assert paths["dflash_oq4+"] == tmp_path / "drafts" / "Qwen3.5-397B-A17B--dflash.oq4+.hfq"
    assert paths["triattn"] == tmp_path / "triattn" / "Qwen3.5-397B-A17B.triattn.hfq"
    assert paths["calib"] == tmp_path / "calib" / "Qwen3.5-397B-A17B.calib.hfq"
    assert paths["bundle"] == tmp_path / "models" / "Qwen3.5-397B-A17B--dflash.triattn.oq4++.hfq"
    assert paths["manifest"] == tmp_path / "induction" / "Qwen3.5-397B-A17B--oq4++" / "manifest.json"

    staged = induct.artifact_layout(
        tmp_path, "Qwen3.5-397B-A17B", "oq4++", ["oq4+"], tmp_path / "staging"
    )
    assert staged["bundle"].parent == tmp_path / "staging"
    assert staged["bundle_partial"].parent == tmp_path / "staging"


def test_default_quant_format_is_mixed_oq425_double_plus():
    assert induct.DEFAULT_QUANT_FORMAT == "oq4.25++"
    assert induct.DEFAULT_DFLASH_FORMATS == ("oq4+",)
    assert induct.DEFAULT_LAYER_PREFETCH_BYTES == 16 * 1024**3
    assert induct.DEFAULT_CALIBRATION_SEGMENT_LAYERS == 0
    assert induct.DEFAULT_MIN_EXPERT_ACTIVATIONS == 2048
    assert induct.DEFAULT_EXPERT_CAPTURE_TARGET == 4096
    assert induct.DEFAULT_EXPERT_CAPTURE_TILE_ROWS == 256
    assert induct.DEFAULT_REQUIRED_EXPERT_FRACTION == 1.0
    assert induct.DEFAULT_SAMPLING_SEED == 1
    assert induct.DEFAULT_EXPERT_COVERAGE_POLICY == "preserve-undercovered"


def test_mixed_opus_format_keeps_calibration_recipe_and_canonical_name(tmp_path):
    paths = induct.artifact_layout(tmp_path, "Qwen3.5-397B-A17B", "oq4.5++", ["oq4+"])

    assert paths["model"].name == "Qwen3.5-397B-A17B--oq4.5++.hfq"
    assert paths["dflash_oq4+"].name == "Qwen3.5-397B-A17B--dflash.oq4+.hfq"
    assert paths["bundle"].name == "Qwen3.5-397B-A17B--dflash.triattn.oq4.5++.hfq"
    assert induct.default_quant_args("oq4.5++") == ["--awq", "--ldlq"]


def test_commands_compose_existing_converters_with_scoped_gpu_stages(tmp_path):
    target, draft = _configs(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("calibration text")
    paths = induct.artifact_layout(
        tmp_path, "Qwen3.5-397B-A17B", "oq4++", ["oq4+"]
    )

    commands = induct.build_stage_commands(
        target=target,
        draft=draft,
        corpus=corpus,
        paths=paths,
        quant_format="oq4++",
        dflash_formats=["oq4+"],
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
        python="python3",
        hipfire="target/release/hipfire",
        coexistence="target/release/hipfire-coexistence",
        quantizer="target/release/hipfire-quantize",
        dflash_converter="target/release/dflash_convert",
        quant_args=["--awq", "--ldlq"],
        reuse_calibration=True,
        calibration_segment_layers=4,
    )

    assert commands["dflash"][0] == [
        "target/release/dflash_convert",
        "--input",
        str(draft),
        "--output",
        str(paths["dflash_oq4+"]),
        "--format",
        "oq4+",
    ]
    target_cmd = commands["target"][0]
    assert target_cmd[:2] == ["python3", "scripts/two_pass_quantize.py"]
    assert target_cmd[target_cmd.index("--calib") + 1] == str(paths["calib"])
    assert target_cmd[target_cmd.index("--cask-output") + 1] == str(paths["triattn"])
    assert target_cmd[target_cmd.index("--coexistence") + 1] == "target/release/hipfire-coexistence"
    assert "--python" not in target_cmd
    assert target_cmd[target_cmd.index("--manifest") + 1] == str(paths["two_pass_manifest"])
    assert target_cmd[target_cmd.index("--batch-size") + 1] == "64"
    assert target_cmd[target_cmd.index("--time-tile") + 1] == "32"
    assert target_cmd[target_cmd.index("--max-rows") + 1] == "2048"
    assert target_cmd[target_cmd.index("--layer-prefetch-bytes") + 1] == str(16 * 1024**3)
    assert target_cmd[target_cmd.index("--min-expert-activations") + 1] == "2048"
    assert target_cmd[target_cmd.index("--expert-capture-target") + 1] == "4096"
    assert target_cmd[target_cmd.index("--expert-capture-tile-rows") + 1] == "256"
    assert target_cmd[target_cmd.index("--required-expert-fraction") + 1] == "1.0"
    assert target_cmd[target_cmd.index("--sampling-seed") + 1] == "1"
    assert target_cmd[target_cmd.index("--expert-coverage-policy") + 1] == "preserve-undercovered"
    assert target_cmd[target_cmd.index("--calibration-segment-layers") + 1] == "4"
    assert "--skip-calib" in target_cmd
    assert target_cmd[-2:] == ["--awq", "--ldlq"]
    bundle = commands["bundle"][0]
    assert bundle[:3] == [
        "target/release/hipfire",
        "model",
        "compose",
    ]
    assert bundle[3:6] == [str(paths["model"]), str(paths["dflash_oq4+"]), str(paths["triattn"])]
    assert bundle[bundle.index("--output") + 1] == str(paths["bundle_partial"])


def test_dflash_format_defaults_use_canonical_non_rotated_opus():
    assert induct._dflash_format_args("oq4+") == ["--format", "oq4+"]
    assert induct._dflash_format_args("oq4.25+") == ["--format", "oq4.25+"]


def test_resume_only_skips_artifacts_with_expected_magic(tmp_path):
    hfq = tmp_path / "model.hfq"
    triattn = tmp_path / "model.triattn.hfq"
    hfq.write_bytes(b"HFQM" + b"\0" * 28)
    triattn.write_bytes(b"TRIA" + b"\0" * 28)

    assert induct.artifact_is_valid(hfq, b"HFQM")
    assert induct.artifact_is_valid(triattn, b"TRIA")
    assert not induct.artifact_is_valid(hfq, b"TRIA")


def test_existing_complete_calibration_is_reused_unless_forced(tmp_path):
    paths = induct.artifact_layout(tmp_path, "Qwen3.5-397B-A17B", "oq4.25++", ["oq4+"])
    assert not induct.should_reuse_calibration(paths, force=False)
    paths["calib"].parent.mkdir(parents=True)
    paths["calib"].write_bytes(b"HFQM" + b"\0" * 28)
    assert not induct.should_reuse_calibration(paths, force=False)
    paths["triattn"].parent.mkdir(parents=True)
    paths["triattn"].write_bytes(b"HFQM" + b"\0" * 28)
    assert induct.should_reuse_calibration(paths, force=False)
    assert not induct.should_reuse_calibration(paths, force=True)


def test_target_rerun_invalidates_and_allows_atomic_bundle_replacement(tmp_path, monkeypatch):
    paths = induct.artifact_layout(tmp_path, "Model", "oq4.25++", [])
    paths["bundle"].parent.mkdir(parents=True)
    paths["bundle"].write_bytes(b"HFQM" + b"stale bundle")
    monkeypatch.setattr(
        induct,
        "_stage_complete",
        lambda stage, _paths, _target_fingerprint=None, _dflash_fingerprint=None: stage == "bundle",
    )

    planned, dependency_invalidated = induct.plan_stages_to_run(
        ["target", "bundle"], paths, "sha256:recipe", force=False
    )

    assert planned == ["target", "bundle"]
    assert dependency_invalidated is True


def test_dflash_reuse_requires_matching_recipe_and_output_digest(tmp_path):
    _target, draft = _configs(tmp_path)
    paths = induct.artifact_layout(tmp_path, "Model", "oq4.25++", ["oq4+"])
    output = paths["dflash_oq4+"]
    output.parent.mkdir(parents=True)
    output.write_bytes(b"HFQM" + b"draft payload" + b"\0" * 32)
    recipe = induct._dflash_recipe_fingerprint(
        draft=draft,
        paths=paths,
        dflash_formats=["oq4+"],
    )
    assert recipe is not None
    paths["draft_manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["draft_manifest"].write_text(
        json.dumps(
            {
                "schema": 1,
                "status": "complete",
                "recipe_fingerprint": recipe,
                "outputs": {str(output): induct.sha256_file(output)},
            }
        )
    )

    assert induct.dflash_stage_complete(paths, recipe)
    assert not induct.dflash_stage_complete(paths, "sha256:different")
    output.write_bytes(b"HFQM" + b"changed payload")
    assert not induct.dflash_stage_complete(paths, recipe)


def test_repo_tool_is_rebuilt_when_a_source_is_newer(tmp_path):
    binary = tmp_path / "tool"
    source = tmp_path / "tool.rs"
    binary.write_bytes(b"binary")
    source.write_text("fn main() {}")
    os.utime(binary, ns=(1_000_000_000, 1_000_000_000))
    os.utime(source, ns=(2_000_000_000, 2_000_000_000))

    assert induct.tool_needs_build(binary, [source])

    os.utime(binary, ns=(3_000_000_000, 3_000_000_000))
    assert not induct.tool_needs_build(binary, [source])


def test_bundle_inspection_requires_exact_roles_and_strong_digests(tmp_path, monkeypatch):
    report = {
        "components": [
            {"role": role, "sha256": character * 64, "byte_len": 128}
            for role, character in (("base", "a"), ("dflash", "b"), ("triattn", "c"))
        ]
    }
    monkeypatch.setattr(
        induct.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(report)),
    )

    assert induct.inspect_bundle("hipfire", tmp_path / "bundle.hfq", expect_dflash=True) == report

    report["components"][1]["sha256"] = "weak"
    try:
        induct.inspect_bundle("hipfire", tmp_path / "bundle.hfq", expect_dflash=True)
    except RuntimeError as error:
        assert "lacks a SHA-256" in str(error)
    else:
        raise AssertionError("weak component digest was accepted")


def test_transfer_refuses_candidate_without_admission(tmp_path):
    bundle = tmp_path / "Model--triattn.oq4.25++.hfq"
    bundle.write_bytes(b"HFQM" + b"candidate")
    manifest = {"admission": {"status": "pending"}}

    try:
        induct.transfer_admitted_bundle(bundle, manifest, "halo")
    except RuntimeError as error:
        assert "admission.status=admitted" in str(error)
    else:
        raise AssertionError("pending candidate was transferred")


def test_transfer_verifies_temporary_and_final_remote_digest(tmp_path, monkeypatch):
    bundle = tmp_path / "Model--triattn.oq4.25++.hfq"
    bundle.write_bytes(b"HFQM" + b"admitted")
    digest = induct.sha256_file(bundle)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(stdout=f"{digest}  remote\n")

    monkeypatch.setattr(induct.subprocess, "run", fake_run)
    delivery = induct.transfer_admitted_bundle(
        bundle,
        {
            "admission": {"status": "admitted"},
            "bundle": {"sha256": digest},
        },
        "halo",
    )

    assert delivery["status"] == "delivered"
    assert delivery["sha256"] == digest
    assert [command[0] for command in commands] == ["ssh", "scp", "ssh", "ssh", "ssh"]


def test_calibration_adapter_source_roots_are_discovered_without_family_list(tmp_path):
    expected = []
    for name in ("hipfire-arch-zeta", "hipfire-arch-alpha"):
        source_root = tmp_path / "crates" / name / "src"
        source_root.mkdir(parents=True)
        (source_root / "calibration_stream.rs").write_text("// adapter\n")
        expected.append(source_root)
    unrelated = tmp_path / "crates" / "hipfire-arch-unrelated" / "src"
    unrelated.mkdir(parents=True)
    (unrelated / "lib.rs").write_text("// no calibration adapter\n")

    assert induct.calibration_adapter_source_roots(tmp_path) == sorted(expected)


def test_target_resume_requires_matching_two_pass_provenance(tmp_path):
    paths = induct.artifact_layout(tmp_path, "Qwen3.5-397B-A17B", "oq4++", ["oq4+"])
    for key in ("model", "calib", "triattn"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_bytes(b"HFQM" + b"\0" * 28)
    paths["two_pass_manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["two_pass_manifest"].write_text(
        json.dumps(
            {
                "status": "complete",
                "recipe_fingerprint": "sha256:expected",
                "fingerprints": {
                        "calibration_artifact": "fnv64:calib",
                        "cask_artifact": "fnv64:cask",
                        "quantized_artifact": "fnv64:model",
                },
                "source_reads": {"missing_logical": [], "duplicate_logical": []},
                "cask": {
                    "artifact_fingerprint": "fnv64:cask",
                    "metadata": {
                        "artifact_kind": "triattn",
                        "package_schema": "hipfire.triattn.v2",
                        "layers": [{"physical_layer": 0}],
                    },
                },
                "calibration_audit": {
                    "schema": "hipfire.calibration_audit.v1",
                    "valid": True,
                    "artifact_fingerprint": "fnv64:calib",
                    "errors": [],
                },
            }
        )
    )

    assert induct.target_stage_complete(paths, "sha256:expected")
    assert not induct.target_stage_complete(paths, "sha256:different")
    manifest = json.loads(paths["two_pass_manifest"].read_text())
    manifest.pop("calibration_audit")
    paths["two_pass_manifest"].write_text(json.dumps(manifest))
    assert not induct.target_stage_complete(paths, "sha256:expected")


def test_induction_target_fingerprint_matches_two_pass_recipe(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("calibration text")
    paths = induct.artifact_layout(tmp_path, "Qwen3.5-397B-A17B", "oq4.25++", ["oq4+"])
    target = tmp_path / "target"
    target.mkdir()
    common = {
        "target": target,
        "corpus": corpus,
        "paths": paths,
        "quant_format": "oq4.25++",
        "n_sequences": 128,
        "ctx_len": 2048,
        "batch_size": 64,
        "time_tile": 32,
        "max_rows": 2048,
        "layer_prefetch_bytes": 16 * 1024**3,
        "kldref_topk": 64,
        "min_expert_activations": 2048,
        "expert_capture_target": 4096,
        "expert_capture_tile_rows": 256,
        "required_expert_fraction": 1.0,
        "sampling_seed": 1,
        "expert_coverage_policy": "preserve-undercovered",
        "quant_args": ["--awq", "--ldlq"],
    }

    induction_fingerprint = induct._target_recipe_fingerprint(**common)
    two_pass_recipe = two_pass.recipe_manifest(
        model=target,
        calib=paths["calib"],
        output=paths["model"],
        quant_format=common["quant_format"],
        corpus=corpus,
        n_sequences=common["n_sequences"],
        ctx_len=common["ctx_len"],
        batch_size=common["batch_size"],
        time_tile=common["time_tile"],
        max_rows=common["max_rows"],
        layer_prefetch_bytes=common["layer_prefetch_bytes"],
        kldref_topk=common["kldref_topk"],
        min_expert_activations=common["min_expert_activations"],
        expert_capture_target=common["expert_capture_target"],
        expert_capture_tile_rows=common["expert_capture_tile_rows"],
        required_expert_fraction=common["required_expert_fraction"],
        sampling_seed=common["sampling_seed"],
        expert_coverage_policy=common["expert_coverage_policy"],
        quant_args=common["quant_args"],
        cask_output=paths["triattn"],
    )

    assert induction_fingerprint == two_pass_recipe["recipe_fingerprint"]


def test_main_dry_run_prints_native_two_pass_plan_without_running_tools(tmp_path, monkeypatch, capsys):
    target, draft = _configs(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("calibration text")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--target",
            str(target),
            "--dflash-source",
            str(draft),
            "--model-name",
            "Qwen3.5-397B-A17B",
            "--corpus",
            str(corpus),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--stage",
            "target",
            "--dry-run",
        ],
    )

    induct.main()

    output = capsys.readouterr().out
    assert "hipfire-coexistence calibrate" not in output  # printed as a safely quoted command
    assert "two_pass_quantize.py" in output
    assert "--manifest" in output
    assert "--python" not in output
    assert "Qwen3.5-397B-A17B--oq4.25++.hfq" in output
    assert "Qwen3.5-397B-A17B--dflash.oq4+.hfq" in output
    assert "Qwen3.5-397B-A17B--dflash.triattn.oq4.25++.hfq" in output
    assert "--format oq4.25++" in output
    assert f"--layer-prefetch-bytes {16 * 1024**3}" in output
    assert "--min-expert-activations 2048" in output
    assert "--expert-capture-target 4096" in output
    assert "--expert-capture-tile-rows 256" in output
    assert "--required-expert-fraction 1.0" in output
    assert "--sampling-seed 1" in output
    assert "--expert-coverage-policy preserve-undercovered" in output
    assert "--calibration-segment-layers 0" in output


def test_target_stage_failure_reflects_interrupted_two_pass_manifest(tmp_path):
    two_pass_manifest = tmp_path / "two-pass.json"
    two_pass_manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "status": "quantization_interrupted",
                "failure": {"kind": "signal", "returncode": 143, "signal": 15},
            }
        )
    )
    paths = {"two_pass_manifest": two_pass_manifest}

    assert (
        induct._stage_failure_status("target", paths, RuntimeError("wrapper exited"))
        == "interrupted"
    )
    assert induct._stage_failure_status("bundle", paths, RuntimeError("failed")) == "failed"
    assert induct._stage_failure_status("dflash", paths, KeyboardInterrupt()) == "interrupted"
