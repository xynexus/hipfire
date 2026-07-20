import importlib.util
import json
import os
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "induct_model.py"
SPEC = importlib.util.spec_from_file_location("induct_model", SCRIPT)
assert SPEC and SPEC.loader
induct = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(induct)


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


def test_artifact_layout_matches_registry_sidecar_names(tmp_path):
    paths = induct.artifact_layout(
        tmp_path, "Qwen3.5-397B-A17B", "oq4++", ["bf16", "f16"]
    )

    assert paths["model"] == tmp_path / "models" / "Qwen3.5-397B-A17B.oq4++.hfq"
    assert paths["dflash_bf16"] == tmp_path / "drafts" / "Qwen3.5-397B-A17B-BF16.dflash.hfq"
    assert paths["dflash_f16"] == tmp_path / "drafts" / "Qwen3.5-397B-A17B-F16.dflash.hfq"
    assert paths["triattn"] == tmp_path / "triattn" / "Qwen3.5-397B-A17B.triattn.hfq"
    assert paths["calib"] == tmp_path / "calib" / "Qwen3.5-397B-A17B.calib.hfq"
    assert paths["manifest"] == tmp_path / "induction" / "Qwen3.5-397B-A17B.oq4++" / "manifest.json"


def test_default_quant_format_is_mixed_oq425_double_plus():
    assert induct.DEFAULT_QUANT_FORMAT == "oq4.25++"
    assert induct.DEFAULT_DFLASH_FORMATS == ("bf16", "f16")
    assert induct.DEFAULT_LAYER_PREFETCH_BYTES == 16 * 1024**3


def test_mixed_opus_format_keeps_calibration_recipe_and_canonical_name(tmp_path):
    paths = induct.artifact_layout(tmp_path, "Qwen3.5-397B-A17B", "oq4.5++", ["f16"])

    assert paths["model"].name == "Qwen3.5-397B-A17B.oq4.5++.hfq"
    assert paths["dflash_f16"].name == "Qwen3.5-397B-A17B-F16.dflash.hfq"
    assert induct.default_quant_args("oq4.5++") == ["--awq", "--ldlq"]


def test_commands_compose_existing_converters_with_scoped_gpu_stages(tmp_path):
    target, draft = _configs(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("calibration text")
    paths = induct.artifact_layout(
        tmp_path, "Qwen3.5-397B-A17B", "oq4++", ["bf16", "f16"]
    )

    commands = induct.build_stage_commands(
        target=target,
        draft=draft,
        corpus=corpus,
        paths=paths,
        quant_format="oq4++",
        dflash_formats=["bf16", "f16"],
        n_sequences=128,
        ctx_len=2048,
        batch_size=64,
        time_tile=32,
        max_rows=2048,
        layer_prefetch_bytes=16 * 1024**3,
        kldref_topk=64,
        triattn_max_tokens=100_000,
        triattn_chunk_len=1024,
        python="python3",
        hipfire="target/release/hipfire",
        coexistence="target/release/hipfire-coexistence",
        quantizer="target/release/hipfire-quantize",
        dflash_converter="target/release/dflash_convert",
        triattn_bin="target/release/examples/triattn_validate",
        quant_args=["--awq", "--ldlq"],
    )

    assert commands["dflash"][0][-4:] == [
        "--input",
        str(draft),
        "--output",
        str(paths["dflash_bf16"]),
    ]
    assert commands["dflash"][1] == [
        "target/release/dflash_convert",
        "--f16",
        "--input",
        str(draft),
        "--output",
        str(paths["dflash_f16"]),
    ]
    target_cmd = commands["target"][0]
    assert target_cmd[:2] == ["python3", "scripts/two_pass_quantize.py"]
    assert target_cmd[target_cmd.index("--calib") + 1] == str(paths["calib"])
    assert target_cmd[target_cmd.index("--coexistence") + 1] == "target/release/hipfire-coexistence"
    assert "--python" not in target_cmd
    assert target_cmd[target_cmd.index("--manifest") + 1] == str(paths["two_pass_manifest"])
    assert target_cmd[target_cmd.index("--batch-size") + 1] == "64"
    assert target_cmd[target_cmd.index("--time-tile") + 1] == "32"
    assert target_cmd[target_cmd.index("--max-rows") + 1] == "2048"
    assert target_cmd[target_cmd.index("--layer-prefetch-bytes") + 1] == str(16 * 1024**3)
    assert target_cmd[-2:] == ["--awq", "--ldlq"]
    triattn = commands["triattn"][0]
    assert triattn[:6] == [
        "target/release/hipfire",
        "lock",
        "run",
        "induct-triattn",
        "--",
        "target/release/examples/triattn_validate",
    ]
    assert str(paths["model"]) in triattn
    assert triattn[triattn.index("--sidecar") + 1] == str(paths["triattn"])


def test_dflash_format_defaults_preserve_bf16_and_keep_f16_explicit():
    assert induct._dflash_format_args("bf16") == []
    assert induct._dflash_format_args("f16") == ["--f16"]


def test_resume_only_skips_artifacts_with_expected_magic(tmp_path):
    hfq = tmp_path / "model.hfq"
    triattn = tmp_path / "model.triattn.hfq"
    hfq.write_bytes(b"HFQM" + b"\0" * 28)
    triattn.write_bytes(b"TRIA" + b"\0" * 28)

    assert induct.artifact_is_valid(hfq, b"HFQM")
    assert induct.artifact_is_valid(triattn, b"TRIA")
    assert not induct.artifact_is_valid(hfq, b"TRIA")


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


def test_target_resume_requires_matching_two_pass_provenance(tmp_path):
    paths = induct.artifact_layout(tmp_path, "Qwen3.5-397B-A17B", "oq4++", ["bf16", "f16"])
    for key in ("model", "calib"):
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
                    "quantized_artifact": "fnv64:model",
                },
                "source_reads": {"missing_logical": [], "duplicate_logical": []},
            }
        )
    )

    assert induct.target_stage_complete(paths, "sha256:expected")
    assert not induct.target_stage_complete(paths, "sha256:different")


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
    assert "Qwen3.5-397B-A17B.oq4.25++.hfq" in output
    assert "Qwen3.5-397B-A17B-BF16.dflash.hfq" in output
    assert "Qwen3.5-397B-A17B-F16.dflash.hfq" in output
    assert "--format oq4.25++" in output
    assert f"--layer-prefetch-bytes {16 * 1024**3}" in output
