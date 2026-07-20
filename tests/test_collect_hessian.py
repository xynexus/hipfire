import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_hessian.py"
SPEC = importlib.util.spec_from_file_location("collect_hessian", SCRIPT)
assert SPEC and SPEC.loader
collect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect)


def test_local_corpus_tokenization_is_truncated_to_calibration_budget(tmp_path):
    corpus = tmp_path / "oversized.txt"
    corpus.write_text(" ".join(f"token-{index}" for index in range(100)))

    class RecordingTokenizer:
        model_max_length = 262144

        def __init__(self):
            self.calls = []

        def __call__(self, text, **kwargs):
            self.calls.append((text, kwargs))
            token_count = len(text.split())
            if kwargs.get("truncation"):
                token_count = min(token_count, kwargs["max_length"])
            return SimpleNamespace(input_ids=torch.arange(token_count).unsqueeze(0))

    tokenizer = RecordingTokenizer()
    sequences = collect.load_calibration_text(str(corpus), 2, 4, tokenizer)

    assert [sequence.tolist() for sequence in sequences] == [list(range(4)), list(range(4, 8))]
    assert len(tokenizer.calls) == 1
    _text, kwargs = tokenizer.calls[0]
    assert len(_text) < len(corpus.read_text())
    assert kwargs == {"return_tensors": "pt", "truncation": True, "max_length": 8}


def _read_hfqm(path: Path):
    data = path.read_bytes()
    magic, version, arch_id, n_entries, metadata_offset, data_offset = struct.unpack_from("<4sIIIQQ", data, 0)
    assert magic == b"HFQM"
    assert version == 2
    decoder = json.JSONDecoder()
    metadata_text = data[metadata_offset:data_offset].decode("utf-8", errors="ignore")
    metadata, json_chars = decoder.raw_decode(metadata_text)
    pos = metadata_offset + len(metadata_text[:json_chars].encode())
    (index_count,) = struct.unpack_from("<I", data, pos)
    pos += 4
    entries = {}
    for _ in range(index_count):
        (name_len,) = struct.unpack_from("<H", data, pos)
        pos += 2
        name = data[pos : pos + name_len].decode()
        pos += name_len
        quant_type, ndim = struct.unpack_from("<BB", data, pos)
        pos += 2
        shape = list(struct.unpack_from(f"<{ndim}I", data, pos))
        pos += ndim * 4
        group_size, data_len, offset_units = struct.unpack_from("<IQQ", data, pos)
        pos += 20
        offset = offset_units * 32
        entries[name] = {
            "quant_type": quant_type,
            "shape": shape,
            "group_size": group_size,
            "payload": data[offset : offset + data_len],
        }
    assert index_count == n_entries
    return arch_id, metadata, entries


def test_writes_canonical_compact_hessian_and_exact_imatrix(tmp_path):
    acc = collect.HessianAccumulator("model.layers.0.self_attn.q_proj", 3, "cpu")
    acc.update(torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]]))
    out = tmp_path / "Tiny-1B.calib.hfq"

    collect.write_calibration_hfq(out, {acc.name: acc}, {"source_format": "safetensors"})

    arch_id, metadata, entries = _read_hfqm(out)
    assert arch_id == 0
    assert metadata["artifact_kind"] == "calibration"
    assert metadata["n_hessian"] == 1
    assert metadata["n_imatrix"] == 1
    h = entries[f"{acc.name}.hessian"]
    im = entries[f"{acc.name}.imatrix"]
    assert h["quant_type"] == 130
    assert h["shape"] == [3, 3]
    assert len(h["payload"]) == 18
    assert im["quant_type"] == 2
    assert np.frombuffer(im["payload"], dtype="<f4").tolist() == [2.5, 2.0, 5.0]
    assert np.frombuffer(h["payload"][:12], dtype="<f4").tolist() == [2.5, 2.0, 5.0]


def test_reference_offload_index_points_at_original_shards(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    index = {
        "weight_map": {
            "model.language_model.layers.0.proj.weight": "model-00001-of-00002.safetensors",
            "model.language_model.layers.1.proj.weight": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    model_to_stored = {
        "model.layers.0.proj.weight": "model.language_model.layers.0.proj.weight",
        "model.layers.1.proj.weight": "model.language_model.layers.1.proj.weight",
    }
    device_map = {"model.layers.0": 0, "model.layers.1": "disk"}

    offload = collect.build_reference_offload_index(tmp_path, model_to_stored, device_map)

    assert set(offload) == {"model.layers.1.proj.weight"}
    assert offload["model.layers.1.proj.weight"]["weight_name"].endswith("layers.1.proj.weight")
    assert offload["model.layers.1.proj.weight"]["safetensors_file"] == str(
        tmp_path / "model-00002-of-00002.safetensors"
    )


def test_resolves_huggingface_cache_root_via_main_ref(tmp_path):
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "main").write_text("abc123\n")

    assert collect.resolve_model_path(str(tmp_path)) == snapshot


def test_expert_name_splits_fused_safetensors_parent():
    parent = "model.language_model.layers.4.mlp.experts.gate_up_proj"
    assert collect.split_expert_name(parent, 17) == "model.language_model.layers.4.mlp.experts.17.gate_up_proj"


def test_dense_projections_with_identical_inputs_share_one_accumulator_group():
    assert collect.shared_input_group("model.layers.3.self_attn.q_proj") == collect.shared_input_group(
        "model.layers.3.self_attn.v_proj"
    )
    assert collect.shared_input_group("model.layers.0.linear_attn.in_proj_qkv") == collect.shared_input_group(
        "model.layers.0.linear_attn.in_proj_a"
    )
    assert collect.shared_input_group("model.layers.2.mlp.shared_expert.gate_proj") == collect.shared_input_group(
        "model.layers.2.mlp.shared_expert.up_proj"
    )
    assert collect.shared_input_group("model.layers.3.self_attn.o_proj") != collect.shared_input_group(
        "model.layers.3.self_attn.q_proj"
    )


def test_router_histogram_matches_native_metadata_shape():
    hist = collect.MoeRouterHistogram(num_experts=4, k_top=2, num_layers=2)
    hist.record(
        1,
        torch.tensor([[2, 1], [2, 3]]),
        torch.tensor([[0.75, 0.25], [0.6, 0.4]]),
    )

    meta = hist.to_metadata()
    assert meta["routed_tokens"] == 2
    assert meta["routed_slots"] == 4
    assert meta["top1_histogram"] == [0, 0, 2, 0]
    assert meta["topk_histogram"] == [0, 1, 2, 1]
    assert meta["per_layer_topk"][1] == [0, 1, 2, 1]
    assert meta["top_cooccurrence"] == [[1, 2, 1], [2, 3, 1]]


def test_fused_moe_capture_preserves_forward_and_emits_split_imatrices():
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 2
            self.gate_up_proj = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 32)
            self.down_proj = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 16)
            self.act_fn = torch.nn.functional.silu

        def forward(self, hidden, indices, weights):
            out = torch.zeros_like(hidden)
            mask = torch.nn.functional.one_hot(indices, num_classes=self.num_experts).permute(2, 1, 0)
            for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten():
                pos, token = torch.where(mask[expert])
                gate, up = torch.nn.functional.linear(hidden[token], self.gate_up_proj[expert]).chunk(2, -1)
                mid = self.act_fn(gate) * up
                value = torch.nn.functional.linear(mid, self.down_proj[expert]) * weights[token, pos, None]
                out.index_add_(0, token, value)
            return out

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = Experts()
    experts = model.model.layers[0].mlp.experts
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    indices = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
    expected = experts(hidden, indices, weights)
    stored = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    }
    accs = {}

    hist, restores = collect.install_fused_moe_capture(model, stored, accs, "cpu", 1)
    actual = experts(hidden, indices, weights)
    for restore in restores:
        restore()

    torch.testing.assert_close(actual, expected)
    assert hist.to_metadata()["routed_tokens"] == 3
    assert set(accs) == {
        "model.language_model.layers.0.mlp.experts.0.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.0.down_proj",
        "model.language_model.layers.0.mlp.experts.1.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.1.down_proj",
    }
    assert all(not acc.has_hessian for acc in accs.values())


def test_calibration_spool_combines_layer_payloads_and_native_kldref(tmp_path):
    layer0 = collect.HessianAccumulator("model.layers.0.self_attn.q_proj", 2, "cpu")
    layer0.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    layer1 = collect.HessianAccumulator("model.layers.1.mlp.experts.7.down_proj", 2, "cpu", hessian=False)
    layer1.update(torch.tensor([[5.0, 6.0]]))
    out = tmp_path / "Tiny-1B.calib.hfq"

    with collect.CalibrationSpool(tmp_path / "spool") as spool:
        spool.add_accumulators({layer0.name: layer0})
        spool.add_accumulators({layer1.name: layer1})
        kldref = collect.KldRefAccumulator(top_k=2)
        kldref.update(torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.0, 2.0]]))
        spool.add_kldref(kldref)
        spool.write_hfq(out, {"source_format": "safetensors"})

    _arch_id, metadata, entries = _read_hfqm(out)
    assert metadata["artifacts"] == ["hessian", "imatrix", "kldref"]
    assert metadata["kldref"] == {"n_positions": 2, "top_k": 2}
    assert metadata["n_hessian"] == 1
    assert metadata["n_imatrix"] == 2
    assert set(entries) == {
        f"{layer0.name}.hessian",
        f"{layer0.name}.imatrix",
        f"{layer1.name}.imatrix",
        "lm_head.kldref_idx",
        "lm_head.kldref_logit",
        "lm_head.kldref_logz",
    }
    assert entries["lm_head.kldref_idx"]["quant_type"] == 2
    assert entries["lm_head.kldref_idx"]["shape"] == [2, 2]
    assert np.frombuffer(entries["lm_head.kldref_idx"]["payload"], dtype="<f4").tolist() == [1.0, 2.0, 0.0, 2.0]
    assert np.frombuffer(entries["lm_head.kldref_logit"]["payload"], dtype="<f4").tolist() == [4.0, 2.0, 3.0, 2.0]
    np.testing.assert_allclose(
        np.frombuffer(entries["lm_head.kldref_logz"]["payload"], dtype="<f4"),
        torch.logsumexp(torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.0, 2.0]]), dim=-1).numpy(),
    )


def test_layer_parameter_plan_keeps_shared_tensors_and_natural_layer_order():
    names = [
        "model.layers.10.mlp.experts.down_proj",
        "model.norm.weight",
        "model.layers.2.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.1.linear_attn.in_proj_qkv.weight",
    ]

    plan = collect.layer_parameter_plan(names)

    assert plan["embedding"] == ["model.embed_tokens.weight"]
    assert list(plan["layers"]) == [1, 2, 10]
    assert plan["layers"][10] == ["model.layers.10.mlp.experts.down_proj"]
    assert plan["final"] == ["lm_head.weight", "model.norm.weight"]


def test_layer_stream_executes_tiny_qwen35_checkpoint_with_one_read_per_tensor(tmp_path):
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=2,
        num_experts=4,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config)
    # Match the published checkpoint's fused-expert safetensors layout instead
    # of save_pretrained's split expert export conversion.
    state = {
        (name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name): value.contiguous()
        for name, value in model.state_dict().items()
    }
    config.save_pretrained(tmp_path)
    save_file(state, tmp_path / "model.safetensors")
    del model, state
    out = tmp_path / "Tiny-Qwen3.5.calib.hfq"

    collect.collect_layer_streamed(
        model_path=tmp_path,
        out_path=out,
        seqs=[torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8])],
        dtype=torch.float32,
        device="cpu",
        accum_device="cpu",
        batch_size=1,
        tensor_filter=None,
        kldref_enabled=True,
        kldref_topk=2,
        logit_batch_tokens=2,
        spool_parent=tmp_path,
    )

    _arch_id, metadata, entries = _read_hfqm(out)
    assert metadata["collector_mode"] == "layer_stream"
    assert metadata["safetensors_teacher_reads"] == len(collect._weight_map(tmp_path))
    assert metadata["kldref"] == {"n_positions": 8, "top_k": 2}
    assert metadata["moe_router_histogram"]["k_top"] == 2
    assert "lm_head.kldref_logz" in entries
