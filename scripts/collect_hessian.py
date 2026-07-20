#!/usr/bin/env python3
"""Collect a canonical Hipfire calibration artifact from HF safetensors.

Phase A Stage B Phase 1.1 per `docs/plans/gptq.md` v2.

`crates/hipfire-runtime/examples/imatrix_collect.rs` is a llama.cpp subprocess
wrapper (no native forward pass), so we use HF transformers + ROCm PyTorch here
for the calibration forward pass. The output HFQM v2 package is consumed
directly by `hipfire-quantize`.

Loads a BF16 model, runs a forward pass on calibration tokens, registers
`nn.Linear` forward hooks that accumulate per-tensor outer products
`H_t = (1/N) * sum_t x_t · x_t^T`, and writes `<model>.calib.hfq`.

Calibration subset matches the GPTQ paper's scale: 128 sequences × 2048
tokens = 262144 calibration tokens. The Hessian is a smoothed expectation
that converges well at this scale; tokenizer disagreement with hipfire's
tokenizer is averaged out (per `docs/plans/gptq_plan_rev_synthesis.md`
Topic 6).

Tensor coverage: all `nn.Linear` modules whose name matches the GPTQ
whitelist (mirrors the quantizer's `awq_eligible` plus the non-AWQ MQ4
projections we want to GPTQ-quantize: o_proj/out_proj/down_proj).

Dense projections carry exact-F32 imatrices and compact Hessians (exact F32
diagonal plus BF16 strict lower triangle). Fused routed experts carry split,
per-expert imatrices plus native-compatible router histogram metadata. With
reference offload, layers page from the original safetensor shards rather than
creating a second disk-offload copy of very large checkpoints.

Usage:
    .venv/bin/python3 scripts/collect_hessian.py \\
        --model    <hf-model-id-or-path> \\
        --output   <Model-Size.calib.hfq> \\
        [--n-sequences 128] \\
        [--ctx-len    2048] \\
        [--corpus     <hf-dataset-id-or-text-file>] \\
        [--device     cuda] \\
        [--dtype      bfloat16]

Examples:
    # Qwen3.5-9B, 128 sequences × 2048 ctx from wikitext-2 train
    .venv/bin/python3 scripts/collect_hessian.py \\
        --model /data/cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202.../ \\
        --output ~/.hipfire/calib/Qwen3.5-9B.calib.hfq
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
import tempfile
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _mask_broken_torchvision_for_text_only_calibration() -> None:
    """Treat broken torchvision as unavailable for text-only HF model loading.

    Some Transformers releases route generic modeling imports through
    `processing_utils` → `image_utils`. A mismatched host torchvision can then
    fail while registering optional image ops even for text-only CausalLM
    calibration. The Hessian collector does not need torchvision, so make the
    availability probe return False when importing torchvision itself fails.
    """
    try:
        import torchvision  # noqa: F401

        return
    except Exception as exc:  # pragma: no cover - host-package dependent.
        print(
            f"      WARN: masking unusable torchvision for text-only calibration: {exc}",
            file=sys.stderr,
        )

    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            del sys.modules[name]

    def _false(*_args, **_kwargs) -> bool:
        return False

    try:
        import transformers.utils as tf_utils
        import transformers.utils.import_utils as import_utils

        import_utils.is_torchvision_available = _false
        import_utils.is_torchvision_v2_available = _false
        tf_utils.is_torchvision_available = _false
        tf_utils.is_torchvision_v2_available = _false
    except Exception as exc:  # pragma: no cover - defensive best effort.
        print(f"      WARN: failed to mask torchvision availability: {exc}", file=sys.stderr)


# Tensor name patterns that should get GPTQ Hessians. Matches hipfire's
# AWQ whitelist plus the non-AWQ MQ4 projections (o_proj/out_proj/down_proj).
# Pattern check is done on the FULL module name (e.g.
# "model.layers.0.self_attn.q_proj"); we match against the last 2-3 components.
GPTQ_TARGET_SUFFIXES = (
    # Attention input projections
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv_proj",
    # Attention output (no AWQ but yes GPTQ)
    "o_proj",
    # MLP
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
    # LFM2 dense FFN uses short w1/w2/w3 names instead of gate/up/down_proj.
    "w1",
    "w2",
    "w3",
    # Linear-attention (Gated DeltaNet)
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    # LFM2 LIV conv mixer has a plain conv.in_proj module.
    "in_proj",
    "out_proj",
    # MoE router
    "gate",
)


def is_gptq_target(name: str) -> bool:
    """Returns True if this module name should have a Hessian collected."""
    last = name.rsplit(".", 1)[-1]
    if last in GPTQ_TARGET_SUFFIXES:
        return True
    # `mlp.gate.weight` (MoE router) — matched by trailing ".gate" — but we
    # need to disambiguate from `gate_proj`. The check above on the literal
    # last segment handles both.
    return False


def _safetensors_keys(model_path: str) -> set[str]:
    """Returns the set of all tensor keys stored in the safetensors files,
    stripped of `.weight` suffix to match against module names.

    Used to translate HF's flattened in-memory module names (e.g.
    `model.layers.0.q_proj` for Qwen3.5 ForCausalLM) to the canonical
    safetensors / .hfq naming (`model.language_model.layers.0.q_proj`).
    """
    import json

    p = Path(model_path)
    keys: set[str] = set()
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        with idx.open() as f:
            data = json.load(f)
        keys = set(data["weight_map"].keys())
    else:
        # Single-file safetensors path
        from safetensors import safe_open  # type: ignore[import-not-found]

        for st_file in p.glob("*.safetensors"):
            with safe_open(st_file, framework="pt") as f:
                keys.update(f.keys())
    # Hooks use module names rather than parameter names. Most checkpoints use
    # `.weight`; fused Qwen MoE tensors (`experts.gate_up_proj`) do not.
    return {k.removesuffix(".weight") for k in keys}


def _translate_to_stored_name(mod_name: str, stored_keys: set[str]) -> str | None:
    """Find the canonical safetensors-stored name for a hooked module.

    Strategy: longest-suffix match. If `model.layers.0.q_proj` is the
    in-memory name, look for any stored key ending in
    `.layers.0.q_proj` — and pick the longest match (typically
    `model.language_model.layers.0.q_proj` for Qwen3.5 ConditionalGen).

    Returns the stored name (without `.weight`) on success, or None if no
    suffix-matching key exists in the safetensors (which would mean the
    module is in-memory only and not stored — unusual; skip with a warning).
    """
    # Strip the leading "model." (common prefix in both naming conventions).
    # We match on the trailing portion after the FIRST "." — typically
    # ".layers.N.<...>" — to be robust against the language_model. insertion.
    if mod_name in stored_keys:
        return mod_name
    # Build a candidate suffix by stripping a single leading component
    # ("model.") — most common HF wrapper pattern.
    trail = mod_name.split(".", 1)[1] if "." in mod_name else mod_name
    # Find any stored key whose suffix matches.
    matches = [k for k in stored_keys if k.endswith(trail)]
    if not matches:
        return None
    # If multiple, pick the longest (most fully-qualified — e.g. prefers
    # `model.language_model.layers.0.q_proj` over a hypothetical bare
    # `layers.0.q_proj`).
    matches.sort(key=len, reverse=True)
    return matches[0]


def _matches_tensor_filter(stored_name: str, pattern: str | None) -> bool:
    """Substring filter over canonical stored tensor names.

    The collector stores hook keys without the trailing `.weight`, while humans
    usually copy names with `.weight` from safetensors or HFQ logs. Accept both.
    """
    if not pattern:
        return True
    pat = pattern.removesuffix(".weight")
    return pat in stored_name


def shared_input_group(stored_name: str) -> str:
    """Return a key for projections that consume the exact same activation.

    Sharing one accumulator avoids repeating identical KxK outer products and
    cuts the 397B collector's dense-accumulator footprint substantially.
    """
    parts = stored_name.rsplit(".", 1)
    if len(parts) != 2:
        return stored_name
    prefix, leaf = parts
    if leaf in {"q_proj", "k_proj", "v_proj"} and prefix.endswith(".self_attn"):
        return f"{prefix}.__qkv_input__"
    if leaf in {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"} and prefix.endswith(".linear_attn"):
        return f"{prefix}.__input__"
    if leaf in {"gate_proj", "up_proj"}:
        return f"{prefix}.__gate_up_input__"
    return stored_name


class HessianAccumulator:
    """One per nn.Linear we hook. Accumulates H = sum_t x_t · x_t^T (FP32, K×K).

    Two accumulation modes (`accum_device`):

    - "gpu" (default, ~20× faster): keep the K×K accumulator on the SAME device
      as the module's activations and do the outer product as an on-GPU
      `addmm_` (H += xᵀx). No per-sequence host transfer of the [N,K]
      activations and no slow CPU BLAS gemm — only the final K×K block is
      copied to host. VRAM cost is Σ K² across hooked linears; fits easily for
      ≤9B on big-VRAM boxes (e.g. Strix Halo 96 GB). The accumulator is
      allocated lazily on first update so it lands on the activation's own
      device (correct under HF `device_map="auto"` offload).

    - "cpu" (fallback for VRAM-constrained big models, e.g. 27B on 24 GB):
      transfer [N,K] to host and accumulate with numpy BLAS — the original
      behavior. Slower but bounded VRAM.

    Promoted to FP64 only at the quantizer for Cholesky (per plan v2 §2).
    """

    def __init__(self, name: str, K: int, accum_device: str = "gpu", hessian: bool = True) -> None:
        self.name = name
        self.K = K
        self.accum_device = accum_device
        self.has_hessian = hessian
        # Lazily allocated on first update (GPU: on the activation's device;
        # CPU: numpy host array). K=12288 → 576 MB per Hessian either way.
        self.H = None  # torch.Tensor (gpu) or np.ndarray (cpu)
        # Kept independently so compact Hessians preserve the exact imatrix
        # diagonal, matching the native collector's consistency contract.
        self.diag = None  # torch.Tensor (gpu) or np.ndarray (cpu)
        self.n_tokens = 0

    def update(self, x: "torch.Tensor") -> None:
        """x: shape [num_tokens, K], any dtype, on the forward device."""
        assert x.ndim == 2 and x.shape[1] == self.K, f"{self.name}: shape mismatch {tuple(x.shape)} vs K={self.K}"
        if self.accum_device == "gpu":
            xf = x.to(dtype=torch.float32)
            if self.diag is None:
                self.diag = torch.zeros(self.K, dtype=torch.float32, device=xf.device)
            self.diag.add_(xf.square().sum(dim=0))
            if self.has_hessian and self.H is None:
                self.H = torch.zeros(self.K, self.K, dtype=torch.float32, device=xf.device)
            if self.has_hessian:
                # H += xᵀ·x as a single fused in-place GPU gemm.
                self.H.addmm_(xf.t(), xf)
        else:
            x_np = x.to(dtype=torch.float32).detach().cpu().numpy()
            if self.diag is None:
                self.diag = np.zeros(self.K, dtype=np.float32)
            self.diag += np.square(x_np, dtype=np.float32).sum(axis=0, dtype=np.float32)
            if self.has_hessian and self.H is None:
                self.H = np.zeros((self.K, self.K), dtype=np.float32)
            if self.has_hessian:
                self.H += x_np.T @ x_np
        self.n_tokens += x.shape[0]

    def finalize(self) -> np.ndarray:
        """Returns H / n_tokens (the expectation E[x xᵀ]) as a host fp32 array."""
        if self.n_tokens == 0 or self.H is None:
            return np.zeros((self.K, self.K), dtype=np.float32)  # caller skips
        if isinstance(self.H, np.ndarray):
            return self.H / self.n_tokens
        # GPU tensor → divide on device, then one K×K host copy.
        return (self.H / self.n_tokens).to(dtype=torch.float32).cpu().numpy()

    def finalize_diag(self) -> np.ndarray:
        """Return exact E[x²] as host fp32."""
        if self.n_tokens == 0 or self.diag is None:
            return np.zeros(self.K, dtype=np.float32)
        if isinstance(self.diag, np.ndarray):
            return self.diag / self.n_tokens
        return (self.diag / self.n_tokens).to(dtype=torch.float32).cpu().numpy()


def split_expert_name(parent: str, expert_idx: int) -> str:
    """Turn a fused `...experts.<projection>` key into Hipfire's split key."""
    marker = ".experts."
    if marker not in parent:
        raise ValueError(f"not a fused expert tensor name: {parent}")
    prefix, projection = parent.rsplit(marker, 1)
    return f"{prefix}.experts.{expert_idx}.{projection}"


class MoeRouterHistogram:
    """CPU-side equivalent of Hipfire's router telemetry metadata."""

    def __init__(self, num_experts: int, k_top: int, num_layers: int) -> None:
        self.num_experts = num_experts
        self.k_top = k_top
        self.routed_tokens = 0
        self.routed_slots = 0
        self.top1 = np.zeros(num_experts, dtype=np.uint64)
        self.topk = np.zeros(num_experts, dtype=np.uint64)
        self.per_layer = np.zeros((num_layers, num_experts), dtype=np.uint64)
        self.cooccurrence: Counter[tuple[int, int]] = Counter()

    def record(self, layer_idx: int, indices: torch.Tensor, weights: torch.Tensor) -> None:  # noqa: ARG002
        selected = indices.detach().to(device="cpu", dtype=torch.int64)
        selected = selected.reshape(-1, selected.shape[-1]).numpy()
        self.routed_tokens += selected.shape[0]
        for row in selected:
            valid = [int(e) for e in row[: self.k_top] if 0 <= int(e) < self.num_experts]
            if valid:
                self.top1[valid[0]] += 1
            for expert in valid:
                self.topk[expert] += 1
                self.per_layer[layer_idx, expert] += 1
                self.routed_slots += 1
            for i, a in enumerate(valid):
                for b in valid[i + 1 :]:
                    self.cooccurrence[min(a, b), max(a, b)] += 1

    def to_metadata(self) -> dict:
        pairs = sorted(self.cooccurrence.items(), key=lambda item: (-item[1], item[0]))[:64]
        return {
            "num_experts": self.num_experts,
            "k_top": self.k_top,
            "routed_tokens": self.routed_tokens,
            "routed_slots": self.routed_slots,
            "top1_histogram": self.top1.tolist(),
            "topk_histogram": self.topk.tolist(),
            "per_layer_topk": self.per_layer.tolist(),
            "top_cooccurrence": [[a, b, count] for ((a, b), count) in pairs],
        }


def install_fused_moe_capture(
    model: torch.nn.Module,
    stored_names: set[str],
    accs: dict[str, HessianAccumulator],
    accum_device: str,
    num_layers: int,
    tensor_filter: str | None = None,
    allowed_layers: set[int] | None = None,
    router_hist: MoeRouterHistogram | None = None,
) -> tuple[MoeRouterHistogram, list]:
    """Instrument fused 3-D expert modules without recomputing their matmuls.

    Transformers exposes Qwen3.5 routed experts as raw `[E,M,K]` parameters,
    not `nn.Linear` children, so ordinary hooks cannot see the down-projection
    input. This wrapper mirrors that tiny expert loop, captures both inputs,
    and retains byte-for-byte forward semantics.
    """
    candidates = []
    for mod_name, module in model.named_modules():
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        if not isinstance(gate_up, torch.nn.Parameter) or gate_up.ndim != 3:
            continue
        if not isinstance(down, torch.nn.Parameter) or down.ndim != 3:
            continue
        match = re.search(r"\.layers\.(\d+)\.", f".{mod_name}.")
        if match is None:
            continue
        if allowed_layers is not None and int(match.group(1)) not in allowed_layers:
            continue
        gate_parent = _translate_to_stored_name(f"{mod_name}.gate_up_proj", stored_names)
        down_parent = _translate_to_stored_name(f"{mod_name}.down_proj", stored_names)
        if gate_parent and down_parent:
            candidates.append((int(match.group(1)), module, gate_parent, down_parent))
    if not candidates:
        return router_hist or MoeRouterHistogram(0, 0, num_layers), []

    num_experts = int(candidates[0][1].gate_up_proj.shape[0])
    hist = router_hist or MoeRouterHistogram(num_experts, 0, num_layers)
    if hist.num_experts not in (0, num_experts):
        raise ValueError(f"router histogram expects {hist.num_experts} experts, layer has {num_experts}")
    if hist.num_experts == 0:
        hist.num_experts = num_experts
        hist.top1 = np.zeros(num_experts, dtype=np.uint64)
        hist.topk = np.zeros(num_experts, dtype=np.uint64)
        hist.per_layer = np.zeros((num_layers, num_experts), dtype=np.uint64)
    restores = []
    for layer_idx, module, gate_parent, down_parent in candidates:
        original = module.forward

        def captured_forward(
            self, hidden_states, top_k_index, top_k_weights, *, _layer=layer_idx, _gate=gate_parent, _down=down_parent
        ):
            if hist.k_top == 0:
                hist.k_top = int(top_k_index.shape[-1])
            hist.record(_layer, top_k_index, top_k_weights)
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_tensor in expert_hit:
                expert_idx = int(expert_tensor[0])
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate_name = split_expert_name(_gate, expert_idx)
                if _matches_tensor_filter(gate_name, tensor_filter):
                    gate_acc = accs.setdefault(
                        gate_name,
                        HessianAccumulator(gate_name, current_state.shape[-1], accum_device, hessian=False),
                    )
                    gate_acc.update(current_state.detach())
                gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                down_name = split_expert_name(_down, expert_idx)
                if _matches_tensor_filter(down_name, tensor_filter):
                    down_acc = accs.setdefault(
                        down_name,
                        HessianAccumulator(down_name, current_hidden_states.shape[-1], accum_device, hessian=False),
                    )
                    down_acc.update(current_hidden_states.detach())
                current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
            return final_hidden_states

        module.forward = types.MethodType(captured_forward, module)
        restores.append(lambda module=module, original=original: setattr(module, "forward", original))
    return hist, restores


def build_hook(acc: HessianAccumulator):
    """Returns a forward-pre hook that captures the input activations.

    Hooks are registered on `nn.Linear`; the input is `x` (the matmul input,
    shape [..., K]). We flatten leading dims and hand the GPU tensor straight
    to the accumulator — the gemm + transfer policy lives in `update`.
    """

    def hook(module, inputs):  # noqa: ARG001 — module is unused but required by API
        x = inputs[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))
        acc.update(x.detach())

    return hook


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _write_compact_hessian(f, acc: HessianAccumulator) -> None:
    """Stream exact-F32 diagonal + row-major BF16 lower strict triangle."""
    diagonal = np.asarray(acc.finalize_diag(), dtype="<f4")
    f.write(diagonal.tobytes())
    hessian = np.asarray(acc.finalize(), dtype=np.float32)
    for row_idx in range(1, acc.K):
        lower = np.ascontiguousarray(hessian[row_idx, :row_idx])
        bf16_bits = torch.from_numpy(lower).to(torch.bfloat16).view(torch.uint16).numpy()
        f.write(bf16_bits.astype("<u2", copy=False).tobytes())


class KldRefAccumulator:
    """Collect native-compatible top-K teacher logits and full-vocab logZ."""

    def __init__(self, top_k: int = 64) -> None:
        if top_k < 1:
            raise ValueError("KLDREF top_k must be at least 1")
        self.top_k = top_k
        self._indices: list[np.ndarray] = []
        self._logits: list[np.ndarray] = []
        self._logz: list[np.ndarray] = []

    @property
    def n_positions(self) -> int:
        return sum(chunk.shape[0] for chunk in self._logz)

    def update(self, logits: torch.Tensor) -> None:
        """Append logits shaped `[positions, vocab]` without retaining vocab rows."""
        if logits.ndim != 2:
            raise ValueError(f"KLDREF logits must be 2-D, got {tuple(logits.shape)}")
        k = min(self.top_k, logits.shape[-1])
        values, indices = torch.topk(logits.float(), k=k, dim=-1, sorted=True)
        if k != self.top_k:
            pad = self.top_k - k
            values = torch.nn.functional.pad(values, (0, pad), value=float("-inf"))
            indices = torch.nn.functional.pad(indices, (0, pad), value=0)
        self._indices.append(indices.to(device="cpu", dtype=torch.float32).numpy())
        self._logits.append(values.to(device="cpu", dtype=torch.float32).numpy())
        self._logz.append(torch.logsumexp(logits.float(), dim=-1).to(device="cpu").numpy())

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._logz:
            empty_topk = np.empty((0, self.top_k), dtype=np.float32)
            return empty_topk, empty_topk.copy(), np.empty((0,), dtype=np.float32)
        return (
            np.concatenate(self._indices).astype("<f4", copy=False),
            np.concatenate(self._logits).astype("<f4", copy=False),
            np.concatenate(self._logz).astype("<f4", copy=False),
        )


class CalibrationSpool:
    """Disk-backed HFQM payload spool for layer-at-a-time collection.

    A 397B checkpoint cannot retain every dense Hessian in RAM while the
    teacher advances through 60 layers. Each layer is finalized into this
    spool, then its accumulators and weights can be released before the next
    layer is read from safetensors.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: list[dict] = []
        self._per_tensor_tokens: dict[str, int] = {}
        self._n_hessian = 0
        self._n_imatrix = 0
        self._kldref: dict | None = None
        self._counter = 0

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        shutil.rmtree(self.path, ignore_errors=True)

    def _payload_path(self, suffix: str) -> Path:
        path = self.path / f"{self._counter:08d}.{suffix}"
        self._counter += 1
        return path

    def _add_entry(self, name: str, quant_type: int, shape: list[int], payload: Path) -> None:
        self._entries.append(
            {
                "name": name,
                "quant_type": quant_type,
                "shape": shape,
                "size": payload.stat().st_size,
                "payload": payload,
            }
        )

    def add_accumulators(self, accs: dict[str, HessianAccumulator]) -> None:
        payloads: dict[int, tuple[Path | None, Path]] = {}
        for name, acc in sorted(accs.items()):
            if not acc.n_tokens:
                print(f"  WARN: {name} accumulated 0 tokens — skipping", file=sys.stderr)
                continue
            identity = id(acc)
            if identity not in payloads:
                h_path = self._payload_path("hessian") if acc.has_hessian else None
                if h_path is not None:
                    with h_path.open("wb") as f:
                        _write_compact_hessian(f, acc)
                i_path = self._payload_path("imatrix")
                i_path.write_bytes(np.asarray(acc.finalize_diag(), dtype="<f4").tobytes())
                payloads[identity] = (h_path, i_path)
            h_path, i_path = payloads[identity]
            if h_path is not None:
                self._add_entry(f"{name}.hessian", 130, [acc.K, acc.K], h_path)
                self._n_hessian += 1
            self._add_entry(f"{name}.imatrix", 2, [acc.K], i_path)
            self._n_imatrix += 1
            self._per_tensor_tokens[name] = acc.n_tokens

    def add_kldref(self, kldref: KldRefAccumulator) -> None:
        if not kldref.n_positions:
            return
        indices, logits, logz = kldref.arrays()
        for name, array in (
            ("lm_head.kldref_idx", indices),
            ("lm_head.kldref_logit", logits),
            ("lm_head.kldref_logz", logz),
        ):
            payload = self._payload_path("kldref")
            payload.write_bytes(np.asarray(array, dtype="<f4").tobytes())
            self._add_entry(name, 2, list(array.shape), payload)
        self._kldref = {"n_positions": kldref.n_positions, "top_k": kldref.top_k}

    def write_hfq(self, out_path: Path, metadata_extra: dict | None = None) -> None:
        if not out_path.name.endswith(".calib.hfq"):
            raise ValueError(f"calibration artifacts must end in .calib.hfq: {out_path}")
        entries = sorted(self._entries, key=lambda entry: entry["name"])
        artifacts = ["hessian", "imatrix"]
        metadata = {
            "artifact_kind": "calibration",
            "n_hessian": self._n_hessian,
            "n_imatrix": self._n_imatrix,
            "per_tensor_tokens": dict(sorted(self._per_tensor_tokens.items())),
            "artifacts": artifacts,
        }
        if self._kldref is not None:
            metadata["kldref"] = self._kldref
            artifacts.append("kldref")
        if metadata_extra:
            metadata.update(metadata_extra)
        if "moe_router_histogram" in metadata and "moe_router_histogram" not in artifacts:
            artifacts.append("moe_router_histogram")
        meta = json.dumps(metadata, separators=(",", ":")).encode()

        index_len = 4
        for entry in entries:
            index_len += 2 + len(entry["name"].encode()) + 2 + 4 * len(entry["shape"]) + 4 + 8 + 8
        metadata_offset = 32
        data_offset = _align(metadata_offset + len(meta) + index_len, 4096)
        offsets: list[int] = []
        cursor = data_offset
        for entry in entries:
            cursor = _align(cursor, 32)
            offsets.append(cursor)
            cursor += entry["size"]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            f.write(struct.pack("<4sIIIQQ", b"HFQM", 2, 0, len(entries), metadata_offset, data_offset))
            f.write(meta)
            f.write(struct.pack("<I", len(entries)))
            for entry, offset in zip(entries, offsets):
                encoded = entry["name"].encode()
                if len(encoded) > 0xFFFF:
                    raise ValueError(f"HFQM entry name too long: {entry['name']}")
                shape = entry["shape"]
                f.write(struct.pack("<H", len(encoded)))
                f.write(encoded)
                f.write(struct.pack("<BB", entry["quant_type"], len(shape)))
                f.write(struct.pack(f"<{len(shape)}I", *shape))
                f.write(struct.pack("<IQQ", 0, entry["size"], offset // 32))
            f.write(b"\0" * (data_offset - f.tell()))
            for entry, offset in zip(entries, offsets):
                if f.tell() < offset:
                    f.write(b"\0" * (offset - f.tell()))
                with entry["payload"].open("rb") as payload:
                    shutil.copyfileobj(payload, f, length=16 * 1024 * 1024)


def write_calibration_hfq(
    out_path: Path,
    accs: dict[str, HessianAccumulator],
    metadata_extra: dict | None = None,
) -> None:
    """Write the canonical streaming HFQM v2 calibration package."""
    with tempfile.TemporaryDirectory(prefix="hipfire-calib-") as temp_dir:
        spool_path = Path(temp_dir) / "payloads"
        with CalibrationSpool(spool_path) as spool:
            spool.add_accumulators(accs)
            spool.write_hfq(out_path, metadata_extra)


def _weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint found under {model_path}")
    from safetensors import safe_open  # type: ignore[import-not-found]

    result: dict[str, str] = {}
    for path in files:
        with safe_open(path, framework="pt") as reader:
            result.update({name: path.name for name in reader.keys()})
    return result


def resolve_model_path(model: str) -> Path:
    """Accept either an HF snapshot or its `models--ORG--NAME` cache root."""
    path = Path(model).expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        main_ref = path / "refs" / "main"
        if main_ref.is_file():
            candidate = snapshots / main_ref.read_text().strip()
            if (candidate / "config.json").is_file():
                return candidate
        candidates = sorted(
            (p for p in snapshots.iterdir() if (p / "config.json").is_file()), key=lambda p: p.stat().st_mtime
        )
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"{model!r} is not a local Hugging Face safetensors snapshot")


def _device_for_parameter(name: str, device_map: dict[str, object]) -> object:
    candidate = name
    while True:
        if candidate in device_map:
            return device_map[candidate]
        if not candidate:
            break
        candidate = candidate.rsplit(".", 1)[0] if "." in candidate else ""
    raise KeyError(f"device map does not cover {name}")


def build_reference_offload_index(
    model_path: Path,
    model_to_stored: dict[str, str],
    device_map: dict[str, object],
) -> dict[str, dict[str, str]]:
    """Map disk-owned parameters to their original safetensor shard."""
    weights = _weight_map(model_path)
    result = {}
    for model_name, stored_name in model_to_stored.items():
        if _device_for_parameter(model_name, device_map) != "disk":
            continue
        result[model_name] = {
            "safetensors_file": str((model_path / weights[stored_name]).resolve()),
            "weight_name": stored_name,
        }
    return result


def _parameter_name_map(model: torch.nn.Module, stored_weights: dict[str, str]) -> dict[str, str]:
    """Map in-memory text-model parameters to checkpoint tensor names."""
    stored = set(stored_weights)
    result = {}
    for model_name, _parameter in model.named_parameters():
        candidates = [model_name]
        if model_name.startswith("model."):
            candidates.append(model_name.replace("model.", "model.language_model.", 1))
        match = next((name for name in candidates if name in stored), None)
        if match is None:
            trail = model_name.split(".", 1)[-1]
            suffix_matches = [name for name in stored if name.endswith(trail)]
            if len(suffix_matches) == 1:
                match = suffix_matches[0]
        if match is None:
            raise KeyError(f"no safetensors tensor matches model parameter {model_name!r}")
        result[model_name] = match
    return result


def layer_parameter_plan(names) -> dict:
    """Partition text-model parameters for one-read layer-major execution."""
    plan = {"embedding": [], "layers": {}, "final": []}
    for name in names:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match is not None:
            plan["layers"].setdefault(int(match.group(1)), []).append(name)
        elif "embed_tokens" in name:
            plan["embedding"].append(name)
        else:
            plan["final"].append(name)
    plan["embedding"].sort()
    plan["layers"] = {index: sorted(plan["layers"][index]) for index in sorted(plan["layers"])}
    plan["final"].sort()
    return plan


class SingleReadSafetensorLoader:
    """Materialize each checkpoint tensor at most once during a teacher pass."""

    def __init__(self, model_path: Path, weight_map: dict[str, str]) -> None:
        self.model_path = model_path
        self.weight_map = weight_map
        self.loaded: set[str] = set()

    def load_parameters(
        self,
        model: torch.nn.Module,
        model_names: list[str],
        model_to_stored: dict[str, str],
        device: str,
        dtype: torch.dtype,
    ) -> None:
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open  # type: ignore[import-not-found]

        by_shard: dict[str, list[tuple[str, str]]] = {}
        for model_name in model_names:
            stored_name = model_to_stored[model_name]
            if stored_name in self.loaded:
                raise RuntimeError(f"teacher pass attempted to reread safetensors tensor {stored_name}")
            by_shard.setdefault(self.weight_map[stored_name], []).append((model_name, stored_name))
        for shard_name, tensors in sorted(by_shard.items()):
            with safe_open(self.model_path / shard_name, framework="pt", device=device) as reader:
                for model_name, stored_name in tensors:
                    value = reader.get_tensor(stored_name)
                    set_module_tensor_to_device(model, model_name, device, value=value, dtype=dtype)
                    self.loaded.add(stored_name)

    @staticmethod
    def unload_parameters(model: torch.nn.Module, model_names: list[str]) -> None:
        from accelerate.utils import set_module_tensor_to_device

        for model_name in model_names:
            set_module_tensor_to_device(model, model_name, "meta")


def _text_causal_config(root_config):
    """Select the text config from multimodal checkpoints when necessary."""
    if not hasattr(root_config, "vocab_size") and getattr(root_config, "text_config", None) is not None:
        return root_config.text_config
    return root_config


def load_safetensors_model(
    model_path: Path,
    dtype: torch.dtype,
    device: str,
    reference_offload: bool,
    gpu_memory: str,
    cpu_memory: str,
):
    """Load a text CausalLM, optionally paging from its original shards."""
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open  # type: ignore[import-not-found]
    from transformers import AutoConfig

    root_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    config = _text_causal_config(root_config)
    if not reference_offload:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    model.tie_weights()

    if device == "cuda":
        no_split = list(getattr(model, "_no_split_modules", []) or [])
        device_map = infer_auto_device_map(
            model,
            max_memory={0: gpu_memory, "cpu": cpu_memory},
            no_split_module_classes=no_split,
            dtype=dtype,
            offload_buffers=False,
        )
    else:
        device_map = {"": device}

    stored_weights = _weight_map(model_path)
    model_to_stored = _parameter_name_map(model, stored_weights)
    resident_by_shard: dict[str, list[tuple[str, str, object]]] = {}
    for model_name, parameter in model.named_parameters():
        target = _device_for_parameter(model_name, device_map)
        if target == "disk":
            continue
        stored_name = model_to_stored[model_name]
        resident_by_shard.setdefault(stored_weights[stored_name], []).append((model_name, stored_name, target))

    for shard_name, tensors in sorted(resident_by_shard.items()):
        shard_path = model_path / shard_name
        for model_name, stored_name, target in tensors:
            load_device = f"cuda:{target}" if isinstance(target, int) else str(target)
            with safe_open(shard_path, framework="pt", device=load_device) as reader:
                value = reader.get_tensor(stored_name)
            set_module_tensor_to_device(model, model_name, target, value=value, dtype=dtype)

    # Replacing a resident tied parameter (typically embed_tokens) breaks the
    # object identity established on the meta model. Restore ties before
    # Accelerate initializes hooks, or its resident lm_head still looks meta.
    model.tie_weights()

    offload_index = build_reference_offload_index(model_path, model_to_stored, device_map)
    if offload_index:
        main_device = next((value for value in device_map.values() if value not in ("cpu", "disk")), "cpu")
        model = dispatch_model(
            model,
            device_map=device_map,
            main_device=main_device,
            offload_index=offload_index,
            offload_buffers=False,
            preload_module_classes=["Qwen3_5MoeExperts"],
            force_hooks=True,
        )
    model.hf_device_map = dict(device_map)
    return model


def load_calibration_text(corpus: str, n_sequences: int, ctx_len: int, tokenizer) -> list[torch.Tensor]:
    """Returns a list of n_sequences token tensors, each of length ctx_len.

    `corpus` is either a HF datasets identifier (e.g. "wikitext") or a
    path to a plain-text file. For HF datasets we use the "train" split
    by default (avoids data leakage against wikitext-2-test which is the
    eval slice).
    """
    needed_tokens = n_sequences * ctx_len

    def tokenize_to_budget(text: str) -> torch.Tensor:
        # The corpus is only a source pool. Passing the full pool through a HF
        # tokenizer without truncation makes Transformers compare its combined
        # length with model_max_length and warn about a sequence that is never
        # actually forwarded. Bound the tokenizer result to the exact amount
        # consumed below; the model still sees n_sequences independent ctx_len
        # sequences.
        return tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=needed_tokens,
        ).input_ids[0]

    if Path(corpus).is_file():
        # Read a generous prefix instead of materializing and tokenizing a
        # multi-million-token local corpus when only `needed_tokens` can be
        # consumed. Retry with another prefix block only for unusually
        # low-token-density text.
        char_budget = max(needed_tokens * 8 * 2, 1)
        prefix_parts: list[str] = []
        all_tokens = torch.empty(0, dtype=torch.long)
        with Path(corpus).open(encoding="utf-8") as corpus_file:
            while chunk := corpus_file.read(char_budget):
                prefix_parts.append(chunk)
                all_tokens = tokenize_to_budget("".join(prefix_parts))
                if all_tokens.shape[0] >= needed_tokens:
                    break
    else:
        # HF datasets path. Default: wikitext-2-raw-v1 train split.
        if "/" in corpus or corpus == "wikitext":
            ds_name = corpus if "/" in corpus else "wikitext"
            cfg = "wikitext-2-raw-v1" if ds_name == "wikitext" else None
            ds = load_dataset(ds_name, cfg, split="train", trust_remote_code=False)
        else:
            ds = load_dataset(corpus, split="train", trust_remote_code=False)
        # Concatenate text rows up to a char budget covering the tokens we need,
        # then tokenize. Avoids tokenizing the ENTIRE dataset (e.g. all of
        # wikitext-train, millions of tokens) to use only n_sequences×ctx_len.
        # ~8 chars/token slack (matches perplexity.rs) + 2× margin for safety.
        text_field = "text" if "text" in ds.column_names else ds.column_names[0]
        char_budget = n_sequences * ctx_len * 8 * 2
        parts: list[str] = []
        n_chars = 0
        for row in ds:
            t = row[text_field]
            if not t.strip():
                continue
            parts.append(t)
            n_chars += len(t) + 2  # +2 for the "\n\n" join
            if n_chars >= char_budget:
                break
        text = "\n\n".join(parts)
        all_tokens = tokenize_to_budget(text)

    n_total_tokens = all_tokens.shape[0]
    if n_total_tokens < needed_tokens:
        raise SystemExit(
            f"corpus has only {n_total_tokens} tokens, need {needed_tokens} ({n_sequences} seqs × {ctx_len} ctx)"
        )

    seqs = [all_tokens[i * ctx_len : (i + 1) * ctx_len] for i in range(n_sequences)]
    return seqs


def collect_layer_streamed(
    *,
    model_path: Path,
    out_path: Path,
    seqs: list[torch.Tensor],
    dtype: torch.dtype,
    device: str,
    accum_device: str,
    batch_size: int,
    tensor_filter: str | None,
    kldref_enabled: bool,
    kldref_topk: int,
    logit_batch_tokens: int,
    spool_parent: Path | None,
) -> None:
    """Run a Qwen3.5 teacher in layer-major order with one read per tensor.

    Boundary activations live in host memory. A layer is loaded once from the
    original safetensor shards, consumes every calibration microbatch, then is
    unloaded before the next layer. This turns the old
    `microbatches × checkpoint-size` paging behavior into one checkpoint sweep.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask

    if not seqs:
        raise ValueError("layer-stream collection requires at least one sequence")
    if logit_batch_tokens < 1:
        raise ValueError("logit_batch_tokens must be at least 1")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")

    root_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    config = _text_causal_config(root_config)
    if getattr(config, "model_type", "") not in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        raise RuntimeError(
            "--layer-stream currently supports Qwen3.5 MoE text checkpoints; "
            "the execution mechanism is deliberately isolated pending generic ParoQuant work"
        )
    if bool(getattr(config, "tie_word_embeddings", False)):
        raise RuntimeError("--layer-stream does not yet support tied embedding/lm-head checkpoints")

    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    model.eval()
    text_model = model.model
    if not hasattr(text_model, "layers") or not hasattr(text_model, "rotary_emb"):
        raise RuntimeError("Qwen3.5 layer-stream model surface changed: missing layers or rotary_emb")

    weights = _weight_map(model_path)
    model_to_stored = _parameter_name_map(model, weights)
    plan = layer_parameter_plan(model_to_stored)
    expected_layers = int(config.num_hidden_layers)
    if list(plan["layers"]) != list(range(expected_layers)):
        raise RuntimeError(
            f"checkpoint/model layer plan is incomplete: found {list(plan['layers'])}, expected 0..{expected_layers - 1}"
        )
    loader = SingleReadSafetensorLoader(model_path, weights)

    n_sequences = len(seqs)
    ctx_len = int(seqs[0].numel())
    if any(int(seq.numel()) != ctx_len for seq in seqs):
        raise ValueError("all calibration sequences must have the same context length")
    hidden_size = int(config.hidden_size)
    pin_memory = device.startswith("cuda") and torch.cuda.is_available()
    hidden = torch.empty(
        (n_sequences, ctx_len, hidden_size), dtype=dtype, device="cpu", pin_memory=pin_memory
    )
    next_hidden = torch.empty_like(hidden)

    print("      loading embedding tensor once and materializing host boundary activations")
    loader.load_parameters(model, plan["embedding"], model_to_stored, device, dtype)
    with torch.no_grad():
        for start in range(0, n_sequences, batch_size):
            stop = min(start + batch_size, n_sequences)
            token_batch = torch.stack(seqs[start:stop]).to(device, non_blocking=pin_memory)
            embedded = model.get_input_embeddings()(token_batch)
            hidden[start:stop].copy_(embedded.to(device="cpu", dtype=dtype), non_blocking=False)
            del token_batch, embedded
    loader.unload_parameters(model, plan["embedding"])

    # Rotary inv_freq buffers are tiny and are not checkpoint tensors. Keep
    # them resident while layer weights stream independently.
    text_model.rotary_emb.to(device=device)
    stored_names = {name.removesuffix(".weight") for name in weights}
    router_hist = MoeRouterHistogram(0, 0, expected_layers)
    kldref = KldRefAccumulator(kldref_topk) if kldref_enabled else None

    spool_root_parent = spool_parent or out_path.parent
    spool_root_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hipfire-calib-spool-", dir=spool_root_parent) as temp_dir:
        with CalibrationSpool(Path(temp_dir) / "payloads") as spool:
            for layer_idx, parameter_names in plan["layers"].items():
                layer_t0 = time.time()
                print(f"      layer {layer_idx + 1}/{expected_layers}: loading {len(parameter_names)} tensors")
                loader.load_parameters(model, parameter_names, model_to_stored, device, dtype)
                layer = text_model.layers[layer_idx]

                accs: dict[str, HessianAccumulator] = {}
                input_groups: dict[str, HessianAccumulator] = {}
                handles = []
                layer_prefix = f"model.layers.{layer_idx}"
                for local_name, module in layer.named_modules():
                    model_name = f"{layer_prefix}.{local_name}" if local_name else layer_prefix
                    if not isinstance(module, torch.nn.Linear) or not is_gptq_target(model_name):
                        continue
                    stored = _translate_to_stored_name(model_name, stored_names)
                    if stored is None or not _matches_tensor_filter(stored, tensor_filter):
                        continue
                    group = shared_input_group(stored)
                    if group in input_groups:
                        accs[stored] = input_groups[group]
                    else:
                        acc = HessianAccumulator(stored, module.in_features, accum_device=accum_device)
                        input_groups[group] = acc
                        accs[stored] = acc
                        handles.append(module.register_forward_pre_hook(build_hook(acc)))

                router_hist, moe_restores = install_fused_moe_capture(
                    model,
                    stored_names,
                    accs,
                    accum_device,
                    expected_layers,
                    tensor_filter,
                    allowed_layers={layer_idx},
                    router_hist=router_hist,
                )

                with torch.no_grad():
                    for start in range(0, n_sequences, batch_size):
                        stop = min(start + batch_size, n_sequences)
                        states = hidden[start:stop].to(device, non_blocking=pin_memory)
                        cache_position = torch.arange(ctx_len, device=device)
                        position_ids = cache_position.view(1, 1, -1).expand(3, stop - start, -1)
                        causal_mask = create_causal_mask(
                            config=config,
                            inputs_embeds=states,
                            attention_mask=None,
                            cache_position=cache_position,
                            past_key_values=None,
                            position_ids=position_ids[0],
                        )
                        linear_mask = text_model._update_linear_attn_mask(None, cache_position)
                        layer_mask = linear_mask if layer.layer_type == "linear_attention" else causal_mask
                        position_embeddings = text_model.rotary_emb(states, position_ids)
                        result = layer(
                            states,
                            position_embeddings=position_embeddings,
                            attention_mask=layer_mask,
                            position_ids=position_ids,
                            past_key_values=None,
                            use_cache=False,
                            cache_position=cache_position,
                        )
                        if isinstance(result, (tuple, list)):
                            result = result[0]
                        next_hidden[start:stop].copy_(result.to(device="cpu", dtype=dtype), non_blocking=False)
                        del states, result, position_embeddings, causal_mask, linear_mask, layer_mask

                for handle in handles:
                    handle.remove()
                for restore in moe_restores:
                    restore()
                spool.add_accumulators(accs)
                loader.unload_parameters(model, parameter_names)
                hidden, next_hidden = next_hidden, hidden
                del accs, input_groups, handles, moe_restores
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"      layer {layer_idx + 1}/{expected_layers}: spooled in {time.time() - layer_t0:.1f}s")

            print("      loading final norm/lm-head tensors once")
            loader.load_parameters(model, plan["final"], model_to_stored, device, dtype)
            with torch.no_grad():
                for start in range(0, n_sequences, batch_size):
                    stop = min(start + batch_size, n_sequences)
                    states = hidden[start:stop].to(device, non_blocking=pin_memory)
                    states = text_model.norm(states).reshape(-1, hidden_size)
                    if kldref is not None:
                        for token_start in range(0, states.shape[0], logit_batch_tokens):
                            logits = model.lm_head(states[token_start : token_start + logit_batch_tokens])
                            kldref.update(logits)
                            del logits
                    del states
            loader.unload_parameters(model, plan["final"])
            if kldref is not None:
                spool.add_kldref(kldref)

            metadata = {
                "source_format": "safetensors",
                "source_model": str(model_path),
                "collector": "scripts/collect_hessian.py",
                "collector_mode": "layer_stream",
                "calibration_tokens": n_sequences * ctx_len,
                "safetensors_teacher_reads": len(loader.loaded),
            }
            if router_hist.routed_tokens:
                metadata["moe_router_histogram"] = router_hist.to_metadata()
            spool.write_hfq(out_path, metadata)


def main():
    ap = argparse.ArgumentParser(description="Collect a canonical .calib.hfq directly from BF16 safetensors.")
    ap.add_argument("--model", required=True, help="Local HF snapshot or models--ORG--NAME cache root.")
    ap.add_argument("--output", required=True, type=Path, help="Canonical output path ending in .calib.hfq.")
    ap.add_argument(
        "--n-sequences", type=int, default=128, help="Calibration sequences (default 128, matches GPTQ paper)."
    )
    ap.add_argument("--ctx-len", type=int, default=2048, help="Tokens per calibration sequence (default 2048).")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Sequences per activation microbatch (default 1; increase when activation memory permits).",
    )
    ap.add_argument(
        "--corpus",
        default="wikitext",
        help="HF dataset ID or path to plain-text file (default 'wikitext' = wikitext-2-raw-v1 train).",
    )
    ap.add_argument("--device", default="cuda", help="Device for the forward pass (default 'cuda').")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default bfloat16).",
    )
    ap.add_argument(
        "--accum",
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Accumulator placement. auto uses CPU with shard paging and GPU otherwise.",
    )
    ap.add_argument(
        "--reference-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Page disk-owned layers from original safetensor shards (default: enabled).",
    )
    ap.add_argument("--gpu-memory", default="55GiB", help="Accelerate GPU weight budget (default: 55GiB).")
    ap.add_argument("--cpu-memory", default="2GiB", help="Accelerate resident CPU weight budget (default: 2GiB).")
    ap.add_argument(
        "--layer-stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load each Qwen3.5 layer once, process all microbatches, then release it.",
    )
    ap.add_argument(
        "--kldref",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bundle native-compatible top-K teacher logits and logZ in the calibration artifact.",
    )
    ap.add_argument("--kldref-topk", type=int, default=64, help="KLDREF top-K width (default: 64).")
    ap.add_argument(
        "--logit-batch-tokens",
        type=int,
        default=64,
        help="Token rows per lm-head call while building KLDREF (default: 64).",
    )
    ap.add_argument(
        "--spool-dir",
        type=Path,
        default=None,
        help="Parent for temporary layer payloads (default: output directory).",
    )
    ap.add_argument(
        "--tensor-filter",
        default=None,
        help="Optional substring over canonical safetensors names "
        "(with or without trailing .weight) to collect a small "
        "canonical smoke artifact.",
    )
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be at least 1")
    if args.kldref_topk < 1:
        ap.error("--kldref-topk must be at least 1")
    if args.logit_batch_tokens < 1:
        ap.error("--logit-batch-tokens must be at least 1")
    model_path = resolve_model_path(args.model)
    if not args.output.name.endswith(".calib.hfq"):
        ap.error("--output must use the canonical .calib.hfq suffix")
    accum_device = "cpu" if args.accum == "auto" and (args.reference_offload or args.layer_stream) else args.accum
    if accum_device == "auto":
        accum_device = "gpu"

    print("=== Hipfire safetensors calibration collector ===")
    print(f"  model:        {model_path}")
    print(f"  output:       {args.output}")
    print(f"  corpus:       {args.corpus}")
    print(
        f"  sequences:    {args.n_sequences} × {args.ctx_len} ctx "
        f"= {args.n_sequences * args.ctx_len} calibration tokens"
    )
    print(f"  device:       {args.device}")
    print(f"  dtype:        {args.dtype}")
    print(f"  accum:        {accum_device}")
    print(f"  layer stream: {args.layer_stream}")
    if args.layer_stream:
        print(f"  kldref:       {args.kldref} (top-k {args.kldref_topk})")
    else:
        print(f"  shard paging: {args.reference_offload} (GPU budget {args.gpu_memory}, CPU budget {args.cpu_memory})")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"\n[1/4] Loading tokenizer{' + model' if not args.layer_stream else ''}...")
    t0 = time.time()
    _mask_broken_torchvision_for_text_only_calibration()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if args.layer_stream:
        seqs = load_calibration_text(args.corpus, args.n_sequences, args.ctx_len, tokenizer)
        print(f"      tokenizer + {len(seqs)} calibration sequences loaded in {time.time() - t0:.1f}s")
        print("\n[2/4] Running one-read layer-streamed teacher pass...")
        t0 = time.time()
        collect_layer_streamed(
            model_path=model_path,
            out_path=args.output,
            seqs=seqs,
            dtype=dtype,
            device=args.device,
            accum_device=accum_device,
            batch_size=args.batch_size,
            tensor_filter=args.tensor_filter,
            kldref_enabled=args.kldref,
            kldref_topk=args.kldref_topk,
            logit_batch_tokens=args.logit_batch_tokens,
            spool_parent=args.spool_dir,
        )
        print(f"\n[4/4] Wrote {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)")
        print(f"      layer-stream pass completed in {time.time() - t0:.1f}s")
        print("\n=== Done ===")
        return
    model = load_safetensors_model(
        model_path,
        dtype,
        args.device,
        args.reference_offload,
        args.gpu_memory,
        args.cpu_memory,
    )
    model.eval()
    print(f"      loaded in {time.time() - t0:.1f}s")
    disk_modules = sum(value == "disk" for value in getattr(model, "hf_device_map", {}).values())
    if disk_modules:
        sweeps = (args.n_sequences + args.batch_size - 1) // args.batch_size
        print(f"      reference offload: {disk_modules} disk modules, {sweeps} full checkpoint sweep(s)")
    if args.device == "cuda" and torch.cuda.is_available():
        print(
            f"      VRAM in use: "
            f"{torch.cuda.memory_allocated() / 1e9:.2f}/"
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    print(f"\n[2/4] Registering Hessian hooks on GPTQ-target Linear modules...")
    # HF's AutoModelForCausalLM flattens multimodal submodules (e.g. Qwen3.5's
    # `language_model.` is dropped from in-memory module names when loaded as
    # a CausalLM). Hipfire's .hfq, however, mirrors the safetensors naming
    # which keeps the full prefix. We translate in-memory names → stored
    # names via safetensors-index matching so the Rust quantizer can look up
    # Hessians by the same key as the .hfq tensors.
    stored_names = _safetensors_keys(str(model_path))
    accs: dict[str, HessianAccumulator] = {}
    input_groups: dict[str, HessianAccumulator] = {}
    handles = []
    name_remap_count = 0
    for mod_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and is_gptq_target(mod_name):
            K = module.in_features
            stored = _translate_to_stored_name(mod_name, stored_names)
            if stored is None:
                print(f"  WARN: no safetensors key matches {mod_name!r} — skipping", file=sys.stderr)
                continue
            if not _matches_tensor_filter(stored, args.tensor_filter):
                continue
            if stored != mod_name:
                name_remap_count += 1
            group = shared_input_group(stored)
            if group in input_groups:
                accs[stored] = input_groups[group]
            else:
                acc = HessianAccumulator(stored, K, accum_device=accum_device)
                input_groups[group] = acc
                accs[stored] = acc
                handles.append(module.register_forward_pre_hook(build_hook(acc)))

    num_layers = int(getattr(model.config, "num_hidden_layers", len(getattr(model.model, "layers", []))))
    router_hist, moe_restores = install_fused_moe_capture(
        model,
        stored_names,
        accs,
        accum_device,
        num_layers,
        args.tensor_filter,
    )
    print(
        f"      {name_remap_count} of {len(accs)} names remapped from in-memory "
        f"→ stored (multimodal-flatten translation)"
    )
    print(
        f"      registered {len(handles)} dense hooks for {len(accs)} tensors "
        f"({len(set(K_ for K_ in [acc.K for acc in accs.values()]))} distinct K dims)"
    )
    dense_acc_bytes = sum(acc.K * acc.K * 4 + acc.K * 4 for acc in input_groups.values())
    print(f"      estimated dense accumulator memory: {dense_acc_bytes / 2**30:.2f} GiB")
    if not accs and router_hist.num_experts == 0:
        detail = f" matching --tensor-filter {args.tensor_filter!r}" if args.tensor_filter else ""
        raise SystemExit(f"no GPTQ-target Linear modules found{detail}")
    if accs:
        print(f"      K range: {min(acc.K for acc in accs.values())}..{max(acc.K for acc in accs.values())}")
    if router_hist.num_experts:
        print(f"      fused MoE capture: {router_hist.num_experts} experts across {num_layers} layers")

    print(f"\n[3/4] Loading calibration corpus + running forward pass...")
    t0 = time.time()
    seqs = load_calibration_text(args.corpus, args.n_sequences, args.ctx_len, tokenizer)
    print(f"      loaded {len(seqs)} calibration sequences in {time.time() - t0:.1f}s")
    print(f"      starting forward pass (no_grad, eval mode)...")

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(seqs), args.batch_size):
            input_device = model.get_input_embeddings().weight.device
            seq_batch = torch.stack(seqs[start : start + args.batch_size]).to(input_device)
            model(seq_batch)
            completed = min(start + args.batch_size, len(seqs))
            if completed % 8 == 0 or completed == len(seqs):
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (len(seqs) - completed) / rate
                print(f"      seq {completed}/{len(seqs)} ({elapsed:.1f}s elapsed, {rate:.2f} seq/s, ETA {eta:.0f}s)")

    print(f"      forward pass complete: {time.time() - t0:.1f}s")
    if accs:
        print(f"      total tokens accumulated per tensor (sample): {next(iter(accs.values())).n_tokens}")
    if not any(acc.n_tokens for acc in accs.values()):
        raise SystemExit("no matching tensor accumulated calibration tokens")

    # Detach hooks before writing — they'll keep accumulating otherwise.
    for h in handles:
        h.remove()
    for restore in moe_restores:
        restore()

    print(f"\n[4/4] Writing calibration package to {args.output}...")
    t0 = time.time()
    metadata = {
        "source_format": "safetensors",
        "source_model": str(model_path),
        "collector": "scripts/collect_hessian.py",
        "calibration_tokens": args.n_sequences * args.ctx_len,
    }
    if router_hist.routed_tokens:
        metadata["moe_router_histogram"] = router_hist.to_metadata()
    write_calibration_hfq(args.output, accs, metadata)
    print(f"      wrote {args.output.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.1f}s")
    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
