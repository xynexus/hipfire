"""
D3: Per-channel activation absmax via forward-pass hooks on bf16 reference.

For each Linear layer in the model (q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj, plus router/shared_expert paths in MoE):
  1. Register a forward_pre_hook that captures the layer's INPUT.
  2. Register a forward_hook that captures the layer's OUTPUT.
  3. For each calibration prompt, run model(input_ids) and accumulate
     per-channel absmax of (input, output) into per-(layer, hook_name)
     buffers.
  4. After all prompts, dump per (layer_idx, hook_point, channel) -> absmax
     to per_channel.jsonl.

The Super-Expert signal (arXiv 2507.23279) is the per-token down_proj
output magnitude per active expert. v1 of D3 captures the bulk
down_proj output absmax (not per-expert per-token); the per-expert
breakdown requires custom MoE-block instrumentation that isn't in
stock transformers and is deferred to v2.

Memory:
  bf16 reference resident across all 4 GPUs via device_map="auto":
    9B  -> 18 GB  (fits on 1 GPU; auto = "cuda:0")
    27B -> 54 GB  (sharded across 2 GPUs)
    A3B -> 70 GB  (sharded across 4 GPUs)

Requires torch + transformers ROCm wheel (task #35; unblocked at
2026-05-06 17:00 UTC commit cd61ef2).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Lazy-import torch / transformers — the weight-side diagnostics import
# this module's parent package and we don't want to hard-fail on
# environments that lack torch.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class D3Accumulator:
    """Per-(layer_idx, hook_point) running max of per-channel absmax.

    Channels are the LAST dim of the activation tensor. Stored as a 1D
    torch tensor on the device the activation lived on; finalize() copies
    to CPU and returns a Python list.
    """
    layer_idx: int
    hook_point: str  # e.g. "layers.5.mlp.down_proj.input" or "...output"
    n_channels: int
    accum: Any = None  # torch.Tensor[n_channels] on hooked layer's device
    n_calls: int = 0

    def update(self, abs_per_channel: Any) -> None:
        if self.accum is None:
            self.accum = abs_per_channel.clone()
        else:
            torch.maximum(self.accum, abs_per_channel, out=self.accum)
        self.n_calls += 1

    def finalize_to_cpu(self) -> list[float]:
        if self.accum is None:
            return []
        return self.accum.detach().to(dtype=torch.float32, device="cpu").tolist()


def _make_input_hook(accumulator: D3Accumulator):
    """forward_pre_hook capture: input[0] is the (batch, seq, channels) activation
    feeding into the linear layer. Reduce to per-channel max(|x|).
    """
    def hook(module, inputs):
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return
        # Per-channel absmax over (batch, seq) dims; channel is last dim.
        with torch.no_grad():
            flat = x.reshape(-1, x.shape[-1]).abs()
            per_ch = flat.max(dim=0).values
        accumulator.update(per_ch)
    return hook


def _make_output_hook(accumulator: D3Accumulator):
    """forward_hook capture: output is (batch, seq, out_channels)."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            return
        with torch.no_grad():
            flat = output.reshape(-1, output.shape[-1]).abs()
            per_ch = flat.max(dim=0).values
        accumulator.update(per_ch)
    return hook


def _classify_hook_point(name: str) -> tuple[int, str] | None:
    """Map a transformers module name like
    'model.language_model.layers.5.mlp.experts.42.down_proj' to
    (5, 'mlp.experts.down_proj').

    Returns None if the module isn't a Linear layer we care about.
    """
    import re
    m = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)$", name)
    if not m:
        return None
    layer_idx = int(m.group(1))
    rest = m.group(2)
    # Collapse per-expert index — we record one entry per
    # (layer, projection-class), summing absmax across experts.
    rest = re.sub(r"\.experts\.\d+\.", ".experts.", rest)
    return layer_idx, rest


def attach_hooks(model, projections_of_interest: set[str]) -> dict[str, dict]:
    """Walk the model, attach pre+post hooks on every Linear matching a
    classified hook point, return a dict keyed by
    (layer_idx, hook_point, side='input'|'output') -> D3Accumulator.

    `projections_of_interest` is a set like
    {"self_attn.q_proj", "mlp.down_proj", "mlp.experts.down_proj", ...}.
    Only matching modules get hooked.
    """
    if not HAS_TORCH:
        raise RuntimeError("D3 requires torch; install via uv pip + rocm7.2 index")

    accumulators: dict[tuple[int, str, str], D3Accumulator] = {}
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        cls = _classify_hook_point(name)
        if cls is None:
            continue
        layer_idx, hook_point = cls
        # Match against requested projections (suffix match).
        if not any(hook_point == p or hook_point.endswith("." + p) or hook_point == "self_attn." + p for p in projections_of_interest):
            continue

        # Channels: input = in_features, output = out_features.
        in_acc = D3Accumulator(layer_idx, hook_point + ".input", module.in_features)
        out_acc = D3Accumulator(layer_idx, hook_point + ".output", module.out_features)
        accumulators[(layer_idx, hook_point, "input")] = in_acc
        accumulators[(layer_idx, hook_point, "output")] = out_acc
        handles.append(module.register_forward_pre_hook(_make_input_hook(in_acc)))
        handles.append(module.register_forward_hook(_make_output_hook(out_acc)))

    return {
        "accumulators": accumulators,
        "handles": handles,
    }


def detach_hooks(state: dict) -> None:
    for h in state["handles"]:
        h.remove()


def run_calibration(
    model,
    tokenizer,
    prompts: list[str],
    accumulators: dict[tuple[int, str, str], D3Accumulator],
    max_seq_len: int = 512,
    device: str = "cuda:0",
) -> int:
    """Run each calibration prompt through the model. Hooks fire and update
    accumulators. Returns total tokens processed.
    """
    if not HAS_TORCH:
        raise RuntimeError("D3 requires torch")

    n_tokens = 0
    model.eval()
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        input_ids = tokens.input_ids.to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=False)
        n_tokens += input_ids.shape[1]
    return n_tokens


def emit_per_channel_jsonl(
    accumulators: dict[tuple[int, str, str], D3Accumulator],
    output_path: Path,
    model_name: str,
) -> int:
    """Dump one record per (layer, hook_point, side) with per-channel
    absmax. Returns number of records emitted.
    """
    n = 0
    with open(output_path, "w") as f:
        for (layer_idx, hook_point, side), acc in sorted(accumulators.items()):
            absmax = acc.finalize_to_cpu()
            if not absmax:
                continue
            rec = {
                "model": model_name,
                "layer_idx": layer_idx,
                "hook_point": hook_point,
                "side": side,
                "n_channels": len(absmax),
                "n_calls": acc.n_calls,
                "absmax_max": float(max(absmax)),
                "absmax_mean": float(sum(absmax) / len(absmax)),
                # Full per-channel array — large but enables synthesis to
                # find specific outlier channels.
                "absmax": absmax,
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Self-test (requires torch; gracefully degrades otherwise)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    if not HAS_TORCH:
        print("[d3 self-test] SKIPPED — torch not installed in this env. "
              "Install on hiptrx via ~/venv-torch (commit cd61ef2 log).")
        return 0

    # Tiny synthetic model: 2 Linear layers wrapped in a module that
    # mimics the transformers 'layers.N.proj' naming.
    import torch.nn as nn
    class _Layer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.down_proj = nn.Linear(dim, dim, bias=False)
        def forward(self, x):
            return self.down_proj(self.q_proj(x))

    class _Mock(nn.Module):
        def __init__(self, dim, n_layers):
            super().__init__()
            self.layers = nn.ModuleList([_Layer(dim) for _ in range(n_layers)])
        def forward(self, x):
            for L in self.layers:
                x = L(x)
            return x

    torch.manual_seed(0)
    dim = 32
    model = _Mock(dim, 4)
    state = attach_hooks(model, {"q_proj", "down_proj"})
    accs = state["accumulators"]
    assert len(accs) == 4 * 2 * 2, f"expected 16 accumulators (4 layers × 2 projections × 2 sides), got {len(accs)}"

    x = torch.randn(2, 8, dim) * 0.1
    x[0, 0, 7] = 5.0  # outlier channel
    with torch.no_grad():
        _ = model(x)

    detach_hooks(state)
    # Layer 0 q_proj input absmax should reflect the outlier in channel 7.
    in_acc = accs[(0, "q_proj", "input")]
    arr = in_acc.finalize_to_cpu()
    assert len(arr) == dim
    assert arr[7] >= 4.5, f"channel 7 absmax should reflect outlier; got {arr[7]}"
    print(f"[d3 self-test] outlier channel 7 absmax = {arr[7]:.3f} (good)")

    out = Path("/tmp/d3-self-test.jsonl")
    n = emit_per_channel_jsonl(accs, out, "_mock")
    assert n == len(accs), f"expected {len(accs)} records, got {n}"
    print(f"[d3 self-test] emitted {n} records to {out}")
    print("[d3 self-test] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
