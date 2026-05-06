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
    # (layer, projection-class), max-merging absmax across experts.
    # Matches BOTH 'mlp.experts.42.down_proj' (with prefix) and
    # 'experts.42.down_proj' (no prefix, e.g. self-test mocks).
    rest = re.sub(r"(^|\.)experts\.\d+\.", r"\1experts.", rest)
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
    handles: list = []
    n_modules_hooked = 0
    n_modules_shared = 0  # count of modules that joined an existing accumulator
    shape_warnings: list[str] = []

    # Wrap the whole hook-registration loop in a try/except so any failure
    # (shape-mismatch ValueError, register_forward_pre_hook OOM, etc.)
    # unwinds the handles already installed on the caller's model. Without
    # this, a partial failure would leave hooks attached pointing into
    # accumulators that the caller never received, mutating model state on
    # every subsequent forward pass.
    try:
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

            # Get-or-create accumulators per collapsed (layer, hook_point) key.
            # Critical for MoE: the classifier collapses
            # 'mlp.experts.42.down_proj' → 'mlp.experts.down_proj' for every
            # expert 0..N-1, so all N experts MUST share the same accumulator
            # instance. Each expert's hook registers against the same shared
            # D3Accumulator so the .update() max-merge spans all experts.
            in_key = (layer_idx, hook_point, "input")
            out_key = (layer_idx, hook_point, "output")

            if in_key not in accumulators:
                accumulators[in_key] = D3Accumulator(layer_idx, hook_point + ".input", module.in_features)
            else:
                # Sanity: all sibling experts should share the same in_features.
                existing = accumulators[in_key]
                if existing.n_channels != module.in_features:
                    shape_warnings.append(
                        f"in_features mismatch at {name}: "
                        f"existing accumulator n_channels={existing.n_channels} "
                        f"vs module.in_features={module.in_features}"
                    )
                n_modules_shared += 1

            if out_key not in accumulators:
                accumulators[out_key] = D3Accumulator(layer_idx, hook_point + ".output", module.out_features)
            else:
                existing = accumulators[out_key]
                if existing.n_channels != module.out_features:
                    shape_warnings.append(
                        f"out_features mismatch at {name}: "
                        f"existing accumulator n_channels={existing.n_channels} "
                        f"vs module.out_features={module.out_features}"
                    )

            in_acc = accumulators[in_key]
            out_acc = accumulators[out_key]
            handles.append(module.register_forward_pre_hook(_make_input_hook(in_acc)))
            handles.append(module.register_forward_hook(_make_output_hook(out_acc)))
            n_modules_hooked += 1

        if shape_warnings:
            # Channel-shape mismatch among experts that classify to the same
            # hook_point is a structural surprise — refuse to proceed silently.
            raise ValueError(
                "D3 attach_hooks: shape mismatch among experts that share a "
                "collapsed hook_point. The accumulator's per-channel max would "
                "be undefined. Refusing to attach hooks. Details:\n  " +
                "\n  ".join(shape_warnings)
            )
    except BaseException:
        # On ANY failure during attach (shape mismatch, OOM during hook
        # registration, KeyboardInterrupt mid-loop), remove all hooks
        # already installed before re-raising. Leaves the caller's model
        # in the same state it was passed in.
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()
        raise

    return {
        "accumulators": accumulators,
        "handles": handles,
        "n_modules_hooked": n_modules_hooked,
        "n_modules_shared_with_existing_acc": n_modules_shared,
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

    # ─── MoE collapse regression test ─────────────────────────────────
    # Mock an MoE layer with 4 experts whose Linears all classify to the
    # same hook_point. The earlier dict-overwrite bug meant only expert 3's
    # hook reached the emitted record, silently dropping experts 0..2.
    class _MoELayer(nn.Module):
        def __init__(self, dim, n_experts):
            super().__init__()
            self.experts = nn.ModuleList([nn.ModuleDict({"down_proj": nn.Linear(dim, dim, bias=False)}) for _ in range(n_experts)])
        def forward(self, x):
            # Apply each expert in sequence with a different input outlier
            # so we can detect whose hook got connected.
            outs = []
            for i, e in enumerate(self.experts):
                xi = x.clone()
                xi[0, 0, i] = 10.0 + i  # expert i sees a unique outlier in channel i
                outs.append(e["down_proj"](xi))
            return sum(outs)

    class _MoEMock(nn.Module):
        def __init__(self, dim, n_experts, n_layers):
            super().__init__()
            self.layers = nn.ModuleList([_MoELayer(dim, n_experts) for _ in range(n_layers)])
        def forward(self, x):
            for L in self.layers:
                x = L(x)
            return x

    moe_dim = 16
    moe_n_experts = 4
    moe = _MoEMock(moe_dim, moe_n_experts, 1)
    moe_state = attach_hooks(moe, {"experts.down_proj"})
    moe_accs = moe_state["accumulators"]
    # ONE accumulator-pair (input+output) per layer per collapsed hook_point.
    assert len(moe_accs) == 2, f"expected 2 accumulators (1 layer × 1 hook_point × 2 sides), got {len(moe_accs)}"
    assert moe_state["n_modules_hooked"] == moe_n_experts, \
        f"expected to hook all {moe_n_experts} expert Linears, got {moe_state['n_modules_hooked']}"
    assert moe_state["n_modules_shared_with_existing_acc"] == moe_n_experts - 1, \
        f"expected experts 1..{moe_n_experts-1} to reuse existing accumulator, got {moe_state['n_modules_shared_with_existing_acc']}"

    x = torch.randn(2, 4, moe_dim) * 0.01
    with torch.no_grad():
        _ = moe(x)
    detach_hooks(moe_state)

    in_acc = moe_accs[(0, "experts.down_proj", "input")]
    in_arr = in_acc.finalize_to_cpu()
    # Each expert should have written its outlier into channel i; max-merge
    # across all experts should leave each channel 0..n-1 reflecting the
    # corresponding outlier value (10+i). If only the LAST expert's hook
    # was connected (the bug), only channel 3 would show ~13 and channels
    # 0..2 would be ~0.01 (typical Gaussian background).
    for i in range(moe_n_experts):
        v = in_arr[i]
        expected = 10.0 + i
        assert v >= expected - 0.5, \
            f"MoE collapse drop bug: expert {i} outlier missing in channel {i} " \
            f"(got {v:.3f}, expected >= {expected - 0.5:.3f}). " \
            f"Earlier-experts' hooks were silently disconnected from the emitted accumulator."
    print(f"[d3 self-test moe] collapse merges all {moe_n_experts} experts: "
          f"channels 0..{moe_n_experts-1} = {[round(in_arr[i], 1) for i in range(moe_n_experts)]}")
    print(f"[d3 self-test moe] hooked {moe_state['n_modules_hooked']} modules into "
          f"{len(moe_accs)} accumulators ({moe_state['n_modules_shared_with_existing_acc']} shared)")

    # ─── Shape-mismatch cleanup test ──────────────────────────────────
    # If sibling experts have mismatched shapes, attach_hooks must raise
    # AND must leave no hooks attached to the caller's model. Otherwise
    # the failed-but-still-installed hooks would mutate state on the
    # next forward pass.
    class _MoEMismatchLayer(nn.Module):
        def __init__(self, dim_a, dim_b):
            super().__init__()
            self.experts = nn.ModuleList([
                nn.ModuleDict({"down_proj": nn.Linear(dim_a, dim_a, bias=False)}),
                nn.ModuleDict({"down_proj": nn.Linear(dim_b, dim_b, bias=False)}),  # different shape
            ])

    class _MoEMismatchMock(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_MoEMismatchLayer(8, 16)])

    mismatch_model = _MoEMismatchMock()

    # Count hooks BEFORE attach so we can verify cleanup fully unwinds.
    def count_hooks(m):
        total = 0
        for sub in m.modules():
            total += len(getattr(sub, "_forward_pre_hooks", {}))
            total += len(getattr(sub, "_forward_hooks", {}))
        return total

    pre_count = count_hooks(mismatch_model)
    raised = False
    try:
        attach_hooks(mismatch_model, {"experts.down_proj"})
    except ValueError as e:
        raised = True
        assert "shape mismatch" in str(e).lower(), f"unexpected ValueError: {e}"
    assert raised, "attach_hooks should have raised ValueError on shape mismatch"
    post_count = count_hooks(mismatch_model)
    assert post_count == pre_count, \
        f"attach_hooks left {post_count - pre_count} stray hooks on the model after refusal " \
        f"(pre={pre_count}, post={post_count}). The caller's model is contaminated."
    print(f"[d3 self-test mismatch] refused on shape mismatch + cleaned up "
          f"({pre_count} hooks before, {post_count} after — no leak)")

    print("[d3 self-test] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
