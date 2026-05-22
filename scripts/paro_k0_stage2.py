#!/usr/bin/env python3
"""paro_k0_stage2.py — ParoQuant K=0 stage-1 + stage-2 prototype for hipfire MQ4G256.

Reference: https://arxiv.org/abs/2511.10645 "ParoQuant: Pairwise Rotation
Quantization for Efficient Reasoning LLM Inference" (Liang et al., ICLR 2026).
Official code: https://github.com/z-lab/paroquant.

This is a K=0 (zero-rotation) reduction of ParoQuant, retargeted to hipfire's
existing MQ4G256 perf-lane. The two ParoQuant components preserved are:

  Stage 1: per-channel scaling `s` (channel_scales), optimized over
    BRECQ-style per-Linear reconstruction loss (5 epochs, Adam LR=0.05).
  Stage 2: weight fine-tuning `W` (5 epochs, Adam LR=1e-5).

The dropped components are:
  - K=8 learned pairwise rotations (paper's main contribution) — requires
    runtime rotation kernel; hipfire MQ4 perf-lane doesn't have one.
  - Learnable quantizer parameters (s_q, z_q) — these are recomputed from
    min/max by `hipfire-quantize`, so optimizing them in PyTorch wouldn't
    survive the hand-off.

What gets exported:
  - BF16 safetensors directory with `W_tuned` replacing `W_orig` for each
    AWQ-target Linear. Non-target Linears + non-Linear tensors are passed
    through unchanged.
  - HFSC v1 sidecar with the optimized `channel_scales` for each target.

This is fed into the standard hipfire-quantize pipeline:
    hipfire-quantize --input <tuned-safetensors-dir> --format mq4 --awq \\
        --awq-scales <hfsc> --awq-alpha 0.55 --awq-scope f1 \\
        --awq-formula paper [--gptq <hessian> --imatrix <imatrix>]
hipfire-quantize re-derives the per-group s_q/z_q via min/max of the
FWHT-rotated W * channel_scales — the same chain our STE uses internally.

Hardware: target is mi300 (gfx942, ROCm 7.0 PyTorch). All math is host-side
PyTorch; no AMD-specific kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FWHT-256 rotation matching hipfire-quantize's gen_fwht_signs (seeds 42, 1042).
# ---------------------------------------------------------------------------

def _gen_fwht_signs(seed: int, n: int = 256) -> torch.Tensor:
    state = seed
    signs = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        signs.append(1.0 if ((state >> 16) & 1) == 1 else -1.0)
    return torch.tensor(signs, dtype=torch.float32)


FWHT_SIGNS1 = _gen_fwht_signs(42)
FWHT_SIGNS2 = _gen_fwht_signs(1042)


def torch_fwht_256(x: torch.Tensor, signs1: torch.Tensor, signs2: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    y = x.reshape(-1, 256).to(torch.float32) * signs1
    stride = 1
    while stride < 256:
        y = y.reshape(-1, 256 // (stride * 2), stride * 2)
        a = y[:, :, :stride].clone()
        b = y[:, :, stride:].clone()
        y[:, :, :stride] = a + b
        y[:, :, stride:] = a - b
        y = y.reshape(-1, 256)
        stride *= 2
    y = y * 0.0625 * signs2
    return y.reshape(orig_shape)


def torch_inverse_fwht_256(x: torch.Tensor, signs1: torch.Tensor, signs2: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    y = x.reshape(-1, 256).to(torch.float32) * signs2
    stride = 1
    while stride < 256:
        y = y.reshape(-1, 256 // (stride * 2), stride * 2)
        a = y[:, :, :stride].clone()
        b = y[:, :, stride:].clone()
        y[:, :, :stride] = a + b
        y[:, :, stride:] = a - b
        y = y.reshape(-1, 256)
        stride *= 2
    y = y * 0.0625 * signs1
    return y.reshape(orig_shape)


def quantize_mq4g256_ste(
    w: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> torch.Tensor:
    """STE pseudo-quantize: forward emits MQ4G256 round-trip, backward passes
    through as if the round/clamp ops were identity. Matches hipfire-quantize's
    `quantize_mq4g256` (main.rs:658+) byte-for-byte modulo numerical precision.
    """
    assert w.dim() == 2, f"MQ4G256 expects 2D weight, got {tuple(w.shape)}"
    m, k = w.shape
    assert k % 256 == 0, f"MQ4G256 requires K%256==0, got K={k}"
    n_groups = k // 256

    w_f32 = w.to(torch.float32)
    rotated = torch_fwht_256(w_f32.reshape(m, n_groups, 256), signs1, signs2)

    with torch.no_grad():
        min_val = rotated.min(dim=-1).values
        max_val = rotated.max(dim=-1).values
        rng = max_val - min_val
        scale = torch.where(rng > 0.0, rng / 15.0, torch.ones_like(rng))
        zero = min_val
        inv_scale = torch.where(rng > 0.0, 1.0 / scale, torch.zeros_like(scale))
        q = torch.clamp(torch.round((rotated - zero.unsqueeze(-1)) * inv_scale.unsqueeze(-1)), 0.0, 15.0)
        dequant_rot = q * scale.unsqueeze(-1) + zero.unsqueeze(-1)
        delta = dequant_rot - rotated

    rotated_quant = rotated + delta.detach()
    w_dq = torch_inverse_fwht_256(rotated_quant.reshape(m, n_groups, 256), signs1, signs2)
    w_dq = w_dq.reshape(m, k)
    return w_dq.to(w.dtype)


# ---------------------------------------------------------------------------
# PseudoQuantizedLinear — K=0 reduction. No rotation kernel, no learnable
# quantizer (s_q/z_q are derived from min/max in STE).
# ---------------------------------------------------------------------------

class K0PseudoQuantizedLinear(nn.Module):
    """ParoQuant K=0 pseudo-quantized Linear. Forward emits the round-trip
    MQ4G256(W * channel_scales) / channel_scales. `weight` and
    `channel_scales` are both `nn.Parameter`s so they participate in autograd.

    Init: weight=BF16 original, channel_scales=closed-form AWQ s = (in_sum2 / mean)^(α/2).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        init_scales: torch.Tensor,
        canonical_name: str,
    ) -> None:
        super().__init__()
        assert weight.dim() == 2
        m, k = weight.shape
        assert init_scales.shape == (k,), f"init scale shape {tuple(init_scales.shape)} != ({k},)"

        # Weight is Parameter — stage 2 will require_grad it.
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)

        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        # Channel scales (stage 1 target).
        self.channel_scales = nn.Parameter(
            init_scales.detach().clone().to(torch.float32),
            requires_grad=False,
        )

        # FWHT sign tables (buffers, never updated).
        self.register_buffer("_fwht_signs1", FWHT_SIGNS1.clone().view(1, 256))
        self.register_buffer("_fwht_signs2", FWHT_SIGNS2.clone().view(1, 256))

        self.canonical_name = canonical_name
        self.k_in = k
        self.m_out = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_safe = torch.clamp(self.channel_scales, min=1e-6).to(torch.float32)

        w_dtype = self.weight.dtype
        # W' = W * s   (broadcast over input channels).
        w_scaled = self.weight.to(torch.float32) * s_safe.unsqueeze(0)
        # MQ4G256 round-trip.
        w_q_dq = quantize_mq4g256_ste(w_scaled, self._fwht_signs1, self._fwht_signs2)
        # Unscale activations: x' = x / s.
        x_scaled = x / s_safe.to(x.dtype).reshape(*[1] * (x.dim() - 1), -1)
        return F.linear(x_scaled, w_q_dq.to(w_dtype), self.bias)

    def set_stage1_optim(self) -> None:
        """Stage 1: only channel_scales is learnable."""
        self.weight.requires_grad = False
        self.channel_scales.requires_grad = True

    def set_stage2_optim(self) -> None:
        """Stage 2: only weight is learnable (paper: weight=1e-5, quantizer=1e-6).
        We drop the quantizer since hipfire-quantize re-derives s_q/z_q from min/max.
        """
        self.weight.requires_grad = True
        self.channel_scales.requires_grad = False

    def freeze(self) -> None:
        self.weight.requires_grad = False
        self.channel_scales.requires_grad = False


# ---------------------------------------------------------------------------
# AWQ scope F1: which Linears to wrap. Matches hipfire-quantize's awq_scope.
# ---------------------------------------------------------------------------

_AWQ_F1_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "qkv_proj.weight",
    "wqkv.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "w_gate.weight",
    "w_up.weight",
    "gate_up_proj.weight",
)


def is_awq_f1_target(safetensors_name: str) -> bool:
    if any(safetensors_name.endswith(sfx) for sfx in _AWQ_F1_SUFFIXES):
        return True
    if ".in_proj_" in safetensors_name:
        return True
    if safetensors_name.endswith("mlp.gate.weight"):
        return True
    return False


# ---------------------------------------------------------------------------
# Closed-form AWQ scales (paper formula) — init for stage 1.
# ---------------------------------------------------------------------------

def compute_awq_scales_paper(in_sum2: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(in_sum2, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("empty imatrix vector")
    half_alpha = float(alpha) * 0.5
    log_s = half_alpha * np.log(np.maximum(values, 1.0e-12))
    log_s -= np.mean(log_s)
    return np.exp(log_s).astype(np.float32)


def load_imatrix_in_sum2(path: Path) -> dict[str, np.ndarray]:
    try:
        import gguf
    except ImportError as exc:
        raise SystemExit(f"gguf package required to read .imatrix.gguf: {exc}")

    reader = gguf.GGUFReader(str(path))
    out: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        if not t.name.endswith(".in_sum2"):
            continue
        stem = t.name[: -len(".in_sum2")]
        out[stem] = np.asarray(t.data).astype(np.float32, copy=False).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# HFSC v1 sidecar export (channel scales hand-off to hipfire-quantize).
# Format defined in `crates/hipfire-quantize/src/awq_scales_io.rs`.
# ---------------------------------------------------------------------------

HFSC_MAGIC = b"HFSC"
HFSC_VERSION = 1


def write_hfsc(path: Path, entries: list[tuple[str, int, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(HFSC_MAGIC)
        f.write(struct.pack("<I", HFSC_VERSION))
        f.write(struct.pack("<Q", len(entries)))
        f.write(struct.pack("<Q", 0))  # reserved
        for name, expert_idx, scales in entries:
            name_bytes = name.encode("utf-8")
            arr = np.asarray(scales, dtype=np.float32).reshape(-1)
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", int(expert_idx)))
            f.write(struct.pack("<I", arr.size))
            f.write(arr.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Safetensors helpers.
# ---------------------------------------------------------------------------

def _safetensors_keys(model_path: str) -> set[str]:
    p = Path(model_path)
    keys: set[str] = set()
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        with idx.open() as f:
            data = json.load(f)
        keys = set(data["weight_map"].keys())
    else:
        from safetensors import safe_open
        for st_file in p.glob("*.safetensors"):
            with safe_open(st_file, framework="pt") as f:
                keys.update(f.keys())
    return {k for k in keys if k.endswith(".weight")}


def _translate_to_stored_name(mod_name: str, stored_keys: set[str]) -> Optional[str]:
    candidate = f"{mod_name}.weight"
    if candidate in stored_keys:
        return candidate
    trail = mod_name.split(".", 1)[1] if "." in mod_name else mod_name
    matches = [k for k in stored_keys if k.endswith(f"{trail}.weight")]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


def _set_module(root: nn.Module, dotted: str, replacement: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], replacement)


def wrap_awq_targets(
    model: nn.Module,
    in_sum2_by_name: dict[str, np.ndarray],
    alpha: float,
    stored_keys: set[str],
) -> dict[str, K0PseudoQuantizedLinear]:
    wrapped: dict[str, K0PseudoQuantizedLinear] = {}
    targets_total = 0
    skip_k_not_256 = 0
    skip_no_imatrix = 0

    target_list: list[tuple[str, nn.Linear, str]] = []
    for mod_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        stored = _translate_to_stored_name(mod_name, stored_keys)
        if stored is None:
            continue
        if not is_awq_f1_target(stored):
            continue
        targets_total += 1
        target_list.append((mod_name, module, stored))

    for mod_name, module, stored in target_list:
        k = module.in_features
        if k % 256 != 0:
            skip_k_not_256 += 1
            continue
        if stored not in in_sum2_by_name:
            skip_no_imatrix += 1
            continue
        in_sum2 = in_sum2_by_name[stored]
        if in_sum2.size != k:
            print(f"  WARN: {stored} imatrix K={in_sum2.size} vs module K={k} - skipping",
                  file=sys.stderr)
            skip_no_imatrix += 1
            continue
        init_s = torch.from_numpy(compute_awq_scales_paper(in_sum2, alpha))
        canonical = stored.removesuffix(".weight")
        weight = module.weight.detach().clone()
        bias = module.bias.detach().clone() if module.bias is not None else None
        device = module.weight.device
        wrapper = K0PseudoQuantizedLinear(weight, bias, init_s, canonical_name=canonical)
        wrapper = wrapper.to(device=device)
        _set_module(model, mod_name, wrapper)
        wrapped[mod_name] = wrapper

    print(
        f"  AWQ-F1 targets in model: {targets_total}, "
        f"wrapped: {len(wrapped)}, "
        f"skipped K%256!=0: {skip_k_not_256}, "
        f"skipped no-imatrix: {skip_no_imatrix}"
    )
    return wrapped


# ---------------------------------------------------------------------------
# Per-Linear output capture hooks (BRECQ reconstruction signal).
# ---------------------------------------------------------------------------

class OutputCapture:
    def __init__(self) -> None:
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles: list = []

    def hook(self, name: str):
        def _hook(module, inputs, output):
            self.outputs[name] = output
        return _hook

    def attach(self, model: nn.Module, target_names: set[str]) -> None:
        for n, m in model.named_modules():
            if n in target_names:
                h = m.register_forward_hook(self.hook(n))
                self.handles.append(h)

    def clear(self) -> None:
        self.outputs.clear()

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def load_calibration(
    corpus: str,
    n_sequences: int,
    ctx_len: int,
    tokenizer,
) -> list[torch.Tensor]:
    p = Path(corpus)
    if p.is_file():
        text = p.read_text()
        all_tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    else:
        raise SystemExit(f"corpus must be a text file path, got: {corpus}")

    n_total = all_tokens.shape[0]
    needed = n_sequences * ctx_len
    if n_total < needed:
        print(f"  WARN: corpus has {n_total} tokens, need {needed} - clamping "
              f"to {n_total // ctx_len} sequences",
              file=sys.stderr)
        n_sequences = n_total // ctx_len
    return [all_tokens[i * ctx_len: (i + 1) * ctx_len] for i in range(n_sequences)]


# ---------------------------------------------------------------------------
# Stage runner: trains one parameter group (channel_scales or weight) for
# `n_epochs` over the BRECQ per-Linear MSE reconstruction signal.
# ---------------------------------------------------------------------------

def run_stage(
    *,
    stage_name: str,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    wrapped: dict[str, K0PseudoQuantizedLinear],
    oracle: nn.Module,
    student: nn.Module,
    oracle_capture: OutputCapture,
    student_capture: OutputCapture,
    seqs: list[torch.Tensor],
    target_names: set[str],
    device: torch.device,
    log_interval: int,
    loss_fn_kind: str,
    grad_clip: Optional[float],
    cosine_floor: float,
    smooth_l1_beta: float,
) -> list[dict]:
    """Run one optimization stage. `stage_name` is informational; the actual
    parameter selection happens via each wrapper's set_stage{1,2}_optim()
    before this is called.
    """
    params: list[nn.Parameter] = []
    for w in wrapped.values():
        for p in w.parameters():
            if p.requires_grad:
                params.append(p)
    if not params:
        print(f"  [{stage_name}] no trainable parameters; skipping")
        return []
    n_params = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
    )
    # Cosine LR schedule decaying to lr * cosine_floor over the full stage.
    total_steps = max(1, n_epochs * len(seqs))
    def lr_lambda(step: int) -> float:
        progress = min(1.0, step / total_steps)
        # cosine from 1.0 → cosine_floor.
        return cosine_floor + 0.5 * (1.0 - cosine_floor) * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    smooth_l1 = nn.SmoothL1Loss(beta=smooth_l1_beta, reduction="mean")

    print(f"  [{stage_name}] lr={lr:g} epochs={n_epochs} "
          f"params={len(params)} elements={n_params}")

    trace: list[dict] = []
    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_count = 0
        for seq_idx, seq in enumerate(seqs):
            input_ids = seq.unsqueeze(0).to(device)
            with torch.no_grad():
                oracle_capture.clear()
                _ = oracle(input_ids)
                y_oracle = {n: oracle_capture.outputs[n].detach()
                            for n in target_names}

            student_capture.clear()
            _ = student(input_ids)

            loss = torch.zeros((), device=device, dtype=torch.float32)
            n_terms = 0
            for n in target_names:
                y_q = student_capture.outputs[n].to(torch.float32)
                y_o = y_oracle[n].to(torch.float32)
                if loss_fn_kind == "smooth_l1":
                    loss = loss + smooth_l1(y_q, y_o)
                else:
                    loss = loss + (y_q - y_o).pow(2).mean()
                n_terms += 1
            loss = loss / max(n_terms, 1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            scheduler.step()

            # Hard-clamp channel_scales to be positive.
            with torch.no_grad():
                for w in wrapped.values():
                    if w.channel_scales.requires_grad:
                        w.channel_scales.clamp_(min=1e-6)

            epoch_loss += float(loss.detach())
            epoch_count += 1

            if (seq_idx + 1) % log_interval == 0:
                running = epoch_loss / max(epoch_count, 1)
                elapsed = time.time() - epoch_start
                cur_lr = scheduler.get_last_lr()[0]
                print(f"      [{stage_name}] epoch {epoch+1}/{n_epochs} "
                      f"seq {seq_idx+1}/{len(seqs)} "
                      f"loss={float(loss.detach()):.6e} running={running:.6e} "
                      f"lr={cur_lr:.3e} ({elapsed:.1f}s)")

            del y_oracle

        mean_loss = epoch_loss / max(epoch_count, 1)
        trace.append({
            "stage": stage_name,
            "epoch": epoch,
            "mean_loss": mean_loss,
            "elapsed_s": time.time() - epoch_start,
            "final_lr": scheduler.get_last_lr()[0],
        })
        print(f"      [{stage_name} epoch {epoch+1}] mean_loss={mean_loss:.6e}  "
              f"epoch_time={time.time() - epoch_start:.1f}s")
    return trace


# ---------------------------------------------------------------------------
# Export tuned weights to a new safetensors directory.
# ---------------------------------------------------------------------------

def export_tuned_safetensors(
    src_dir: Path,
    dst_dir: Path,
    wrapped: dict[str, K0PseudoQuantizedLinear],
    stored_keys: set[str],
) -> None:
    """Copy the source HF model directory verbatim, but replace the .weight
    tensors of wrapped modules with the tuned weights.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    import shutil

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Copy all non-safetensors files (config, tokenizer, etc.)
    for fname in src_dir.iterdir():
        if fname.is_dir():
            continue
        if fname.suffix == ".safetensors":
            continue
        # Skip the safetensors index — we'll regenerate.
        if fname.name == "model.safetensors.index.json":
            continue
        shutil.copy2(fname, dst_dir / fname.name)

    # Map mod_name -> tuned weight tensor.
    tuned_by_stored_key: dict[str, torch.Tensor] = {}
    for mod_name, wrapper in wrapped.items():
        stored = _translate_to_stored_name(mod_name, stored_keys)
        if stored is None:
            print(f"  WARN: could not translate {mod_name} for export", file=sys.stderr)
            continue
        # The tuned weight (post-stage-2). Store in same dtype as original (BF16).
        w = wrapper.weight.detach().cpu()
        tuned_by_stored_key[stored] = w

    # Read source safetensors and write new ones with replacements.
    src_files = sorted(src_dir.glob("*.safetensors"))
    new_index_weight_map: dict[str, str] = {}
    total_bytes_written = 0
    for src_file in src_files:
        out_name = src_file.name
        out_path = dst_dir / out_name
        all_tensors: dict[str, torch.Tensor] = {}
        with safe_open(src_file, framework="pt") as f:
            for k in f.keys():
                if k in tuned_by_stored_key:
                    t = tuned_by_stored_key[k]
                    src_t = f.get_tensor(k)
                    # Coerce to the original safetensors dtype to keep BF16.
                    if t.dtype != src_t.dtype:
                        t = t.to(src_t.dtype)
                    all_tensors[k] = t.contiguous()
                else:
                    all_tensors[k] = f.get_tensor(k)
                new_index_weight_map[k] = out_name
        save_file(all_tensors, str(out_path), metadata={"format": "pt"})
        total_bytes_written += out_path.stat().st_size
        print(f"  wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    # Write the index json if the source had one.
    src_index = src_dir / "model.safetensors.index.json"
    if src_index.exists():
        with src_index.open() as f:
            src_idx = json.load(f)
        new_idx = {
            "metadata": src_idx.get("metadata", {"total_size": total_bytes_written}),
            "weight_map": new_index_weight_map,
        }
        with (dst_dir / "model.safetensors.index.json").open("w") as f:
            json.dump(new_idx, f, indent=2)

    print(f"  exported {len(wrapped)} tuned tensors; "
          f"total safetensors written: {total_bytes_written / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ParoQuant K=0 stage-1 + stage-2 prototype for hipfire MQ4G256.")
    ap.add_argument("--model", required=True, help="HF BF16 model dir path.")
    ap.add_argument("--imatrix", required=True, type=Path, help="Path to .imatrix.gguf.")
    ap.add_argument("--corpus", required=True, help="Path to calibration text file.")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Output directory: tuned safetensors + hfsc + meta.")
    ap.add_argument("--n-sequences", type=int, default=128)
    ap.add_argument("--ctx-len", type=int, default=2048,
                    help="Paper uses 2048; lower if memory pressure.")
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="Closed-form paper-formula alpha for scale initialization.")
    # Paper Stage 1: lr=0.05 for channel_scales, 10 epochs (full PARO 4bit.sh: 5+5).
    ap.add_argument("--stage1-lr", type=float, default=0.05)
    ap.add_argument("--stage1-epochs", type=int, default=5)
    # Paper Stage 2: lr=1e-5 for weight, 10 epochs (full PARO 4bit.sh: 5+5).
    ap.add_argument("--stage2-lr", type=float, default=1e-5)
    ap.add_argument("--stage2-epochs", type=int, default=5)
    # AdamW defaults from paper (Appendix A.1).
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-10)
    ap.add_argument("--cosine-floor", type=float, default=0.05,
                    help="Final LR = lr * cosine_floor (paper: ~1/20).")
    ap.add_argument("--loss-fn", choices=["smooth_l1", "mse"], default="smooth_l1")
    ap.add_argument("--smooth-l1-beta", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--log-interval", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--skip-stage1", action="store_true", help="Diagnostic.")
    ap.add_argument("--skip-stage2", action="store_true", help="Diagnostic.")
    ap.add_argument("--skip-export-weights", action="store_true",
                    help="Diagnostic: only emit HFSC, not tuned safetensors.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("ParoQuant K=0 stage-1 + stage-2 (MQ4G256 perf-lane prototype)")
    print("=" * 80)
    print(f"  model:           {args.model}")
    print(f"  imatrix:         {args.imatrix}")
    print(f"  corpus:          {args.corpus}")
    print(f"  output dir:      {args.output_dir}")
    print(f"  n-sequences:     {args.n_sequences}")
    print(f"  ctx-len:         {args.ctx_len}")
    print(f"  alpha (init):    {args.alpha}")
    print(f"  stage1: lr={args.stage1_lr}  epochs={args.stage1_epochs}  (channel_scales)")
    print(f"  stage2: lr={args.stage2_lr}  epochs={args.stage2_epochs}  (weight)")
    print(f"  loss:            {args.loss_fn}")
    print(f"  grad-clip:       {args.grad_clip}")
    print()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print("[1/7] Loading BF16 oracle model + tokenizer ...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    oracle = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    oracle.eval()
    for p in oracle.parameters():
        p.requires_grad_(False)
    print(f"      loaded oracle in {time.time() - t0:.1f}s. "
          f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n[2/7] Loading fresh BF16 student model + wrapping AWQ-F1 targets ...")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    print(f"      loaded student in {time.time() - t0:.1f}s. "
          f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n[3/7] Reading imatrix + initializing channel_scales (paper formula) ...")
    t0 = time.time()
    in_sum2_by_name = load_imatrix_in_sum2(args.imatrix)
    print(f"      imatrix tensors: {len(in_sum2_by_name)}")
    stored_keys = _safetensors_keys(args.model)
    print(f"      safetensors keys: {len(stored_keys)}")
    wrapped = wrap_awq_targets(student, in_sum2_by_name, args.alpha, stored_keys)
    print(f"      wrap done in {time.time() - t0:.1f}s. "
          f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    if not wrapped:
        raise SystemExit("no AWQ targets wrapped - bailing")

    target_names = set(wrapped.keys())
    oracle_capture = OutputCapture()
    student_capture = OutputCapture()
    oracle_capture.attach(oracle, target_names)
    student_capture.attach(student, target_names)
    print(f"      hooks attached: oracle={len(oracle_capture.handles)}, "
          f"student={len(student_capture.handles)}")

    print("\n[4/7] Tokenizing calibration corpus ...")
    t0 = time.time()
    seqs = load_calibration(args.corpus, args.n_sequences, args.ctx_len, tokenizer)
    print(f"      {len(seqs)} sequences x {args.ctx_len} ctx "
          f"= {len(seqs) * args.ctx_len} tokens in {time.time() - t0:.1f}s")

    device = next(student.parameters()).device

    # Snapshot init state for diagnostics.
    init_scale_norms = {n: float(w.channel_scales.detach().norm().cpu()) for n, w in wrapped.items()}
    init_weight_norms = {n: float(w.weight.detach().to(torch.float32).norm().cpu()) for n, w in wrapped.items()}

    full_trace: list[dict] = []
    train_start = time.time()

    if not args.skip_stage1:
        print(f"\n[5/7] Stage 1 — channel_scales optimization "
              f"({args.stage1_epochs} epochs, lr={args.stage1_lr}) ...")
        for w in wrapped.values():
            w.set_stage1_optim()
        trace1 = run_stage(
            stage_name="stage1-channel_scales",
            n_epochs=args.stage1_epochs,
            lr=args.stage1_lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            wrapped=wrapped,
            oracle=oracle, student=student,
            oracle_capture=oracle_capture, student_capture=student_capture,
            seqs=seqs, target_names=target_names,
            device=device,
            log_interval=args.log_interval,
            loss_fn_kind=args.loss_fn,
            grad_clip=args.grad_clip,
            cosine_floor=args.cosine_floor,
            smooth_l1_beta=args.smooth_l1_beta,
        )
        full_trace.extend(trace1)
    else:
        print("\n[5/7] Stage 1 SKIPPED (--skip-stage1)")

    if not args.skip_stage2:
        print(f"\n[6/7] Stage 2 — weight fine-tuning "
              f"({args.stage2_epochs} epochs, lr={args.stage2_lr}) ...")
        for w in wrapped.values():
            w.set_stage2_optim()
        trace2 = run_stage(
            stage_name="stage2-weight",
            n_epochs=args.stage2_epochs,
            lr=args.stage2_lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            wrapped=wrapped,
            oracle=oracle, student=student,
            oracle_capture=oracle_capture, student_capture=student_capture,
            seqs=seqs, target_names=target_names,
            device=device,
            log_interval=args.log_interval,
            loss_fn_kind=args.loss_fn,
            grad_clip=args.grad_clip,
            cosine_floor=args.cosine_floor,
            smooth_l1_beta=args.smooth_l1_beta,
        )
        full_trace.extend(trace2)
    else:
        print("\n[6/7] Stage 2 SKIPPED (--skip-stage2)")

    train_elapsed = time.time() - train_start
    print(f"\n      total train time: {train_elapsed:.1f}s")

    # Smoke verification.
    print("\n      param-movement summary:")
    for n, w in list(wrapped.items())[:8]:
        s_init = init_scale_norms[n]
        s_final = float(w.channel_scales.detach().norm().cpu())
        w_init = init_weight_norms[n]
        w_final = float(w.weight.detach().to(torch.float32).norm().cpu())
        print(f"        {n}:")
        print(f"           channel_scales norm {s_init:.4f} -> {s_final:.4f} "
              f"(rel {abs(s_init - s_final) / max(s_init, 1e-9):.2e})")
        print(f"           weight norm        {w_init:.4f} -> {w_final:.4f} "
              f"(rel {abs(w_init - w_final) / max(w_init, 1e-9):.2e})")

    oracle_capture.detach()
    student_capture.detach()

    print(f"\n[7/7] Exporting tuned safetensors + HFSC ...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) HFSC sidecar with optimized channel_scales.
    hfsc_path = args.output_dir / "paro_channel_scales.hfsc"
    entries: list[tuple[str, int, np.ndarray]] = []
    for mod_name, wrapper in sorted(wrapped.items()):
        scales = wrapper.channel_scales.detach().to(torch.float32).cpu().numpy().astype(np.float32)
        entries.append((wrapper.canonical_name, 0, scales))
    write_hfsc(hfsc_path, entries)
    print(f"      wrote {len(entries)} HFSC entries -> {hfsc_path} "
          f"({hfsc_path.stat().st_size / 1e6:.2f} MB)")

    # 2) Tuned BF16 safetensors directory.
    if not args.skip_export_weights:
        tuned_dir = args.output_dir / "tuned-safetensors"
        export_tuned_safetensors(Path(args.model), tuned_dir, wrapped, stored_keys)
        print(f"      tuned safetensors dir: {tuned_dir}")
    else:
        print("      SKIPPED tuned safetensors export (--skip-export-weights)")

    # 3) Meta.
    meta = {
        "model": args.model,
        "imatrix": str(args.imatrix),
        "corpus": args.corpus,
        "n_sequences": args.n_sequences,
        "ctx_len": args.ctx_len,
        "alpha": args.alpha,
        "stage1_lr": args.stage1_lr,
        "stage1_epochs": args.stage1_epochs,
        "stage2_lr": args.stage2_lr,
        "stage2_epochs": args.stage2_epochs,
        "loss_fn": args.loss_fn,
        "grad_clip": args.grad_clip,
        "cosine_floor": args.cosine_floor,
        "n_wrapped": len(wrapped),
        "loss_trace": full_trace,
        "train_elapsed_s": train_elapsed,
        "skip_stage1": args.skip_stage1,
        "skip_stage2": args.skip_stage2,
        "skip_export_weights": args.skip_export_weights,
    }
    meta_path = args.output_dir / "paro_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"      meta -> {meta_path}")

    print("\n=== Done ===")
    print(f"Next step: hipfire-quantize --input {args.output_dir}/tuned-safetensors \\")
    print(f"            --awq --awq-scales {hfsc_path} --awq-formula paper \\")
    print(f"            --awq-alpha {args.alpha} --awq-scope f1 ...")


if __name__ == "__main__":
    main()
