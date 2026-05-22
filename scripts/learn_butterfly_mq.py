#!/usr/bin/env python3
"""learn_butterfly_mq.py — Tier 3 butterfly residual learner for hipfire MQ4G256.

Reference: ButterflyQuant (arXiv:2509.09679, Xu et al., Feb 2026). Adapted from
PARO K=0's pipeline structure (scripts/paro_k0_stage2.py on the
feat/paro-stage2-mq4-prototype branch). The math choice is described in the
docstring of scripts/butterfly_core.py — residual butterfly composes WITH the
existing FWHT, not REPLACES it. At theta = 0 the pipeline reduces to current
MQ4+AWQ byte-equal.

Pipeline (per AWQ-F1 target Linear):

  1. Channel scales  s_AWQ = (in_sum2)^(alpha/2)  (closed-form, FROZEN).
  2. Weight          W = BF16 original  (FROZEN).
  3. Rotation        T(theta) = B_residual(theta) ∘ FWHT_canonical.
  4. theta_residual  shape [8, 128] per Linear, LEARNED via SGD+cosine, init 0.

Forward in pseudo-quantized Linear:
    x'      = x / s_AWQ
    W_rot   = T(theta)(W * s_AWQ)        # per-256-group rotation
    W_q     = MQ4_min_max_RTN(W_rot)     # 4-bit, 256-group
    W_dq    = T(theta)^{-1}(W_q)         # inverse rotation
    y_q     = Linear(x', W_dq)           # output in original (unrotated) space

Loss (BRECQ-style per-Linear MSE):
    L(theta) = sum_t || y_q(t) - y_oracle(t) ||²

where t indexes wrapped Linears in the model. Optimizer: SGD with momentum and
cosine LR schedule (paper hparams: 128 samples × 2048 ctx, ~500-700 steps total).

Outputs:
    <output-dir>/butterfly_residuals.npz      # key = canonical Linear name → [8, 128] F32
    <output-dir>/butterfly_residuals.hfbf     # HFBF v1 binary sidecar for Rust port
    <output-dir>/paper_awq_scales.hfsc        # HFSC v1 sidecar for AWQ scales (unchanged)
    <output-dir>/butterfly_meta.json          # hparams + loss trace + per-tensor norms

The HFBF format is documented in `_write_hfbf` below.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FWHT-256 rotation — matches hipfire-quantize's gen_fwht_signs (seeds 42, 1042).
# Mirrored from PARO K=0 (scripts/paro_k0_stage2.py).
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
    """In-place stride-doubling FWHT on the last axis (length 256)."""
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
    y = y * 0.0625 * signs2  # 1/sqrt(256) = 1/16 = 0.0625
    return y.reshape(orig_shape)


def torch_inverse_fwht_256(x: torch.Tensor, signs1: torch.Tensor, signs2: torch.Tensor) -> torch.Tensor:
    """Inverse FWHT-256 — swaps signs1/signs2 at endpoints (the FWHT is its own
    inverse up to normalization)."""
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


# ---------------------------------------------------------------------------
# Differentiable butterfly residual — PyTorch port of butterfly_core.butterfly256.
# ---------------------------------------------------------------------------

LAYER_COUNT = 8
N = 256
PAIRS_PER_LAYER = N // 2  # 128


def torch_butterfly256(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply B_residual(theta) along the last axis.

    Args:
        x: tensor with last axis = 256.
        theta: [8, 128] tensor of Givens angles, broadcast across all prefix dims.

    Returns:
        rotated tensor of same shape as x.
    """
    assert x.shape[-1] == N
    assert theta.shape == (LAYER_COUNT, PAIRS_PER_LAYER)
    y = x.to(torch.float32)
    for layer in range(LAYER_COUNT):
        stride = 1 << layer
        n_blocks = N // (2 * stride)
        prefix_shape = y.shape[:-1]
        y_blocks = y.reshape(*prefix_shape, n_blocks, 2 * stride)
        a = y_blocks[..., :stride]
        b = y_blocks[..., stride:]
        # Pair p in [0, 128) maps to (block, within) = (p // stride, p % stride).
        cos_lay = torch.cos(theta[layer]).reshape(n_blocks, stride)
        sin_lay = torch.sin(theta[layer]).reshape(n_blocks, stride)
        new_a = cos_lay * a - sin_lay * b
        new_b = sin_lay * a + cos_lay * b
        y = torch.cat([new_a, new_b], dim=-1).reshape(*prefix_shape, N)
    return y


def torch_butterfly256_inverse(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply B_residual(theta)^T = B_residual(-theta) along the last axis."""
    assert y.shape[-1] == N
    assert theta.shape == (LAYER_COUNT, PAIRS_PER_LAYER)
    x = y.to(torch.float32)
    for layer in range(LAYER_COUNT - 1, -1, -1):
        stride = 1 << layer
        n_blocks = N // (2 * stride)
        prefix_shape = x.shape[:-1]
        x_blocks = x.reshape(*prefix_shape, n_blocks, 2 * stride)
        a = x_blocks[..., :stride]
        b = x_blocks[..., stride:]
        cos_lay = torch.cos(theta[layer]).reshape(n_blocks, stride)
        sin_lay = torch.sin(theta[layer]).reshape(n_blocks, stride)
        new_a = cos_lay * a + sin_lay * b
        new_b = -sin_lay * a + cos_lay * b
        x = torch.cat([new_a, new_b], dim=-1).reshape(*prefix_shape, N)
    return x


# ---------------------------------------------------------------------------
# MQ4G256 STE with butterfly residual on top of FWHT.
# ---------------------------------------------------------------------------

def quantize_mq4g256_bfly_ste(
    w: torch.Tensor,
    theta: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> torch.Tensor:
    """MQ4G256 round-trip with learnable butterfly residual.

    Forward: emit dequant(quant(B_res(theta) ∘ FWHT(W))) → unrotate.
    Backward: STE through min/max/round/clamp; theta gradients flow through
    both B_res and B_res^{-1} (no detach on the rotation operations).
    """
    assert w.dim() == 2
    m, k = w.shape
    assert k % N == 0, f"K={k} must be divisible by {N}"
    n_groups = k // N

    w_f32 = w.to(torch.float32)
    # FWHT step (theta-independent).
    w_fwht = torch_fwht_256(w_f32.reshape(m, n_groups, N), signs1, signs2)
    # Butterfly residual step (theta-dependent, autograd-tracked).
    w_rotated = torch_butterfly256(w_fwht, theta)

    # MQ4 min-max RTN with STE.
    with torch.no_grad():
        min_val = w_rotated.min(dim=-1).values
        max_val = w_rotated.max(dim=-1).values
        rng = max_val - min_val
        scale = torch.where(rng > 0.0, rng / 15.0, torch.ones_like(rng))
        zero = min_val
        inv_scale = torch.where(rng > 0.0, 1.0 / scale, torch.zeros_like(scale))
        q = torch.clamp(
            torch.round((w_rotated - zero.unsqueeze(-1)) * inv_scale.unsqueeze(-1)),
            0.0, 15.0,
        )
        dequant_rot = q * scale.unsqueeze(-1) + zero.unsqueeze(-1)
        delta = dequant_rot - w_rotated

    w_quant_rot = w_rotated + delta.detach()  # STE through round/clamp
    # Inverse butterfly (theta-dependent).
    w_inv_bfly = torch_butterfly256_inverse(w_quant_rot, theta)
    # Inverse FWHT (theta-independent).
    w_dq = torch_inverse_fwht_256(w_inv_bfly, signs1, signs2)
    return w_dq.reshape(m, k).to(w.dtype)


# ---------------------------------------------------------------------------
# Pseudo-quantized Linear with frozen weight + frozen AWQ scales + learnable theta.
# ---------------------------------------------------------------------------

class ButterflyResidualPseudoQuantLinear(nn.Module):
    """MQ4G256 + AWQ + learnable butterfly residual.

    At theta = 0 this reduces to current MQ4+AWQ pseudo-quant (the baseline
    we're trying to beat). theta is the only learnable parameter.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        channel_scales_init: torch.Tensor,
        canonical_name: str,
    ) -> None:
        super().__init__()
        assert weight.dim() == 2
        m, k = weight.shape
        assert k % N == 0, f"K={k} must be divisible by {N}"
        assert channel_scales_init.shape == (k,)

        # Frozen weight + bias.
        self.register_buffer("weight", weight.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        # Frozen AWQ channel scales (paper-formula closed form).
        self.register_buffer(
            "channel_scales",
            channel_scales_init.detach().clone().to(torch.float32),
        )

        # Learnable butterfly residual.
        self.theta_residual = nn.Parameter(
            torch.zeros(LAYER_COUNT, PAIRS_PER_LAYER, dtype=torch.float32),
            requires_grad=False,
        )

        self.register_buffer("_fwht_signs1", FWHT_SIGNS1.clone().view(1, N))
        self.register_buffer("_fwht_signs2", FWHT_SIGNS2.clone().view(1, N))

        self.canonical_name = canonical_name
        self.k_in = k
        self.m_out = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_safe = torch.clamp(self.channel_scales, min=1e-6).to(torch.float32)
        w_dtype = self.weight.dtype
        w_scaled = self.weight.to(torch.float32) * s_safe.unsqueeze(0)
        w_q_dq = quantize_mq4g256_bfly_ste(
            w_scaled, self.theta_residual, self._fwht_signs1, self._fwht_signs2,
        )
        x_scaled = x / s_safe.to(x.dtype).reshape(*[1] * (x.dim() - 1), -1)
        return F.linear(x_scaled, w_q_dq.to(w_dtype), self.bias)

    def set_optim(self) -> None:
        self.theta_residual.requires_grad = True


# ---------------------------------------------------------------------------
# AWQ-F1 target detection + closed-form scale init (from PARO K=0).
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


def compute_awq_scales_paper(in_sum2: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form AWQ paper-formula: s = exp((alpha/2) * (log(in_sum2) - mean))."""
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
# HFSC v1 sidecar export (AWQ scales — unchanged from PARO K=0).
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
# HFBF v1 sidecar export (butterfly residuals — NEW format for Phase 8 Rust port).
# Format:
#   4 B  magic   = "HFBF"
#   4 B  version = u32 LE (= 1)
#   8 B  n_entries = u64 LE
#   8 B  reserved  = u64 LE (must be 0)
# Per entry:
#   4 B  name_len = u32 LE
#   N B  name = UTF-8 (canonical Linear name, e.g. "model.layers.0.q_proj")
#   4 B  expert_idx = u32 LE (0 for dense, MoE expert index for MoE)
#   4 B  layer_count = u32 LE (= 8)
#   4 B  pairs_per_layer = u32 LE (= 128)
#   8*128*4 B  theta = F32 LE row-major [layer_count, pairs_per_layer]
# ---------------------------------------------------------------------------

HFBF_MAGIC = b"HFBF"
HFBF_VERSION = 1


def write_hfbf(path: Path, entries: list[tuple[str, int, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(HFBF_MAGIC)
        f.write(struct.pack("<I", HFBF_VERSION))
        f.write(struct.pack("<Q", len(entries)))
        f.write(struct.pack("<Q", 0))  # reserved
        for name, expert_idx, theta in entries:
            name_bytes = name.encode("utf-8")
            arr = np.asarray(theta, dtype=np.float32).reshape(LAYER_COUNT, PAIRS_PER_LAYER)
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", int(expert_idx)))
            f.write(struct.pack("<I", LAYER_COUNT))
            f.write(struct.pack("<I", PAIRS_PER_LAYER))
            f.write(arr.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Safetensors helpers (from PARO K=0).
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


# ---------------------------------------------------------------------------
# Model wrapping.
# ---------------------------------------------------------------------------

import re


def wrap_awq_targets(
    model: nn.Module,
    in_sum2_by_name: dict[str, np.ndarray],
    alpha: float,
    stored_keys: set[str],
    target_pattern: Optional[str],
) -> dict[str, ButterflyResidualPseudoQuantLinear]:
    """Wrap all AWQ-F1 Linears whose stored name matches `target_pattern`.

    Returns dict mod_name → wrapper.
    """
    pat = re.compile(target_pattern) if target_pattern else None
    wrapped: dict[str, ButterflyResidualPseudoQuantLinear] = {}
    skipped_no_imatrix = 0
    skipped_k_not_256 = 0
    skipped_pattern = 0
    total_f1 = 0

    targets: list[tuple[str, nn.Linear, str]] = []
    for mod_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        stored = _translate_to_stored_name(mod_name, stored_keys)
        if stored is None:
            continue
        if not is_awq_f1_target(stored):
            continue
        total_f1 += 1
        if pat is not None and not pat.search(stored):
            skipped_pattern += 1
            continue
        targets.append((mod_name, module, stored))

    for mod_name, module, stored in targets:
        k = module.in_features
        if k % N != 0:
            skipped_k_not_256 += 1
            continue
        if stored not in in_sum2_by_name:
            skipped_no_imatrix += 1
            continue
        in_sum2 = in_sum2_by_name[stored]
        if in_sum2.size != k:
            print(f"  WARN: {stored} imatrix K={in_sum2.size} vs module K={k} - skipping",
                  file=sys.stderr)
            skipped_no_imatrix += 1
            continue
        init_s = torch.from_numpy(compute_awq_scales_paper(in_sum2, alpha))
        canonical = stored.removesuffix(".weight")
        weight = module.weight.detach().clone()
        bias = module.bias.detach().clone() if module.bias is not None else None
        device = module.weight.device
        wrapper = ButterflyResidualPseudoQuantLinear(weight, bias, init_s, canonical)
        wrapper = wrapper.to(device=device)
        _set_module(model, mod_name, wrapper)
        wrapped[mod_name] = wrapper

    print(
        f"  AWQ-F1 targets: {total_f1} total, "
        f"{len(wrapped)} wrapped, "
        f"{skipped_pattern} excluded by --target-pattern, "
        f"{skipped_k_not_256} skipped K%256, "
        f"{skipped_no_imatrix} skipped no-imatrix"
    )
    return wrapped


# ---------------------------------------------------------------------------
# Output capture hooks (BRECQ per-Linear MSE signal).
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


# ---------------------------------------------------------------------------
# Calibration corpus tokenization.
# ---------------------------------------------------------------------------

def load_calibration(
    corpus: str,
    n_sequences: int,
    ctx_len: int,
    tokenizer,
) -> list[torch.Tensor]:
    p = Path(corpus)
    if not p.is_file():
        raise SystemExit(f"corpus must be a text file path, got: {corpus}")
    text = p.read_text()
    needed = n_sequences * ctx_len
    cap_chars = needed * 6
    if len(text) > cap_chars:
        text = text[:cap_chars]
        print(f"  capped corpus to {cap_chars} chars (~{needed} tokens budget)",
              file=sys.stderr)
    print(f"  tokenizing {len(text)} chars ...", file=sys.stderr)
    t0 = time.time()
    all_tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    print(f"  tokenized {all_tokens.shape[0]} tokens in {time.time() - t0:.1f}s",
          file=sys.stderr)
    n_total = all_tokens.shape[0]
    if n_total < needed:
        print(f"  WARN: corpus has {n_total} tokens, need {needed} - clamping to "
              f"{n_total // ctx_len} sequences", file=sys.stderr)
        n_sequences = n_total // ctx_len
    return [all_tokens[i * ctx_len: (i + 1) * ctx_len] for i in range(n_sequences)]


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------

def run_butterfly_training(
    *,
    wrapped: dict[str, ButterflyResidualPseudoQuantLinear],
    oracle: nn.Module,
    student: nn.Module,
    oracle_capture: OutputCapture,
    student_capture: OutputCapture,
    seqs: list[torch.Tensor],
    target_names: set[str],
    device: torch.device,
    n_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    cosine_floor: float,
    grad_clip: Optional[float],
    l2_theta: float,
    log_interval: int,
) -> list[dict]:
    """SGD + cosine training over butterfly residuals."""
    params: list[nn.Parameter] = []
    for w in wrapped.values():
        w.set_optim()
        for p in w.parameters():
            if p.requires_grad:
                params.append(p)
    if not params:
        print("  no trainable parameters; skipping")
        return []
    n_params = sum(p.numel() for p in params)
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False,
    )
    total_steps = max(1, n_epochs * len(seqs))

    def lr_lambda(step: int) -> float:
        progress = min(1.0, step / total_steps)
        return cosine_floor + 0.5 * (1.0 - cosine_floor) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  SGD: lr={lr:g} momentum={momentum} wd={weight_decay} "
          f"epochs={n_epochs} steps={total_steps} params={len(params)} "
          f"elements={n_params} l2_theta={l2_theta}")

    trace: list[dict] = []
    global_step = 0
    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_count = 0
        for seq_idx, seq in enumerate(seqs):
            input_ids = seq.unsqueeze(0).to(device)
            with torch.no_grad():
                oracle_capture.clear()
                _ = oracle(input_ids)
                y_oracle = {n: oracle_capture.outputs[n].detach() for n in target_names}

            student_capture.clear()
            _ = student(input_ids)

            loss = torch.zeros((), device=device, dtype=torch.float32)
            n_terms = 0
            for n in target_names:
                y_q = student_capture.outputs[n].to(torch.float32)
                y_o = y_oracle[n].to(torch.float32)
                loss = loss + (y_q - y_o).pow(2).mean()
                n_terms += 1
            loss = loss / max(n_terms, 1)

            if l2_theta > 0.0:
                reg = torch.zeros((), device=device, dtype=torch.float32)
                for w in wrapped.values():
                    reg = reg + w.theta_residual.pow(2).sum()
                loss = loss + l2_theta * reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += float(loss.detach())
            epoch_count += 1

            if (seq_idx + 1) % log_interval == 0:
                running = epoch_loss / max(epoch_count, 1)
                cur_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                print(f"      epoch {epoch + 1}/{n_epochs} seq {seq_idx + 1}/{len(seqs)} "
                      f"loss={float(loss.detach()):.6e} running={running:.6e} "
                      f"lr={cur_lr:.3e} ({elapsed:.1f}s)")

            del y_oracle

        mean_loss = epoch_loss / max(epoch_count, 1)
        trace.append({
            "epoch": epoch,
            "mean_loss": mean_loss,
            "elapsed_s": time.time() - epoch_start,
            "final_lr": scheduler.get_last_lr()[0],
        })
        print(f"      [epoch {epoch + 1}] mean_loss={mean_loss:.6e}  "
              f"time={time.time() - epoch_start:.1f}s")

    return trace


# ---------------------------------------------------------------------------
# Logit-KLD smoke eval (Phase 3 gate).
# ---------------------------------------------------------------------------

def compute_logit_kld(
    oracle: nn.Module,
    student: nn.Module,
    seqs: list[torch.Tensor],
    device: torch.device,
) -> float:
    """Mean per-token KL(oracle || student) over the eval slice.

    KLD = sum_v p(v) * (log p(v) - log q(v)) with p = softmax(oracle logits),
    q = softmax(student logits). Averaged over all eval token positions.
    """
    total_kld = 0.0
    total_tokens = 0
    with torch.no_grad():
        for seq in seqs:
            input_ids = seq.unsqueeze(0).to(device)
            logits_oracle = oracle(input_ids).logits.float()  # [1, T, V]
            logits_student = student(input_ids).logits.float()
            log_p = F.log_softmax(logits_oracle, dim=-1)
            log_q = F.log_softmax(logits_student, dim=-1)
            p = log_p.exp()
            kl = (p * (log_p - log_q)).sum(dim=-1)  # [1, T]
            total_kld += float(kl.sum())
            total_tokens += int(kl.numel())
    return total_kld / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Self-test (no real model required — random weights, single-Linear sanity).
# ---------------------------------------------------------------------------

def self_test(device_str: str = "cpu") -> int:
    """Verify (a) torch butterfly matches the numpy reference at random theta,
    (b) theta=0 in the pseudo-quant Linear reproduces the MQ4+AWQ baseline,
    and (c) SGD step on a synthetic Linear moves theta away from 0.
    """
    print("Self-test: butterfly residual training step + torch/numpy parity")
    print("=" * 70)
    device = torch.device(device_str)

    # Parity vs numpy reference (catches indexing/reshape bugs in torch port).
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    from scripts.butterfly_core import (  # noqa: E402
        butterfly256 as np_butterfly256,
        butterfly256_inverse as np_butterfly256_inverse,
    )
    rng = np.random.default_rng(7)
    parity_max_err = 0.0
    parity_inv_err = 0.0
    for _ in range(64):
        x_np = rng.standard_normal((3, N)).astype(np.float32)
        theta_np = rng.uniform(-np.pi, np.pi, size=(LAYER_COUNT, PAIRS_PER_LAYER)).astype(np.float32)
        # Forward parity.
        y_np = np_butterfly256(x_np, theta_np)
        y_torch = torch_butterfly256(
            torch.from_numpy(x_np), torch.from_numpy(theta_np),
        ).numpy()
        parity_max_err = max(parity_max_err, float(np.abs(y_np - y_torch).max()))
        # Inverse parity.
        x_back_np = np_butterfly256_inverse(y_np, theta_np)
        x_back_torch = torch_butterfly256_inverse(
            torch.from_numpy(y_np), torch.from_numpy(theta_np),
        ).numpy()
        parity_inv_err = max(parity_inv_err, float(np.abs(x_back_np - x_back_torch).max()))
    print(
        f"  [parity] torch vs numpy butterfly   forward max_err = {parity_max_err:.3e}, "
        f"inverse max_err = {parity_inv_err:.3e}"
    )
    if parity_max_err > 1e-5 or parity_inv_err > 1e-5:
        print("  FAIL: torch/numpy parity drift > 1e-5 — check indexing/reshape")
        return 1

    torch.manual_seed(0)
    m, k = 64, 512
    weight = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    bias = torch.zeros(m, device=device, dtype=torch.bfloat16)
    scales = torch.ones(k, device=device, dtype=torch.float32)

    wrapper = ButterflyResidualPseudoQuantLinear(
        weight, bias, scales, canonical_name="test.linear",
    ).to(device=device)

    # Sanity 1: theta=0 → output equals direct MQ4G256+AWQ pseudo-quant.
    x = torch.randn(2, 32, k, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        wrapper.theta_residual.zero_()
        y_theta0 = wrapper(x)
        # Reference: same pipeline with no butterfly residual (theta=0).
        s_safe = torch.clamp(wrapper.channel_scales, min=1e-6).to(torch.float32)
        x_scaled = x / s_safe.to(x.dtype).reshape(*[1] * (x.dim() - 1), -1)
        w_scaled = wrapper.weight.to(torch.float32) * s_safe.unsqueeze(0)
        w_q_dq_ref = quantize_mq4g256_bfly_ste(
            w_scaled, torch.zeros(LAYER_COUNT, PAIRS_PER_LAYER, device=device),
            wrapper._fwht_signs1, wrapper._fwht_signs2,
        )
        y_ref = F.linear(x_scaled, w_q_dq_ref.to(wrapper.weight.dtype), wrapper.bias)
    diff = (y_theta0 - y_ref).abs().max().item()
    print(f"  [check 1] theta=0 reproduces baseline      max_err = {diff:.3e}")
    if diff > 1e-3:
        print("  FAIL: theta=0 does not reproduce baseline")
        return 1

    # Sanity 2: a single SGD step on a random oracle moves theta.
    wrapper.set_optim()
    optimizer = torch.optim.SGD([wrapper.theta_residual], lr=0.1, momentum=0.0)
    torch.manual_seed(1)
    y_target = torch.randn_like(y_ref)
    initial_theta_norm = float(wrapper.theta_residual.detach().norm())
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        y_q = wrapper(x).to(torch.float32)
        loss = (y_q - y_target.to(torch.float32)).pow(2).mean()
        loss.backward()
        # Gradient must be nonzero on theta after backward.
        if step == 0:
            grad_norm = float(wrapper.theta_residual.grad.norm())
            print(f"  [check 2] gradient flow through theta    ||grad|| = {grad_norm:.3e}")
            if grad_norm == 0.0:
                print("  FAIL: theta gradient is zero — autograd not connected")
                return 1
        optimizer.step()
    final_theta_norm = float(wrapper.theta_residual.detach().norm())
    moved = abs(final_theta_norm - initial_theta_norm)
    print(f"  [check 3] theta moved after SGD steps     ||theta||: {initial_theta_norm:.3e} → "
          f"{final_theta_norm:.3e}  (Δ={moved:.3e})")
    if moved < 1e-6:
        print("  FAIL: theta did not move after 10 SGD steps")
        return 1

    print("=" * 60)
    print("ALL SELF-TEST CHECKS PASS")
    return 0


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true",
                    help="Run a synthetic self-test (no model required) and exit.")
    ap.add_argument("--model", help="HF BF16 model dir path.")
    ap.add_argument("--imatrix", type=Path, help="Path to .imatrix.gguf.")
    ap.add_argument("--corpus", help="Path to calibration text file.")
    ap.add_argument("--output-dir", type=Path,
                    help="Output directory for residuals + meta + sidecars.")
    ap.add_argument("--target-pattern", default=None,
                    help="Optional regex restricting wrapped tensors (e.g. for "
                         "Phase 3 smoke: 'layers\\.0\\.(self_attn|mlp)\\..*proj').")
    ap.add_argument("--n-sequences", type=int, default=128,
                    help="ButterflyQuant paper recipe: 128 samples.")
    ap.add_argument("--ctx-len", type=int, default=2048,
                    help="ButterflyQuant paper recipe: 2048 ctx.")
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="AWQ closed-form paper-formula alpha.")
    ap.add_argument("--n-epochs", type=int, default=4,
                    help="Total steps = n_epochs * n_sequences. Paper guidance "
                         "500-700 steps; default 4*128=512 sits in that range.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="SGD LR. Paper hparams uncertain; conservative default.")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Weight decay on theta (paper does not regularize "
                         "explicitly; --l2-theta is the alternative knob).")
    ap.add_argument("--cosine-floor", type=float, default=0.05,
                    help="Final LR = initial * floor (paper ~1/20).")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--l2-theta", type=float, default=0.0,
                    help="Optional small L2 penalty on theta magnitude.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--log-interval", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--smoke-eval", action="store_true",
                    help="Run logit-KLD smoke eval (baseline theta=0 vs trained) "
                         "before AND after training. Phase 3 gate.")
    ap.add_argument("--smoke-eval-seqs", type=int, default=16,
                    help="Number of sequences for --smoke-eval KLD computation.")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"))

    # Validate required args for non-self-test mode.
    missing = [name for name in ("model", "imatrix", "corpus", "output_dir")
               if getattr(args, name) is None]
    if missing:
        ap.error(f"missing required args for training mode: {', '.join('--' + m.replace('_', '-') for m in missing)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("Butterfly residual learner (Tier 3 MQ4G256)")
    print("=" * 80)
    print(f"  model:           {args.model}")
    print(f"  imatrix:         {args.imatrix}")
    print(f"  corpus:          {args.corpus}")
    print(f"  output dir:      {args.output_dir}")
    print(f"  target-pattern:  {args.target_pattern or '(all AWQ-F1 targets)'}")
    print(f"  n-sequences:     {args.n_sequences}")
    print(f"  ctx-len:         {args.ctx_len}")
    print(f"  alpha:           {args.alpha}")
    print(f"  optimizer:       SGD lr={args.lr} momentum={args.momentum} wd={args.weight_decay}")
    print(f"  schedule:        cosine_floor={args.cosine_floor}")
    print(f"  n_epochs:        {args.n_epochs}  (=> {args.n_epochs * args.n_sequences} steps total)")
    print(f"  grad-clip:       {args.grad_clip}")
    print(f"  l2-theta:        {args.l2_theta}")
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
    print(f"      loaded oracle in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"      VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n[2/7] Loading fresh BF16 student model ...")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    print(f"      loaded student in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"      VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n[3/7] Reading imatrix + initializing AWQ channel_scales ...")
    t0 = time.time()
    in_sum2_by_name = load_imatrix_in_sum2(args.imatrix)
    print(f"      imatrix tensors: {len(in_sum2_by_name)}")
    stored_keys = _safetensors_keys(args.model)
    print(f"      safetensors keys: {len(stored_keys)}")
    wrapped = wrap_awq_targets(
        student, in_sum2_by_name, args.alpha, stored_keys, args.target_pattern,
    )
    print(f"      wrap done in {time.time() - t0:.1f}s")
    if not wrapped:
        raise SystemExit("no AWQ targets wrapped — bailing")

    target_names = set(wrapped.keys())
    oracle_capture = OutputCapture()
    student_capture = OutputCapture()
    oracle_capture.attach(oracle, target_names)
    student_capture.attach(student, target_names)
    print(f"      capture hooks: oracle={len(oracle_capture.handles)}, "
          f"student={len(student_capture.handles)}")

    print("\n[4/7] Tokenizing calibration corpus ...")
    t0 = time.time()
    seqs = load_calibration(args.corpus, args.n_sequences, args.ctx_len, tokenizer)
    print(f"      {len(seqs)} sequences × {args.ctx_len} ctx "
          f"= {len(seqs) * args.ctx_len} tokens ({time.time() - t0:.1f}s)")

    device = next(student.parameters()).device

    initial_theta_norms = {
        n: float(w.theta_residual.detach().norm().cpu()) for n, w in wrapped.items()
    }

    smoke_kld_baseline: Optional[float] = None
    smoke_kld_trained: Optional[float] = None
    if args.smoke_eval:
        n_eval = max(1, args.smoke_eval_seqs)
        print(f"\n[pre-train] Smoke eval baseline (theta=0): {n_eval} seqs × {args.ctx_len} ctx ...")
        smoke_kld_baseline = compute_logit_kld(
            oracle, student, seqs[:n_eval], device,
        )
        print(f"      baseline KLD (theta=0):  {smoke_kld_baseline:.6f}")

    print("\n[5/7] Training butterfly residuals ...")
    train_start = time.time()
    trace = run_butterfly_training(
        wrapped=wrapped,
        oracle=oracle,
        student=student,
        oracle_capture=oracle_capture,
        student_capture=student_capture,
        seqs=seqs,
        target_names=target_names,
        device=device,
        n_epochs=args.n_epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        cosine_floor=args.cosine_floor,
        grad_clip=args.grad_clip,
        l2_theta=args.l2_theta,
        log_interval=args.log_interval,
    )
    train_elapsed = time.time() - train_start
    print(f"\n      total train time: {train_elapsed:.1f}s")

    final_theta_norms = {
        n: float(w.theta_residual.detach().norm().cpu()) for n, w in wrapped.items()
    }
    print("\n      theta-norm summary (first 8 tensors):")
    for n in list(wrapped.keys())[:8]:
        init_n = initial_theta_norms[n]
        final_n = final_theta_norms[n]
        # The plan warns: ||theta|| > π suggests rotations exceed half-turn.
        warn = " (warn: > π)" if final_n > math.pi else ""
        print(f"        {n}:   {init_n:.3e} → {final_n:.3e}{warn}")

    oracle_capture.detach()
    student_capture.detach()

    if args.smoke_eval:
        n_eval = max(1, args.smoke_eval_seqs)
        print(f"\n[post-train] Smoke eval (theta=trained): {n_eval} seqs × {args.ctx_len} ctx ...")
        smoke_kld_trained = compute_logit_kld(
            oracle, student, seqs[:n_eval], device,
        )
        print(f"      trained KLD:  {smoke_kld_trained:.6f}")
        print(f"      baseline KLD: {smoke_kld_baseline:.6f}")
        delta = smoke_kld_trained - smoke_kld_baseline
        rel = delta / max(smoke_kld_baseline, 1e-12) * 100.0
        print(f"      Δ:            {delta:+.6f}  ({rel:+.2f}%)")
        passed = smoke_kld_trained < smoke_kld_baseline
        print(f"      {'[PASS]' if passed else '[FAIL]'} smoke gate "
              f"({'trained < baseline' if passed else 'trained ≥ baseline — STOP'})")

    print("\n[6/7] Exporting butterfly residuals + AWQ sidecar ...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = args.output_dir / "butterfly_residuals.npz"
    npz_dict: dict[str, np.ndarray] = {}
    for mod_name, wrapper in sorted(wrapped.items()):
        npz_dict[wrapper.canonical_name] = wrapper.theta_residual.detach().cpu().numpy().astype(
            np.float32,
        )
    np.savez(npz_path, **npz_dict)
    print(f"      wrote {len(npz_dict)} residuals → {npz_path} "
          f"({npz_path.stat().st_size / 1e6:.2f} MB)")

    hfbf_path = args.output_dir / "butterfly_residuals.hfbf"
    bfly_entries: list[tuple[str, int, np.ndarray]] = []
    for mod_name, wrapper in sorted(wrapped.items()):
        bfly_entries.append((
            wrapper.canonical_name,
            0,
            wrapper.theta_residual.detach().cpu().numpy().astype(np.float32),
        ))
    write_hfbf(hfbf_path, bfly_entries)
    print(f"      wrote {len(bfly_entries)} HFBF entries → {hfbf_path} "
          f"({hfbf_path.stat().st_size / 1e3:.1f} KB)")

    hfsc_path = args.output_dir / "paper_awq_scales.hfsc"
    awq_entries: list[tuple[str, int, np.ndarray]] = []
    for mod_name, wrapper in sorted(wrapped.items()):
        scales = wrapper.channel_scales.detach().to(torch.float32).cpu().numpy().astype(np.float32)
        awq_entries.append((wrapper.canonical_name, 0, scales))
    write_hfsc(hfsc_path, awq_entries)
    print(f"      wrote {len(awq_entries)} HFSC entries → {hfsc_path} "
          f"({hfsc_path.stat().st_size / 1e6:.2f} MB)")

    print("\n[7/7] Writing meta.json ...")
    meta = {
        "model": args.model,
        "imatrix": str(args.imatrix),
        "corpus": args.corpus,
        "target_pattern": args.target_pattern,
        "n_sequences": args.n_sequences,
        "ctx_len": args.ctx_len,
        "alpha": args.alpha,
        "optimizer": {
            "kind": "SGD",
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "cosine_floor": args.cosine_floor,
            "grad_clip": args.grad_clip,
            "l2_theta": args.l2_theta,
        },
        "n_epochs": args.n_epochs,
        "n_wrapped": len(wrapped),
        "loss_trace": trace,
        "train_elapsed_s": train_elapsed,
        "initial_theta_norms": initial_theta_norms,
        "final_theta_norms": final_theta_norms,
        "dtype": args.dtype,
        "seed": args.seed,
        "smoke_eval": {
            "enabled": args.smoke_eval,
            "n_eval_seqs": args.smoke_eval_seqs if args.smoke_eval else None,
            "kld_baseline_theta_0": smoke_kld_baseline,
            "kld_trained": smoke_kld_trained,
        },
    }
    meta_path = args.output_dir / "butterfly_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"      meta → {meta_path}")

    print("\n=== Done ===")
    print(f"Next step: hipfire-quantize --input {args.model} \\")
    print(f"            --awq --awq-scales {hfsc_path} --awq-formula paper \\")
    print(f"            --awq-alpha {args.alpha} --awq-scope f1 \\")
    print(f"            --butterfly-residual {hfbf_path}    # (Phase 8 wiring)")


if __name__ == "__main__":
    main()
