#!/usr/bin/env python3
"""grad_scale_learn.py - gradient-based AWQ scale learning prototype.

The "5-epoch" ParoQuant mechanism reframed for hipfire's MQ4G256 quant
format:

  forall AWQ-target nn.Linear:
    s in R^K   (learnable, init = closed-form paper-formula scales)
    y(x) = (x / s) @ STE_quantize_mq4g256( W * broadcast(s) ).T

  Loss = BRECQ-style block-wise reconstruction (MSE of per-Linear outputs
  vs the original BF16 model's per-Linear outputs on the same tokens).
  Optimize s via Adam over 5 epochs.

The straight-through estimator gives us a non-zero gradient through the
quantize op - the forward emits the actual rounded weight, the backward
passes through as if no quantization happened.

When training completes we serialize the learned `s` per tensor into an
HFSC sidecar (`crates/hipfire-quantize/src/awq_scales_io.rs` v1 format)
that hipfire-quantize will consume via its existing `--awq-scales` flag.

Usage (mi300 droplet):
    python3 scripts/grad_scale_learn.py \\
        --model /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc.../ \\
        --imatrix /workspace/qwen3.5-0.8b.mix.ctx4096.imatrix.gguf \\
        --corpus /workspace/calibration-mix-v1.txt \\
        --output /workspace/v3/0.8b/gradscale/awq_scales.hfsc \\
        --n-sequences 128 --ctx-len 4096 \\
        --alpha 0.55 --lr 1e-3 --epochs 5

Result (2026-05-22, Qwen3.5-0.8B, 128 seqs x 4096 ctx, 5 epochs, Adam lr=1e-3,
calibration-mix-v1, mi300 droplet, total wall clock 25.0 min training + ~33 min eval):
  loss trace (per-Linear MSE block-wise reconstruction):
    epoch 1: 2.888e-2
    epoch 2: 2.749e-2  (-4.8%)
    epoch 3: 2.724e-2  (-0.9%)
    epoch 4: 2.725e-2  (+0.0%)
    epoch 5: 2.703e-2  (-0.8%)
  Final KLD (gradient-learned scales + GPTQ): 0.159296
  Baseline KLD (closed-form paper-formula scales + GPTQ): 0.132733
  Delta: +20.01% — REGRESSION.

Interpretation: end-to-end BRECQ-style activation MSE went DOWN (-6.4% over 5
epochs) yet the downstream KL divergence vs the BF16 oracle went UP. The
gradient learning IS optimizing what we asked it to (per-Linear output
reconstruction), but that proxy objective is mis-aligned with the GPTQ
post-processing step that runs after the AWQ scales are emitted. The damped
AWQ scales the closed-form path produces are a 1st-order minimizer of the
GPTQ objective; our learned scales minimize a different objective (no-GPTQ
reconstruction), so when GPTQ then re-quantizes on top of our scales it
introduces extra error.

The "ParoQuant 5-epoch" mechanism does work — but in a different geometry
(MR-GPTQ, joint scale+rotation, and no post-hoc GPTQ correction). Replicating
it on hipfire's MQ4 stack would need either (a) joint scale+GPTQ
differentiation, or (b) training with the GPTQ step IN the loss (currently
not differentiable through Cholesky), or (c) abandoning GPTQ post-processing
when learning scales. NEGATIVE RESULT for the simple BRECQ+STE recipe on
this particular pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MQ4QuantizedLinear(nn.Module):
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

        self.register_buffer("weight", weight.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        self.s = nn.Parameter(init_scales.detach().clone().to(torch.float32))

        self.register_buffer("_fwht_signs1", FWHT_SIGNS1.clone().view(1, 256))
        self.register_buffer("_fwht_signs2", FWHT_SIGNS2.clone().view(1, 256))

        self.canonical_name = canonical_name
        self.k_in = k
        self.m_out = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.s
        s_safe = torch.clamp(s, min=1e-6).to(torch.float32)

        w_dtype = self.weight.dtype
        w_f32 = self.weight.to(torch.float32) * s_safe.unsqueeze(0)
        w_q = quantize_mq4g256_ste(w_f32, self._fwht_signs1, self._fwht_signs2)
        w_q = w_q.to(w_dtype)

        x_scaled = x / s_safe.to(x.dtype).reshape(*[1] * (x.dim() - 1), -1)
        return F.linear(x_scaled, w_q, self.bias)


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


def compute_awq_scales_paper(in_sum2: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(in_sum2, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("empty imatrix vector")
    half_alpha = float(alpha) * 0.5
    log_s = half_alpha * np.log(np.maximum(values, 1.0e-12))
    log_s -= np.mean(log_s)
    return np.exp(log_s).astype(np.float32)


HFSC_MAGIC = b"HFSC"
HFSC_VERSION = 1


def write_hfsc(
    path: Path,
    entries: list[tuple[str, int, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(HFSC_MAGIC)
        f.write(struct.pack("<I", HFSC_VERSION))
        f.write(struct.pack("<Q", len(entries)))
        f.write(struct.pack("<Q", 0))
        for name, expert_idx, scales in entries:
            name_bytes = name.encode("utf-8")
            arr = np.asarray(scales, dtype=np.float32).reshape(-1)
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", int(expert_idx)))
            f.write(struct.pack("<I", arr.size))
            f.write(arr.tobytes(order="C"))


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
    *,
    only_module: Optional[str] = None,
) -> dict[str, MQ4QuantizedLinear]:
    wrapped: dict[str, MQ4QuantizedLinear] = {}
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
        if only_module is not None and mod_name != only_module:
            continue
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
        wrapper = MQ4QuantizedLinear(weight, bias, init_s, canonical_name=canonical)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Gradient-based AWQ scale learning prototype.")
    ap.add_argument("--model", required=True, help="HF BF16 model dir path.")
    ap.add_argument("--imatrix", required=True, type=Path, help="Path to .imatrix.gguf.")
    ap.add_argument("--corpus", required=True, help="Path to calibration text file.")
    ap.add_argument("--output", required=True, type=Path, help="HFSC sidecar output path.")
    ap.add_argument("--n-sequences", type=int, default=128)
    ap.add_argument("--ctx-len", type=int, default=4096)
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="Closed-form paper-formula alpha for scale initialization.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--smoke-one-module", default=None,
                    help="If set, only wrap the named module (Phase-2 smoke).")
    ap.add_argument("--log-interval", type=int, default=8)
    ap.add_argument("--save-trace", type=Path, default=None,
                    help="Optional JSON file to dump loss trace.")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("gradient-based AWQ scale learning")
    print("=" * 80)
    print(f"  model:         {args.model}")
    print(f"  imatrix:       {args.imatrix}")
    print(f"  corpus:        {args.corpus}")
    print(f"  output HFSC:   {args.output}")
    print(f"  n-sequences:   {args.n_sequences}")
    print(f"  ctx-len:       {args.ctx_len}")
    print(f"  alpha (init):  {args.alpha}")
    print(f"  Adam lr:       {args.lr}")
    print(f"  epochs:        {args.epochs}")
    print(f"  smoke-only:    {args.smoke_one_module}")
    print()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print("[1/6] Loading BF16 oracle model + tokenizer ...")
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

    print("\n[2/6] Loading fresh BF16 student model + wrapping AWQ-F1 targets ...")
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

    print("\n[3/6] Reading imatrix in_sum2 + initializing scales (paper formula) ...")
    t0 = time.time()
    in_sum2_by_name = load_imatrix_in_sum2(args.imatrix)
    print(f"      imatrix tensors: {len(in_sum2_by_name)}")
    stored_keys = _safetensors_keys(args.model)
    print(f"      safetensors keys: {len(stored_keys)}")
    wrapped = wrap_awq_targets(
        student, in_sum2_by_name, args.alpha, stored_keys,
        only_module=args.smoke_one_module,
    )
    print(f"      wrap done in {time.time() - t0:.1f}s. "
          f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    if not wrapped:
        raise SystemExit("no AWQ targets wrapped - bailing")

    params = [w.s for w in wrapped.values()]
    n_scale_elements = sum(p.numel() for p in params)
    print(f"      learnable scale elements: {n_scale_elements} across "
          f"{len(params)} tensors")
    optimizer = torch.optim.Adam(params, lr=args.lr)

    target_names = set(wrapped.keys())
    oracle_capture = OutputCapture()
    student_capture = OutputCapture()
    oracle_capture.attach(oracle, target_names)
    student_capture.attach(student, target_names)
    print(f"      hooks attached: oracle={len(oracle_capture.handles)}, "
          f"student={len(student_capture.handles)}")

    print("\n[4/6] Tokenizing calibration corpus ...")
    t0 = time.time()
    seqs = load_calibration(args.corpus, args.n_sequences, args.ctx_len, tokenizer)
    print(f"      {len(seqs)} sequences x {args.ctx_len} ctx "
          f"= {len(seqs) * args.ctx_len} tokens in {time.time() - t0:.1f}s")

    print(f"\n[5/6] Training for {args.epochs} epochs ...")
    device = next(student.parameters()).device
    loss_trace: list[dict] = []
    step_global = 0
    train_start = time.time()

    # Capture initial scale norm for smoke verification
    init_scale_norms = {n: float(w.s.detach().norm().cpu()) for n, w in wrapped.items()}

    for epoch in range(args.epochs):
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
                y_q = student_capture.outputs[n]
                y_o = y_oracle[n]
                diff = (y_q.to(torch.float32) - y_o.to(torch.float32))
                loss = loss + diff.pow(2).mean()
                n_terms += 1
            loss = loss / max(n_terms, 1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p in params:
                    p.clamp_(min=1e-6)

            epoch_loss += float(loss.detach())
            epoch_count += 1
            step_global += 1

            if (seq_idx + 1) % args.log_interval == 0:
                running = epoch_loss / max(epoch_count, 1)
                elapsed = time.time() - epoch_start
                print(f"      epoch {epoch+1}/{args.epochs} seq {seq_idx+1}/{len(seqs)} "
                      f"loss={float(loss.detach()):.6e}  running={running:.6e}  "
                      f"({elapsed:.1f}s)")

            del y_oracle

        mean_loss = epoch_loss / max(epoch_count, 1)
        loss_trace.append({
            "epoch": epoch,
            "mean_loss": mean_loss,
            "elapsed_s": time.time() - epoch_start,
        })
        print(f"      [epoch {epoch+1}] mean_loss={mean_loss:.6e}  "
              f"epoch_time={time.time() - epoch_start:.1f}s")

    train_elapsed = time.time() - train_start
    print(f"      total train time: {train_elapsed:.1f}s")

    # Smoke verification: did scales move?
    scale_movements: list[tuple[str, float, float]] = []
    for n, w in wrapped.items():
        init_norm = init_scale_norms[n]
        final_norm = float(w.s.detach().norm().cpu())
        rel_change = abs(final_norm - init_norm) / max(init_norm, 1e-9)
        scale_movements.append((n, init_norm, final_norm))
    moved = [m for m in scale_movements if abs(m[1] - m[2]) / max(m[1], 1e-9) > 1e-4]
    print(f"      scales moved >1e-4 rel: {len(moved)}/{len(wrapped)}")
    if scale_movements:
        sample = scale_movements[:3]
        for n, i, f in sample:
            print(f"        {n}: norm {i:.4f} -> {f:.4f} "
                  f"(rel delta {abs(i-f)/max(i,1e-9):.2e})")

    oracle_capture.detach()
    student_capture.detach()

    print(f"\n[6/6] Exporting learned scales to {args.output} ...")
    entries: list[tuple[str, int, np.ndarray]] = []
    for mod_name, wrapper in sorted(wrapped.items()):
        scales = wrapper.s.detach().to(torch.float32).cpu().numpy().astype(np.float32)
        entries.append((wrapper.canonical_name, 0, scales))
    write_hfsc(args.output, entries)
    print(f"      wrote {len(entries)} HFSC entries -> {args.output} "
          f"({args.output.stat().st_size / 1e6:.2f} MB)")

    if args.save_trace is not None:
        args.save_trace.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "imatrix": str(args.imatrix),
            "corpus": args.corpus,
            "n_sequences": args.n_sequences,
            "ctx_len": args.ctx_len,
            "alpha": args.alpha,
            "lr": args.lr,
            "epochs": args.epochs,
            "n_wrapped": len(wrapped),
            "loss_trace": loss_trace,
            "train_elapsed_s": train_elapsed,
            "scale_movement_sample": scale_movements[:8],
        }
        with args.save_trace.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"      saved trace -> {args.save_trace}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
