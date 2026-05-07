#!/usr/bin/env python3
"""dump_dequant_reference.py — produce a PyTorch reference activation dump
where the model uses the SAME dequantized weights that hipfire's runtime
GEMV consumes.

Why this exists
───────────────
The original reference (dump_reference_activations.py) runs gemma-4-31B in
bf16 with full-precision weights. When verify_against_torch.rs Phase 2
compares hipfire's MQ4-class GEMV output against that reference, any delta
mixes (a) quantization-noise-from-MG4 with (b) actual kernel error. On
gemma-4-31b that means projection NRMSE settles at 5-12% even when the
kernels are bit-perfect, which is uninformative for kernel correctness
debugging.

This script:
  1. Loads the gemma-4-31B HF bf16 model (same device_map=auto path).
  2. Opens an .mq4 file and reads each quantized weight tensor's BYTES.
  3. Dequantizes them in NumPy via the production round-trip (FWHT inverse
     with seeds 42 / 1042) — same dequant the hipfire GEMV kernel implements.
  4. Casts to bf16 and copy_-replaces the corresponding model parameter.
  5. Registers the SAME hooks as dump_reference_activations.py.
  6. Forwards the same prompt; dumps activations to <out-dir>.

After this dump, verify_against_torch.rs Phase 2 with VERIFY_MODEL=...mq4
compares hipfire's GEMV output against PyTorch's GEMV output WHERE BOTH USE
THE SAME WEIGHTS. Residual NRMSE is now pure kernel error (should drop to
1e-3 to 1e-4 if kernels are correct).

Tensor format support
─────────────────────
  qt=1   (F16)    → cast to f32, no calibration
  qt=3   (Q8F16)  → block (34B/32elems): f16 scale + 32 int8 values
  qt=13  (MQ4G256) | qt=19 (MG4G256) → 136 B/group: scale + min + 128 nibbles,
                    apply inv_fwht_256 with production seeds.
  qt=14  (MQ8G256) → 258 B/group: scale + 256 int8 + sign byte; not yet supported (panic).
  qt=4,6-12,17,18 → not yet supported (will panic).

Usage
─────
  # Run on hiptrx (where /home/kaden/.hipfire/models/gemma-4-31b.mq4 lives)
  python3 scripts/arch-intake/dump_dequant_reference.py \\
      --mq4 ~/.hipfire/models/gemma-4-31b.mq4 \\
      --hf-model google/gemma-4-31B \\
      --prompt "The capital of France is" \\
      --out-dir /tmp/gemma4-ref-dequant \\
      --max-seq 32

Memory: works in CPU memory tensor-by-tensor (worst-case ~350 MB FP32 per
weight matrix). The HF bf16 model occupies ~62 GB across the 4 R9700s on
hiptrx via device_map=auto.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

# Path to scripts/quant-survey so we can pull in the production FWHT/dequant
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quant-survey"))

import numpy as np
from quant_ops import (
    GROUP_SIZE,
    BLOCK_BYTES,
    PRODUCTION_SIGNS1_SEED,
    PRODUCTION_SIGNS2_SEED,
    gen_fwht_signs,
    inv_fwht_256_batch,
)


# ─── HFQ file reader ────────────────────────────────────────────────────────


class HfqReader:
    """Pure-Python HFQ reader. Mirrors crates/hipfire-runtime/src/hfq.rs::open."""

    HEADER_SIZE = 32

    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "rb")
        # Header: "HFQM" | u32 version | u32 arch_id | u32 n_tensors |
        #         u64 metadata_offset | u64 data_offset
        head = self.fh.read(self.HEADER_SIZE)
        assert head[:4] == b"HFQM", f"not an HFQ file: {path}"
        self.version, self.arch_id, self.n_tensors = struct.unpack("<III", head[4:16])
        self.metadata_offset, self.data_offset = struct.unpack("<QQ", head[16:32])

        # Metadata: JSON blob from offset metadata_offset; ends at start of
        # tensor index. Index format: u32 n + per-tensor records.
        # Scan JSON braces to find end.
        self.fh.seek(self.metadata_offset)
        json_blob = self.fh.read(self.data_offset - self.metadata_offset)
        json_end = self._find_json_end(json_blob)
        self.metadata = json.loads(json_blob[:json_end].decode("utf-8"))

        # Parse tensor index
        self.fh.seek(self.metadata_offset + json_end)
        idx_n = struct.unpack("<I", self.fh.read(4))[0]
        assert idx_n == self.n_tensors, (idx_n, self.n_tensors)

        self.tensors: dict[str, dict[str, Any]] = {}
        cumulative = self.data_offset
        for _ in range(self.n_tensors):
            name_len = struct.unpack("<H", self.fh.read(2))[0]
            name = self.fh.read(name_len).decode("utf-8")
            quant_type = self.fh.read(1)[0]
            n_dims = self.fh.read(1)[0]
            shape = list(struct.unpack(f"<{n_dims}I", self.fh.read(n_dims * 4)))
            group_size = struct.unpack("<I", self.fh.read(4))[0]
            data_size = struct.unpack("<Q", self.fh.read(8))[0]
            self.tensors[name] = {
                "quant_type": quant_type,
                "shape": shape,
                "group_size": group_size,
                "data_offset": cumulative,
                "data_size": data_size,
            }
            cumulative += data_size

    @staticmethod
    def _find_json_end(blob: bytes) -> int:
        """Match the Rust scan (handles braces, escaping, strings)."""
        depth = 0
        in_str = False
        escape = False
        for i, b in enumerate(blob):
            if escape:
                escape = False
                continue
            if b == ord('\\') and in_str:
                escape = True
                continue
            if b == ord('"'):
                in_str = not in_str
                continue
            if not in_str:
                if b == ord('{'):
                    depth += 1
                elif b == ord('}'):
                    depth -= 1
                    if depth == 0:
                        return i + 1
        raise ValueError("unterminated JSON metadata block")

    def read_tensor_bytes(self, name: str) -> bytes:
        info = self.tensors[name]
        self.fh.seek(info["data_offset"])
        return self.fh.read(info["data_size"])

    def has(self, name: str) -> bool:
        return name in self.tensors

    def info(self, name: str) -> dict[str, Any]:
        return self.tensors[name]


# ─── Dequant primitives ─────────────────────────────────────────────────────


def dequantize_mq4g256_block_batch(
    blob: bytes, n_groups: int, signs1: np.ndarray, signs2: np.ndarray
) -> np.ndarray:
    """Dequantize n_groups consecutive 136-byte MQ4/MG4 blocks at once.

    Returns float32 array of shape (n_groups * 256,).
    """
    expected = n_groups * BLOCK_BYTES
    assert len(blob) == expected, (len(blob), expected)

    # View: (n_groups, 136). Slice scale (4 B), min (4 B), nibbles (128 B).
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(n_groups, BLOCK_BYTES)
    scale = np.frombuffer(arr[:, 0:4].tobytes(), dtype=np.float32)
    min_val = np.frombuffer(arr[:, 4:8].tobytes(), dtype=np.float32)
    nibbles_packed = arr[:, 8:8 + 128]  # (n_groups, 128) bytes

    # Unpack low/high nibbles per byte → (n_groups, 256)
    lo = (nibbles_packed & 0x0F).astype(np.float32)
    hi = ((nibbles_packed >> 4) & 0x0F).astype(np.float32)
    # Interleave so even index = lo, odd index = hi.
    rotated = np.empty((n_groups, 256), dtype=np.float32)
    rotated[:, 0::2] = lo
    rotated[:, 1::2] = hi

    # Apply scale + min: rotated_value = scale * q + min
    rotated *= scale[:, None]
    rotated += min_val[:, None]

    # Inverse FWHT to recover original-space values
    recovered = inv_fwht_256_batch(rotated, signs1, signs2)
    return recovered.reshape(-1)


def dequantize_q8_0_block(blob: bytes, n_elements: int) -> np.ndarray:
    """Q8_0: 34 bytes per 32 elements (f16 scale + 32 int8 values)."""
    assert n_elements % 32 == 0, n_elements
    n_blocks = n_elements // 32
    expected = n_blocks * 34
    assert len(blob) == expected, (len(blob), expected)

    arr = np.frombuffer(blob, dtype=np.uint8).reshape(n_blocks, 34)
    scales_f16 = np.frombuffer(arr[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
    values = np.frombuffer(arr[:, 2:34].tobytes(), dtype=np.int8).reshape(n_blocks, 32).astype(np.float32)
    out = values * scales_f16[:, None]
    return out.reshape(-1)


def dequantize_tensor(
    blob: bytes,
    quant_type: int,
    shape: list[int],
    signs1: np.ndarray,
    signs2: np.ndarray,
) -> np.ndarray:
    """Dispatch dequant by quant_type. Returns flat float32 array sized prod(shape)."""
    n_elements = int(np.prod(shape)) if shape else 1
    if quant_type == 1:  # F16
        return np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    if quant_type == 0:  # F32 (rare in HFQ but support it)
        return np.frombuffer(blob, dtype=np.float32).copy()
    if quant_type == 3:  # Q8F16 / Q8_0
        return dequantize_q8_0_block(blob, n_elements)
    if quant_type in (13, 19):  # MQ4G256 / MG4G256 (same byte layout)
        # 256 elements per group, 136 bytes per block
        n_padded = ((n_elements + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
        n_groups = n_padded // GROUP_SIZE
        recovered = dequantize_mq4g256_block_batch(blob, n_groups, signs1, signs2)
        return recovered[:n_elements]
    raise NotImplementedError(
        f"dequantize: quant_type={quant_type} not implemented "
        f"(needed for tensor with shape {shape})"
    )


# ─── HF model patcher ───────────────────────────────────────────────────────


# Reuse the original dump script's hook pattern + step name patterns
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_reference_activations import (  # noqa: E402
    STEP_PATTERNS,
    classify_step,
    _safe_save,
)


def patch_model_with_dequant(model, hfq: HfqReader, signs1, signs2) -> dict[str, str]:
    """Iterate all named_parameters; for each parameter that has a matching
    tensor in the HFQ file, dequantize the HFQ bytes and copy into the param.

    Returns a dict {hf_param_name: status_string} for reporting. Only weight
    matrices are matched; biases / position embeddings / layer scalars are
    handled by name suffix or skipped.
    """
    import torch
    status: dict[str, str] = {}
    n_patched = 0
    n_skipped = 0
    n_missing = 0

    for hf_name, param in model.named_parameters():
        # Map HF parameter name → HFQ tensor name (typically identical).
        # Some weights need name remapping (rare; skip if not found).
        hfq_name = hf_name
        if not hfq.has(hfq_name):
            # Try without "weight" suffix or other small remaps
            n_missing += 1
            status[hf_name] = f"SKIP (no HFQ tensor named {hfq_name})"
            continue

        info = hfq.info(hfq_name)
        try:
            blob = hfq.read_tensor_bytes(hfq_name)
            f32 = dequantize_tensor(blob, info["quant_type"], info["shape"], signs1, signs2)
        except NotImplementedError as e:
            n_skipped += 1
            status[hf_name] = f"SKIP ({e.__class__.__name__}: qt={info['quant_type']})"
            continue

        # Reshape; some HF params might be transposed vs HFQ. Check size first.
        expected_n = int(param.numel())
        if f32.size != expected_n:
            n_skipped += 1
            status[hf_name] = (
                f"SKIP (numel mismatch: HFQ={f32.size} HF={expected_n} "
                f"HFQ shape={info['shape']} HF shape={list(param.shape)})"
            )
            continue

        # Reshape to HF shape, cast to bf16 (or matching), copy.
        t = torch.from_numpy(f32.reshape(list(param.shape))).to(
            dtype=param.dtype, device=param.device
        )
        with torch.no_grad():
            param.data.copy_(t)
        n_patched += 1
        status[hf_name] = f"OK qt={info['quant_type']} bytes={info['data_size']:,}"
        del t, f32

    print(
        f"# patched: {n_patched} params, skipped: {n_skipped}, missing: {n_missing}",
        file=sys.stderr,
    )
    return status


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--mq4", required=True, type=Path,
                    help="Path to the .mq4 file (e.g. ~/.hipfire/models/gemma-4-31b.mq4)")
    ap.add_argument("--hf-model", required=True,
                    help="HF model ID (e.g. google/gemma-4-31B)")
    ap.add_argument("--prompt", default="The capital of France is",
                    help="prompt to forward (must match original dump for verifier)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-seq", type=int, default=32)
    ap.add_argument("--hf-cache", default=str(Path.home() / ".cache/huggingface/hub"))
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--report-status", type=Path, default=None,
                    help="Optional: write the patch status dict to this JSON file")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open HFQ, generate FWHT signs
    print(f"# opening {args.mq4}", file=sys.stderr)
    hfq = HfqReader(args.mq4)
    print(
        f"# HFQ: arch_id={hfq.arch_id}, n_tensors={hfq.n_tensors}, "
        f"metadata_keys={list(hfq.metadata.keys())[:5]}",
        file=sys.stderr,
    )
    signs1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    signs2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)

    # Load HF model
    print(f"# loading HF model {args.hf_model} ({args.dtype}, device_map=auto)", file=sys.stderr)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model, cache_dir=args.hf_cache, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, cache_dir=args.hf_cache,
        torch_dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model.eval()
    embed_device = next(model.parameters()).device
    print(f"# loaded in {time.time()-t0:.1f}s; embed on {embed_device}", file=sys.stderr)

    # Patch weights from HFQ
    print(f"# dequantizing + patching weights from {args.mq4}...", file=sys.stderr)
    t0 = time.time()
    status = patch_model_with_dequant(model, hfq, signs1, signs2)
    print(f"# patch complete in {time.time()-t0:.1f}s", file=sys.stderr)
    if args.report_status:
        args.report_status.write_text(json.dumps(status, indent=2))

    # Hooks: copy from dump_reference_activations.py
    captures: list[dict[str, Any]] = []
    handles = []
    n_hooks = 0
    for name, module in model.named_modules():
        cls = classify_step(name)
        if cls is None:
            continue
        layer_idx, step = cls

        def make_pre_hook(layer_idx, step, name):
            def hook(mod, inputs):
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return
                path = out_dir / f"layer_{layer_idx:02d}" / f"{step}.input.safetensors"
                _safe_save(x, path)
                captures.append({
                    "layer_idx": layer_idx, "step": step, "side": "input",
                    "module_name": name, "shape": list(x.shape),
                    "dtype": str(x.dtype),
                    "path": str(path.relative_to(out_dir)),
                })
            return hook

        def make_post_hook(layer_idx, step, name):
            def hook(mod, inputs, output):
                if isinstance(output, tuple):
                    output = output[0]
                if not isinstance(output, torch.Tensor):
                    return
                path = out_dir / f"layer_{layer_idx:02d}" / f"{step}.output.safetensors"
                _safe_save(output, path)
                captures.append({
                    "layer_idx": layer_idx, "step": step, "side": "output",
                    "module_name": name, "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "path": str(path.relative_to(out_dir)),
                })
            return hook

        handles.append(module.register_forward_pre_hook(make_pre_hook(layer_idx, step, name)))
        handles.append(module.register_forward_hook(make_post_hook(layer_idx, step, name)))
        n_hooks += 2
    print(f"# registered {n_hooks} hooks", file=sys.stderr)

    # Tokenize + forward
    tokens = tokenizer(args.prompt, return_tensors="pt", truncation=True,
                       max_length=args.max_seq)
    input_ids = tokens.input_ids.to(embed_device)
    _safe_save(input_ids, out_dir / "input_ids.safetensors")
    print(f"# prompt: {args.prompt!r} → tokens: {input_ids.tolist()}", file=sys.stderr)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    print(f"# forward in {time.time()-t0:.1f}s; logits shape {out.logits.shape}",
          file=sys.stderr)

    _safe_save(out.logits, out_dir / "final_logits.safetensors")
    for h in handles:
        h.remove()

    manifest = {
        "model": args.hf_model,
        "weights_source": str(args.mq4),
        "weights_arch_id": hfq.arch_id,
        "prompt": args.prompt,
        "input_ids": input_ids.tolist(),
        "n_tokens": int(input_ids.shape[1]),
        "dtype": args.dtype,
        "config_summary": {
            "architectures": getattr(model.config, "architectures", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        },
        "captures": captures,
        "n_captures": len(captures),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"# wrote {out_dir / 'manifest.json'} ({len(captures)} captures)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
