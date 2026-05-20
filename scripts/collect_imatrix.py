#!/usr/bin/env python3
"""collect_imatrix.py — PyTorch-based imatrix oracle for Tier 1 BF16 parity.

This is the imatrix companion to `scripts/collect_hessian.py`: it runs a
calibration-time forward pass through HF transformers (the SAME stack the
model was trained with), accumulates `Σ x² per input channel` on every
`nn.Linear`, and writes a GGUF file byte-near-compatible with
`llama-imatrix`'s output.

WHY this exists:

Comparing hipfire Tier 1 (our native BF16 forward) directly to `llama-imatrix`
hits a structural floor at ~0.91 NRMSE because llama.cpp and hipfire implement
the same architecture differently (kernel fusion, RoPE phasing, eps placement,
FullAttn vs flash-attn batching, etc.). To break that floor we need a
DIFFERENT reference: HF transformers in PyTorch. Qwen3.5 was TRAINED in this
stack, so it's the canonical impl.

KV format: stock HF transformers `eager` attention uses fp32/bf16 KV (matches
`torch_dtype`); no Q8_0 quantization. This is the cleanest BF16 oracle. A
future tighter-tier hipfire comparison may swap Tier 1 to F16/BF16 KV to
match; for now, the Q8_0-KV residual is a known structural delta.

Output format: GGUF, llama-imatrix-compatible. Per-tensor we emit
  `{ggml_name}.in_sum2`   F32[k]   sum of squared activations per channel
  `{ggml_name}.counts`    F32[1]   total tokens contributing
Plus metadata:
  general.type           = "imatrix"
  imatrix.datasets       = [<corpus_basename>]
  imatrix.chunk_count    = <n_sequences>
  imatrix.chunk_size     = <ctx_len>
Names use llama.cpp GGUF style (`blk.{layer}.<slot>.weight`); the
HF-safetensors → GGUF translation mirrors `safetensors_to_ggml_name` in
`crates/hipfire-quantize/src/main.rs:2506` and `scripts/compare_imatrix.py`.

Counts semantics: matches `llama-imatrix` — each linear that fires on every
token of every chunk accumulates `n_sequences * ctx_len` tokens. Sparse
linears (MoE experts) would accumulate fewer; we record actual hit counts.
(For dense Qwen3.5 every Linear sees every token, so all counts are equal.)

Usage:
    .venv/bin/python3 scripts/collect_imatrix.py \\
        --hf-model <hf-snap-dir> \\
        --corpus   benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \\
        --output   /tmp/qwen3.5-0.8b.pytorch.imatrix.gguf \\
        --n-sequences 32 --ctx-len 2048 \\
        [--process-output] \\
        [--device cuda] [--dtype bfloat16]

Verification:
    # Well-formed GGUF + tensor count
    python3 -c "import gguf; r = gguf.GGUFReader('/tmp/...gguf'); \\
        print(f'tensors={len(r.tensors)}')"

    # Compare against existing llama-imatrix oracle
    python3 scripts/compare_imatrix.py \\
        --oracle benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf \\
        --target /tmp/qwen3.5-0.8b.pytorch.imatrix.gguf \\
        --max-nrmse 0.5

Hard out-of-scope: Hessian collection (see scripts/collect_hessian.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import gguf
    from gguf import GGUFWriter
except ImportError:
    sys.stderr.write("error: requires `gguf` python package (pip install gguf)\n")
    sys.exit(3)

from transformers import AutoModelForCausalLM, AutoTokenizer


# Mirror of `safetensors_to_ggml_name` in
# crates/hipfire-quantize/src/main.rs:2506 and scripts/compare_imatrix.py.
# Maps the per-layer slot of a safetensors-stored tensor name to the
# llama.cpp GGUF tensor name slot.
_PER_LAYER_MAP = {
    # MLP — present on every layer.
    "mlp.gate_proj":         "ffn_gate",
    "mlp.up_proj":           "ffn_up",
    "mlp.down_proj":         "ffn_down",
    # FullAttention layer tensors.
    "self_attn.q_proj":      "attn_q",
    "self_attn.k_proj":      "attn_k",
    "self_attn.v_proj":      "attn_v",
    "self_attn.o_proj":      "attn_output",
    # LinearAttention (Gated DeltaNet) layer tensors — SSM-style naming.
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z":   "attn_gate",
    "linear_attn.in_proj_a":   "ssm_alpha",
    "linear_attn.in_proj_b":   "ssm_beta",
    "linear_attn.out_proj":    "ssm_out",
}

_TOP_LEVEL_MAP = {
    "embed_tokens.weight":  "token_embd.weight",
    "lm_head.weight":       "output.weight",
    "norm.weight":           "output_norm.weight",
}


def safetensors_to_ggml_name(name: str) -> Optional[str]:
    """Convert HF safetensors-stored name to llama.cpp GGUF tensor name.

    Returns None for tensors that don't have an imatrix counterpart
    (norms, biases, conv1d, A_log, dt_bias, etc.).
    """
    normalized = name
    for prefix in ("model.language_model.", "model."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    if normalized in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[normalized]

    m = re.match(r"layers\.(\d+)\.(.+)\.weight$", normalized)
    if not m:
        return None
    layer_idx, slot = m.group(1), m.group(2)
    translated = _PER_LAYER_MAP.get(slot)
    if translated is None:
        return None
    return f"blk.{layer_idx}.{translated}.weight"


def _safetensors_keys(model_path: str) -> set[str]:
    """Returns the set of all tensor keys stored in the safetensors files,
    stripped of `.weight` suffix to match against module names.

    HF's `AutoModelForCausalLM` flattens multimodal submodules (e.g.
    Qwen3.5 in CausalLM mode drops the `language_model.` prefix from
    in-memory module names). The safetensors files keep the full prefix.
    """
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
    return {k.removesuffix(".weight") for k in keys if k.endswith(".weight")}


def _translate_to_stored_name(mod_name: str, stored_keys: set[str]) -> Optional[str]:
    """Find the canonical safetensors-stored name for a hooked module.

    Strategy: longest-suffix match (same logic as collect_hessian.py).
    """
    if mod_name in stored_keys:
        return mod_name
    trail = mod_name.split(".", 1)[1] if "." in mod_name else mod_name
    matches = [k for k in stored_keys if k.endswith(trail)]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


# Modules we capture activations for. Same set as collect_hessian.py — every
# linear that participates in the forward path (attention + MLP + DeltaNet +
# MoE router). `lm_head` / `output` is opt-in via --process-output to match
# llama-imatrix's CLI semantics.
IMATRIX_TARGET_SUFFIXES = (
    # Attention input projections
    "q_proj", "k_proj", "v_proj", "qkv_proj",
    # Attention output (gets imatrix even without AWQ)
    "o_proj",
    # MLP
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
    # Linear-attention (Gated DeltaNet)
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
    "out_proj",
    # MoE router
    "gate",
)


def is_imatrix_target(name: str, capture_output: bool) -> bool:
    """Returns True if this nn.Linear module should have its inputs captured."""
    last = name.rsplit(".", 1)[-1]
    if last in IMATRIX_TARGET_SUFFIXES:
        return True
    if capture_output and last == "lm_head":
        return True
    return False


class ImatrixAccumulator:
    """One per nn.Linear we hook. Accumulates Σ x[..., k]² per channel.

    We accumulate in F64 to keep numerical headroom over the ~2M+ token
    activation sum (BF16 mantissa is 8 bits; a naive F32 sum of ~1M positive
    values stops adding new contributions once the running sum exceeds
    ~16M·ulp). The final cast to F32 matches the llama-imatrix output dtype.
    """

    def __init__(self, ggml_name: str, K: int) -> None:
        self.ggml_name = ggml_name
        self.K = K
        # F64 channel sum; ~K*8 bytes per linear, negligible vs the model.
        self.in_sum2 = np.zeros(K, dtype=np.float64)
        self.n_tokens = 0

    def update(self, x_sumsq: np.ndarray, n_tokens_this_batch: int) -> None:
        """x_sumsq: shape [K], F64 — already reduced on GPU to the per-channel
        sum-of-squares contribution from this hook firing.
        """
        assert x_sumsq.shape == (self.K,), \
            f"{self.ggml_name}: shape mismatch {x_sumsq.shape} vs K={self.K}"
        self.in_sum2 += x_sumsq
        self.n_tokens += n_tokens_this_batch


def build_hook(acc: ImatrixAccumulator):
    """Returns a forward-pre hook that computes per-channel Σx² on the GPU.

    Reducing on the GPU first means we transfer K floats per hook firing
    instead of N*K (saves ~2048× bandwidth at ctx_len=2048).
    """
    def hook(module, inputs):  # noqa: ARG001
        x = inputs[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))
        # Sum on GPU in F32 then transfer K F32 values (small). Final
        # accumulation into the F64 host buffer keeps drift bounded.
        sumsq = (x.to(dtype=torch.float32) ** 2).sum(dim=0)
        sumsq_host = sumsq.detach().cpu().numpy().astype(np.float64, copy=False)
        n_tokens = int(x.size(0))
        acc.update(sumsq_host, n_tokens)
    return hook


def load_calibration_text(corpus: Path, n_sequences: int, ctx_len: int,
                           tokenizer) -> list[torch.Tensor]:
    """Returns a list of `n_sequences` token tensors of length `ctx_len`.

    Matches llama-imatrix's chunking exactly: tokenize the corpus as-is
    (no extra BOS/EOS), take the first `n_sequences * ctx_len` tokens,
    split into non-overlapping chunks.
    """
    text = corpus.read_text()
    enc = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    all_tokens = enc.input_ids[0]
    n_total = all_tokens.shape[0]
    needed = n_sequences * ctx_len
    if n_total < needed:
        raise SystemExit(
            f"corpus has only {n_total} tokens, need {needed} "
            f"({n_sequences} seqs × {ctx_len} ctx). "
            "Use a larger calibration slice or reduce --n-sequences."
        )
    return [all_tokens[i * ctx_len: (i + 1) * ctx_len] for i in range(n_sequences)]


def write_imatrix_gguf(
    out_path: Path,
    accs: dict[str, ImatrixAccumulator],
    corpus_basename: str,
    n_sequences: int,
    ctx_len: int,
) -> None:
    """Serialize accumulators to llama-imatrix-compatible GGUF.

    Per-tensor we write `.in_sum2` (F32[K]) and `.counts` (F32[1]). The
    `general.type=imatrix` metadata tag tells consumers this is an imatrix
    file rather than a model. `arch` on GGUFWriter must be set to *some*
    string; we use "imatrix" — readers (including hipfire's loader and
    `scripts/compare_imatrix.py`) key on the tensor names, not `arch`, so
    the extra `general.architecture` field is harmless.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(out_path), arch="imatrix")

    # Metadata — exact field set used by llama-imatrix.
    writer.add_string("general.type", "imatrix")
    writer.add_array("imatrix.datasets", [corpus_basename])
    writer.add_uint32("imatrix.chunk_count", int(n_sequences))
    writer.add_uint32("imatrix.chunk_size", int(ctx_len))

    # Tensors — sorted for byte-deterministic output across runs. The
    # canonical reference writes them in iteration order; we sort by
    # (layer_idx, slot) so diffs are stable.
    def _sort_key(name: str) -> tuple[int, str]:
        m = re.match(r"blk\.(\d+)\.(.+)\.weight", name)
        if m:
            return (int(m.group(1)), m.group(2))
        # Top-level (token_embd / output / output_norm) — sort after layers
        # by very large layer index so they end up at the bottom.
        return (10**9, name)

    n_skipped = 0
    for ggml_name in sorted(accs.keys(), key=_sort_key):
        acc = accs[ggml_name]
        if acc.n_tokens == 0:
            print(f"  WARN: {ggml_name} accumulated 0 tokens — skipping", file=sys.stderr)
            n_skipped += 1
            continue
        in_sum2_f32 = acc.in_sum2.astype(np.float32, copy=False)
        counts_f32 = np.array([float(acc.n_tokens)], dtype=np.float32)
        writer.add_tensor(f"{ggml_name}.in_sum2", in_sum2_f32)
        writer.add_tensor(f"{ggml_name}.counts", counts_f32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if n_skipped:
        print(f"  skipped {n_skipped} empty accumulators", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PyTorch-based imatrix oracle for Tier 1 BF16 parity."
    )
    ap.add_argument("--hf-model", required=True, type=str,
                    help="HF model dir (snapshot path) or HF hub identifier.")
    ap.add_argument("--corpus", required=True, type=Path,
                    help="Plain-text calibration corpus file.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output imatrix GGUF path.")
    ap.add_argument("--n-sequences", type=int, default=32,
                    help="Calibration chunks (default 32 for fast oracle; 128 for full).")
    ap.add_argument("--ctx-len", type=int, default=2048,
                    help="Tokens per chunk (default 2048, matches reference).")
    ap.add_argument("--process-output", action="store_true",
                    help="Also capture lm_head input activations (default off, matches "
                         "llama-imatrix default).")
    ap.add_argument("--device", default="cuda",
                    help="Device for the forward pass (default 'cuda').")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Model + KV dtype (default bfloat16, matches Tier 1).")
    ap.add_argument("--attn-impl", default="eager",
                    choices=["eager", "sdpa"],
                    help="HF attention implementation (default eager — most "
                         "faithful to the trained reference; sdpa for speed).")
    args = ap.parse_args()

    print(f"=== PyTorch imatrix oracle ===")
    print(f"  model:      {args.hf_model}")
    print(f"  corpus:     {args.corpus}")
    print(f"  output:     {args.output}")
    print(f"  sequences:  {args.n_sequences} × {args.ctx_len} ctx "
          f"= {args.n_sequences * args.ctx_len} calibration tokens")
    print(f"  device:     {args.device}")
    print(f"  dtype:      {args.dtype}")
    print(f"  attn_impl:  {args.attn_impl}")
    print(f"  --process-output: {args.process_output}")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }[args.dtype]

    print(f"\n[1/4] Loading tokenizer + model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    )
    if args.device != "cuda":
        model = model.to(args.device)
    model.eval()
    print(f"      loaded in {time.time() - t0:.1f}s")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"      VRAM in use: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f}/"
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Sanity: tokenizer parity check vs the llama-imatrix oracle's tokenizer.
    # Both should agree on the same BPE; if they disagree by more than a few
    # tokens on a short prompt, the imatrix counts will be off-by-tokenization
    # rather than off-by-numerics.
    sanity_text = "hello world\n"
    hf_ids = tokenizer(sanity_text, add_special_tokens=False).input_ids
    print(f"      tokenizer sanity: 'hello world\\n' → {hf_ids} ({len(hf_ids)} tokens)")

    print(f"\n[2/4] Registering imatrix hooks on target Linear modules...")
    stored_names = _safetensors_keys(args.hf_model)
    accs: dict[str, ImatrixAccumulator] = {}
    handles = []
    name_remap_count = 0
    skipped_no_ggml = []
    for mod_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not is_imatrix_target(mod_name, args.process_output):
            continue
        K = module.in_features

        # Translate in-memory name → safetensors stored name (multimodal-
        # flatten compensation), then → llama.cpp GGUF tensor name.
        stored = _translate_to_stored_name(mod_name, stored_names)
        if stored is None:
            # No safetensors counterpart — unusual; warn.
            print(f"  WARN: no safetensors key for module {mod_name!r} — skipping",
                  file=sys.stderr)
            continue
        if stored != mod_name:
            name_remap_count += 1

        ggml_name = safetensors_to_ggml_name(stored + ".weight")
        if ggml_name is None:
            # lm_head → output mapping happens via _TOP_LEVEL_MAP; if we got
            # here it means the slot isn't in our translation table.
            skipped_no_ggml.append(stored)
            continue

        accs[ggml_name] = ImatrixAccumulator(ggml_name, K)
        handles.append(module.register_forward_pre_hook(build_hook(accs[ggml_name])))

    print(f"      registered {len(accs)} hooks "
          f"({name_remap_count} names remapped HF→stored)")
    if accs:
        K_vals = [a.K for a in accs.values()]
        print(f"      K range: {min(K_vals)}..{max(K_vals)}  "
              f"({len(set(K_vals))} distinct)")
    if skipped_no_ggml:
        print(f"      skipped {len(skipped_no_ggml)} modules with no GGUF mapping "
              f"(e.g. {skipped_no_ggml[:2]})", file=sys.stderr)

    print(f"\n[3/4] Loading calibration corpus + running forward pass...")
    t0 = time.time()
    seqs = load_calibration_text(args.corpus, args.n_sequences, args.ctx_len, tokenizer)
    print(f"      tokenized {len(seqs)} chunks of {args.ctx_len} tokens "
          f"in {time.time() - t0:.1f}s")
    print(f"      starting forward pass (no_grad, eval mode)...")

    t0 = time.time()
    target_device = "cuda" if args.device == "cuda" else args.device
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            seq = seq.unsqueeze(0).to(target_device)
            model(seq)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(seqs) - i - 1) / rate
            # Print every chunk for visibility on short smoke runs; the spam
            # is negligible vs the per-chunk runtime.
            print(f"      seq {i+1}/{len(seqs)} "
                  f"({elapsed:.1f}s elapsed, {rate:.2f} seq/s, ETA {eta:.0f}s)")

    print(f"      forward pass complete: {time.time() - t0:.1f}s")
    if accs:
        sample_n = next(iter(accs.values())).n_tokens
        print(f"      tokens per tensor (sample): {sample_n} "
              f"(expected {args.n_sequences * args.ctx_len})")

    # Detach hooks before writing.
    for h in handles:
        h.remove()

    print(f"\n[4/4] Writing imatrix GGUF to {args.output}...")
    t0 = time.time()
    write_imatrix_gguf(
        args.output,
        accs,
        corpus_basename=args.corpus.name,
        n_sequences=args.n_sequences,
        ctx_len=args.ctx_len,
    )
    print(f"      wrote {args.output.stat().st_size / 1e6:.2f} MB "
          f"in {time.time() - t0:.1f}s")
    print(f"      {len(accs)} tensors ({len(accs) * 2} GGUF entries)")
    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
