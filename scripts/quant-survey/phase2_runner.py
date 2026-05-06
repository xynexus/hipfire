"""phase2_runner.py — Pre-registered SE ablation perplexity test.

Runs ONE (model, variant, trial) per invocation. Loads bf16 reference
via transformers, mutates weights in-place per the variant's quant
plan, runs the eval corpus through forward, computes cross-entropy,
emits one phase2_results.jsonl line.

Variants per `03-super-expert-confirmation.md`:
  V1     All-MQ4        baseline
  V2     All-Q8         ceiling
  V3a    D2-pinned      17 layer-0 expert down_proj rows pinned at Q8
  V3b    D3-pinned      19 layer-38/39 expert down_proj rows pinned at Q8
  V3c    Union          36 expert down_proj rows pinned at Q8

Quantization scope: every Linear + every MoE-experts 3D weight in the
model EXCEPT norms, router gates (`mlp.gate.weight`), embed_tokens,
lm_head, biases. This mirrors what hipfire-quantize touches in
production.

Usage:

  python3 scripts/quant-survey/phase2_runner.py \\
      --model qwen3.5-a3b \\
      --variant V3b \\
      --trial 1 \\
      --output-dir /tmp/hiptrx-survey/runs/phase2-3.5-a3b/

The output dir's `phase2_results.jsonl` is APPENDED to. After 5
variants × 3 trials per model = 15 lines per model file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from quant_ops import (  # noqa: E402
    quantize_then_dequantize_mq4g256_fwht_vectorized,
    quantize_then_dequantize_q8g256,
    PRODUCTION_SIGNS1_SEED,
    PRODUCTION_SIGNS2_SEED,
    gen_fwht_signs,
)


MODEL_REGISTRY: dict[str, str] = {
    "qwen3.5-a3b": "Qwen/Qwen3.5-35B-A3B",
    "qwen3.6-a3b": "Qwen/Qwen3.6-35B-A3B",
}

# D2-derived SE candidates (layer 0 down_proj weight ratio_p99 cliff).
D2_PIN_SET: set[tuple[int, int]] = {(0, e) for e in
    [3, 8, 42, 49, 70, 115, 119, 132, 164, 167,
     190, 195, 203, 225, 237, 239, 253]}

# D3-derived SE candidates (activation absmax_max output side, layers 38-39).
D3_PIN_SET: set[tuple[int, int]] = {
    (38, 48), (38, 103), (38, 209),
    (39, 5), (39, 21), (39, 27), (39, 37), (39, 101),
    (39, 108), (39, 113), (39, 149), (39, 155), (39, 170),
    (39, 200), (39, 209), (39, 229), (39, 238), (39, 251), (39, 255),
}

UNION_PIN_SET = D2_PIN_SET | D3_PIN_SET

# Skip these tensor name patterns when applying quant.
SKIP_PATTERNS = (
    "norm",         # all RMSNorm weights
    "embed_tokens",
    "lm_head",
    "mlp.gate.weight",     # MoE router (small)
    "shared_expert_gate",  # shared-expert binary gate
)


def _should_skip(name: str) -> bool:
    return any(p in name for p in SKIP_PATTERNS)


def _is_moe_3d_expert_tensor(name: str) -> bool:
    """True for `model.layers.X.mlp.experts.{gate_up_proj,down_proj}` —
    3D parameters with shape [n_experts, ...]."""
    return ".mlp.experts.gate_up_proj" in name or ".mlp.experts.down_proj" in name


def _layer_idx_of(name: str) -> int | None:
    import re
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _is_down_proj(name: str) -> bool:
    return name.endswith(".down_proj") or name.endswith(".down_proj.weight")


def _quant_2d(arr_f32: np.ndarray, q8: bool, signs1, signs2) -> np.ndarray:
    """Round-trip a 2D weight tensor. Quant is per-row groups of 256 along
    the LAST axis (consistent with hipfire-quantize's per-row group layout).
    """
    n_rows, n_cols = arr_f32.shape
    out = np.empty_like(arr_f32)
    for r in range(n_rows):
        if q8:
            out[r] = quantize_then_dequantize_q8g256(arr_f32[r])
        else:
            out[r] = quantize_then_dequantize_mq4g256_fwht_vectorized(arr_f32[r], signs1, signs2)
    return out


def _quant_3d_expert(arr_f32: np.ndarray, layer_idx: int, is_down_proj: bool,
                     pin_set: set[tuple[int, int]], signs1, signs2) -> np.ndarray:
    """Round-trip a 3D MoE expert tensor [n_experts, M, K]. Per-expert
    quant choice: if (layer, expert) is in pin_set AND this is a
    down_proj tensor, use Q8; else MQ4G256+FWHT.

    Per arXiv 2507.23279 SE definition, only down_proj is the pinning
    target. gate_up_proj is always MQ4 in V3 variants.
    """
    n_experts = arr_f32.shape[0]
    out = np.empty_like(arr_f32)
    for e in range(n_experts):
        pin_q8 = is_down_proj and ((layer_idx, e) in pin_set)
        out[e] = _quant_2d(arr_f32[e], q8=pin_q8, signs1=signs1, signs2=signs2)
    return out


def apply_variant(model, variant: str) -> dict[str, int]:
    """Walk every parameter, apply the variant's quant plan in-place via
    .data.copy_(round_tripped). Returns counts of {q4, q8, skipped}.

    Memory: each tensor is moved to CPU as float32 for round-trip, then
    cast back to bf16 and copied to its original device. Peak overhead
    is ~one tensor's worth of f32 + the output buffer.
    """
    import torch

    if variant == "V1":
        pin_set: set[tuple[int, int]] = set()
        all_q8 = False
    elif variant == "V2":
        pin_set = set()
        all_q8 = True
    elif variant == "V3a":
        pin_set = D2_PIN_SET
        all_q8 = False
    elif variant == "V3b":
        pin_set = D3_PIN_SET
        all_q8 = False
    elif variant == "V3c":
        pin_set = UNION_PIN_SET
        all_q8 = False
    else:
        raise ValueError(f"unknown variant {variant!r}")

    signs1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    signs2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)

    counts = {"q4": 0, "q8": 0, "skipped": 0, "moe_3d": 0}
    t0 = time.time()
    n_params = sum(1 for _ in model.named_parameters())
    print(f"phase2: applying variant {variant} ({n_params} parameters)", file=sys.stderr)

    for i, (name, p) in enumerate(model.named_parameters()):
        if _should_skip(name):
            counts["skipped"] += 1
            continue
        if p.dim() < 2:
            counts["skipped"] += 1
            continue

        orig_device = p.device
        orig_dtype = p.dtype
        arr_f32 = p.detach().to(dtype=torch.float32, device="cpu").numpy()

        if _is_moe_3d_expert_tensor(name) and arr_f32.ndim == 3:
            layer_idx = _layer_idx_of(name) or -1
            is_down = ".mlp.experts.down_proj" in name
            if all_q8:
                # V2: Q8 everywhere — every expert slice gets Q8.
                out = np.empty_like(arr_f32)
                for e in range(arr_f32.shape[0]):
                    rows = arr_f32[e]
                    out_e = np.empty_like(rows)
                    for r in range(rows.shape[0]):
                        out_e[r] = quantize_then_dequantize_q8g256(rows[r])
                    out[e] = out_e
                counts["q8"] += arr_f32.shape[0]
            else:
                out = _quant_3d_expert(arr_f32, layer_idx, is_down, pin_set, signs1, signs2)
                pinned = sum(1 for e in range(arr_f32.shape[0]) if is_down and (layer_idx, e) in pin_set)
                counts["q8"] += pinned
                counts["q4"] += arr_f32.shape[0] - pinned
            counts["moe_3d"] += 1
        elif arr_f32.ndim == 2:
            q8 = all_q8
            out = _quant_2d(arr_f32, q8=q8, signs1=signs1, signs2=signs2)
            if q8:
                counts["q8"] += 1
            else:
                counts["q4"] += 1
        else:
            counts["skipped"] += 1
            continue

        # Cast back + copy to original device.
        import torch
        out_t = torch.from_numpy(out.astype(np.float32))
        out_t = out_t.to(dtype=orig_dtype, device=orig_device)
        with torch.no_grad():
            p.data.copy_(out_t)
        del out_t, out, arr_f32

        if (i + 1) % 50 == 0 or i == n_params - 1:
            dt = time.time() - t0
            print(f"  param {i+1}/{n_params}  q4={counts['q4']} q8={counts['q8']} "
                  f"skipped={counts['skipped']} dt={dt:.1f}s", file=sys.stderr)

    return counts


def compute_perplexity(model, tokenizer, eval_records, max_seq_len: int, device) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    model.eval()
    ce_sum = 0.0
    n_tokens = 0
    t0 = time.time()
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    for i, rec in enumerate(eval_records):
        tokens = tokenizer(rec["prompt"], return_tensors="pt", truncation=True, max_length=max_seq_len)
        input_ids = tokens.input_ids.to(device)
        if input_ids.shape[1] < 8:
            continue
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits  # [1, seq, vocab]
            shift_logits = logits[..., :-1, :].contiguous().to(dtype=torch.float32)
            shift_labels = input_ids[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ce_sum += float(loss.item())
        n_tokens += shift_labels.numel()
        if (i + 1) % 25 == 0:
            print(f"  eval {i+1}/{len(eval_records)}  ce_sum={ce_sum:.1f}  n={n_tokens}",
                  file=sys.stderr)
    elapsed = time.time() - t0
    ppl = float(np.exp(ce_sum / n_tokens)) if n_tokens > 0 else float("inf")
    return {
        "ce_sum": float(ce_sum),
        "n_tokens": int(n_tokens),
        "ppl": ppl,
        "wall_time_s": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--variant", required=True, choices=["V1", "V2", "V3a", "V3b", "V3c"])
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hf-cache", default=str(Path.home() / ".cache/huggingface/hub"))
    ap.add_argument("--eval-corpus", default=str(_THIS / "phase2_eval_corpus.jsonl"))
    ap.add_argument("--max-seq-len", type=int, default=1024)
    args = ap.parse_args()

    repo = MODEL_REGISTRY[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "phase2_results.jsonl"

    # md5 corpus for record-keeping.
    eval_path = Path(args.eval_corpus)
    h = hashlib.md5(eval_path.read_bytes()).hexdigest()
    with open(eval_path) as f:
        eval_records = [json.loads(line) for line in f if line.strip()]
    print(f"phase2: corpus {eval_path} md5={h} n_records={len(eval_records)}", file=sys.stderr)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"phase2: loading {repo} (bf16, device_map=auto)", file=sys.stderr)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=args.hf_cache, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        cache_dir=args.hf_cache,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    embed_device = next(model.parameters()).device
    print(f"phase2: loaded in {time.time()-t0:.1f}s, embed on {embed_device}", file=sys.stderr)

    # Apply variant quant.
    counts = apply_variant(model, args.variant)
    print(f"phase2: variant {args.variant} applied: {counts}", file=sys.stderr)

    # Compute PPL on eval corpus.
    eval_result = compute_perplexity(model, tokenizer, eval_records, args.max_seq_len, embed_device)
    print(f"phase2: PPL={eval_result['ppl']:.4f} on {eval_result['n_tokens']} tokens "
          f"({eval_result['wall_time_s']}s)", file=sys.stderr)

    # Append result.
    record = {
        "model": args.model,
        "repo": repo,
        "variant": args.variant,
        "trial": args.trial,
        "eval_corpus_md5": h,
        "max_seq_len": args.max_seq_len,
        "quant_counts": counts,
        **eval_result,
    }
    with open(results_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"phase2: appended to {results_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
