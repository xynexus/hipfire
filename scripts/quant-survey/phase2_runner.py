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
    PRODUCTION_SIGNS1_SEED,
    PRODUCTION_SIGNS2_SEED,
    gen_fwht_signs,
    GROUP_SIZE,
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


# Chunk size for FWHT batched processing. Keeps GPU working memory under
# ~64 MB per call so the quant fits in the headroom remaining after the
# model is loaded (R9700 = 32 GB, A3B fills ~28 GB → ~2 GB free per GPU).
FWHT_CHUNK_GROUPS = 65536  # = 16 MB per pass × 8 passes ≈ small enough


def _torch_fwht_256_chunk(x_chunk, signs1_t, signs2_t, inverse: bool = False):
    """One FWHT pass over a small chunk shape [n_chunk, 256]. Caller
    chunks for memory; this is the inner kernel."""
    import torch
    if inverse:
        out = x_chunk * signs2_t
    else:
        out = x_chunk * signs1_t
    n_chunk = out.shape[0]
    stride = 1
    while stride < 256:
        n_pairs = 256 // (2 * stride)
        viewed = out.reshape(n_chunk, n_pairs, 2, stride)
        a = viewed[:, :, 0, :]
        b = viewed[:, :, 1, :].clone()  # clone so write-back below doesn't alias
        # Write in-place into viewed: a = a + b; then overwrite b slot.
        # We need both new_a and new_b, but the cheap path is a fresh
        # buffer at half the previous stack-based cost since we use
        # cat along the stride axis.
        new_a = a + b
        new_b = a - b
        out = torch.cat((new_a.unsqueeze(2), new_b.unsqueeze(2)), dim=2).reshape(n_chunk, 256)
        del a, b, new_a, new_b, viewed
        stride <<= 1
    out = out * (0.0625 * (signs1_t if inverse else signs2_t))
    return out


def _torch_fwht_256_batch(x, signs1_t, signs2_t, inverse: bool = False):
    """Chunked FWHT on shape [..., 256]. Reshapes to [n_groups, 256], runs
    `_torch_fwht_256_chunk` over chunks of FWHT_CHUNK_GROUPS, concatenates.
    Returns same shape as input.
    """
    import torch
    out_shape = x.shape
    flat = x.reshape(-1, 256).contiguous()
    n_groups = flat.shape[0]
    if n_groups <= FWHT_CHUNK_GROUPS:
        return _torch_fwht_256_chunk(flat, signs1_t, signs2_t, inverse).reshape(out_shape)
    # Chunked: allocate output, fill chunk by chunk, free temporaries.
    out = torch.empty_like(flat)
    for start in range(0, n_groups, FWHT_CHUNK_GROUPS):
        end = min(start + FWHT_CHUNK_GROUPS, n_groups)
        out[start:end] = _torch_fwht_256_chunk(flat[start:end], signs1_t, signs2_t, inverse)
    return out.reshape(out_shape)


def _torch_quant_2d(arr, q8: bool, signs1_t, signs2_t):
    """Round-trip a 2D tensor on its home device. Chunks the per-group
    work to keep GPU working memory bounded. arr is [n_rows, n_cols] with
    n_cols % 256 == 0.
    """
    import torch
    n_rows, n_cols = arr.shape
    n_groups_per_row = n_cols // GROUP_SIZE
    n_total_groups = n_rows * n_groups_per_row

    grouped = arr.reshape(n_total_groups, GROUP_SIZE).contiguous().to(dtype=torch.float32)
    levels = 255.0 if q8 else 15.0

    out = torch.empty_like(grouped)
    for start in range(0, n_total_groups, FWHT_CHUNK_GROUPS):
        end = min(start + FWHT_CHUNK_GROUPS, n_total_groups)
        chunk = grouped[start:end]
        rotated = chunk if q8 else _torch_fwht_256_chunk(chunk, signs1_t, signs2_t, inverse=False)
        grp_min = rotated.amin(dim=1)
        grp_max = rotated.amax(dim=1)
        grp_range = grp_max - grp_min
        safe = grp_range > 0
        grp_scale = torch.where(safe, grp_range / levels, torch.ones_like(grp_range))
        grp_inv_scale = torch.where(safe, 1.0 / grp_scale, torch.zeros_like(grp_scale))
        centered = rotated - grp_min[:, None]
        q = torch.round(centered * grp_inv_scale[:, None] + 1e-6).to(dtype=torch.int32)
        q = torch.clamp(q, 0, int(levels))
        deq = q.to(dtype=torch.float32) * grp_scale[:, None] + grp_min[:, None]
        recon = deq if q8 else _torch_fwht_256_chunk(deq, signs1_t, signs2_t, inverse=True)
        out[start:end] = recon
        del rotated, grp_min, grp_max, grp_range, safe, grp_scale, grp_inv_scale
        del centered, q, deq, recon
    return out.reshape(n_rows, n_cols)


def apply_variant(model, variant: str) -> dict[str, int]:
    """Walk every parameter, apply the variant's quant plan in-place on the
    parameter's home device via torch ops. NO CPU round-trip — keeps
    everything on the R9700 GPUs that already hold the model. Returns
    counts of {q4, q8, skipped, moe_3d}.

    The FWHT signs are generated once via numpy (LCG seeded by production
    seeds 42 / 1042), then copied to torch tensors on each device as
    needed. Quant proceeds entirely in fp32 GPU compute and writes back
    to the parameter's original dtype.
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

    # Pre-build sign tables on CPU once; they're tiny (1KB each).
    signs1_np = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    signs2_np = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)
    # Cache per-device torch copies so we don't re-upload per-tensor.
    signs_cache: dict[str, tuple] = {}

    def signs_on(device):
        key = str(device)
        if key not in signs_cache:
            s1 = torch.from_numpy(signs1_np).to(device=device, dtype=torch.float32)
            s2 = torch.from_numpy(signs2_np).to(device=device, dtype=torch.float32)
            signs_cache[key] = (s1, s2)
        return signs_cache[key]

    counts = {"q4": 0, "q8": 0, "skipped": 0, "moe_3d": 0}
    t0 = time.time()
    n_params = sum(1 for _ in model.named_parameters())
    print(f"phase2: applying variant {variant} ({n_params} parameters)", file=sys.stderr)

    with torch.no_grad():
        for i, (name, p) in enumerate(model.named_parameters()):
            if _should_skip(name):
                counts["skipped"] += 1
                continue
            if p.dim() < 2:
                counts["skipped"] += 1
                continue

            orig_dtype = p.dtype
            device = p.device
            s1_t, s2_t = signs_on(device)

            if _is_moe_3d_expert_tensor(name) and p.dim() == 3:
                layer_idx = _layer_idx_of(name) or -1
                is_down = ".mlp.experts.down_proj" in name
                n_experts, n_rows, n_cols = p.shape
                pinned_set = (
                    set(e for e in range(n_experts) if is_down and (layer_idx, e) in pin_set)
                    if is_down else set()
                )

                # Chunk the experts to bound peak working memory. With
                # 8 experts per chunk on (8, n_rows, n_cols) bf16 + fp32
                # working buffers, peak ≈ ~150 MB even on the larger
                # gate_up_proj. Fits comfortably in the ~2 GB headroom
                # left after device_map="auto" packs the model.
                EXPERT_CHUNK = 8
                n_q4 = 0
                n_q8 = 0
                for e0 in range(0, n_experts, EXPERT_CHUNK):
                    e1 = min(e0 + EXPERT_CHUNK, n_experts)
                    for e in range(e0, e1):
                        use_q8 = all_q8 or (e in pinned_set)
                        rt = _torch_quant_2d(p[e], q8=use_q8, signs1_t=s1_t, signs2_t=s2_t)
                        p.data[e] = rt.to(dtype=orig_dtype)
                        if use_q8:
                            n_q8 += 1
                        else:
                            n_q4 += 1
                    # Free fragmented blocks between chunks.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                counts["q8"] += n_q8
                counts["q4"] += n_q4
                counts["moe_3d"] += 1
            elif p.dim() == 2:
                n_rows, n_cols = p.shape
                if n_cols % GROUP_SIZE != 0:
                    counts["skipped"] += 1
                    continue
                rt = _torch_quant_2d(p, q8=all_q8, signs1_t=s1_t, signs2_t=s2_t)
                p.data.copy_(rt.to(dtype=orig_dtype))
                if all_q8:
                    counts["q8"] += 1
                else:
                    counts["q4"] += 1
            else:
                counts["skipped"] += 1
                continue

            if (i + 1) % 50 == 0 or i == n_params - 1:
                dt = time.time() - t0
                print(f"  param {i+1}/{n_params}  q4={counts['q4']} q8={counts['q8']} "
                      f"skipped={counts['skipped']} moe_3d={counts['moe_3d']} dt={dt:.1f}s",
                      file=sys.stderr)

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
