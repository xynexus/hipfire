"""d3_runner.py — Phase 1B-bis: per-channel activation absmax via forward hooks.

Wires diagnostics/d3_activation.py's attach_hooks() onto a transformers
bf16-reference model loaded with device_map="auto", runs the calibration
corpus, and emits per_channel.jsonl + summary.json.

D3 is the activation-side counterpart to D1/D2/D4 (which measure
weight-side properties). The headline target is the per-channel absmax
of MoE down_proj inputs and outputs across all experts at every layer —
the direct measurement of arXiv 2507.23279's "Super-Expert" output
magnitude signal.

Hardware:
  9B   -> 1 R9700 (18 GB bf16) — HIP_VISIBLE_DEVICES=0
  27B  -> 2 R9700 (54 GB bf16) — device_map="auto" picks 0,1
  A3B  -> 4 R9700 (70 GB bf16) — device_map="auto" picks 0,1,2,3

Output (per --output-dir):
  per_channel.jsonl   one record per (layer_idx, hook_point, side)
                      with full per-channel absmax array
  summary.json        manifest + aggregate stats + top-N outlier hooks
                      by absmax_max
  manifest.json       model/repo/corpus_md5/projections/runner_schema_version
                      for resume safety

Usage:

  python3 scripts/quant-survey/d3_runner.py \\
      --model qwen3.5-9b \\
      --hf-cache ~/.cache/huggingface/hub \\
      --output-dir /tmp/hiptrx-survey/runs/qwen3.5-9b-d3/ \\
      --calibration-corpus scripts/quant-survey/calibration_corpus.jsonl \\
      --max-seq-len 512 \\
      --projections q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,shared_expert.gate_proj,shared_expert.up_proj,shared_expert.down_proj

This MUST run on a host with ROCm torch installed (hiptrx ~/venv-torch
per cd61ef2). It is NOT runnable on k9lin (no ROCm wheels for that
GPU's gfx1010).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS / "diagnostics"))

from safetensors_reader import find_hf_snapshot, read_config  # noqa: E402
from diagnostics.d3_activation import (  # noqa: E402
    attach_hooks,
    detach_hooks,
    attach_moe_per_expert_hooks,
    detach_moe_per_expert_hooks,
    emit_per_expert_jsonl,
    run_calibration,
    emit_per_channel_jsonl,
)


MODEL_REGISTRY: dict[str, str] = {
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",
    "qwen3.5-a3b": "Qwen/Qwen3.5-35B-A3B",
    "qwen3.6-a3b": "Qwen/Qwen3.6-35B-A3B",
}

# Default projections — covers the full set of Linear layers we care
# about across both dense and MoE Qwen 3.5/3.6 architectures. The
# d3_activation module collapses per-expert layers into a single hook
# point, so "down_proj" matches both `mlp.down_proj` (dense) and
# `mlp.experts.X.down_proj` (MoE, collapsed across X).
DEFAULT_PROJECTIONS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
]


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_corpus(path: Path) -> tuple[list[dict[str, Any]], str]:
    md5 = _md5_file(path)
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records, md5


def _load_model_and_tokenizer(repo: str, hf_cache: Path):
    """Load bf16 reference with device_map="auto".

    Returns (model, tokenizer, embed_device). embed_device is where
    input_ids must live for the forward pass.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"d3_runner: loading {repo} (bf16, device_map=auto)", file=sys.stderr)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=str(hf_cache), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        cache_dir=str(hf_cache),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    embed_device = next(model.parameters()).device
    elapsed = time.time() - t0
    print(f"d3_runner: model loaded in {elapsed:.1f}s, embed on {embed_device}", file=sys.stderr)
    if hasattr(model, "hf_device_map"):
        unique_devs = sorted({str(v) for v in model.hf_device_map.values() if v is not None})
        print(f"d3_runner: hf_device_map spans {unique_devs}", file=sys.stderr)
    return model, tokenizer, embed_device


def _classify_outliers(records: list[dict[str, Any]], top_n: int = 50) -> list[dict[str, Any]]:
    """Sort hook records by absmax_max descending; return top_n metadata
    (without the full absmax array).
    """
    annotated = []
    for r in records:
        annotated.append({
            "layer_idx": r["layer_idx"],
            "hook_point": r["hook_point"],
            "side": r["side"],
            "absmax_max": r["absmax_max"],
            "absmax_mean": r["absmax_mean"],
            "n_calls": r["n_calls"],
            "n_channels": r["n_channels"],
        })
    annotated.sort(key=lambda d: -d["absmax_max"])
    return annotated[:top_n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--hf-cache", default=str(Path.home() / ".cache/huggingface/hub"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--calibration-corpus", default=str(_THIS / "calibration_corpus.jsonl"))
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--projections", default=",".join(DEFAULT_PROJECTIONS))
    ap.add_argument("--limit-prompts", type=int, default=-1, help="cap prompts for smoke testing; -1 = full corpus")
    args = ap.parse_args()

    projections = sorted({p.strip() for p in args.projections.split(",") if p.strip()})
    repo = MODEL_REGISTRY[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.calibration_corpus)

    per_channel_path = output_dir / "per_channel.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    corpus, corpus_md5 = _load_corpus(corpus_path)
    if args.limit_prompts > 0:
        corpus = corpus[:args.limit_prompts]
    print(f"d3_runner: corpus {corpus_path} md5={corpus_md5} n_records={len(corpus)}", file=sys.stderr)

    manifest_now = {
        "diagnostic": "d3",
        "model": args.model,
        "repo": repo,
        "projections": projections,
        "calibration_corpus_md5": corpus_md5,
        "calibration_corpus_path": str(corpus_path),
        "max_seq_len": args.max_seq_len,
        "n_prompts": len(corpus),
        "runner_schema_version": 1,
    }

    if per_channel_path.exists() and not manifest_path.exists():
        raise SystemExit(
            f"d3_runner: ERROR {per_channel_path} exists but {manifest_path} does not. "
            f"Cannot safely treat as resume base — model/seeds/corpus could differ. "
            f"Either point --output-dir at a fresh path, OR delete {per_channel_path} "
            f"and rerun."
        )

    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                prior = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"d3_runner: ERROR cannot read existing manifest: {e}")
        # D3 has no per-tensor resume — it is whole-model forward-pass.
        # Refuse any existing manifest with mismatched fields or just refuse
        # resume entirely (user must delete dir to rerun).
        for k in ("diagnostic", "model", "repo", "calibration_corpus_md5",
                  "max_seq_len", "n_prompts", "runner_schema_version"):
            if prior.get(k) != manifest_now.get(k):
                raise SystemExit(
                    f"d3_runner: ERROR manifest mismatch at {manifest_path}: "
                    f"field {k!r} prior={prior.get(k)!r} != requested={manifest_now.get(k)!r}. "
                    f"Resume is unsafe; delete the output dir to rerun."
                )
        if set(prior.get("projections", [])) != set(projections):
            raise SystemExit(
                f"d3_runner: ERROR projections mismatch at {manifest_path}: "
                f"prior={sorted(prior.get('projections', []))} "
                f"requested={projections}. Delete output dir to rerun."
            )
        # Manifest matched — but D3 is non-incremental; if per_channel.jsonl
        # exists alongside a matching manifest we assume the prior run completed
        # and refuse to redo work silently.
        if per_channel_path.exists() and per_channel_path.stat().st_size > 0:
            raise SystemExit(
                f"d3_runner: prior run already produced {per_channel_path} "
                f"with matching manifest. D3 is non-incremental — delete the "
                f"output dir if you want to rerun."
            )
    else:
        with open(manifest_path, "w") as f:
            json.dump(manifest_now, f, indent=2)

    snapshot = find_hf_snapshot(Path(args.hf_cache).expanduser(), repo)
    config = read_config(snapshot)
    print(f"d3_runner: arch={config.get('architectures', ['?'])} "
          f"hidden_size={config.get('hidden_size')} "
          f"num_hidden_layers={config.get('num_hidden_layers')} "
          f"num_experts={config.get('num_experts', 'N/A')}",
          file=sys.stderr)

    model, tokenizer, embed_device = _load_model_and_tokenizer(repo, Path(args.hf_cache).expanduser())

    state = attach_hooks(model, set(projections))
    print(f"d3_runner: hooked {state['n_modules_hooked']} modules into "
          f"{len(state['accumulators'])} accumulators "
          f"({state['n_modules_shared_with_existing_acc']} shared via MoE-collapse)",
          file=sys.stderr)

    # Per-expert MoE monkey-patch (silent no-op for dense models since they
    # have no Qwen3_5MoeExperts module). For A3B, this captures
    # per-(layer, expert, side) down_proj input + output absmax during
    # the routed-experts loop — the activation-side counterpart of the
    # weight-side D2 ratio cliff (already known from 02-survey-results.md).
    moe_state = attach_moe_per_expert_hooks(model)
    print(f"d3_runner: monkey-patched {moe_state['n_modules_patched']} MoE-experts "
          f"modules ({moe_state['n_experts_total']} experts total)", file=sys.stderr)

    t_calib = time.time()
    try:
        prompts = [r["prompt"] for r in corpus]
        n_tokens = run_calibration(
            model, tokenizer, prompts,
            state["accumulators"],
            max_seq_len=args.max_seq_len,
            device=str(embed_device),
        )
    finally:
        # ALWAYS detach + un-patch even on error so the model isn't left mutated.
        detach_hooks(state)
        detach_moe_per_expert_hooks(moe_state)
    elapsed_calib = time.time() - t_calib
    print(f"d3_runner: calibration {n_tokens} tokens in {elapsed_calib:.1f}s "
          f"({n_tokens / elapsed_calib:.1f} tok/s)", file=sys.stderr)

    # Emit per-channel (Linear-hooked) and per-expert (MoE-patched) JSONL.
    n_records = emit_per_channel_jsonl(state["accumulators"], per_channel_path, args.model)
    print(f"d3_runner: wrote {n_records} per-channel records to {per_channel_path}", file=sys.stderr)
    per_expert_path = output_dir / "per_expert.jsonl"
    n_expert_records = 0
    if moe_state["accumulators"]:
        n_expert_records = emit_per_expert_jsonl(moe_state["accumulators"], per_expert_path, args.model)
        print(f"d3_runner: wrote {n_expert_records} per-expert records to {per_expert_path}", file=sys.stderr)

    # Re-read the JSONL to build the summary (avoids holding everything in memory twice).
    records: list[dict[str, Any]] = []
    with open(per_channel_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    outliers = _classify_outliers(records, top_n=50)
    layers_seen = sorted({r["layer_idx"] for r in records})
    hook_points_seen = sorted({r["hook_point"] for r in records})

    # Per-expert summary: top-N (layer, expert) by absmax_max (down_proj output side).
    per_expert_outliers: list[dict[str, Any]] = []
    if n_expert_records > 0:
        # Re-read per_expert.jsonl for top-N selection (avoid keeping all in mem).
        with open(per_expert_path) as f:
            ex_records = [json.loads(line) for line in f if line.strip()]
        # Sort by absmax_max within side="output" (the SE signal target).
        out_recs = [r for r in ex_records if r["side"] == "output"]
        out_recs.sort(key=lambda d: -d["absmax_max"])
        per_expert_outliers = [
            {
                "layer_idx": r["layer_idx"],
                "expert_idx": r["expert_idx"],
                "side": r["side"],
                "absmax_max": r["absmax_max"],
                "absmax_mean": r["absmax_mean"],
                "n_tokens_routed": r["n_tokens_routed"],
            }
            for r in out_recs[:50]
        ]

    summary = {
        "manifest": manifest_now,
        "snapshot": str(snapshot),
        "config": {
            "architectures": config.get("architectures"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_experts": config.get("num_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "moe_intermediate_size": config.get("moe_intermediate_size"),
        },
        "n_modules_hooked": state["n_modules_hooked"],
        "n_modules_shared_with_existing_acc": state["n_modules_shared_with_existing_acc"],
        "n_accumulators": len(state["accumulators"]),
        "n_records_emitted": n_records,
        "n_layers_observed": len(layers_seen),
        "n_hook_points_observed": len(hook_points_seen),
        "n_tokens_calibrated": n_tokens,
        "wall_time_calibration_seconds": round(elapsed_calib, 1),
        "outliers_top50_by_absmax_max": outliers,
        "moe_per_expert": {
            "n_modules_patched": moe_state["n_modules_patched"],
            "n_experts_total": moe_state["n_experts_total"],
            "n_records_emitted": n_expert_records,
            "outliers_top50_by_absmax_max_output_side": per_expert_outliers,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"d3_runner: wrote {summary_path}", file=sys.stderr)
    if outliers[:5]:
        print(f"d3_runner: top-5 absmax_max:", file=sys.stderr)
        for o in outliers[:5]:
            print(f"  layer={o['layer_idx']:3d} {o['hook_point']:.<48s} {o['side']:6s} "
                  f"absmax_max={o['absmax_max']:.3e} absmax_mean={o['absmax_mean']:.3e}",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
