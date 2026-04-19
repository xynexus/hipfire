#!/usr/bin/env python3
"""
dflash_train_poc.py — DFlash draft training loop, paper-aligned.

Targets AMD MI300X (ROCm 7.2, PyTorch 2.11+rocm7.2). Dependencies installed
by scripts/amd_quickdeploy.sh.

Based on: Chen, Liang, Liu. *DFlash: Block Diffusion for Flash Speculative
Decoding.* arXiv:2602.06036 (Feb 2026). Section 4.2 spells out five training
techniques; this script implements:
  T1 KV injection          — ref arch (.dflash-reference) handles this. ✓
  T2 Multi-anchor blocks   — K masked blocks per sequence, random anchors.
  T3 Flex attention mask   — STUBBED as K-loop of small forwards for now
                             (correctness-equivalent; will be Flex in v2).
  T4 Position-weighted CE  — w_k = exp(-(k-1)/γ), γ=3 default.
  T5 Shared embed/lm_head  — reuses target's; draft has neither. ✓

What happens per step:
  1. Sample a contiguous seq_len window from the corpus.
  2. Sample K stratified-random anchor positions (K = --masked-blocks-per-seq).
  3. Target forward over the CLEAN seq_len sequence (grad disabled) extracts
     per-layer hidden features for all positions.
  4. For each anchor: build a masked block (pos 0 = real, 1..B-1 = mask),
     run draft forward conditioned on the context feature slice before the
     anchor, compute per-position cross-entropy, apply exp-decay weights.
  5. Accumulate losses across all K anchors × batch_size examples, backprop.
  6. Ckpt every `--ckpt-every` steps as safetensors + JSON metadata.

Scale knobs:
  --target-repo            HF repo of the target (e.g. Qwen/Qwen3.5-4B).
  --draft-layers           Number of decoder layers in the draft (paper=5).
  --block-size             B (paper=16; models trained at 16 generalize to 8).
  --seq-len                Training sequence length (paper uses long ctx).
  --batch-size             Examples per step.
  --masked-blocks-per-seq  K — anchors per sequence (paper §4.2 default unclear,
                           4-8 is reasonable).
  --loss-gamma             Exp decay rate for position weighting; 0 disables.
  --lr / --steps / --warmup  AdamW + cosine.
  --corpus                 Plain-text corpus file (one doc per blank line).
  --grad-ckpt-target       Enable gradient checkpointing on target for 27B+.

For a 30-min Qwen3.5-4B validation run at batch=1 K=4 steps=5000:
  bash scripts/amd_quickdeploy.sh
  bash scripts/fetch_calibration_corpus.sh /root/agent.txt --recipe agentic
  python3 scripts/dflash_train_poc.py \
      --target-repo Qwen/Qwen3.5-4B \
      --corpus /root/agent.txt \
      --seq-len 4096 --batch-size 1 --masked-blocks-per-seq 4 \
      --steps 5000 --ckpt-every 1000 \
      --out /root/dflash_4b_agentic

Expect loss 12 → 2-3 by step 5000; if that holds, scale up to 3.6-A3B.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.optim.lr_scheduler import LambdaLR

# Pull the reference model.py off .dflash-reference/ without having to
# `pip install -e` it (avoids transformers-version conflicts).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".dflash-reference"))
from dflash.model import DFlashDraftModel, build_target_layer_ids, extract_context_feature  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target-repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--draft-layers", type=int, default=5)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--corpus", default="/root/wikitext_calib.txt")
    p.add_argument("--out", default="/root/dflash_train_poc_out")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", default=None, help="Path to checkpoint safetensors to resume from.")
    p.add_argument("--seed", type=int, default=0)
    # Paper's multi-anchor training (§4.2): concatenate K masked blocks into
    # each training sequence, each block with a random anchor position inside
    # the seq_len window. Each block contributes B-1 supervised predictions.
    p.add_argument("--masked-blocks-per-seq", type=int, default=4,
                   help="Number of anchor-masked blocks per training example (paper §4.2 'Random sampling of masked blocks').")
    # Paper's loss weighting (§4.2 'Loss weighting for faster convergence'):
    # w_k = exp(-(k-1)/gamma) where k is position within block (1-indexed).
    # Earlier positions weighted more because errors at k=1 invalidate
    # subsequent accepts. gamma=3 gives meaningful emphasis without nuking
    # the gradient on later positions.
    p.add_argument("--loss-gamma", type=float, default=3.0,
                   help="Exponential decay rate for per-position loss weighting (paper eq. 4). <=0 disables weighting.")
    p.add_argument("--grad-ckpt-target", action="store_true",
                   help="Enable gradient checkpointing on the frozen target (doesn't affect correctness; saves VRAM on 27B/35B targets).")
    return p.parse_args()


def read_corpus_tokens(corpus_path: str, tokenizer) -> list[int]:
    """Tokenize the whole corpus into a flat list of IDs, respecting BOS at doc boundaries."""
    print(f"[data] tokenizing {corpus_path}...", flush=True)
    text = Path(corpus_path).read_text()
    # Slice into docs on blank lines to avoid one giant sequence.
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    print(f"[data]   {len(docs):,} docs", flush=True)
    bos = tokenizer.bos_token_id
    ids: list[int] = []
    for d in docs:
        if bos is not None:
            ids.append(bos)
        ids.extend(tokenizer.encode(d, add_special_tokens=False))
    print(f"[data]   {len(ids):,} tokens", flush=True)
    return ids


def sample_batch(
    ids: list[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Random contiguous slices. Simple and fast; no packing tricks."""
    out = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    max_start = len(ids) - seq_len - 1
    for b in range(batch_size):
        start = random.randint(0, max_start)
        out[b] = torch.tensor(ids[start : start + seq_len], dtype=torch.long, device=device)
    return out


def build_draft_config(target_config, draft_layers: int, block_size: int, mask_token_id: int):
    """Clone a DFlash draft config from the target config — same hidden/heads, fewer layers."""
    import copy

    cfg = copy.deepcopy(target_config)
    cfg.num_hidden_layers = draft_layers
    # Required by the DFlashDraftModel __init__.
    cfg.num_target_layers = target_config.num_hidden_layers
    cfg.block_size = block_size
    cfg.dflash_config = {
        "mask_token_id": mask_token_id,
        "target_layer_ids": build_target_layer_ids(target_config.num_hidden_layers, draft_layers),
    }
    return cfg


def cosine_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load target (frozen, bf16) ────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[target] loading {args.target_repo} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target_repo)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_repo,
        torch_dtype=dtype,
        attn_implementation="eager",   # safer on ROCm; swap to sdpa once verified
    ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    print(f"[target]   {target.config.num_hidden_layers} layers, "
          f"hidden={target.config.hidden_size}, vocab={target.config.vocab_size}", flush=True)

    # Pick a mask token id that's unlikely to appear in data; reference uses 248070.
    mask_token_id = min(248070, target.config.vocab_size - 1)

    # ── build draft ───────────────────────────────────────────────────
    draft_cfg = build_draft_config(
        target.config, args.draft_layers, args.block_size, mask_token_id
    )
    draft = DFlashDraftModel(draft_cfg).to(device=device, dtype=dtype)
    if args.resume:
        from safetensors.torch import load_file
        sd = load_file(args.resume)
        missing, unexpected = draft.load_state_dict(sd, strict=False)
        print(f"[draft] resumed from {args.resume}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    n_draft_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"[draft]   {args.draft_layers} layers, {n_draft_params / 1e6:.1f}M params, block={args.block_size}", flush=True)

    # ── data ──────────────────────────────────────────────────────────
    ids = read_corpus_tokens(args.corpus, tokenizer)
    if len(ids) < args.seq_len + args.block_size + 16:
        print(f"[data] ERROR: corpus has only {len(ids)} tokens, need >{args.seq_len + args.block_size}", flush=True)
        return 2

    # ── optimizer ─────────────────────────────────────────────────────
    optim = torch.optim.AdamW(
        (p for p in draft.parameters() if p.requires_grad),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    sched = LambdaLR(optim, lambda s: cosine_schedule(s, args.warmup, args.steps))

    # Precompute per-position loss weights. Paper eq (4): w_k = exp(-(k-1)/γ).
    # k is 1-indexed position within the block; we predict positions 1..B-1.
    B = args.block_size
    if args.loss_gamma > 0:
        ks = torch.arange(1, B, device=device, dtype=torch.float32)
        pos_weights = torch.exp(-(ks - 1) / args.loss_gamma)
        pos_weights = pos_weights / pos_weights.sum()  # normalize so expected grad magnitude is comparable to uniform
    else:
        pos_weights = torch.full((B - 1,), 1.0 / (B - 1), device=device)
    print(f"[train] position weights (γ={args.loss_gamma:.2f}): "
          + ", ".join(f"{w:.3f}" for w in pos_weights.tolist()[:8])
          + (" ..." if B - 1 > 8 else ""), flush=True)

    if args.grad_ckpt_target:
        try:
            target.gradient_checkpointing_enable()
            print("[target] gradient checkpointing enabled (VRAM saver for 27B/35B targets)", flush=True)
        except Exception as e:
            print(f"[target] WARN: gradient checkpointing failed to enable: {e}", flush=True)

    # ── train ─────────────────────────────────────────────────────────
    print(f"[train] {args.steps} steps, batch={args.batch_size}, seq={args.seq_len}, "
          f"blocks_per_seq={args.masked_blocks_per_seq}, lr={args.lr}", flush=True)
    loss_ema: Optional[float] = None
    t_start = time.time()
    draft.train()
    # Spacing math: with K anchors each needing B tokens of lookahead, and
    # needing >=1 context token before each anchor, the maximum number of
    # non-overlapping anchors in a sequence of length seq_len is roughly
    # (seq_len - 1) // B. Clamp user input to that.
    max_k = max(1, (args.seq_len - 1) // B)
    anchors_per_seq = min(args.masked_blocks_per_seq, max_k)
    if anchors_per_seq < args.masked_blocks_per_seq:
        print(f"[train] clamped masked-blocks-per-seq from {args.masked_blocks_per_seq} to {anchors_per_seq} "
              f"(seq_len={args.seq_len}, B={B}: max non-overlapping anchors = {max_k})", flush=True)

    for step in range(args.steps + 1):
        optim.zero_grad(set_to_none=True)

        batch = sample_batch(ids, args.seq_len, args.batch_size, device)  # [batch, seq_len]

        # Multi-anchor sampling per sequence (paper §4.2 "Random sampling of
        # masked blocks"). For each example we sample K anchor positions
        # uniformly within [1, seq_len - B] without overlap. Each anchor
        # contributes B-1 supervised predictions, so a single target forward
        # amortizes K×(B-1) losses. Different K per example is fine — we
        # enforce the same K here for simpler batching.
        K = anchors_per_seq
        # Pool of candidate positions: each anchor + B-1 mask slots must fit
        # within seq_len. Using stratified sampling (divide seq into K
        # equal windows, sample anchor per window) both avoids overlaps and
        # improves coverage across the sequence.
        window_size = (args.seq_len - B) // K
        anchors = torch.stack([
            torch.tensor(
                [w * window_size + 1 + random.randint(0, max(0, window_size - B))
                 for w in range(K)],
                dtype=torch.long, device=device
            )
            for _ in range(args.batch_size)
        ])  # [batch, K]

        # Single target forward per batch element on the clean sequence,
        # extracting hidden features for all positions. Enumerated loop to
        # keep memory tractable for 35B targets; small models could batch.
        # Loss is accumulated across blocks per example.
        all_logits = []
        all_labels = []
        for b in range(args.batch_size):
            clean_seq = batch[b : b + 1]  # [1, seq_len]
            with torch.no_grad():
                t_out = target(
                    input_ids=clean_seq,
                    output_hidden_states=True,
                    use_cache=False,
                )
                layer_ids = draft_cfg.dflash_config["target_layer_ids"]
                # context feature shape: [1, seq_len, hidden * len(layer_ids)]
                tgt_ctx = extract_context_feature(t_out.hidden_states, layer_ids)

            # For each anchor, build an independent draft forward. This is
            # simpler than the paper's single-sequence concatenated-block mask
            # (which needs Flex Attention to be fast); correctness-equivalent
            # as long as the draft sees the same context slice + same masking
            # per block. Latency-equivalent to the paper once Flex Attention
            # lands — for now the K-loop is a readable stand-in.
            for k in range(K):
                s = int(anchors[b, k].item())
                block_ids = clean_seq[:, s : s + B]            # [1, B]
                # First position = real token (anchor = "bonus from prev verify");
                # positions 1..B-1 masked.
                masked_block = block_ids.clone()
                masked_block[:, 1:] = mask_token_id
                noise_embedding = target.model.embed_tokens(masked_block).to(dtype)

                # Context for this block = all tokens before the anchor.
                ctx_slice = tgt_ctx[:, :s, :]

                position_ids = torch.arange(s, s + B, device=device).unsqueeze(0)
                draft_hidden = draft(
                    noise_embedding=noise_embedding,
                    target_hidden=ctx_slice,
                    position_ids=position_ids,
                    use_cache=False,
                )
                # Predict positions 1..B-1 inside the block from draft states
                # at those same positions. [1, B-1, vocab].
                logits = target.lm_head(draft_hidden[:, -B + 1 :, :])
                all_logits.append(logits)
                all_labels.append(block_ids[:, 1:])

        # Stack: [B*K, B-1, vocab] and [B*K, B-1].
        draft_logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Per-position weighted cross-entropy. Compute CE per position, apply
        # paper's exp-decay weights, mean across (batch × anchor) samples.
        ce = F.cross_entropy(
            draft_logits.reshape(-1, draft_logits.size(-1)).float(),
            labels.reshape(-1),
            reduction="none",
        ).view(-1, B - 1)  # [B*K, B-1]
        loss = (ce * pos_weights.unsqueeze(0)).sum(dim=1).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optim.step()
        sched.step()

        lv = float(loss.item())
        loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            rate = (step + 1) / max(1e-6, elapsed)
            print(
                f"[step {step:6d}] loss={lv:.4f} ema={loss_ema:.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} rate={rate:.2f} step/s",
                flush=True,
            )

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = out_dir / f"draft_step{step}.safetensors"
            save_file(draft.state_dict(), str(ckpt_path))
            meta = {
                "step": step,
                "loss_ema": loss_ema,
                "target_repo": args.target_repo,
                "draft_layers": args.draft_layers,
                "block_size": args.block_size,
                "mask_token_id": mask_token_id,
                "target_layer_ids": draft_cfg.dflash_config["target_layer_ids"],
            }
            (out_dir / f"draft_step{step}.json").write_text(json.dumps(meta, indent=2))
            print(f"[ckpt]   wrote {ckpt_path}", flush=True)

    final = out_dir / "draft_final.safetensors"
    save_file(draft.state_dict(), str(final))
    (out_dir / "draft_final.json").write_text(json.dumps({
        "steps": args.steps,
        "loss_ema": loss_ema,
        "target_repo": args.target_repo,
        "draft_layers": args.draft_layers,
        "block_size": args.block_size,
        "mask_token_id": mask_token_id,
    }, indent=2))
    print(f"[done] final ckpt at {final}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
