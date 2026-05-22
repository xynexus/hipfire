# Phase 4-7 launch commands (mi300)

Reference paths verified 2026-05-22. Use `python3 -u` for unbuffered logs.

## Phase 4 — Qwen3.5-0.8B (running, PID 567138)

```bash
ssh mi300 'cd /workspace/hipfire && nohup python3 -u scripts/learn_butterfly_mq.py \
    --model /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
    --imatrix /workspace/qwen3.5-0.8b.mix.ctx4096.imatrix.gguf \
    --corpus /workspace/calibration-mix-v1.txt \
    --output-dir /workspace/butterfly-phase4-0.8b \
    --n-sequences 128 --ctx-len 2048 --n-epochs 4 --lr 1e-3 \
    --smoke-eval --smoke-eval-seqs 64 --log-interval 16 \
    > /workspace/butterfly-phase4-0.8b/train.log 2>&1 </dev/null &
disown; echo "PID $!"'
```

## Phase 5 — Qwen3.5-9B

```bash
ssh mi300 'cd /workspace/hipfire && nohup python3 -u scripts/learn_butterfly_mq.py \
    --model /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --imatrix /workspace/qwen3.5-9b.tier1.mix.imatrix.gguf \
    --corpus /workspace/calibration-mix-v1.txt \
    --output-dir /workspace/butterfly-phase5-9b \
    --n-sequences 128 --ctx-len 2048 --n-epochs 4 --lr 1e-3 \
    --smoke-eval --smoke-eval-seqs 64 --log-interval 16 \
    > /workspace/butterfly-phase5-9b/train.log 2>&1 </dev/null &
disown; echo "PID $!"'
```

Estimated wall-clock: ~5 hr (12× per-step cost vs 0.8B). Within plan budget (8 hr).

## Phase 6 — Qwen3.6-27B

```bash
ssh mi300 'cd /workspace/hipfire && nohup python3 -u scripts/learn_butterfly_mq.py \
    --model /workspace/hf-models/qwen3.6-27b \
    --imatrix /workspace/qwen3.6-27b.tier1.imatrix.gguf \
    --corpus /workspace/calibration-mix-v1.txt \
    --output-dir /workspace/butterfly-phase6-27b \
    --n-sequences 128 --ctx-len 2048 --n-epochs 3 --lr 1e-3 \
    --smoke-eval --smoke-eval-seqs 32 --log-interval 16 \
    > /workspace/butterfly-phase6-27b/train.log 2>&1 </dev/null &
disown; echo "PID $!"'
```

Reduced epochs (3 vs 4) and smoke eval seqs (32 vs 64) to keep within plan
budget (16 hr). 384 steps still in 500-700 paper range.

## Phase 7 — Qwen3.6-35B-A3B (asymptote search)

```bash
ssh mi300 'cd /workspace/hipfire && nohup python3 -u scripts/learn_butterfly_mq.py \
    --model /workspace/hf-models/qwen3.6-35b-a3b \
    --imatrix /workspace/imatrix/Qwen3.6-35B-A3B-GGUF/imatrix_unsloth.gguf_file \
    --corpus /workspace/calibration-mix-v1.txt \
    --output-dir /workspace/butterfly-phase7-a3b \
    --n-sequences 128 --ctx-len 2048 --n-epochs 5 --lr 1e-3 \
    --smoke-eval --smoke-eval-seqs 32 --log-interval 16 \
    > /workspace/butterfly-phase7-a3b/train.log 2>&1 </dev/null &
disown; echo "PID $!"'
```

A3B is MoE — needs handling for per-expert butterfly (256 experts × layer).
The script currently wraps Linears generically; may need to extend to wrap
expert MLP Linears (mlp.experts.N.gate_proj etc) explicitly. Audit before fire.

Asymptote stop condition (no KLD Δ > 0.001 across 5 iters) is NOT implemented
yet in learn_butterfly_mq.py — currently it just runs fixed n_epochs.
Implement by running multiple short trains in sequence, comparing trained
KLD after each iteration, halting when delta drops below threshold.

## Resource notes (mi300, 192 GB VRAM)

| Model | Params | BF16 × 2 | Autograd state | Total est. |
|---|---:|---:|---:|---:|
| 0.8B | 0.8 B | 3 GB | 0.5 GB | 4 GB |
| 9B | 9 B | 36 GB | 5 GB | 42 GB |
| 27B | 27 B | 108 GB | 15 GB | 130 GB |
| 35B-A3B | 35 B | 140 GB | ~10 GB (MoE sparse) | 160 GB |

27B is tightest; if OOM, reduce n_sequences or use gradient checkpointing.
