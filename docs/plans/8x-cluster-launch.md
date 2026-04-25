# 8× MI300X Cluster Launch Plan

**Trigger:** 9B draft+sidecar validated on 1× MI300X run. User spins the 8× cluster. This doc is the fast-start checklist.

## Scope

Training jobs that meaningfully benefit from 8× (each job gets its own GPU):
1. **Qwen3.6-35B-A3B draft** — main agentic target (user priority)
2. **Qwen3.5-27B draft** — largest dense target
3. **Qwen3.5-9B agentic_xl draft** — re-run 9B with the bigger corpus as a parallel research contribution
4. **Qwen3.5-4B agentic_xl draft** — re-run 4B with the bigger corpus
5. **Target sidecar calibrations for each of the above** (agentic_xl corpus)

Grid: 5 draft trainings + 4 sidecar cals = 9 jobs across 8 GPUs. First 8 run concurrently, last queues.

## Corpus (staged, ready)

- Recipe: `agentic_xl` → blends `hermes + nemotron_agentic + hermes_filtered + tool_calls_mt + xlam`
- `scripts/fetch_calibration_corpus.sh --recipe agentic_xl /root/agentic_xl_corpus.txt`
- Source breakdown:
  | source | rows | rough token est |
  |---|---|---|
  | lambda/hermes | 14.7k | ~365M (our current) |
  | **nvidia/Nemotron-Agentic-v1** | **335k** | **~32B** |
  | DJLougen/hermes-filtered | 3.7k | ~120M |
  | interstellarninja/tool-calls-multiturn | 1.9k | ~6M |
  | Salesforce/xlam-60k | 60k | ~180M |
- Expected final corpus: ~32-33B tokens flattened to ChatML (~80GB text).
- `max_rows` param caps if we want a smaller mix for first validation.

## Pre-flight

```bash
# 1. quickdeploy cluster (fetches corpus with agentic_xl recipe)
for host in $NEW_HOSTS; do
    scp scripts/amd_quickdeploy.sh $host:/root/
done
# 2. On each host (in parallel):
for host in $NEW_HOSTS; do
    ssh $host 'bash /root/amd_quickdeploy.sh && \
               bash /root/hipfire/scripts/fetch_calibration_corpus.sh \
                   --recipe agentic_xl /root/agentic_xl_corpus.txt' &
done
wait
```

## Training jobs (one per GPU)

Each job uses `HIP_VISIBLE_DEVICES=N` on a single host with 8× visible, OR 1 GPU on separate hosts.

```bash
TRAIN=/root/pytorch_env/bin/python3
ARG() { printf '%s ' "$@"; }
COMMON="$(ARG --corpus /root/agentic_xl_corpus.txt \
              --seq-len 4096 --batch-size 1 --masked-blocks-per-seq 4 \
              --ckpt-every 2500 --log-every 250 \
              --lr 5e-5 --warmup 500 \
              --loss-gamma 3.0 \
              --match-zlab-arch)"

# GPU 0: Qwen3.6-35B-A3B (MoE — grad ckpt target MANDATORY)
HIP_VISIBLE_DEVICES=0 $TRAIN -u scripts/dflash_train_poc.py \
    --target-repo Qwen/Qwen3.6-35B-A3B \
    $COMMON --grad-ckpt-target \
    --steps 25000 \
    --out /root/dflash_36a3b_scratch_xl > /root/36a3b.log 2>&1 &

# GPU 1: Qwen3.5-27B (dense, needs grad ckpt)
HIP_VISIBLE_DEVICES=1 $TRAIN -u scripts/dflash_train_poc.py \
    --target-repo Qwen/Qwen3.5-27B \
    $COMMON --grad-ckpt-target \
    --steps 25000 \
    --out /root/dflash_27b_scratch_xl > /root/27b.log 2>&1 &

# GPU 2: Qwen3.5-9B (bigger corpus; may beat 1× run)
HIP_VISIBLE_DEVICES=2 $TRAIN -u scripts/dflash_train_poc.py \
    --target-repo Qwen/Qwen3.5-9B \
    $COMMON --grad-ckpt-target \
    --steps 25000 \
    --out /root/dflash_9b_scratch_xl > /root/9b_xl.log 2>&1 &

# GPU 3: Qwen3.5-4B (bigger corpus)
HIP_VISIBLE_DEVICES=3 $TRAIN -u scripts/dflash_train_poc.py \
    --target-repo Qwen/Qwen3.5-4B \
    $COMMON \
    --steps 25000 \
    --out /root/dflash_4b_scratch_xl > /root/4b_xl.log 2>&1 &

# GPUs 4-7: sidecar cals on the 4 MQ4 targets
for i in 4 5 6 7; do
    case $i in
        4) tgt=/root/models/qwen3.6-35b-a3b.mq4 ;;
        5) tgt=/root/models/qwen3.5-27b.mq4 ;;
        6) tgt=/root/models/qwen3.5-9b.mq4 ;;
        7) tgt=/root/models/qwen3.5-4b.mq4 ;;
    esac
    HIP_VISIBLE_DEVICES=$i /root/hipfire/target/release/examples/triattn_validate \
        "$tgt" --sidecar "${tgt}.triattn.agentic_xl.bin" \
        --corpus /root/agentic_xl_corpus.txt --max-tokens 5000000 --chunk-len 1024 \
        > /root/sidecar_$i.log 2>&1 &
done
wait
```

## Expected timing (rough)

- 4B @ 25k: ~4hr
- 9B @ 25k: ~4-5hr (same as 1× run)
- 27B @ 25k: ~8-10hr (grad ckpt + larger target)
- 35B-A3B @ 25k: ~6-8hr (MoE activates only routed experts per token)
- Sidecar cals @ 5M tokens: ~30-60 min each

**Longest pole:** 27B draft at ~10hr. Cluster rental: 10hr × 8× @ $48/hr = **~$480 for the full matrix.**

If budget-constrained, drop 27B and it's ~5hr × 8× = $240.

## Evaluation

Post-training, scp all `.hfq` + sidecar files back to local for Rust-engine eval on `.dflash-runs/prompts/hermes_full_system.txt` + agent traces, comparing:
- target alone (AR baseline)
- target + z-lab draft (baseline for 4B/9B/27B/35B where available)
- target + our new agentic_xl draft
- target + our new agentic_xl draft + agentic_xl sidecar

Report τ + mean_committed for each, with confidence intervals over ≥3 diverse prompts.

## Publishable result

If our agentic_xl draft + sidecar beats z-lab's (on agentic specifically) for 9B or 3.6-A3B, that's a domain-specialization paper. We'd publish:
- Drafts as safetensors on HF (`kai-os/Qwen3.5-9B-DFlash-Agentic`, etc.)
- Sidecars as HF models
- A short technical note on the training recipe + data mix + eval

Reference paper: Chen/Liang/Liu, *DFlash: Block Diffusion for Flash Speculative Decoding.* arXiv:2602.06036.
