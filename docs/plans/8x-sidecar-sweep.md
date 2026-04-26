# 8× MI300X Sidecar Sweep Plan

Status: **DEFERRED** (2026-04-19). Superseded by domain-trained DFlash draft
training. Resurrect when draft training is de-risked and we want to refresh
all sidecars with recipe-matched calibration.

**Scope clarification (2026-04-25):** This doc is sidecar calibration
*only* (producing `.triattn.bin`). For DFlash **draft model training**
(producing `.hfq` weights) and its corpus discussion — including
`Jackrong/Qwen3.5-reasoning-700x` and the broader prompt mix — see
`docs/plans/task-93-path-c-trained-draft.prd`. Both flows feed off
`scripts/fetch_calibration_corpus.sh` recipes but produce different
artifacts.

## Context

Today we proved two things from single-GPU tests on the current MI300X:

1. **Domain-calibrated TriAttention sidecars dramatically outperform the
   generic (wikitext-trained) default** on agentic deployment. On a kimi
   hermes tool-call trace, carnice-9b + hermes-cal sidecar produced a
   clean `<tool_call>...<|im_end|>`; the wikitext-cal sidecar degenerated
   into `<function=write>` loops.

2. **Sidecar calibration parallelizes trivially across GPUs** because each
   run is a single-process job. `HIP_VISIBLE_DEVICES=N` per job and
   queue-backfill wall-clocks 8-10 jobs in a single run.

## Deployment chain (ready to fire)

Committed to `dflash` branch in `scripts/`:

```bash
scp scripts/amd_quickdeploy.sh NEW_HOST:/root/
ssh NEW_HOST 'bash /root/amd_quickdeploy.sh --fetch-corpus'
ssh NEW_HOST 'bash /root/hipfire/scripts/stage_models.sh'
ssh NEW_HOST 'bash /root/hipfire/scripts/calibrate_multigpu.sh \
    --models $(ls /root/models/*.mq[46] | tr "\n" ",") \
    --corpus /root/calibration_corpus.txt'
rsync -avP NEW_HOST:/root/models/*.triattn.bin ~/.hipfire/models/
```

## Scripts shipped

- `scripts/amd_quickdeploy.sh` — pod bring-up, bakes
  `HIPFIRE_ROCBLAS_OFF=1` + ROCm PATH into `/root/.bashrc`, counts visible
  GPUs, prints right next-step command.
- `scripts/fetch_calibration_corpus.sh` — recipe-driven corpus builder
  (`agentic | reasoning | chat | blended | all`). Pulls from 5 HF
  datasets, flattens to ChatML-wrapped text.
- `scripts/stage_models.sh` — pulls safetensors for the full model matrix
  and quantizes on-pod (Qwen3.5-{4B,9B,27B} × mq4+mq6, Qwen3.5/3.6-A3B
  × mq4, kai-os/Carnice-{9B,27B} × mq4+mq6). MoE mq6 is a quantizer TODO
  (main.rs:1318 hard-codes MQ4 for routed experts).
- `scripts/calibrate_multigpu.sh` — fans N calibration jobs across N
  GPUs via `HIP_VISIBLE_DEVICES`, queue + backfill when more models than
  GPUs. Captures per-model logs at `/root/calib_logs/`.

## Per-model recipe recommendation

Based on today's empirical finding (domain-match > diversity for known
deployment):

| model | recipe | rationale |
|-------|--------|-----------|
| carnice-{9B,27B}.{mq4,mq6} | agentic | Hermes-tuned, deployed as agent |
| qwen3.5-35b-a3b.mq4 | agentic | Primary agent target |
| **qwen3.6-35b-a3b.mq4** | **agentic** | **Main agent target** |
| qwen3.5-9b.mq4 | blended | General-purpose default |
| qwen3.5-27b.mq4 | blended | General-purpose default |
| qwen3.5-4b.mq4 | blended | General-purpose default |

`calibrate_multigpu.sh` doesn't currently take per-model recipe flags
— that's a future addition. For now: run two waves (agentic corpus,
then blended corpus) with different `--corpus` each time.

## Cost / time estimates

- 8× MI300X rental at ~$48/hr total
- Pod boot → calib corpus fetched: ~5 min
- `stage_models.sh` (quantize 12 artifacts on pod): ~15 min
- Carnice scp (if not using kai-os HF pull): ~5 min
- `calibrate_multigpu.sh` wall-clock: ~30-40 min (all models one wave)
- **Total: ~1 hour, ~$50** for the full matrix

vs sequential 1× = ~3-4 hours for the same matrix, ~$20-25 at $2/hr.
The 8× premium only pencils if you're doing multiple parallel jobs
(training + cal + SFT) in the same rental window.

## Known gaps (to fix before running)

- [x] `calibrate_multigpu.sh` doesn't take `--recipe`. **FIXED 2026-04-25**:
      script now accepts `--recipe NAME` and auto-builds the corpus via
      `fetch_calibration_corpus.sh`, caching at
      `/root/calib_corpus_<NAME>.txt`. Per-model recipe mapping (different
      recipe per model in one call) still requires multiple waves.
- [ ] MoE MQ6 support in `hipfire-quantize` (main.rs:1318). Affects
      A3B mq6 artifacts. Skip for now.
- [ ] `stage_models.sh` quantizes even if artifact exists but not if it
      exists at a different path. Add `--force` flag if we want
      overwrites.
- [ ] Validation after calibration: no automated "did the sidecar work"
      check. Manually bench against a held-out eval prompt before shipping.

## Re-ship trigger

Fire this plan when:
1. DFlash draft training is validated and domain-trained drafts exist.
2. We want to re-calibrate sidecars against the NEW drafts (if the
   draft's KV attention patterns meaningfully diverge from the
   wikitext-trained defaults).
3. We decide to publish sidecars alongside drafts on HF.

## Related

- `scripts/calibrate_multigpu.sh` (queue logic is reusable for draft
  training if we make it generic).
- `scripts/dflash_train_poc.py` (the draft trainer — priority #1 now).
- `.dflash-reference/dflash/model.py` (reference architecture).
