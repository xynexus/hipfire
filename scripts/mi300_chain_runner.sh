#!/usr/bin/env bash
# mi300_chain_runner.sh — sequential auto-chain for MI300X training jobs.
#
# Runs jobs back-to-back so the GPU isn't idle between training runs.
# Each job block:
#   - logs to /root/chain_logs/${name}.log
#   - writes status marker /root/chain_status/${name}.{started,done,failed}
#   - on failure, the chain stops (next jobs NOT started)
#
# Usage (on MI300X):
#   nohup bash /root/hipfire/scripts/mi300_chain_runner.sh \
#     > /root/chain_runner.log 2>&1 &
#
# Set SKIP_UNTIL=<name> to resume after a manual kill (skip completed jobs).
#
# After editing this file on the controller, just scp/git-pull + start fresh.

set -uo pipefail
cd /root/hipfire

LOG_DIR=/root/chain_logs
STATUS_DIR=/root/chain_status
mkdir -p "$LOG_DIR" "$STATUS_DIR"

PY=/root/pytorch_env/bin/python3
CORPUS=/root/agentic_corpus.txt
SKIP_UNTIL="${SKIP_UNTIL:-}"

SKIPPING=1
[ -z "$SKIP_UNTIL" ] && SKIPPING=0

# ── jobs ─────────────────────────────────────────────────────────────
# Each job function should:
#   - return 0 on success, non-zero on failure
#   - print [chain-runner] markers for progress

run_if_pending() {
    local name=$1; shift
    if [ "$SKIPPING" = "1" ]; then
        if [ "$name" = "$SKIP_UNTIL" ]; then
            SKIPPING=0
            echo "[chain-runner] resuming at: $name"
        else
            echo "[chain-runner] skipping: $name (--skip-until=$SKIP_UNTIL)"
            return 0
        fi
    fi
    if [ -f "$STATUS_DIR/$name.done" ]; then
        echo "[chain-runner] already done: $name"
        return 0
    fi
    echo "[chain-runner] starting: $name"
    date -Is > "$STATUS_DIR/$name.started"
    rm -f "$STATUS_DIR/$name.failed"
    "$@" > "$LOG_DIR/$name.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        date -Is > "$STATUS_DIR/$name.done"
        echo "[chain-runner] finished: $name (rc=0)"
    else
        date -Is > "$STATUS_DIR/$name.failed"
        echo "[chain-runner] FAILED:   $name (rc=$rc)"
        echo "[chain-runner] tail of $LOG_DIR/$name.log:"
        tail -20 "$LOG_DIR/$name.log"
        return $rc
    fi
}

# ── job definitions ──────────────────────────────────────────────────

job_4b_scratch_50k() {
    PYTHONUNBUFFERED=1 "$PY" -u scripts/dflash_train_poc.py \
        --target-repo Qwen/Qwen3.5-4B \
        --corpus "$CORPUS" \
        --seq-len 4096 --batch-size 1 --masked-blocks-per-seq 4 \
        --steps 50000 --ckpt-every 2500 --log-every 250 \
        --lr 5e-5 --warmup 500 \
        --loss-gamma 3.0 \
        --match-zlab-arch \
        --out /root/dflash_4b_scratch_50k
}

job_4b_scratch_convert() {
    # Convert final safetensors → .hfq (caller pulls + sidecars locally).
    ./target/release/dflash_convert \
        --input /root/dflash_4b_scratch_50k \
        --output /root/dflash_4b_scratch_50k.hfq \
        --mq4
}

job_9b_scratch_50k() {
    PYTHONUNBUFFERED=1 "$PY" -u scripts/dflash_train_poc.py \
        --target-repo Qwen/Qwen3.5-9B \
        --corpus "$CORPUS" \
        --seq-len 4096 --batch-size 1 --masked-blocks-per-seq 4 \
        --steps 50000 --ckpt-every 2500 --log-every 250 \
        --lr 5e-5 --warmup 500 \
        --loss-gamma 3.0 \
        --match-zlab-arch \
        --grad-ckpt-target \
        --out /root/dflash_9b_scratch_50k
}

job_9b_scratch_convert() {
    ./target/release/dflash_convert \
        --input /root/dflash_9b_scratch_50k \
        --output /root/dflash_9b_scratch_50k.hfq \
        --mq4
}

# ── sidecar cal jobs (gated on draft success — require MQ4 target to exist) ──

sidecar_cal() {
    local tgt=$1 sc_out=$2
    /root/hipfire/target/release/examples/triattn_validate \
        --model "$tgt" \
        --corpus "$CORPUS" \
        --out "$sc_out" \
        --max-tokens 1000000 \
        --chunk-len 1024
}

job_4b_sidecar_cal() {
    # Uses the existing MQ4 target (not the new draft — sidecars are for
    # the TARGET's attention, not the draft). Produces
    # qwen3.5-4b.mq4.triattn.bin which pairs with any 4B draft.
    local tgt=/root/models/qwen3.5-4b.mq4
    [ -f "$tgt" ] || { echo "no target at $tgt — stage with stage_models.sh first" >&2; return 3; }
    sidecar_cal "$tgt" "${tgt}.triattn.agentic.bin"
}

job_9b_sidecar_cal() {
    local tgt=/root/models/qwen3.5-9b.mq4
    [ -f "$tgt" ] || { echo "no target at $tgt — stage with stage_models.sh first" >&2; return 3; }
    sidecar_cal "$tgt" "${tgt}.triattn.agentic.bin"
}

# ── main chain ───────────────────────────────────────────────────────

echo "[chain-runner] starting chain at $(date -Is)"
echo "[chain-runner] skip_until='${SKIP_UNTIL:-<none>}'"

run_if_pending 4b_scratch_50k           job_4b_scratch_50k           || exit 1
run_if_pending 4b_scratch_convert       job_4b_scratch_convert       || exit 1
run_if_pending 4b_sidecar_cal           job_4b_sidecar_cal           || exit 1
run_if_pending 9b_scratch_50k           job_9b_scratch_50k           || exit 1
run_if_pending 9b_scratch_convert       job_9b_scratch_convert       || exit 1
run_if_pending 9b_sidecar_cal           job_9b_sidecar_cal           || exit 1

echo "[chain-runner] ALL JOBS DONE at $(date -Is)"
