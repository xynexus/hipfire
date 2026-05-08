#!/usr/bin/env bash
# PFlash + DFlash hetero compose bench (TB5 drafter).
#
# Spawns the daemon, loads target + DFlash drafter (pinned to a separate
# device via HIPFIRE_DFLASH_DRAFTER_DEVICE) + PFlash drafter (on target),
# runs a long-context NIAH prompt, captures prefill/decode/τ + PFlash
# kept_tokens. Validates the two levers compose without crashing and
# without τ drift vs hetero-DFlash-only.
#
# Usage: ./pflash_dflash_compose.sh <config>
# Configs:
#   solo        — DFlash drafter on same device as target (HIPFIRE_DFLASH_DRAFTER_DEVICE unset)
#   hetero      — DFlash drafter on TB5 (HIPFIRE_DFLASH_DRAFTER_DEVICE=1)
#   hetero+pf   — hetero + PFlash always-on with 0.8b drafter
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-hetero+pf}"
TARGET="$HOME/.hipfire/models/qwen3.5-27b.mq4"
DFLASH_DRAFT="$HOME/.hipfire/models/qwen35-27b-dflash-mq4.hfq"
PFLASH_DRAFT="$HOME/.hipfire/models/qwen3.5-0.8b.mq4"
PROMPT_FILE="benchmarks/longctx/niah/niah_16k.jsonl"

# Single-line filler+needle+question prompt extracted from NIAH JSONL line 1.
PROMPT=$(python3 -c "
import json
with open('$PROMPT_FILE') as f:
    rec = json.loads(f.readline())
print(rec['filler_text'] + '\n\n' + rec['question'], end='')
")
EXPECTED=$(python3 -c "
import json
with open('$PROMPT_FILE') as f:
    rec = json.loads(f.readline())
print(rec.get('expected_answer_substring', rec.get('needle', '')), end='')
")

case "$CONFIG" in
    solo)
        unset HIPFIRE_DFLASH_DRAFTER_DEVICE 2>/dev/null || true
        LOAD_PARAMS=$(jq -nc --arg draft "$DFLASH_DRAFT" \
            '{max_seq:18000,draft:$draft,kv_mode:"asym3"}')
        ;;
    hetero)
        export HIPFIRE_DFLASH_DRAFTER_DEVICE=1
        LOAD_PARAMS=$(jq -nc --arg draft "$DFLASH_DRAFT" \
            '{max_seq:18000,draft:$draft,kv_mode:"asym3"}')
        ;;
    hetero+pf)
        export HIPFIRE_DFLASH_DRAFTER_DEVICE=1
        LOAD_PARAMS=$(jq -nc --arg draft "$DFLASH_DRAFT" --arg pfd "$PFLASH_DRAFT" \
            '{max_seq:18000,draft:$draft,kv_mode:"asym3",
              prefill_compression:"always",
              prefill_drafter:$pfd,
              prefill_threshold:4096,
              prefill_keep_ratio:0.5,
              prefill_min_keep:1024,
              prefill_profile:true}')
        ;;
    ar)
        # Plain AR on full prompt — no DFlash, no PFlash. Decode rate
        # baseline for the long-context workload.
        unset HIPFIRE_DFLASH_DRAFTER_DEVICE 2>/dev/null || true
        LOAD_PARAMS=$(jq -nc '{max_seq:18000,kv_mode:"asym3"}')
        ;;
    ar+pf)
        # AR + PFlash: prompt compressed, AR decode on smaller KV.
        # Compares directly to hetero+pf to show whether spec-decode
        # actually wins on this workload.
        unset HIPFIRE_DFLASH_DRAFTER_DEVICE 2>/dev/null || true
        LOAD_PARAMS=$(jq -nc --arg pfd "$PFLASH_DRAFT" \
            '{max_seq:18000,kv_mode:"asym3",
              prefill_compression:"always",
              prefill_drafter:$pfd,
              prefill_threshold:4096,
              prefill_keep_ratio:0.5,
              prefill_min_keep:1024,
              prefill_profile:true}')
        ;;
    *)
        echo "unknown config: $CONFIG" >&2
        exit 1
        ;;
esac

LOAD_JSON=$(jq -nc --arg model "$TARGET" --argjson params "$LOAD_PARAMS" \
    '{type:"load",model:$model,params:$params}')
GEN_JSON=$(jq -nc --arg prompt "$PROMPT" \
    '{type:"generate",id:"r1",prompt:$prompt,temperature:0.0,max_tokens:60,raw:true}')

echo "=== config: $CONFIG ===" >&2
echo "=== HIPFIRE_DFLASH_DRAFTER_DEVICE: ${HIPFIRE_DFLASH_DRAFTER_DEVICE:-unset} ===" >&2
echo "=== prompt size (chars): ${#PROMPT} ===" >&2
echo "=== expected needle: $EXPECTED ===" >&2

# Daemon needs ROCm path
export PATH="/opt/rocm-7.2.2/bin:$PATH"
export HIPFIRE_NORMALIZE_PROMPT=1

LOG=$(mktemp /tmp/pflash_compose_${CONFIG}_XXXXXX.log)
echo "=== log: $LOG ===" >&2

# Send load → wait loaded → send generate → read until done.
{
    echo "$LOAD_JSON"
    # Wait for target load (~30s for 27B) + DFlash drafter (~1s) + PFlash
    # drafter (~30s with first-run JIT compile of gemm_hfq4g256_residual_mmq).
    # Extend on cold cache; warm cache cuts to ~5s.
    sleep 90
    echo "$GEN_JSON"
    sleep 120
    echo '{"type":"unload"}'
    sleep 2
} | timeout 360 ./target/release/examples/daemon 2>"$LOG.stderr" | tee "$LOG" | \
    while IFS= read -r line; do
        case "$line" in
            *'"type":"loaded"'*)    echo "[LOADED]    $line" >&2 ;;
            *'"type":"pflash"'*)    echo "[PFLASH-CFG] $line" >&2 ;;
            *'"type":"pflash_load_failed"'*) echo "[PFLASH-FAIL] $line" >&2 ;;
            *'"type":"error"'*)     echo "[ERROR]     $line" >&2 ;;
            *'"type":"done"'*)      echo "[DONE]      $line" >&2 ;;
            *'"type":"unloaded"'*)  echo "[UNLOADED]  $line" >&2 ;;
            *) ;;  # tokens — silent
        esac
    done

echo
echo "=== summary ($CONFIG) ===" >&2
grep -E '"type":"done"|"type":"pflash"' "$LOG" | jq -c . 2>/dev/null || cat "$LOG" | grep -E "done|pflash"
