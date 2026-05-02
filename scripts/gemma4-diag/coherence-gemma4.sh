#!/usr/bin/env bash
# Gemma 4 slice of the coherence battery — same daemon JSONL flow as
# scripts/coherence-gate.sh but with the four Gemma 4 dense + MoE models
# we currently have quantized at MG4. Mirrors the gate's hard-error
# criteria (exit≠0, zero tokens, panic) and prints per-model output for
# eyeball review.
set -u
cd "$(dirname "$0")/../.."

EXE="./target/release/examples/daemon"
[ -x "$EXE" ] || { echo "missing $EXE — build with cargo build --release --features deltanet --example daemon -p engine" >&2; exit 2; }

OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-gemma4-$(date +%Y%m%d-%H%M%S).md}"

# GPU lock — match coherence-gate.sh. Block until the GPU is free, then
# release on exit. Skipping this races whatever else is on the card and
# typically manifests as 0 tokens emitted (daemon load partially fails,
# generate returns immediately).
LOCK_SCRIPT="./scripts/gpu-lock.sh"
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gemma4" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# (model_path|id|prompt|max_tok)
TESTS=(
    "$HOME/.hipfire/models/gemma-4-e2b/gemma-4-e2b.mg4.hfq|e2b-cap|What is the capital of France? Answer in one short sentence.|80"
    "$HOME/.hipfire/models/gemma-4-e4b/gemma-4-e4b.mg4.hfq|e4b-cap|What is the capital of France? Answer in one short sentence.|80"
    "$HOME/.hipfire/models/gemma-4-31b/gemma-4-31b.mg4.hfq|31b-cap|What is the capital of France? Answer in one short sentence.|80"
    "$HOME/.hipfire/models/gemma-4-31b/gemma-4-31b.mg4.hfq|31b-reason|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
    "$HOME/.hipfire/models/gemma-4-26b-a4b/gemma-4-26b-a4b.mg4.hfq|26b-a4b-cap|What is the capital of France? Answer in one short sentence.|80"
    "$HOME/.hipfire/models/gemma-4-26b-a4b/gemma-4-26b-a4b.mg4.hfq|26b-a4b-reason|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
)

{
    echo "# Gemma 4 coherence battery"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- date: $(date -Iseconds)"
    echo
} > "$OUT"

hard_errors=0
for entry in "${TESTS[@]}"; do
    IFS='|' read -r model_path prompt_id prompt max_tok <<< "$entry"
    if [ ! -f "$model_path" ]; then
        echo "## $prompt_id — SKIPPED (model not present: $model_path)" >> "$OUT"; echo >> "$OUT"
        continue
    fi
    echo "== $prompt_id ($model_path) =="
    in_file="/tmp/cohg4_in_$$.jsonl"
    out_file="/tmp/cohg4_out_$$.log"
    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")
    cat > "$in_file" <<JL
{"type":"load","model":"$model_path","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":$max_tok,"repeat_penalty":1.05}
{"type":"unload"}
JL
    t0=$(date +%s.%N)
    timeout 600 "$EXE" < "$in_file" > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    n_tokens=$(grep -ac '"type":"token"' "$out_file")
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$out_file" | head -1)
    status="OK"
    if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ]; then
        status="HARD_ERROR (exit=$ec tokens=$n_tokens panic=${panic:+yes})"
        hard_errors=$((hard_errors + 1))
    fi
    text=$(python3 -c "
import sys, json
out=[]
for line in open(sys.argv[1]):
    line=line.strip()
    if not line.startswith('{'): continue
    try: m=json.loads(line)
    except Exception: continue
    if m.get('type')=='token' and 'text' in m: out.append(m['text'])
print(''.join(out))
" "$out_file")

    {
        echo "## $prompt_id ($model_path)"
        echo
        echo "- status: $status"
        echo "- tokens: $n_tokens"
        echo "- wall:   ${wall}s"
        echo
        echo "### prompt"
        echo
        echo '```'
        echo "$prompt"
        echo '```'
        echo
        echo "### output"
        echo
        echo '```'
        echo "$text"
        echo '```'
        echo
        if [ -n "$panic" ]; then
            echo "### panic line"; echo; echo '```'; echo "$panic"; echo '```'; echo
        fi
    } >> "$OUT"
    rm -f "$in_file" "$out_file"
done

echo
echo "wrote $OUT"
echo "hard_errors=$hard_errors"
exit $hard_errors
