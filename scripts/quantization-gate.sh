#!/usr/bin/env bash
# Quantization validation/perf gate.
#
# Runs the mandatory layers for KV quantization work:
#   1. synthetic KV parity: asym4+Q8 V reference vs asym4+TQV4/TQV2
#   2. logit/top-5 divergence: fixed prompts, q8 baseline, per-mode CSV/JSON
#   3. prompt-generation smoke: daemon JSONL output + decoded text per mode
#   4. split perf: prefill/decode matrix per mode and context size
#   5. optional coherence gate pass per mode (--coherence)
#
# Default is intentionally small enough for iterative work. Use --full for
# larger prompt/context coverage, and --coherence before claiming quality.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${HIPFIRE_QUANT_MODEL:-$HOME/.hipfire/models/qwen3.5-2b.mq4}"
MODES="${HIPFIRE_QUANT_MODES:-q8,asym4,asym4_tqv4,asym4_tqv2}"
OUT="${HIPFIRE_QUANT_OUT:-benchmarks/results/quantization-$(date +%Y%m%d-%H%M%S)}"
FULL=0
COHERENCE=0
RUNS=1

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --full) FULL=1; shift ;;
        --coherence) COHERENCE=1; shift ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -f "$MODEL" ]; then
    echo "missing model: $MODEL" >&2
    exit 2
fi

mkdir -p "$OUT"/{kernel,logits,prompts,perf,coherence}

LOCK="./scripts/gpu-lock.sh"
if [ -r "$LOCK" ]; then
    # shellcheck disable=SC1090
    . "$LOCK"
    gpu_acquire "quantization-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

IFS=',' read -r -a MODE_ARR <<< "$MODES"
if [ "${#MODE_ARR[@]}" -lt 2 ]; then
    echo "--modes must include at least q8 plus one comparison mode" >&2
    exit 2
fi
BASE_MODE="${MODE_ARR[0]}"

echo "quantization-gate output: $OUT"
{
    echo "# Quantization Gate"
    echo
    echo "- date: $(date -Iseconds)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- model: $MODEL"
    echo "- model_md5: $(md5sum "$MODEL" | awk '{print $1}')"
    echo "- modes: $MODES"
    echo "- full: $FULL"
    echo "- coherence: $COHERENCE"
    echo
} > "$OUT/report.md"

echo "== build examples =="
cargo build --release --features deltanet -p engine \
    --example daemon \
    --example bench_qwen35_mq4 \
    --example greedy_dump_top5 \
    --example kv_quant_parity

for exe in daemon bench_qwen35_mq4 greedy_dump_top5 kv_quant_parity; do
    bin="target/release/examples/$exe"
    if [ -x "$bin" ]; then
        echo "- ${exe}_md5: $(md5sum "$bin" | awk '{print $1}')" >> "$OUT/report.md"
    fi
done
echo >> "$OUT/report.md"

echo "== kernel-level KV parity =="
if [ "$FULL" -eq 1 ]; then
    KV_SEQS="128,512,2048,8192"
    KV_HEADS="128,256"
    LOGIT_STEPS=192
else
    KV_SEQS="128,512"
    KV_HEADS="256"
    LOGIT_STEPS=64
fi
target/release/examples/kv_quant_parity --seq "$KV_SEQS" --head-dim "$KV_HEADS" \
    > "$OUT/kernel/kv_quant_parity.csv" \
    2> "$OUT/kernel/kv_quant_parity.stderr"
{
    echo "## Kernel KV Parity"
    echo
    echo '```csv'
    cat "$OUT/kernel/kv_quant_parity.csv"
    echo '```'
    echo
} >> "$OUT/report.md"

PROMPTS=(
    "cap|inline|What is the capital of France? Answer in one short sentence.|"
    "reason|inline|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|"
    "code|file|benchmarks/prompts/lru_cache_pep8_strict.txt|"
    "tool|file|benchmarks/prompts/tool_call_read_file.txt|benchmarks/prompts/tool_call_system.txt"
)
if [ "$FULL" -eq 1 ]; then
    PROMPTS+=(
        "humaneval0|file|benchmarks/prompts/humaneval_0_has_close_elements.txt|"
        "merge_sort|file|benchmarks/prompts/merge_sort_thinking_off.txt|"
    )
fi

prompt_text() {
    local kind="$1" value="$2"
    if [ "$kind" = "file" ]; then
        cat "$value"
    else
        printf '%s' "$value"
    fi
}

prompt_md5() {
    local kind="$1" value="$2"
    if [ "$kind" = "file" ]; then
        md5sum "$value" | awk '{print $1}'
    else
        printf '%s' "$value" | md5sum | awk '{print $1}'
    fi
}

echo "== logit/top-5 divergence =="
{
    echo "## Logit/Top-5 Divergence"
    echo
} >> "$OUT/report.md"
for entry in "${PROMPTS[@]}"; do
    IFS='|' read -r pid pkind pval _system_file <<< "$entry"
    ptxt="$(prompt_text "$pkind" "$pval")"
    pmd5="$(prompt_md5 "$pkind" "$pval")"
    mkdir -p "$OUT/logits/$pid"
    echo "prompt=$pid md5=$pmd5"
    specs=()
    for mode in "${MODE_ARR[@]}"; do
        prefix="$OUT/logits/$pid/$mode"
        PROMPT_MODE=thinking HIPFIRE_KV_MODE="$mode" \
            target/release/examples/greedy_dump_top5 "$MODEL" "$prefix" --max-gen "$LOGIT_STEPS" "$ptxt" \
            > "$prefix.stdout" 2> "$prefix.stderr"
        if [ "$mode" != "$BASE_MODE" ]; then
            specs+=("$mode=$prefix")
        fi
    done
    ./scripts/quant_compare_top5.py "$OUT/logits/$pid/$BASE_MODE" "${specs[@]}" \
        > "$OUT/logits/$pid/compare.json"
    {
        echo "### $pid"
        echo
        echo "- prompt_md5: $pmd5"
        echo
        echo '```json'
        cat "$OUT/logits/$pid/compare.json"
        echo '```'
        echo
    } >> "$OUT/report.md"
done

echo "== prompt-generation smoke =="
{
    echo "## Prompt Smoke"
    echo
} >> "$OUT/report.md"
for entry in "${PROMPTS[@]}"; do
    IFS='|' read -r pid pkind pval system_file <<< "$entry"
    ptxt="$(prompt_text "$pkind" "$pval")"
    pmd5="$(prompt_md5 "$pkind" "$pval")"
    pjson="$(python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' <<< "$ptxt")"
    system_json=""
    if [ -n "${system_file:-}" ]; then
        smd5="$(md5sum "$system_file" | awk '{print $1}')"
        sjson="$(python3 -c 'import json,sys; print(json.dumps(open(sys.argv[1]).read()))' "$system_file")"
        system_json=",\"system\":$sjson"
    else
        smd5=""
    fi
    for mode in "${MODE_ARR[@]}"; do
        in_file="$OUT/prompts/${pid}.${mode}.jsonl"
        out_file="$OUT/prompts/${pid}.${mode}.jsonl.out"
        cat > "$in_file" <<JSONL
{"type":"load","model":"$MODEL","params":{"max_seq":4096,"kv_mode":"$mode","dflash_mode":"off"}}
{"type":"generate","id":"$pid-$mode","prompt":$pjson,"temperature":0.0,"max_tokens":96,"repeat_penalty":1.05,"top_p":1.0,"max_think_tokens":1$system_json}
{"type":"unload"}
JSONL
        HIPFIRE_DFLASH_DRAFT= timeout 240 target/release/examples/daemon \
            < "$in_file" > "$out_file" 2> "$OUT/prompts/${pid}.${mode}.stderr"
        python3 - "$out_file" > "$OUT/prompts/${pid}.${mode}.txt" <<'PY'
import json, sys
print("".join(json.loads(l).get("text", "") for l in open(sys.argv[1]) if '"type":"token"' in l))
PY
        done_line="$(python3 - "$out_file" <<'PY'
import sys
for line in open(sys.argv[1]):
    if '"type":"done"' in line:
        print(line.strip())
        break
PY
)"
        {
            echo "### $pid / $mode"
            echo
            echo "- prompt_md5: $pmd5"
            if [ -n "$smd5" ]; then echo "- system_md5: $smd5"; fi
            echo "- stats: \`${done_line:-missing done line}\`"
            echo
            echo '```'
            cat "$OUT/prompts/${pid}.${mode}.txt"
            echo '```'
            echo
        } >> "$OUT/report.md"
    done
done

echo "== split perf =="
if [ "$FULL" -eq 1 ]; then
    PREFILLS=(512 2048 8192 32768)
    GEN=128
else
    PREFILLS=(512 2048)
    GEN=64
fi
echo "mode,prefill,run,prefill_tok_s,decode_tok_s,summary" > "$OUT/perf/perf.csv"
for mode in "${MODE_ARR[@]}"; do
    for pp in "${PREFILLS[@]}"; do
        for run in $(seq 1 "$RUNS"); do
            log="$OUT/perf/${mode}.pp${pp}.run${run}.log"
            HIPFIRE_KV_MODE="$mode" HIPFIRE_GRAPH=1 HIPFIRE_DPM_WARMUP_SECS=5 \
                target/release/examples/bench_qwen35_mq4 "$MODEL" \
                --prefill "$pp" --prefill-runs 2 --warmup 5 --gen "$GEN" \
                > "$log" 2>&1
            python3 - "$mode" "$pp" "$run" "$log" >> "$OUT/perf/perf.csv" <<'PY'
import re, sys
mode, pp, run, path = sys.argv[1:]
text = open(path, errors="ignore").read()
summary = ""
for line in text.splitlines():
    if line.startswith("SUMMARY"):
        summary = line
pref = re.search(r"prefill_tok_s=([0-9.]+)", summary)
dec = re.search(r"gen_tok_s=([0-9.]+)", summary)
print(f"{mode},{pp},{run},{pref.group(1) if pref else ''},{dec.group(1) if dec else ''},{summary!r}")
PY
        done
    done
done
{
    echo "## Split Perf"
    echo
    echo '```csv'
    cat "$OUT/perf/perf.csv"
    echo '```'
    echo
} >> "$OUT/report.md"

if [ "$COHERENCE" -eq 1 ]; then
    echo "== coherence =="
    {
        echo "## Coherence"
        echo
    } >> "$OUT/report.md"
    for mode in "${MODE_ARR[@]}"; do
        coh_out="$OUT/coherence/${mode}.md"
        HIPFIRE_KV_MODE="$mode" HIPFIRE_DFLASH_DRAFT= HIPFIRE_COHERENCE_OUT="$coh_out" \
            ./scripts/coherence-gate.sh
        echo "- $mode: $coh_out" >> "$OUT/report.md"
    done
    echo >> "$OUT/report.md"
fi

echo "report: $OUT/report.md"
