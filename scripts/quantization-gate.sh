#!/usr/bin/env bash
# Quantization validation/perf gate.
#
# Runs the mandatory layers for KV quantization work:
#   1. synthetic KV parity: asym4+Q8 V reference vs asym4+TQV4/TQV2
#   2. logit/top-5 divergence: fixed prompts, fp32 baseline, per-mode CSV/JSON
#   3. prompt-generation smoke: daemon JSONL output + decoded text per mode
#   4. split perf: prefill/decode matrix per mode and context size
#   5. optional coherence gate pass per mode (--coherence)
#
# Default is intentionally small enough for iterative work. Use --full for
# larger prompt/context coverage, and --coherence before claiming quality.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${HIPFIRE_QUANT_MODEL:-${XDG_DATA_HOME:-$HOME/.local/share}/hipfire/models/qwen3.5-2b.mq4}"
MODES="${HIPFIRE_QUANT_MODES:-fp32,q8,asym4,asym3,asym2,asym4_tqv4,asym4_tqv3,asym4_tqv2,asym4_tqv1}"
OUT="${HIPFIRE_QUANT_OUT:-benchmarks/results/quantization-$(date +%Y%m%d-%H%M%S)}"
FULL=0
COHERENCE=0
RUNS=1
STRICT=0
SKIP_BUILD=0
MODEL_NAME="${HIPFIRE_QUANT_MODEL_NAME:-}"
CHECK_TOOL_JSON="${HIPFIRE_QUANT_CHECK_TOOL_JSON:-warn}" # off|warn|strict
TQV4_COS_FLOOR="${HIPFIRE_TQV4_COS_FLOOR:-0.995}"
TQV3_COS_FLOOR="${HIPFIRE_TQV3_COS_FLOOR:-0.990}"
TQV2_COS_FLOOR="${HIPFIRE_TQV2_COS_FLOOR:-0.985}"
TOP1_WARN_FLOOR="${HIPFIRE_QUANT_TOP1_WARN_FLOOR:-0.70}"
TOP5_WARN_FLOOR="${HIPFIRE_QUANT_TOP5_WARN_FLOOR:-2.50}"
WARNINGS=()
ERRORS=()

warn() {
    WARNINGS+=("$*")
    echo "warning: $*" >&2
}

hard_error() {
    ERRORS+=("$*")
    echo "error: $*" >&2
}

strict_or_warn() {
    if [ "$STRICT" -eq 1 ]; then
        hard_error "$*"
    else
        warn "$*"
    fi
}

collect_diag() {
    local kind="$1" msg="$2"
    case "$kind" in
        WARN) warn "$msg" ;;
        ERROR) hard_error "$msg" ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --check-tool-json) CHECK_TOOL_JSON="$2"; shift 2 ;;
        --full) FULL=1; shift ;;
        --coherence) COHERENCE=1; shift ;;
        --strict) STRICT=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        -h|--help)
            sed -n '2,28p' "$0"
            echo
            echo "Options: --model PATH --modes CSV --out DIR --runs N --full --coherence --strict --skip-build"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -f "$MODEL" ]; then
    echo "missing model: $MODEL" >&2
    exit 2
fi
if [[ "$CHECK_TOOL_JSON" != "off" && "$CHECK_TOOL_JSON" != "warn" && "$CHECK_TOOL_JSON" != "strict" ]]; then
    echo "--check-tool-json must be off, warn, or strict" >&2
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
    echo "--modes must include a baseline plus at least one comparison mode" >&2
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
    if [ -n "$MODEL_NAME" ]; then echo "- model_name: $MODEL_NAME"; fi
    echo "- model_md5: $(md5sum "$MODEL" | awk '{print $1}')"
    echo "- model_size_bytes: $(stat -c%s "$MODEL" 2>/dev/null || wc -c < "$MODEL")"
    echo "- modes: $MODES"
    echo "- full: $FULL"
    echo "- coherence: $COHERENCE"
    echo "- strict: $STRICT"
    echo "- check_tool_json: $CHECK_TOOL_JSON"
    echo "- thresholds:"
    echo "  - tqv4_cos_floor: $TQV4_COS_FLOOR"
    echo "  - tqv3_cos_floor: $TQV3_COS_FLOOR"
    echo "  - tqv2_cos_floor: $TQV2_COS_FLOOR"
    echo "  - top1_warn_floor: $TOP1_WARN_FLOOR"
    echo "  - top5_warn_floor: $TOP5_WARN_FLOOR"
    echo
} > "$OUT/report.md"

if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "== build examples =="
    cargo build --release --features deltanet -p engine \
        --example daemon \
        --example bench_qwen35_mq4 \
        --example greedy_dump_top5 \
        --example kv_quant_parity
else
    echo "== build examples: skipped =="
fi

for exe in daemon bench_qwen35_mq4 greedy_dump_top5 kv_quant_parity; do
    bin="target/release/examples/$exe"
    if [ -x "$bin" ]; then
        echo "- ${exe}_md5: $(md5sum "$bin" | awk '{print $1}')" >> "$OUT/report.md"
    else
        hard_error "missing built example: $bin"
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
while IFS='|' read -r kind msg; do
    collect_diag "$kind" "$msg"
done < <(python3 - "$OUT/kernel/kv_quant_parity.csv" "$TQV4_COS_FLOOR" "$TQV3_COS_FLOOR" "$TQV2_COS_FLOOR" <<'PY'
import csv
import math
import sys

path = sys.argv[1]
tqv4_floor, tqv3_floor, tqv2_floor = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
floors = {"asym4_tqv4": tqv4_floor, "asym4_tqv3": tqv3_floor, "asym4_tqv2": tqv2_floor}
rows = list(csv.DictReader(open(path, newline="")))
if not rows:
    print("ERROR|kernel parity produced no rows")
for row in rows:
    label = f"kernel parity {row.get('mode')} head_dim={row.get('head_dim')} seq={row.get('seq')}"
    for key in ("max_abs", "mean_abs", "cosine", "write_ms_per_token"):
        try:
            value = float(row[key])
        except Exception:
            print(f"ERROR|{label}: missing numeric {key}")
            continue
        if not math.isfinite(value):
            print(f"ERROR|{label}: nonfinite {key}={value}")
    mode = row.get("mode", "")
    if mode in floors:
        try:
            cosine = float(row["cosine"])
        except Exception:
            continue
        if cosine < floors[mode]:
            print(f"ERROR|{label}: cosine {cosine:.6f} below floor {floors[mode]:.6f}")
PY
)
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
    "largest_island|inline|What is the largest island? Answer with only one word.|"
    "red_planet|inline|Which planet is known as the Red Planet? Answer with one word.|"
    "author_1984|inline|Who wrote 1984? Answer with only the surname.|"
    "boiling_water|inline|At sea level, water boils at what temperature in Celsius? Answer with only the number.|"
    "speed_light|inline|What is the speed of light in vacuum, rounded to the nearest thousand km/s? Answer with only the number.|"
    "square_root|inline|What is the square root of 144? Answer with only the number.|"
    "chemical_water|inline|What is the chemical formula for water? Answer with only the formula.|"
    "largest_ocean|inline|What is the largest ocean on Earth? Answer with one word.|"
    "currency_japan|inline|What is the currency of Japan? Answer with one word.|"
    "primary_blue_yellow|inline|What color do blue and yellow paint make when mixed? Answer with one word.|"
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
    IFS='|' read -r pid pkind pval system_file <<< "$entry"
    ptxt="$(prompt_text "$pkind" "$pval")"
    pmd5="$(prompt_md5 "$pkind" "$pval")"
    system_args=()
    if [ -n "${system_file:-}" ]; then
        system_args=(--system "$(cat "$system_file")")
    fi
    mkdir -p "$OUT/logits/$pid"
    printf '%s' "$ptxt" > "$OUT/prompts/${pid}.prompt.txt"
    if [ -n "${system_file:-}" ]; then
        cat "$system_file" > "$OUT/prompts/${pid}.system.txt"
    fi
    echo "prompt=$pid md5=$pmd5"
    specs=()
    for mode in "${MODE_ARR[@]}"; do
        prefix="$OUT/logits/$pid/$mode"
        PROMPT_MODE=thinking HIPFIRE_KV_MODE="$mode" \
            target/release/examples/greedy_dump_top5 "$MODEL" "$prefix" --max-gen "$LOGIT_STEPS" "${system_args[@]}" "$ptxt" \
            > "$prefix.stdout" 2> "$prefix.stderr"
        if [ "$mode" != "$BASE_MODE" ]; then
            specs+=("$mode=$prefix")
        fi
    done
    ./scripts/quant_compare_top5.py "$OUT/logits/$pid/$BASE_MODE" "${specs[@]}" \
        > "$OUT/logits/$pid/compare.json"
    while IFS='|' read -r kind msg; do
        collect_diag "$kind" "$msg"
    done < <(python3 - "$OUT/logits/$pid/compare.json" "$pid" "$TOP1_WARN_FLOOR" "$TOP5_WARN_FLOOR" "$STRICT" <<'PY'
import json
import math
import sys

path, pid = sys.argv[1], sys.argv[2]
top1_floor, top5_floor = float(sys.argv[3]), float(sys.argv[4])
strict = sys.argv[5] == "1"
rows = json.load(open(path))
if not rows:
    print(f"ERROR|{pid}: top5 comparison produced no rows")
for row in rows:
    mode = row["mode"]
    top1 = float(row["top1_agreement"])
    overlap = float(row["mean_top5_overlap"])
    steps = int(row["steps_compared"])
    prefix = "ERROR" if strict else "WARN"
    if steps <= 0 or not math.isfinite(top1) or not math.isfinite(overlap):
        print(f"ERROR|{pid}/{mode}: invalid comparison metrics steps={steps} top1={top1} overlap={overlap}")
        continue
    if top1 < top1_floor:
        print(f"{prefix}|{pid}/{mode}: top1 agreement {top1:.3f} below warn floor {top1_floor:.3f}")
    if overlap < top5_floor:
        print(f"{prefix}|{pid}/{mode}: mean top5 overlap {overlap:.3f} below warn floor {top5_floor:.3f}")
PY
)
    asym4_specs=()
    for mode in "${MODE_ARR[@]}"; do
        if [[ "$mode" == asym4_tqv* ]]; then
            asym4_specs+=("$mode=$OUT/logits/$pid/$mode")
        fi
    done
    if [ -f "$OUT/logits/$pid/asym4.tokens" ] && [ "${#asym4_specs[@]}" -gt 0 ]; then
        ./scripts/quant_compare_top5.py "$OUT/logits/$pid/asym4" "${asym4_specs[@]}" \
            > "$OUT/logits/$pid/compare_vs_asym4.json"
        while IFS='|' read -r kind msg; do
            collect_diag "$kind" "$msg"
        done < <(python3 - "$OUT/logits/$pid/compare_vs_asym4.json" "$pid vs asym4" "$TOP1_WARN_FLOOR" "$TOP5_WARN_FLOOR" "$STRICT" <<'PY'
import json
import math
import sys

path, pid = sys.argv[1], sys.argv[2]
top1_floor, top5_floor = float(sys.argv[3]), float(sys.argv[4])
strict = sys.argv[5] == "1"
prefix = "ERROR" if strict else "WARN"
for row in json.load(open(path)):
    mode = row["mode"]
    top1 = float(row["top1_agreement"])
    overlap = float(row["mean_top5_overlap"])
    steps = int(row["steps_compared"])
    if steps <= 0 or not math.isfinite(top1) or not math.isfinite(overlap):
        print(f"ERROR|{pid}/{mode}: invalid comparison metrics steps={steps} top1={top1} overlap={overlap}")
        continue
    if top1 < top1_floor:
        print(f"{prefix}|{pid}/{mode}: top1 agreement {top1:.3f} below warn floor {top1_floor:.3f}")
    if overlap < top5_floor:
        print(f"{prefix}|{pid}/{mode}: mean top5 overlap {overlap:.3f} below warn floor {top5_floor:.3f}")
PY
)
    fi
    {
        echo "### $pid"
        echo
        echo "- prompt_md5: $pmd5"
        echo
        echo "$BASE_MODE baseline:"
        echo
        echo '```json'
        cat "$OUT/logits/$pid/compare.json"
        echo '```'
        echo
        if [ -f "$OUT/logits/$pid/compare_vs_asym4.json" ]; then
            echo "asym4 baseline:"
            echo
            echo '```json'
            cat "$OUT/logits/$pid/compare_vs_asym4.json"
            echo '```'
            echo
        fi
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
        token_count="$(python3 - "$out_file" <<'PY'
import sys
n = 0
for line in open(sys.argv[1]):
    if '"type":"token"' in line:
        n += 1
print(n)
PY
)"
        if [ -z "$done_line" ]; then
            hard_error "$pid/$mode: missing daemon done line"
        fi
        if [ "${token_count:-0}" -eq 0 ]; then
            hard_error "$pid/$mode: daemon emitted zero tokens"
        fi
        if [ "$pid" = "tool" ]; then
            while IFS='|' read -r kind msg; do
                collect_diag "$kind" "$msg"
            done < <(python3 - "$OUT/prompts/${pid}.${mode}.txt" "$mode" "$STRICT" "$CHECK_TOOL_JSON" <<'PY'
import json
import re
import sys

text = open(sys.argv[1], errors="ignore").read()
mode = sys.argv[2]
strict = sys.argv[3] == "1"
check_tool_json = sys.argv[4]
prefix = "ERROR" if strict else "WARN"
json_prefix = "ERROR" if (strict or check_tool_json == "strict") else "WARN"
if "<|im_start|>" in text or "<|im_end|>" in text:
    print(f"WARN|tool/{mode}: emitted chat special token around tool response")
matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
if not matches:
    print(f"{prefix}|tool/{mode}: no <tool_call> block emitted")
elif check_tool_json != "off":
    for idx, payload in enumerate(matches):
        try:
            json.loads(payload)
        except Exception as exc:
            print(f"{json_prefix}|tool/{mode}: tool_call #{idx} is not valid JSON: {exc}")
PY
)
        fi
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

diag_file="$OUT/logits/tqv3-collapse.diag"
python3 - "$OUT" "$MODES" > "$diag_file" <<'PY'
import filecmp
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
modes = [m.strip() for m in sys.argv[2].split(",") if m.strip()]
if "asym4_tqv2" not in modes or "asym4_tqv3" not in modes:
    raise SystemExit(0)

prompt_dirs = [p for p in (out / "logits").iterdir() if p.is_dir()]
checked = 0
same = 0
for prompt_dir in prompt_dirs:
    t2 = prompt_dir / "asym4_tqv2.tokens"
    t3 = prompt_dir / "asym4_tqv3.tokens"
    k2 = prompt_dir / "asym4_tqv2.top5.csv"
    k3 = prompt_dir / "asym4_tqv3.top5.csv"
    if not (t2.exists() and t3.exists() and k2.exists() and k3.exists()):
        continue
    checked += 1
    if filecmp.cmp(t2, t3, shallow=False) and filecmp.cmp(k2, k3, shallow=False):
        same += 1

if checked > 0 and checked == same:
    print(f"WARN|asym4_tqv3 exactly matched asym4_tqv2 for all {checked} prompts; TQV3 is probably collapsed to the TQV2 table/path")
PY
while IFS='|' read -r kind msg; do
    collect_diag "$kind" "$msg"
done < "$diag_file"

echo "== split perf =="
if [ "$FULL" -eq 1 ]; then
    PREFILLS=(512 2048 8192 32768)
    GEN=128
else
    PREFILLS=(512 2048)
    GEN=64
fi
printf '%s\n' "mode,prefill,run,prefill_tok_s,decode_tok_s,summary" > "$OUT/perf/perf.csv"
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
while IFS='|' read -r kind msg; do
    collect_diag "$kind" "$msg"
done < <(python3 - "$OUT/perf/perf.csv" <<'PY'
import csv
import math
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
if not rows:
    print("ERROR|perf produced no rows")
for row in rows:
    label = f"perf {row.get('mode')} prefill={row.get('prefill')} run={row.get('run')}"
    for key in ("prefill_tok_s", "decode_tok_s"):
        raw = row.get(key, "")
        try:
            value = float(raw)
        except Exception:
            print(f"ERROR|{label}: missing numeric {key}")
            continue
        if not math.isfinite(value) or value <= 0:
            print(f"ERROR|{label}: invalid {key}={raw}")
PY
)
{
    echo "## Split Perf"
    echo
    echo '```csv'
    cat "$OUT/perf/perf.csv"
    echo '```'
    echo
} >> "$OUT/report.md"

echo "== identicality dashboard =="
dashboard_args=(scripts/kv_quality_dashboard.py "$OUT" --baseline "$BASE_MODE" --model "$MODEL" --out "$OUT/dashboard.html" --json-out "$OUT/identicality.json")
if [ -n "$MODEL_NAME" ]; then dashboard_args+=(--model-name "$MODEL_NAME"); fi
if python3 "${dashboard_args[@]}"; then
    {
        echo "## Identicality Dashboard"
        echo
        echo "- html: $OUT/dashboard.html"
        echo "- json: $OUT/identicality.json"
        echo
    } >> "$OUT/report.md"
else
    warn "identicality dashboard generation failed"
fi

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

{
    echo "## Gate Diagnostics"
    echo
    echo "### Warnings"
    echo
    if [ "${#WARNINGS[@]}" -eq 0 ]; then
        echo "- none"
    else
        for msg in "${WARNINGS[@]}"; do
            echo "- $msg"
        done
    fi
    echo
    echo "### Hard Errors"
    echo
    if [ "${#ERRORS[@]}" -eq 0 ]; then
        echo "- none"
    else
        for msg in "${ERRORS[@]}"; do
            echo "- $msg"
        done
    fi
    echo
} >> "$OUT/report.md"

echo "report: $OUT/report.md"
if [ "${#ERRORS[@]}" -gt 0 ]; then
    exit 1
fi
