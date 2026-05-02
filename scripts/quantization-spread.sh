#!/usr/bin/env bash
# Run quantization-gate across a small spread of model sizes.

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${HIPFIRE_QUANT_SPREAD_MODELS:-$HOME/.hipfire/models/qwen3.5-0.8b.mq4,$HOME/.hipfire/models/qwen3.5-2b.mq4,$HOME/.hipfire/models/qwen3.5-9b.mq4}"
MODES="${HIPFIRE_QUANT_MODES:-q8,asym4,asym4_tqv4,asym4_tqv3,asym4_tqv2}"
OUT="${HIPFIRE_QUANT_SPREAD_OUT:-benchmarks/results/quantization-spread-$(date +%Y%m%d-%H%M%S)}"
RUNS=1
FULL=0
COHERENCE=0
STRICT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --full) FULL=1; shift ;;
        --coherence) COHERENCE=1; shift ;;
        --strict) STRICT=1; shift ;;
        -h|--help)
            echo "usage: $0 [--models CSV] [--modes CSV] [--out DIR] [--runs N] [--full] [--coherence] [--strict]"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT"
IFS=',' read -r -a MODEL_ARR <<< "$MODELS"

{
    echo "# Quantization Spread"
    echo
    echo "- date: $(date -Iseconds)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- modes: $MODES"
    echo "- runs: $RUNS"
    echo "- full: $FULL"
    echo "- coherence: $COHERENCE"
    echo "- strict: $STRICT"
    echo
} > "$OUT/report.md"

build_flag=()
exit_code=0
for model in "${MODEL_ARR[@]}"; do
    if [ ! -f "$model" ]; then
        echo "- missing model: $model" >> "$OUT/report.md"
        exit_code=1
        continue
    fi
    stem="$(basename "$model")"
    stem="${stem//[^A-Za-z0-9_.-]/_}"
    model_out="$OUT/$stem"
    args=(--model "$model" --modes "$MODES" --runs "$RUNS" --out "$model_out")
    if [ "$FULL" -eq 1 ]; then args+=(--full); fi
    if [ "$COHERENCE" -eq 1 ]; then args+=(--coherence); fi
    if [ "$STRICT" -eq 1 ]; then args+=(--strict); fi
    args+=("${build_flag[@]}")

    echo "== spread model: $model =="
    if ./scripts/quantization-gate.sh "${args[@]}"; then
        status="pass"
    else
        status="fail"
        exit_code=1
    fi
    build_flag=(--skip-build)

    {
        echo "## $stem"
        echo
        echo "- model: $model"
        echo "- status: $status"
        echo "- report: $model_out/report.md"
        if [ -f "$model_out/perf/perf.csv" ]; then
            echo
            echo '```csv'
            cat "$model_out/perf/perf.csv"
            echo '```'
        fi
        echo
    } >> "$OUT/report.md"
done

echo "spread report: $OUT/report.md"
exit "$exit_code"
