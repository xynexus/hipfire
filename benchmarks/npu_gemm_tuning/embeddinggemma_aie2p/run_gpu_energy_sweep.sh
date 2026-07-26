#!/usr/bin/env bash
# Benchmark all local EmbeddingGemma Opus artifacts on one fixed GPU workload.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_DIR="${HIPFIRE_EMBGEMMA_MODEL_DIR:-$HOME/.hipfire/models/embeddinggemma-300m}"
M="${HIPFIRE_EMBGEMMA_ENERGY_M:-256}"
ITERS="${HIPFIRE_EMBGEMMA_ENERGY_ITERS:-30}"
ROUNDS="${HIPFIRE_EMBGEMMA_ENERGY_ROUNDS:-3}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${1:-$ROOT/benchmarks/npu_gemm_tuning/results/embeddinggemma-opus-gpu-energy-$STAMP.csv}"
LOG="${OUT%.csv}.log"
BINARY="$ROOT/target/release/examples/embed_e2e"

mkdir -p "$(dirname "$OUT")"
cd "$ROOT"
cargo build --release -p hipfire-arch-embeddinggemma --example embed_e2e

mapfile -t opus_models < <(find "$MODEL_DIR" -maxdepth 1 -type f \
  -name 'EmbeddingGemma-300M*oq*.hfq' -printf '%p\n' | sort)
models=("$MODEL_DIR/EmbeddingGemma-300M.bf16.hfq" "${opus_models[@]}")
for model in "${models[@]}"; do
  [[ -f "$model" ]] || { echo "missing model: $model" >&2; exit 1; }
done

printf 'timestamp,host,arch,round,position,m,iters,label,path,bytes,tok_s,idle_w,pkg_w,dyn_w,pkg_tok_j,dyn_tok_j\n' >"$OUT"
: >"$LOG"
host="$(hostname -s)"
arch="$(rocminfo 2>/dev/null | sed -n 's/^[[:space:]]*Name:[[:space:]]*\(gfx[0-9]*\).*/\1/p' | head -n1)"
arch="${arch:-unknown}"

hipfire lock acquire embeddinggemma-opus-energy
trap 'hipfire lock release >/dev/null 2>&1 || true' EXIT

for ((round = 1; round <= ROUNDS; round++)); do
  order=("${models[@]}")
  if ((round % 2 == 0)); then
    reversed=()
    for ((index = ${#order[@]} - 1; index >= 0; index--)); do
      reversed+=("${order[index]}")
    done
    order=("${reversed[@]}")
  fi
  position=0
  for model in "${order[@]}"; do
    position=$((position + 1))
    label="$(basename "$model" .hfq)"
    echo "[$round/$ROUNDS $position/${#order[@]}] $label" | tee -a "$LOG" >&2
    output="$($BINARY --hfq "$model" --bench-m "$M" --bench-iters "$ITERS" 2>>"$LOG")"
    metrics="$(printf '%s\n' "$output" | sed -n '/^gpu_tok_s=/p' | tail -n1)"
    [[ -n "$metrics" ]] || { echo "missing benchmark metrics for $model" >&2; exit 1; }
    eval "$metrics"
    bytes="$(stat -c %s "$model")"
    printf '%s,%s,%s,%d,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$host" "$arch" "$round" "$position" \
      "$M" "$ITERS" "$label" "$model" "$bytes" "$gpu_tok_s" "$idle_w" \
      "$pkg_w" "$dyn_w" "$pkg_tok_j" "$dyn_tok_j" | tee -a "$OUT"
    sleep 1
  done
done

echo "wrote $OUT" >&2
echo "wrote $LOG" >&2
