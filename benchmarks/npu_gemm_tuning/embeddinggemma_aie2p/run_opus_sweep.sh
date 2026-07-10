#!/usr/bin/env bash
# Run the EmbeddingGemma-300M AIE2P Opus GEMM sweep.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="${HIPFIRE_NPU_CACHE_DIR:-$HOME/.hipfire/npu}"
OUT="${1:-benchmarks/npu_gemm_tuning/results/embeddinggemma-aie2p-opus-$(date -u +%Y%m%dT%H%M%SZ).csv}"
W8_MTS="${HIPFIRE_EMBGEMMA_NPU_W8_MTS:-4}"
if [[ "$OUT" = /* ]]; then
  OUT_PATH="$OUT"
else
  OUT_PATH="$ROOT/$OUT"
fi
mkdir -p "$(dirname "$OUT_PATH")"

args=()
w8_ready=1
for n in 256 768 1152 3072; do
  args+=(--cache "w4:$n:$BASE/embgemma_aie2p_w4_4x4x16_c8_nb$((n / 64))")
  for w8_mt in $W8_MTS; do
    w8_dir="$BASE/embgemma_aie2p_w8_${w8_mt}x4x32_c8_nb$((n / 64))_m8k8_w8"
    if [ -d "$w8_dir" ] && { [ -f "$w8_dir/VERIFIED" ] || [ "${HIPFIRE_EMBGEMMA_NPU_ALLOW_UNVERIFIED_W8:-0}" = 1 ]; }; then
      args+=(--cache "w8:$n:$w8_dir")
    else
      w8_ready=0
    fi
  done
done

if [ "$w8_ready" = 1 ]; then
  default_formats="oq4++,oq8++,oq4.25-policy"
else
  default_formats="oq4++"
  echo "W8 caches are not present; defaulting to HIPFIRE_EMBGEMMA_NPU_FORMATS=$default_formats" >&2
fi

(
  cd "$ROOT"
  cargo run --release -p hipfire-xdna --example npu_embeddinggemma_opus_sweep -- \
    "${args[@]}" \
    --formats "${HIPFIRE_EMBGEMMA_NPU_FORMATS:-$default_formats}" \
    --batches "${HIPFIRE_EMBGEMMA_NPU_BATCHES:-32,128,512}" \
    --warmup "${HIPFIRE_EMBGEMMA_NPU_WARMUP:-2}" \
    --iters "${HIPFIRE_EMBGEMMA_NPU_ITERS:-10}"
) | tee "$OUT_PATH"

echo "wrote $OUT_PATH"
