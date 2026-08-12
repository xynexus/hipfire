#!/usr/bin/env bash
# Confirm the FLUX.2 oq8 admission rejection is trajectory divergence (sampler
# chaos), not weight corruption. Two independent probes:
#
#   (1) per-tensor weight-space reconstruction error  [CPU, no GPU]
#         hipfire diffusion quant-diff <ref> <cand>
#       If the worst tensor is near-lossless, the quant is faithful.
#
#   (2) step-by-step trajectory divergence            [GPU]
#       Render ref + cand with identical seed/prompt, dumping the denoise trace,
#       then compare step-1 velocity (identical inputs) vs final-latent drift.
#
# Defaults mirror the failing admission case
# (benchmarks/results/flux2-klein-oq8-admission-diagnostic-2026-07-12).
set -euo pipefail

HIPFIRE="${HIPFIRE:-./target/release/hipfire}"
REF="${REF:-/srv/huggingface/FLUX.2-klein-base-4B.diffusers.p0.hfq}"
CAND="${CAND:-/srv/huggingface/FLUX.2-klein-base-4B--oq8.hfq}"
PROMPT_FILE="${PROMPT_FILE:-benchmarks/prompts/flux2_image_admission_object.txt}"
SEED="${SEED:-7}"
STEPS="${STEPS:-4}"
CFG="${CFG:-4.0}"
W="${W:-64}"
H="${H:-64}"
DEVICE="${DEVICE:-0}"
OUT="${OUT:-/tmp/flux2-chaos-$$}"

PROMPT="$(cat "$PROMPT_FILE")"
mkdir -p "$OUT/ref_trace" "$OUT/cand_trace" "$OUT/img"
echo "workdir: $OUT"

echo "=============================================================="
echo "(1) weight-space reconstruction error  [CPU]"
echo "=============================================================="
"$HIPFIRE" diffusion quant-diff "$REF" "$CAND" --top 30

echo
echo "=============================================================="
echo "(2) trajectory divergence  [GPU device $DEVICE]"
echo "=============================================================="
# Non-daemon GPU binaries do not self-lock (AGENTS.md): hold the GPU lock across
# both renders.
"$HIPFIRE" lock acquire
trap '"$HIPFIRE" lock release || true' EXIT

for pair in "ref:$REF" "cand:$CAND"; do
  tag="${pair%%:*}"; model="${pair#*:}"
  echo "-- rendering $tag ($model)"
  HIPFIRE_DUMP_DENOISE_TRACE="$OUT/${tag}_trace" \
    "$HIPFIRE" diffusion txt2img \
      --model "$model" \
      --prompt "$PROMPT" \
      --output "$OUT/img/${tag}.png" \
      --seed "$SEED" --steps "$STEPS" --cfg-scale "$CFG" \
      --width "$W" --height "$H" \
      --rocm-device-id "$DEVICE"
done

"$HIPFIRE" lock release
trap - EXIT

echo
echo "=============================================================="
echo "trajectory comparison"
echo "=============================================================="
python3 scripts/flux2_trajectory_divergence.py "$OUT/ref_trace" "$OUT/cand_trace"

echo
echo "images: $OUT/img/ref.png  vs  $OUT/img/cand.png  (eyeball: same content, moved?)"
