#!/usr/bin/env bash
# Per-role quantization sensitivity ablation for a FLUX.2 DiT.
#
# For each role (a set of tensor-name substrings), render one denoise step of the
# bf16 model with those tensors forced to low-bit fold (HIPFIRE_DIFFUSION_ABLATE),
# dump the step-1 velocity, and compare to the un-ablated bf16 velocity. The rank
# (flux2_sensitivity_rank.py) shows which roles tolerate low bit — the data the
# mixed-precision allocator needs to push footprint past the static precision map.
#
# In-process, artifact-free: quantizes on the fly from the bf16 weights (no
# per-role .hfq). One model load + one forward per role.
set -euo pipefail

HIPFIRE="${HIPFIRE:-./target/release/hipfire}"
MODEL="${MODEL:-/srv/huggingface/FLUX.2-klein-base-4B.diffusers.p0.hfq}"
BITS="${BITS:-4}"
SEED="${SEED:-7}"
CFG="${CFG:-4.0}"
W="${W:-64}"
H="${H:-64}"
DEVICE="${DEVICE:-0}"
PROMPT_FILE="${PROMPT_FILE:-benchmarks/prompts/flux2_image_admission_object.txt}"
OUT="${OUT:-/tmp/flux2-sens-$$}"
PROMPT="$(cat "$PROMPT_FILE")"
mkdir -p "$OUT"

# role label -> space-separated name substrings (`.weight`-suffixed to be precise;
# "to_q.weight" does not match "to_qkv_mlp_proj.weight").
ROLES=(
  "ff_up|ff.linear_in.weight ff_context.linear_in.weight"
  "attn_out|to_out.0.weight to_add_out.weight attn.to_out.weight"
  "attn_qk|to_q.weight to_k.weight add_q_proj.weight add_k_proj.weight"
  "attn_v|to_v.weight add_v_proj.weight"
  "single_qkvmlp|to_qkv_mlp_proj.weight"
  "modulation|stream_modulation"
)

render() { # <trace_dir> <ablate_substrings|"">
  local dir="$1" ablate="$2"
  mkdir -p "$dir"
  HIPFIRE_DUMP_DENOISE_TRACE="$dir" \
  HIPFIRE_DIFFUSION_ABLATE="$ablate" \
  HIPFIRE_DIFFUSION_ABLATE_BITS="$BITS" \
    "$HIPFIRE" diffusion txt2img --model "$MODEL" --prompt "$PROMPT" \
      --output "$dir/img.png" --seed "$SEED" --steps 1 --cfg-scale "$CFG" \
      --width "$W" --height "$H" --rocm-device-id "$DEVICE" >"$dir/render.log" 2>&1
}

"$HIPFIRE" lock acquire flux2-sensitivity >/dev/null 2>&1
trap '"$HIPFIRE" lock release >/dev/null 2>&1 || true' EXIT

echo "baseline (bf16)..."
render "$OUT/baseline" ""

RANK_ARGS=()
for entry in "${ROLES[@]}"; do
  label="${entry%%|*}"; subs="${entry#*|}"
  echo "ablate $label ($BITS-bit): $subs"
  render "$OUT/$label" "$subs"
  RANK_ARGS+=("$label:$OUT/$label")
done

"$HIPFIRE" lock release >/dev/null 2>&1
trap - EXIT

echo
echo "=== step-1 velocity sensitivity (role forced to ${BITS}-bit) ==="
python3 scripts/flux2_sensitivity_rank.py "$OUT/baseline" "${RANK_ARGS[@]}"
echo
echo "traces: $OUT"
