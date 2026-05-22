#!/bin/bash
# paro_k0_eval.sh — drive the hipfire-quantize + eval pipeline for the
# paro_k0_stage2.py output. Runs two cells:
#   A) paro + GPTQ  — replaces closed-form AWQ + GPTQ baseline
#   B) paro only    — control to isolate stage-2 weight tuning contribution
#
# Inputs:
#   PARO_DIR      directory written by paro_k0_stage2.py
#                 (contains tuned-safetensors/ and paro_channel_scales.hfsc)
#   HESSIAN       path to the GPTQ hessian.bin (same one used by baseline)
#   IMATRIX       path to the .imatrix.gguf
#   KLDREF        path to .kldref.bin
#   ALPHA         AWQ alpha (must match the one paro_k0_stage2.py was trained with)
#   GPTQ_DAMP     GPTQ damping (default 0.0 to match baseline)
#   OUT_DIR       cell output directory root

set -euo pipefail

PARO_DIR=${PARO_DIR:?must set PARO_DIR}
HESSIAN=${HESSIAN:?must set HESSIAN}
IMATRIX=${IMATRIX:?must set IMATRIX}
KLDREF=${KLDREF:?must set KLDREF}
ALPHA=${ALPHA:-0.55}
GPTQ_DAMP=${GPTQ_DAMP:-0.0}
OUT_DIR=${OUT_DIR:?must set OUT_DIR}

BIN_QUANT=${BIN_QUANT:-/workspace/hipfire/target/release/hipfire-quantize}
BIN_EVAL=${BIN_EVAL:-/workspace/hipfire/target/release/examples/eval_hipfire}
HARNESS_DIR=${HARNESS_DIR:-/workspace/hipfire/benchmarks/quality-baselines/harness}

TUNED_DIR="${PARO_DIR}/tuned-safetensors"
HFSC="${PARO_DIR}/paro_channel_scales.hfsc"

if [[ ! -d "${TUNED_DIR}" ]]; then
    echo "error: tuned safetensors dir not found: ${TUNED_DIR}" >&2
    exit 2
fi
if [[ ! -f "${HFSC}" ]]; then
    echo "error: HFSC not found: ${HFSC}" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

run_cell() {
    local cell_name="$1"
    local use_gptq="$2"
    local cell_dir="${OUT_DIR}/${cell_name}"
    mkdir -p "${cell_dir}"
    local model_hfq="${cell_dir}/model.mq4.hfq"
    local kldseq="${cell_dir}/eval.kldseq"
    local log="${cell_dir}/run.log"

    echo "=== Cell: ${cell_name} (GPTQ=${use_gptq}) ==="
    echo "  quantize -> ${model_hfq}"
    local q_args=(
        --input "${TUNED_DIR}"
        --output "${model_hfq}"
        --format mq4
        --awq
        --awq-alpha "${ALPHA}"
        --awq-scope f1
        --awq-formula paper
        --awq-scales "${HFSC}"
        --imatrix "${IMATRIX}"
    )
    if [[ "${use_gptq}" == "1" ]]; then
        q_args+=(--gptq "${HESSIAN}" --gptq-damp "${GPTQ_DAMP}")
    fi

    HIPFIRE_GPTQ_HIP_OBS=1 \
    "${BIN_QUANT}" "${q_args[@]}" > "${log}" 2>&1
    grep -E "Output size|Done:|Skipped|Total params|Quantized params" "${log}" | tee -a "${cell_dir}/summary.txt"

    echo "  eval -> ${kldseq}"
    HIPFIRE_KV_MODE=q8 \
    "${BIN_EVAL}" \
        --model "${model_hfq}" \
        --ref "${KLDREF}" \
        --output "${kldseq}" \
        --kv-mode q8 \
        > "${cell_dir}/eval.log" 2>&1
    grep -E "slice-mean KLD|mean NLL|PPL =" "${cell_dir}/eval.log" | tee -a "${cell_dir}/summary.txt"

    # Reduce per-seq KLD into headline numbers.
    python3 - <<PY | tee -a "${cell_dir}/summary.txt"
import sys, math, json
sys.path.insert(0, "${HARNESS_DIR}")
from kldref_format import read_per_seq_kld
m, p, n = read_per_seq_kld("${kldseq}")
mk = sum(m) / len(m)
vn = [x for x in n if not math.isnan(x)]
mn = sum(vn) / len(vn) if vn else float("nan")
ppl = math.exp(mn) if vn and not math.isnan(mn) else float("nan")
out = {
    "cell": "${cell_name}",
    "gptq": "${use_gptq}" == "1",
    "alpha": ${ALPHA},
    "gptq_damp": ${GPTQ_DAMP},
    "kld": mk,
    "ppl": ppl,
    "n_chunk": len(m),
}
print(json.dumps(out, indent=2))
with open("${cell_dir}/headline.json", "w") as f:
    json.dump(out, f, indent=2)
PY
    echo ""
}

run_cell "paro_plus_gptq" "1"
run_cell "paro_only" "0"

echo "=== All cells complete. Summary: ==="
for cell in paro_plus_gptq paro_only; do
    cat "${OUT_DIR}/${cell}/headline.json" 2>/dev/null || true
    echo
done
