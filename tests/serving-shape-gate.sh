#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — serving-shape parity gate (GPU; runs beside a serving daemon).
#
# The tiny fixtures (hidden 256, head_dim 128, one KV head) cannot reach the routes
# the deployed models take: K >= 5120, the wide multicol (<= 16 rows), the BN=32 and
# BN=64 tiles, the default and 128x64 prefill tiles, head_dim 256 with GQA, and the
# coder's compact grouped MoE GEMM at n >= 64. A tiny pass said nothing about them,
# and the split-K bug and the B % 4 overlay fault both shipped through that gap.
#
# These parity examples check exactly those routes at those shapes, each against an
# independent reference (another route, a per-session path, or a CPU oracle). They
# need a few hundred MiB, so unlike the model gates they run while the daemon holds
# the GPU -- the pre-commit hook runs this whenever a serving route's code is staged.
#
#   ./tests/serving-shape-gate.sh
#
# Exit: 0 all pass, 1 a case failed or crashed, 2 the examples did not build.
set -u
cd "$(git rev-parse --show-toplevel)"

# package example [args...] -- what it covers
CASES=(
    "hipfire-rdna parity_oq_compact_route"                            # dense compact routes, 27B K=5120/17408: multicol, BN=32/64, default, 128x64, chunked
    "hipfire-rdna parity_oq_overlay_tr"                               # overlay correction _tr/_trs/_t, B % 4 != 0 (the B=78 fault), nothing written past B
    "hipfire-rdna parity_kvarn_routed"                                # routed batched KVarN attention, head_dim 256, GQA
    "hipfire-runtime parity_gemv_oq_compact_moe"                      # compact-resident indexed MoE GEMVs (coder decode)
    "hipfire-runtime parity_gemm_oq_compact_moe_grouped 1024 2048 16 64"  # 35B-A3B gate/up, grouped f32 at n >= 64 (prefill)
    "hipfire-runtime parity_gemm_oq_compact_moe_grouped 2048 512 16 128"  # 35B-A3B down
)

declare -A EXAMPLES
for c in "${CASES[@]}"; do
    read -r pkg ex _ <<<"$c"
    EXAMPLES[$pkg]+=" --example $ex"
done
for pkg in "${!EXAMPLES[@]}"; do
    # shellcheck disable=SC2086
    if ! cargo build -q --release -p "$pkg" ${EXAMPLES[$pkg]} 2>/dev/null; then
        echo "serving-shape gate: could not build the $pkg examples" >&2
        exit 2
    fi
done

fail=0
for c in "${CASES[@]}"; do
    read -r pkg ex args <<<"$c"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    if out=$(./target/release/examples/$ex $args 2>&1); then
        echo "  pass  $ex $args ($(( $(date +%s) - t0 ))s)"
    else
        fail=1
        echo "  FAIL  $ex $args"
        echo "$out" | grep -E "FAIL|MISMATCH|panicked|error" | head -10 | sed 's/^/        /'
    fi
done
[ "$fail" = 0 ] && echo "serving-shape gate: PASS" || echo "serving-shape gate: FAIL"
exit "$fail"
