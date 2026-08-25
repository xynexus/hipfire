#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# Byte-exactness cover for the gated-DeltaNet kernel families.
#
# The f32 kernels (linear / tree / routed) must agree BIT-FOR-BIT, and so must
# the f16 trio: tree replay reproduces the linear path, and routed replay
# reproduces per-session linear. When they disagree, spec-decode commits a state
# sequential decode never reaches and acceptance collapses — a failure that
# shows up as a few percent of tok/s, not as a crash.
#
# None of these tests was run by any gate until now, which is how
# `-ffp-contract=fast` came to fuse `x += y*z` differently in the routed and
# linear kernels despite character-identical source. The f16 test caught it only
# because someone ran it by hand.
#
# Fast (a few seconds each) and needs only a GPU — no model artifacts.
set -uo pipefail
cd "$(dirname "$0")/.."

EXAMPLES=(
    parity_gated_delta_net_f64acc
    parity_gated_delta_net_f64acc_routed
    test_gated_delta_net_tree_f32
    test_gated_delta_net_routed_f32
    test_gated_delta_net_routed_f16
)

echo "tiny-deltanet-gate: building..."
if ! cargo build --release -p hipfire-rdna --features deltanet \
        $(printf -- '--example %s ' "${EXAMPLES[@]}") 2>&1 | tail -3; then
    echo "tiny-deltanet-gate: BUILD FAILED"
    exit 2
fi

status=0
for e in "${EXAMPLES[@]}"; do
    out="$(./target/release/examples/"$e" 2>&1)"
    rc=$?
    if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q PASS; then
        echo "  pass  $e"
    else
        echo "  FAIL  $e"
        printf '%s\n' "$out" | grep -E "byte-exact|mismatches|FAIL|rel L2" | sed 's/^/        /'
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "tiny-deltanet-gate: PASS (${#EXAMPLES[@]} cells)"
else
    echo "tiny-deltanet-gate: FAIL"
fi
exit "$status"
