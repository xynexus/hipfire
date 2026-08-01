#!/usr/bin/env bash
# Run N decoder layers through the role architecture, chained.
#
# Each layer is two dispatches with the seams crossing files:
#
#     groups_ab  x_L -> attention out        (P1 -> P2, q' core to core)
#     group_c    attention out -> x_{L+1}    (P3 -> P4 -> P5, one dispatch)
#
# Layer 0 starts from a random activation; every layer after it consumes the
# previous layer's real x_out, which is what makes this a chain rather than N
# independent layers. The error should FALL after layer 0 -- a saved x_out is
# already bf16, so reference and device then consume bit-identical inputs.
#
#     ./chain_layers.sh 4 [pos] [seq]
#
# Needs the mlir-aie/XRT env, same as the designs themselves.
# No `set -e`: group_c exits nonzero because it still runs p1p2_chain's
# attention check, which has no attention to check here and reports FAIL. Under
# `set -e` that ended the chain after layer 0 while looking like it had finished.
set -uo pipefail

N=${1:-2}
POS=${2:-30}
SEQ=${3:-31}
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
cd "$(dirname "$0")"

for ((L = 0; L < N; L++)); do
    echo "=== layer $L ==="
    # Layer 0 starts from the design's own random x; later layers read the
    # previous x_out. Unset rather than empty -- groups_ab tests presence.
    if [ "$L" -gt 0 ]; then export AB_X_FROM="$D/x$L.npy"; else unset AB_X_FROM; fi
    # `|| true` on every grep: under `set -e` a grep that matches nothing kills
    # the whole run, which is how the first version stopped silently after
    # layer 0 while looking like the chain had simply ended.
    AB_EMIT_ATTN="$D/a$L.npy" \
        python3 groups_ab.py --p1-cores 8 --pos "$POS" --layer "$L" 2>&1 |
        { grep -E "attn:|k' |v' " || true; } | sed 's/^/  /' | tr -s ' '
    C_ATTN_FROM="$D/a$L.npy" C_EMIT_XOUT="$D/x$((L + 1)).npy" CHAIN_NLAY=1 \
        python3 group_c.py --seq "$SEQ" --layer "$L" 2>&1 |
        { grep -E "P3 h|P4 sw|P5 x_out" || true; } | sed 's/^/  /' | tr -s ' '
done
