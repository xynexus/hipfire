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
# `set -e` is back: group_c's verdict now reflects its own phases rather than an
# inherited attention check with no attention to check, so a nonzero exit means a
# real failure and SHOULD stop the chain. The greps keep their `|| true` -- a
# grep that matches nothing is not a failure, and that one silently ended the
# chain after layer 0 while printing what a successful run prints.
set -euo pipefail

N=${1:-2}
POS=${2:-30}
SEQ=${3:-31}
# X0: an initial activation for layer 0 instead of the design's random vector.
# With a real token embedding here AND --pos 0, the run is self-consistent: at
# position 0 attention sees exactly one KV entry, the one this layer's own P1 just
# wrote, so nothing synthetic enters the computation.
X0=${X0:-}
# XOUT: where to keep the final layer's x_out, which otherwise dies with the
# temp dir.
XOUT=${XOUT:-}
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
cd "$(dirname "$0")"

for ((L = 0; L < N; L++)); do
    echo "=== layer $L ==="
    # Layer 0 starts from the design's own random x; later layers read the
    # previous x_out. Unset rather than empty -- groups_ab tests presence.
    if [ "$L" -gt 0 ]; then export AB_X_FROM="$D/x$L.npy"
    elif [ -n "$X0" ]; then export AB_X_FROM="$X0"
    else unset AB_X_FROM; fi
    # `|| true` on every grep: under `set -e` a grep that matches nothing kills
    # the whole run, which is how the first version stopped silently after
    # layer 0 while looking like the chain had simply ended.
    AB_EMIT_ATTN="$D/a$L.npy" \
        python3 groups_ab.py --p1-cores 8 --pos "$POS" --layer "$L" 2>&1 |
        { grep -E "attn:|ADVISORY|k' |v' " || true; } | sed 's/^/  /' | tr -s ' '
    # C_X_FROM is the layer INPUT (P3's and P5's residual); C_ATTN_FROM is P3's
    # ACTIVATION. Passing only the latter left group_c using a random residual.
    if [ "$L" -gt 0 ]; then CX="$D/x$L.npy"; else CX="${X0:-}"; fi
    C_X_FROM="$CX" C_ATTN_FROM="$D/a$L.npy" C_EMIT_XOUT="$D/x$((L + 1)).npy" CHAIN_NLAY=1 \
        python3 group_c.py --seq "$SEQ" --layer "$L" 2>&1 |
        { grep -E "P3 h|P4 sw|P5 x_out" || true; } | sed 's/^/  /' | tr -s ' '
done

if [ -n "$XOUT" ]; then cp "$D/x$N.npy" "$XOUT" && echo "final x_out -> $XOUT"; fi
