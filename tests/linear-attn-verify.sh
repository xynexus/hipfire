#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# Cross-check the hipfire-train linear_attn / MoE host implementations against
# the inference kernels, AT THE GEOMETRIES REAL MODELS USE.
#
# These checks existed but were referenced by nothing, so they ran only when
# someone remembered — and each one defaulted to a toy config (2 heads, h=64,
# n_k == n_v, 8 experts). That is how a GQA bug and a 256-expert routing bug
# could both have hidden behind a passing check. The geometries below are the
# ones Qwen3.5-0.8B and Qwen3.6-35B-A3B actually run.
#
# Usage: ./tests/linear-attn-verify.sh
# Requires a GPU. Takes the resource lock, as non-daemon GPU binaries must.

set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="linear-attn-verify-$$"
./target/release/hipfire lock acquire "$LABEL" >/dev/null
trap './target/release/hipfire lock release '"$LABEL"' >/dev/null 2>&1 || true' EXIT

cargo build --release -q -p hipfire-train --features deltanet \
  --example verify_la_core_vs_kernels \
  --example verify_deltanet_vs_kernel \
  --example verify_moe_router

fail=0
run() {
  local name="$1"; shift
  printf '\n=== %s %s ===\n' "$name" "$*"
  if ! ./target/release/examples/"$name" "$@"; then
    echo "FAILED: $name $*"
    fail=1
  fi
}

# args: seq n_value_heads n_key_heads hidden
run verify_la_core_vs_kernels 5 2 0 64        # toy, the historical default
run verify_la_core_vs_kernels 5 16 16 1024    # Qwen3.5-0.8B
run verify_la_core_vs_kernels 5 32 16 2048    # Qwen3.6-35B-A3B — GQA
run verify_la_core_vs_kernels 5 32 32 2048    # same width, no GQA

# args: seq n_heads
run verify_deltanet_vs_kernel 6 2
run verify_deltanet_vs_kernel 64 32           # real head count, real depth

# args: seq n_experts (top-8 is the kernel's compile-time K)
run verify_moe_router 8 8
run verify_moe_router 32 256                  # the 35B's expert count

if [ "$fail" -ne 0 ]; then
  echo && echo "linear-attn-verify: FAIL"
  exit 1
fi
echo && echo "linear-attn-verify: PASS"
