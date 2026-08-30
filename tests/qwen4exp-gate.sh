#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.

# Qwen3.8-Flash-Next (qwen4_exp) correctness gate.
#
# The port's family-specific kernels are each differenced against an independent
# CPU implementation, and its offline half is pinned against the SHIPPED
# checkpoint's own config and tensor list. Without a gate, all of that is
# protected by someone remembering to run it.
#
# Two stages:
#
#   1. CPU  — the oracles and the checkpoint-pinned tests. No GPU, seconds.
#             Includes the fixture emit + quantize round trip, which is what
#             catches a quant-policy regression on the 102 GB n-gram table.
#   2. GPU  — `parity_qwen4exp_kernels`, which differences all eight kernels
#             against those oracles and runs six negative controls. Skipped with
#             a clear message when no GPU is present, so the CPU half still
#             gates in CI.
#
# The negative controls are the point. Each one asserts an EXACT value that a
# plausible wrong implementation would miss, and two have already caught real
# defects: a sum-then-ReLU reading, and a 64-lane `__shfl_xor` reduction that
# silently doubled on wave32 (see BUGS.md).
#
# Exit codes:
#   0  everything passed (GPU stage may have been skipped)
#   1  a test or parity check failed
#   2  build / environment error
#
# Usage:
#   ./tests/qwen4exp-gate.sh              # both stages
#   ./tests/qwen4exp-gate.sh --cpu-only   # skip the GPU stage
#
# This gate does NOT take the resource lock: the parity example allocates a few
# MB and runs for under a second, and wrapping it would deadlock against a
# caller that already holds the lock.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CPU_ONLY=0
[ "${1:-}" = "--cpu-only" ] && CPU_ONLY=1

fail=0
step() { printf '\n== %s ==\n' "$1"; }

step "CPU: oracles + checkpoint-pinned tests"
if ! cargo test --quiet -p hipfire-arch-qwen4exp -p hipfire-arch-qwen4exp-spec 2>&1 | tail -25; then
  echo "qwen4exp-gate: CPU tests FAILED"
  fail=1
fi

# The reference-oracle tests SKIP when the artifact has not been generated, so
# say which happened — a silent skip reads exactly like a pass.
if [ -f crates/hipfire-arch-qwen4exp/tests/oracle/oracle.json ]; then
  echo "  reference oracle: present (plan differenced against transformers @5f8ab9bb)"
else
  echo "  reference oracle: ABSENT — those tests skipped."
  echo "                    regenerate with scripts/qwen4exp_oracle.py (see its header)"
fi

step "CPU: fixture emit + quantize round trip"
# Proves the arch resolves, the stacked experts split, and the quant policy still
# lands the n-gram table and the indexer at source precision. A regression here
# is silent in the artifact and only shows up as quality loss much later.
if ! cargo build --quiet --release -p hipfire-quantize --bin hipfire-quantize 2>&1 | tail -5; then
  echo "qwen4exp-gate: quantizer build FAILED"
  exit 2
fi
# Build and use the IN-TREE `hipfire`, not ~/.hipfire/bin. That symlink points at
# whatever was last installed, so a gate reading through it can report on a binary
# that predates the change under test — which is how arch 26 showed up as
# `arch_name: None` here while the registration was in fact present.
if ! cargo build --quiet --release --bin hipfire 2>&1 | tail -5; then
  echo "qwen4exp-gate: hipfire build FAILED"
  exit 2
fi
HF=./target/release/hipfire
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if ./target/release/hipfire-quantize --emit-fixture qwen4_exp --out "$TMP/fx" >/dev/null 2>&1 \
   && ./target/release/hipfire-quantize --input "$TMP/fx" --output "$TMP/fx.hfq" \
        --format oq4 >/dev/null 2>&1; then
  # The n-gram shards and the indexer must NOT be quantized. `precision_class` is
  # inert until an arch opts into the source-precision site, so this is the check
  # that the opt-in is still wired.
  rows=$("$HF" inspect "$TMP/fx.hfq" --tensors 2>/dev/null \
         | grep -cE "ngram_embedding\.shard_|indexer\." || true)
  bad=$("$HF" inspect "$TMP/fx.hfq" --tensors 2>/dev/null \
        | grep -E "ngram_embedding\.shard_|indexer\." \
        | grep -vcE "F16|BF16" || true)
  # Guard against passing VACUOUSLY: if `inspect`'s format drifts and the pattern
  # stops matching, `bad` is 0 and this would report success while checking
  # nothing. The default fixture has 4 n-gram shards + 3 indexer tensors.
  if [ "${rows:-0}" -lt 7 ]; then
    echo "qwen4exp-gate: only $rows n-gram/indexer rows found (expected >= 7) —"
    echo "               the inspect format probably drifted; this check is not testing anything"
    fail=1
  elif [ "${bad:-0}" != "0" ]; then
    echo "qwen4exp-gate: $bad of $rows n-gram/indexer tensors are NOT at source precision"
    fail=1
  else
    echo "  fixture quantized; all $rows n-gram + indexer tensors at source precision"
  fi
else
  echo "qwen4exp-gate: fixture emit/quantize FAILED"
  fail=1
fi

step "CPU: config survives into the SERVED artifact"
# The source config parsing is unit-tested; this checks the quantizer still
# carries it into the `.hfq` metadata envelope, which nothing else would notice.
if ! cargo build --quiet --release -p hipfire-arch-qwen4exp --example parse_metadata 2>&1 | tail -3; then
  echo "qwen4exp-gate: parse_metadata build FAILED"
  fail=1
elif [ -f "$TMP/fx.hfq" ]; then
  arch=$("$HF" inspect "$TMP/fx.hfq" --json 2>/dev/null \
         | python3 -c 'import json,sys; j=json.load(sys.stdin); print(j.get("arch_id"), j.get("arch_name"))')
  echo "  artifact arch: $arch"
  case "$arch" in
    "26 qwen4-exp") ;;
    *) echo "qwen4exp-gate: artifact arch is '$arch', expected '26 qwen4-exp'"; fail=1 ;;
  esac
  if "$HF" inspect "$TMP/fx.hfq" --json 2>/dev/null \
     | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["metadata"]))' \
     | ./target/release/examples/parse_metadata; then
    :
  else
    echo "qwen4exp-gate: config did NOT survive into the artifact metadata"
    fail=1
  fi
else
  echo "qwen4exp-gate: no fixture artifact to check"
  fail=1
fi

step "GPU: load a real artifact through the Architecture impl"
# Everything else in this gate uses synthetic weights uploaded from host arrays.
# This is the only check that exercises the LOADER: config out of the artifact's
# metadata, weights off the HFQ, per-layer state, real logits.
if ! cargo build --quiet --release -p hipfire-arch-qwen4exp \
     --example load_hfq_decode 2>&1 | tail -5; then
  echo "qwen4exp-gate: load_hfq_decode build FAILED"
  exit 2
fi
if ./target/release/hipfire-quantize --input "$TMP/fx" --output "$TMP/fx.bf16.hfq" \
     --format bf16 >/dev/null 2>&1; then
  ld="$(./target/release/examples/load_hfq_decode "$TMP/fx.bf16.hfq" 2>&1 \
        | grep -E "argmax over|WARNING|load_hfq_decode:" || true)"
  echo "$ld" | sed 's/^/  /'
  case "$ld" in
    *"load_hfq_decode: OK"*) ;;
    *skipped*) ;;
    *) echo "qwen4exp-gate: artifact load/decode FAILED"; fail=1 ;;
  esac
  case "$ld" in
    *WARNING*) echo "qwen4exp-gate: the argmax never moved — the load is suspect"; fail=1 ;;
  esac
else
  echo "qwen4exp-gate: bf16 artifact build FAILED"
  fail=1
fi

step "GPU: serve through the REGISTERED factory"
# The only check that the SERVING seam is wired, as opposed to the trunk being
# correct. It fails for four distinct reasons, each of which has happened: arch 26
# not resolving to a factory (a missing link edge in hipfire-archs), the factory
# failing to build a backend, a dead forward (finite but frozen logits), and an
# incomplete reset leaking recurrent state between requests.
if ! cargo build --quiet --release -p hipfire-arch-qwen4exp \
     --example serve_fixture 2>&1 | tail -5; then
  echo "qwen4exp-gate: serve_fixture build FAILED"
  exit 2
fi
if [ -f "$TMP/fx.bf16.hfq" ]; then
  sv="$(./target/release/examples/serve_fixture "$TMP/fx.bf16.hfq" 2>&1 || true)"
  echo "$sv" | sed 's/^/  /'
  case "$sv" in
    *"serve_fixture: OK"*) ;;
    *skipped*) ;;
    *) echo "qwen4exp-gate: serving through the factory FAILED"; fail=1 ;;
  esac
  # A frozen argmax is the signature of a dead forward, and it is reported rather
  # than thrown, so check the line explicitly.
  bf16_argmax="$(echo "$sv" | grep -oE 'argmax [0-9]+' | head -1 | awk '{print $2}')"
  case "$sv" in
    *"argmax moved = false"*)
      echo "qwen4exp-gate: decode ran but the argmax never moved — dead forward"
      fail=1 ;;
  esac
else
  echo "qwen4exp-gate: no bf16 artifact to serve"
  fail=1
fi

step "GPU: serve QUANTISED artifacts, against the bf16 control"
# The trunk dequantises oq4/oq8 at load. What makes this a real check rather than
# a smoke test is comparing the ARGMAX to the bf16 run above: a dequant that is
# subtly wrong (mismatched FWHT basis, wrong block stride) still yields finite
# logits, and only diverges from the float control.
if [ -n "${bf16_argmax:-}" ]; then
  for q in oq4 oq8; do
    if ! ./target/release/hipfire-quantize --input "$TMP/fx" --output "$TMP/fx.$q.hfq" \
         --format "$q" >/dev/null 2>&1; then
      echo "qwen4exp-gate: could not quantize the fixture to $q"; fail=1; continue
    fi
    qout="$(./target/release/examples/serve_fixture "$TMP/fx.$q.hfq" 2>&1 || true)"
    case "$qout" in
      *"serve_fixture: OK"*) ;;
      *) echo "  $q: FAILED to serve"; echo "$qout" | tail -3 | sed 's/^/    /'; fail=1; continue ;;
    esac
    qam="$(echo "$qout" | grep -oE 'argmax [0-9]+' | head -1 | awk '{print $2}')"
    qdt="$(echo "$qout" | sed -nE 's/.*routed experts resident as Some\(([A-Za-z0-9]+)\).*/\1/p' | head -1)"
    if [ "$qam" != "$bf16_argmax" ]; then
      echo "qwen4exp-gate: $q argmax $qam != bf16 argmax $bf16_argmax — dequant is wrong"
      fail=1
    # The experts must stay QUANTISED. Falling back to F32 serves identically and
    # costs ~8x the memory, so it is invisible in the logits — on the shipped
    # geometry the routed experts are 97.3% of the trunk, which is the whole
    # difference between ~32 GB and ~258 GB resident.
    elif [ "$qdt" = "F32" ] || [ -z "$qdt" ]; then
      echo "qwen4exp-gate: $q serves but its routed experts are resident as '${qdt:-unknown}'"
      echo "               — they were silently dequantised; the model would not fit"
      fail=1
    else
      echo "  $q: serves, argmax $qam matches bf16, experts resident as $qdt"
    fi
  done
else
  echo "qwen4exp-gate: no bf16 argmax recorded; cannot compare quantised runs"
  fail=1
fi

if [ "$CPU_ONLY" = "1" ]; then
  [ "$fail" = "0" ] && echo && echo "qwen4exp-gate: PASS (cpu-only)"
  exit "$fail"
fi

step "GPU: kernel parity + negative controls"
if ! cargo build --quiet --release -p hipfire-rdna --example parity_qwen4exp_kernels 2>&1 | tail -5; then
  echo "qwen4exp-gate: parity build FAILED"
  exit 2
fi
# GPU-vs-CPU for the composed Gated DeltaNet step. The kernel-level parity below
# checks individual kernels; this checks the SEQUENCE of them against the CPU
# reference that `reference_oracle.rs` pins to upstream — so the two together say
# the GPU path computes GDN correctly, which neither says alone.
# Composed GPU-vs-CPU parities. Each name below is a SEQUENCE of kernels checked
# against the CPU reference that `reference_oracle.rs` pins to upstream.
for ex in parity_gdn_gpu_vs_cpu parity_hc_gpu_vs_cpu parity_ple_gpu_vs_cpu \
          parity_indexer_gpu_vs_cpu parity_qsa_attn_gpu_vs_cpu \
          parity_moe_gpu_vs_cpu parity_trunk_gpu_vs_cpu \
          parity_vision_gpu_vs_cpu; do
  if ! cargo build --quiet --release -p hipfire-arch-qwen4exp --example "$ex" 2>&1 | tail -5; then
    echo "qwen4exp-gate: $ex build FAILED"
    exit 2
  fi
  line="$(./target/release/examples/"$ex" 2>&1 | grep "^$ex" || true)"
  echo "  $line"
  case "$line" in
    *OK*|*skipped*) ;;
    *) echo "qwen4exp-gate: $ex FAILED"; fail=1 ;;
  esac
done

out="$(./target/release/examples/parity_qwen4exp_kernels 2>&1)"
echo "$out" | sed 's/^/  /'
if echo "$out" | grep -q "no GPU"; then
  echo "  (GPU stage skipped)"
elif ! echo "$out" | grep -q "parity_qwen4exp_kernels: OK"; then
  echo "qwen4exp-gate: kernel parity FAILED"
  fail=1
fi

echo
if [ "$fail" = "0" ]; then
  echo "qwen4exp-gate: PASS"
else
  echo "qwen4exp-gate: FAIL"
fi
exit "$fail"
