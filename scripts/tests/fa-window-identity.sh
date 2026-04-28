#!/usr/bin/env bash
# fa-window-identity.sh — sliding-window FA regression test suite.
#
# Lucebox PR #26 explicitly mentions catching a mask off-by-one bug
# RED-to-GREEN with `--test-window`'s 23 assertions. The hipfire
# equivalent: greedy decode with HIPFIRE_FA_WINDOW=W at ctx <= W must
# produce byte-identical token streams to HIPFIRE_FA_WINDOW=0 (the window
# is mathematically inactive when kv_len <= W).
#
# Failure modes this catches:
#   - Off-by-one in kv_start (drops one valid key from softmax → output drift).
#   - Per-row vs uniform mask in batched verify (asymmetric for rows < B-1).
#   - Tile-clip vs explicit-mask divergence at edge cases.
#   - Incorrect tree_bias interaction with windowing.
#
# Failure modes this does NOT catch (need long-ctx + draft-aware tests):
#   - Acceptance regressions when window < kv_len (draft alignment).
#   - Per-row sliding semantics that only matter at B>1.

set -e
cd "$(dirname "$0")/../.."

DAEMON="./target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

if [ ! -x "$DAEMON" ]; then
    echo "ERROR: $DAEMON not built. Run: cargo build --release --features deltanet --example daemon -p engine" >&2
    exit 2
fi

source "$LOCK_SCRIPT"
gpu_acquire "fa-window-identity-test" || { echo "could not acquire GPU lock" >&2; exit 2; }
trap 'gpu_release 2>/dev/null || true' EXIT

# Cases: model_file | prompt | match_tokens | gen_tokens | window
# match_tokens — number of leading tokens that must match byte-identically.
# gen_tokens   — total tokens to generate (only the first `match_tokens`
#                are checked).
# Why a window of `match_tokens`: greedy decode is bit-exact at the kernel
# level when the window is mathematically inactive (kv_len <= window), but
# softmax-reduction FP rounding compounds across 32 layers and can flip an
# argmax around token ~30 even with identical kernel inputs. Lucebox's own
# `cosine_sim=1.0` short-context check is FP-aware (compares hidden state
# tensors with tolerance), not byte-identity. We approximate by limiting the
# match window to where divergence is structural rather than FP drift.
# An off-by-one in the mask diverges from token 1; FP drift from token ~25.
CASES=(
    "qwen3.5-0.8b.mq4|Tell me about France in three sentences.|10|30|4096"
    "qwen3.5-4b.mq4|Write a Python function that returns the nth prime.|10|30|4096"
    "qwen3.5-9b.mq4|What is the difference between TCP and UDP?|10|30|4096"
)

pass=0
fail=0
skip=0

run_decode() {
    local model="$1" prompt="$2" max="$3" window="$4"
    local out
    out=$(env HIPFIRE_FA_WINDOW="$window" timeout 180 "$DAEMON" <<JL 2>/dev/null
{"type":"load","model":"$model","params":{"max_seq":4096,"kv_mode":"asym3"}}
{"type":"generate","id":"r1","prompt":"$prompt","temperature":0.0,"max_tokens":$max,"repeat_penalty":1.0}
{"type":"unload"}
JL
)
    # Extract tokens as ID stream. Daemon emits {"type":"token","id":"...","text":"..."}
    # We use the text for identity check (deterministic given same model + prompt).
    echo "$out" | grep -aE '^\{"type":"token"' | sed -nE 's/.*"text":"((\\.|[^"\\])*)".*/\1/p'
}

for case in "${CASES[@]}"; do
    IFS='|' read -r model_file prompt match_n gen_n window <<< "$case"
    model_path="$MODELS_DIR/$model_file"
    if [ ! -f "$model_path" ]; then
        printf "  %-22s SKIP  (model not present)\n" "$model_file"
        skip=$((skip + 1))
        continue
    fi
    printf "  %-22s " "$model_file"
    base=$(run_decode "$model_path" "$prompt" "$gen_n" 0 | head -n "$match_n")
    win=$(run_decode "$model_path" "$prompt" "$gen_n" "$window" | head -n "$match_n")
    if [ -z "$base" ] || [ -z "$win" ]; then
        printf "FAIL  (zero tokens — daemon panic?)\n"
        fail=$((fail + 1))
        continue
    fi
    if [ "$base" = "$win" ]; then
        printf "PASS  (first %d tokens identical at window=%d inactive)\n" "$match_n" "$window"
        pass=$((pass + 1))
    else
        printf "FAIL  (mask math bug — divergence in first %d tokens)\n" "$match_n"
        diff <(printf '%s\n' "$base") <(printf '%s\n' "$win") | head -20
        fail=$((fail + 1))
    fi
done

echo
if [ "$fail" -gt 0 ]; then
    echo "FA-WINDOW IDENTITY: $fail failure(s), $pass pass, $skip skip — INVESTIGATE"
    exit 1
fi
echo "FA-WINDOW IDENTITY: $pass pass, $skip skip — clean"
