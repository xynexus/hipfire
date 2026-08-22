#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Probe a list of commits for 9B decode perf. Output per commit:
#   <hash>  <tok_s>  <short message>   (metric selected by PROBE_METRIC)
#
# YOUR WORKING TREE IS NEVER TOUCHED. Every checkout happens in a scratch git
# worktree; this script does not stash, does not check out in the main tree, and
# does not move HEAD.
#
# It used to do `git stash push -u` and then `git checkout -f <hash>` in the main
# tree. Both are unsafe here:
#   * `git stash` does not work in this repo at all — the untracked `.agents/`
#     symlink tree makes it fail, and a failed stash followed by `stash pop`
#     restores an unrelated OLDER stash over your work. AGENTS.md forbids it.
#   * because the stash silently failed, the `checkout -f` that followed then
#     DISCARDED uncommitted changes rather than being blocked by them.
#
# Env vars:
#   BENCH_MODEL        Path under ~/.hipfire/models/ (default qwen3.5-9b-mq4.hfq).
#                      Bench is dtype-agnostic — pass qwen3.5-9b-lloyd-mq3.hfq to
#                      bench Lloyd-MQ3. Decoder dtype is detected from the .hfq
#                      quant-type ID in qwen35::load_weights.
#   HIPFIRE_KV_MODE    KV-cache mode (default asym3). See bench_qwen35_speed.
#   HIPFIRE_GRAPH      Set to 1 to capture the decode loop as a graph (default 1).
#   PROBE_TARGET_DIR   Build directory (default ~/.cache/hipfire-probe-target).
#                      Deliberately NOT your ./target — probing would otherwise
#                      leave it built at some other commit. Persisting it across
#                      invocations keeps builds incremental.
#   PROBE_WORKTREE     Scratch worktree path (default a mktemp dir).
#   PROBE_BENCH_ARGS   bench_qwen35_speed args (default "--prefill 16 --warmup 3
#                      --gen 30", i.e. decode). For PREFILL work, pass something
#                      like "--prefill 2059 --warmup 1 --gen 4" together with
#                      PROBE_METRIC=prefill_tok_s.
#   PROBE_METRIC       Which SUMMARY field to report (default gen_tok_s).
set -u

BENCH_MODEL="${BENCH_MODEL:-qwen3.5-9b-mq4.hfq}"
COMMITS=("$@")

if [ "${#COMMITS[@]}" -eq 0 ]; then
    echo "usage: $0 <commit> [commit ...]" >&2
    echo "  probes each commit's 9B decode tok/s; see header for env vars" >&2
    exit 2
fi

# Resolve to full hashes up front so a bad ref fails before any building, and so
# a branch name cannot drift under us mid-run.
RESOLVED=()
for h in "${COMMITS[@]}"; do
    if ! full=$(git rev-parse --verify --quiet "$h^{commit}"); then
        echo "$h  BAD_REF" >&2
        exit 2
    fi
    RESOLVED+=("$full")
done

TARGET_DIR="${PROBE_TARGET_DIR:-$HOME/.cache/hipfire-probe-target}"
mkdir -p "$TARGET_DIR"

if [ -n "${PROBE_WORKTREE:-}" ]; then
    WT="$PROBE_WORKTREE"
    OWN_TMP=""
else
    OWN_TMP=$(mktemp -d -t hipfire-probe-XXXXXX)
    WT="$OWN_TMP/wt"
fi

cleanup() {
    git worktree remove --force "$WT" >/dev/null 2>&1 || true
    [ -n "$OWN_TMP" ] && rm -rf "$OWN_TMP"
    git worktree prune >/dev/null 2>&1 || true
}
trap cleanup EXIT

if ! git worktree add --detach "$WT" "${RESOLVED[0]}" >/dev/null 2>&1; then
    echo "WORKTREE_FAIL: could not create scratch worktree at $WT" >&2
    exit 1
fi

for h in "${RESOLVED[@]}"; do
    msg=$(git show --no-patch --format="%s" "$h" | head -c 50)
    echo -n "${h:0:9}  "
    # -f is safe HERE: it only forces the scratch worktree, never your tree.
    # Cargo may have dirtied Cargo.lock inside the worktree on the last pass.
    if ! git -C "$WT" checkout -f --detach "$h" >/dev/null 2>&1; then
        echo "CHECKOUT_FAIL  $msg"
        continue
    fi
    rm -f "$TARGET_DIR/release/examples/bench_qwen35_speed"
    if ! (cd "$WT" && CARGO_TARGET_DIR="$TARGET_DIR" \
            cargo build --release --features deltanet \
            -p hipfire-runtime --example bench_qwen35_speed) >/tmp/probe_build.log 2>&1; then
        echo "BUILD_FAIL  $msg"
        continue
    fi
    out=$(HIPFIRE_KV_MODE="${HIPFIRE_KV_MODE:-asym3}" HIPFIRE_GRAPH="${HIPFIRE_GRAPH:-1}" \
        "$TARGET_DIR/release/examples/bench_qwen35_speed" \
        "$HOME/.hipfire/models/$BENCH_MODEL" \
        ${PROBE_BENCH_ARGS:---prefill 16 --warmup 3 --gen 30} 2>&1)
    tok_s=$(echo "$out" | grep -oE "${PROBE_METRIC:-gen_tok_s}=[0-9.]+" | head -1 | sed "s/${PROBE_METRIC:-gen_tok_s}=//")
    if [ -z "$tok_s" ]; then
        echo "BENCH_FAIL  $msg"
    else
        printf '%7s tok/s  %s\n' "$tok_s" "$msg"
    fi
done
