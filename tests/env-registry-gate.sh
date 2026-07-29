#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — enforce that env vars are declared in `hipfire-env`.
#
# Every environment variable must be declared in crates/hipfire-env so
# `hipfire help env` can list it with a description a user can act on, and so a
# read site cannot name a variable the docs have never heard of. `clippy.toml`
# carries the message; this gate applies the level.
#
# WHY AN EXPLICIT clippy INVOCATION rather than `[lints]` in Cargo.toml:
#   1. `.cargo/config.toml` sets `rustflags = ["-Aclippy::all", ...]`, which
#      blanket-allows every clippy lint workspace-wide. Flags from the `[lints]`
#      table lose to it.
#   2. Independently, `[lints.clippy] disallowed_methods = "deny"` was measured
#      NOT to take effect here even with those rustflags emptied, whereas
#      `-D` on the command line does. So `[lints]` would be a silent no-op.
# A `-D` after `--` is applied last and wins, which is what this does.
#
# `--no-deps` is required: unmigrated crates are still full of bare reads, and
# without it clippy fails on the first dependency instead of the crate under
# test.
#
# MIGRATION: a crate joins ENFORCED once all of its env reads go through the
# table. This list is the migration progress — it only grows.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENFORCED=(
    hipfire-env
    hipfire-model
)

status=0
for crate in "${ENFORCED[@]}"; do
    # Capture first, then match. Piping clippy straight into `grep -q` under
    # `set -o pipefail` reports the PIPELINE status, which is clippy's 101 when
    # the lint fires — so `if ... | grep -q` reads false exactly when a
    # violation exists, and the gate passes precisely when it should fail.
    out="$(cargo clippy -q -p "$crate" --all-targets --no-deps -- \
        -D clippy::disallowed_methods 2>&1)"
    if printf '%s\n' "$out" | grep -q 'disallowed method'; then
        echo "  FAIL $crate: undeclared env var read"
        printf '%s\n' "$out" | grep -B1 -A3 'disallowed method' | head -12
        status=1
    else
        echo "  ok   $crate"
    fi
done

# Every declared variable must be reachable from the table, and the table must
# stay parseable — cheap structural cover for the registry itself.
if ! cargo test -q -p hipfire-env >/dev/null 2>&1; then
    echo "  FAIL hipfire-env: registry tests failed"
    status=1
fi

remaining=$(grep -rl 'std::env::var' --include=*.rs crates/*/src 2>/dev/null | wc -l)
echo "env-registry-gate: enforced ${#ENFORCED[@]} crate(s); $remaining file(s) still hold un-migrated reads"
[ "$status" -eq 0 ] && echo "env-registry-gate: PASS" || echo "env-registry-gate: FAIL"
exit "$status"
