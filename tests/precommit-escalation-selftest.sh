#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — pre-commit tiny-affected verdict self-test (no GPU).
#
# Guards ONE property: a tiny-affected-gate FAIL (exit 1) must block the commit.
#
# It used to escalate to the coherence battery instead — and that battery says of
# itself that it "only fails on hard daemon/error signals ... correctness is
# assessed qualitatively on the report", so it cannot go red on the drift the
# tiny tier just measured. A definite red was therefore laundered into a green.
# That is the same shape that let `tiny-spec-gate` stay broken on master for its
# whole life (f212ae076): a red that escalates into something which only fails on
# hard errors is a red that never blocks.
#
# The test runs the hook's REAL verdict region — extracted between
# `RUN_COHERENCE=1` and the `# END tiny-affected verdict` marker — against stub
# gates, so it cannot drift from the hook by being a copy of it.
#
# Exit: 0 all cases correct, 1 a case is wrong, 2 infra.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK="$ROOT/.githooks/pre-commit"
[ -r "$HOOK" ] || { echo "precommit-escalation-selftest: no $HOOK"; exit 2; }

W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
mkdir -p "$W/tests"
awk '/^RUN_COHERENCE=1$/{p=1} p{print} /^# END tiny-affected verdict$/{exit}' \
    "$HOOK" > "$W/region.sh"
grep -q 'TINY_FAILED' "$W/region.sh" || {
    echo "precommit-escalation-selftest: could not extract the verdict region"
    echo "  (did the RUN_COHERENCE=1 / '# END tiny-affected verdict' markers move?)"
    exit 2
}

# Stub gates. The coherence battery always exits 0 — that IS the scenario: it is
# clean precisely because it cannot see output drift.
printf '#!/bin/sh\nexit ${TINY_RC:-0}\n'          > "$W/tests/tiny-affected-gate.sh"
printf '#!/bin/sh\necho "  (battery clean)"\n'    > "$W/tests/coherence-gate.sh"
chmod +x "$W/tests"/*.sh

run_case() {  # tiny_rc -> hook exit
    ( cd "$W" && TINY_RC="$1" GPU_GATES=1 PFLASH_GATE_ENV="HIPFIRE_SELFTEST=1" \
        PFLASH_GATE_NOTE="(stub)" bash ./region.sh >/dev/null 2>&1 )
    echo $?
}

fail=0
check() {  # label tiny_rc want_exit
    local got; got="$(run_case "$2")"
    if [ "$got" = "$3" ]; then
        printf '  OK    tiny exit=%s -> hook exit=%s  (%s)\n' "$2" "$got" "$1"
    else
        printf '  FAIL  tiny exit=%s -> hook exit=%s, want %s  (%s)\n' "$2" "$got" "$3" "$1"
        fail=1
    fi
}

echo "precommit-escalation-selftest: hook verdict region vs stubbed gates"
check "pass skips the battery"              0 0
check "FAIL BLOCKS even with a clean battery" 1 1
check "inconclusive escalates, does not block" 3 0
check "could-not-run escalates, does not block" 2 0

# Negative control: the assertion above is worthless unless it can fail. Revert
# the one line that carries the fix and confirm the FAIL case goes green again —
# reproducing the original bug on purpose.
sed 's/^           TINY_FAILED=1 ;;/           ;;/' "$W/region.sh" > "$W/region_old.sh"
if ! cmp -s "$W/region.sh" "$W/region_old.sh"; then
    ( cd "$W" && TINY_RC=1 GPU_GATES=1 PFLASH_GATE_ENV="HIPFIRE_SELFTEST=1" \
        PFLASH_GATE_NOTE="(stub)" bash ./region_old.sh >/dev/null 2>&1 )
    old_rc=$?
    if [ "$old_rc" = 0 ]; then
        echo "  OK    negative control: without the fix, tiny exit=1 -> hook exit=0 (the bug)"
    else
        echo "  FAIL  negative control did not reproduce the bug (got exit=$old_rc, want 0)"
        echo "        This test can no longer prove it is testing anything."
        fail=1
    fi
else
    echo "  FAIL  negative control could not patch the region — the fix line moved"
    fail=1
fi

if [ "$fail" = 0 ]; then echo "precommit-escalation-selftest: PASS"; exit 0; fi
echo "precommit-escalation-selftest: FAIL"; exit 1
