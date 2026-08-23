#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — self-test for tiny-affected-gate's verdict aggregation.
#
# The gates used to be chained on `[ "$status" -eq 0 ]`, so the first non-zero
# exit skipped every later gate — including exit 3, which the gate's own header
# defines as "inconclusive", not "failed". Two missing baselines were therefore
# enough to stop the state, spec and prefill gates from running at all.
#
# This runs tiny-affected-gate against STUB sub-gates whose exit codes are set
# by env vars, so the aggregation logic is checked in about a second with no GPU
# and no models. It asserts BOTH the aggregate exit code AND that every selected
# gate actually ran.
#
# Exit: 0 all cases pass, 1 some case failed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/tests"
cp "$ROOT/tests/tiny-affected-gate.sh" "$WORK/tests/"
for g in quant state spec prefill; do
    var="STUB_$(echo "$g" | tr '[:lower:]' '[:upper:]')"
    cat > "$WORK/tests/tiny-$g-gate.sh" <<EOF
#!/usr/bin/env bash
echo "  [stub] tiny-$g ran"
exit \${$var:-0}
EOF
    chmod +x "$WORK/tests/tiny-$g-gate.sh"
done

# Selects quant + state + prefill; spec stays off, so three gates should run.
printf 'crates/hipfire-model/foo.rs\ncrates/hipfire-arch-qwen35/src/qwen35/prefill.rs\n' \
    > "$WORK/files.txt"

pass=0
fail=0
check() {
    local desc=$1 want_exit=$2 want_ran=$3
    shift 3
    local out rc ran
    out="$(cd "$WORK" && env "$@" ./tests/tiny-affected-gate.sh --files-from files.txt 2>&1)"
    rc=$?
    ran="$(printf '%s' "$out" | grep -c '\[stub\]')"
    if [ "$rc" = "$want_exit" ] && [ "$ran" = "$want_ran" ]; then
        pass=$((pass + 1))
        printf '  ok    %-34s exit=%s ran=%s\n' "$desc" "$rc" "$ran"
    else
        fail=$((fail + 1))
        printf '  FAIL  %-34s exit=%s ran=%s (want exit=%s ran=%s)\n' \
            "$desc" "$rc" "$ran" "$want_exit" "$want_ran"
    fi
}

# Every case expects ran=3: a non-zero gate must not suppress the others.
check "all pass"                     0 3 STUB_QUANT=0
check "quant inconclusive (3)"       3 3 STUB_QUANT=3
check "quant failed (1)"             1 3 STUB_QUANT=1
check "state failed (1)"             1 3 STUB_STATE=1
check "failure outranks inconclusive" 1 3 STUB_QUANT=3 STUB_STATE=1
check "infra (2) outranks 3"         2 3 STUB_QUANT=2 STUB_STATE=3
check "state inconclusive alone"     3 3 STUB_STATE=3
# tiny-prefill's exit 3 means "no batched-prefill family selected", which is a
# skip rather than an inconclusive verdict, so it must NOT colour the result.
check "prefill 3 is a skip, not 3"   0 3 STUB_PREFILL=3
check "prefill failed (1)"           1 3 STUB_PREFILL=1

echo "tiny-affected-gate-selftest: pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
