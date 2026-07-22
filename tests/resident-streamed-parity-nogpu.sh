#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNNER="$ROOT/benchmarks/calib/resident-streamed-parity.sh"
TMP_DIR=$(mktemp -d /tmp/hipfire-resident-parity-nogpu.XXXXXX)
trap 'find "$TMP_DIR" -mindepth 1 -delete; rmdir "$TMP_DIR"' EXIT

fail() {
    printf 'resident-streamed-parity-nogpu: %s\n' "$*" >&2
    exit 1
}

bash -n "$RUNNER"
help=$($RUNNER --help)
[[ $help == *'Existing evidence is never'* ]] || fail 'help omits no-overwrite contract'

streamed=$(mktemp "$TMP_DIR/streamed.XXXXXX.calib.hfq")
resident_model=$(mktemp "$TMP_DIR/resident.XXXXXX.bf16.hfq")

set +e
invalid_stem_output=$(
    "$RUNNER" \
        --streamed-calib "$streamed" \
        --resident-model "$resident_model" \
        --output-dir "$TMP_DIR/invalid-stem-output" \
        --artifact-stem '../Qwen3.5-0.8B' \
        --collector /bin/true \
        --coexistence /bin/true \
        --hipfire /bin/true 2>&1
)
invalid_stem_status=$?
set -e
[[ $invalid_stem_status == 2 ]] || fail "invalid stem exited $invalid_stem_status, expected 2"
[[ $invalid_stem_output == *'canonical path-safe model stem'* ]] \
    || fail 'invalid stem did not report the canonical-name contract'

nonempty="$TMP_DIR/nonempty"
mkdir "$nonempty"
mktemp "$nonempty/existing.XXXXXX" >/dev/null
set +e
nonempty_output=$(
    "$RUNNER" \
        --streamed-calib "$streamed" \
        --resident-model "$resident_model" \
        --output-dir "$nonempty" \
        --artifact-stem Qwen3.5-0.8B \
        --collector /bin/true \
        --coexistence /bin/true \
        --hipfire /bin/true 2>&1
)
nonempty_status=$?
set -e
[[ $nonempty_status == 2 ]] || fail "nonempty output exited $nonempty_status, expected 2"
[[ $nonempty_output == *'output directory is not empty'* ]] \
    || fail 'nonempty output did not refuse evidence reuse'

printf 'resident-streamed-parity-nogpu: PASS\n'
