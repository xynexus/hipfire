#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNNER="$ROOT/benchmarks/calib/zaya-ragged-slice-parity.sh"
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/hipfire-ragged-slice-nogpu.XXXXXX")
trap 'find "$TMP_DIR" -mindepth 1 -delete; rmdir "$TMP_DIR"' EXIT

fail() {
    printf 'zaya-ragged-slice-parity-nogpu: %s\n' "$*" >&2
    exit 1
}

# Expect the runner to reject an argument set before it runs anything.
expect_usage_error() {
    local description=$1 needle=$2
    shift 2
    local output status
    set +e
    output=$("$RUNNER" "$@" 2>&1)
    status=$?
    set -e
    [[ $status == 2 ]] || fail "$description exited $status, expected 2"
    [[ $output == *"$needle"* ]] || fail "$description did not report: $needle"
}

bash -n "$RUNNER"
[[ -x $RUNNER ]] || fail 'runner is not executable'

help=$($RUNNER --help)
# The three contracts that are load-bearing and easy to regress by "cleanup".
[[ $help == *'Existing evidence is never'* ]] || fail 'help omits no-overwrite contract'
[[ $help == *'NO LOCK WRAPPER'* ]] || fail 'help omits the self-lock contract'
[[ $help == *'KLDREF IS OFF'* ]] || fail 'help omits the kldref contract'

model=$(mktemp "$TMP_DIR/model.XXXXXX.bf16.hfq")

expect_usage_error 'invalid stem' 'canonical path-safe model stem' \
    --model "$model" --output-dir "$TMP_DIR/stem" --artifact-stem '../ZAYA1-8B' \
    --coexistence /bin/true

expect_usage_error 'missing model' 'model not found' \
    --model "$TMP_DIR/absent.hfq" --output-dir "$TMP_DIR/absent-model" \
    --artifact-stem ZAYA1-8B --coexistence /bin/true

# The guards that keep the job from passing vacuously. A width of 1 makes every
# slice one row, and fewer sequences than the slice width means the slice never
# fills -- either way no slice is ever narrower than its scratch, which is the
# only condition this job exists to test.
expect_usage_error 'slice width 1' 'no narrow slice is ever built' \
    --model "$model" --output-dir "$TMP_DIR/width1" --artifact-stem ZAYA1-8B \
    --slice-width 1 --coexistence /bin/true

expect_usage_error 'sequences below slice width' 'the slice never fills' \
    --model "$model" --output-dir "$TMP_DIR/toofew" --artifact-stem ZAYA1-8B \
    --sequences 2 --slice-width 4 --coexistence /bin/true

expect_usage_error 'non-integer sequences' 'must be a positive integer' \
    --model "$model" --output-dir "$TMP_DIR/badint" --artifact-stem ZAYA1-8B \
    --sequences 0 --coexistence /bin/true

expect_usage_error 'absent corpus' 'corpus not found' \
    --model "$model" --output-dir "$TMP_DIR/nocorpus" --artifact-stem ZAYA1-8B \
    --corpus "$TMP_DIR/absent-corpus.txt" --coexistence /bin/true

nonempty="$TMP_DIR/nonempty"
mkdir "$nonempty"
mktemp "$nonempty/existing.XXXXXX" >/dev/null
expect_usage_error 'nonempty output' 'output directory is not empty' \
    --model "$model" --output-dir "$nonempty" --artifact-stem ZAYA1-8B \
    --coexistence /bin/true

# The corpus the runner generates must actually be ragged: equal-length
# sequences make every slice full width and the whole job proves nothing. Drive
# a real generation with a stub binary, then measure the paragraphs it wrote.
corpus_dir="$TMP_DIR/corpus-shape"
"$RUNNER" --model "$model" --output-dir "$corpus_dir" --artifact-stem ZAYA1-8B \
    --sequences 4 --context 128 --slice-width 4 \
    --coexistence /bin/true >/dev/null 2>&1 || true
corpus="$corpus_dir/ragged-corpus.txt"
[[ -f $corpus ]] || fail 'runner did not generate a corpus'

# One paragraph per sequence, split on a blank line.
paragraphs=$(awk 'BEGIN{RS="";n=0} NF{n++} END{print n}' "$corpus")
[[ $paragraphs == 4 ]] || fail "generated corpus has $paragraphs paragraphs, expected 4"

# Every paragraph must be a different length, and all must stay under the
# context so the loader emits one short sample each instead of splitting one
# paragraph into several full-length ones.
lengths=$(awk 'BEGIN{RS="";ORS=" "} NF{print NF}' "$corpus")
distinct=$(printf '%s\n' $lengths | sort -u | wc -l)
[[ $distinct == 4 ]] || fail "generated corpus lengths are not all distinct: $lengths"
for length in $lengths; do
    ((length < 128)) || fail "paragraph of $length words is not under the context"
    ((length >= 2)) || fail "paragraph of $length words is too short to be a sequence"
done

printf 'zaya-ragged-slice-parity-nogpu: PASS (paragraph words: %s)\n' "$lengths"
