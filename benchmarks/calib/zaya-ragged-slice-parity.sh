#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: benchmarks/calib/zaya-ragged-slice-parity.sh \
  --model DIR --output-dir DIR --artifact-stem MODEL_NAME [options]

--model is a Hugging Face SNAPSHOT DIRECTORY (one holding config.json) or a
cache root (one holding snapshots/). Streamed calibration reads the raw
checkpoint, not a packed container, so an .hfq artifact is NOT a valid source
however convenient it looks sitting next to one.

Run one ZAYA streamed calibration job at two slice widths over a RAGGED corpus
and compare the two artifacts. Catches capture bugs that only appear when a
position slice is narrower than the scratch it was allocated for.

Why ragged: the planner emits rows position-major, so a slice holds one token
per sequence that still has a token at that position. When every sequence is
the same length, every slice is full width and a stride bug is invisible --
which is exactly why the uniform-width parity runs passed while dense captures
were reading stale scratch. Sequences of DIFFERENT lengths make the slice
narrow as the short ones run out, which is the case that must be proven.

The reference run uses --sequence-batch 1, where every slice holds exactly one
row, so it cannot express a row-stride error at all. The candidate uses
--slice-width (default 4). Any real difference beyond float reassociation is a
capture defect.

Options:
  --corpus FILE          Ragged corpus (default: generated into the output dir)
  --sequences N          Sequences to calibrate (default: 4)
  --context N            Max tokens per sequence (default: 128)
  --slice-width N        Candidate --sequence-batch (default: 4, must be > 1)
  --coexistence PATH     hipfire-coexistence binary
                         (default: target/release/hipfire-coexistence)
  --atol F32             Absolute comparison tolerance (default: 1e-5)
  --rtol F32             Relative comparison tolerance (default: 5e-3)
  --max-mismatch-fraction F
                         Share of compared values allowed to differ
                         (default: 0.001). See VERDICT below.
  -h, --help             Show this help

VERDICT: the run passes when no expert tensor differs, no value is non-finite,
no tensor is missing or structurally wrong, and the share of differing values
is within --max-mismatch-fraction. It deliberately does NOT pass/fail on the
comparator's own exit status. Two slice widths sum the same rows in a different
order, so a bf16 Hessian off-diagonal that nearly cancels differs by a few ULPs
on a near-zero value: negligible absolutely, huge relatively (measured on
ZAYA1-8B: abs 0.125 at rel 1.9). No atol/rtol separates that from corruption.
Density does -- reassociation is a sparse scatter in dense Hessians only, while
a stride defect rewrites whole rows and corrupts nearly every value, including
the imatrix. Tolerances stay tight so the comparator remains a sensitive
detector; the fraction and the structural invariants are the verdict.

The output directory must be absent or empty. Existing evidence is never
overwritten. A parity mismatch is recorded in evidence.json and exits 4.

NO LOCK WRAPPER: `coexistence calibrate` acquires the shared GPU lock itself
and blocks until it wins. Wrapping it in `hipfire lock run` deadlocks silently
against the lock this script already holds, so both runs are invoked directly.
Do not "fix" that by adding a wrapper.

KLDREF IS OFF in both runs. The lm_head.kldref_* position map holds the same
set of rows in a different order at different geometries, and the comparator
diffs positionally, so any two slice widths disagree there for reasons that are
not numerical. Leaving it on would report a wholesale mismatch on every run.
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=
OUTPUT_DIR=
ARTIFACT_STEM=
CORPUS=
SEQUENCES=4
CONTEXT=128
SLICE_WIDTH=4
COEXISTENCE="$ROOT/target/release/hipfire-coexistence"
ATOL=1e-5
RTOL=5e-3
MAX_MISMATCH_FRACTION=0.001

while (($#)); do
    case "$1" in
        --model) MODEL=${2:?missing value for --model}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?missing value for --output-dir}; shift 2 ;;
        --artifact-stem) ARTIFACT_STEM=${2:?missing value for --artifact-stem}; shift 2 ;;
        --corpus) CORPUS=${2:?missing value for --corpus}; shift 2 ;;
        --sequences) SEQUENCES=${2:?missing value for --sequences}; shift 2 ;;
        --context) CONTEXT=${2:?missing value for --context}; shift 2 ;;
        --slice-width) SLICE_WIDTH=${2:?missing value for --slice-width}; shift 2 ;;
        --coexistence) COEXISTENCE=${2:?missing value for --coexistence}; shift 2 ;;
        --atol) ATOL=${2:?missing value for --atol}; shift 2 ;;
        --rtol) RTOL=${2:?missing value for --rtol}; shift 2 ;;
        --max-mismatch-fraction) MAX_MISMATCH_FRACTION=${2:?missing value for --max-mismatch-fraction}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n $MODEL ]] || { printf 'error: --model is required\n' >&2; exit 2; }
[[ -n $OUTPUT_DIR ]] || { printf 'error: --output-dir is required\n' >&2; exit 2; }
[[ -n $ARTIFACT_STEM ]] || { printf 'error: --artifact-stem is required\n' >&2; exit 2; }
# Mirror resolve_hf_snapshot() so a wrong source fails here in milliseconds
# instead of after the run has taken the GPU lock and started loading.
if [[ ! -d $MODEL ]]; then
    if [[ $MODEL == *.hfq ]]; then
        printf 'error: --model must be a Hugging Face snapshot directory, not an .hfq: %s\n' \
            "$MODEL" >&2
    else
        printf 'error: model directory not found: %s\n' "$MODEL" >&2
    fi
    exit 2
fi
if [[ ! -f $MODEL/config.json && ! -d $MODEL/snapshots ]]; then
    printf 'error: %s is neither a Hugging Face snapshot nor a cache root\n' "$MODEL" >&2
    exit 2
fi
[[ -x $COEXISTENCE ]] || { printf 'error: executable not found: %s\n' "$COEXISTENCE" >&2; exit 2; }
for command in jq sha256sum; do
    command -v "$command" >/dev/null || { printf 'error: %s is required\n' "$command" >&2; exit 2; }
done
if [[ ! $ARTIFACT_STEM =~ ^[A-Za-z0-9][A-Za-z0-9.+-]*$ ]]; then
    printf 'error: --artifact-stem must be a canonical path-safe model stem\n' >&2
    exit 2
fi
for pair in "sequences:$SEQUENCES" "context:$CONTEXT" "slice-width:$SLICE_WIDTH"; do
    if [[ ! ${pair#*:} =~ ^[1-9][0-9]*$ ]]; then
        printf 'error: --%s must be a positive integer\n' "${pair%%:*}" >&2
        exit 2
    fi
done
if [[ -n $CORPUS && ! -f $CORPUS ]]; then
    printf 'error: corpus not found: %s\n' "$CORPUS" >&2
    exit 2
fi

# A vacuous pass is worse than a failure: if the candidate geometry cannot
# produce a slice narrower than its scratch, the run proves nothing. The engine
# sizes scratch as min(sequence_batch, max_rows, sequences), so a width of 1 --
# or more sequences requested than the slice holds -- means every slice is
# either width 1 or full width, and the narrow-slice case is never exercised.
if ((SLICE_WIDTH < 2)); then
    printf 'error: --slice-width must be > 1 or no narrow slice is ever built\n' >&2
    exit 2
fi
if ((SEQUENCES < SLICE_WIDTH)); then
    printf 'error: --sequences (%d) must be >= --slice-width (%d) or the slice never fills\n' \
        "$SEQUENCES" "$SLICE_WIDTH" >&2
    exit 2
fi
if [[ -d $OUTPUT_DIR && -n $(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    printf 'error: output directory is not empty: %s\n' "$OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

# Emit `SEQUENCES` paragraphs whose lengths differ by construction. The corpus
# loader splits on a blank line, accumulates tokens within a paragraph, and
# pushes the remainder as a short sample -- so a paragraph under `context`
# tokens becomes exactly one sequence of its own length. Halving the word count
# each time keeps every paragraph well under the context while making the
# lengths maximally unequal, which is what forces narrow slices.
generate_ragged_corpus() {
    local path=$1 index words line
    local -a vocab=(calibration ragged slice parity token stream boundary router)
    : >"$path"
    # Start at a quarter of the context, not half: these are ordinary words but a
    # tokenizer may still split one into two, and a paragraph that reaches
    # `context` tokens would be cut into a full-length sample plus a tail, which
    # changes the sequence count and the length ladder.
    words=$((CONTEXT / 4))
    for ((index = 0; index < SEQUENCES; index++)); do
        ((words < 2)) && words=2
        line=
        for ((word = 0; word < words; word++)); do
            line+="${vocab[word % ${#vocab[@]}]} "
        done
        printf '%s\n\n' "${line% }" >>"$path"
        words=$((words / 2))
    done
}

GENERATED_CORPUS=false
if [[ -z $CORPUS ]]; then
    CORPUS="$OUTPUT_DIR/ragged-corpus.txt"
    generate_ragged_corpus "$CORPUS"
    GENERATED_CORPUS=true
fi

LOG="$OUTPUT_DIR/run.log"
REFERENCE_CALIB="$OUTPUT_DIR/${ARTIFACT_STEM}.slice1.calib.hfq"
CANDIDATE_CALIB="$OUTPUT_DIR/${ARTIFACT_STEM}.slice${SLICE_WIDTH}.calib.hfq"
COMPARISON="$OUTPUT_DIR/calibration-comparison.json"

printf '%s zaya ragged-slice parity start\n' "$(date --iso-8601=seconds)" | tee "$LOG"

# Both runs must differ ONLY in --sequence-batch. A shared sampling seed keeps
# the corpus selection identical, and --no-resume stops a stale checkpoint from
# either run being replayed into the other.
run_calibrate() {
    local width=$1 output=$2
    local -a command=(
        "$COEXISTENCE" calibrate
        --model "$MODEL"
        --corpus "$CORPUS"
        --output "$output"
        --sequences "$SEQUENCES"
        --context "$CONTEXT"
        --sequence-batch "$width"
        --sampling-seed 0
        --no-resume
        --no-kldref
        --min-expert-activations 1
        --required-expert-fraction 0.0001
        --expert-coverage-policy preserve-undercovered
    )
    printf '%s command=' "$(date --iso-8601=seconds)" | tee -a "$LOG"
    printf '%q ' "${command[@]}" | tee -a "$LOG"
    printf '\n' | tee -a "$LOG"
    "${command[@]}" 2>&1 | tee -a "$LOG"
}

run_calibrate 1 "$REFERENCE_CALIB"
run_calibrate "$SLICE_WIDTH" "$CANDIDATE_CALIB"

set +e
"$COEXISTENCE" artifact compare-calibration \
    --reference "$REFERENCE_CALIB" \
    --candidate "$CANDIDATE_CALIB" \
    --atol "$ATOL" \
    --rtol "$RTOL" \
    2>>"$LOG" | tee "$COMPARISON"
comparison_status=${PIPESTATUS[0]}
set -e

# Per-value tolerance CANNOT be the verdict here, and this is the subtle part of
# the whole job. Two slice widths sum the same rows in a different order, so a
# bf16-stored Hessian off-diagonal that nearly cancels lands a few ULPs apart on
# a near-zero value -- tiny absolute, enormous relative. Measured on ZAYA1-8B:
# max_abs_error 0.125 at max_rel_error 1.9. No fixed atol/rtol separates that
# from real corruption, and loosening rtol past 1.0 would excuse anything.
#
# What DOES separate them is density and location. Reassociation is a sparse
# scatter confined to dense Hessians. The stride defect this job exists to catch
# reads whole rows from stale scratch, so it corrupts essentially every value of
# every affected tensor -- percent-scale, not 0.001%-scale, and it reaches the
# imatrix too. So gate on the mismatch FRACTION plus the structural invariants,
# and keep the tolerance tight so it stays a sensitive detector rather than a
# verdict.
EXPERT_MISMATCHES=$(jq '[.tensor_mismatches[]? | select(.name | test("expert"; "i"))] | length' "$COMPARISON")
NON_FINITE=$(jq '.non_finite_values // 0' "$COMPARISON")
STRUCTURAL=$(jq '((.structural_errors//[])|length) + ((.missing_from_reference//[])|length) + ((.missing_from_candidate//[])|length)' "$COMPARISON")
MISMATCH_FRACTION=$(jq 'if (.compared_values // 0) > 0 then (.mismatched_values / .compared_values) else 1 end' "$COMPARISON")
FRACTION_OK=$(jq -n --argjson f "$MISMATCH_FRACTION" --argjson b "$MAX_MISMATCH_FRACTION" 'if $f <= $b then 1 else 0 end')

for check in "expert tensors mismatched:$EXPERT_MISMATCHES" "non-finite values:$NON_FINITE" \
             "structural errors:$STRUCTURAL"; do
    if [[ ${check#*:} != 0 ]]; then
        printf '%s FAIL %s = %s (not reassociation)\n' \
            "$(date --iso-8601=seconds)" "${check%%:*}" "${check#*:}" | tee -a "$LOG"
    fi
done
if [[ $FRACTION_OK != 1 ]]; then
    printf '%s FAIL mismatch fraction %s exceeds budget %s\n' \
        "$(date --iso-8601=seconds)" "$MISMATCH_FRACTION" "$MAX_MISMATCH_FRACTION" | tee -a "$LOG"
fi

# The model is a directory, so hash what identifies the weights rather than the
# weights themselves: the safetensors index names every shard and its offsets.
# config.json is the fallback for a checkpoint stored as a single shard.
MODEL_FINGERPRINT_FILE=$(
    find "$MODEL" -maxdepth 3 \( -name 'model.safetensors.index.json' -o -name 'config.json' \) \
        -print 2>/dev/null | sort | head -1
)
if [[ -n $MODEL_FINGERPRINT_FILE ]]; then
    MODEL_SHA256=$(sha256sum "$MODEL_FINGERPRINT_FILE" | awk '{print $1}')
else
    MODEL_FINGERPRINT_FILE=
    MODEL_SHA256=
fi
CORPUS_SHA256=$(sha256sum "$CORPUS" | awk '{print $1}')
REFERENCE_SHA256=$(sha256sum "$REFERENCE_CALIB" | awk '{print $1}')
CANDIDATE_SHA256=$(sha256sum "$CANDIDATE_CALIB" | awk '{print $1}')
COEXISTENCE_SHA256=$(sha256sum "$COEXISTENCE" | awk '{print $1}')
GIT_COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)
if [[ -n $(git -C "$ROOT" status --porcelain 2>/dev/null) ]]; then
    GIT_DIRTY=true
else
    GIT_DIRTY=false
fi
if [[ $EXPERT_MISMATCHES == 0 && $NON_FINITE == 0 && $STRUCTURAL == 0 && $FRACTION_OK == 1 ]]; then
    STATUS=pass
else
    STATUS=fail
fi

jq -n \
    --arg status "$STATUS" \
    --arg generated_at "$(date --iso-8601=seconds)" \
    --arg model "$MODEL" \
    --arg model_sha256 "$MODEL_SHA256" \
    --arg model_fingerprint "$MODEL_FINGERPRINT_FILE" \
    --arg corpus "$CORPUS" \
    --arg corpus_sha256 "$CORPUS_SHA256" \
    --argjson corpus_generated "$GENERATED_CORPUS" \
    --arg reference_calib "$REFERENCE_CALIB" \
    --arg reference_sha256 "$REFERENCE_SHA256" \
    --arg candidate_calib "$CANDIDATE_CALIB" \
    --arg candidate_sha256 "$CANDIDATE_SHA256" \
    --arg coexistence "$COEXISTENCE" \
    --arg coexistence_sha256 "$COEXISTENCE_SHA256" \
    --arg git_commit "$GIT_COMMIT" \
    --argjson git_dirty "$GIT_DIRTY" \
    --argjson comparison_status "$comparison_status" \
    --argjson expert_mismatches "$EXPERT_MISMATCHES" \
    --argjson non_finite "$NON_FINITE" \
    --argjson structural "$STRUCTURAL" \
    --argjson mismatch_fraction "$MISMATCH_FRACTION" \
    --argjson max_mismatch_fraction "$MAX_MISMATCH_FRACTION" \
    --argjson sequences "$SEQUENCES" \
    --argjson context "$CONTEXT" \
    --argjson slice_width "$SLICE_WIDTH" \
    --arg atol "$ATOL" \
    --arg rtol "$RTOL" \
    --slurpfile comparison "$COMPARISON" \
    '{
        schema: "hipfire.ragged_slice_parity_evidence.v1",
        status: $status,
        generated_at: $generated_at,
        mechanism: "layer_streamed_slice_width_1_vs_n_over_ragged_corpus",
        inputs: {
            model: {
                path: $model,
                fingerprint_file: (if $model_fingerprint == "" then null else $model_fingerprint end),
                sha256: (if $model_sha256 == "" then null else $model_sha256 end)
            },
            corpus: {path: $corpus, sha256: $corpus_sha256, generated: $corpus_generated}
        },
        geometry: {
            sequences: $sequences,
            context: $context,
            reference_sequence_batch: 1,
            candidate_sequence_batch: $slice_width,
            kldref: false
        },
        output: {
            reference_calibration: {path: $reference_calib, sha256: $reference_sha256},
            candidate_calibration: {path: $candidate_calib, sha256: $candidate_sha256}
        },
        producer: {
            git_commit: $git_commit,
            git_dirty: $git_dirty,
            coexistence: {path: $coexistence, sha256: $coexistence_sha256}
        },
        tolerances: {atol: $atol, rtol: $rtol},
        comparison: {
            calibration_exit_status: $comparison_status,
            expert_tensor_mismatches: $expert_mismatches,
            non_finite_values: $non_finite,
            structural_errors: $structural,
            mismatch_fraction: $mismatch_fraction,
            max_mismatch_fraction: $max_mismatch_fraction,
            calibration: $comparison[0]
        }
    }' >"$OUTPUT_DIR/evidence.json.tmp"
mv "$OUTPUT_DIR/evidence.json.tmp" "$OUTPUT_DIR/evidence.json"
printf '%s zaya ragged-slice parity status=%s evidence=%s\n' \
    "$(date --iso-8601=seconds)" "$STATUS" "$OUTPUT_DIR/evidence.json" | tee -a "$LOG"

if [[ $STATUS != pass ]]; then
    exit 4
fi
