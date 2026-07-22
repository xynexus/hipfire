#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: benchmarks/calib/resident-streamed-parity.sh \
  --streamed-calib FILE --resident-model FILE --output-dir DIR \
  --artifact-stem MODEL_NAME [options]

Generate a fresh resident calibration artifact from the streamed artifact's
frozen job, compare the two mechanisms, and write evidence.json.

Options:
  --streamed-residuals FILE  Also capture and compare bounded residual probes
  --residual-probe-rows N    Residual rows when enabled (default: 16)
  --collector PATH           collect_artifacts binary
                             (default: target/release/examples/collect_artifacts)
  --coexistence PATH         hipfire-coexistence binary
                             (default: target/release/hipfire-coexistence)
  --hipfire PATH             hipfire CLI used for the shared lock
                             (default: target/release/hipfire)
  --atol F32                 Absolute comparison tolerance (default: 1e-5)
  --rtol F32                 Relative comparison tolerance (default: 5e-3)
  -h, --help                 Show this help

The output directory must be absent or empty. Existing evidence is never
overwritten. A parity mismatch is recorded in evidence.json and exits 4.
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STREAMED_CALIB=
STREAMED_RESIDUALS=
RESIDENT_MODEL=
OUTPUT_DIR=
ARTIFACT_STEM=
COLLECTOR="$ROOT/target/release/examples/collect_artifacts"
COEXISTENCE="$ROOT/target/release/hipfire-coexistence"
HIPFIRE="$ROOT/target/release/hipfire"
RESIDUAL_PROBE_ROWS=16
ATOL=1e-5
RTOL=5e-3

while (($#)); do
    case "$1" in
        --streamed-calib) STREAMED_CALIB=${2:?missing value for --streamed-calib}; shift 2 ;;
        --streamed-residuals) STREAMED_RESIDUALS=${2:?missing value for --streamed-residuals}; shift 2 ;;
        --resident-model) RESIDENT_MODEL=${2:?missing value for --resident-model}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?missing value for --output-dir}; shift 2 ;;
        --artifact-stem) ARTIFACT_STEM=${2:?missing value for --artifact-stem}; shift 2 ;;
        --collector) COLLECTOR=${2:?missing value for --collector}; shift 2 ;;
        --coexistence) COEXISTENCE=${2:?missing value for --coexistence}; shift 2 ;;
        --hipfire) HIPFIRE=${2:?missing value for --hipfire}; shift 2 ;;
        --residual-probe-rows) RESIDUAL_PROBE_ROWS=${2:?missing value for --residual-probe-rows}; shift 2 ;;
        --atol) ATOL=${2:?missing value for --atol}; shift 2 ;;
        --rtol) RTOL=${2:?missing value for --rtol}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n $STREAMED_CALIB ]] || { printf 'error: --streamed-calib is required\n' >&2; exit 2; }
[[ -n $RESIDENT_MODEL ]] || { printf 'error: --resident-model is required\n' >&2; exit 2; }
[[ -n $OUTPUT_DIR ]] || { printf 'error: --output-dir is required\n' >&2; exit 2; }
[[ -n $ARTIFACT_STEM ]] || { printf 'error: --artifact-stem is required\n' >&2; exit 2; }
[[ -f $STREAMED_CALIB ]] || { printf 'error: streamed artifact not found: %s\n' "$STREAMED_CALIB" >&2; exit 2; }
[[ -f $RESIDENT_MODEL ]] || { printf 'error: resident model not found: %s\n' "$RESIDENT_MODEL" >&2; exit 2; }
if [[ -n $STREAMED_RESIDUALS && ! -f $STREAMED_RESIDUALS ]]; then
    printf 'error: streamed residual artifact not found: %s\n' "$STREAMED_RESIDUALS" >&2
    exit 2
fi
for binary in "$COLLECTOR" "$COEXISTENCE" "$HIPFIRE"; do
    [[ -x $binary ]] || { printf 'error: executable not found: %s\n' "$binary" >&2; exit 2; }
done
for command in jq sha256sum; do
    command -v "$command" >/dev/null || { printf 'error: %s is required\n' "$command" >&2; exit 2; }
done
if [[ ! $ARTIFACT_STEM =~ ^[A-Za-z0-9][A-Za-z0-9.+-]*$ ]]; then
    printf 'error: --artifact-stem must be a canonical path-safe model stem\n' >&2
    exit 2
fi
if [[ ! $RESIDUAL_PROBE_ROWS =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: --residual-probe-rows must be a positive integer\n' >&2
    exit 2
fi
if [[ -d $OUTPUT_DIR && -n $(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    printf 'error: output directory is not empty: %s\n' "$OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

LOG="$OUTPUT_DIR/run.log"
RESIDENT_CALIB="$OUTPUT_DIR/${ARTIFACT_STEM}.resident.calib.hfq"
RESIDENT_RESIDUALS="$OUTPUT_DIR/${ARTIFACT_STEM}.resident.residuals.hfq"
CALIBRATION_COMPARISON="$OUTPUT_DIR/calibration-comparison.json"
RESIDUAL_COMPARISON="$OUTPUT_DIR/residual-comparison.json"

printf '%s resident/streamed parity start\n' "$(date --iso-8601=seconds)" | tee "$LOG"
collect_command=(
    "$COLLECTOR"
    --model "$RESIDENT_MODEL"
    --job-from "$STREAMED_CALIB"
    --output "$RESIDENT_CALIB"
)
if [[ -n $STREAMED_RESIDUALS ]]; then
    collect_command+=(
        --residual-probe-output "$RESIDENT_RESIDUALS"
        --residual-probe-rows "$RESIDUAL_PROBE_ROWS"
    )
fi
printf '%s command=' "$(date --iso-8601=seconds)" | tee -a "$LOG"
printf '%q ' "$HIPFIRE" lock run "calibration-parity-$ARTIFACT_STEM" -- "${collect_command[@]}" \
    | tee -a "$LOG"
printf '\n' | tee -a "$LOG"
"$HIPFIRE" lock run "calibration-parity-$ARTIFACT_STEM" -- "${collect_command[@]}" \
    2>&1 | tee -a "$LOG"

set +e
"$COEXISTENCE" artifact compare-calibration \
    --reference "$RESIDENT_CALIB" \
    --candidate "$STREAMED_CALIB" \
    --atol "$ATOL" \
    --rtol "$RTOL" \
    2>>"$LOG" | tee "$CALIBRATION_COMPARISON"
calibration_status=${PIPESTATUS[0]}
set -e

residual_status=0
if [[ -n $STREAMED_RESIDUALS ]]; then
    set +e
    "$COEXISTENCE" artifact compare-residuals \
        --reference "$RESIDENT_RESIDUALS" \
        --candidate "$STREAMED_RESIDUALS" \
        --atol "$ATOL" \
        --rtol "$RTOL" \
        2>>"$LOG" | tee "$RESIDUAL_COMPARISON"
    residual_status=${PIPESTATUS[0]}
    set -e
else
    printf 'null\n' >"$RESIDUAL_COMPARISON"
fi

STREAMED_SHA256=$(sha256sum "$STREAMED_CALIB" | awk '{print $1}')
RESIDENT_MODEL_SHA256=$(sha256sum "$RESIDENT_MODEL" | awk '{print $1}')
RESIDENT_SHA256=$(sha256sum "$RESIDENT_CALIB" | awk '{print $1}')
COLLECTOR_SHA256=$(sha256sum "$COLLECTOR" | awk '{print $1}')
COEXISTENCE_SHA256=$(sha256sum "$COEXISTENCE" | awk '{print $1}')
GIT_COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)
if [[ -n $(git -C "$ROOT" status --porcelain 2>/dev/null) ]]; then
    GIT_DIRTY=true
else
    GIT_DIRTY=false
fi
if ((calibration_status == 0 && residual_status == 0)); then
    STATUS=pass
else
    STATUS=fail
fi

jq -n \
    --arg status "$STATUS" \
    --arg generated_at "$(date --iso-8601=seconds)" \
    --arg streamed_calib "$STREAMED_CALIB" \
    --arg streamed_sha256 "$STREAMED_SHA256" \
    --arg resident_model "$RESIDENT_MODEL" \
    --arg resident_model_sha256 "$RESIDENT_MODEL_SHA256" \
    --arg resident_calib "$RESIDENT_CALIB" \
    --arg resident_sha256 "$RESIDENT_SHA256" \
    --arg collector "$COLLECTOR" \
    --arg collector_sha256 "$COLLECTOR_SHA256" \
    --arg coexistence "$COEXISTENCE" \
    --arg coexistence_sha256 "$COEXISTENCE_SHA256" \
    --arg git_commit "$GIT_COMMIT" \
    --argjson git_dirty "$GIT_DIRTY" \
    --argjson calibration_status "$calibration_status" \
    --argjson residual_status "$residual_status" \
    --arg streamed_residuals "$STREAMED_RESIDUALS" \
    --slurpfile calibration "$CALIBRATION_COMPARISON" \
    --slurpfile residual "$RESIDUAL_COMPARISON" \
    '{
        schema: "hipfire.calibration_parity_evidence.v2",
        status: $status,
        generated_at: $generated_at,
        mechanism: "resident_vs_layer_streamed",
        inputs: {
            streamed_calibration: {path: $streamed_calib, sha256: $streamed_sha256},
            streamed_residuals: (if $streamed_residuals == "" then null else $streamed_residuals end),
            resident_model: {path: $resident_model, sha256: $resident_model_sha256}
        },
        output: {resident_calibration: $resident_calib, sha256: $resident_sha256},
        producer: {
            git_commit: $git_commit,
            git_dirty: $git_dirty,
            collector: {path: $collector, sha256: $collector_sha256},
            coexistence: {path: $coexistence, sha256: $coexistence_sha256}
        },
        comparison: {
            calibration_exit_status: $calibration_status,
            calibration: $calibration[0],
            residual_exit_status: (if $streamed_residuals == "" then null else $residual_status end),
            residual: (if $streamed_residuals == "" then null else $residual[0] end)
        }
    }' >"$OUTPUT_DIR/evidence.json.tmp"
mv "$OUTPUT_DIR/evidence.json.tmp" "$OUTPUT_DIR/evidence.json"
printf '%s resident/streamed parity status=%s evidence=%s\n' \
    "$(date --iso-8601=seconds)" "$STATUS" "$OUTPUT_DIR/evidence.json" | tee -a "$LOG"

if [[ $STATUS != pass ]]; then
    exit 4
fi
