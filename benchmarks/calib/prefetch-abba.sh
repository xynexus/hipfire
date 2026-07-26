#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: benchmarks/calib/prefetch-abba.sh \
  --model SAFETENSORS_DIR --corpus CORPUS --output-dir DIR \
  --artifact-stem MODEL_NAME [options]

Run fresh two-layer native-calibration trials in ABBA order (off,on,on,off),
then emit evidence.json with matched hashes and layer-1 timing summaries.

Options:
  --binary PATH             hipfire-coexistence binary
                            (default: target/release/hipfire-coexistence)
  --order CSV               Trial order using off/on (default: off,on,on,off)
  --prefetch-bytes N        On-trial byte budget (default: 17179869184)
  --sequences N             Independent samples (default: 8)
  --context N               Tokens per sample (default: 256)
  --sequence-batch N        Samples per batch (default: 8)
  --time-tile N             Time tile (default: 32)
  --max-rows N              Hessian rows (default: 512)
  -h, --help                Show this help

The native calibrator owns the shared hipfire lock for each trial. The output
directory must be absent or empty; trials are deliberately fresh and are not
resumable production artifacts.
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=
CORPUS=
OUTPUT_DIR=
ARTIFACT_STEM=
COEXISTENCE_BIN="$ROOT/target/release/hipfire-coexistence"
ORDER=off,on,on,off
PREFETCH_BYTES=17179869184
SEQUENCES=8
CONTEXT=256
SEQUENCE_BATCH=8
TIME_TILE=32
MAX_ROWS=512

while (($#)); do
    case "$1" in
        --model) MODEL=${2:?missing value for --model}; shift 2 ;;
        --corpus) CORPUS=${2:?missing value for --corpus}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?missing value for --output-dir}; shift 2 ;;
        --artifact-stem) ARTIFACT_STEM=${2:?missing value for --artifact-stem}; shift 2 ;;
        --binary) COEXISTENCE_BIN=${2:?missing value for --binary}; shift 2 ;;
        --order) ORDER=${2:?missing value for --order}; shift 2 ;;
        --prefetch-bytes) PREFETCH_BYTES=${2:?missing value for --prefetch-bytes}; shift 2 ;;
        --sequences) SEQUENCES=${2:?missing value for --sequences}; shift 2 ;;
        --context) CONTEXT=${2:?missing value for --context}; shift 2 ;;
        --sequence-batch) SEQUENCE_BATCH=${2:?missing value for --sequence-batch}; shift 2 ;;
        --time-tile) TIME_TILE=${2:?missing value for --time-tile}; shift 2 ;;
        --max-rows) MAX_ROWS=${2:?missing value for --max-rows}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n $MODEL ]] || { printf 'error: --model is required\n' >&2; usage >&2; exit 2; }
[[ -n $CORPUS ]] || { printf 'error: --corpus is required\n' >&2; usage >&2; exit 2; }
[[ -n $OUTPUT_DIR ]] || { printf 'error: --output-dir is required\n' >&2; usage >&2; exit 2; }
[[ -n $ARTIFACT_STEM ]] || {
    printf 'error: --artifact-stem is required\n' >&2
    usage >&2
    exit 2
}
[[ -d "$MODEL" ]] || { printf 'error: model directory not found: %s\n' "$MODEL" >&2; exit 2; }
[[ -f "$CORPUS" ]] || { printf 'error: corpus not found: %s\n' "$CORPUS" >&2; exit 2; }
[[ -x "$COEXISTENCE_BIN" ]] || {
    printf 'error: executable not found: %s\n' "$COEXISTENCE_BIN" >&2
    exit 2
}
command -v jq >/dev/null || { printf 'error: jq is required\n' >&2; exit 2; }
command -v md5sum >/dev/null || { printf 'error: md5sum is required\n' >&2; exit 2; }
command -v sha256sum >/dev/null || { printf 'error: sha256sum is required\n' >&2; exit 2; }

if [[ ! $ARTIFACT_STEM =~ ^[A-Za-z0-9][A-Za-z0-9.+-]*$ ]]; then
    printf 'error: --artifact-stem must be a canonical path-safe model stem\n' >&2
    exit 2
fi
for numeric in PREFETCH_BYTES SEQUENCES CONTEXT SEQUENCE_BATCH TIME_TILE MAX_ROWS; do
    if [[ ! ${!numeric} =~ ^[0-9]+$ ]] || (( ${!numeric} == 0 )); then
        printf 'error: %s must be a positive integer\n' "$numeric" >&2
        exit 2
    fi
done
if ((SEQUENCE_BATCH > SEQUENCES)); then
    printf 'error: --sequence-batch cannot exceed --sequences\n' >&2
    exit 2
fi

IFS=, read -r -a MODES <<<"$ORDER"
off_count=0
on_count=0
for mode in "${MODES[@]}"; do
    case "$mode" in
        off) off_count=$((off_count + 1)) ;;
        on) on_count=$((on_count + 1)) ;;
        *) printf 'error: --order entries must be off or on, got %s\n' "$mode" >&2; exit 2 ;;
    esac
done
if ((off_count == 0 || on_count == 0)); then
    printf 'error: --order must contain at least one off and one on trial\n' >&2
    exit 2
fi

if [[ -d "$OUTPUT_DIR" ]] && [[ -n $(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    printf 'error: output directory is not empty: %s\n' "$OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

CORPUS_MD5=$(md5sum "$CORPUS" | awk '{print $1}')
CORPUS_SHA256=$(sha256sum "$CORPUS" | awk '{print $1}')
BINARY_SHA256=$(sha256sum "$COEXISTENCE_BIN" | awk '{print $1}')
GIT_COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)
if [[ -n $(git -C "$ROOT" status --porcelain 2>/dev/null) ]]; then
    GIT_DIRTY=true
else
    GIT_DIRTY=false
fi

trial=0
for mode in "${MODES[@]}"; do
    trial=$((trial + 1))
    printf -v trial_tag 'trial-%02d-%s' "$trial" "$mode"
    trial_dir="$OUTPUT_DIR/$trial_tag"
    mkdir -p "$trial_dir/boundary"
    artifact="$trial_dir/${ARTIFACT_STEM}-prefetch-${mode}-t$(printf '%02d' "$trial").calib.hfq"
    if [[ $mode == on ]]; then
        budget=$PREFETCH_BYTES
    else
        budget=0
    fi
    command=(
        "$COEXISTENCE_BIN" calibrate
        --model "$MODEL"
        --corpus "$CORPUS"
        --output "$artifact"
        --sequences "$SEQUENCES"
        --context "$CONTEXT"
        --sequence-batch "$SEQUENCE_BATCH"
        --time-tile "$TIME_TILE"
        --max-rows "$MAX_ROWS"
        --no-kldref
        --boundary-dir "$trial_dir/boundary"
        --pause-after-layers 2
        --layer-prefetch-bytes "$budget"
    )
    printf -v command_shell '%q ' "${command[@]}"
    started_at=$(date -Is)
    started_epoch=$(date +%s)
    awk '/^(MemAvailable|Cached|SwapFree):/ {printf "%s=%s ", $1, $2} END {print ""}' \
        /proc/meminfo >"$trial_dir/host-before.txt"
    "${command[@]}" 2>&1 | tee "$trial_dir/run.log"
    finished_epoch=$(date +%s)
    finished_at=$(date -Is)

    artifact_name=$(basename "$artifact")
    progress0="$trial_dir/.${artifact_name}.layer-0000.progress.json"
    progress1="$trial_dir/.${artifact_name}.layer-0001.progress.json"
    [[ -f "$progress0" && -f "$progress1" ]] || {
        printf 'error: trial %s did not produce both layer progress files\n' "$trial_tag" >&2
        exit 3
    }
    host_before=$(<"$trial_dir/host-before.txt")
    jq -n \
        --arg mode "$mode" \
        --argjson trial "$trial" \
        --arg artifact "$artifact" \
        --arg command "$command_shell" \
        --arg started_at "$started_at" \
        --arg finished_at "$finished_at" \
        --arg host_before "$host_before" \
        --argjson elapsed_seconds "$((finished_epoch - started_epoch))" \
        --slurpfile layer0 "$progress0" \
        --slurpfile layer1 "$progress1" \
        '{
            trial: $trial,
            mode: $mode,
            artifact: $artifact,
            command: $command,
            started_at: $started_at,
            finished_at: $finished_at,
            elapsed_seconds: $elapsed_seconds,
            host_before: $host_before,
            run_fingerprint: $layer1[0].run_fingerprint,
            engine_build: $layer1[0].engine_build,
            layer0: {
                part_hash: $layer0[0].part_hash,
                read_ledger: {
                    planned_count: ($layer0[0].read_ledger.planned_logical | length),
                    consumed_count: ($layer0[0].read_ledger.consumed_logical | length),
                    missing_count: ($layer0[0].read_ledger.missing_logical | length),
                    duplicate_count: ($layer0[0].read_ledger.duplicate_logical | length),
                    logical_bytes_read: $layer0[0].read_ledger.logical_bytes_read
                },
                timing: $layer0[0].timing
            },
            layer1: {
                part_hash: $layer1[0].part_hash,
                read_ledger: {
                    planned_count: ($layer1[0].read_ledger.planned_logical | length),
                    consumed_count: ($layer1[0].read_ledger.consumed_logical | length),
                    missing_count: ($layer1[0].read_ledger.missing_logical | length),
                    duplicate_count: ($layer1[0].read_ledger.duplicate_logical | length),
                    logical_bytes_read: $layer1[0].read_ledger.logical_bytes_read
                },
                timing: $layer1[0].timing
            }
        }' >"$trial_dir/trial.json"
done

jq -s \
    --arg model "$MODEL" \
    --arg corpus "$CORPUS" \
    --arg corpus_md5 "$CORPUS_MD5" \
    --arg corpus_sha256 "$CORPUS_SHA256" \
    --arg binary "$COEXISTENCE_BIN" \
    --arg binary_sha256 "$BINARY_SHA256" \
    --arg git_commit "$GIT_COMMIT" \
    --argjson git_dirty "$GIT_DIRTY" \
    --arg order "$ORDER" \
    --argjson prefetch_bytes "$PREFETCH_BYTES" \
    --argjson sequences "$SEQUENCES" \
    --argjson context "$CONTEXT" \
    --argjson sequence_batch "$SEQUENCE_BATCH" \
    --argjson time_tile "$TIME_TILE" \
    --argjson max_rows "$MAX_ROWS" '
    def median:
        sort as $s | ($s | length) as $n |
        if $n == 0 then null
        elif ($n % 2) == 1 then $s[($n / 2) | floor]
        else (($s[$n / 2 - 1] + $s[$n / 2]) / 2)
        end;
    def summary($mode):
        [.[] | select(.mode == $mode)] as $rows |
        {
            trials: ($rows | length),
            median_elapsed_seconds: ([$rows[].elapsed_seconds] | median),
            median_total_before_checkpoint_us:
                ([$rows[].layer1.timing.total_before_checkpoint_us] | median),
            median_foreground_load_us:
                ([$rows[] | (.layer1.timing.prefetch_wait_us + .layer1.timing.load_upload_us)] | median),
            median_background_read_us:
                ([$rows[].layer1.timing.prefetch_read_us] | median),
            median_execute_us: ([$rows[].layer1.timing.execute_us] | median),
            median_staged_bytes: ([$rows[].layer1.timing.prefetch_staged_bytes] | median)
        };
    . as $trials |
    (summary("off")) as $off |
    (summary("on")) as $on |
    {
        schema: "hipfire.calibration_prefetch_abba.v1",
        evidence_scope: "two-layer timing and exact mechanism parity; not quality evidence",
        model: $model,
        corpus: {path: $corpus, md5: $corpus_md5, sha256: $corpus_sha256},
        engine: {
            binary: $binary,
            binary_sha256: $binary_sha256,
            git_commit: $git_commit,
            git_dirty: $git_dirty
        },
        recipe: {
            order: ($order | split(",")),
            prefetch_bytes: $prefetch_bytes,
            sequences: $sequences,
            context: $context,
            sequence_batch: $sequence_batch,
            time_tile: $time_tile,
            max_rows: $max_rows,
            pause_after_layers: 2,
            kldref: false
        },
        checks: {
            engine_identity_present: (all(.[]; .engine_build != null)),
            one_engine_identity: ([.[].engine_build] | unique | length) == 1,
            one_run_fingerprint: ([.[].run_fingerprint] | unique | length) == 1,
            exact_layer0_parts: ([.[].layer0.part_hash] | unique | length) == 1,
            exact_layer1_parts: ([.[].layer1.part_hash] | unique | length) == 1,
            off_has_no_prefetch: all(.[] | select(.mode == "off");
                .layer1.timing.prefetch_bytes == 0 and
                .layer1.timing.prefetch_staged_bytes == 0 and
                .layer1.timing.prefetch_errors == 0),
            on_prefetch_admitted: all(.[] | select(.mode == "on");
                .layer1.timing.prefetch_bytes > 0 and
                .layer1.timing.prefetch_staged_bytes > 0 and
                .layer1.timing.prefetch_errors == 0)
        },
        timing: {
            off: $off,
            on: $on,
            on_minus_off_total_before_checkpoint_us:
                ($on.median_total_before_checkpoint_us - $off.median_total_before_checkpoint_us),
            on_minus_off_foreground_load_us:
                ($on.median_foreground_load_us - $off.median_foreground_load_us)
        },
        trials: $trials
    } |
    .status = if all(.checks[]; . == true) then "valid" else "inconclusive" end
    ' "$OUTPUT_DIR"/trial-*/trial.json >"$OUTPUT_DIR/evidence.json"

jq '{status, checks, timing}' "$OUTPUT_DIR/evidence.json"
if [[ $(jq -r '.status' "$OUTPUT_DIR/evidence.json") != valid ]]; then
    printf 'error: prefetch comparison is inconclusive; inspect %s\n' "$OUTPUT_DIR/evidence.json" >&2
    exit 3
fi
