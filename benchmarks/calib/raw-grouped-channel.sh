#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: benchmarks/calib/raw-grouped-channel.sh --output-dir DIR [options]

Build and run the cross-architecture raw F16/BF16 grouped-MoE channel test,
then write a fingerprinted evidence.json row.

Options:
  --binary PATH       Prebuilt channel-test binary
  --lock-bin PATH     hipfire binary providing `lock run`
  --no-build          Do not build the channel-test example
  --output-dir DIR    New or empty evidence directory (required)
  -h, --help          Show this help
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT_DIR=
CHANNEL_BIN="$ROOT/target/release/examples/test_moe_grouped_wmma_f16_bf16"
LOCK_BIN=${HIPFIRE_BIN:-$ROOT/target/release/hipfire}
BUILD=true

while (($#)); do
    case "$1" in
        --output-dir) OUTPUT_DIR=${2:?missing value for --output-dir}; shift 2 ;;
        --binary) CHANNEL_BIN=${2:?missing value for --binary}; shift 2 ;;
        --lock-bin) LOCK_BIN=${2:?missing value for --lock-bin}; shift 2 ;;
        --no-build) BUILD=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n $OUTPUT_DIR ]] || { printf 'error: --output-dir is required\n' >&2; usage >&2; exit 2; }
command -v jq >/dev/null || { printf 'error: jq is required\n' >&2; exit 2; }
command -v sha256sum >/dev/null || { printf 'error: sha256sum is required\n' >&2; exit 2; }
command -v rg >/dev/null || { printf 'error: rg is required\n' >&2; exit 2; }
if [[ -d $OUTPUT_DIR ]] && [[ -n $(find "$OUTPUT_DIR" -mindepth 1 -print -quit) ]]; then
    printf 'error: output directory is not empty: %s\n' "$OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

if [[ $BUILD == true ]]; then
    CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2} cargo build --release \
        --manifest-path "$ROOT/Cargo.toml" \
        -p hipfire-rdna \
        --example test_moe_grouped_wmma_f16_bf16 2>&1 | tee "$OUTPUT_DIR/build.log"
fi
[[ -x $CHANNEL_BIN ]] || { printf 'error: channel binary not found: %s\n' "$CHANNEL_BIN" >&2; exit 2; }
if [[ ! -x $LOCK_BIN ]]; then
    if command -v hipfire >/dev/null 2>&1; then
        LOCK_BIN=$(command -v hipfire)
    else
        printf 'error: hipfire lock binary not found: %s\n' "$LOCK_BIN" >&2
        exit 2
    fi
fi

CHANNEL_BIN_SHA256=$(sha256sum "$CHANNEL_BIN" | awk '{print $1}')
LOCK_BIN_SHA256=$(sha256sum "$LOCK_BIN" | awk '{print $1}')
started_at=$(date -Is)
set +e
"$LOCK_BIN" lock run calibration-raw-grouped-channel -- "$CHANNEL_BIN" \
    2>&1 | tee "$OUTPUT_DIR/channel.log"
channel_status=${PIPESTATUS[0]}
set -e
finished_at=$(date -Is)

LOG_SHA256=$(sha256sum "$OUTPUT_DIR/channel.log" | awk '{print $1}')
mapfile -t arches < <(sed -n 's/^  arch=//p' "$OUTPUT_DIR/channel.log" | sort -u)
arch=${arches[0]:-}
portable_f16=$(rg -c '^  portable F16 max_abs=' "$OUTPUT_DIR/channel.log" || true)
portable_bf16=$(rg -c '^  portable BF16 max_abs=' "$OUTPUT_DIR/channel.log" || true)
wmma_cases=$(rg -c '^  wmma max_abs=' "$OUTPUT_DIR/channel.log" || true)
wmma_skips=$(rg -c '^  WMMA SKIP - ' "$OUTPUT_DIR/channel.log" || true)
indexed_cases=$(rg -c '^=== (F16|BF16) indexed-gate-up' "$OUTPUT_DIR/channel.log" || true)
case_passes=$(rg -c '^  PASS$' "$OUTPUT_DIR/channel.log" || true)
portable_f16=${portable_f16:-0}
portable_bf16=${portable_bf16:-0}
wmma_cases=${wmma_cases:-0}
wmma_skips=${wmma_skips:-0}
indexed_cases=${indexed_cases:-0}
case_passes=${case_passes:-0}
final_pass=false
if rg -q '^All executed portable F16/BF16 grouped MoE cases PASS\.$' \
    "$OUTPUT_DIR/channel.log"; then
    final_pass=true
fi
checks_valid=false
if ((channel_status == 0)) && [[ -n $arch ]] && ((${#arches[@]} == 1)) \
    && ((portable_f16 == 4 && portable_bf16 == 4)) \
    && [[ $final_pass == true ]]; then
    checks_valid=true
fi

git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)
if [[ -n $(git -C "$ROOT" status --porcelain 2>/dev/null) ]]; then
    git_dirty=true
else
    git_dirty=false
fi
jq -n \
    --arg arch "$arch" \
    --arg binary "$CHANNEL_BIN" \
    --arg binary_sha256 "$CHANNEL_BIN_SHA256" \
    --arg lock_binary "$LOCK_BIN" \
    --arg lock_binary_sha256 "$LOCK_BIN_SHA256" \
    --arg log "$OUTPUT_DIR/channel.log" \
    --arg log_sha256 "$LOG_SHA256" \
    --arg started_at "$started_at" \
    --arg finished_at "$finished_at" \
    --arg git_commit "$git_commit" \
    --argjson git_dirty "$git_dirty" \
    --argjson portable_f16 "$portable_f16" \
    --argjson portable_bf16 "$portable_bf16" \
    --argjson wmma_cases "$wmma_cases" \
    --argjson wmma_skips "$wmma_skips" \
    --argjson indexed_cases "$indexed_cases" \
    --argjson case_passes "$case_passes" \
    --argjson channel_exit_code "$channel_status" \
    --argjson checks_valid "$checks_valid" \
    '{
        schema: "hipfire.raw_grouped_channel.v1",
        status: (if $checks_valid then "pass"
                 elif $channel_exit_code != 0 then "fail"
                 else "inconclusive" end),
        scope: "portable F16/BF16 grouped execution plus architecture-specific fast paths",
        arch: $arch,
        started_at: $started_at,
        finished_at: $finished_at,
        engine: {
            binary: $binary,
            binary_sha256: $binary_sha256,
            git_commit: $git_commit,
            git_dirty: $git_dirty
        },
        lock: {binary: $lock_binary, binary_sha256: $lock_binary_sha256},
        log: {path: $log, sha256: $log_sha256},
        checks: {
            channel_exit_code: $channel_exit_code,
            portable_f16_cases: $portable_f16,
            portable_bf16_cases: $portable_bf16,
            wmma_cases: $wmma_cases,
            wmma_skips: $wmma_skips,
            indexed_cases: $indexed_cases,
            passing_cases: $case_passes,
            exact_portable_case_matrix: $checks_valid
        }
    }' >"$OUTPUT_DIR/evidence.json.tmp"
mv "$OUTPUT_DIR/evidence.json.tmp" "$OUTPUT_DIR/evidence.json"
jq '{status, arch, checks}' "$OUTPUT_DIR/evidence.json"

if [[ $checks_valid != true ]]; then
    printf 'error: raw grouped channel evidence is inconclusive: %s\n' \
        "$OUTPUT_DIR/evidence.json" >&2
    if ((channel_status != 0)); then
        exit "$channel_status"
    fi
    exit 3
fi
