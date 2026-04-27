#!/bin/bash
# scripts/_detect-gpu.sh — sourced helper that exports detected GPU
# arch / marketing name / VRAM. Mirrors the detection chain in
# scripts/speed-gate.sh so every bench/test script picks up the same
# arch the gate already trusts.
#
# Usage (in another script):
#   . "$(dirname "$0")/_detect-gpu.sh"
#   echo "$HIPFIRE_DETECTED_ARCH"
#   echo "$HIPFIRE_DETECTED_NAME"
#   echo "$HIPFIRE_DETECTED_VRAM_GB"
#
# Each var can be pre-set in the environment (e.g. for CI on a
# headless runner without rocminfo) and the helper will respect it
# rather than re-detecting.
#
# Exit-status is always 0 — failure to detect leaves the vars empty
# (or as the caller-set defaults). Callers decide whether unknown
# arch is fatal or not.

# ── Arch (gfx1100, gfx1010, ...) ────────────────────────────────
if [ -z "${HIPFIRE_DETECTED_ARCH:-}" ]; then
    for probe in amdgpu-arch offload-arch \
                 /opt/rocm/bin/amdgpu-arch /opt/rocm/bin/offload-arch \
                 /opt/rocm/llvm/bin/amdgpu-arch; do
        if command -v "$probe" >/dev/null 2>&1 || [ -x "$probe" ]; then
            HIPFIRE_DETECTED_ARCH="$("$probe" 2>/dev/null | head -1)"
            [ -n "$HIPFIRE_DETECTED_ARCH" ] && break
        fi
    done
    case "${HSA_OVERRIDE_GFX_VERSION:-}" in
        10.1.0|10.1) HIPFIRE_DETECTED_ARCH="gfx1010" ;;
        10.3.0|10.3) HIPFIRE_DETECTED_ARCH="gfx1030" ;;
        11.0.0|11.0) HIPFIRE_DETECTED_ARCH="gfx1100" ;;
    esac
fi

# ── Marketing name ("Radeon RX 7900 XTX") ──────────────────────
if [ -z "${HIPFIRE_DETECTED_NAME:-}" ]; then
    if command -v rocminfo >/dev/null 2>&1; then
        HIPFIRE_DETECTED_NAME="$(rocminfo 2>/dev/null \
            | awk '/^  Name:/{n=$2} /^  Marketing Name:/{$1=""; $2=""; sub(/^[ \t]+/,""); if (n ~ /^gfx/) {print; exit}}')"
    fi
    [ -z "$HIPFIRE_DETECTED_NAME" ] && HIPFIRE_DETECTED_NAME="Unknown GPU"
fi

# ── VRAM GB ──────────────────────────────────────────────────────
if [ -z "${HIPFIRE_DETECTED_VRAM_GB:-}" ]; then
    # rocminfo's "Pool Info" sections enumerate per-agent memory; first
    # GPU agent's GLOBAL pool is the VRAM. We grep "Size:" lines under
    # the first gfx-named agent.
    if command -v rocminfo >/dev/null 2>&1; then
        HIPFIRE_DETECTED_VRAM_GB="$(rocminfo 2>/dev/null \
            | awk '
                /^  Name:/                { in_gpu = ($2 ~ /^gfx/) }
                in_gpu && /^      Size:/  { size_kb = $2; print int(size_kb/1024/1024); exit }
            ')"
    fi
    [ -z "$HIPFIRE_DETECTED_VRAM_GB" ] && HIPFIRE_DETECTED_VRAM_GB="?"
fi

export HIPFIRE_DETECTED_ARCH HIPFIRE_DETECTED_NAME HIPFIRE_DETECTED_VRAM_GB

# Convenience: human-readable banner string.
hipfire_gpu_banner() {
    if [ "$HIPFIRE_DETECTED_VRAM_GB" = "?" ]; then
        echo "$HIPFIRE_DETECTED_NAME ($HIPFIRE_DETECTED_ARCH)"
    else
        echo "$HIPFIRE_DETECTED_NAME (${HIPFIRE_DETECTED_VRAM_GB}GB VRAM, $HIPFIRE_DETECTED_ARCH)"
    fi
}
