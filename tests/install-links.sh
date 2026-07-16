#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

HIPFIRE_DIR="$TMP/hipfire"
LOCAL_BIN="$TMP/local-bin"
TARGET_DIR="$TMP/target"
mkdir -p "$HIPFIRE_DIR/bin" "$LOCAL_BIN" "$TARGET_DIR"

for bin in hipfire hipfire-daemon hipfire-quantize hipfire-eval hipfire-monitor \
    hipfire-priv-helper hipfire-host-profile; do
    : >"$TARGET_DIR/$bin"
done

# Simulate links from an older install, including retired names whose targets no
# longer exist. A link to another installation and an unrelated file must stay.
for bin in hipfire-daemon hipfire-host-profile hipfire-priv-helper \
    hipfire-system-monitor hipfire-tui; do
    ln -s "$HIPFIRE_DIR/bin/$bin" "$LOCAL_BIN/$bin"
done
ln -s /opt/other-hipfire/hipfire-eval "$LOCAL_BIN/hipfire-eval"
: >"$LOCAL_BIN/unrelated"

make -s -C "$ROOT" link \
    HIPFIRE_DIR="$HIPFIRE_DIR" \
    LOCAL_BIN="$LOCAL_BIN" \
    TARGET_DIR="$TARGET_DIR"

for bin in hipfire hipfire-quantize hipfire-monitor; do
    [ "$(readlink "$LOCAL_BIN/$bin")" = "$HIPFIRE_DIR/bin/$bin" ] || {
        echo "install-links: missing public link for $bin" >&2
        exit 1
    }
done

for bin in hipfire-daemon hipfire-host-profile hipfire-priv-helper \
    hipfire-system-monitor hipfire-tui; do
    [ ! -L "$LOCAL_BIN/$bin" ] || {
        echo "install-links: retained owned non-public link for $bin" >&2
        exit 1
    }
done

[ "$(readlink "$LOCAL_BIN/hipfire-eval")" = /opt/other-hipfire/hipfire-eval ]
[ -f "$LOCAL_BIN/unrelated" ]

# Repeating the sync is idempotent.
HIPFIRE_DIR="$HIPFIRE_DIR" LOCAL_BIN="$LOCAL_BIN" \
    bash "$ROOT/scripts/sync-install-links.sh" >/dev/null

grep -q -- '--path crates/hipfire-quantize' "$ROOT/install.sh"
grep -q -- '--bin hipfire-quantize' "$ROOT/install.sh"
grep -q -- 'scripts/sync-install-links.sh' "$ROOT/install.sh"

echo "install-links: PASS"
