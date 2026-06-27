#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# dev-link.sh — symlink the in-tree release binaries into the install dirs so
# `cargo build --release` updates the installed commands in place. This is the
# developer alternative to `install.sh` (which `cargo install`s copies, and so
# goes stale until you reinstall).
#
# Covers BOTH resolution paths that bite us:
#   - PATH commands: ~/.local/bin (first on PATH) and ~/.hipfire/bin
#   - daemon discovery: ~/.hipfire/bin/hipfire-daemon (find_daemon_bin candidate
#     #2, preferred over target/release) — a stale copy here silently wins.
#
# Usage:
#   scripts/dev-link.sh            # symlink existing target/release binaries
#   scripts/dev-link.sh --build    # cargo build --release first, then symlink
#   scripts/dev-link.sh --dry-run  # print what it would do, change nothing
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO/target/release"
HIPFIRE_BIN="${HIPFIRE_DIR:-$HOME/.hipfire}/bin"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"

# Workspace binaries that ship as installed commands.
BINS=(hipfire hipfire-daemon hipfire-quantize hipfire-eval hipfire-host-profile hipfire-tui hipfire-atlas)

BUILD=0
DRY=0
for a in "$@"; do
    case "$a" in
        --build) BUILD=1 ;;
        --dry-run) DRY=1 ;;
        *) echo "unknown arg: $a" >&2; exit 2 ;;
    esac
done

if [ "$BUILD" = 1 ]; then
    echo "==> cargo build --release"
    (cd "$REPO" && cargo build --release)
fi

link() {  # link <src> <dest> — atomic replace so a concurrent reader (e.g. a
          # running daemon-discovery) never sees a missing path.
    local src="$1" dest="$2"
    if [ "$DRY" = 1 ]; then
        printf "  %s -> %s\n" "$dest" "$src"
        return
    fi
    local tmp="${dest}.dev-link.$$"
    ln -sfn "$src" "$tmp"
    mv -Tf "$tmp" "$dest"   # rename over the existing entry (atomic)
    printf "  %s -> %s\n" "$dest" "$src"
}

[ "$DRY" = 1 ] || mkdir -p "$HIPFIRE_BIN" "$LOCAL_BIN"

echo "==> symlinking in-tree binaries from $TARGET"
linked=0
for b in "${BINS[@]}"; do
    src="$TARGET/$b"
    if [ ! -x "$src" ]; then
        echo "  skip $b (not built — run with --build or 'cargo build --release')" >&2
        continue
    fi
    link "$src" "$HIPFIRE_BIN/$b"
    link "$src" "$LOCAL_BIN/$b"
    linked=$((linked + 1))
done

# `daemon` alias some tooling expects alongside hipfire-daemon.
if [ -x "$TARGET/hipfire-daemon" ]; then
    link "$TARGET/hipfire-daemon" "$HIPFIRE_BIN/daemon"
fi

echo "==> linked $linked binaries into $HIPFIRE_BIN and $LOCAL_BIN"
echo "    'cargo build --release' (or 'make build') now updates the installed commands in place."
