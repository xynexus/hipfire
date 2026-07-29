#!/usr/bin/env bash
set -euo pipefail

# Keep the public PATH surface smaller than the private runner directory.
# hipfire dispatches daemon/eval/host-profile itself, and doctor resolves its
# privileged helper beside the installed CLI. Those binaries still belong in
# ~/.hipfire/bin, but they are not standalone public commands.
HIPFIRE_DIR="${HIPFIRE_DIR:-$HOME/.hipfire}"
BIN_DIR="$HIPFIRE_DIR/bin"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"

PUBLIC_BINS=(
    hipfire
    hipfire-quantize
    hipfire-coexistence
    hipfire-monitor
)

# Names previously published into ~/.local/bin. Remove only links owned by this
# Hipfire install; preserve regular files and links to any other destination.
NON_PUBLIC_BINS=(
    hipfire-daemon
    hipfire-eval
    hipfire-host-profile
    hipfire-priv-helper
    hipfire-system-monitor
    hipfire-tui
)

mkdir -p "$LOCAL_BIN"
echo "Synchronizing public commands in $LOCAL_BIN..."

remove_owned_link() {
    local name="$1"
    local link="$LOCAL_BIN/$name"
    local owned_target="$BIN_DIR/$name"

    if [ -L "$link" ] && [ "$(readlink "$link")" = "$owned_target" ]; then
        rm -f "$link"
        echo "  removed retired $link"
    fi
}

for bin in "${NON_PUBLIC_BINS[@]}"; do
    remove_owned_link "$bin"
done

for bin in "${PUBLIC_BINS[@]}"; do
    source="$BIN_DIR/$bin"
    link="$LOCAL_BIN/$bin"

    [ -f "$source" ] || continue

    if [ -e "$link" ] && [ ! -L "$link" ]; then
        echo "  WARNING: preserving non-symlink $link" >&2
        continue
    fi
    if [ -L "$link" ]; then
        target="$(readlink "$link")"
        if [ "$target" != "$source" ]; then
            echo "  WARNING: preserving user-managed $link -> $target" >&2
            continue
        fi
    fi

    ln -sfn "$source" "$link"
    echo "  $link -> $source ✓"
done
