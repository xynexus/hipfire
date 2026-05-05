#!/bin/bash
# hipfire installer — detects GPU, installs deps, downloads binary + kernels.
# Usage: curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
set -euo pipefail

XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
XDG_STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
HIPFIRE_CONFIG_DIR="$XDG_CONFIG_HOME/hipfire"
HIPFIRE_DATA_DIR="$XDG_DATA_HOME/hipfire"
HIPFIRE_CACHE_DIR="$XDG_CACHE_HOME/hipfire"
HIPFIRE_STATE_DIR="$XDG_STATE_HOME/hipfire"
BIN_DIR="$HOME/.local/lib/hipfire"
CLI_DIR="$BIN_DIR/cli"
MODELS_DIR="$HIPFIRE_DATA_DIR/models"
SRC_DIR="$HIPFIRE_DATA_DIR/src"
LOCAL_BIN_DIR="$HOME/.local/bin"
GITHUB_REPO="Kaden-Schutt/hipfire"
GITHUB_BRANCH="master"

echo "=== hipfire installer ==="
echo ""

# ─── Interactive prompts (safe for curl|bash) ────────────
ask() {
    # Usage: result=$(ask "prompt [Y/n] " "Y")
    # Safe for curl|bash: reads from /dev/tty, falls back to default if non-interactive
    local prompt="$1" default="$2"
    if can_prompt; then
        printf "%s" "$prompt" >/dev/tty
        local reply
        read -r reply </dev/tty 2>/dev/null || reply="$default"
        echo "${reply:-$default}"
    else
        echo "$default"
    fi
}

can_prompt() {
    tty -s
}

run_pkg_cmd() {
    # Avoid noisy sudo auth failures in non-interactive shells. Root-run
    # installers can still execute sudo-free package managers, while
    # interactive users get the normal sudo prompt.
    local cmd="$1"
    if [[ "$cmd" == sudo\ * ]] && [ "$(id -u)" -ne 0 ] && ! can_prompt; then
        echo "  Skipping non-interactive package install that requires sudo:"
        echo "    $cmd"
        echo "  Re-run interactively or run it manually, then re-run this installer."
        return 1
    fi
    eval "$cmd"
}

install_file() {
    # Atomic-ish file install that is safe to rerun over existing files.
    # Usage: install_file SRC DEST [MODE]
    local src="$1" dest="$2" mode="${3:-}"
    local dest_dir tmp
    if [ ! -f "$src" ]; then
        echo "  ERROR: missing source file: $src" >&2
        return 1
    fi
    dest_dir="$(dirname "$dest")"
    mkdir -p "$dest_dir"
    if [ -d "$dest" ]; then
        echo "  ERROR: destination is a directory, expected file: $dest" >&2
        return 1
    fi

    # If a local install source already points at the destination, do nothing.
    if [ -e "$dest" ] && [ "$(readlink -f "$src" 2>/dev/null || echo "$src")" = "$(readlink -f "$dest" 2>/dev/null || echo "$dest")" ]; then
        [ -n "$mode" ] && chmod "$mode" "$dest"
        return 0
    fi

    tmp="$(mktemp "$dest_dir/.tmp.$(basename "$dest").XXXXXX")"
    cp "$src" "$tmp"
    if [ -n "$mode" ]; then
        chmod "$mode" "$tmp"
    elif [ -x "$src" ]; then
        chmod 755 "$tmp"
    else
        chmod 644 "$tmp"
    fi
    mv -f "$tmp" "$dest"
}

install_text_file() {
    # Read stdin, write destination atomically, set mode.
    local dest="$1" mode="${2:-644}"
    local dest_dir tmp
    dest_dir="$(dirname "$dest")"
    mkdir -p "$dest_dir"
    tmp="$(mktemp "$dest_dir/.tmp.$(basename "$dest").XXXXXX")"
    cat > "$tmp"
    chmod "$mode" "$tmp"
    mv -f "$tmp" "$dest"
}

# Pick the right HIP runtime package name for dnf-based distros. Fedora's
# rocm-hip package is what ships libamdhip64.so.6; the rocm-hip-runtime
# meta-package only exists on RHEL / Rocky / Alma via AMD's repo. Detect
# via /etc/os-release ID + ID_LIKE. Reported in #64 (kamikazechaser).
dnf_hip_pkg() {
    local id="" id_like=""
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        id="${ID:-}"
        id_like="${ID_LIKE:-}"
    fi
    case "$id" in
        fedora) echo "rocm-hip" ;;
        rhel|rocky|almalinux|centos|ol) echo "rocm-hip-runtime" ;;
        *)
            case "$id_like" in
                *fedora*) echo "rocm-hip" ;;
                *rhel*|*centos*) echo "rocm-hip-runtime" ;;
                *) echo "rocm-hip-runtime" ;;
            esac
            ;;
    esac
}

# ─── OS Detection ────────────────────────────────────────
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$OS" in
    linux) ;;
    darwin)
        echo "macOS is not supported (AMD GPUs only). Exiting."
        exit 1
        ;;
    mingw*|msys*|cygwin*)
        echo "Windows detected (via $OS)."
        echo ""
        echo "hipfire has native Windows support. Install options:"
        echo "  1. PowerShell (recommended):"
        echo "     irm https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/scripts/install.ps1 | iex"
        echo "  2. WSL2 (alternative):"
        echo "     wsl --install"
        echo "     # Then inside WSL:"
        echo "     curl -L https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/scripts/install.sh | bash"
        exit 1
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac
echo "OS: $OS ($ARCH)"

# ─── GPU Detection ───────────────────────────────────────
echo ""
echo "Checking for AMD GPU..."
if [ ! -e /dev/kfd ]; then
    echo "ERROR: /dev/kfd not found. No AMD GPU detected."
    echo ""
    echo "Possible fixes:"
    echo "  - Install amdgpu driver: sudo apt install linux-firmware (Ubuntu)"
    echo "  - Reboot after driver install"
    echo "  - Check: lspci | grep -i amd"
    echo ""
    echo "Run 'hipfire diag' after install for automated troubleshooting."
    exit 1
fi
echo "  /dev/kfd: found ✓"

# Detect GPU arch via kfd topology (most reliable on modern kernels)
GPU_ARCH="unknown"
for node_props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ -f "$node_props" ] || continue
    ver=$(grep -oP 'gfx_target_version\s+\K\d+' "$node_props" 2>/dev/null || true)
    case "$ver" in
        90006)          GPU_ARCH="gfx906";  break ;;
        90008)          GPU_ARCH="gfx908";  break ;;
        100100)         GPU_ARCH="gfx1010"; break ;;
        100300|100302)  GPU_ARCH="gfx1030"; break ;;
        110000|110001)  GPU_ARCH="gfx1100"; break ;;
        110501)         GPU_ARCH="gfx1151"; break ;;
        120000)         GPU_ARCH="gfx1200"; break ;;
        120001)         GPU_ARCH="gfx1201"; break ;;
    esac
done

# Fallback: rocm-smi
if [ "$GPU_ARCH" = "unknown" ] && command -v rocm-smi &>/dev/null; then
    GPU_ARCH=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
fi

# Fallback: ask user
if [ "$GPU_ARCH" = "unknown" ]; then
    echo "  WARNING: Could not detect GPU architecture."
    echo "  Supported: gfx906 (Vega 20), gfx908 (MI100), gfx1010 (5700 XT), gfx1030 (6800 XT), gfx1100 (7900 XTX), gfx1151 (Strix Halo), gfx1200 (R9700), gfx1201 (9070 XT)"
    GPU_ARCH=$(ask "  Enter your GPU arch [or Enter to skip]: " "unknown")
fi
echo "  GPU arch: $GPU_ARCH"

# ─── HIP Runtime ─────────────────────────────────────────
echo ""
echo "Checking HIP runtime..."
HIP_FOUND=false
HIP_LIB=""
# Probe each known install directory for either the unversioned .so symlink
# or a versioned ABI variant (.so.6 / .so.7 / .so.8). Fedora's `rocm-hip`
# package installs only `libamdhip64.so.6` — the unversioned symlink ships
# in `rocm-hip-devel` which most users don't have. So checking only `.so`
# misses Fedora installs entirely.
for dir in /opt/rocm/lib /opt/rocm/lib64 \
           /usr/lib /usr/lib64 \
           /usr/lib/x86_64-linux-gnu /usr/lib64/rocm; do
    for suffix in "" ".6" ".7" ".8"; do
        lib="$dir/libamdhip64.so${suffix}"
        if [ -f "$lib" ]; then
            echo "  libamdhip64.so: found at $lib ✓"
            HIP_FOUND=true
            HIP_LIB="$lib"
            break 2
        fi
    done
done

# Fallback: ask ldconfig if none of the hardcoded paths matched. Match both
# unversioned (`libamdhip64.so`) and versioned (`libamdhip64.so.N`) entries
# — the trailing-space pattern from the previous version only matched the
# unversioned line, missing Fedora's `.so.6` SONAME.
if ! $HIP_FOUND; then
    ldconfig_hit=$(ldconfig -p 2>/dev/null | grep -m1 -E '\blibamdhip64\.so(\.[0-9]+)?\b' | awk '{print $NF}' || true)
    if [ -n "$ldconfig_hit" ] && [ -f "$ldconfig_hit" ]; then
        echo "  libamdhip64.so: found via ldconfig at $ldconfig_hit ✓"
        HIP_FOUND=true
        HIP_LIB="$ldconfig_hit"
    fi
fi

# Check HIP version matches GPU arch requirements
if $HIP_FOUND; then
    HIP_VER=""
    if command -v /opt/rocm/bin/hipconfig &>/dev/null; then
        HIP_VER=$(/opt/rocm/bin/hipconfig --version 2>/dev/null | grep -oP '^\d+\.\d+' || true)
    elif command -v hipconfig &>/dev/null; then
        HIP_VER=$(hipconfig --version 2>/dev/null | grep -oP '^\d+\.\d+' || true)
    fi

    if [ -n "$HIP_VER" ]; then
        HIP_MAJOR=$(echo "$HIP_VER" | cut -d. -f1)
        HIP_MINOR=$(echo "$HIP_VER" | cut -d. -f2)
        echo "  HIP version: $HIP_VER"

        # Minimum HIP versions per GPU arch
        MIN_MAJOR=5; MIN_MINOR=0
        case "$GPU_ARCH" in
            gfx1200|gfx1201) MIN_MAJOR=6; MIN_MINOR=4 ;;
            gfx1150|gfx1151|gfx1152) MIN_MAJOR=7; MIN_MINOR=2 ;;
            gfx1100|gfx1101) MIN_MAJOR=5; MIN_MINOR=5 ;;
        esac

        NEEDS_UPGRADE=false
        if [ "$HIP_MAJOR" -lt "$MIN_MAJOR" ] 2>/dev/null; then
            NEEDS_UPGRADE=true
        elif [ "$HIP_MAJOR" -eq "$MIN_MAJOR" ] && [ "$HIP_MINOR" -lt "$MIN_MINOR" ] 2>/dev/null; then
            NEEDS_UPGRADE=true
        fi

        if $NEEDS_UPGRADE; then
            echo ""
            echo "  WARNING: HIP $HIP_VER is too old for $GPU_ARCH (needs $MIN_MAJOR.$MIN_MINOR+)"
            echo "  Kernels may fail to load. Upgrading HIP runtime is recommended."
            PKG_CMD=""
            if command -v apt &>/dev/null; then
                PKG_CMD="sudo apt install -y rocm-hip-runtime"
            elif command -v dnf &>/dev/null; then
                PKG_CMD="sudo dnf install -y $(dnf_hip_pkg)"
            elif command -v pacman &>/dev/null; then
                PKG_CMD="sudo pacman -S --noconfirm rocm-hip-runtime"
            fi
            if [ -n "$PKG_CMD" ]; then
                reply=$(ask "  Upgrade now? ($PKG_CMD) [Y/n] " "Y")
                if [ "$reply" != "n" ] && [ "$reply" != "N" ]; then
                    echo "  Running: $PKG_CMD"
                    run_pkg_cmd "$PKG_CMD" || echo "  Upgrade failed or skipped. You may need to add the ROCm repo first."
                fi
            else
                echo "  Upgrade manually: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
            fi
        fi
    fi
fi

if ! $HIP_FOUND; then
    echo "  libamdhip64.so: NOT FOUND"
    echo ""
    echo "  hipfire needs the HIP runtime library (libamdhip64.so)."
    echo "  This is a small package (~50MB), NOT the full ROCm SDK."

    # Detect package manager and offer guided install
    PKG_CMD=""
    if command -v apt &>/dev/null; then
        PKG_CMD="sudo apt install -y rocm-hip-runtime"
    elif command -v dnf &>/dev/null; then
        PKG_CMD="sudo dnf install -y $(dnf_hip_pkg)"
    elif command -v pacman &>/dev/null; then
        PKG_CMD="sudo pacman -S --noconfirm rocm-hip-runtime"
    elif command -v zypper &>/dev/null; then
        PKG_CMD="sudo zypper install -y rocm-hip-runtime"
    fi

    if [ -n "$PKG_CMD" ]; then
        reply=$(ask "  Install now? ($PKG_CMD) [Y/n] " "Y")
        if [ "$reply" != "n" ] && [ "$reply" != "N" ]; then
            echo "  Running: $PKG_CMD"
            run_pkg_cmd "$PKG_CMD" || {
                echo ""
                echo "  HIP runtime install failed. Try manually:"
                echo "    $PKG_CMD"
                echo "  Or see: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
                echo ""
                echo "  hipfire can still be installed, but won't run without libamdhip64.so."
                reply=$(ask "  Continue anyway? [y/N] " "N")
                if [ "$reply" != "y" ] && [ "$reply" != "Y" ]; then
                    exit 1
                fi
            }
        else
            echo "  Skipping. Install later: $PKG_CMD"
        fi
    else
        echo "  Unknown package manager. Install libamdhip64.so manually:"
        echo "  https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        reply=$(ask "  Continue without HIP runtime? [y/N] " "N")
        if [ "$reply" != "y" ] && [ "$reply" != "Y" ]; then
            exit 1
        fi
    fi
fi

# ─── Install Bun (needed for CLI) ───────────────────────
echo ""
if command -v bun &>/dev/null; then
    echo "Bun: found ✓"
else
    echo "Installing Bun (runtime for hipfire CLI)..."
    if ! command -v unzip &>/dev/null; then
        echo "  ERROR: 'unzip' is required by the Bun installer but is not present."
        echo "  Install it first:  apt install unzip   # or pacman -S unzip / dnf install unzip"
        echo "  hipfire CLI requires Bun to run."
        exit 1
    fi
    curl -fsSL https://bun.sh/install | bash || {
        echo "  Bun install failed. Visit https://bun.sh"
        echo "  hipfire CLI requires Bun to run."
        exit 1
    }
    # Source bun into current session
    export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
    export PATH="$BUN_INSTALL/bin:$PATH"
    if command -v bun &>/dev/null; then
        echo "  Bun installed ✓"
    else
        echo "  Bun installed but not in PATH. Restart your shell or run:"
        echo "    export PATH=\"\$HOME/.bun/bin:\$PATH\""
    fi
fi

# ─── Create directories ─────────────────────────────────
mkdir -p "$BIN_DIR" "$CLI_DIR" "$MODELS_DIR" "$HIPFIRE_CONFIG_DIR" "$HIPFIRE_CACHE_DIR" "$HIPFIRE_STATE_DIR" "$LOCAL_BIN_DIR"

# ─── Determine install mode ──────────────────────────────
# Local: running from within a repo checkout (./scripts/install.sh)
# Remote: running via curl|bash — clone the repo
INSTALL_MODE="remote"
REPO_DIR=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd 2>/dev/null)" || true
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../Cargo.toml" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    INSTALL_MODE="local"
fi

echo ""
if [ "$INSTALL_MODE" = "local" ]; then
    echo "Install mode: local (repo at $REPO_DIR)"
else
    echo "Install mode: remote (cloning repository)"

    if [ ! -d "$SRC_DIR/.git" ]; then
        if ! command -v git &>/dev/null; then
            echo "  ERROR: git is required for remote install."
            echo "  Install git and re-run, or clone manually:"
            echo "    git clone https://github.com/$GITHUB_REPO.git $SRC_DIR"
            exit 1
        fi
        echo "  Cloning https://github.com/$GITHUB_REPO.git ..."
        git clone --depth 1 --branch "$GITHUB_BRANCH" \
            "https://github.com/$GITHUB_REPO.git" "$SRC_DIR" || {
            echo "  Clone failed. Check your connection or try:"
            echo "    git clone https://github.com/$GITHUB_REPO.git $SRC_DIR"
            exit 1
        }
        echo "  Cloned ✓"
    else
        echo "  Existing clone found at $SRC_DIR"
        # Stash any local modifications (Cargo.lock rewritten by cargo build,
        # autocrlf line-ending drift, user edits, etc.) so that the subsequent
        # reset can't abort with "local changes would be overwritten by merge".
        # The stash is named so the user can recover via `git stash pop`.
        if [ -n "$(git -C "$SRC_DIR" status --porcelain 2>/dev/null)" ]; then
            stamp=$(date -u +%Y-%m-%dT%H-%M-%SZ)
            stash_msg="hipfire-install-${stamp}"
            echo "  Local modifications detected — stashing as '$stash_msg'"
            if git -C "$SRC_DIR" stash push --include-untracked -m "$stash_msg" >/dev/null 2>&1; then
                echo "  Recover later with: git -C $SRC_DIR stash pop"
            else
                echo "  WARNING: git stash failed; reset may drop local changes."
            fi
        fi
        echo "  Updating..."
        # Fetch + hard-reset is safe now (tree is clean post-stash). Reset
        # handles both fast-forward and diverged-history cases uniformly.
        git -C "$SRC_DIR" fetch origin "$GITHUB_BRANCH" --depth 1 2>/dev/null && \
        git -C "$SRC_DIR" reset --hard "origin/$GITHUB_BRANCH" 2>/dev/null || {
            echo "  Update failed (non-fatal). Using existing checkout."
        }
    fi
    REPO_DIR="$SRC_DIR"
fi

# ─── Build / Install binaries ────────────────────────────
echo ""
echo "Installing hipfire..."

if [ -f "$REPO_DIR/target/release/examples/daemon" ] && [ -f "$REPO_DIR/target/release/hipfire-quantize" ]; then
    echo "  Pre-built binaries found ✓"
else
    echo "  No pre-built binaries. Building from source..."
    if ! command -v cargo &>/dev/null; then
        echo "  Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>/dev/null
        . "$HOME/.cargo/env"
    fi
    (cd "$REPO_DIR" && \
        echo "  cargo build --release (this may take several minutes)..." && \
        cargo build --release --features deltanet --example daemon --example infer --example infer_hfq -p engine 2>&1 | tail -5 && \
        cargo build --release -p hipfire-quantize 2>&1 | tail -5)
    if [ ! -f "$REPO_DIR/target/release/examples/daemon" ]; then
        echo ""
        echo "  BUILD FAILED."
        echo "  Common causes:"
        echo "    - Missing ROCm SDK (needed to compile, not just run)"
        echo "    - Missing system libs (check error above)"
        echo ""
        echo "  After fixing, re-run this installer or build manually:"
        echo "    cd $REPO_DIR && cargo build --release --features deltanet --example daemon -p engine"
        exit 1
    fi
    echo "  Build complete ✓"
fi

# Copy binaries. Use temp-file + rename so rerunning the installer over
# existing binaries does not leave half-written files if a copy is interrupted.
install_file "$REPO_DIR/target/release/examples/daemon" "$BIN_DIR/daemon" 755
if [ -f "$REPO_DIR/target/release/examples/infer" ]; then
    install_file "$REPO_DIR/target/release/examples/infer" "$BIN_DIR/infer" 755
fi
if [ -f "$REPO_DIR/target/release/examples/infer_hfq" ]; then
    install_file "$REPO_DIR/target/release/examples/infer_hfq" "$BIN_DIR/infer_hfq" 755
fi
if [ -f "$REPO_DIR/target/release/hipfire-quantize" ]; then
    install_file "$REPO_DIR/target/release/hipfire-quantize" "$BIN_DIR/hipfire-quantize" 755
fi

# Copy CLI
mkdir -p "$CLI_DIR"
# Order: registry.json BEFORE index.ts. The CLI imports the JSON at startup;
# if we wrote the new index.ts before the JSON and the JSON copy then failed,
# the install would be stranded — new TS that can't resolve its own data file.
# JSON-first means a partial-failure window leaves a recoverable state.
if [ ! -f "$REPO_DIR/cli/registry.json" ] || [ ! -f "$REPO_DIR/cli/index.ts" ]; then
    echo "ERROR: cli/registry.json or cli/index.ts missing in $REPO_DIR" >&2
    echo "       Repo checkout may be incomplete; aborting install." >&2
    exit 1
fi
install_file "$REPO_DIR/cli/registry.json" "$CLI_DIR/registry.json" 644
install_file "$REPO_DIR/cli/package.json"  "$CLI_DIR/package.json" 644
install_file "$REPO_DIR/cli/index.ts"      "$CLI_DIR/index.ts" 644

# Create hipfire wrapper. The shim resolves `bun` even when it isn't on
# $PATH — rustup and bun both install to under-home bindirs that shell
# profiles load, but non-interactive SSH / cron / systemd sessions often
# get a minimal PATH. Without this probe the first line that calls the
# shim dies with "exec: bun: not found" before dep-autodetect inside the
# TS CLI has a chance to run.
install_text_file "$LOCAL_BIN_DIR/hipfire" 755 << 'WRAPPER'
#!/bin/bash
set -e
if command -v bun >/dev/null 2>&1; then
    BUN=bun
elif [ -x "$HOME/.bun/bin/bun" ]; then
    BUN="$HOME/.bun/bin/bun"
elif [ -x "/usr/local/bin/bun" ]; then
    BUN="/usr/local/bin/bun"
else
    echo "hipfire: 'bun' not found in PATH, ~/.bun/bin/, or /usr/local/bin/." >&2
    echo "         Install it: curl -fsSL https://bun.sh/install | bash" >&2
    exit 127
fi
exec "$BUN" run "$HOME/.local/lib/hipfire/cli/index.ts" "$@"
WRAPPER
echo "  Binaries + CLI installed to $BIN_DIR/ ✓"
echo "  hipfire wrapper installed to $LOCAL_BIN_DIR/hipfire ✓"

# ─── Install kernels ────────────────────────────────────
# Engine probes for kernels at {exe_dir}/kernels/compiled/{arch}/
# so we place them at ~/.local/lib/hipfire/kernels/compiled/{arch}/
echo ""
if [ "$GPU_ARCH" != "unknown" ]; then
    echo "Setting up kernels for $GPU_ARCH..."
    KERNEL_DEST="$BIN_DIR/kernels/compiled/$GPU_ARCH"
    mkdir -p "$KERNEL_DEST"

    if [ -d "$REPO_DIR/kernels/compiled/$GPU_ARCH" ]; then
        copied=0
        for kernel in "$REPO_DIR/kernels/compiled/$GPU_ARCH"/*.hsaco "$REPO_DIR/kernels/compiled/$GPU_ARCH"/*.hash; do
            [ -f "$kernel" ] || continue
            install_file "$kernel" "$KERNEL_DEST/$(basename "$kernel")"
            copied=$((copied + 1))
        done
        count=$(ls "$KERNEL_DEST"/*.hsaco 2>/dev/null | wc -l)
        echo "  Installed $copied kernel artifact(s); $count hsaco file(s) present in $KERNEL_DEST/ ✓"
    else
        echo "  No pre-compiled kernels for $GPU_ARCH in repo — will JIT from source below."
    fi
else
    echo "Skipping pre-built kernel copy (GPU arch unknown) — daemon will still"
    echo "  auto-detect at runtime. Missing kernels compile on first use."
fi

# Fill in any kernels missing from the pre-compiled set, including MQ4/asym3
# defaults that aren't always shipped for newer arches. Uses hipcc in parallel
# and writes back to ~/.local/lib/hipfire/kernels/compiled/<arch>/ so first
# `hipfire run` isn't a 2-minute compile wall. Runs regardless of install-time
# arch detection — the daemon's own Gpu::init resolves the active arch.
if [ -x "$BIN_DIR/daemon" ]; then
    echo ""
    echo "Pre-compiling GPU kernels (first run will be instant afterward)..."
    if "$BIN_DIR/daemon" --precompile; then
        echo "  Pre-compile complete ✓"
    else
        echo "  Pre-compile finished with warnings — any missing kernels will compile on first use."
    fi
fi

# ─── Config ──────────────────────────────────────────────
CONFIG="$HIPFIRE_CONFIG_DIR/config.json"
if [ ! -f "$CONFIG" ]; then
    cat > "$CONFIG" << CONF
{
  "temperature": 0.3,
  "top_p": 0.8,
  "max_tokens": 512,
  "gpu_arch": "$GPU_ARCH"
}
CONF
    echo ""
    echo "Config: $CONFIG"
fi

# ─── PATH setup ─────────────────────────────────────────
echo ""
if [[ ":$PATH:" != *":$LOCAL_BIN_DIR:"* ]]; then
    SHELL_RC=""
    case "$(basename "${SHELL:-bash}")" in
        bash) SHELL_RC="$HOME/.bashrc" ;;
        zsh)  SHELL_RC="$HOME/.zshrc" ;;
    esac

    PATH_LINE="export PATH=\"\$HOME/.local/bin:\$PATH\""
    if [ -n "$SHELL_RC" ] && [ -f "$SHELL_RC" ]; then
        if ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
            reply=$(ask "Add ~/.local/bin to PATH in $SHELL_RC? [Y/n] " "Y")
            if [ "$reply" != "n" ] && [ "$reply" != "N" ]; then
                printf '\n# hipfire\n%s\n' "$PATH_LINE" >> "$SHELL_RC"
                echo "  Added to $SHELL_RC ✓"
            else
                echo "  Add manually: $PATH_LINE"
            fi
        fi
    else
        echo "Add to your shell profile:"
        echo "  $PATH_LINE"
    fi
fi

echo ""
echo "=== hipfire installed ==="
echo ""
echo "Quick start:"
echo "  source ${SHELL_RC:-~/.bashrc}                    # reload PATH (or restart shell)"
echo "  hipfire list                                      # see local models"
echo "  hipfire run <model.hfq> \"Hello\"                  # generate text"
echo "  hipfire serve                                     # start OpenAI-compatible API"
echo ""
echo "Models go in $MODELS_DIR/ or the repo's models/ directory."
echo ""
