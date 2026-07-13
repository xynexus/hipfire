#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

R72_Q_MODE=adjacent "$SCRIPT_DIR/../r72/build_r72.sh" "$@"
