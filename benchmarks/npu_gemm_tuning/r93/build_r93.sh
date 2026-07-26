#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${R93_CACHE_DIR:-$HOME/.hipfire/npu/embgemma_r93_bf16_to_r25_w4_activation_m256_k768}
rm -rf -- "$OUT"
mkdir -p -- "$OUT"
: "${HIPFIRE_NPU_VENV:=$HOME/.venv}"
# shellcheck source=/dev/null
source "$HIPFIRE_NPU_VENV/bin/activate"
PEANO="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
MA_ROOT=$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')
export PATH="/opt/xilinx/xrt/bin:$PEANO/bin:$MA_ROOT/bin:$PATH"

"$PEANO/bin/clang++" "$HERE/r93_ffn_activation_prep.cc" -c -o "$OUT/r93.o" \
  -I"$MA_ROOT/include" -std=c++20 -Os -DNDEBUG \
  ${R93_EXTRA_CXX_FLAGS:-} ${R93_VECTOR_PREP:+-DR93_VECTOR_PREP} \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-deprecated-declarations --target=aie2p-none-unknown-elf
python "$HERE/r93_gen.py" > "$OUT/aie.mlir"

LOG="$OUT/aiecc.log"
if ! aiecc "$OUT/aie.mlir" --no-compile-host --no-xchesscc --no-xbridge \
  --peano="$PEANO" --aie-generate-npu-insts \
  --npu-insts-name="$OUT/insts.bin" --aie-generate-xclbin \
  --xclbin-name="$OUT/final.xclbin" --tmpdir="$OUT" >"$LOG" 2>&1; then
  cat "$LOG" >&2
  exit 1
fi
test -s "$OUT/final.xclbin"
test -s "$OUT/insts.bin"

python - "$OUT" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for elf in sorted(out.glob("main_core_*.elf")):
    text = subprocess.check_output(["llvm-size", "-A", str(elf)], text=True)
    match = re.search(r"^\.text\s+(\d+)", text, re.MULTILINE)
    if match:
        size = int(match.group(1))
        rows.append((elf.name, size))
        if size > 16_384:
            raise SystemExit(f"{elf.name}: .text {size} exceeds 16384")
(out / "core-sizes.txt").write_text("".join(f"{name},{size}\n" for name, size in rows))
if len(rows) != 32:
    raise SystemExit(f"expected 32 core ELFs, found {len(rows)}")
PY

printf '%s\n' \
  'op=embeddinggemma-ffn-activation-prep' \
  'mode=w4-scaled' \
  'm=256' \
  'k=768' \
  'input=canonical-bf16-pre-ffn-norm' \
  'output=resident-r25-w4-activation' \
  'block-bytes=6656' \
  'prefix-bytes=6240' \
  'replicas=3' > "$OUT/shape.txt"
if [[ -n "${R93_VECTOR_PREP:-}" ]]; then
  printf '%s\n' 'prep=vector-r25-params' >> "$OUT/shape.txt"
fi
echo "$OUT"
