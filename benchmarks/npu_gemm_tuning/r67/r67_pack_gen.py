#!/usr/bin/env python3
"""Retarget the proven R28 graph to R67's padded joined staging buffer."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
R28 = HERE.parent / "r28" / "r28_gen.py"
ROWS = 4
PAIR, RAW_JOIN = 8192, 32768
PAIRS_PER_ROLE = int(sys.argv[1]) if len(sys.argv) > 1 else 36
if PAIRS_PER_ROLE not in (36, 37):
    raise SystemExit("records per role must be 36 or 37")
ROLES = 5
PARAMS = 2048
OLD_RAW_BYTES = 1_310_720
VALUE_STAGE_BYTES = ROLES * PAIRS_PER_ROLE * PAIR
STAGE_BYTES = VALUE_STAGE_BYTES + PARAMS
RAW_Q_BYTES = 786_432
RAW_K_BYTES = 262_144

text = subprocess.run(
    [sys.executable, str(R28)],
    check=True,
    capture_output=True,
    text=True,
).stdout

old_signature = (
    f"aie.runtime_sequence(%R: memref<{OLD_RAW_BYTES}xi8>, "
    "%P: memref<2048xi8>, %Q: memref<393216xi8>, %KV: memref<262144xi8>)"
)
new_signature = (
    f"aie.runtime_sequence(%R: memref<{STAGE_BYTES}xi8>, "
    "%Q: memref<393216xi8>, %KV: memref<262144xi8>)"
)
if old_signature not in text:
    raise SystemExit("R28 runtime signature changed")
text = text.replace(old_signature, new_signature, 1)
text = text.replace(
    "%P : memref<2048xi8>, 0, 2048",
    f"%R : memref<{STAGE_BYTES}xi8>, {VALUE_STAGE_BYTES}, 2048",
)
text = text.replace(
    f"%R : memref<{OLD_RAW_BYTES}xi8>",
    f"%R : memref<{STAGE_BYTES}xi8>",
)


offset_replacements = []


def replace_input_offset(old_offset, new_offset):
    offset_replacements.append((old_offset, new_offset))


for group in range(6):
    role, half = divmod(group, 2)
    for row in range(ROWS):
        old_offset = (group * ROWS + row) * RAW_JOIN
        new_pair = role * PAIRS_PER_ROLE + half * 16 + row * 4
        replace_input_offset(old_offset, new_pair * PAIR)

for phase, role, old_base in (
    ("k", 3, RAW_Q_BYTES),
    ("v", 4, RAW_Q_BYTES + RAW_K_BYTES),
):
    del phase
    for wave in range(2):
        for row in range(ROWS):
            old_offset = old_base + (wave * ROWS + row) * RAW_JOIN
            new_pair = role * PAIRS_PER_ROLE + wave * 16 + row * 4
            replace_input_offset(old_offset, new_pair * PAIR)

for index, (old_offset, _) in enumerate(offset_replacements):
    old = f"%R : memref<{STAGE_BYTES}xi8>, {old_offset}, {RAW_JOIN},"
    marker = f"%R : memref<{STAGE_BYTES}xi8>, R67_OFFSET_{index}, {RAW_JOIN},"
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one R28 input offset {old_offset}, found {count}")
    text = text.replace(old, marker, 1)
for index, (_, new_offset) in enumerate(offset_replacements):
    text = text.replace(f"R67_OFFSET_{index}", str(new_offset), 1)

print(text, end="")
