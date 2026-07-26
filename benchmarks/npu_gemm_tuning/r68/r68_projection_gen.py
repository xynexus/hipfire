#!/usr/bin/env python3
"""Retarget R65's 24x32 output objects to overlapping joined R28 records."""

from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
R65 = HERE.parent / "r65" / "r65_gen.py"
OLD_STAGE_BYTES = 2_457_600
PAIR, PAIRS_PER_ROLE, ROLES = 8192, 37, 5
PARAMS = 2048
STAGE_BYTES = ROLES * PAIRS_PER_ROLE * PAIR + PARAMS

text = subprocess.run(
    [sys.executable, str(R65)], check=True, capture_output=True, text=True
).stdout
text = text.replace(
    f"memref<{OLD_STAGE_BYTES}xi8>", f"memref<{STAGE_BYTES}xi8>"
)
text = text.replace(
    "[<size = 4, stride = 40960>, <size = 4, stride = 10240>, "
    "<size = 8, stride = 512>, <size = 64, stride = 1>]",
    "[<size = 4, stride = 24576>, <size = 4, stride = 8192>, "
    "<size = 8, stride = 512>, <size = 64, stride = 1>]",
)

for outblock in range(6):
    m_macro, n_macro = divmod(outblock, 2)
    for slice_index in range(3):
        for col in range(8):
            stripe32 = n_macro * 24 + col * 3 + slice_index
            if stripe32 >= ROLES * 8:
                continue
            role, role_stripe = divmod(stripe32, 8)
            new_offset = (
                (role * PAIRS_PER_ROLE + m_macro * 12) * PAIR
                + role_stripe * 64
            )
            task = f"to{outblock}_{slice_index}_{col}"
            pattern = re.compile(
                rf"(%{task} = aiex\.dma_configure_task_for @osh{col} \{{\n"
                rf"\s+aie\.dma_bd\(%R : memref<{STAGE_BYTES}xi8>, )"
                rf"\d+(, 2048,)"
            )
            text, count = pattern.subn(rf"\g<1>{new_offset}\g<2>", text, count=1)
            if count != 1:
                raise SystemExit(f"could not retarget {task}")

if "stride = 40960" in text or "stride = 10240" in text:
    raise SystemExit("stale R65 record stride remains")
print(text, end="")
