#!/usr/bin/env python3
"""Fuse R80 paired projection with R66 external Q/K/V packing."""

from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
R80 = HERE.parent / "r80" / "r80_gen.py"
R66 = HERE.parent / "r66" / "r66_gen.py"
SOURCE_ACTIVATION_BYTES = 589_824
ACTIVATION_BYTES = 737_280
WEIGHT_BYTES = 2_359_296
STAGE_BYTES = 2_457_600
Q_BYTES = 393_216
KV_BYTES = 262_144
PROJECTION_BLOCKS = 18
SINGLE_GROUP_FUNCTION = "--single-group-function" in sys.argv[1:]
DYNAMIC_SLICE_LOOP = "--dynamic-slice-loop" in sys.argv[1:]


def generate(path, *args):
    return subprocess.run(
        [sys.executable, str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def projection_graph():
    args = []
    if SINGLE_GROUP_FUNCTION:
        args.append("--single-group-function")
    if DYNAMIC_SLICE_LOOP:
        args.append("--dynamic-slice-loop")
    text = generate(R80, *args)
    for name in ("ash", "abc"):
        text = re.sub(
            rf"(@{name}\d+[^\n]*memref<)8192(xi8>)",
            rf"\g<1>10240\g<2>",
            text,
        )
    lines = []
    for line in text.splitlines():
        if "aie.objectfifo.acquire @abc" in line or "aie.objectfifo.subview.access %a" in line:
            line = line.replace("memref<8192xi8>", "memref<10240xi8>")
        lines.append(line)
    text = "\n".join(lines) + "\n"
    text = text.replace(
        "(memref<8192xi8>, memref<16384xi8>, memref<2304xi32>)",
        "(memref<10240xi8>, memref<16384xi8>, memref<2304xi32>)",
    )
    text = text.replace(
        "(memref<8192xi8>, memref<16384xi8>, memref<2304xi32>, i32)",
        "(memref<10240xi8>, memref<16384xi8>, memref<2304xi32>, i32)",
    )
    text = text.replace(
        f"memref<{SOURCE_ACTIVATION_BYTES}xi8>",
        f"memref<{ACTIVATION_BYTES}xi8>",
    )
    old_dims = "[<size = 18, stride = 8192>, <size = 16, stride = 512>, <size = 512, stride = 1>]"
    new_dims = "[<size = 18, stride = 10240>, <size = 20, stride = 512>, <size = 512, stride = 1>]"
    text = text.replace(old_dims, new_dims)
    for row in range(4):
        text = text.replace(
            f", {row * 147456}, 147456, {new_dims}",
            f", {row * 184320}, 184320, {new_dims}",
        )
    for row in range(4):
        odd = ", ".join(f"%c{col}_{row}" for col in range(1, 8, 2))
        all_cores = ", ".join(f"%c{col}_{row}" for col in range(8))
        text = text.replace(
            f"@abc{row}(%mt{row}, {{{odd}}}",
            f"@abc{row}(%mt{row}, {{{all_cores}}}",
        )
    if "memref<589824xi8>" in text or old_dims in text:
        raise SystemExit("stale compact activation geometry remains")
    return text


def pack_graph():
    return (
        generate(R66, "--split-kv-columns", "--r78-odd-attention")
        .replace("@rsh", "@ash")
        .replace("@rbc", "@abc")
    )


def top_level_ops(text):
    lines = text.splitlines()
    if lines[:2] != ["module {", "  aie.device(npu2) {"] or lines[-2:] != ["  }", "}"]:
        raise SystemExit("graph wrapper changed")
    body = lines[2:-2]
    ops = []
    index = 0
    while index < len(body):
        op = [body[index]]
        balance = body[index].count("{") - body[index].count("}")
        index += 1
        while balance > 0:
            op.append(body[index])
            balance += body[index].count("{") - body[index].count("}")
            index += 1
        ops.append(op)
    return ops


def core_key(op):
    match = re.search(r"%core(\d+)_(\d+) = aie\.core", op[0])
    return tuple(map(int, match.groups())) if match else None


def is_runtime(op):
    return "aie.runtime_sequence(" in op[0]


def outer_parts(op):
    outer = next(i for i, line in enumerate(op) if "scf.for %outer" in line)
    end = next(i for i in range(len(op) - 1, outer, -1) if op[i] == "      }")
    return op[:outer], op[outer], op[outer + 1 : end], op[end:]


def add_prefix_constants(prefix, source_prefix):
    existing = {line.strip().split(" =", 1)[0] for line in prefix if " =" in line}
    for line in source_prefix[1:]:
        if " =" not in line:
            continue
        symbol = line.strip().split(" =", 1)[0]
        if symbol not in existing:
            prefix.append(line)
            existing.add(symbol)


def merge_odd(projection, pack):
    prefix, outer, projection_body, suffix = outer_parts(projection)
    pack_prefix, _, pack_body, _ = outer_parts(pack)
    add_prefix_constants(prefix, pack_prefix)
    suffix = [line.replace("stack_size = 2048", "stack_size = 4096") for line in suffix]
    return prefix + [outer] + projection_body + pack_body + suffix


def even_with_projection_drain(pack, row):
    prefix, outer, pack_body, suffix = outer_parts(pack)
    prefix.append(f"      %pblocks = arith.constant {PROJECTION_BLOCKS} : index")
    drain = [
        "        scf.for %pdrain = %z to %pblocks step %one {",
        f"          %pda = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<10240xi8>>",
        f"          aie.objectfifo.release @abc{row}(Consume, 1)",
        "        }",
    ]
    return prefix + [outer] + drain + pack_body + suffix


def runtime_body(op):
    return op[1:-1]


projection_ops = top_level_ops(projection_graph())
pack_ops = top_level_ops(pack_graph())
projection_cores = {core_key(op): op for op in projection_ops if core_key(op) is not None}
pack_cores = {core_key(op): op for op in pack_ops if core_key(op) is not None}
if len(projection_cores) != 16 or len(pack_cores) != 32:
    raise SystemExit("unexpected paired projection or pack core grid")

projection_runtime = next(op for op in projection_ops if is_runtime(op))
pack_runtime = next(op for op in pack_ops if is_runtime(op))
combined = []
for op in projection_ops:
    if core_key(op) is None and not is_runtime(op):
        combined.extend(op)

for op in pack_ops:
    if core_key(op) is not None or is_runtime(op):
        continue
    first = op[0]
    if re.match(r"    %(?:shim|mt|c)\d", first):
        continue
    if "@ash" in first or "@abc" in first:
        continue
    output_match = re.search(r"@(?:oc|osh)(\d+)", first)
    if output_match and int(output_match.group(1)) % 2 == 1:
        continue
    combined.extend(op)

for key in sorted(pack_cores):
    col, row = key
    if col % 2 == 1:
        combined.extend(merge_odd(projection_cores[key], pack_cores[key]))
    else:
        combined.extend(even_with_projection_drain(pack_cores[key], row))

combined.append(
    "    aie.runtime_sequence("
    f"%A: memref<{ACTIVATION_BYTES}xi8>, %W: memref<{WEIGHT_BYTES}xi8>, "
    f"%R: memref<{STAGE_BYTES}xi8>, %Q: memref<{Q_BYTES}xi8>, "
    f"%KV: memref<{KV_BYTES}xi8>) {{"
)
combined.extend(runtime_body(projection_runtime))
combined.extend(runtime_body(pack_runtime))
combined.append("    }")
print("\n".join(["module {", "  aie.device(npu2) {", *combined, "  }", "}"]))
