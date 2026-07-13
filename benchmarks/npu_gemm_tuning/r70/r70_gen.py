#!/usr/bin/env python3
"""Fuse W4 projection and inline QKV packing in one two-input-channel graph."""

from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
R65 = HERE.parent / "r65" / "r65_gen.py"
R66 = HERE.parent / "r66" / "r66_gen.py"
R71_PACK_FREE = "--r71-pack-free-4-7" in sys.argv[1:]
R72_DIRECT_Q = "--r72-direct-q" in sys.argv[1:]
R72_LOCAL_Q = "--r72-local-q" in sys.argv[1:]
R73_ADJACENT_Q = "--r73-adjacent-q" in sys.argv[1:]
R78_ODD_ATTENTION = "--r78-odd-attention" in sys.argv[1:]

SOURCE_ACTIVATION_BYTES = 589_824
ACTIVATION_BYTES = 737_280
WEIGHT_BYTES = 2_359_296
STAGE_BYTES = 2_457_600
Q_BYTES = 393_216
KV_BYTES = 262_144


def generate(path, *args):
    return subprocess.run(
        [sys.executable, str(path), *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def projection_graph():
    """Pad each 8-KiB activation token to the 10-KiB FIFO reused by packing."""
    text = generate(R65)
    for name in ("ash", "abc"):
        text = re.sub(
            rf"(@{name}\d+[^\n]*memref<)8192(xi8>)",
            rf"\g<1>10240\g<2>",
            text,
        )
    for marker in ("aie.objectfifo.acquire @abc", "aie.objectfifo.subview.access %a"):
        lines = []
        for line in text.splitlines():
            if marker in line:
                line = line.replace("memref<8192xi8>", "memref<10240xi8>")
            lines.append(line)
        text = "\n".join(lines) + "\n"
    text = text.replace(
        "(memref<8192xi8>, memref<16384xi8>, memref<2304xi32>)",
        "(memref<10240xi8>, memref<16384xi8>, memref<2304xi32>)",
    )
    init_decl = (
        '    func.func private @r15_w4_scaled_init(memref<10240xi8>, '
        'memref<16384xi8>, memref<2304xi32>) attributes {link_with = "r15.o"}'
    )
    accum_decl = (
        '    func.func private @r15_w4_scaled_accum(memref<10240xi8>, '
        'memref<16384xi8>, memref<2304xi32>) attributes {link_with = "r15.o"}'
    )
    group_decl = (
        '    func.func private @r70_w4_scaled_group(memref<10240xi8>, '
        'memref<16384xi8>, memref<2304xi32>, i32) '
        'attributes {link_with = "r70group.o"}'
    )
    if init_decl not in text or accum_decl not in text:
        raise SystemExit("R65 projection declarations changed")
    text = text.replace(init_decl, group_decl, 1).replace(accum_decl + "\n", "", 1)
    call_pattern = re.compile(
        r"func\.call @r15_w4_scaled_(init|accum)\((%a[^,]+), (%w[^,]+), "
        r"(%acc[^)]+)\) : \(memref<10240xi8>, memref<16384xi8>, "
        r"memref<2304xi32>\) -> \(\)"
    )

    def replace_group_call(match):
        flag = "%h0" if match.group(1) == "init" else "%h1"
        return (
            f"func.call @r70_w4_scaled_group({match.group(2)}, {match.group(3)}, "
            f"{match.group(4)}, {flag}) : (memref<10240xi8>, memref<16384xi8>, "
            "memref<2304xi32>, i32) -> ()"
        )

    text, call_count = call_pattern.subn(replace_group_call, text)
    if call_count == 0 or "@r15_w4_scaled_" in text:
        raise SystemExit("failed to merge split W4 projection calls")
    text = text.replace(
        f"%A: memref<{SOURCE_ACTIVATION_BYTES}xi8>",
        f"%A: memref<{ACTIVATION_BYTES}xi8>",
    ).replace(
        f"%A : memref<{SOURCE_ACTIVATION_BYTES}xi8>",
        f"%A : memref<{ACTIVATION_BYTES}xi8>",
    )
    old_dims = "[<size = 18, stride = 8192>, <size = 16, stride = 512>, <size = 512, stride = 1>]"
    new_dims = "[<size = 18, stride = 10240>, <size = 20, stride = 512>, <size = 512, stride = 1>]"
    text = text.replace(old_dims, new_dims)
    for row in range(4):
        text = text.replace(
            f", {row * 147456}, 147456, {new_dims}",
            f", {row * 184320}, 184320, {new_dims}",
        )
    if "memref<589824xi8>" in text or old_dims in text:
        raise SystemExit("stale compact activation geometry remains")
    return text


def pack_graph():
    # The pack phase reuses the projection activation channel after projection.
    args = ["--split-kv-columns"]
    if R71_PACK_FREE:
        args.append("--r71-pack-free-4-7")
    if R72_DIRECT_Q:
        args.append("--r72-direct-q")
    if R72_LOCAL_Q:
        args.append("--r72-local-q")
    if R73_ADJACENT_Q:
        args.append("--r73-adjacent-q")
    if R78_ODD_ATTENTION:
        args.append("--r78-odd-attention")
    return (
        generate(R66, *args)
        .replace("@rsh", "@ash")
        .replace("@rbc", "@abc")
    )


def top_level_ops(text):
    lines = text.splitlines()
    if lines[:2] != ["module {", "  aie.device(npu2) {"] or lines[-2:] != ["  }", "}"]:
        raise SystemExit("source graph wrapper changed")
    body = lines[2:-2]
    ops = []
    index = 0
    while index < len(body):
        if not re.match(r"^    \S", body[index]):
            raise SystemExit(f"unexpected top-level continuation: {body[index]}")
        op = [body[index]]
        balance = body[index].count("{") - body[index].count("}")
        index += 1
        while balance > 0:
            if index >= len(body):
                raise SystemExit("unterminated top-level operation")
            op.append(body[index])
            balance += body[index].count("{") - body[index].count("}")
            index += 1
        if balance != 0:
            raise SystemExit(f"unbalanced top-level operation: {op[0]}")
        ops.append(op)
    return ops


def core_key(op):
    match = re.search(r"%core(\d+)_(\d+) = aie\.core", op[0])
    return tuple(map(int, match.groups())) if match else None


def is_runtime(op):
    return "aie.runtime_sequence(" in op[0]


def outer_parts(op):
    outer = next(
        (index for index, line in enumerate(op) if "scf.for %outer" in line),
        None,
    )
    end = next(
        (
            index
            for index in range(len(op) - 1, outer or 0, -1)
            if op[index] == "      }"
        ),
        None,
    )
    if outer is None or end is None:
        raise SystemExit(f"core loop shape changed: {op[0]}")
    return op[:outer], op[outer], op[outer + 1 : end], op[end:]


def merge_core(projection, pack):
    prefix, outer, projection_body, suffix = outer_parts(projection)
    pack_prefix, _, pack_body, _ = outer_parts(pack)
    existing = {line.strip().split(" =", 1)[0] for line in prefix if " =" in line}
    for line in pack_prefix[1:]:
        if " =" not in line:
            continue
        symbol = line.strip().split(" =", 1)[0]
        if symbol not in existing:
            prefix.append(line)
            existing.add(symbol)
    suffix = [line.replace("stack_size = 2048", "stack_size = 4096") for line in suffix]
    return prefix + [outer] + projection_body + pack_body + suffix


def runtime_body(op):
    if not is_runtime(op) or op[-1] != "    }":
        raise SystemExit("runtime sequence shape changed")
    return op[1:-1]


projection_ops = top_level_ops(projection_graph())
pack_ops = top_level_ops(pack_graph())

projection_cores = {core_key(op): op for op in projection_ops if core_key(op) is not None}
pack_cores = {core_key(op): op for op in pack_ops if core_key(op) is not None}
if projection_cores.keys() != pack_cores.keys() or len(projection_cores) != 32:
    raise SystemExit("projection/pack core grids differ")

projection_runtime = next(op for op in projection_ops if is_runtime(op))
pack_runtime = next(op for op in pack_ops if is_runtime(op))

combined = []
for op in projection_ops:
    if core_key(op) is None and not is_runtime(op):
        combined.extend(op)

for op in pack_ops:
    first = op[0]
    if core_key(op) is not None or is_runtime(op):
        continue
    if re.match(r"    %(?:shim|mt|c)\d", first):
        continue
    if "@oc" in first or "@osh" in first or "@ash" in first or "@abc" in first:
        continue
    combined.extend(op)

for key in sorted(projection_cores):
    combined.extend(merge_core(projection_cores[key], pack_cores[key]))

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
