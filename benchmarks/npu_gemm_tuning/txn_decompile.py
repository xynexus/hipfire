#!/usr/bin/env python3
"""Decompile an AIE transaction binary (insts.bin) to its operation stream.

Handles both txn versions hipfire and FastFlowLM emit — v0.1 (what our mlir-aie
checkout produces) and v1.0 (what FLM ships) — which have DIFFERENT field
offsets for the same opcodes, so a v0.1 parser silently mis-walks a v1.0 blob.
Field layouts follow mlir-aie's lib/Conversion/AIEToConfiguration/
AIEToConfiguration.cpp; opcodes follow third_party/aie-rt .../xaie_txn.h.

mlir-aie's own txn2mlir.py refuses FLM's binaries ("Failed to parse binary")
because they end in MERGE_SYNC (0x84), which that tree has ONLY as an enum
value — no emitter and no parser case. Hence this script.

Usage: txn_decompile.py FILE [--order]
       --order prints the op sequence with column numbers (addr >> 25),
       run-length collapsed, which is how you see whether a schedule
       configures all columns then syncs, or interleaves config and sync.
"""
import collections
import struct
import sys

OPS = {0: "WRITE", 1: "BLOCKWRITE", 3: "MASKWRITE",
       0x80: "TCT", 0x81: "DDR_PATCH", 0x84: "MERGE_SYNC"}


def parse(d):
    """Yield (opcode, offset, size, detail) for each op in the blob."""
    v1 = d[0] == 1
    nops = struct.unpack_from("<I", d, 8)[0]
    r32 = lambda o: struct.unpack_from("<I", d, o)[0]
    i, n = 16, 0
    while i < len(d) and n < nops:
        op = d[i]
        detail = {}
        if op == 0:  # WRITE: v1 [op][addr][val]; v0.1 carries its own size
            sz = 12 if v1 else r32(i + 20)
            detail["addr"] = r32(i + 4) if v1 else r32(i + 8)
        elif op == 1:  # BLOCKWRITE: header 12 (v1) / 16 (v0.1), then payload
            detail["addr"] = r32(i + 4) if v1 else r32(i + 8)
            sz = r32(i + 8) if v1 else r32(i + 12)
            detail["payload"] = sz - (12 if v1 else 16)
        elif op == 3:  # MASKWRITE
            sz = 16 if v1 else r32(i + 24)
            detail["addr"] = r32(i + 4) if v1 else r32(i + 8)
        elif op == 0x80:  # CUSTOM_OP_TCT — one task-completion wait
            sz = r32(i + 4)
            desc, cfg = r32(i + 8), r32(i + 12)
            detail = dict(col=(desc >> 16) & 0xFF, row=(desc >> 8) & 0xFF,
                          dir=desc & 0xFF, ch=(cfg >> 24) & 0xFF,
                          ncol=(cfg >> 16) & 0xFF, nrow=(cfg >> 8) & 0xFF)
        elif op == 0x81:  # CUSTOM_OP_DDR_PATCH — relocate a BD's DDR address
            sz = r32(i + 4)
            detail = dict(arg=struct.unpack_from("<i", d, i + 32)[0],
                          plus=struct.unpack_from("<i", d, i + 40)[0])
        elif 0x82 <= op <= 0x85:  # newer custom ops; body is opaque here
            sz = r32(i + 4)
            detail["words"] = [r32(i + k) for k in range(0, sz, 4)]
        else:
            print(f"!! unknown opcode 0x{op:02x} at 0x{i:x} after {n} ops")
            return
        if sz <= 0:
            print(f"!! zero-size op 0x{op:02x} at 0x{i:x}")
            return
        yield op, i, sz, detail
        i += sz
        n += 1


def main():
    path = sys.argv[1]
    d = open(path, "rb").read()
    nops, tsz = struct.unpack_from("<II", d, 8)
    print(f"v{d[0]}.{d[1]} devgen={d[2]} rows={d[3]} cols={d[4]} memrows={d[5]} "
          f"ops={nops} size={tsz} file={len(d)}")
    parsed = list(parse(d))
    cnt = collections.Counter(op for op, _, _, _ in parsed)
    end = parsed[-1][1] + parsed[-1][2] if parsed else 16
    print(f"parsed {len(parsed)}/{nops} ops, consumed {end}/{len(d)} bytes")
    print("mix:", {OPS.get(k, hex(k)): v for k, v in sorted(cnt.items())})

    if "--order" in sys.argv:
        seq = []
        for op, _, _, det in parsed:
            if op in (0, 1, 3):
                seq.append(f"{OPS[op][0]}:c{det['addr'] >> 25}")
            elif op == 0x80:
                seq.append(f"TCT:c{det['col']}")
            elif op == 0x81:
                seq.append(f"PATCH:arg{det['arg']}")
            else:
                seq.append(OPS.get(op, f"OP{op:02x}"))
        out, prev, run = [], None, 0
        for s in seq:
            if s == prev:
                run += 1
            else:
                if prev:
                    out.append(prev + (f"x{run}" if run > 1 else ""))
                prev, run = s, 1
        out.append(prev + (f"x{run}" if run > 1 else ""))
        print("order:", " ".join(out))
        return

    bw = [det["payload"] for op, _, _, det in parsed if op == 1]
    print(f"blockwrite payload = {sum(bw)} bytes over {len(bw)} ops")
    for op, label in ((0x80, "TCT syncs"), (0x81, "DDR patches")):
        items = [det for o, _, _, det in parsed if o == op]
        print(f"{label}: {len(items)}", items[:8])


if __name__ == "__main__":
    main()
