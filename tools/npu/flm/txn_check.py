#!/usr/bin/env python3
"""Cross-check mlir-aie's txn2mlir against an independent decode of a v1.0 TXN binary.

Two things have to agree before a decode of FLM's transaction binaries is
trustworthy: a raw structural walk of the bytes, and what txn2mlir produces.
This runs both and asserts they match. It is the check that caught mlir-aie
reading FLM's 24-byte DDR_PATCH with the 44-byte layout's offsets, which walks
8 bytes into the *following* operation and yields plausible garbage.

    python3 txn_check.py ../../../docs/npu/flm_c8_a.txn [...]

Set MLIR_AIE_BUILD to the mlir-aie build dir (default ~/build/mlir-aie/build).

The v1.0 opcode encoding is aie-rt's "optimized" family (xaiegbl.h):

    WRITE      0x00  12 bytes   XAie_Write32Hdr_opt
    BLOCKWRITE 0x01  size@+8    XAie_BlockWrite32Hdr_opt
    MASKWRITE  0x03  16 bytes   XAie_MaskWrite32Hdr_opt
    DDR_PATCH  0x81  size@+4    XAie_CustomOpHdr_opt + patch_op_opt_t (24 B)
    MERGE_SYNC 0x84  size@+4    3 words, payload (tokens | cols<<8)

patch_op_opt_t is u32 regaddr, u8 argidx, u8 pad[3], u64 argplus -- note the
32-bit regaddr and the absence of the v0.1 form's `action` field.
"""

import os
import re
import struct
import subprocess
import sys

WRITE, BLOCKWRITE, MASKWRITE, DDR_PATCH, MERGE_SYNC = 0x00, 0x01, 0x03, 0x81, 0x84

# XAIE_DEV_GEN_* values that mean npu2 (AIE2P, AIE2PS, Strix A0/B0).
NPU2_DEVGENS = {4, 5, 8, 9}


def decode(path):
    """Structural walk of a TXN binary. Returns (header, ops)."""
    d = open(path, "rb").read()
    if len(d) < 16:
        raise SystemExit(f"{path}: too small for a TXN header")
    major, minor, devgen, rows, cols, memtile_rows = d[0], d[1], d[2], d[3], d[4], d[5]
    num_ops, txn_size = struct.unpack_from("<II", d, 8)
    if (major, minor) != (1, 0):
        raise SystemExit(f"{path}: only v1.0 is handled here, got {major}.{minor}")

    ops, i = [], 16
    while i < len(d):
        op = d[i]
        if op == WRITE:
            size = 12
        elif op == BLOCKWRITE:
            size = struct.unpack_from("<I", d, i + 8)[0]
        elif op == MASKWRITE:
            size = 16
        else:
            size = struct.unpack_from("<I", d, i + 4)[0]
        if size <= 0 or i + size > len(d):
            raise SystemExit(f"{path}: bad size {size} for opcode {op:#x} at {i:#x}")

        rec = {"op": op, "off": i, "size": size}
        if op == DDR_PATCH:
            if size != 24:
                raise SystemExit(f"{path}: DDR_PATCH size {size} != 24 at {i:#x}")
            rec["addr"] = struct.unpack_from("<I", d, i + 8)[0]
            rec["argidx"] = d[i + 12]
            rec["argplus"] = struct.unpack_from("<Q", d, i + 16)[0]
        elif op == MERGE_SYNC:
            word = struct.unpack_from("<I", d, i + 8)[0]
            rec["num_tokens"], rec["num_cols"] = word & 0xFF, (word >> 8) & 0xFF
        ops.append(rec)
        i += size

    # Two independent header cross-checks: the walk must land exactly on the
    # declared byte count, and produce exactly the declared operation count.
    if i != len(d):
        raise SystemExit(f"{path}: walk ended at {i}, file is {len(d)} bytes")
    if txn_size != len(d):
        raise SystemExit(f"{path}: header txn_size {txn_size} != file size {len(d)}")
    if len(ops) != num_ops:
        raise SystemExit(f"{path}: walked {len(ops)} ops, header says {num_ops}")

    header = dict(major=major, minor=minor, devgen=devgen, rows=rows, cols=cols,
                  memtile_rows=memtile_rows, num_ops=num_ops, txn_size=txn_size)
    return header, ops


def run_txn2mlir(path, build):
    env = dict(os.environ, PYTHONPATH=os.path.join(build, "python"))
    # A stale mlir_aie wheel in site-packages shadows the build tree and silently
    # decodes with an older parser, so PYTHONPATH is not optional here.
    r = subprocess.run([sys.executable, os.path.join(build, "bin", "txn2mlir.py"),
                        "-f", path], capture_output=True, text=True, env=env)
    if r.returncode != 0 or "Failed to parse" in r.stdout + r.stderr:
        raise SystemExit(f"{path}: txn2mlir failed\n{r.stdout}\n{r.stderr}")
    return r.stdout


def check(path, build):
    header, ops = decode(path)
    mlir = run_txn2mlir(path, build)
    fails = []

    def eq(what, a, b):
        if a != b:
            fails.append(f"{what}: raw={a!r} txn2mlir={b!r}")

    want_dev = "npu2" if header["devgen"] in NPU2_DEVGENS else "npu1"
    m = re.search(r"aie\.device\((\w+)\)", mlir)
    got_dev = m.group(1) if m else None
    # npu1/npu2 with all columns has no _Ncol suffix; fewer columns do.
    expect = want_dev if header["cols"] in (4, 8) else f"{want_dev}_{header['cols']}col"
    eq("device", expect, got_dev)

    for opcode, name in ((WRITE, "write32"), (MASKWRITE, "maskwrite32"),
                         (BLOCKWRITE, "blockwrite"), (DDR_PATCH, "address_patch"),
                         (MERGE_SYNC, "merge_sync")):
        eq(f"count({name})", sum(o["op"] == opcode for o in ops),
           len(re.findall(rf"aiex\.npu\.{name}\b", mlir)))

    patches = [o for o in ops if o["op"] == DDR_PATCH]
    got = re.findall(r"aiex\.npu\.address_patch\(%c(-?\d+)_i32 : i32\) "
                     r"\{addr = (\d+) : ui32, arg_idx = (-?\d+) : i32\}", mlir)
    eq("address_patch parsed", len(patches), len(got))
    for p, (argplus, addr, argidx) in zip(patches, got):
        eq(f"patch@{p['off']:#x} addr", p["addr"], int(addr))
        eq(f"patch@{p['off']:#x} arg_idx", p["argidx"], int(argidx))
        eq(f"patch@{p['off']:#x} argplus", p["argplus"], int(argplus))

    for ms in (o for o in ops if o["op"] == MERGE_SYNC):
        m = re.search(r"merge_sync \{num_cols = (\d+) : ui8, "
                      r"num_tokens = (\d+) : ui8\}", mlir)
        if not m:
            fails.append("merge_sync: absent or printed in generic form")
        else:
            eq("merge_sync num_cols", ms["num_cols"], int(m.group(1)))
            eq("merge_sync num_tokens", ms["num_tokens"], int(m.group(2)))

    name = os.path.basename(path)
    if fails:
        print(f"FAIL {name}")
        for f in fails:
            print(f"       {f}")
        return False
    print(f"PASS {name}: v{header['major']}.{header['minor']} devgen={header['devgen']} "
          f"cols={header['cols']} {header['num_ops']} ops -> {got_dev}, "
          f"{len(patches)} patches, argidx={sorted({p['argidx'] for p in patches})}")
    return True


def main():
    args = sys.argv[1:]
    if not args:
        here = os.path.dirname(os.path.abspath(__file__))
        args = [os.path.join(here, "../../../docs/npu", f)
                for f in ("flm_c8_a.txn", "flm_c8_b.txn")]
    build = os.environ.get("MLIR_AIE_BUILD",
                           os.path.expanduser("~/build/mlir-aie/build"))
    sys.exit(0 if all([check(p, build) for p in args]) else 1)


if __name__ == "__main__":
    main()
