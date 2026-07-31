#!/usr/bin/env python3
"""Find every embedded AIE2P transaction binary in a file and classify what it programs.

FLM ships its runtime sequences inside the per-kernel shared objects, not the
xclbins. This locates them by structural validation (not a magic number -- a TXN
header has none) and then decodes each one far enough to say *what it does*:
which tile rows it touches, which DMA channels and in which direction, and how
many bytes its buffer descriptors move.

    python3 txn_scan.py /opt/fastflowlm/lib/*.so
    python3 txn_scan.py --dump out/ /opt/fastflowlm/lib/libgemm.so

Direction is the useful discriminator. A weight-ingress program drives shim
**MM2S** (DDR -> AIE); an egress program drives shim **S2MM** (AIE -> DDR).
`flm_c8_a/b.txn` turned out to be pure egress, which is why they could not be
the weight-streaming layer dispatch they were assumed to be.

Register offsets are from aie-rt's own AIE2P definitions
(third_party/aie-rt/driver/src/global/xaie2pgbl_params.h), not inferred.
"""

import argparse
import os
import struct
import sys
from collections import Counter, defaultdict

WRITE, BLOCKWRITE, MASKWRITE, DDR_PATCH, MERGE_SYNC = 0x00, 0x01, 0x03, 0x81, 0x84
COL_SHIFT = 25                      # AIE2P column stride 0x2000000
ROW_SHIFT, ROW_MASK = 20, 0x1F      # tile row within a column

# (row_kind, direction, channel) keyed by register offset within the tile.
DMA_REGS = {}
for ch in range(2):
    DMA_REGS[0x1D200 + ch * 8] = ("shim", "S2MM", ch)      # CTRL
    DMA_REGS[0x1D204 + ch * 8] = ("shim", "S2MM", ch)      # TASK_QUEUE
    DMA_REGS[0x1D210 + ch * 8] = ("shim", "MM2S", ch)
    DMA_REGS[0x1D214 + ch * 8] = ("shim", "MM2S", ch)
    DMA_REGS[0x1DE00 + ch * 8] = ("core", "S2MM", ch)
    DMA_REGS[0x1DE04 + ch * 8] = ("core", "S2MM", ch)
    DMA_REGS[0x1DE10 + ch * 8] = ("core", "MM2S", ch)
    DMA_REGS[0x1DE14 + ch * 8] = ("core", "MM2S", ch)
for ch in range(6):
    DMA_REGS[0xA0600 + ch * 8] = ("memtile", "S2MM", ch)
    DMA_REGS[0xA0604 + ch * 8] = ("memtile", "S2MM", ch)
    DMA_REGS[0xA0630 + ch * 8] = ("memtile", "MM2S", ch)
    DMA_REGS[0xA0634 + ch * 8] = ("memtile", "MM2S", ch)

# BD array bases, for turning a blockwrite target into "BD #n of <tile>".
BD_BASE = {"shim": 0x1D000, "core": 0x1D000, "memtile": 0xA0000}
# Word 7 valid bit differs between shim/core (NOC/MEMORY) and memtile.
VALID_BIT = {"shim": 0x02000000, "core": 0x02000000, "memtile": 0x80000000}


def op_size(d, i):
    op = d[i]
    if op == WRITE:
        return 12
    if op == BLOCKWRITE:
        return struct.unpack_from("<I", d, i + 8)[0]
    if op == MASKWRITE:
        return 16
    return struct.unpack_from("<I", d, i + 4)[0]


def walk(d, start):
    """Validate a TXN v1.0 header at `start`. Return (header, ops) or None.

    There is no magic number, so validity IS the test: the op walk must land
    exactly on the declared txn_size and produce exactly num_ops operations.
    Two independent conditions, which makes false positives very unlikely.
    """
    if start + 16 > len(d):
        return None
    major, minor, devgen = d[start], d[start + 1], d[start + 2]
    rows, cols, memtile_rows = d[start + 3], d[start + 4], d[start + 5]
    if (major, minor) != (1, 0) or devgen != 4:      # v1.0, XAIE_DEV_GEN_AIE2P
        return None
    if not (1 <= cols <= 8) or not (1 <= rows <= 16):
        return None
    num_ops, txn_size = struct.unpack_from("<II", d, start + 8)
    if not (16 < txn_size <= 1 << 20) or start + txn_size > len(d):
        return None
    if not (0 < num_ops <= 1 << 16):
        return None

    ops, i, end = [], start + 16, start + txn_size
    while i < end:
        op = d[i]
        if op not in (WRITE, BLOCKWRITE, MASKWRITE, DDR_PATCH, MERGE_SYNC):
            return None
        try:
            sz = op_size(d, i)
        except struct.error:
            return None
        if sz <= 0 or i + sz > end:
            return None
        ops.append((op, i, sz))
        i += sz
    if i != end or len(ops) != num_ops:
        return None
    return dict(offset=start, cols=cols, rows=rows, memtile_rows=memtile_rows,
                num_ops=num_ops, txn_size=txn_size), ops


def classify(d, hdr, ops):
    """Summarise which tiles/DMA directions a transaction programs."""
    dma = Counter()
    rows_touched = Counter()
    bds = []
    counts = Counter()
    for op, i, sz in ops:
        counts[op] += 1
        if op == WRITE:
            addr = struct.unpack_from("<I", d, i + 4)[0]
        elif op == MASKWRITE:
            addr = struct.unpack_from("<I", d, i + 4)[0]
        elif op == BLOCKWRITE:
            addr = struct.unpack_from("<I", d, i + 4)[0]
        else:
            continue
        row = (addr >> ROW_SHIFT) & ROW_MASK
        reg = addr & 0xFFFFF
        rows_touched[row] += 1
        hit = DMA_REGS.get(reg)
        if hit:
            dma[(hit[0], hit[1])] += 1
        if op == BLOCKWRITE and sz >= 12 + 32:
            words = struct.unpack_from("<8I", d, i + 12)
            kind = "memtile" if row == 1 else ("shim" if row == 0 else "core")
            bds.append((row, reg, kind, words))
    return counts, rows_touched, dma, bds


def describe_bds(bds):
    out = []
    for row, reg, kind, w in bds:
        base = BD_BASE[kind]
        bd_n = (reg - base) // 0x20 if reg >= base else -1
        length_words = w[0]
        valid = bool(w[7] & VALID_BIT[kind])
        iter_field = w[6]
        out.append(dict(row=row, kind=kind, bd=bd_n, length_words=length_words,
                        bytes=length_words * 4, valid=valid, iteration=iter_field))
    return out


import re as _re

# v1.0 + AIE2P is a 3-byte anchor. Finding candidates with a compiled regex runs
# the search in C; a Python `while i < len(d)` byte loop is fine on a 2 MB shared
# object but takes many minutes on the multi-gigabyte device BO regions that
# txn_memscan walks.
ANCHOR = _re.compile(rb"\x01\x00\x04")


def find_txns(d, start=0, end=None):
    """Yield (header, ops) for every valid transaction in d[start:end]."""
    end = len(d) if end is None else end
    pos = start
    while pos < end:
        m = ANCHOR.search(d, pos, end)
        if not m:
            return
        r = walk(d, m.start())
        if r:
            yield r
            pos = m.start() + r[0]["txn_size"]
        else:
            pos = m.start() + 1


def scan(path, dump_dir=None, verbose=False):
    d = open(path, "rb").read()
    found = list(find_txns(d))

    if not found:
        return []
    print(f"\n=== {path} — {len(found)} transaction(s) ===")
    print(f"{'offset':>10s} {'cols':>4s} {'ops':>5s} {'bytes':>7s}  "
          f"{'rows':<8s} {'DMA programmed':<34s} {'DDR bytes':>12s}")
    for hdr, ops in found:
        counts, rows, dma, bds = classify(d, hdr, ops)
        bdinfo = describe_bds(bds)
        # DDR traffic is the SHIM side only. The memtile BDs are the other end of
        # the same transfer (source or sink), so summing both double-counts. The
        # BD list already spans every column, so there is no further x cols.
        ddr_bytes = sum(b["bytes"] for b in bdinfo if b["valid"] and b["kind"] == "shim")
        dma_s = ", ".join(f"{t}:{dirn}x{n}" for (t, dirn), n in sorted(dma.items())) or "-"
        rows_s = ",".join(str(r) for r in sorted(rows))
        print(f"{hdr['offset']:#10x} {hdr['cols']:4d} {hdr['num_ops']:5d} "
              f"{hdr['txn_size']:7d}  {rows_s:<8s} {dma_s:<34s} {ddr_bytes:12,d}")
        if verbose:
            uniq = Counter((b["kind"], b["bd"], b["bytes"], b["valid"], b["iteration"])
                           for b in bdinfo)
            for (kind, bd, byts, valid, it), n in sorted(uniq.items()):
                print(f"             BD {kind}#{bd:<3d} x{n:<3d} {byts:>9,d} B  "
                      f"valid={valid} iter={it:#x}")
            print("             ops: " + ", ".join(
                f"{hex(k)}x{v}" for k, v in sorted(counts.items())))
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            name = f"{os.path.basename(path)}.{hdr['offset']:#x}.c{hdr['cols']}.txn"
            with open(os.path.join(dump_dir, name), "wb") as f:
                f.write(d[hdr["offset"]:hdr["offset"] + hdr["txn_size"]])
    return found


def main():
    p = argparse.ArgumentParser(description="Scan for embedded AIE2P TXN binaries")
    p.add_argument("files", nargs="+")
    p.add_argument("--dump", metavar="DIR", help="write each transaction out")
    p.add_argument("-v", "--verbose", action="store_true", help="per-BD detail")
    o = p.parse_args()
    total = 0
    for f in o.files:
        try:
            total += len(scan(f, o.dump, o.verbose))
        except (OSError, struct.error) as e:
            print(f"{f}: {e}", file=sys.stderr)
    print(f"\n{total} transaction(s) total")


if __name__ == "__main__":
    main()
