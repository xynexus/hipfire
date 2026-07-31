#!/usr/bin/env python3
"""Decode the DMA buffer descriptors an AIE PDI's CDO programs, per tile.

`cdo.py` extracts core *program* memory from the CDO. This extracts the other
half: the DMA configuration. Every buffer descriptor written by the CDO is
decoded into (L1 base address, length, dimensions, locks), which is what maps a
core's operand pointers onto DMA streams — and therefore onto producers.

    python3 cdo_dma.py /opt/fastflowlm/share/flm/xclbins/Llama-3.2-1B-NPU2/layer.xclbin
    python3 cdo_dma.py layer.xclbin --tile 0,2

Field definitions are read at run time from aie-rt's AIE2P register header
(`xaie2pgbl_params.h`) rather than hardcoded, so the decode cannot drift from
the vendor's definition. Point at a different tree with AIE_RT_PARAMS.

Register blocks decoded:
  core tile (rows >= 2)  MEMORY_MODULE  BD base 0x1D000, 6 words, stride 0x20
  memtile   (row 1)      MEM_TILE       BD base 0xA0000, 8 words, stride 0x20
  shim      (row 0)      NOC            BD base 0x1D000, 8 words, stride 0x20
"""

import argparse
import os
import re
import struct
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdo import parse, split  # noqa: E402

DEFAULT_PARAMS = os.path.expanduser(
    "~/build/mlir-aie/third_party/aie-rt/driver/src/global/xaie2pgbl_params.h")

BLOCKS = {
    "core":    ("MEMORY_MODULE", 0x1D000, 6),
    "memtile": ("MEM_TILE_MODULE", 0xA0000, 8),
    "shim":    ("NOC_MODULE", 0x1D000, 8),
}


def load_fields(params_path, prefix):
    """{word_index: [(field_name, mask, lsb)]} for <prefix>_DMA_BD0_<w>_<field>."""
    txt = open(params_path).read()
    masks, lsbs = {}, {}
    rx = re.compile(rf"#define XAIE2PGBL_{prefix}_DMA_BD0_(\d+)_([A-Z0-9_]+)_(MASK|LSB)\s+(0[xX][0-9A-Fa-f]+|\d+)")
    for m in rx.finditer(txt):
        w, name, kind, val = int(m.group(1)), m.group(2), m.group(3), int(m.group(4), 0)
        (masks if kind == "MASK" else lsbs)[(w, name)] = val
    out = defaultdict(list)
    for (w, name), mask in masks.items():
        lsb = lsbs.get((w, name))
        if lsb is None:                      # derive from the mask
            lsb = (mask & -mask).bit_length() - 1
        out[w].append((name, mask, lsb))
    for w in out:
        out[w].sort(key=lambda t: -t[1])
    return out


def collect_writes(ops):
    """{(col,row): {offset: value}} for every scalar/block write in the CDO."""
    regs = defaultdict(dict)
    for cmd, p in ops:
        if cmd == 0x105:                      # BLOCKWRITE64 (hi, lo, data...)
            if len(p) < 2:
                continue
            addr, data = p[1], p[2:]
        elif cmd == 0x103:                    # WRITE64 (addr, val)
            addr, data = p[0], p[1:2]
        elif cmd == 0x102:                    # MASKWRITE64 (addr, mask, val)
            addr, data = p[0], p[2:3]
        else:
            continue
        c, r, o = split(addr)
        for k, val in enumerate(data):
            regs[(c, r)][o + 4 * k] = val
    return regs


def decode_bds(regs_for_tile, base, nwords, fields):
    """Yield (bd_index, {field: value}) for each BD with any register written."""
    for bd in range(64):
        off = base + bd * 0x20
        words = [regs_for_tile.get(off + 4 * w) for w in range(nwords)]
        if all(x is None for x in words):
            continue
        vals = {}
        for w, word in enumerate(words):
            if word is None:
                continue
            for name, mask, lsb in fields.get(w, []):
                v = (word & mask) >> lsb
                if v:
                    vals[name] = v
        yield bd, vals, words


def kind_for(row):
    return "shim" if row == 0 else ("memtile" if row == 1 else "core")


# --- stream-switch decoding -------------------------------------------------
#
# A "master" port is an output of the switch; its config register selects which
# slave (switch input) feeds it. So MASTER_CONFIG_DMA0 names the source of the
# data arriving at DMA channel 0. Physical port indices are architecture
# specific and are read from aie-rt's own port maps rather than assumed.

DEFAULT_REGINIT = os.path.expanduser(
    "~/build/mlir-aie/third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c")

PORTMAP_ARRAY = {
    "core":    ("Aie2PTileStrmSwMasterPortMap", "Aie2PTileStrmSwSlavePortMap"),
    "memtile": ("Aie2PMemTileStrmSwMasterPortMap", "Aie2PMemTileStrmSwSlavePortMap"),
    "shim":    ("Aie2PShimStrmSwMasterPortMap", "Aie2PShimStrmSwSlavePortMap"),
}
SS_MASTER_BASE = {"core": 0x3F000, "memtile": 0xB0000, "shim": 0x3F000}
SS_SLAVE_BASE = {"core": 0x3F100, "memtile": 0xB0100, "shim": 0x3F100}


def load_portmaps(reginit_path):
    txt = open(reginit_path).read()
    out = {}
    for kind, (mname, sname) in PORTMAP_ARRAY.items():
        pair = []
        for arr in (mname, sname):
            i = txt.index(f"{arr}[] =")
            body = txt[i:txt.index("};", i)]
            m = {int(g.group(1)): f"{g.group(2)}{g.group(3)}" for g in re.finditer(
                r"/\* PhyPort (\d+) \*/\s*\.PortType = (\w+),\s*\.PortNum = (\d+)", body)}
            pair.append(m)
        out[kind] = tuple(pair)
    return out


# Which neighbour a directional port actually reaches, as (dcol, drow).
NEIGHBOUR = {"EAST": (1, 0), "WEST": (-1, 0), "NORTH": (0, 1), "SOUTH": (0, -1)}


def ss_edges(regs_for_tile, kind, portmaps):
    """[(master_port, slave_port, mode)] for every enabled master on this tile."""
    masters, slaves = portmaps[kind]
    base = SS_MASTER_BASE[kind]
    out = []
    for phy, mname in sorted(masters.items()):
        v = regs_for_tile.get(base + 4 * phy)
        if v is None or not (v >> 31) & 1:
            continue
        src = slaves.get(v & 0x7F, f"phy{v & 0x7F}")
        out.append((mname, src, "packet" if (v >> 30) & 1 else "circuit"))
    return out


def main():
    ap = argparse.ArgumentParser(description="Decode DMA BDs from an AIE PDI/xclbin CDO")
    ap.add_argument("path")
    ap.add_argument("--tile", help="only this tile, as col,row")
    ap.add_argument("--params", default=os.environ.get("AIE_RT_PARAMS", DEFAULT_PARAMS))
    ap.add_argument("--raw", action="store_true", help="also print raw BD words")
    ap.add_argument("--graph", action="store_true",
                    help="print the stream-switch connectivity graph instead of BDs")
    ap.add_argument("--reginit", default=os.environ.get("AIE_RT_REGINIT", DEFAULT_REGINIT))
    o = ap.parse_args()

    if not os.path.exists(o.params):
        sys.exit(f"aie-rt params header not found: {o.params}\n"
                 f"set AIE_RT_PARAMS to xaie2pgbl_params.h")

    fields = {k: load_fields(o.params, BLOCKS[k][0]) for k in BLOCKS}
    _ver, ops = parse(o.path)
    regs = collect_writes(ops)

    want = None
    if o.tile:
        c, r = (int(x) for x in o.tile.split(","))
        want = (c, r)

    if o.graph:
        if not os.path.exists(o.reginit):
            sys.exit(f"aie-rt reginit not found: {o.reginit}")
        portmaps = load_portmaps(o.reginit)
        print(f"{'tile':9s} {'kind':8s} {'master':22s} <- {'slave':12s} mode")
        edges = 0
        for (c, r) in sorted(regs, key=lambda t: (t[1], t[0])):
            if want and (c, r) != want:
                continue
            kind = kind_for(r)
            for mname, src, mode in ss_edges(regs[(c, r)], kind, portmaps):
                # Annotate directional sources with the tile they actually reach.
                note = ""
                for d, (dc, dr) in NEIGHBOUR.items():
                    if src.startswith(d):
                        note = f"   from ({c + dc},{r + dr})"
                        break
                print(f"({c},{r})     {kind:8s} {mname:22s} <- {src:12s} {mode}{note}")
                edges += 1
        print(f"\n{edges} enabled stream-switch route(s)")
        return

    total = 0
    for (c, r) in sorted(regs):
        if want and (c, r) != want:
            continue
        kind = kind_for(r)
        _prefix, base, nwords = BLOCKS[kind]
        found = list(decode_bds(regs[(c, r)], base, nwords, fields[kind]))
        if not found:
            continue
        print(f"\n=== tile ({c},{r})  [{kind}] — {len(found)} BD(s) ===")
        for bd, vals, words in found:
            # Core-tile BASE_ADDRESS and BUFFER_LENGTH are both in 32-bit WORDS.
            # Both fields are 14 bits, which spans exactly the 64 KB data-memory
            # module at 4 bytes per unit -- and the resulting addresses match the
            # GEMM core's operand pointers exactly (p1 -> 0x2800, p4 -> 0x4000,
            # p5 -> 0xc000), which is the check that fixes the unit.
            #
            # The core addresses its own module through a 0x70000 window, so the
            # core-visible pointer is 0x70000 + base*4.
            note = ""
            if kind == "core" and "BASE_ADDRESS" in vals:
                a = vals["BASE_ADDRESS"] * 4
                note = f"   module 0x{a:05x}  core-view 0x{0x70000 + a:05x}"
            if "BUFFER_LENGTH" in vals:
                note += f"   len {vals['BUFFER_LENGTH'] * 4} B"
            print(f"  BD{bd:<3d}{note}")
            print("       " + ", ".join(f"{k}={v}" for k, v in sorted(vals.items())))
            if o.raw:
                print("       raw: " + " ".join(
                    "----" if w is None else f"{w:08x}" for w in words))
            total += 1
    print(f"\n{total} buffer descriptor(s) decoded")


if __name__ == "__main__":
    main()
