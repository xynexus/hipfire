#!/usr/bin/env python3
"""Parse the CDO blob inside an AIE PDI; classify ops and extract core program memory.

Command word: bits[15:0]=cmd, bits[23:16]=len (0xFF -> next word is 32-bit len).
Validated against FLM's layer.pdi: walks to exactly the header's word count.
"""
import struct, sys, collections, os

CMD = {0x102: 'MASKWRITE64', 0x103: 'WRITE64', 0x105: 'BLOCKWRITE64', 0x111: 'NOP'}

# AIE2 tile address: col<<25 | row<<20 | offset
def split(a): return (a >> 25) & 0x3f, (a >> 20) & 0x1f, a & 0xfffff

PM_BASE, PM_END = 0x20000, 0x30000   # AIE2 core program memory window


def parse(path):
    d = open(path, 'rb').read()
    off = d.find(b'CDO\x00')
    magic, ver, nwords, cksum = struct.unpack_from('<4sIII', d, off)
    w = struct.unpack_from(f'<{nwords}I', d, off + 16)
    ops, i = [], 0
    while i < len(w):
        cw = w[i]; cmd = cw & 0xffff; ln = (cw >> 16) & 0xff; j = i + 1
        if ln == 0xff:
            ln = w[j]; j += 1
        ops.append((cmd, w[j:j + ln]))
        i = j + ln
    assert i == len(w), f'walk ended {i} != {len(w)}'
    return ver, ops


def main(path, dump_dir=None):
    ver, ops = parse(path)
    print(f'{os.path.basename(path)}  CDO v{ver>>8}.{ver&0xff}  {len(ops)} ops')

    hist = collections.Counter()
    tiles = collections.Counter()
    pm = collections.defaultdict(dict)   # (col,row) -> {offset: word}
    for cmd, p in ops:
        hist[CMD.get(cmd, hex(cmd))] += 1
        # address position differs per command:
        #   0x102 MASKWRITE (addr, mask, val)   0x103 WRITE (addr, val)
        #   0x105 BLOCKWRITE64 (addr_hi, addr_lo, data...)
        if cmd == 0x105:
            if len(p) < 2:
                continue
            addr, data = p[1], p[2:]
        elif cmd in (0x102, 0x103):
            addr, data = p[0], (p[1:2] if cmd == 0x103 else ())
        else:
            continue
        c, r, o = split(addr)
        tiles[(c, r)] += 1
        if PM_BASE <= o < PM_END:
            for k, val in enumerate(data):
                pm[(c, r)][o + 4 * k] = val

    print('  ops:', dict(hist))
    print('  tiles touched:', len(tiles), '->', sorted(tiles)[:12], '...' if len(tiles) > 12 else '')
    print('  cores with program memory:', sorted(pm))
    for (c, r), words in sorted(pm.items()):
        lo, hi = min(words), max(words)
        print(f'    core({c},{r}): {len(words)*4} B  0x{lo:05x}-0x{hi:05x}')
        if dump_dir:
            buf = bytearray(hi - lo + 4)
            for o, val in words.items():
                struct.pack_into('<I', buf, o - lo, val)
            out = f'{dump_dir}/core_{c}_{r}.bin'
            open(out, 'wb').write(buf)
    return pm


if __name__ == '__main__':
    dd = sys.argv[2] if len(sys.argv) > 2 else None
    if dd:
        os.makedirs(dd, exist_ok=True)
    main(sys.argv[1], dd)
