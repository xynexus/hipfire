#!/usr/bin/env python3
"""Disassemble a raw AIE2p program-memory image.

llvm-objdump has no -b binary here, so bytes are wrapped via .incbin into a real
aie2p object. A linear sweep halts at the first undecodable word (constant pools
and data live in program memory too), so restart past each failure and stitch.
"""
import os, re, subprocess, sys, tempfile

BIN = '/home/sadara/.venv/lib/python3.14/site-packages/llvm-aie/bin'
LINE = re.compile(r'^\s*([0-9a-f]+):\s*(.*)$')


def sweep(data, tmp):
    """Disassemble `data` from offset 0; return (lines, bytes_consumed)."""
    src, obj = f'{tmp}/w.s', f'{tmp}/w.o'
    open(f'{tmp}/w.bin', 'wb').write(data)
    open(src, 'w').write(f'.text\n.globl _start\n_start:\n.incbin "{tmp}/w.bin"\n')
    subprocess.run([f'{BIN}/clang', '--target=aie2p', '-x', 'assembler', '-c', src,
                    '-o', obj], check=True, capture_output=True)
    # the aie2p disassembler is an assertions build and aborts on some byte
    # patterns; keep whatever it printed before dying and step past the fault.
    r = subprocess.run([f'{BIN}/llvm-objdump', '-d', '--no-show-raw-insn', obj],
                       capture_output=True, text=True)
    out = r.stdout
    lines, end = [], 0
    for ln in out.splitlines():
        m = LINE.match(ln)
        if not m:
            continue
        addr, text = int(m.group(1), 16), m.group(2).strip()
        if text.startswith('<unknown>'):
            return lines, addr + 2          # resume past the bad halfword
        lines.append((addr, text))
        end = addr
    if r.returncode != 0:                   # crashed: keep output, step past last decode
        return lines, (end + 2) if lines else 2
    return lines, len(data)


def disassemble(path, max_restarts=4000):
    data = open(path, 'rb').read()
    out, pos, n = [], 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        while pos < len(data) and n < max_restarts:
            lines, used = sweep(data[pos:], tmp)
            out += [(pos + a, t) for a, t in lines]
            if used == 0:
                used = 2
            pos += used
            n += 1
    return out, n


if __name__ == '__main__':
    insns, restarts = disassemble(sys.argv[1])
    for a, t in insns:
        print(f'{a:8x}:\t{t}')
    print(f'# {len(insns)} instructions, {restarts} sweeps', file=sys.stderr)
