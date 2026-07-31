#!/usr/bin/env python3
"""Static issue-rate microbenchmark for AIE2P MAC modes.

Measures how many cycles the machine needs per MAC *instruction*, by compiling a
loop of independent MAC chains and counting scheduled VLIW bundles. The llvm-aie
scheduler models AIE2P's reservation tables, so bundles/MAC is the issue interval.

Register pressure is the trap: v256int4 is 1024 bits and v64acc32 is 2048, so
unrolling past a couple of chains spills to stack and the "measurement" becomes
stack traffic. So: operands are SHARED across chains (only the accumulators
differ), and any loop body containing a stack reference is rejected outright.

Two independent accumulator chains are enough to cover the MAC latency; a body
that still shows 1 MAC/bundle at 2 chains is issue-limited, not latency-limited.
"""
import re, subprocess, tempfile

BIN = '/home/sadara/.venv/lib/python3.14/site-packages/llvm-aie/bin'
INC = '/home/sadara/.venv/lib/python3.14/site-packages/llvm-aie/lib/clang/21/include'

# label -> (a_type, b_type, acc_type, intrinsic, MACs per instruction)
MODES = [
    ('W4A8  mac_4x16_16x16 (int4)',  'v64int8',      'v256int4',     'v64acc32',    'mac_4x16_16x16', 4*16*16),
    ('W4A8  mac_4x16_16x16 (uint4)', 'v64int8',      'v256uint4',    'v64acc32',    'mac_4x16_16x16', 4*16*16),
    ('W8A8  mac_8x8_8x8',            'v64int8',      'v64int8',      'v64acc32',    'mac_8x8_8x8',    8*8*8),
    ('BFP16 mac_8x8_8x8T',           'v64bfp16ebs8', 'v64bfp16ebs8', 'v64accfloat', 'mac_8x8_8x8T',   8*8*8),
    ('bf16  mac_elem_64',            'v64bfloat16',  'v64bfloat16',  'v64accfloat', 'mac_elem_64',    64),
    ('bf16  mac_elem_32',            'v32bfloat16',  'v32bfloat16',  'v32accfloat', 'mac_elem_32',    32),
    ('bf16  mac_elem_16  (FLM)',     'v32bfloat16',  'v32bfloat16',  'v16accfloat', 'mac_elem_16',    16),
]

SRC = """
#include <aie2pintrin.h>
void bench({A} *ap, {B} *bp, {ACC} *op, int n) {{
  {A} a0 = ap[0];
  {B} b0 = bp[0];
{decl}
  for (int i = 0; i < n; ++i) {{
{body}
  }}
{store}
}}
"""


def compile_mode(A, B, ACC, fn, chains):
    src = SRC.format(A=A, B=B, ACC=ACC,
                     decl='\n'.join(f'  {ACC} c{j} = op[{j}];' for j in range(chains)),
                     body='\n'.join(f'    c{j} = {fn}(a0, b0, c{j});' for j in range(chains)),
                     store='\n'.join(f'  op[{j}] = c{j};' for j in range(chains)))
    with tempfile.TemporaryDirectory() as t:
        open(f'{t}/b.cpp', 'w').write(src)
        r = subprocess.run([f'{BIN}/clang++', '--target=aie2p-none-unknown-elf', '-O2',
                            '-I', INC, '-c', f'{t}/b.cpp', '-o', f'{t}/b.o'],
                           capture_output=True, text=True)
        if r.returncode:
            return None, r.stderr.strip().splitlines()[0][:90]
        d = subprocess.run([f'{BIN}/llvm-objdump', '-d', '--no-show-raw-insn', f'{t}/b.o'],
                           capture_output=True, text=True).stdout
    return d, None


def loop_body(disasm):
    """Body = the bundles from the loop-head label through the .L_LEnd bundle."""
    lines = disasm.splitlines()
    end_i = next((i for i, l in enumerate(lines) if '<.L_LEnd' in l), None)
    if end_i is None:
        return None
    start_i = max((i for i, l in enumerate(lines[:end_i]) if re.search(r'<\.LBB', l)), default=None)
    if start_i is None:
        return None
    body = []
    for l in lines[start_i:]:
        m = re.match(r'\s*([0-9a-f]+):\s*(.*)', l)
        if m:
            body.append(m.group(2).strip())
            if len(body) > 1 and any('<.L_LEnd' in x for x in lines[start_i:lines.index(l)+1]) \
               and l is not None and lines.index(l) > end_i:
                break
    # trim to the bundle that follows the .L_LEnd label
    body = []
    seen_end = False
    for i, l in enumerate(lines[start_i:], start_i):
        if '<.L_LEnd' in l:
            seen_end = True
            continue
        m = re.match(r'\s*([0-9a-f]+):\s*(.*)', l)
        if not m:
            continue
        body.append(m.group(2).strip())
        if seen_end:
            break
    return body


if __name__ == '__main__':
    print(f'{"mode":30s} {"chains":>6s} {"bundles":>7s} {"macs":>5s} {"cyc/MACinsn":>11s} {"MAC/cycle":>9s}')
    for label, A, B, ACC, fn, nmac in MODES:
        best = None
        for chains in (2, 3, 4):
            d, err = compile_mode(A, B, ACC, fn, chains)
            if d is None:
                if chains == 2:
                    print(f'{label:30s} {"":>6s} COMPILE FAIL: {err}')
                break
            body = loop_body(d)
            if not body:
                continue
            if any('[sp' in b for b in body):      # spilled -> not a MAC measurement
                continue
            vmacs = sum(1 for b in body if re.search(r'\bvmac', b))
            if not vmacs:
                continue
            # CORRECTED 2026-07-31. This used to divide bundles by the *vmac
            # instruction* count and then multiply by MACs per *intrinsic call*,
            # mixing two different units. It is only safe when one intrinsic
            # lowers to one vmac, which is not true in general:
            # mac_4x16_16x16 emits TWO vmacs per call (the v256int4 operand is
            # unpacked into a y1/y2 pair and each half is MAC'd), so the loop is
            # 4 bundles / 4 vmacs / 2 calls. The old form read that as
            # 1.0 cyc per MAC and reported 1024 MACs/cycle; the truth is 2.0 cyc
            # per call and 512 MACs/cycle. Hardware agrees (macbench_hw.py
            # measures 2.073 cyc/call) -- see docs/npu/flm-refe-log.md.
            #
            # The loop body performs exactly `chains` intrinsic calls, so that
            # is the correct divisor.
            cyc = len(body) / chains
            row = (chains, len(body), vmacs, cyc, nmac / cyc)
            if best is None or row[4] > best[4]:
                best = row
        if best:
            ch, nb, nm, cyc, rate = best
            print(f'{label:30s} {ch:6d} {nb:7d} {nm:5d} {cyc:11.2f} {rate:9.0f}')
        else:
            print(f'{label:30s} {"":>6s} no spill-free loop found')
