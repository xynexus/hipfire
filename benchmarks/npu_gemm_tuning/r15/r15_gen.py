#!/usr/bin/env python3
"""4x4 AIE2P W4 schedule retaining scaled C across all K groups."""
import sys

MODE = sys.argv[1]
GROUPS, OUTBLOCKS = map(int, sys.argv[2:4])
COLS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
if COLS not in (4, 8):
    raise SystemExit("COLS must be 4 or 8")
if MODE == "w4":
    AB, WB, CB, CJ = 8192, 16384, 2304, 9216
elif MODE == "w8":
    AB, WB, CB, CJ = 8192, 16384, 1152, 4608
else:
    raise SystemExit("MODE must be w4 or w8")
INBLOCKS, INF, GC, GR = GROUPS * OUTBLOCKS, 9223372036854775807, range(COLS), range(4)

def dims(count, block):
    # Power-of-two padded payloads lower to legal 4-D DMA descriptors.
    return f"[<size = {count}, stride = {block}>, <size = {block // 512}, stride = 512>, <size = 512, stride = 1>]"

out = ["module {", "  aie.device(npu2) {"]
for col in GC:
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in GR: out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")
for col in GC:
    cores = ", ".join(f"%c{col}_{row}" for row in GR)
    out += [f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
            f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
            f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])"]
for row in GR:
    cores = ", ".join(f"%c{col}_{row}" for col in GC)
    out += [f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
            f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
            f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])"]
for col in GC:
    inputs = ", ".join(f"@cc{col}_{row}" for row in GR)
    offsets = ", ".join(str(row * CB) for row in GR)
    for row in GR: out.append(f"    aie.objectfifo @cc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{CB}xi32>>")
    out += [f"    aie.objectfifo @csh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CJ}xi32>>",
            f"    aie.objectfifo.link [{inputs}] -> [@csh{col}] ([{offsets}] [])"]
for name in (f"r15_{MODE}_scaled_init", f"r15_{MODE}_scaled_accum"):
    out.append(f'    func.func private @{name}(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r15.o"}}')
for col in GC:
  for row in GR:
    out += [f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index", f"      %m = arith.constant {INF} : index",
            f"      %groups = arith.constant {GROUPS} : index", "      %o = arith.constant 1 : index",
            "      scf.for %outer = %z to %m step %o {",
            f"        %c = aie.objectfifo.acquire @cc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>",
            f"        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>",
            f"        %a0 = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
            f"        %av0 = aie.objectfifo.subview.access %a0[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
            f"        %w0 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"        %wv0 = aie.objectfifo.subview.access %w0[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            f"        func.call @r15_{MODE}_scaled_init(%av0, %wv0, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
            f"        aie.objectfifo.release @abc{row}(Consume, 1)", f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
            "        scf.for %group = %o to %groups step %o {",
            f"          %a = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
            f"          %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
            f"          %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            f"          func.call @r15_{MODE}_scaled_accum(%av, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
            f"          aie.objectfifo.release @abc{row}(Consume, 1)", f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
            "        }", f"        aie.objectfifo.release @cc{col}_{row}(Produce, 1)",
            "      }", "      aie.end", "    }"]
AT, WT, CT = INBLOCKS * AB, INBLOCKS * WB, OUTBLOCKS * CJ
out.append(f"    aie.runtime_sequence(%A: memref<{4*AT}xi8>, %W: memref<{COLS*WT}xi8>, %C: memref<{COLS*CT}xi32>) {{")
for row in GR:
    out += [f"      %ta{row} = aiex.dma_configure_task_for @ash{row} {{",
            f"        aie.dma_bd(%A : memref<{4*AT}xi8>, {row*AT}, {AT}, {dims(INBLOCKS, AB)}) {{burst_length = 0 : i32}}",
            "        aie.end", "      }", f"      aiex.dma_start_task(%ta{row})"]
for col in GC:
    out += [f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
            f"        aie.dma_bd(%W : memref<{COLS*WT}xi8>, {col*WT}, {WT}, {dims(INBLOCKS, WB)}) {{burst_length = 0 : i32}}",
            "        aie.end", "      }", f"      aiex.dma_start_task(%tw{col})",
            f"      %tc{col} = aiex.dma_configure_task_for @csh{col} {{",
            f"        aie.dma_bd(%C : memref<{COLS*CT}xi32>, {col*CT}, {CT}, {dims(OUTBLOCKS, CJ)}) {{burst_length = 0 : i32}}",
            "        aie.end", "      } {issue_token = true}", f"      aiex.dma_start_task(%tc{col})"]
for col in GC: out += [f"      aiex.dma_await_task(%tc{col})", f"      aiex.dma_free_task(%tc{col})"]
for row in GR: out.append(f"      aiex.dma_free_task(%ta{row})")
for col in GC: out.append(f"      aiex.dma_free_task(%tw{col})")
out += ["    }", "  }", "}"]
print("\n".join(out))
