#!/usr/bin/env python3
"""Generate a one-core AIE2P int32-to-float conversion probe."""

import sys

OBJECT = sys.argv[1]
INF = 9223372036854775807

print(f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %core_tile = aie.tile(0, 2)
    aie.objectfifo @input(%shim, {{%core_tile}}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scales(%shim, {{%core_tile}}, 1 : i32) : !aie.objectfifo<memref<16xf32>>
    aie.objectfifo @f32(%core_tile, {{%shim}}, 1 : i32) : !aie.objectfifo<memref<16xf32>>
    aie.objectfifo @scaled(%core_tile, {{%shim}}, 1 : i32) : !aie.objectfifo<memref<16xf32>>
    func.func private @r6_fix2float_probe(memref<16xi32>, memref<16xf32>, memref<16xf32>, memref<16xf32>) attributes {{link_with = \"{OBJECT}\"}}
    %core = aie.core(%core_tile) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {{
        %is = aie.objectfifo.acquire @input(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %iv = aie.objectfifo.subview.access %is[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %ss = aie.objectfifo.acquire @scales(Consume, 1) : !aie.objectfifosubview<memref<16xf32>>
        %sv = aie.objectfifo.subview.access %ss[0] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
        %fs = aie.objectfifo.acquire @f32(Produce, 1) : !aie.objectfifosubview<memref<16xf32>>
        %fv = aie.objectfifo.subview.access %fs[0] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
        %os = aie.objectfifo.acquire @scaled(Produce, 1) : !aie.objectfifosubview<memref<16xf32>>
        %ov = aie.objectfifo.subview.access %os[0] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
        func.call @r6_fix2float_probe(%iv, %sv, %fv, %ov) : (memref<16xi32>, memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
        aie.objectfifo.release @input(Consume, 1)
        aie.objectfifo.release @scales(Consume, 1)
        aie.objectfifo.release @f32(Produce, 1)
        aie.objectfifo.release @scaled(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%I: memref<16xi32>, %S: memref<16xf32>, %F: memref<16xf32>, %O: memref<16xf32>) {{
      %ti = aiex.dma_configure_task_for @input {{
        aie.dma_bd(%I : memref<16xi32>, 0, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ti)
      %ts = aiex.dma_configure_task_for @scales {{
        aie.dma_bd(%S : memref<16xf32>, 0, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ts)
      %tf = aiex.dma_configure_task_for @f32 {{
        aie.dma_bd(%F : memref<16xf32>, 0, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tf)
      %to = aiex.dma_configure_task_for @scaled {{
        aie.dma_bd(%O : memref<16xf32>, 0, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%tf)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ti)
      aiex.dma_free_task(%ts)
    }}
  }}
}}""")
