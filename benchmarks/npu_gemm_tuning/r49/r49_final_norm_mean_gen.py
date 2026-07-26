#!/usr/bin/env python3
"""Single-core resident BF16x2 final RMSNorm and mean-pool graph."""

ROWS, PAD_M, HIDDEN = 256, 288, 768
ROW_BYTES = 2 * HIDDEN * 2
INPUT_BYTES = PAD_M * ROW_BYTES
ACTIVE_INPUT_BYTES = ROWS * ROW_BYTES
PARAM_BYTES = 4096
OUTPUT_BYTES = HIDDEN * 4
INF = 9223372036854775807

print(
    f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %core_tile = aie.tile(0, 2)
    aie.objectfifo @xsh(%shim, {{%mt}}, 2 : i32) : !aie.objectfifo<memref<{ROW_BYTES}xi8>>
    aie.objectfifo @xc(%mt, {{%core_tile}}, 2 : i32) : !aie.objectfifo<memref<{ROW_BYTES}xi8>>
    aie.objectfifo.link [@xsh] -> [@xc] ([] [])
    aie.objectfifo @psh(%shim, {{%mt}}, 1 : i32) : !aie.objectfifo<memref<{PARAM_BYTES}xi8>>
    aie.objectfifo @pc(%mt, {{%core_tile}}, 1 : i32) : !aie.objectfifo<memref<{PARAM_BYTES}xi8>>
    aie.objectfifo.link [@psh] -> [@pc] ([] [])
    aie.objectfifo @oc(%core_tile, {{%mt}}, 1 : i32) : !aie.objectfifo<memref<{HIDDEN}xf32>>
    aie.objectfifo @osh(%mt, {{%shim}}, 1 : i32) : !aie.objectfifo<memref<{HIDDEN}xf32>>
    aie.objectfifo.link [@oc] -> [@osh] ([] [])
    func.func private @r49_final_norm_mean_row(memref<{ROW_BYTES}xi8>, memref<{PARAM_BYTES}xi8>, memref<{HIDDEN}xf32>, i32) attributes {{link_with = "r49.o"}}
    %core = aie.core(%core_tile) {{
      %z = arith.constant 0 : index
      %inf = arith.constant {INF} : index
      %one = arith.constant 1 : index
      %rows = arith.constant {ROWS} : index
      scf.for %outer = %z to %inf step %one {{
        %p = aie.objectfifo.acquire @pc(Consume, 1) : !aie.objectfifosubview<memref<{PARAM_BYTES}xi8>>
        %pv = aie.objectfifo.subview.access %p[0] : !aie.objectfifosubview<memref<{PARAM_BYTES}xi8>> -> memref<{PARAM_BYTES}xi8>
        %o = aie.objectfifo.acquire @oc(Produce, 1) : !aie.objectfifosubview<memref<{HIDDEN}xf32>>
        %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{HIDDEN}xf32>> -> memref<{HIDDEN}xf32>
        scf.for %row = %z to %rows step %one {{
          %x = aie.objectfifo.acquire @xc(Consume, 1) : !aie.objectfifosubview<memref<{ROW_BYTES}xi8>>
          %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{ROW_BYTES}xi8>> -> memref<{ROW_BYTES}xi8>
          %rowi = arith.index_cast %row : index to i32
          func.call @r49_final_norm_mean_row(%xv, %pv, %ov, %rowi) : (memref<{ROW_BYTES}xi8>, memref<{PARAM_BYTES}xi8>, memref<{HIDDEN}xf32>, i32) -> ()
          aie.objectfifo.release @xc(Consume, 1)
        }}
        aie.objectfifo.release @oc(Produce, 1)
        aie.objectfifo.release @pc(Consume, 1)
      }}
      aie.end
    }} {{stack_size = 1024 : i32}}
    aie.runtime_sequence(%X: memref<{INPUT_BYTES}xi8>, %P: memref<{PARAM_BYTES}xi8>, %O: memref<{HIDDEN}xf32>) {{
      %to = aiex.dma_configure_task_for @osh {{
        aie.dma_bd(%O : memref<{HIDDEN}xf32>, 0, {HIDDEN}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      %tp = aiex.dma_configure_task_for @psh {{
        aie.dma_bd(%P : memref<{PARAM_BYTES}xi8>, 0, {PARAM_BYTES}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tp)
      %tx = aiex.dma_configure_task_for @xsh {{
        aie.dma_bd(%X : memref<{INPUT_BYTES}xi8>, 0, {ACTIVE_INPUT_BYTES}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tx)
      aiex.dma_await_task(%tx)
      aiex.dma_free_task(%tx)
      aiex.dma_await_task(%tp)
      aiex.dma_free_task(%tp)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%to)
    }}
  }}
}}"""
)
