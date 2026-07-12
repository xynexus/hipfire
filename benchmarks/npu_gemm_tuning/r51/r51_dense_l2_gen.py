#!/usr/bin/env python3
"""Single-core resident EmbeddingGemma Dense heads and L2 normalization."""

INPUT, INTERMEDIATE, OUTPUT = 768, 3072, 768
INPUT_BYTES = INPUT * 4
W0_ROW_BYTES = INPUT * 2
W0_WEIGHT_BYTES = INTERMEDIATE * W0_ROW_BYTES
W0_BYTES = INPUT_BYTES + W0_WEIGHT_BYTES
W1_ROW_BYTES = INTERMEDIATE * 2
W1_BYTES = OUTPUT * W1_ROW_BYTES
OUTPUT_BYTES = OUTPUT * 4
INF = 9223372036854775807

print(
    f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %shim1 = aie.tile(1, 0)
    %mt1 = aie.tile(1, 1)
    %core_tile = aie.tile(0, 2)
    %input = aie.buffer(%core_tile) {{sym_name = "input"}} : memref<{INPUT}xf32>
    %intermediate = aie.buffer(%core_tile) {{sym_name = "intermediate"}} : memref<{INTERMEDIATE}xf32>
    aie.objectfifo @w0sh(%shim, {{%mt}}, 2 : i32) : !aie.objectfifo<memref<{W0_ROW_BYTES}xi8>>
    aie.objectfifo @w0c(%mt, {{%core_tile}}, 2 : i32) : !aie.objectfifo<memref<{W0_ROW_BYTES}xi8>>
    aie.objectfifo.link [@w0sh] -> [@w0c] ([] [])
    aie.objectfifo @w1sh(%shim1, {{%mt1}}, 2 : i32) : !aie.objectfifo<memref<{W1_ROW_BYTES}xi8>>
    aie.objectfifo @w1c(%mt1, {{%core_tile}}, 2 : i32) : !aie.objectfifo<memref<{W1_ROW_BYTES}xi8>>
    aie.objectfifo.link [@w1sh] -> [@w1c] ([] [])
    aie.objectfifo @oc(%core_tile, {{%mt}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT}xf32>>
    aie.objectfifo @osh(%mt, {{%shim}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT}xf32>>
    aie.objectfifo.link [@oc] -> [@osh] ([] [])
    func.func private @r51_copy_input(memref<{W0_ROW_BYTES}xi8>, memref<{INPUT}xf32>, i32) attributes {{link_with = "r51.o"}}
    func.func private @r51_dense0_row(memref<{INPUT}xf32>, memref<{W0_ROW_BYTES}xi8>, memref<{INTERMEDIATE}xf32>, i32) attributes {{link_with = "r51.o"}}
    func.func private @r51_dense1_row(memref<{INTERMEDIATE}xf32>, memref<{W1_ROW_BYTES}xi8>, memref<{OUTPUT}xf32>, i32) attributes {{link_with = "r51.o"}}
    func.func private @r51_l2_normalize(memref<{OUTPUT}xf32>) attributes {{link_with = "r51.o"}}
    %core = aie.core(%core_tile) {{
      %z = arith.constant 0 : index
      %inf = arith.constant {INF} : index
      %one = arith.constant 1 : index
      %halves = arith.constant 2 : index
      %rows0 = arith.constant {INTERMEDIATE} : index
      %rows1 = arith.constant {OUTPUT} : index
      scf.for %outer = %z to %inf step %one {{
        %o = aie.objectfifo.acquire @oc(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT}xf32>>
        %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{OUTPUT}xf32>> -> memref<{OUTPUT}xf32>
        scf.for %half = %z to %halves step %one {{
          %w = aie.objectfifo.acquire @w0c(Consume, 1) : !aie.objectfifosubview<memref<{W0_ROW_BYTES}xi8>>
          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{W0_ROW_BYTES}xi8>> -> memref<{W0_ROW_BYTES}xi8>
          %halfi = arith.index_cast %half : index to i32
          func.call @r51_copy_input(%wv, %input, %halfi) : (memref<{W0_ROW_BYTES}xi8>, memref<{INPUT}xf32>, i32) -> ()
          aie.objectfifo.release @w0c(Consume, 1)
        }}
        scf.for %row = %z to %rows0 step %one {{
          %w = aie.objectfifo.acquire @w0c(Consume, 1) : !aie.objectfifosubview<memref<{W0_ROW_BYTES}xi8>>
          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{W0_ROW_BYTES}xi8>> -> memref<{W0_ROW_BYTES}xi8>
          %rowi = arith.index_cast %row : index to i32
          func.call @r51_dense0_row(%input, %wv, %intermediate, %rowi) : (memref<{INPUT}xf32>, memref<{W0_ROW_BYTES}xi8>, memref<{INTERMEDIATE}xf32>, i32) -> ()
          aie.objectfifo.release @w0c(Consume, 1)
        }}
        scf.for %row = %z to %rows1 step %one {{
          %w = aie.objectfifo.acquire @w1c(Consume, 1) : !aie.objectfifosubview<memref<{W1_ROW_BYTES}xi8>>
          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{W1_ROW_BYTES}xi8>> -> memref<{W1_ROW_BYTES}xi8>
          %rowi = arith.index_cast %row : index to i32
          func.call @r51_dense1_row(%intermediate, %wv, %ov, %rowi) : (memref<{INTERMEDIATE}xf32>, memref<{W1_ROW_BYTES}xi8>, memref<{OUTPUT}xf32>, i32) -> ()
          aie.objectfifo.release @w1c(Consume, 1)
        }}
        func.call @r51_l2_normalize(%ov) : (memref<{OUTPUT}xf32>) -> ()
        aie.objectfifo.release @oc(Produce, 1)
      }}
      aie.end
    }} {{stack_size = 1024 : i32}}
    aie.runtime_sequence(%W0: memref<{W0_BYTES}xi8>, %W1: memref<{W1_BYTES}xi8>, %O: memref<{OUTPUT}xf32>) {{
      %to = aiex.dma_configure_task_for @osh {{
        aie.dma_bd(%O : memref<{OUTPUT}xf32>, 0, {OUTPUT}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      %tw0 = aiex.dma_configure_task_for @w0sh {{
        aie.dma_bd(%W0 : memref<{W0_BYTES}xi8>, 0, {W0_BYTES}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tw0)
      %tw1 = aiex.dma_configure_task_for @w1sh {{
        aie.dma_bd(%W1 : memref<{W1_BYTES}xi8>, 0, {W1_BYTES}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tw1)
      aiex.dma_await_task(%tw0)
      aiex.dma_free_task(%tw0)
      aiex.dma_await_task(%tw1)
      aiex.dma_free_task(%tw1)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%to)
    }}
  }}
}}"""
)
