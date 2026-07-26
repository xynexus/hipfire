---
title: "Load and Store Using Pointers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Load-and-Store-Using-Pointers"
toc_id: e5UUF~0QfEuXcDlmN8UYoA
content_id: qxsGZUSq~hz4NuxFFxLPxw
---

#### Load and Store Using Pointers

Applications can load from DM into vector registers and store the contents of vector registers into DM. Memory instructions in the AI Engine that operate on vectors have alignment requirements. Functions are provided for both aligned and unaligned accesses:

- **`aie::load_v`:** Load a vector of Elems size whose elements have type T (for example, `aie::load_v<Elems>(T*)`. The pointer is assumed to meet the alignment requirements for a vector load of this size.
- **`aie::store_v`:** Store a vector of Elems size whose elements have type T (for example, `aie::store_v<Elems>(T*)`. The pointer is assumed to meet the alignment requirements for a vector store of this size.
- **`aie::load_unaligned_v`:** Load a vector of Elems size whose elements have type T. The pointer is assumed to be aligned to T.
- **`aie::store_unaligned_v`:** Store a vector of Elems size whose elements have type T. The pointer is assumed to be aligned to T.

```
alignas(aie::vector_decl_align) int16 delay_value[N]={...};
aie::vector<int16,8> va=aie::load_v<8>(delay_value);
aie::store_v(delay_value,va);
aie::vector<int16,8> vv=aie::load_unaligned_v<8>((int16*)scatter_value);
aie::store_unaligned_v((int16*)scatter_value,vv);
```

The compiler supports standard pointer de-referencing and pointer arithmetic for vectors. For using vector iterators to access memory, see Iterators.

It is mandatory to use the buffer port in the kernel function prototype as inputs and outputs. However, in the kernel code, it is possible to use a direct pointer to load/store data.

```
void func(input_buffer<int16> &w_input, output_buffer<cint16> &w_output){
  ......
  aie::vector<int16,16> datain=aie::load_v<16>((int16*)w_input.data());
  aie::vector<cint16,8> dataout=datain.cast_to<cint16>();
  aie::store_v((cint16*)w_output.data(),dataout);
  ......
}
```

The buffer structure is responsible for managing buffer locks tracking buffer type (ping/pong) and this can add to the cycle count. This is especially true when load/store are out-of-order (scatter-gather). Using pointers can help reduce the cycle count required for load and store.

**Note:** If using pointers to load and store data, it is your responsibility to avoid out-of-bound memory access.
