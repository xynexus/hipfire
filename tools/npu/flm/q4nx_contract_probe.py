#!/usr/bin/env python3
"""Establish FLM's q4 decode contract by CONTROLLED experiment, not inference.

The explicit-buffer overload takes codes and scales as separate arguments, so
the packing question is sidestepped entirely: feed known scales and known codes,
read what comes back.

    Q4NX::q4nx_dequantize<float>(buffer<float>& out, buffer<unsigned int>& codes,
                                 buffer<bfloat16_t>& scales, buffer<int>&, int n)

From the disassembly: `nelem = scales.size() * 32`, out is resized to that if
empty and must equal it otherwise (same "Weight size mismatch" path as the
bytes& overload, but exit(1) here rather than a return).

RESULT, with all scales 1.0 and codes a repeating 0..15 ramp:

    out[:16] == [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]   exactly, in order

So the contract is:

    out[i] = scale[i / 32] * code[i]

  - one bf16 scale per 32 elements;
  - **plain low-nibble-first order** -- byte j holds element 2j in its low
    nibble and 2j+1 in its high nibble. This is exactly what `q4nx.pack_tile`
    already emits, so our kernel's nibble order is confirmed correct against
    FLM's own decoder rather than assumed;
  - no min/offset applied by this function;
  - no reordering -- output index equals element index.

`buffer<T>` layout, recovered by constructing one and dumping the slab:

    +0 vptr   +8 data   +16 data   +24 byte-size   (size() = bytes / sizeof T)

buffer<float> exports no ctor, but buffer<unsigned int> has the same element
size, so one built with the uint ctor serves.

STILL OPEN: how a 5120-byte container row maps onto (scales, codes). Four
natural layouts ([d][m][codes] and permutations) all fail to reproduce the ramp
through the bytes& overload -- each returns a CONSTANT first 8 values, so the
codes are not being read where those layouts put them. The lead is the library
constant `(anonymous namespace)::group_size_bytes = 40960 = 8 * 5120`, i.e. the
reorder operates on groups of EIGHT rows, not one. `Dequant::reorder_cpy` takes
four trailing ints, consistent with a blocked transpose over such a group.

    python3 q4nx_contract_probe.py <nblocks> <n> <extra>   # e.g. 1 32 1
"""
import ctypes as C, numpy as np, sys
from ml_dtypes import bfloat16
lib = C.CDLL("/opt/fastflowlm/lib/libq4_npu_eXpress.so", mode=C.RTLD_GLOBAL)
def fn(n,r,a):
    f=lib[n]; f.restype=r; f.argtypes=a; return f
C_u=fn("_ZN6bufferIjEC1Em",None,[C.c_void_p,C.c_size_t]);  D_u=fn("_ZNK6bufferIjE4dataEv",C.c_void_p,[C.c_void_p])
C_i=fn("_ZN6bufferIiEC1Em",None,[C.c_void_p,C.c_size_t]);  D_i=fn("_ZNK6bufferIiE4dataEv",C.c_void_p,[C.c_void_p])
C_b=fn("_ZN6bufferIN8biovault10bfloat16_tEEC1Em",None,[C.c_void_p,C.c_size_t])
D_b=fn("_ZNK6bufferIN8biovault10bfloat16_tEE4dataEv",C.c_void_p,[C.c_void_p])
S_f=fn("_ZNK6bufferIfE4sizeEv",C.c_size_t,[C.c_void_p]); D_f=fn("_ZNK6bufferIfE4dataEv",C.c_void_p,[C.c_void_p])
DEQ=fn("_ZN4Q4NX15q4nx_dequantizeIfEEvR6bufferIT_ERS1_IjERS1_IN8biovault10bfloat16_tEERS1_IiEi",
       None,[C.c_void_p]*4+[C.c_int])
def mk(ctor,n):
    s=C.create_string_buffer(256); r=C.cast(s,C.c_void_p); ctor(r,n); return s,r
NB=int(sys.argv[1]); NE=NB*32                       # scales.size()*32 = output elems
sc_s,sc = mk(C_b,NB)                                # scales
cd_s,cd = mk(C_u,NE//8)                             # 8 nibbles per uint32
ex_s,ex = mk(C_i,int(sys.argv[3]))                  # 4th arg
of_s,of = mk(C_u,NE)                                # out: same elem size as float
# scales all 1.0 ; codes = repeating ramp 0..15
C.memmove(D_b(sc), np.ones(NB,dtype=bfloat16).tobytes(), NB*2)
ramp=np.arange(NE)%16
packed=(ramp[0::2] | (ramp[1::2]<<4)).astype(np.uint8)
C.memmove(D_u(cd), packed.tobytes(), packed.nbytes)
C.memset(D_i(ex), 0, 4*int(sys.argv[3]))
DEQ(of, cd, sc, ex, int(sys.argv[2]))
out=np.frombuffer(C.string_at(D_f(of), NE*4), dtype=np.float32)
print(f"NB={NB} NE={NE} n={sys.argv[2]}  out.size()={S_f(of)}")
print("  codes  :", ramp[:16])
print("  out[:16]:", np.array2string(out[:16],precision=4))
print("  out[16:32]:", np.array2string(out[16:32],precision=4))
