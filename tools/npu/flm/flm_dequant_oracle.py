"""Call FastFlowLM's OWN q4nx dequantizer as a ground-truth oracle.

`libq4_npu_eXpress.so` exports `Q4NX::q4nx_dequantize<float>` and the whole
`bytes` accessor set, so FLM's decoder runs directly from ctypes -- no patching,
no source. That matters because §1.3 is otherwise unanswerable: the container's
values match no layout of the checkpoint, so there is nothing to check a
candidate decode against except FLM's own.

**The signature is (out, in, n), not (in, out, n).** Read off the disassembly:

    nrows = in.size() / 5120                 # magic-number divide, s=12
    need  = nrows * 8192 * 4                 # f32 out, 8192 elements per row
    if out.size() == 0: out.resize(need)     # but bytes(0) THROWS, so pre-size
    else if out.size() != need: "Weight size mismatch: <out.size()> != <need>"

That formula reproduces all five observed mismatches exactly. `n` is consumed as
`n / 256`, so pass **n = 256 * nrows**. `bytes(0)` raises
`std::runtime_error: Invalid size for bytes allocation`, so always pre-size the
output; the resize branch is unreachable from ctypes.

WHAT IT RETURNS is NOT yet identified. Two things are solid:

  - the output is **100% non-negative** (neg fraction 0.0000 over all 4.19M
    elements of layers.0 q_proj), so it is not finished weights;
  - it is deterministic and f32-computed (only 0.09% of values are bf16-exact).

`d*code` was the closest of six candidate formulas by sorted-multiset distance
(mean|diff| 0.00084 vs 0.041-0.078 for d*code+-m, |d*code+m|, d*(code+-8)) and
was briefly recorded here as the answer. **That is retracted.** Per-element tests
refute it:

  - the output contains **zero zeros** in 8192 values, impossible for d*code
    when codes span 0..15;
  - for only 20% of outputs does ANY of the 256 block scales divide the value
    to an integer 0..15 -- and with 256 candidates and a 1e-3 tolerance that is
    about what chance alone gives.

So the multiset agreement reflects similar distribution SHAPE, not matching
values. Whatever the function computes, it is not d*code under any block
pairing. Deriving the map by division (recover.py in scratch) yields a unique
block for only 788/8192 outputs, which is consistent with no true pairing
existing rather than with a reorder waiting to be found.

Next: the explicit-buffer overload takes codes and scales as SEPARATE arguments
    q4nx_dequantize<float>(buffer<float>&, buffer<unsigned int>&,
                           buffer<bfloat16_t>&, buffer<int>&, int)
so it sidesteps the packing question entirely -- feed known codes and known
scales, see what comes back. buffer<T> exports ctor/data/size/resize for uint,
int and bfloat16, so it is constructible the same way `bytes` was.
"""

import ctypes as C
import sys

LIB = "/opt/fastflowlm/lib/libq4_npu_eXpress.so"
lib = C.CDLL(LIB, mode=C.RTLD_GLOBAL)

# bytes:: accessors
_ctor = lib["_ZN5bytesC1Em"]; _ctor.restype = None
_ctor.argtypes = [C.c_void_p, C.c_size_t]
_data = lib["_ZNK5bytes4dataEv"]; _data.restype = C.c_void_p
_data.argtypes = [C.c_void_p]
_size = lib["_ZNK5bytes4sizeEv"]; _size.restype = C.c_size_t
_size.argtypes = [C.c_void_p]
_dtor = lib["_ZN5bytesD1Ev"]; _dtor.restype = None; _dtor.argtypes = [C.c_void_p]

_deq_f = lib["_ZN4Q4NX15q4nx_dequantizeIfEEvR5bytesS2_i"]
_deq_f.restype = None
_deq_f.argtypes = [C.c_void_p, C.c_void_p, C.c_int]


class Bytes:
    """A `bytes` instance built in an over-allocated slab (sizeof is unknown)."""
    SLAB = 512

    def __init__(self, n):
        self._slab = C.create_string_buffer(self.SLAB)
        self.ref = C.cast(self._slab, C.c_void_p)
        _ctor(self.ref, n)

    def ptr(self):
        return _data(self.ref)

    def size(self):
        return _size(self.ref)

    def write(self, raw):
        C.memmove(self.ptr(), raw, len(raw))

    def read(self, n):
        return C.string_at(self.ptr(), n)

    def __del__(self):
        try:
            _dtor(self.ref)
        except Exception:
            pass


def dequantize_f32(raw, n):
    """raw: packed q4nx bytes. n: the library's count argument. -> bytes of f32."""
    src = Bytes(len(raw))
    src.write(raw)
    dst = Bytes(max(1, n) * 4)
    _deq_f(src.ref, dst.ref, n)
    return dst.read(dst.size()), dst.size()


if __name__ == "__main__":
    b = Bytes(64)
    print(f"bytes(64): size()={b.size()} data()={hex(b.ptr() or 0)}", file=sys.stderr)
    assert b.size() == 64, "bytes ctor/size disagree — slab layout wrong"
    print("bytes accessors work", file=sys.stderr)
