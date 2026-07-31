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

WHAT IT RETURNS -- and this is the finding: the output is **100% non-negative**
(neg fraction 0.0000 over all 4.19M elements of layers.0 q_proj), so it is not
finished weights. Against ground truth, of six candidate formulas:

    d*code        mean|diff| 0.00084   <-- 50-90x better than any other
    d*code + m    mean|diff| 0.0716
    d*code - m    mean|diff| 0.0700
    |d*code + m|  mean|diff| 0.0411
    d*(code-8)    mean|diff| 0.0776
    |d*(code-8)|  mean|diff| 0.0428

**FLM's dequant applies the scale but NOT the q4_1 min.** That is consistent
with factoring the min out of the per-element path -- sum((d*c+m)*x) is
d*sum(c*x) + m*sum(x), so `m` need only meet a per-block activation sum, never
each weight. Mathematically identical to what our kernel does per element; the
difference is where the work happens, not the result.

STILL OPEN: no nibble order matches elementwise (all four orders sit at
mean|diff| ~5.2e-02 with the natural block-to-scale pairing), so the container
is REORDERED -- which is what `Q4NX::_q4nx_reorder` and `Dequant::reorder_cpy(u8*,
buffer<u8>&, quant_block_t, int,int,int,int)` exist to do. The multiset matches
but the pairing does not. That is now a well-posed search WITH an oracle, rather
than the unanswerable question §1.3 was.
"""Call FastFlowLM's OWN q4nx dequantizer as a ground-truth oracle.

`libq4_npu_eXpress.so` exports `Q4NX::q4nx_dequantize<float>(bytes&, bytes&, int)`
along with the whole `bytes` accessor set (ctor/data/size/resize/dtor), so the
decoder can be driven directly from ctypes. That is worth having because §1.3 is
otherwise unanswerable: the container's values do not match the checkpoint under
ANY layout (Frobenius differs by 1.16x, and permutation preserves Frobenius), so
the q4_1 interpretation is wrong and there is nothing to check a candidate
against except FLM's own decoder.

**Status: the bridge works, the calling convention does not yet.** `bytes` is
driven correctly — `Bytes(64).size() == 64` and `data()` returns a live pointer.
But `q4nx_dequantize` rejects a bare container row with

    Weight size mismatch: 5120 != 196608

and the expected size is a function of the DESTINATION size alone (`n` is
ignored — dst=32768 expects 196608 whether n is 1, 256, or 8192). The expected
values are not whole multiples of the 5120-byte row:

    dst 5120 -> 32768   dst 32768 -> 196608   dst 131072 -> 819200
        ratios 6.400            6.000                6.250

so this entry point does not take a raw row. It almost certainly wants the blob
`SafeTensors::load_weights(bytes&, string)` produces, with tensor metadata
attached — that is the next thing to try, along with the explicit-buffer
overload `q4nx_dequantize<float>(buffer<float>&, buffer<unsigned int>&,
buffer<bfloat16>&, buffer<int>&, int)`, whose separate code/scale arguments
would sidestep the packing question entirely.

Do not read the ratio column as a formula; five points fit several and none
survives all five. It is recorded as data, not as a rule.
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
