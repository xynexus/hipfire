// R5: one core of a K-cascade W4A8 column. The ROWS cores in a column split the K
// contraction; each computes its K-slice partial and the 512-bit cascade stream
// carries the running accumulator core->core (put_mcd / get_scd), so C is
// accumulated in-flight and stored ONCE by the tail core — eliminating the per-tile
// C load/store that pins the memtile dataflow (and SOTA FastFlowLM) to ~5 TOPS.
//
// Cascade API (from aie_kernels/aie2/cascade_mm.cc): the 512-bit cascade moves one
// v16 per beat via put_mcd / get_scd. The mmul accumulator is size_C acc32 =
// size_C/16 beats. The graph wires the physical link with aie.cascade_flow(src,dst).
// Built -DROLE={0 head,1 mid,2 tail}.
//
// ARCH: the aie2p (Strix Halo/npu2) int8xint4 mmul is <4,16,16> (size_C=64) and the
// cascade read is the get_scd_v16acc32 builtin, keeping the sum in the acc32 domain.
// XDNA1/aie2 (Phoenix/npu1) provides int8xint4 only as <4,16,8> (size_C=32) and has
// NO get_scd_v16acc32 — its cascade read is get_scd_v16int32, so we sum in the int32
// domain (partials are exact small ints; acc32->int32 at shift 0 is lossless). Select
// XDNA1 with -DNPU_AIE2 -DMMUL_N=8; aie2p is the default.
#include <aie_api/aie.hpp>

#ifndef KSLICE
#define KSLICE 16          // 16x16 mmul steps this core contracts (its K-slice)
#endif
#ifndef ROLE
#define ROLE 2             // 0=head (put only), 1=middle (get+put), 2=tail (get, store C)
#endif
#ifndef MMUL_N
#define MMUL_N 16          // aie2p int8xint4 = <4,16,16>; XDNA1/aie2 = <4,16,8> (-DMMUL_N=8)
#endif

using MMUL = aie::mmul<4, 16, MMUL_N, int8, int4>;
using ACC = aie::accum<acc32, MMUL::size_C>;   // 4*MMUL_N acc32 partial C
static constexpr int CN = MMUL::size_C;
static constexpr int BEATS = CN / 16;          // cascade beats (16 acc32/int32 each)

// This core's K-slice partial, in a register accumulator (II=1 recipe). Arch-agnostic.
static inline ACC kslice_partial(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  MMUL c;
  const int4 *w = reinterpret_cast<const int4 *>(wbytes);
  c.mul(aie::load_v<MMUL::size_A>(pA), aie::load_v<MMUL::size_B>(w));
  for (int j = 1; j < KSLICE; j++)
      chess_prepare_for_pipelining {
    aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(pA + j * MMUL::size_A);
    // Weight stride in BYTES on the int8 buffer, then reinterpret (int4* arithmetic
    // is byte-addressed — the R3a fix); size_B/2 bytes per 16xN tile.
    const int4 *bj = reinterpret_cast<const int4 *>(wbytes + j * (MMUL::size_B / 2));
    c.mac(a, aie::load_v<MMUL::size_B>(bj));
  }
  return c.to_accum();
}

#if defined(NPU_AIE2)
// XDNA1/aie2: cascade carries v16int32; sum in the int32 domain.
using CVEC = aie::vector<int32, CN>;
static inline CVEC to_cvec(ACC a) { return a.template to_vector<int32>(); }
static inline CVEC csum(CVEC a, CVEC b) { return aie::add(a, b); }
static inline void store_c(int32 *__restrict pC, CVEC v) { aie::store_v(pC, v); }
static inline void cascade_put(CVEC v) {
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    put_mcd((v16int32)v.template extract<16>(i));
}
static inline CVEC cascade_get() {
  CVEC v;
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    v.insert(i, aie::vector<int32, 16>(get_scd_v16int32()));
  return v;
}
#else
// aie2p: keep the accumulator in the acc32 domain end-to-end.
using CVEC = ACC;
static inline CVEC to_cvec(ACC a) { return a; }
static inline CVEC csum(CVEC a, CVEC b) { return add(a, b); }
static inline void store_c(int32 *__restrict pC, CVEC a) { aie::store_v(pC, a.template to_vector<int32>()); }
static inline void cascade_put(CVEC acc) {
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    put_mcd(acc.template extract<16>(i).to_native());
}
static inline CVEC cascade_get() {
  ACC acc;
#pragma unroll
  for (int i = 0; i < BEATS; i++)
    acc.insert(i, aie::accum<acc32, 16>(get_scd_v16acc32()));
  return acc;
}
#endif

#if ROLE == 0   // HEAD: seed the cascade with this slice's partial.
extern "C" void r5_cascade_head(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  cascade_put(to_cvec(kslice_partial(pA, wbytes)));
}
#elif ROLE == 1 // MIDDLE: add cascade-in + this slice, pass on.
extern "C" void r5_cascade_mid(const int8 *__restrict pA, const int8 *__restrict wbytes) {
  cascade_put(csum(cascade_get(), to_cvec(kslice_partial(pA, wbytes))));
}
#else           // TAIL: add cascade-in + this slice, STORE C once.
extern "C" void r5_cascade_tail(const int8 *__restrict pA, const int8 *__restrict wbytes,
                                int32 *__restrict pC) {
  store_c(pC, csum(cascade_get(), to_cvec(kslice_partial(pA, wbytes))));
}
#endif
