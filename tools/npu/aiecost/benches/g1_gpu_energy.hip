// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// G1: the GPU side of the NPU-vs-GPU split, measured the same way E1 measures
// the NPU so the two numbers are comparable.
//
// Why this is fair: the NPU and the GPU (gfx1103) are on the SAME die and share
// the same package rails, so RAPL package-0 sees both. Same instrument, same
// matched-rate discipline, same null subtraction. The only honest way to compare
// "should this kernel run on the NPU or the GPU" on tok/J.
//
// Three kernels mirroring E1's:
//   null    — empty launch. Isolates host+launch cost so it cancels.
//   feed    — grid-stride read of a large buffer. The GPU analogue of C2: bytes
//             delivered from DDR to the compute units.
//   compute — resident FMA chain in registers, no memory traffic. Analogue of
//             K1's VMAC chain.
//
// Reports nothing itself: the host harness (g1_gpu.py) integrates RAPL around
// these launches.

#include <hip/hip_runtime.h>
#include <cstdio>

extern "C" __global__ void g1_null(int *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 1;
}

// Grid-stride read. `sink` keeps the loads live; the compiler cannot drop them.
extern "C" __global__ void g1_feed(const int4 *__restrict__ src, int n4, int *__restrict__ sink) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int4 acc = make_int4(0, 0, 0, 0);
  for (int i = idx; i < n4; i += stride) {
    int4 v = src[i];
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }
  int s = acc.x + acc.y + acc.z + acc.w;
  // Never true in practice; keeps `s` live without a real store per thread.
  if (s == 0x7FFFFFFF) sink[idx] = s;
}

// Resident int8 matrix chain, the GPU's analogue of the NPU's VMAC chain.
//
// Uses v_wmma_i32_16x16x16_iu8 — the same instruction hipfire's own gfx1103
// kernels use for int8 (71 call sites). It is the fair comparison to the NPU's
// mmul: both are the architecture's native int8 matrix op. (sdot4 is NOT
// available here — gfx1103 lacks dot1-insts; it is a GCN-era path that hipfire
// gates to older archs.)
//
// Each wave32 WMMA computes 16x16x16 = 4096 MACs. Four independent accumulator
// chains hide the latency, mirroring K1's structure.
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

extern "C" __global__ void g1_compute(const int *__restrict__ seed, int iters, int *__restrict__ sink) {
  v4i a = {seed[0], seed[1], seed[2], seed[3]};
  v4i b = {seed[4], seed[5], seed[6], seed[7]};
  v8i c0 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8i c1 = c0, c2 = c0, c3 = c0;
  for (int i = 0; i < iters; ++i) {
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c0, false);
    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c1, false);
    c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c2, false);
    c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c3, false);
  }
  int s = c0[0] + c1[0] + c2[0] + c3[0];
  if (s == 0x7FFFFFFF) sink[0] = s;
}

// ── C ABI launchers ─────────────────────────────────────────────────────────
// The launch loop lives here rather than in ctypes: the Python side only needs
// to integrate RAPL around the call. Each returns the number of launches.
//
// `rate` pads each launch to a fixed period, exactly as E1 does for the NPU.
// This is load-bearing, not cosmetic — free-running, package power tracks launch
// rate and swamps the kernel's own energy (E1 measured a NULL NPU kernel burning
// 10.3 W at 4326 dispatch/s while a compute kernel burned 1.1 W at 133/s).

#include <chrono>
#include <thread>

namespace {
constexpr int FEED_BYTES = 256 * 1024 * 1024;
// Sized so one launch fits inside a 50/s slot (20 ms). If this grows, the
// matched-rate discipline breaks and g1_gpu.py refuses the result.
// KEEP IN SYNC with COMPUTE_ITERS in g1_gpu.py (used for MAC accounting).
constexpr int COMPUTE_ITERS = 6000;
constexpr int THREADS = 256;
constexpr int BLOCKS = 12 * 8;

double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Pad to the slot, then return. Sleeping (not spinning) keeps the host idle so
// its power does not contaminate the measurement.
void pad(double slot_end) {
  double rest = slot_end - now_s();
  if (rest > 0) std::this_thread::sleep_for(std::chrono::duration<double>(rest));
}
}  // namespace

extern "C" double g1_launch_null(double seconds, double rate) {
  int *out;
  hipMalloc(&out, sizeof(int) * 1024);
  hipLaunchKernelGGL(g1_null, dim3(1), dim3(64), 0, 0, out);
  hipDeviceSynchronize();  // warm
  double period = rate > 0 ? 1.0 / rate : 0.0;
  double t0 = now_s();
  long n = 0;
  while (now_s() - t0 < seconds) {
    double slot = t0 + n * period;
    hipLaunchKernelGGL(g1_null, dim3(1), dim3(64), 0, 0, out);
    hipDeviceSynchronize();
    n++;
    if (period > 0) pad(slot + period);
  }
  hipFree(out);
  return (double)n;
}

extern "C" double g1_launch_feed(double seconds, double rate) {
  int4 *src;
  int *sink;
  hipMalloc(&src, FEED_BYTES);
  hipMemset(src, 1, FEED_BYTES);
  hipMalloc(&sink, sizeof(int) * THREADS * BLOCKS);
  int n4 = FEED_BYTES / (int)sizeof(int4);
  hipLaunchKernelGGL(g1_feed, dim3(BLOCKS), dim3(THREADS), 0, 0, src, n4, sink);
  hipDeviceSynchronize();
  double period = rate > 0 ? 1.0 / rate : 0.0;
  double t0 = now_s();
  long n = 0;
  while (now_s() - t0 < seconds) {
    double slot = t0 + n * period;
    hipLaunchKernelGGL(g1_feed, dim3(BLOCKS), dim3(THREADS), 0, 0, src, n4, sink);
    hipDeviceSynchronize();
    n++;
    if (period > 0) pad(slot + period);
  }
  hipFree(src);
  hipFree(sink);
  return (double)n;
}

extern "C" double g1_launch_compute(double seconds, double rate) {
  int *seed, *sink;
  hipMalloc(&seed, sizeof(int) * 32);
  hipMemset(seed, 3, sizeof(int) * 32);
  hipMalloc(&sink, sizeof(int) * 64);
  hipLaunchKernelGGL(g1_compute, dim3(BLOCKS), dim3(THREADS), 0, 0, seed, COMPUTE_ITERS, sink);
  hipDeviceSynchronize();
  double period = rate > 0 ? 1.0 / rate : 0.0;
  double t0 = now_s();
  long n = 0;
  while (now_s() - t0 < seconds) {
    double slot = t0 + n * period;
    hipLaunchKernelGGL(g1_compute, dim3(BLOCKS), dim3(THREADS), 0, 0, seed, COMPUTE_ITERS, sink);
    hipDeviceSynchronize();
    n++;
    if (period > 0) pad(slot + period);
  }
  hipFree(seed);
  hipFree(sink);
  return (double)n;
}
