// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// P1: the CPU leg of the NPU/GPU/CPU split, measured like E1 and G1.
//
// Same die, same package rails, same RAPL counter as the NPU and iGPU — so all
// three legs are directly comparable. The CPU is simpler to measure than the
// other two: there is no dispatch, so no rate-matching is needed (E1's confound
// was the HOST submit loop, which here IS the workload).
//
// Two kernels mirroring E1/G1:
//   feed    — threaded streaming read of a large buffer. DDR bandwidth.
//   compute — resident VNNI int8 dot chain. _mm512_dpbusd_epi32 does 64 int8
//             MACs per instruction (16 int32 lanes x 4 products), the CPU's
//             native int8 path on Zen 4. 8 independent accumulators hide the
//             latency, mirroring K1's chain structure.
//
// Prints work done; the Python harness integrates RAPL around it.

#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 256 MiB: far beyond the 32 MiB L3, so this is real DDR traffic like G1's.
#define FEED_BYTES (256ull * 1024 * 1024)
#define CHAINS 8

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "feed";
  double seconds = argc > 2 ? atof(argv[2]) : 8.0;
  int threads = argc > 3 ? atoi(argv[3]) : omp_get_max_threads();
  omp_set_num_threads(threads);

  if (!strcmp(mode, "feed")) {
    int32_t *buf = aligned_alloc(64, FEED_BYTES);
    if (!buf) return 1;
    memset(buf, 1, FEED_BYTES);
    const size_t n = FEED_BYTES / sizeof(int32_t);
    long passes = 0;
    double t0 = now_s();
    int64_t sink = 0;
    while (now_s() - t0 < seconds) {
      int64_t total = 0;
#pragma omp parallel for reduction(+ : total) schedule(static)
      for (size_t i = 0; i < n; i += 16) {
        __m512i v = _mm512_load_si512((const void *)(buf + i));
        total += _mm512_reduce_add_epi32(v);
      }
      sink += total;
      passes++;
    }
    double el = now_s() - t0;
    // sink is printed so the loop cannot be optimised away.
    printf("mode=feed threads=%d passes=%ld elapsed=%.6f bytes=%llu sink=%lld\n",
           threads, passes, el, (unsigned long long)FEED_BYTES * passes, (long long)sink);
    free(buf);
    return 0;
  }

  // compute: resident VNNI chain, no memory traffic in the inner loop.
  const long iters = 2000000;
  long reps = 0;
  double t0 = now_s();
  int64_t sink = 0;
  while (now_s() - t0 < seconds) {
    int64_t part = 0;
#pragma omp parallel reduction(+ : part)
    {
      __m512i a = _mm512_set1_epi8(3), b = _mm512_set1_epi8(5);
      __m512i c[CHAINS];
      for (int j = 0; j < CHAINS; j++) c[j] = _mm512_setzero_si512();
      for (long i = 0; i < iters; i++) {
        for (int j = 0; j < CHAINS; j++) c[j] = _mm512_dpbusd_epi32(c[j], a, b);
      }
      __m512i s = _mm512_setzero_si512();
      for (int j = 0; j < CHAINS; j++) s = _mm512_add_epi32(s, c[j]);
      part += _mm512_reduce_add_epi32(s);
    }
    sink += part;
    reps++;
  }
  double el = now_s() - t0;
  // 64 int8 MACs per dpbusd (16 int32 lanes x 4 products).
  double macs = (double)reps * iters * CHAINS * 64.0 * threads;
  printf("mode=compute threads=%d reps=%ld elapsed=%.6f macs=%.0f sink=%lld\n",
         threads, reps, el, macs, (long long)sink);
  return 0;
}
