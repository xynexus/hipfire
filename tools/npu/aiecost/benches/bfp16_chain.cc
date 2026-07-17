#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#ifndef ITERS
#define ITERS 100000
#endif
#ifndef CHAINS
#define CHAINS 4
#endif
extern "C" void bfp16_chain(uint8_t *__restrict pA_, uint8_t *__restrict pB_, float *__restrict pOut) {
  bfp16ebs8 *pA = reinterpret_cast<bfp16ebs8 *>(pA_);
  bfp16ebs8 *pB = reinterpret_cast<bfp16ebs8 *>(pB_);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sA(pA);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sB(pB);
  aie::block_vector<bfp16ebs8, 64> a0 = sA.pop();
  aie::block_vector<bfp16ebs8, 64> a1 = sA.pop();
  aie::block_vector<bfp16ebs8, 64> b0 = sB.pop();
  aie::block_vector<bfp16ebs8, 64> b1 = sB.pop();
  aie::accum<accfloat, 64> c0 = aie::zeros<accfloat, 64>();
  aie::accum<accfloat, 64> c1 = aie::zeros<accfloat, 64>();
  aie::accum<accfloat, 64> c2 = aie::zeros<accfloat, 64>();
  aie::accum<accfloat, 64> c3 = aie::zeros<accfloat, 64>();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < ITERS; i++) {
    c0 = mac_8x8_8x8T(a0, b0, c0);
#if CHAINS > 1
    c1 = mac_8x8_8x8T(a0, b1, c1);
#endif
#if CHAINS > 2
    c2 = mac_8x8_8x8T(a1, b0, c2);
#endif
#if CHAINS > 3
    c3 = mac_8x8_8x8T(a1, b1, c3);
#endif
  }
  auto s = aie::add(aie::add(c0.to_vector<float>(), c1.to_vector<float>()),
                    aie::add(c2.to_vector<float>(), c3.to_vector<float>()));
  aie::store_v(pOut + 16, s);
  for (int i = 0; i < 8; i++) pOut[i] = (float)(ITERS);
  pOut[1] = (float)CHAINS;
}
