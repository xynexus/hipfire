// no-op body: this probe measures the dataflow, not arithmetic
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" __attribute__((noinline)) void
barrier_noop(const bfloat16 *restrict in, bfloat16 *restrict out) {
  out[0] = in[0];
}
