// SPDX-License-Identifier: Apache-2.0
// Vector-load/store stream Q handoff for the R72 graph-local query cache.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int Q_HALF_BYTES = 2048;
constexpr int Q_GROUPS = 6;
}

extern "C" {

__attribute__((noinline, minsize)) void
r72_send_q(const int8_t *restrict query) {
  const int32_t *words = reinterpret_cast<const int32_t *>(query);
#ifdef R72_SCALAR_STREAM
  for (int word = 0; word < Q_HALF_BYTES / (int)sizeof(int32_t); ++word)
    put_ms(words[word]);
#else
  auto input = aie::begin_restrict_vector<16>(words);
  for (int chunk = 0; chunk < Q_HALF_BYTES / 64; ++chunk)
    put_ms((*input++).to_native());
#endif
}

__attribute__((noinline, minsize)) void
r72_recv_q(int8_t *restrict cache, int32_t group, int32_t lane) {
  int32_t *words = reinterpret_cast<int32_t *>(
      cache + (group * 2 + lane) * Q_HALF_BYTES);
#ifdef R72_SCALAR_STREAM
  for (int word = 0; word < Q_HALF_BYTES / (int)sizeof(int32_t); ++word)
    words[word] = get_ss_int();
#else
  auto output = aie::begin_restrict_vector<16>(words);
  for (int chunk = 0; chunk < Q_HALF_BYTES / 64; ++chunk)
    *output++ = aie::vector<int32_t, 16>(get_ss_v16int32());
#endif
}

} // extern "C"
