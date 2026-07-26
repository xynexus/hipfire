// Eight-way row-striped activation pack with direct-stream ring all-gather.
#include "../r21/r21_w4_pack_down.cc"

constexpr int FRAGMENT_ROWS = 3;
constexpr int FRAGMENT_BYTES = FRAGMENT_ROWS * GROUP + 16;
constexpr int FRAGMENT_WORDS = FRAGMENT_BYTES / sizeof(int);

extern "C" void r22_pack3(const float *__restrict input,
                           const int8 *__restrict weight_payload,
                           int8 *__restrict activation_payload,
                           float *__restrict scratch,
                           int8 *__restrict fragment, int owner) {
  const float *owned_input = input + (owner & 1) * FRAGMENT_ROWS * GROUP;
  for (int row = 0; row < FRAGMENT_ROWS; row++)
    r21_w4_pack_row(owned_input + row * GROUP, weight_payload,
                    activation_payload, scratch,
                    owner * FRAGMENT_ROWS + row);

  const float *scales =
      reinterpret_cast<const float *>(activation_payload + A_DATA);
  for (int row = 0; row < FRAGMENT_ROWS; row++) {
    const int local_row = owner * FRAGMENT_ROWS + row;
    const int lm = local_row / 4;
    const int rr = local_row % 4;
    for (int kt = 0; kt < 16; kt++) {
      const int source = (lm * 16 + kt) * 64 + rr * 16;
      aie::store_v(fragment + row * GROUP + kt * 16,
                   aie::load_v<16>(activation_payload + source));
    }
    reinterpret_cast<float *>(fragment + FRAGMENT_ROWS * GROUP)[row] =
        scales[local_row];
  }
  reinterpret_cast<int *>(fragment)[FRAGMENT_WORDS - 1] = 0;
}

extern "C" void r22_insert_fragment(const int8 *__restrict fragment,
                                     int8 *__restrict activation_payload,
                                     int owner) {
  float *scales = reinterpret_cast<float *>(activation_payload + A_DATA);
  for (int row = 0; row < FRAGMENT_ROWS; row++) {
    const int local_row = owner * FRAGMENT_ROWS + row;
    const int lm = local_row / 4;
    const int rr = local_row % 4;
    for (int kt = 0; kt < 16; kt++) {
      const int target = (lm * 16 + kt) * 64 + rr * 16;
      aie::store_v(activation_payload + target,
                   aie::load_v<16>(fragment + row * GROUP + kt * 16));
    }
    scales[local_row] =
        reinterpret_cast<const float *>(fragment + FRAGMENT_ROWS * GROUP)[row];
  }
}

extern "C" void r22_send_fragment(const int8 *__restrict fragment) {
  const int *words = reinterpret_cast<const int *>(fragment);
  for (int word = 0; word < FRAGMENT_WORDS; word++) put_ms(words[word]);
}

extern "C" void r22_receive_fragment(int8 *__restrict fragment) {
  int *words = reinterpret_cast<int *>(fragment);
  for (int word = 0; word < FRAGMENT_WORDS; word++) words[word] = get_ss_int();
}
