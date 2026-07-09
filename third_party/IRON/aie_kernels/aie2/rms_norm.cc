// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "aie2_math.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T, int N>
void rms_norm_general(const T *restrict input, const T *restrict input2, T *restrict output, int32_t cols)
{
    event0();
    constexpr float epsilon = 1e-5f;
    ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();

    int vector_chunks = cols / N;
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
        ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
        add_res = ::aie::add(add_res, square_v);
    }
    float sum_sq = ::aie::reduce_add(add_res);

    int remaining = cols % N;
    if (remaining > 0) {
        int start_idx = vector_chunks * N;
        for (int i = 0; i < remaining; i++) {
            T val = input[start_idx + i];
            float square = static_cast<float>(val) * static_cast<float>(val);
            sum_sq += square;
        }
    }

    float rms = sum_sq / cols + epsilon;
    float inv_rms = invsqrt(rms);
    ::aie::vector<T, N> inv_rms_v = ::aie::broadcast<T, N>(static_cast<T>(inv_rms));

    for (int i = 0; i < vector_chunks; i++) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
        ::aie::vector<T, N> norm_v = ::aie::mul(reg_a, inv_rms_v);
        ::aie::vector<T, N> out_v;
        if (input2) {
            ::aie::vector<T, N> reg_b = ::aie::load_v<N>(input2 + i * N);
            out_v = ::aie::mul(norm_v, reg_b);
        } else {
            out_v = norm_v;
        }
        ::aie::store_v(output + i * N, out_v);
    }

    if (remaining > 0) {
        int start_idx = vector_chunks * N;
        for (int i = 0; i < remaining; i++) {
            T val = input[start_idx + i];
            T norm_val = static_cast<T>(static_cast<float>(val) * inv_rms);
            if (input2) {
                T mul_val = input2[start_idx + i];
                output[start_idx + i] = static_cast<T>(static_cast<float>(norm_val) * static_cast<float>(mul_val));
            } else {
                output[start_idx + i] = norm_val;
            }
        }
    }
    event1();
}

extern "C" {
void rms_norm_bf16_vector(bfloat16 *input, bfloat16 *output, int32_t size)
{
    rms_norm_general<bfloat16, 16>(input, nullptr, output, size);
}

void weighted_rms_norm(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int32_t size)
{
    rms_norm_general<bfloat16, 16>(a_in, b_in, c_out, size);
}
}
