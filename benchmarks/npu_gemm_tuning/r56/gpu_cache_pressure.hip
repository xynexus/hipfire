// SPDX-License-Identifier: Apache-2.0
//
// GPU cache/DRAM pressure control for NPU cache-path experiments. A small pair
// of buffers (for example 8 MiB each) fits below Strix Halo's reported 32 MiB
// MALL capacity after warm-up; a large pair (for example 256 MiB each) does not.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>

#define HIP_OK(call)                                                            \
  do {                                                                          \
    hipError_t error = (call);                                                   \
    if (error != hipSuccess) {                                                   \
      std::cerr << #call << ": " << hipGetErrorString(error) << '\n';           \
      return 2;                                                                 \
    }                                                                           \
  } while (false)

__global__ void copy_pressure(const uint4 *__restrict__ source,
                              uint4 *__restrict__ destination,
                              std::size_t elements) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    destination[i] = source[i];
}

int main(int argc, char **argv) {
  const std::size_t mib_per_buffer =
      argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 8;
  const double seconds = argc > 2 ? std::strtod(argv[2], nullptr) : 20.0;
  if (mib_per_buffer == 0 || seconds <= 0.0) {
    std::cerr << "usage: gpu_cache_pressure [MiB-per-buffer] [seconds]\n";
    return 2;
  }
  const std::size_t bytes = mib_per_buffer << 20;
  const std::size_t elements = bytes / sizeof(uint4);
  uint4 *source = nullptr;
  uint4 *destination = nullptr;
  HIP_OK(hipMalloc(&source, bytes));
  HIP_OK(hipMalloc(&destination, bytes));
  HIP_OK(hipMemset(source, 0x5a, bytes));
  HIP_OK(hipMemset(destination, 0, bytes));

  for (int i = 0; i < 20; ++i)
    copy_pressure<<<320, 256>>>(source, destination, elements);
  HIP_OK(hipDeviceSynchronize());

  const auto started = std::chrono::steady_clock::now();
  std::uint64_t launches = 0;
  do {
    copy_pressure<<<320, 256>>>(source, destination, elements);
    std::swap(source, destination);
    ++launches;
    if ((launches & 63) == 0)
      HIP_OK(hipDeviceSynchronize());
  } while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         started)
               .count() < seconds);
  HIP_OK(hipDeviceSynchronize());
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  const double logical_bytes =
      2.0 * static_cast<double>(bytes) * static_cast<double>(launches);
  std::cout << "mib_per_buffer=" << mib_per_buffer << " launches=" << launches
            << " seconds=" << elapsed
            << " logical_gbs=" << logical_bytes / elapsed / 1.0e9 << '\n';
  HIP_OK(hipFree(source));
  HIP_OK(hipFree(destination));
}
