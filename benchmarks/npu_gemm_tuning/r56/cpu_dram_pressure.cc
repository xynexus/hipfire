// SPDX-License-Identifier: Apache-2.0
//
// Sustained host-memory pressure for NPU fabric-contention experiments.
// Each worker repeatedly copies between private, page-populated buffers. The
// printed bandwidth counts one logical read plus one logical write per byte.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

int main(int argc, char **argv) {
  const unsigned threads =
      argc > 1 ? static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10))
               : std::thread::hardware_concurrency();
  const std::size_t mib_per_buffer =
      argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 128;
  const double seconds = argc > 3 ? std::strtod(argv[3], nullptr) : 20.0;
  if (threads == 0 || mib_per_buffer == 0 || seconds <= 0.0) {
    std::cerr << "usage: cpu_dram_pressure [threads] [MiB-per-buffer] [seconds]\n";
    return 2;
  }

  const std::size_t bytes = mib_per_buffer << 20;
  std::atomic<unsigned> ready{0};
  std::atomic<bool> go{false};
  std::atomic<std::uint64_t> copies{0};
  std::vector<std::thread> workers;
  workers.reserve(threads);

  for (unsigned worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&, worker] {
      auto source = std::make_unique_for_overwrite<std::byte[]>(bytes);
      auto destination = std::make_unique_for_overwrite<std::byte[]>(bytes);
      std::memset(source.get(), static_cast<int>(worker + 1), bytes);
      std::memset(destination.get(), 0, bytes);
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire))
        std::this_thread::yield();
      const auto stop = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(seconds);
      std::uint64_t local = 0;
      while (std::chrono::steady_clock::now() < stop) {
        std::memcpy(destination.get(), source.get(), bytes);
        std::swap(source, destination);
        ++local;
      }
      copies.fetch_add(local, std::memory_order_relaxed);
    });
  }

  while (ready.load(std::memory_order_acquire) != threads)
    std::this_thread::yield();
  const auto started = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);
  for (auto &worker : workers)
    worker.join();
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  const double logical_bytes = 2.0 * static_cast<double>(bytes) *
                               static_cast<double>(copies.load());
  std::cout << "threads=" << threads << " mib_per_buffer=" << mib_per_buffer
            << " copies=" << copies.load() << " seconds=" << elapsed
            << " logical_gbs=" << logical_bytes / elapsed / 1.0e9 << '\n';
}
