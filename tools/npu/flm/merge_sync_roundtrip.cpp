// Round-trip test for mlir-aie's txn_append_merge_sync().
//
// The encoding was recovered from XRT's aie-rt (XAie_Txn_MergeSync in
// xaie_helper.c) and must reproduce the MERGE_SYNC operation that FastFlowLM's
// shipped v1.0 transaction binaries actually contain. This test emits one and
// searches for those exact bytes in an FLM blob -- if the encoder drifts, or if
// the payload packing is wrong, the search fails.
//
//   g++ -std=c++17 -I <mlir-aie>/include merge_sync_roundtrip.cpp -o rt
//   ./rt <mlir-aie>/../hipfire-npu/docs/npu/flm_c8_a.txn
//
// FLM's 8-column layer dispatch closes with num_tokens=16, num_cols=8: two
// task-completion tokens per column, one merged wait replacing eight TCTs.

#include "aie/Runtime/TxnEncoding.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

constexpr uint32_t kFlmNumTokens = 16;
constexpr uint32_t kFlmNumCols = 8;

std::vector<uint8_t> readFile(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return {};
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

} // namespace

int main(int argc, char **argv) {
  // 1. Encoder shape: three words, opcode 0x84, size 12, packed payload.
  std::vector<uint32_t> txn;
  aie_runtime::txn_append_merge_sync(txn, kFlmNumTokens, kFlmNumCols);

  assert(txn.size() == 3 && "merge_sync must emit exactly 3 words");
  assert(txn[0] == 0x84 && "opcode must be TXN_OPC_CUSTOM_OP_BEGIN + 4");
  assert(txn[1] == 12 && "operation size must be 12 bytes, not TCT's 16");
  assert(txn[2] == 0x0810 && "payload must pack num_tokens | num_cols << 8");

  // 2. Field packing is order-sensitive; a swap still yields 3 words and would
  //    pass the shape checks above.
  std::vector<uint32_t> swapped;
  aie_runtime::txn_append_merge_sync(swapped, kFlmNumCols, kFlmNumTokens);
  assert(swapped[2] != txn[2] && "num_tokens and num_cols must not commute");

  // 3. Byte-exact match against a real FLM transaction binary.
  if (argc < 2) {
    printf("shape checks passed; pass an FLM .txn to check bytes\n");
    return 0;
  }
  std::vector<uint8_t> blob = readFile(argv[1]);
  if (blob.empty()) {
    fprintf(stderr, "could not read %s\n", argv[1]);
    return 2;
  }

  const auto *needle = reinterpret_cast<const uint8_t *>(txn.data());
  const size_t n = txn.size() * sizeof(uint32_t);
  size_t found = 0, at = 0;
  for (size_t i = 0; i + n <= blob.size(); ++i)
    if (memcmp(blob.data() + i, needle, n) == 0) {
      if (!found)
        at = i;
      ++found;
    }

  if (found != 1) {
    fprintf(stderr,
            "FAIL: expected exactly one MERGE_SYNC in %s, found %zu\n"
            "      encoder emitted:",
            argv[1], found);
    for (uint32_t w : txn)
      fprintf(stderr, " %08x", w);
    fprintf(stderr, "\n");
    return 1;
  }

  printf("PASS: encoder output matches FLM's MERGE_SYNC byte-for-byte\n"
         "      %s @ 0x%zx: [%08x %08x %08x] -> num_tokens=%u num_cols=%u\n",
         argv[1], at, txn[0], txn[1], txn[2], txn[2] & 0xff,
         (txn[2] >> 8) & 0xff);
  return 0;
}
