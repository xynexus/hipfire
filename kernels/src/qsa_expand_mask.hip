// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Expand a QSA block selection into a per-token attention mask.
//
// The reference emits a MASK and ANDs it into the causal mask rather than
// gathering a compacted KV set, so this matches that shape — and it needs no
// prefix scan, which a compacted list would.
//
// Two rules, both of which are silent if wrong:
//
//   * a selected block contributes ALL `compress_ratio` of its positions;
//   * the ragged tail past the last complete block is ALWAYS visible — it does
//     not compete for budget and is not represented in the block mask at all.
//
// The second is why the mask cannot simply be derived from the block mask: the
// tail has no block to be selected.
//
// `visible` lists the causally-legal cache positions in increasing order, so the
// mask is written at CACHE positions and a masked or paged cache stays correct.
//
// Grid: (ceil(n_visible / BLOCK)). Block = BLOCK.

#define BLOCK 256

__launch_bounds__(BLOCK)
extern "C" __global__ void qsa_expand_mask(
    const unsigned int* __restrict__ block_mask,  // [n_blocks], 1 = selected
    const int* __restrict__ visible,              // [n_visible] cache positions
    unsigned int* __restrict__ token_mask,        // [n_positions], zeroed by caller
    int n_visible,
    int n_blocks,
    int compress_ratio
) {
    const int j = blockIdx.x * BLOCK + threadIdx.x;
    if (j >= n_visible) return;

    const int b = j / compress_ratio;
    // Past the last COMPLETE block: the ragged tail, unconditionally visible.
    const bool keep = (b >= n_blocks) ? true : (block_mask[b] != 0u);
    if (keep) {
        token_mask[visible[j]] = 1u;
    }
}
