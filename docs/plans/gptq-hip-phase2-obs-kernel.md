# GPTQ-HIP Phase 2: GPU OBS Column-Sequential Update via MFMA (block-128)

**Status:** Plan — not yet implemented.
**Parent:** `feat/gptq-hip-impl` HEAD `4f082d22` (Phase A-D GPU Cholesky via rocSOLVER).
**Branch:** `feat/gptq-hip-phase2-plan`
**Target hardware:** gfx942 (MI300X CDNA3). CPU fallback preserved for all other arches.
**Projected 9B Tier 1 quantize wall:** ~8-10 min (from Phase D ~1 hr, from CPU ~3 hr).

---

## §1 — Goal and Non-Goals

### Goal

Port the OBS (Optimal Brain Surgeon) column-sequential update in `gptq.rs:860-915` to GPU via an MFMA-optimized block-128 variant of the GPTQ paper §3.2 algorithm. The existing GPU path (Phase A-D) offloads only the Cholesky-inverse computation; the serial column loop at lines 860-915 still runs on CPU via Rayon. For ffn_down (M=4096, K=12288) this loop accounts for roughly 310 GFLOP of rank-1 propagation per tensor at ~30 GFLOP/s on a 20-core CPU, dominating total quantize wall by 2:1 over the Phase D GPU Cholesky step.

The block-128 paper variant (§3.2) converts that BLAS-2 stream into a single BLAS-3 GEMM per block, which MFMA on gfx942 can execute near peak. The implementation adds three new HIP kernels to `crates/hipfire-quantize/` and a new orchestrator `gptq_column_sequential_hip` in `crates/hipfire-quantize/src/gptq_hip.rs`, reusing the `RocSolver` FFI struct from Phase A-D.

### Non-Goals

- Multi-GPU OBS distribution across XGMI peers.
- Changing paper algorithm parameters. Block size B=128 is the canonical Frantar et al. choice; no tuning or adaptive sizing in v1.
- Reordering tensors across layers (batching all K=4096 tensors together). That is Phase 3.
- Changing the MQ4 quantize grid or pack format. Byte-level output must match the CPU path.
- RDNA / consumer-GPU support for the OBS GPU path. The CDNA FP64-MFMA gate that Phase D already enforces (`detect_is_cdna` at `main.rs:94-108`) applies here too. On RDNA the CPU path runs unchanged.
- hipGraph capture to amortize per-block kernel launch overhead. That is a Phase 3 optimization. Phase 2 launches serially per column step.
- AWQ pre-scaling on GPU. That transform is cheap CPU-side (`apply_awq_rescaling` at `gptq.rs:500-514`) and not worth moving.
- WEIGHT-mode actorder GPU sort. `weight_mode_actorder` at `gptq.rs:737-744` is an argsort taking microseconds.

---

## §2 — Algorithm: Paper §3.2 Block Variant

The current CPU loop in `gptq_column_sequential` (lines 860-915) processes each column step serially: quantize column `j_orig = perm[step]`, compute per-row error `err = (w - q) / U[step, step]`, then for every remaining step `k > step` subtract `err * U[step, k]` from `W_residual[:, perm[k]]`. This is O(K²·M) with a BLAS-2 memory pattern — each step reads a column of W and a row of U.

The paper §3.2 block variant processes B=128 consecutive steps together:

```
for block_start in 0..K step B:
    block_end = min(block_start + B, K)

    # Phase A: serial within the block (B serial quantize+intra-block-rank1 steps)
    for step in block_start..block_end:
        j_orig = perm[step]
        u_ss   = U[step, step]
        # M-parallel: quantize column, compute err_col[step]
        # err_col[step][row] = (W_residual[row, j_orig] - q) / u_ss
        quantize_column(W_flat, W_residual, frozen_grids, j_orig, u_ss, m, k_dim,
                        -> err_col_block[:, step - block_start])

        # M-parallel rank-1 update within block only
        # W_residual[:, perm[k]] -= err_col[step] * U[step, k]
        #   for k in step+1..block_end
        for k in step+1..block_end:
            rank1_update_column(W_residual, err_col_block, U, step, k, perm, m, k_dim)

    # Phase B: single rank-B GEMM to flush accumulated block errors
    # to the remaining K-block_end columns — the BLAS-3 payoff.
    # W_residual[:, perm[block_end..K]] -= err_block @ U_remaining
    #
    # err_block shape:      M × B       (columns = err_col_block[block_start..block_end])
    # U_remaining shape:    B × (K - block_end)  (rows block_start..block_end of U,
    #                                              cols block_end..K)
    block_gemm_mfma(W_residual, err_block, U, perm, block_start, block_end, m, k_dim)
```

Phase A does B quantize-column ops plus B(B-1)/2 within-block rank-1 updates = O(B²·M) BLAS-2 work per block. Total over all K/B blocks: O(B·K·M) — identical leading term to the unblocked loop.

Phase B is the leverage: one M×(K-block_end) GEMM with inner dim B applied to the remaining columns. Total Phase B work over all blocks: approximately K²·M/2 FLOPs (same as Phase A) but structured as BLAS-3, matching what MFMA tiles love. For the first block (block_start=0, block_end=128) on ffn_down: 4096 × 12160 × 128 × 2 FLOPs = 12.8 GFLOP. Summed across K/B=96 blocks (geometric series, average remaining = K/2): ~310 GFLOP total.

**Device memory layout** for ffn_down (M=4096, K=12288):

| Buffer | Shape | Dtype | Size |
|---|---|---|---|
| `d_W` (weights_residual) | M × K row-major | F64 | 384 MB |
| `d_W_quant` (weights_flat output) | M × K row-major | F64 | 384 MB |
| `d_U` (from Phase D, stays on device) | K × K upper-tri | F64 | 1.15 GB |
| `d_err_block` (accumulator) | M × B | F64 | 4 MB |
| `d_frozen_grids` | (M·K/256) × 16 B | F32 × 2 | 24 MB |
| `d_perm` | K | U32 | 49 KB |

Total: ~1.95 GB per tensor for ffn_down. Fits MI300X 192 GB with many tensors in flight; marginal on 24 GB consumer cards (Phase G / streaming for those, not in scope here).

For the 27B case: ffn_down has M=7168, K=18432. `d_U` alone is 18432²×8 = 2.72 GB. Total ~4.8 GB per tensor. Comfortably MI300X-only.

---

## §3 — Kernel Breakdown

### Kernel 1: `quantize_mq4_column_f64` (~120 LOC)

**File:** `kernels/src/gptq_obs_phase2.gfx942.hip` (new, Phase 2A)

**Purpose:** quantize one column of W_residual to MQ4 and write the per-row error into a column slot of the `d_err_block` accumulator.

**Launch geometry:** grid = `(M + 255) / 256`, block = `256`. One thread per row (M up to 7168 for 27B ffn_down, 28 blocks of 256).

**Inputs:**
- `d_W_residual`: `double* [M × K]` row-major — current residual weights
- `d_W_quant`: `double* [M × K]` — output quantized weight buffer (same layout)
- `d_frozen_grids`: `packed_grid_t* [(M * K) / 256]` — device-side copy of `BlockGrid`, each 16 bytes: `{float scale, float min_val, char _pad[8]}`
- `col_orig`: `int` — which column to quantize (the un-permuted index `j_orig = perm[step]`)
- `u_ss`: `double` — `U[step, step]` diagonal entry, divisor for error
- `k_dim`: `int` — K
- `d_err_block`: `double* [M × B]` — output accumulator; this call writes column `(step - block_start)`
- `err_col_slot`: `int` — the column index within `d_err_block` to write (= `step - block_start`)

**Per-thread work:**
```c
int row = blockIdx.x * 256 + threadIdx.x;
if (row >= M) return;
int block_idx = (row * k_dim + col_orig) / 256;
double scale    = (double)d_frozen_grids[block_idx].scale;
double min_val  = (double)d_frozen_grids[block_idx].min_val;
double w = d_W_residual[row * k_dim + col_orig];
double q = quantize_mq4_f64(w, scale, min_val);   // inline, mirrors gptq.rs:92-101
double err = (w - q) / u_ss;
d_W_quant[row * k_dim + col_orig] = q;             // write quantized value
d_err_block[row * B + err_col_slot] = err;         // write to accumulator
```

The inline `quantize_mq4_f64` device function mirrors `quantize_mq4_element_with_clamp` (gptq.rs:92-101): `q_raw = floor((w - min_val) / scale + 0.5); q = clamp(q_raw, 0.0, 15.0); return q * scale + min_val`. Clamp counters are omitted in the GPU kernel (the CPU diagnostic at gptq.rs:917-928 becomes a post-D2H count if needed; not critical for performance).

**Note:** This kernel writes `d_W_quant` but does NOT update `d_W_residual`. The residual is updated in-place by Kernel 2 (within-block) and Kernel 3 (cross-block). The split matches the CPU loop's two-buffer design (`weights_flat` for output, `weights_residual` for the live residual at gptq.rs:842).

### Kernel 2: `gptq_obs_within_block_rank1_f64` (~150 LOC)

**File:** `kernels/src/gptq_obs_phase2.gfx942.hip` (same file, Phase 2B)

**Purpose:** apply within-block rank-1 update for one column step to the remaining `(block_end - step - 1)` columns inside the current block. This is the inner loop at gptq.rs:907-914 restricted to `next_step in (step+1)..block_end`.

**Launch geometry:** grid = `((M + 63) / 64, block_end - step - 1)`, block = `64`. dim-y iterates over the remaining columns inside the block; dim-x iterates over rows. At B=128, max dim-y = 127, max dim-x blocks = ceil(7168/64) = 112. Total threads per call: up to 112 × 127 × 64 = ~910K — fine.

**Inputs:**
- `d_W_residual`: `double* [M × K]` — mutated in place
- `d_err_block`: `double* [M × B]` — read column `(step - block_start)`
- `d_U_block_row`: `const double*` — pointer to row `step` of U, length K (upper-triangular, so only columns `>= step` are meaningful)
- `d_perm`: `const int* [K]` — permutation
- `step`, `block_start`, `block_end`, `k_dim`, `M`, `B`: `int`

**Per-thread work:**
```c
int row  = blockIdx.x * 64 + threadIdx.x;
int slot = blockIdx.y;                           // 0-based index within remaining
int next_step = step + 1 + slot;
if (next_step >= block_end) return;
if (row >= M) return;

int kk_orig = d_perm[next_step];
double u_sn = d_U_block_row[next_step];         // U[step, next_step] (col-major in U)
if (u_sn == 0.0) return;

int err_slot = step - block_start;
double err = d_err_block[row * B + err_slot];
if (err == 0.0) return;

d_W_residual[row * k_dim + kk_orig] -= err * u_sn;
```

This is the exact GPU translation of gptq.rs:902-914's inner loop, restricted to within-block columns. No MFMA — this is BW-bound scalar F64, which is fine because this kernel only processes B(B-1)/2 ≈ 8128 column-pairs total across all steps in one block: 4096 × 127 × 8 = 4 MB of writes per block, executed in ~1 ms at 100 GB/s MI300X HBM3.

**U storage:** Phase D's `compute_damped_inv_cholesky_upper_hip` currently does a D2H copy of U back to CPU (gptq_hip.rs:255-258). Phase 2 needs U to **stay on device**. This is a key interface change to Phase D: the orchestrator must keep `d_u` allocated and pass a device pointer through to Phase 2 kernels, rather than copying to the CPU `Mat<f64>`. See §4 for the interface design.

### Kernel 3: `gptq_obs_block_apply_mfma_f64` (~400 LOC, or ~250 LOC for BF16 variant)

**File:** `kernels/src/gptq_obs_block_apply_mfma.gfx942.hip` (new file, Phase 2C/2D)

**Purpose:** apply the accumulated B columns of block errors to all remaining K-block_end columns via a single GEMM. This is the BLAS-3 payoff:

```
W_residual[:, perm[block_end..K]] -= err_block @ U_submatrix
where U_submatrix = U[block_start:block_end, block_end:K]
```

Dimensions: M × (K-block_end) output, M × B left operand, B × (K-block_end) right operand.

**MFMA variant decision (FP64 vs BF16) — critical design choice, see §8 Q1.**

#### FP64 path (Phase 2C)

Use `__builtin_amdgcn_mfma_f64_16x16x4_f64`. On gfx942 (CDNA3) this instruction computes a 16×16 F64 D tile with 4 F64 elements of inner-product accumulation per call. Peak throughput: 190 TFLOP/s FP64. 

Inner-dim B=128 requires 128/4 = 32 MFMA calls to fully accumulate one output tile. Each call contributes 16×16×4×2 = 2048 FLOPs. 32 calls = 65536 FLOPs = one 16×16×128 output tile.

**Tile geometry (FP64 path):**

- WG size: 128 threads = 2 wave-64
- 2 waves arranged as 2×1 over the M-dim (wave 0: m_off=0, wave 1: m_off=16 — a 32×16 M×N output tile per WG)
- Grid: `((M+31)/32, (K-block_end+15)/16)`
- LDS B-tile: `16 × B × sizeof(double)` = 16 × 128 × 8 = 16 KB per WG. gfx942 has 64 KB LDS. For 2-WG occupancy, 32 KB is consumed, leaving 32 KB for the 2nd WG — double-buffer still feasible.
- Cooperative LDS load: 128 threads load 16-col × B-row slice of U_submatrix (16 × 128 = 2048 doubles = 16 KB) once per 16-wide N-tile.

**MFMA inner loop (FP64 path) — per wave:**
```c
// wave holds a 16×16 output sub-tile
// outer loop over B in 4-element increments
double4 c_acc = {0,0,0,0};   // 16×16 tile spread across 64 lanes
for (int bi = 0; bi < B; bi += 4) {
    double4 a_reg = load_err_block_f64(d_err_block, m_off, bi);   // from HBM
    double4 b_reg = load_u_submatrix_f64_from_lds(lds_u, n_off, bi);
    c_acc = __builtin_amdgcn_mfma_f64_16x16x4_f64(a_reg, b_reg, c_acc, 0, 0, 0);
}
// subtract: W_residual[m, n] -= c_acc[...]
```

The `err_block` A operand is loaded from HBM directly (M×B F64, ~4 MB for ffn_down — fits in L2). The U_submatrix B operand is loaded into LDS cooperatively (16 columns of K-wide U subrow, 16 KB). Per block: 16 LDS loads of 16 KB = 256 KB transferred from U HBM to LDS (L2-resident across blocks given U's temporal reuse).

**MFMA variant (BF16 path — Phase 2D alternative):**

Cast `err_block` and U submatrix entries from F64 to BF16 on load, use `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` (the same intrinsic used by `gemm_bf16_mfma.gfx942.hip` at line 72 of that file). Accumulate in F32. Peak throughput: ~1.3 PFLOP/s BF16 MFMA on gfx942 — roughly 7× faster than the FP64 path.

The BF16 path requires a down-cast (F64 → BF16) per loaded element. The `__builtin_bit_cast(s16x4, bf16x4)` pattern from `gemm_bf16_mfma.gfx942.hip:70-72` applies directly. Inner dim B=128 requires 128/16 = 8 MFMA calls to accumulate one 16×16×128 tile (vs 32 for FP64). LDS B-tile is `16 × B × sizeof(__bf16)` = 16 × 128 × 2 = 4 KB per WG — trivial.

The BF16 tile geometry matches `gemm_bf16_mfma.gfx942.hip` (4 wave64 / WG, 32×32 output tile, K_CHUNK=128, N_TILE=32) almost exactly. The only difference is the W_residual update is a subtract-accumulate rather than overwrite. The tile code is a near-verbatim adaptation of lines 84-175 of that file.

**Output write for both variants:** `d_W_residual[m * k_dim + perm[block_end + n]] -= result[m, n]`. The subtraction is a read-modify-write on d_W_residual (not d_W_quant — the residual is the live working buffer). Use `atomicAdd` with negation on FP64, or simply a non-atomic load-subtract-store since each (m, n) output cell is written by exactly one WG with no aliasing.

---

## §4 — Rust Orchestrator

### Interface change to Phase D: keep U on device

The current `compute_damped_inv_cholesky_upper_hip` (gptq_hip.rs:125-270) copies U back to CPU at lines 255-258 and frees `d_u`. Phase 2 needs U to stay on device so Kernel 2 and Kernel 3 can read it without a D2H + re-H2D round-trip. The interface change is a new return variant:

**New function in `gptq_hip.rs`:**

```rust
pub struct GpuU {
    pub d_u: *mut c_void,       // device pointer, K×K F64
    pub k: usize,
    pub effective_damp: f64,
}
// impl Drop calls fn_hip_free(d_u)
```

```rust
pub fn compute_damped_inv_cholesky_upper_hip_keep(
    solver: &RocSolver,
    h: &Mat<f64>,
    perm: Option<&[usize]>,
    initial_damp: f64,
    max_damp_multiplier: f64,
) -> Result<(GpuU, f64), CholeskyError>
```

This is Phase D's existing function with the D2H copy of `d_u` omitted and `d_u` ownership transferred to the returned `GpuU`. The existing `compute_damped_inv_cholesky_upper_hip` (which does the D2H copy) is kept unchanged for the Cholesky-only callers.

### New function: `gptq_column_sequential_hip`

**Location:** `crates/hipfire-quantize/src/gptq_hip.rs`, gated on `#[cfg(feature = "gptq-hip")]`.

**Signature:**
```rust
pub fn gptq_column_sequential_hip(
    solver: &RocSolver,
    weights_flat: &mut [f64],      // M×K row-major, mutated in place (quant output)
    h_target: &Mat<f64>,           // K×K Hessian in GPTQ basis
    m: usize,
    k_dim: usize,
    frozen_grids: &[BlockGrid],
    initial_damp: f64,
    max_damp_multiplier: f64,
    tensor_name: &str,
) -> Result<f64, CholeskyError>    // returns effective_damp
```

**Steps:**

1. **CPU actorder** (unchanged from gptq.rs:803-804): `let perm = weight_mode_actorder(&h_diag);` — sorting is ~50µs, not worth GPUing.

2. **GPU Cholesky** (new `_keep` variant): call `compute_damped_inv_cholesky_upper_hip_keep(solver, h_target, Some(&perm), ...)` to get `GpuU { d_u, k, effective_damp }`. U stays on device.

3. **H2D: W and grids**: allocate `d_W_residual`, `d_W_quant`, `d_frozen_grids`, `d_perm` on device. Copy `weights_flat` to `d_W_residual` and `d_W_quant` (Phase A quantize updates `d_W_quant`; `d_W_residual` starts as the same initial weights). Copy `frozen_grids` to device. Copy `perm` (as `Vec<u32>`) to `d_perm`.

4. **Allocate `d_err_block`**: `M × B` F64 = M × 128 × 8 bytes. Zeroed before each block.

5. **Block loop** (K/B = up to 96 blocks for K=12288):

   ```rust
   for block_start in (0..k_dim).step_by(B) {
       let block_end = (block_start + B).min(k_dim);
       // Kernel 1 + Kernel 2 launches (B serial steps within block)
       for step in block_start..block_end {
           let j_orig = perm[step] as u32;
           let u_ss = /* D2H scalar read of U[step, step] from d_u */ ...;
           if u_ss <= 0.0 { continue; }
           let err_col_slot = (step - block_start) as u32;
           launch_quantize_mq4_column_f64(d_W_residual, d_W_quant, d_frozen_grids,
               j_orig, u_ss, k_dim, m, d_err_block, err_col_slot);
           if step + 1 < block_end {
               launch_gptq_obs_within_block_rank1_f64(d_W_residual, d_err_block,
                   d_u_row_ptr(d_u, step, k_dim), d_perm,
                   step, block_start, block_end, k_dim, m, B);
           }
       }
       // Kernel 3: rank-B GEMM for remaining columns
       if block_end < k_dim {
           launch_gptq_obs_block_apply_mfma(d_W_residual, d_err_block, d_u,
               d_perm, block_start, block_end, k_dim, m, B);
       }
       // Clear d_err_block for next block
       hipMemset(d_err_block, 0, m * B * size_of::<f64>());
   }
   ```

6. **D2H** for `d_W_quant` back to `weights_flat`. Synchronize device. Free all device buffers (except `d_u` which `GpuU::drop` handles).

7. Return `effective_damp`.

**Note on `u_ss` D2H scalar read:** `U[step, step]` is one `f64` scalar. A single `hipMemcpy(..., 8, DEVICE_TO_HOST)` per step costs ~5µs of overhead. Since there are K=12288 such reads, that adds 60 ms. Mitigation: batch all K diagonal entries into a `Vec<f64>` in one D2H copy immediately after step 2 (the diagonal of U is known immediately after Cholesky). This is an 8-byte × K = 98 KB copy, one call, ~1µs.

**Wiring into `gptq_column_sequential` (gptq.rs:816-836):**

Add a new GPU-OBS arm after the existing GPU Cholesky arm. The insertion point is gptq.rs:816 — replace the `#[cfg(feature = "gptq-hip")]` block that currently just calls `compute_damped_inv_cholesky_upper_hip` with:

```rust
#[cfg(feature = "gptq-hip")]
{
    if let Some(s) = solver {
        let obs_on_gpu = std::env::var("HIPFIRE_GPTQ_HIP_OBS").is_ok();
        if obs_on_gpu {
            return crate::gptq_hip::gptq_column_sequential_hip(
                s, weights_flat, h_target, m, k_dim, frozen_grids,
                initial_damp, max_damp_multiplier, tensor_name,
            );
        }
        // Cholesky-only GPU path (Phase D — keep for regression comparison)
        let (u, effective_damp) = crate::gptq_hip::compute_damped_inv_cholesky_upper_hip(
            s, h_target, Some(&perm), initial_damp, max_damp_multiplier,
        )?;
        (u, effective_damp)
    } else { /* CPU fallback */ }
}
```

The `HIPFIRE_GPTQ_HIP_OBS` env var makes the GPU OBS path opt-in during Phase 2E validation. After Phase 2F confirms end-to-end quality, the logic can be flipped to opt-out (`--gptq-hip-cholesky-only`) or auto-on based on arch detection, following the existing PR merge policy (selectable additions land freely; default-flip needs discussion per `feedback_pr_gating_policy.md`).

---

## §5 — Implementation Phases (Multi-Commit)

| Phase | Scope | New files / LOC | Expected wall |
|---|---|---|---|
| **2A** | `quantize_mq4_column_f64` kernel + Rust `fn_hip_launch_kernel` FFI + parity test vs CPU per-column | `gptq_obs_phase2.gfx942.hip` ~120 LOC; `gptq_hip.rs` +150 LOC | 1 day |
| **2B** | `gptq_obs_within_block_rank1_f64` kernel + parity test (CPU within-block-only update vs GPU) | same `.hip` +150 LOC; test +80 LOC | 1 day |
| **2C** | `gptq_obs_block_apply_mfma_f64` kernel (FP64 MFMA path) + unit test on synthetic M=64, K=512, B=128 | `gptq_obs_block_apply_mfma.gfx942.hip` ~400 LOC | 2 days |
| **2D** | BF16 cast-trick variant of Kernel 3 + correctness gate vs FP64 | same file +250 LOC alt kernel | 1 day |
| **2E** | `gptq_column_sequential_hip` orchestrator + `HIPFIRE_GPTQ_HIP_OBS` wiring; `GpuU` struct; diagonal D2H batch | `gptq_hip.rs` +300 LOC; `gptq.rs` +20 LOC | 1 day |
| **2F** | End-to-end 9B validation: byte-compare `weights_flat` CPU vs GPU for 5 tensors (ffn_down, ffn_up, ffn_gate, q_proj, v_proj); coherence gate on resulting .mq4 | 0 new LOC; test scripts | 1 day |
| **Total** | | ~1370 LOC | **7 days** |

Phase order rationale: 2A and 2B ship the simple M-parallel kernels first, establishing the D2H comparison harness before Kernel 3's MFMA complexity is introduced. Phase 2C ships the FP64-correct path before the BF16 optimization is attempted in 2D. Phase 2E plumbs everything together only after all three kernels have individual parity tests. Phase 2F is the end-to-end gate.

**Commit discipline:** each phase gets its own commit on `feat/gptq-hip-phase2-plan` (which will be renamed to `feat/gptq-hip-phase2` during implementation). Failed parity gates are committed as `test(gptq-hip-phase2): FAILING N tests — [description]` per the project's failure-documentation rule (CLAUDE.md §Rules 2+3).

---

## §6 — Numerical Equivalence Policy

**FP64 path:** max-element error between `weights_flat` (CPU output) and GPU output must satisfy:

```
max_i |W_gpu[i] - W_cpu[i]| / ||W||_F < 1e-10
```

This is the same tolerance used by Phase D's `parity_gate_k256` and `parity_gate_k4096` tests (gptq_hip.rs:326-344). MQ4 codewords should be byte-identical except at tie positions (where `floor(q_raw + 0.5)` lands exactly on a half-integer and hardware rounding differs). Track tie positions as a diagnostic; don't fail the gate on them.

**BF16 cast-trick path (Kernel 3 only):** the F64→BF16 cast introduces ~1e-3 relative error per element, and the F32 accumulation adds ~1e-7 per inner-product step. For B=128 accumulations the expected max error is:

```
~sqrt(B) * max(BF16_ULP, F32_accum_ULP) * ||err_block||_F * ||U_submatrix||_F / ||W||_F
~ sqrt(128) * 1e-3 * (loose bound) ~ 1e-2
```

This is too loose for a tight gate. Use the empirical approach: run both FP64 and BF16 on the same tensor, record the distribution of MQ4 codeword differences (not weight differences), and gate on: the fraction of codewords that differ between FP64 and BF16 variants must be < 0.1% (i.e., OBS error from BF16 accumulation is small enough to flip fewer than 1 in 1000 codewords compared to FP64). The `coherence-gate.sh` pass on the resulting `.mq4` is the final quality arbiter.

If BF16 causes more than 0.1% codeword divergence vs FP64, ship FP64 for Kernel 3 and document the quality cost of BF16 in the commit message. The 7× wall-time difference (~0.1 sec vs ~0.7 sec per ffn_down) does not justify a measurable quality regression given the goal is to match CPU GPTQ output.

**Clamp diagnostic:** the GPU orchestrator must preserve the per-tensor clamp statistics that `gptq_column_sequential` prints to stderr at gptq.rs:917-928. The GPU path should perform a single D2H count of clamped elements after the block loop and emit the same `[gptq-clamp]` line format for pipeline log compatibility.

---

## §7 — Performance Projections

### Per-tensor wall (ffn_down M=4096, K=12288)

| Step | Time (FP64 MFMA) | Time (BF16 MFMA) | Notes |
|---|---|---|---|
| Phase D Cholesky (GPU) | 12 s | 12 s | Unchanged from Phase D |
| H2D W + grids | ~20 ms | ~20 ms | 384 MB + 24 MB at ~40 GB/s PCIe |
| Per-block Phase A (96 blocks × B steps) | ~60 ms | ~60 ms | Scalar F64, BW-bound |
| Per-block Phase B GEMM (96 blocks) | ~700 ms | ~70 ms | FP64 at 50% peak vs BF16 |
| D2H W_quant | ~10 ms | ~10 ms | 384 MB |
| Kernel launch overhead (K + K/B) | ~75 ms | ~75 ms | 12288 + 96 calls × ~5µs |
| **Total Phase 2** | **~880 ms** | **~240 ms** | Per ffn_down |
| **Total w/ Cholesky** | **~13 s** | **~12.3 s** | |

Current Phase D (Cholesky on GPU, OBS on CPU): ~3 min per ffn_down (Cholesky ~12 s + OBS ~168 s).

Phase 2 FP64 target: ~0.9 s per ffn_down = 197× speedup on the OBS step, 14× total.

Phase 2 BF16 target: ~0.24 s per ffn_down = 700× speedup on OBS, 15× total.

### 9B model end-to-end

9B has ~146 GPTQ tensors, of which ~36 are ffn_down-class (K=4096, K=8192), ~36 ffn_up/gate-class (similar K), and ~74 attention projections (smaller K, faster). Using the Phase 2 FP64 estimate with average tensor cost ~300 ms (geometric mean of small/large tensors):

- **Phase D:** ~1 hr (measured per project_mi300x_rental_2026_05_18_delivery memory entry)
- **Phase 2 FP64 projection:** ~73 s for OBS + Cholesky baseline = ~9 min total
- **Phase 2 BF16 projection:** ~7 min total

### 27B model end-to-end

27B has ~186 tensors, ffn_down at M=7168 K=18432 = 2.7 GB d_U alone. MI300X handles it. Estimated per-tensor OBS cost BF16: ~1 s (M×K larger, K/B more blocks). Total: ~186 s = ~3 min (vs projected Phase D ~8 hr).

---

## §8 — Risks

**Risk 1: Kernel launch overhead at K serial steps.**

K=12288 column steps = 12288 Kernel 1 launches + up to 12288 Kernel 2 launches = ~24K launches per tensor. At 5µs/launch (HIP minimum), that is 120 ms overhead alone — non-negligible relative to the ~240 ms BF16 OBS wall. Mitigation (available now without hipGraph): **fuse steps per block into a single kernel** that loops internally over `step in block_start..block_end`, doing Kernel 1 + Kernel 2 work sequentially within one GPU kernel. This eliminates 127 out of 128 launches per block, reducing total launches from ~24K to K/B=96 (Kernel 3 calls) + 96 (fused A/B calls) = 192. Overhead drops to ~1 ms. The fused kernel's serial inner loop is fine because step N+1 within a block depends on step N's output — there is no intra-block parallelism at the step level, only row-level parallelism.

Decision for v1: **fuse Phase A steps** (Kernel 1 + Kernel 2 per block) into one `gptq_obs_block_phase_a_f64` kernel that takes `block_start`, `block_end`, `d_perm`, `d_U`, and loops over steps internally. This replaces the two separate per-step kernel launches in the orchestrator with one per-block launch. Phase 2A and 2B plan their kernels individually for testability; the fused version is implemented in Phase 2E.

**Risk 2: BF16 precision for Kernel 3.**

As computed in §6, BF16 + F32-accumulate can flip codewords. The 0.1% gate is the empirical tripwire. If the BF16 path fails the gate, the fallback is pure FP64 which projects ~880 ms per ffn_down (still 204× vs CPU). The FP64 path is the guaranteed ship target; BF16 is a bonus.

Additionally: the `__builtin_amdgcn_mfma_f64_16x16x4_f64` intrinsic has an inner-dim of 4, so the B=128 inner loop requires 32 MFMA calls. gfx942 FP64 MFMA throughput per wave is ~3 TFLOP/s (190 TFLOP/s / 64 waves). Per MFMA call: 16×16×4×2 = 2048 FLOPs / ~500 ps = ~4 TFLOP/s peak. The FP64 path projects ~50% MFMA utilization due to LDS and HBM latency; the numbers in §7 use that assumption. If utilization is lower (e.g. 25%), FP64 rises to ~1.7 s per ffn_down, still ~106× faster than CPU.

**Risk 3: U storage changes break Phase D callers.**

The `GpuU` wrapper (new in Phase 2E) keeps `d_u` alive across the OBS loop. The existing `compute_damped_inv_cholesky_upper_hip` (which D2H-copies and frees `d_u`) is kept unchanged. No existing test or caller is modified. The new `_keep` variant is additive. Low risk.

**Risk 4: 27B VRAM per-tensor peak (4.8 GB).**

MI300X has 192 GB HBM3; 4.8 GB per tensor is fine. For 24 GB consumer cards (the RDNA path the CDNA gate already excludes), this would require tensor streaming. Phase 3 can add a chunked-W path for such hardware. This plan is explicitly MI300X-only for Phase 2, consistent with the `detect_is_cdna` gate at `main.rs:94-108` that already excludes RDNA from the GPU Cholesky path.

**Risk 5: Correctness of U submatrix indexing.**

Phase D stores U on device in column-major order matching rocSOLVER's Fortran convention (the `dgeam` transpose at gptq_hip.rs:246-251 converts from rocSOLVER's column-major L output to the row-of-U-for-step-j access pattern the Rust CPU loop uses). Phase 2 kernels must use the same memory layout assumption. Kernel 2 reads `U[step, k]` for `k >= step`; in the device buffer this is `d_u[step * k + k]` if stored row-major, or `d_u[k * k + step]` if column-major. The `dgeam` step in gptq_hip.rs:246-251 transposes into `d_u` with leading dimension K; the resulting `d_u[col * K + row]` layout is column-major. Kernel 2 must use `d_u[next_step * k_dim + step]` to read `U[step, next_step]` (column index = next_step, row index = step in column-major). This is a common indexing trap — unit tests in Phase 2B must explicitly verify against the CPU's `u[(step, next_step)]` access.

---

## §9 — Out of Scope

These are deferred to Phase 3 or later, and must not be included in Phase 2 implementation:

- **hipGraph capture** for the block loop. Would reduce 192 kernel launches to a single replay call, saving ~1 ms — negligible vs the OBS kernel wall at this scale, but valuable at B=16 or for 27B where per-block overhead grows. Phase 3.
- **Multi-GPU distribution** of the OBS loop across XGMI peers on MI300X (all 8 GCDs share HBM3 via infinity fabric, enabling a data-parallel block distribution). Phase 3.
- **Batching same-K tensors** (processing all K=4096 attention projections as a batched GEMM). This requires model-level orchestration outside `gptq_column_sequential`. Phase 3.
- **Adaptive block size** (B < 128 for small K, B > 128 for large K). B=128 is the paper's canonical choice and the right default; tuning adds complexity without clear gain at K >= 4096.
- **Streaming W in chunks** for consumer GPUs with < 8 GB VRAM. RDNA is already excluded from this path by the `detect_is_cdna` gate.

---

## §10 — Open Questions for the User

**Q1: BF16 cast-trick for Kernel 3, or stay pure FP64?**

Recommendation: implement both (FP64 in Phase 2C, BF16 in Phase 2D), ship BF16 if and only if the 0.1% codeword-divergence gate passes AND `coherence-gate.sh` passes on the resulting model. Document the wall delta (FP64 ~880 ms vs BF16 ~240 ms per ffn_down) in the commit. If BF16 fails the gate, ship FP64 — it is still a 14× end-to-end speedup over Phase D.

The primary concern from Frantar et al. is FP32 precision being insufficient for the **Cholesky solve** at K=12288 (gptq.rs:37-38). The OBS propagation (Kernel 3) is a rank-128 GEMM applied to the residual — the accumulated error per output element is bounded by the B=128 inner product, which has relative error ~B × BF16_ULP ≈ 128 × 1e-3 = 13%. Whether that 13% per-element error flips MQ4 codewords depends on the distribution of `err_block` magnitudes, which varies by tensor and calibration data. The empirical gate (§6) is the right arbiter.

**Q2: Should `HIPFIRE_GPTQ_HIP_OBS` be opt-in (separate from `--gptq-hip`) or auto-on when both rocSOLVER and the new kernels are available?**

Recommendation: opt-in via `HIPFIRE_GPTQ_HIP_OBS=1` for Phase 2E and 2F validation. After Phase 2F confirms end-to-end quality parity, discuss a default-flip: auto-on whenever `detect_is_cdna()` returns true (same predicate as Phase D). This matches the "selectable additions land freely; default-flip needs discussion" policy in `feedback_pr_gating_policy.md`. A separate `--gptq-hip-cholesky-only` flag for users who want Phase D but not Phase 2 is a reasonable escape hatch.

**Q3: For 27B specifically — chunk-stream W from CPU in blocks if VRAM is tight?**

For MI300X (192 GB), no streaming needed. If Phase 3 targets consumer CDNA (MI100 = 32 GB HBM2), ffn_down at 27B requires 4.8 GB — borderline. The architectural answer is to process W in horizontal M-slices: split the M=7168 rows into 2 halves, process each half independently with independent `d_W_residual`/`d_err_block` allocations. U is read-only and shared across slices. This halves peak VRAM per tensor at the cost of two passes over U. Defer to Phase 3; for now MI300X is the only deployment target and 4.8 GB is well within budget.

---

## Appendix A — Key Source Anchors

| Location | Role in Phase 2 |
|---|---|
| `gptq.rs:860-915` | CPU loop being replaced |
| `gptq.rs:869-890` | Phase A: quantize column + err computation (Kernel 1 target) |
| `gptq.rs:896-914` | Phase B: OBS propagation (Kernel 2 + Kernel 3 target) |
| `gptq.rs:92-101` | `quantize_mq4_element_with_clamp` — device function template for Kernel 1 |
| `gptq.rs:698-716` | `compute_frozen_block_grids` — grid layout for H2D copy |
| `gptq.rs:724-727` | `block_idx_for` — row×col → grid index formula, needed in Kernel 1 |
| `gptq.rs:737-744` | `weight_mode_actorder` — stays on CPU |
| `gptq_hip.rs:125-270` | `compute_damped_inv_cholesky_upper_hip` — extended by `_keep` variant |
| `gptq_hip.rs:246-251` | `dgeam` transpose: confirms d_u is column-major in device memory |
| `kernels/src/gemm_bf16_mfma.gfx942.hip` | BF16 MFMA tile template for Kernel 3 Phase 2D |
| `kernels/src/gemm_hfq4g256_residual_mfma_v3.gfx942.hip` | FP16 MFMA tile template; LDS B-tile pattern |
| `main.rs:88-108` | `GPTQ_SOLVER` + `detect_is_cdna` — unchanged, gates both Phase D and Phase 2 |
| `main.rs:4907-4910` | `gptq_solver_ref` dispatch — unchanged; Phase 2 wires inside `gptq_column_sequential` |

---

## Appendix B — Git Sequence

```bash
# In the gptq-hip-impl worktree:
cd /home/kaden/ClaudeCode/autorocm/hipfire/.claude/worktrees/gptq-hip-impl
git checkout -b feat/gptq-hip-phase2-plan
# Write this file to docs/plans/gptq-hip-phase2-obs-kernel.md
git add docs/plans/gptq-hip-phase2-obs-kernel.md
git commit -m "docs(gptq-hip): Phase 2 OBS GPU kernel plan (block-128 MFMA)"
```

---
