# Gemma 4 Phase 4 mathematical primitives and lowered dense forward

Date: 2026-07-15. Status: passed.

## Exit-gate evidence

The generic partial half-split RoPE ABI now carries `basis_dim` independently
from `rotary_dim`. Every existing MiniMax and Qwen 3.5 caller passes its former
denominator explicitly, while Gemma 4 can lower proportional RoPE with a full
head-width basis. The operator-golden run on gfx1103 passed full/local RoPE,
proportional RoPE, weighted Q/K RMSNorm, weightless V RMSNorm, query scaling,
the learned layer scalar, and final vector softcap:

```text
$ hipfire lock run gemma4-phase4-operators -- \
    cargo run -q -p hipfire-runtime --example gemma4_operator_goldens
gemma4_operator_goldens: PASS
```

The example contains an output captured before the ABI change and compares the
raw F32 bits after the change. Qwen's half-split partial-RoPE output remained
byte-identical when `basis_dim == rotary_dim`.

The dense oracle and lowered executor share one immutable weight set and use
independent state arenas. A ten-token mixed local/global run crosses the
eight-token SWA ring boundary at positions 7, 8, and 9 and captures every layer
boundary and final logits on every token:

```text
$ hipfire lock run gemma4-phase4-final -- \
    cargo run -q -p hipfire-arch-gemma4 \
      --example dense_reference_lowered_parity
GPU: gfx1103
layer 0: max_abs=0.00000000
layer 1: max_abs=0.00000000
logits: max_abs=0.00000000
dense_reference_lowered_parity: PASS
```

After this exact dual-run result, the dense API was flipped to the lowered path
by default. `HIPFIRE_GEMMA4_FORWARD_ORACLE=1` retains the straightforward
reference path as an explicit diagnostic opt-out.

The changed HIP sources compiled directly with `hipcc --genco` for gfx1030
(RDNA2), gfx1100 (RDNA3), gfx1151 (RDNA3.5), and gfx1200 (RDNA4): 24 of 24
source/target builds passed. This is cross-target compile evidence; runtime
execution was performed only on the local gfx1103 GPU.

The targeted CPU suites passed after the final typed-plan validation:

```text
$ cargo test -p hipfire-dispatch
test result: ok. 144 passed; 0 failed

$ cargo test -p hipfire-arch-gemma4 --lib
test result: ok. 6 passed; 0 failed

$ cargo test -p hipfire-runtime --lib
test result: ok. 167 passed; 0 failed; 4 ignored
```

The canonical kernel correctness wrapper exited zero and wrote
`/tmp/coherence-dflash-20260715-052855.md`. Its exact result was:

```text
27b-dflash-prose: OK
27b-dflash-code: WARN (paragraph-level repetition; hard_fails=0, soft_warns=1)
27b-ddtree-b12-prose: OK
27b-ddtree-b12-code: OK
```

The warning is recorded rather than promoted to a clean result. There were no
hard errors.

The new final softcap is a generic `vector_softcap_f32(x, out, n, cap)` kernel
with no LDS allocation. Weightless RMSNorm is explicit rather than represented
by a synthetic weight tensor; its reduction uses the same normal RMSNorm
reduction structure and is not a pointwise operation.

## Lowered representation

- `RopeFlavor::PartialHalfRotate` carries both the rotated width and basis.
- Every lowered attention op carries exact Q/KV/head geometry and the resolved
  physical cache group, slot, and producer. The executor compares the entire
  lowered flavor against the live resolved layer plan before running it.
- Learned scale, final softcap, and the reserved later PLE operation are regular
  super-ops. No routine Gemma 4 operation was added as an `EscapeKind`.
- The dense implementation supports separate V and Gemma's K=V pre-norm copy,
  weighted per-head Q/K norm, weightless V norm, query compensation,
  proportional RoPE, full attention, SWA ring staging/attention, four norm and
  residual points, GeGLU, the learned layer scalar, final norm/head, and final
  softcap.

## Reuse and cleanup ledger

- Existing primitives reused: the current RMSNorm, scale, residual GEMV,
  GeGLU, full-attention, SWA ring write/stage/attention, RoPE launch machinery,
  `LayeredKvArena`, and common transformer weight loader.
- Duplicate removed or retained: no Gemma-specific copies of attention, GeGLU,
  scale, residual, or softcap launch policy were introduced. The readable
  reference path is deliberately retained as the oracle.
- Generic seam added or changed: partial RoPE gains an explicit basis; generic
  weightless RMSNorm and vector softcap APIs are available to all architectures;
  the lowered program gains exact attention geometry/cache binding plus regular
  act, residual, scale, softcap, and future PLE kinds.
- Generic abstraction consumers: all prior partial-RoPE callers use the new ABI;
  Gemma 4 consumes the mixed layered-cache and typed-lowering seams.
- Stale assumption removed: partial RoPE no longer assumes that the number of
  rotated coordinates is also the frequency exponent denominator, and lowered
  attention no longer assumes homogeneous geometry or an implicit cache slot.
- Oracle retained: the frozen pre-change Qwen bytes, the Gemma 4 readable
  forward selected by `HIPFIRE_GEMMA4_FORWARD_ORACLE=1`, and the existing
  DFlash coherence wrapper remain independent regression anchors.
