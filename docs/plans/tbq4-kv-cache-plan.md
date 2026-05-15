# TBQ4 KV-cache quantization for hipfire

> **Branch:** `feat/tbq` — seed commit only. No implementation yet.
> **Status:** OPEN. Pick this up cold; everything you need is in this doc.
> **Reference impl:** `DrBearJew/llama.cpp@tbq4-rdna3-experiment` (MIT). Survey at the bottom.

## What it is

TurboQuant 4-bit (**TBQ4**) is a KV-cache quantization format. Pre-RoPE Q/K
vectors are rotated by a signed Fast Walsh-Hadamard Transform (FWHT) into a
flatter "domain" where 4-bit centroid quantization preserves attention
fidelity better than naive int4. Concrete shape from the reference:

- **Block size:** 128 elements (one block ≈ one head × head-dim chunk)
- **Per-block storage:** L2 norm (1× FP16 = 2 B) + 128 packed 4-bit centroid
  indices (64 B) + small overhead → ~68 B / block = **4.25 bits per value**
- **Domain:** signed-FWHT (Hadamard) rotation; same family as our existing
  `MQ4G256` weight quantizer (which also uses signed FWHT, group=256)
- **Drop-in scope:** KV cache only (`-ctk tbq4_0 -ctv tbq4_0` in llama.cpp).
  Does NOT touch lm_head, weights, or activation paths.

llama.cpp registers TBQ4 as a `ggml_type` (`tbq4_0`) so any model that
selects TBQ4 KV gets the savings without retraining.

## Why hipfire wants it

Today's KV-cache quant on hipfire is **asym3** (3-bit asymmetric, ~3 bpv) —
already excellent compression but pre-RoPE Q/K reconstruction adds compute.
TBQ4 trade-offs vs asym3:

- **Higher bpv** (4.25 vs 3.0) — 41% bigger KV cache footprint
- **No de-rotation at scoring time** — pre-RoPE storage means K is rotated
  ONCE at write, FWHT is involution-ish (signed-FWHT inverse = transpose
  with flipped signs), so attention can score directly in the rotated
  domain
- **Better PPL preservation** at long context — Hadamard rotation flattens
  outlier distributions that hurt int4 rounding

For RDNA3 (gfx1100, 7900 XTX) the reference reports 38.6 tok/s decode on
Qwen3.6-27B + MTP at 64k context with TBQ4 KV — comparable to asym3 territory
with strictly better quality at long ctx (their claim, not yet verified).

## Where it lives in hipfire

KV cache code is in:
- `crates/rdna-compute/src/kvcache.rs` — KV alloc + write paths (per-arch)
- `crates/rdna-compute/src/triattn.rs` — TriAttention sidecar (orthogonal,
  doesn't touch quant format directly)
- `crates/rdna-compute/src/dispatch.rs` — kernel dispatch tables for
  per-format K/V write + scoring kernels
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — model uses `Mode::Asym3`
  (search `Mode::` enum for the existing format zoo)
- `kernels/triattn_score_asym3.hip` and friends — HIP kernels per format
- `crates/hipfire-runtime/src/lib.rs` — `--kv-mode` flag plumbing

Pattern to follow: look at how **asym3** is wired end-to-end. Add a parallel
`tbq4` mode in the same shape: format enum variant, kernel files (write +
score), dispatch arms, CLI flag value, daemon config field.

## First task (smoke gate, ~1 day)

1. **Implement TBQ4 K-write kernel** at `kernels/tbq4_write.hip`:
   - Input: pre-RoPE K vector `[head_dim]` F32
   - Per 128-element block: signed-FWHT rotate, compute L2 norm,
     normalize, find 4-bit centroid index per element, pack to 64 B
   - Output: 68 B per block (FP16 L2 + 64 B packed)
2. **Implement TBQ4 attention scoring kernel** at `kernels/tbq4_score.hip`:
   - Input: F32 query (already RoPE-rotated to position p_q) + tbq4_0 K
     blocks at positions [0..p_k]
   - Per block: dequantize on-the-fly (FWHT inverse + L2 scale + centroid
     lookup) and compute Q·K dot. Or do attention scoring in the rotated
     domain directly if both Q and K are pre-rotated.
3. **Wire into asym3-shaped `Mode::Tbq4` enum variant.** Don't add to the
   default fallthrough — gate behind explicit `--kv-mode tbq4` selection.
4. **Smoke test:** modify `coherence_probe` (in
   `crates/hipfire-runtime/examples/`) to run canonical 27B-3.5 LRU PEP-8
   prompt with `--kv-mode tbq4 --max-n 1 --temp 0.0`. Coherent code out
   = format is shipped. Compare PPL vs asym3 on a small longctx sample.

## Constraints

- **CLAUDE.md applies.** No Python in inference path. MQ4 default for
  weights still — TBQ4 is KV-format only, doesn't change weight quant.
- **Must respect** the existing TriAttention sidecar interface
  (`crates/rdna-compute/src/triattn.rs`). TBQ4 is at the same layer as
  asym3 — TriAttention scoring should see TBQ4 K through the same
  abstraction (or you add a TBQ4-aware path to TriAttention's scoring
  kernel).
- **Coherence-gate** required before commit per `CLAUDE.md`'s coherence
  protocol. KV-format changes are exactly the kind of thing the
  `./scripts/coherence-gate.sh` battery catches.
- Reuse existing FWHT machinery if possible — `kernels/mq_*.hip` already
  has signed-FWHT for the MQ family.

## Reference

- DrBearJew fork README: <https://raw.githubusercontent.com/DrBearJew/llama.cpp/tbq4-rdna3-experiment/README.md>
- llama.cpp ggml type registration: search the fork for `tbq4_0`
- Theory: signed-FWHT for outlier suppression is the same trick the
  MQ-family uses on weights; this paper applies it to KV.

## Bench gate

Match or beat asym3 on:
- decode tok/s on canonical 27B-3.5 LRU bench
- coherence-gate full battery
- longctx PPL on a 16k-32k prompt (this is where TBQ4 should pull ahead)

If TBQ4 doesn't clear all three, document the regression and close the
branch as a falsified-experiment — same disposition as the prior FP8 /
gfx11-fdot2 attempts (see MEMORY.md falsified entries for the format).

## Sister branch

`feat/rtq` ships RotorQuant — same KV-cache scope, different rotation
(2D-Givens / 4D-quaternion instead of FWHT). That branch can land first if
its kernels are simpler; both branches stay independent and bench-comparable.
