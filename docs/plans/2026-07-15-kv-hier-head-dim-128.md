# Scope: generalize the hierarchical KV cache to head_dim=128

Status: **scoping** (not started). Branch: `chaingun`. Date: 2026-07-15.
Companion: `docs/plans/2026-07-15-8bit-hot-ring-kv-hier.md` (the 8-bit hot tier,
landed; this doc removes the head_dim=256 ceiling that gates all of it).

## Why

The two-tier hierarchical KV cache (`HierKvState`, `HIPFIRE_KV_HIERARCHICAL=1`)
— and therefore the whole 8-bit hot-ring stack — is hard-gated to **head_dim==256**
(`kv_hier.rs:58 const HD = 256`, enable gate at `:254 head_dim == HD`). The FWHT-256
rotation and the two cold/merge attention kernels are compiled for CHD=256.

head_dim=128 is the rest of the world: Qwen3/Llama/Gemma are almost all 128. Within
the qwen35 arch itself, **Qwen3.5-122B-A10B is head_dim=128** (verified in its HF
config) while the 0.8b/9b are 256. So the concrete near-term beneficiary is the
122B MoE — exactly the model where thousands of batched sessions × a smaller hot
tier matters most. (Other 128 arches — llama/gemma/qwen3-base — additionally lack
the hier *hook*; that is a separate, larger integration and out of scope here. This
doc only removes the head_dim ceiling **within the qwen35 arch**.)

## What is ALREADY general (no work)

Verified by reading the kernels + CPU primitives:

- **`kv_hot_quant_q8` / `kv_hot_dequant_q8`** (the 8-bit hot ring, this project's own
  kernels) — take `head_dim` as a runtime arg, `block = min(256, head_dim)`. Work at
  128 unchanged.
- **`kvarn_quantize_tile` / `kvarn_dequant_tile`** (cold tier) — `r_dim`/`c_dim`
  runtime params; `r_dim=128` is within the `<=256` bound; block `[256]` is a thread
  count, not a head-dim. Work at 128 unchanged.
- **CPU FWHT stack** — `signed_fwht` and `gen_fwht_signs(seed, n)`
  (`hipfire-primitives/src/fwht.rs`) are fully power-of-two-general; only their *call
  sites* pass a literal 256.
- **FWHT-128 rotation kernel already exists**: `mq_rotate_x_128`
  (`gemv_mq4g128.hip`), orthonormal (butterfly + 1/√128), dispatched by
  `gpu.rotate_x_mq_128()` (`rope.rs:83`). No new rotation kernel needed — just select
  it by head_dim.

## Blockers (the actual work)

### K1 — `attention_cold_slots.hip`: CHD=256/CPL=8 compile-time (new 128 variant)
`#define CHD 256`, `#define CPL 8` (`= CHD/32`, one 32-lane wave per q-head, 8
elements/lane in `float q[CPL]/acc[CPL]/kk/vv` register arrays, `#pragma unroll`
over CPL, 32-lane `__shfl_xor` butterfly). At head_dim=128, CPL=4 — **structurally
identical**, only the per-lane element count + address arithmetic change; the 32-lane
wave and butterfly are unchanged. → **new `attention_cold_slots_128` variant**
(CHD=128, CPL=4). ~40 LOC clone.

### K2 — `flash_tier_merge.hip`: same CHD=256/CPL=8 (new 128 variant)
Same shape (per-head CPL register loop). → **new `flash_tier_merge_128`**. ~20 LOC.

**Variant vs runtime CHD:** prefer two compiled variants (the `mq_rotate_x` vs
`mq_rotate_x_128` precedent) over a single runtime-CHD kernel — the register arrays
and unroll are the whole point of the LDS-free gfx1103-safe design; a runtime bound
would size arrays to `[8]` and drop the unroll (perf regression on the decode hot
path). Two variants keep both dims optimal. Cost: SRC consts in `kernels.rs` +
dispatch fns that pick the variant by head_dim.

### R1 — `kv_hier.rs`: `const HD` → runtime `head_dim` field
`const HD: usize = 256` → a `head_dim: usize` field on `HierKvState`, set from the
`from_env` arg (which already receives it). ~8–10 sites; the audit shows ~60% are
pure sizing (`nkv*hb*HD`, `nkv*HD`, `kv_dim()`) that just deref the field, and the
256-specific ones are:
- **Rotation calls** (`append_token`/`two_tier_read`: `rotate_x_mq(.., nkv*HD)`) →
  select `rotate_x_mq` (256) vs `rotate_x_mq_128` (128) by head_dim.
- **Attention/merge dispatch** → select the CHD=256 vs `_128` kernel (K1/K2).
- **TriAttn bands** (`let n_bands = HD/2`) → `self.head_dim/2`. Optional feature;
  the sidecar must be calibrated at the model's head_dim (it already is).
- Enable gate `:254 head_dim == HD` → `head_dim == 256 || head_dim == 128`.

### C1 — `kv_compact.rs`: drop the 256 assert + parametrize signs
`assert_eq!(head_dim, 256, "KVarN v1 FWHT is 256-wide")` → remove; and
`gen_fwht_signs(42/1042, 256)` → `gen_fwht_signs(42/1042, head_dim)` (already
general). `signed_fwht`/Sinkhorn/`quantize_tile_qmax` are already head_dim-agnostic.
Low.

### Q1 — qwen35 caller: relax the two 256 guards
- `HierKvState::from_env` gate (via R1's enable relaxation).
- **Single-tier KVarN rotate** (`qwen35/mod.rs:3124 if kvarn_rotate &&
  config.head_dim == 256`, `rotate_x_mq_batched(.., nkv*head_dim, 1)`): the batched
  rotate hardcodes `k/256`. A `rotate_x_mq_128_batched` does **not** exist yet — only
  the non-batched `rotate_x_mq_128`. Either add a batched 128 variant or relax this
  guard carefully (see risk RQ below).

## Risks / open questions (resolve BEFORE coding)

- **RQ [RESOLVED 2026-07-15 — NO double-rotation]** Traced the live per-token decode
  path (`qwen35/mod.rs kv_cache_attention_dispatch`). The hier branch calls
  `append_token(&s.fa_k, ..)` then **`return`s `two_tier_read(..)` at :3105**, so the
  single-tier KVarN rotate at `:3132` (`rotate_x_mq_batched(&s.fa_k, .., in-place)`)
  is UNREACHABLE when hier is enabled. The two are mutually exclusive:
  - hier ON → only `append_token`'s FWHT runs, and it rotates into the `rot_k`/`q_rot`
    *scratch* (`fa_k`/`fa_q` are never mutated). One rotation.
  - hier OFF → only `:3132` runs (in-place). One rotation.
  Confirmed two ways: (1) control flow (early return), (2) exhaustive grep — the ONLY
  `s.fa_k`/`s.fa_q` rotation site in the qwen35 crate is `:3132–3133`; every other
  `rotate_x_mq*` call rotates activations (`x_norm`/`hidden`/`gate_up`) for the MQ4
  weight GEMMs, not the KV key. So the 128 port needs NO double-rotation coordination:
  just route `append_token`'s internal `rotate_x_mq` → `rotate_x_mq_128` by head_dim.
  Side fix: the `:3071` call-site comment ("NO KVarN rotation … fa_q consumed
  un-rotated") was stale post-Phase-0 (default int8 hot forces rotation) and is
  corrected in the same commit.
- **Validation model gap.** `parity_kv_hier` hardcodes `NH/NKV/HD=256` — parametrize
  it to also run HD=128 (cheap, synthetic; the strongest proof). But an *end-to-end*
  decode test needs a head_dim=128 **qwen35-arch** model — that is **Qwen3.5-122B-A10B**
  (huge; halo/medusa only, may need conversion to `.hfq`). Without it, 128 hier ships
  on parity-oracle evidence alone. Confirm the 122B artifact exists / is convertible
  before committing to end-to-end validation.
- **Kernel precompile / cache keys.** New `_128` kernels need `.hsaco`/hash sidecar
  entries and `ensure_kernel` cache keys (mirror the `mq_rotate_x_128` wiring).

## Phasing

0. **[DONE]** Resolve RQ — traced: no double-rotation (hier `return`s before the
   single-tier rotate; the two are mutually exclusive). Clears the gate.
1. **[DONE 2026-07-15]** R1: `HD` const → runtime `head_dim` field on `HierKvState`
   (256-only still; gate is now the literal `head_dim == 256`). Pure refactor,
   ~40 sites; `parity_kv_hier` byte-identical (q8 4.3e-5 / 5.1e-5, f16 5.2e-5,
   +defrag 5.1e-5 — same as pre-refactor). `Q8Ring::alloc` gained a `head_dim` arg.
2. **K1 + K2**: `attention_cold_slots_128` + `flash_tier_merge_128` kernels + SRC
   consts + dispatch-by-head_dim. Parity: extend `parity_kv_hier` to run HD=128.
3. **C1 + Q1**: cold-compaction sign parametrization + relax the qwen35 gates
   (incl. `rotate_x_mq_128_batched` if the single-tier path is in scope).
4. **End-to-end**: HD=128 decode on Qwen3.5-122B-A10B (halo/medusa) if available;
   else document parity-only coverage. Coherence gate.

## Effort

~2 new small kernels (K1/K2, ~60 LOC + consts/dispatch), one mechanical const→field
refactor (R1), and ~3 low-risk relaxations (C1/Q1). Real cost is in the **validation**
(parity parametrization is easy; a 122B end-to-end run is not) and in **resolving RQ**
first. Estimate: 2–3 focused days for code; validation depends on 122B availability.
No new algorithms — the FWHT-128 rotation and the parametric cold/q8 kernels already
exist; this is dimension-plumbing + two kernel clones.
