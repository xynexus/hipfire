# Scope: Opus support in `is_batchable_la` — the Q8-removal prerequisite

Status: scope, 2026-08-27. Driver: retiring `Q8_0` from model weights, re-encoding
those tensors as `oq8++` (Q8 ≈ oq6, so same cost) or `oq4.25` (less cost, less
accuracy).

**This must land before Q8 is removed, or batched prefill dies silently.**

---

## 1. Why this blocks the migration

Two predicates decide whether a layer may batch, and they are **inverses**:

| predicate | gates | `Q8_0` | Opus |
|---|---|---|---|
| `is_batchable_la` | attention projections | ✅ | ❌ |
| `moe_prefill_quant_family` ladder | MoE FFN | ❌ | ✅ |

Today's oq4 MoE artifacts batch **only** because attention is `Q8_0` and the MoE
side is Opus. Measured on the real `Qwen3.6-35B-A3B--oq4.hfq`: `wqkv=Q8_0`, 271
`Q8_0` tensors of 21,094 — the attention projection set plus `shared_expert_gate`
and `lm_head`.

Re-encode those to Opus and attention stops being admissible. The forward drops
to the per-token path — **4.2 tok/s**, measured — and it does so *silently*,
because a decline is not an error.

The inverse is already observable. A tiny fixture whose `shared_expert.down_proj`
is `Q8_0` declines for the opposite reason, now visible with the trace fix:

```
3x qwen35 moe_prefill_quant_family ladder `_` arm -> dtype not batched-admissible
   [Q8_0 arch=gfx1103]
3x qwen35 moe_ffn_batched DECLINED (dtype unsupported on this arch)
   [shared_gate_up=Oq4G256/true shared_down=Q8_0/false routed=Uniform(Oq4G256)/true]
```

So a half-migrated artifact can decline on *either* side. Both predicates have to
accept the target encoding simultaneously.

## 2. The rotation question — mostly already answered

Opus weights are FWHT(+AWQ)-rotated offline, so the activation must be rotated to
match. The exclusion in `is_batchable_la` is justified by a comment in
`prefill_chunk.rs` warning that an unrotated `x` reaching an Opus GEMM produces
"garbage: PPL 3.5e6".

**That comment predates the current code.** It sits in a block that also says the
LA body "used to be ~790 lines of hand-rolled dtype dispatch" before being folded
onto the shared lowered super-ops. Those super-ops carry the machinery:

- `prefill_lowered.rs` has **18 `Oq4G256`, 18 `Oq8G256`, 17 `OqCompactG256`** arms
  and 31 `FWHT` / 5 `rotate_x` references;
- the `moe_prefill_quant_family` ladder admits all three Opus dtypes on RDNA
  *specifically because* "both attention arms rotate for it (the
  `is_mq`/`qkv_is_mq` predicates)".

So the codebase already asserts, in the MoE path, that the attention arms rotate
for Opus.

### ⚠️ RETRACTED: there is no `is_mq` / `qkv_is_mq` gap

An earlier revision of this scope claimed `is_mq` covered `Oq4G256` only while
`qkv_is_mq` covered all three, made that "step 1", and called it "small, and
everything else depends on it". **That was wrong — I truncated the `sed` window
and reported the truncation as a finding.** Parsed in full, the two are
identical:

```
qkv_is_mq (8): MFP4G32 MQ3G256 MQ3G256Lloyd MQ4G256 MQ6G256 Oq4G256 Oq8G256 OqCompactG256
is_mq     (8): MFP4G32 MQ3G256 MQ3G256Lloyd MQ4G256 MQ6G256 Oq4G256 Oq8G256 OqCompactG256
```

(Same class of error as the `head -8` that made an earlier panic-count look like
it had grown. A truncated view of a list is not a finding about the list.)

**This makes the scope easier, not harder.** The rotation support for all three
Opus dtypes is already complete on both attention paths — there is no
prerequisite to fix before widening `is_batchable_la`.

It also locates the "garbage: PPL 3.5e6" number that justifies the exclusion. It
is not a general claim about Opus attention; it is the comment recording a
historical bug in *these very predicates*:

> *"Opus W8A8 needs the SAME FWHT rotation as W4A4 — its weights are rotated
> offline too. Omitting it here fed the oq8 GEMM an unrotated activation
> (garbage: PPL 3.5e6)."*

That omission was already fixed — which is why both predicates list `Oq8G256`
today. So the failure the `is_batchable_la` exclusion cites has **already been
repaired at the layer that actually rotates**, strengthening the case that the
exclusion is vestigial.

## 3. Widening `is_batchable_la` alone is NOT sufficient

Measured. With a temporary flag making `is_batchable_la` accept all three Opus
dtypes, an all-Opus fixture **still** declined (`rc=3`, batched prefill did not
execute). The flag was reverted; it demonstrated nothing.

Two reasons, both now visible:

1. that fixture's LA projections were already `Q8_0`, so the predicate was never
   the blocker for it — the experiment tested a gate that was not in the way;
2. the MoE ladder refused `shared_down=Q8_0` independently.

So the work is **both predicates plus their downstream arms**, not a one-line
`matches!` edit. Any plan that budgets for the latter is wrong.

## 4. Verification — and a warning against assuming benignity

Batched-vs-per-token divergence on this arch is **real and already measured**, so
"it's just accumulation order" cannot be assumed:

| configuration | vs per-token reference |
|---|---|
| MoE path-1, KVarN KV | DIFFER (shared prefix 255 of 433) |
| MoE path-1, **q8 KV** | **DIFFER** (shared prefix 221 of 417) |

The q8 row matters. The code warns that under KVarN the per-token fallback "is
MEASURED to emit a different token stream than the batched path … while f32 and
q8 KV agree between the two". Re-running under q8 removes that confound and the
divergence persists — so it is the batched arm's own, not a KV artifact. Any
Opus-in-attention change must clear the same bar rather than inherit the
assumption.

**Required evidence to land:**

- `tiny-prefill-gate` KLD parity for an artifact with Opus attention, with the
  corrupt-prefix self-check firing (a cell that cannot fail is not evidence);
- the check run under **q8 or f32 KV**, not KVarN, for the reason above;
- a positive path probe — the batched counter must move — since a decline
  produces a clean pass for a path that never ran.

Note the fixture problem from
`docs/todo/2026-08-27-oq4-moe-prefill-coverage.md`: the tiny quantizer emits
`Q8_0` attention for `--format oq4`, so a fixture with Opus attention does not
exist yet and has to be produced first.

## 5. Order of work

1. ~~Resolve the `is_mq` / `qkv_is_mq` asymmetry~~ — **retracted, no gap exists**
   (§2). Both predicates already cover all three Opus dtypes.
2. **Produce a fixture with Opus attention.** ⚠️ **Corrected**: this does NOT
   need new per-tensor format control. `--tensor-format <GLOB=FMT>` already
   exists, is repeatable, and has `--tensor-source` / `--copy-untargeted`
   companions. Used on the tiny fixture it does put Opus on the attention
   projections (`in_proj_qkv`/`in_proj_z`/`out_proj`/`k_proj`/`o_proj` → qt=35
   Oq8G256, Q8_0 count 11 → 4).

   What blocks it instead is a **bug it exposed**: the requant path ignores the
   `K % 256` fallback the direct path applies, so the K=128 `in_proj_a`/`in_proj_b`
   get `Oq4G256` and the artifact panics at load with "OQ4G256 requires
   K % 256 == 0 (got K=128)". Filed in BUGS.md. Either fix that guard, or give
   the fixture dims ≥256 on every quantized tensor.
3. **Widen `is_batchable_la`** to the Opus dtypes the rotation predicates
   actually cover.
4. **Widen the MoE ladder** to accept whatever the shared-expert tensors become,
   so a half-migrated artifact does not decline on the other side.
5. **Gate it** (§4), then migrate the quantizer's Q8 output to `oq8++`/`oq4.25`.
6. **Delete `HIPFIRE_MOE_OQ4_UNIFORM_PATH1`** once path-1 parity is established
   by the same gate — it is opt-in today precisely because that evidence is
   missing.

Step 2 is the only prerequisite and can start now. **Re-encoding Q8 (step 5)
must not land before step 3**, or every oq4 MoE artifact loses batched prefill
with no error to point at.

## 6. What made this findable

The declines were recorded all along but never printed: `kernel_trace::report()`
returned `None` whenever no kernels had dispatched, which is always true of an
admission decline. Fixed in `fix-kernel-trace-predispatch-declines`. Before that,
`[pbs-gate]` printed `verdict=false` with every *named* term reading true —
recomputed proxies, not the deciding terms — which is exactly the failure its own
comment ("Name every term") warns against.
