# The v2 daemon: module-major execution, multi-stream cohabitation, deadline admission

Status: scope, 2026-08-09. Supersedes stages M4d and M5–M8 of
`docs/plans/2026-07-25-daemon-merge-training-induction-scheduler.md`; M0–M4c of that
document stand as landed history and its §1.2 ("the executor stays serial —
deliberately") is still correct and is assumed here rather than restated.

**All line references in this document are against `origin/master` at `da38cc16f`** and
were verified there, not on a feature branch. Where in-flight branch work is assumed,
that is called out explicitly (see §0.5).

Six scoping decisions were taken by the user on 2026-08-09 and are treated as settled
premises, not open options:

1. **A new branch replaces the core of `hipfire-daemon`.** Breaking the existing daemon
   and rebuilding features back up is explicitly allowed. **No backward compatibility is
   required** — which retires M2b (typed `DaemonResponse` serialization) as a
   compatibility problem and lets the wire protocol be redesigned once rather than
   migrated.
2. **The scheduler and the HTTP service move into the daemon process.**
3. **Priority order is latency > throughput > capacity.** The headline goal is one GPU
   serving low-latency work while training, quantizing, embedding, eval, calibration and
   image generation run underneath it.
4. **One forward/backward system for all workloads**, built as one module graph with two
   numeric bodies per module — *not* by differentiating the fused inference kernels.
5. **The realtime guarantee is admission-controlled and entry-latency-bounded**, not a
   per-frame deadline. See §2; this is the decision that most shapes the scope.
6. **First target is qwen3.5 MoE** (`crates/hipfire-arch-qwen35`, arch id 6). TTS, STT
   and realtime video in/out are **stubbed** — declared workload classes with synthetic
   executors, no models.

## Context

hipfire owns one GPU per box and rations it by serializing whole requests.
`hipfire-daemon` is a single-threaded executor whose smallest quantum is one protocol
frame: a `Generate` runs prompt→EOS holding `&mut Gpu`
(`hipfire-daemon/src/main.rs:1293-1294`), a `Calibrate` runs a whole layer, a `TrainLora`
a whole micro-step. The only real preemption boundary in the system — one fused
multi-session decode step with a LIFO parked stack — lives *outside* the GPU-owning
process, in `hipfire-server/src/batch_runner.rs:923`, so every preemption decision costs
a process round trip. Image generation is worse: it preempts at a sampler step by
setting an `AtomicBool` and **restarting from the seed**, discarding the work done.

So hipfire cannot do the thing being asked for. Low-priority workloads either take the
GPU lock and block inference outright, or are admitted at a priority that controls
*ordering* but never *interruption*.

The v2 daemon's premise is a single unit — the **module** — playing three roles:

1. the **residency and eviction unit** (one expert of one expert layer);
2. the **suspension and preemption quantum**;
3. the **node of one unified forward/backward tape** shared by inference, training,
   calibration, eval, embedding and image generation.

Role 2 is why the other two matter. Modules are the means; latency is the point.

---

# Part 0 — Findings that contradict the naive reading

Recorded first because four of the five change the ordering.

## 0.1 The lowered super-op substrate is already LIVE and default-ON

This was the decisive fact, and two independent design passes disagreed on it, so it was
verified directly. `hipfire-arch-qwen35/src/qwen35/lowered.rs:638`:

```rust
*F.get_or_init(|| std::env::var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() != Some("0"))
```

Default **on**, opt **out**, since 2026-06-07 — the doc comment above it records
validation "byte-identical to the hand path via fleet decode byte-parity (RDNA3 / RDNA4 /
RDNA3.5, dense + MoE) and the full coherence battery." The gate is consumed at
`decode_layers.rs:42`. The same pattern is live in `hipfire-arch-qwen2`,
`hipfire-arch-deepseek4`, `hipfire-arch-minimax` and `hipfire-arch-lfm2moe`.

**Eight doc comments across five files say the opposite** — the substrate header
(`superop.rs:34`, "nothing here is on a live path yet … default off"), the doc on
`run_layer_program` itself (`superop.rs:531`, "It is NOT on any live path"), and one or
two per arch in `qwen35/decode_layers.rs`, `qwen2.rs`, `minimax/forward.rs` and
`lfm2moe/forward.rs`. Every one is contradicted by the gate body directly below it, and
minimax's contradicts itself two lines later ("Default off (opt-in)" then "Lowered is the
default fast path"). Only deepseek4's comments were already correct, and they are the
model the rest were rewritten to. One of the two design passes read these and concluded
the substrate was unproven, which would have added an entire de-risking stage this plan
does not need.

**Fixed 2026-08-09 in the commit following this doc** — comment-only, no behavior change.
The original count in this section said "two"; a sweep found eight, and that is the
finding: the stale claim propagated by copy-paste as each arch was flipped.

Why it matters: **the lowered forward is the only place in hipfire where a forward pass
is data rather than control flow.** `LoweredForward = Vec<LayerProgram>` of POD `SuperOp`s
carrying `WeightSlot`/`ScratchSlot` indices is exactly the representation a module-major
executor needs. You cannot reorder a `for layer_idx in 0..n` loop across streams; you can
reorder a cursor into a `Vec<SuperOp>`. Building v2 on anything else means rewriting the
~63k lines of `hipfire-arch-qwen35`.

Caveat that costs a stage: **only decode is lowered.** Prefill (`prefill_chunk.rs` 8517 L,
`prefill_batch.rs` 5694 L) is hand-written control flow. And there are four escapes back
to the hand path at `decode_layers.rs:42-49` — hidden-state ring buffer, GDN tape
capture, `HIPFIRE_RQ_HAND=1`, and `hipfire_steer::is_active()`.

## 0.2 `SuperOpKind::Moe` is one op for a whole MoE layer, not per-expert

`superop.rs:183`. The substrate exists but its granularity is the layer. Splitting `Moe`
into `MoeRoute` / `MoeExpert(e)` / `MoeCombine` *is* the "make the module the quantum"
work — one stage, not a rewrite. It is also **the only super-op whose duration is
unbounded and data-dependent**: at decode with one stream it touches at most `top_k`
experts; at a 512-token prefill it can touch all of them.

## 0.3 The residency unit, the dispatch unit and the preemption point must be three different things

At ROCm's ~8 µs kernel-launch overhead and gfx1103's ~100 GB/s effective UMA bandwidth,
one routed expert of a 35B-A3B-class MQ4 artifact (1.59 MiB, from the archived
`chaingun-moe-module-layout.md`) reads in **16.7 µs**. Launch overhead is therefore ~32 %
of a per-expert dispatch, and a full sweep is worse:

| regime | expert-dispatches / step | launch overhead | weight read |
|---|---|---|---|
| decode, 1 stream, 48 L × top-8 | 384 | 3.1 ms | 6.4 ms |
| saturated sweep, 48 L × 128 exp | 6144 | **49 ms** | 102 ms |
| the 10 × 512 shape in the original ask | 5120 | **41 ms** | 85 ms |

A module is the unit of *addressing and residency* — one expert. Execution groups a run
of co-resident modules into **one** launch through the device pointer tables
`qwen35/prefill_batch.rs` already builds. Preemption happens *between launch groups*.
Reading "module-granular preemption" as "one kernel per module" builds something ~50 %
slower than today.

Corollary, and it must be a declared policy rather than a later discovery:
**module-major loses at low stream counts.** At N=1–4 it replaces one
`gemv_*_indexed_batched` launch covering all touched experts with one launch per expert —
8–32× more launches for identical FLOPs and identical bytes. The executor needs a switch
that coalesces `MoeExpert(e)` back into a fused `Moe` when residency is trivially
satisfied.

## 0.4 The amortization curve saturates fast, and it is the whole capacity argument

Distinct experts touched per layer under uniform routing — pessimistic, since real
routing is skewed and the true curve is better at low slot counts:

| token-slots in flight | distinct experts (of 512, top-8) | weight bytes / token-slot |
|---|---|---|
| 1 | 8 (1.6 %) | 8.00 |
| 8 | 61 (11.8 %) | 7.58 |
| 32 | 203 (39.6 %) | 6.33 |
| 64 | 325 (63.5 %) | 5.08 |
| 128 | 444 (86.7 %) | 3.47 |
| 256 | 503 (98.2 %) | 1.96 |

Past saturation the cost is `n_exp / slots`. This yields a computable admission rule
rather than a tunable: **above roughly `n_exp / k` token-slots, sweep the whole expert
layer in residency order — sequential, prefetchable, no routing lookahead needed; below
it, page only what routes.**

Be honest about what is new. The amortization is the classic MoE batching argument and
does not require module-major execution. What module-major adds is (a) the working set is
bounded to one launch group instead of the whole layer, decoupling capacity from `n_exp`,
and (b) a suspension point every ~1.6 MiB of weights. **If the N at which the curve pays
does not fit in VRAM on the target box, the capacity half of the thesis does not pay on
that box** — report that, do not tune around it.

## 0.6 Paged residency and OQ4 routed experts do not currently meet

Measured on gfx1103 with the artifact above. Two distinct problems, and the second is a
v2 concern in its own right.

**(a) Full residency OOMs; paging works, but paged OQ4 decode is refused.**

| load mode | result |
|---|---|
| pinned (default) | **OOM** — `hipMalloc: out of memory`, 15 MB free of 43,008 MB, for a 19.1 GB artifact |
| paged (`HIPFIRE_QWEN35_PAGED_EXPERTS=1`, 8 GiB cache) | **loaded in 10.6 s**, streaming only 2.54 GiB of a 17.77 GiB payload |

The pinned OOM is the allocator overhead `moe-expert-residency-unification.md` already
records — "20,480 BOs on a 256-expert/40-layer artifact = 4.35 GB pure allocator overhead"
is *this exact model shape*. So the capacity half of the v2 thesis is demonstrable on this
box: the model does not fit pinned and does fit paged.

But generation then fails with `moe.decode-routed-dtype-unsupported-no-fallback`
(`hipfire-dispatch/src/pipeline/mod.rs:220`). The refusal is
`!use_gpu_topk && !routed_experts_resident`, and under paged residency
`routed_experts_resident` is **false by design** — the doc at `:196-197` says only the
GPU-top-K path is available. So paging *requires* a GPU-top-K-indexable routed dtype.
Traced further, and **there are two causes, not one**:

**Cause 1 — the indexed OQ decode kernels are disabled because they are numerically
broken.** `routed_dtype_indexable_oq4 = oq_indexed_decode && …`
(`qwen35/mod.rs:1774`), and `oq_indexed_decode` is
`qwen35_moe_oq_indexed_decode_enabled()` (`:1824`), whose own comment says:
*"`HIPFIRE_QWEN35_MOE_OQ_INDEXED=1` re-enables the experimental indexed routed OQ decode
kernels **while debugging their finite-KLD failure**."* So this is not "experimental,
opt-in" — the premier quant family's indexed routed decode is switched off because it
produces wrong output, and nothing else can serve paged residency.

**This is the sharpest instance yet of the pattern already recorded for the fused dense
batch path: the fast paths admit deprecated and special-use formats and exclude the premier
one.** Here the premier path exists, is wired, and is disabled pending a numerics fix.

**Cause 2 — setting the flag is not sufficient.** With
`HIPFIRE_QWEN35_MOE_OQ_INDEXED=1` the refusal is unchanged, so dtype resolution under paged
residency is failing independently. The suspect is `MoePrefillDtypes::from_ffn`
(`qwen35/mod.rs:1644`): with `ffn.experts` empty it takes a branch requiring
`ffn.expert_gate_up_dtype`/`expert_down_dtype` to be `Some`, and the pager's
`oq4_canonical_to_moe_blocks` repack may present a different `DType` than `Oq4G256`
anyway. **Resolving which of those it is, is where the next session starts** — it is a
two-line instrumentation question, not a design question.

**Consequence for sequencing.** M4 is *dispatch* work and does not need paging, so it can
proceed against a pinned artifact on a box with a large enough carveout (halo, 128 GB), or
against a smaller MoE here. M5 on this box needs cause 1 fixed. Do **not** requantize to
`mq4` to get moving: magnum is deprecated, and routing around the gap buys a measurement on
a dying format while leaving the premier path broken.

**(c) Paged Opus is CPU-bound — but NOT on the host repack. That attribution is
RETRACTED.**

The original claim here was that `module_requires_host_repack` (`weight_pager.rs:648`),
true for exactly `Oq4G256 | Oq8G256 | OqPlusCompact`, put an `oq4_canonical_to_moe_blocks`
transform on every page-in and that this was the ~96 %-of-one-core cost. That was inferred
from the predicate existing, not measured, and it is **wrong**.

Tested by building the pre-transformed artifact (qt 53, verified: 10,240 modules,
1,622,016 B each) and running both through the same paged configuration. The packed
artifact does **no** repack — and is still pegged at **99.6 % of one core with the GPU at
0 %**, 10 minutes of CPU for a single-token generation. Identical symptom to canonical.
So the cost is common to both formats and is somewhere else in the paged path.

What this does and does not invalidate:
- The **qt 53 format still stands on its own merits** — it removes a real transform, and
  storage that matches the consuming layout is right regardless. It is simply not the fix
  for this symptom.
- **The "prefetch must also repack" note is withdrawn** along with the attribution.
- **What is actually eating the CPU is unknown.** Candidates that are common to both
  formats: `touch_module_lru`'s O(n) `iter().position()` + `VecDeque::remove` over 10,240
  entries (§1.6 already flags it), `would_fit_expert_module_set`, the per-layer
  `patch_expert_module_ptr_table`, or the per-MoE-layer D2H top-k readback. **Do not guess
  again — profile.**
- **Profiling is currently blocked on this box:** `perf_event_paranoid=4` refuses
  `perf record`, and `ptrace_scope` refuses `gdb`/`eu-stack` attach ("Operation not
  permitted"). Either relax one of those, or add timing instrumentation to the pager the
  way `HIPFIRE_QWEN35_MOE_DTYPE_DEBUG` was added — that env-gated trace is what located the
  dtype bug after two wrong hypotheses.

The methodological point, since this is the third time the same shape has bitten in this
work: a predicate that *could* explain a symptom is not evidence that it *does*. The
earlier `moe.decode-routed-dtype-unsupported` chase went the same way — two confident wrong
causes before an instrumented run gave the real one in a single line. `module_requires_host_repack` (`weight_pager.rs:648`) returns true
for exactly `Oq4G256 | Oq8G256 | OqPlusCompact`; magnum and the rest take the verbatim
`transport.fetch` path. So every Opus expert page-in runs
`oq4_canonical_to_moe_blocks` on the CPU, single-threaded, synchronously inside the
dispatch — 10,240 modules x 1.52 MiB on the M4 artifact. Observed directly: with the
refusal fixed, the daemon sits at ~96 % of ONE core with ~415 MB RSS while the GPU idles.

### 0.6.1 The fix is an artifact format, not a runtime workaround

Scoped 2026-08-09 at the user's direction: the repack exists because the artifact is not
stored in the layout the paged path consumes. Store it pre-transformed and page-in becomes
a verbatim `transport.fetch`. This supersedes the "prefetch must also repack" note above —
prefetch then only has to move bytes.

**New quant code: `Oq4G256MoeBlocks = 53`** (52 is the highest in use). Document it exactly
as `Oq4G256ArchPacked = 37` is documented, because it is the same idea one layout over:
*the quant_type code IS the layout version*, so a stale derived artifact is refused at load
rather than read as garbage, and the byte length is validated against
`oq4_moe_packed_len` instead of the 130 B/group canonical form. `Oq8G256` and
`OqPlusCompact` want siblings later — `module_requires_host_repack` names all three.

**Consumer — LANDED 2026-08-09.** `Oq4G256MoeBlocks = 53` exists, `block_bytes()` reports
132 (f32 scale + 128 nibbles, versus canonical's f16 at 130), `module_tensor_resident_len`
validates it against `oq4_moe_packed_len`, and `oq_gpu_dtype_for_quant_type` maps it to
`DType::Oq4G256` so dispatch, kernel selection and the indexed path are untouched. It is
deliberately **absent** from `module_requires_host_repack`, which is the entire saving —
that one predicate is what makes page-in a verbatim fetch. Two tests pin the contract: the
new code needs no repack while canonical still does, and both resolve to the *same* resident
length so a slab can never be sized wrong.

**Producer — `hipfire optimize` needs a real upgrade, not a flag.**

1. **It refuses every artifact this applies to, today.** `optimize.rs` bails when
   `hfq.modules()` is non-empty, and the message says the file "bundles N module(s)"
   and blames MTP/DFlash sidecars. But `HfqFile::modules()` returns
   `hfq_modules::HfqModuleRecord` — the **routed-expert table** — so it refuses precisely
   every paged MoE artifact, for a reason that misnames what it found. Fix the message
   and the behaviour together.
2. **Apply the right transform.** Routed-expert tensors take
   `oq4_canonical_to_moe_blocks` (→ qt 53), NOT the existing dense
   `oq4_pack_arch_combined` (→ qt 37) which the pager explicitly refuses. The two are
   per-tensor exclusive: a routed expert gets MoE-blocks, a dense weight gets
   arch-combined. An artifact can legitimately carry both.
3. **Round-trip the module table with recomputed offsets — and expose the layout planner
   rather than duplicating it.** `write_hfqm_package_mem` passes `hfq.metadata_json`
   through **unchanged**, so module records keep stale offsets; that is the real reason the
   tool refuses these artifacts. The writer's layout is deterministic and visible
   (`hfq.rs:484-503`): `metadata_offset = 32`, `index_offset = 32 + meta.len()`,
   `index_len = 4 + Σ(2 + name + 1 + 1 + 4·rank + 4 + 8 + 8)`, `data_offset =
   align_4096(index_offset + index_len)`, then a sequential cursor.

   **Do not re-derive that in the tool.** The repo's own rule is that the transform is "the
   SAME function the loader calls, so the tool and the loader can never drift" — the layout
   deserves the same treatment. Add a public planner in `hfq.rs` that both
   `write_hfqm_package_streaming` and the tool call, then iterate metadata↔offsets to a
   fixed point in memory (only digit widths change, so it converges in 2–3 rounds). That
   is the same fixed-point `hipfire-quantize/src/hfq_out.rs:1078-1090` already runs, and it
   needs no extra write of a 19 GB payload. Rewriting tensor bytes moves
   every `data_offset` / `data_size` / `rel_offset`. The circularity — metadata contains
   the offsets, and the offsets depend on the metadata's own length — is **already solved
   in-repo**: `hipfire-quantize/src/hfq_out.rs:1078-1090` iterates layout and metadata to
   a fixed point. Reuse that shape rather than inventing one. Module `data_offset` is an
   **absolute** file offset (`weight_pager.rs:1223`/`:1230` hand it straight to
   `read_host`/`fetch`); `rel_offset` is within the module.

**Consumer — `weight_pager`, four small changes.**
`module_requires_host_repack` (`:648`) returns false for qt 53 — that one predicate is the
entire CPU cost. `prepare_expert_module`'s transform match gains a passthrough arm.
`module_tensor_resident_len` / `module_resident_len` use the on-disk length, validated
against `oq4_moe_packed_len`. `register_expert_module`'s refusal list must not reject it.

**Runtime dtype.** `paged_moe_dtype_for_quant` (`qwen35/loading.rs:2463`) maps 53 →
`DType::Oq4G256`, so dispatch resolution, kernel selection and the indexed path are all
unchanged — this is a storage change only.

**Open question worth settling before baking it in:** qt 37's own comment says the combined
layout is identical across current RDNA/CDNA arches and the arch only tags the output name.
If the MoE-block layout is likewise arch-independent, the derived artifact should NOT take
a `.gfx1103` tag, and the naming follows the plain machine-section rule instead.

**Verification:** logits byte-identical between canonical-paged and moeblocks-paged on one
prompt, plus the CPU-time collapse — the before number is the measurement in §0.6(c),
~96 % of one core with the GPU at 0 %.

**(d) The paged CPU stall: what is known, and six things it is NOT.** Still open.

An earlier version of this section claimed the cause was the executor dropping to
`run_moe_decode_cpu_fallback`. **That is retracted.** Instrumenting
`MoeResolution::resolve` itself (`HIPFIRE_MOE_RESOLVE_DEBUG=1`) shows the executor and the
arch layer **agree**:

```
[moe-resolve] k=8 oq_gate=true router=Q8_0 routed(gu/dn)=Oq4G256/Oq4G256
              => idx(...oq4...)=true use_gpu_topk=true needs_x_rot=true
[moe-dtype]   layer=0 ... profile=Uniform(Oq4G256) k=8 path=Oq4 use_gpu_topk=true
```

`use_gpu_topk=true`, so `if !res.use_gpu_topk { return run_moe_decode_cpu_fallback(..) }`
is not taken. The dual-resolution split is real and did cause the routed-dtype bug, but it
is **not** the cause here — that inference was made from "the fallback exists inside the
bounded window" without checking the branch condition.

**Bisected 2026-08-09 (`HIPFIRE_MOE_STEP_DEBUG=1`).** Timestamped markers through
`run_moe_decode` put everything up to and including the gate-side GEMV at **0.7 ms**:

```
[moe-step]   0.0ms enter run_moe_decode
[moe-step]   0.0ms after check_moe_decode_supported
[moe-step]   0.1ms after x_rot_local block
[moe-step]   0.7ms after gate-side GEMV
```

then silence. Extending the same markers through the rest of the function narrows it to a
single dispatch — the whole prologue costs **1.9 ms**:

```
[moe-step]   0.5ms after gate-side GEMV
[moe-step]   0.7ms after softmax_f32
[moe-step]   1.0ms after topk + any host readback
[moe-step]   1.8ms after shared-expert down
[moe-step]   1.9ms after routed geometry bind      <- then ~160 s of silence
```

So resolution, rotation, gate side, softmax, top-K, the host readback and the shared expert
are all exonerated, and **the time is inside the indexed routed-expert GEMV dispatch**
(`gemv_oq4g256_moe_gate_up_k8_indexed_batched` and what follows it).

**Runtime JIT is eliminated (candidate #7).** The cached
`gemv_oq4g256_moe_gate_up_indexed_batched.hip` in `~/.hipfire/kernels/gfx1103` is
**byte-identical** to `kernels/src/`, so `ensure_kernel` finds a valid hash match and loads
the `.hsaco` rather than recompiling. A newer source mtime is not sufficient — the cache
keys on content.

**The reframing this forces, which should have come first.** This whole investigation has
been conducted with `HIPFIRE_QWEN35_MOE_OQ_INDEXED=1`, which I set to get past the earlier
dispatch refusal. That gate's own comment says it "re-enables the experimental indexed
routed OQ decode kernels **while debugging their finite-KLD failure**" — i.e. the path is
off by default *because it is known broken*. Nothing says that defect is confined to
numerics; a kernel that computes wrong answers can equally loop pathologically.

**Confirmed 2026-08-09 by A/B on the tiny fixture — one variable, opposite verdicts:**

| `HIPFIRE_QWEN35_MOE_OQ_INDEXED` | `tiny-quant qwen3_5_moe` |
|---|---|
| `1` | **FAIL — 7/7 Opus cells "non-finite KLD"** |
| unset | **PASS — 7/7**, drift `-0` on every cell |

The indexed OQ decode path is broken, on its own, independent of paging, the 35B artifact,
the pre-transformed format, and everything else this investigation touched. It reproduces
in minutes on a fixture instead of 160 s per layer on a 35B, and `tiny-quant` is therefore
the right harness for fixing it.

**Consequences for this plan.** Paged Opus MoE decode is blocked on repairing these
kernels, not on anything in §1.6's residency design — M5 cannot be measured on Opus until
they are fixed. `oq4`/`oq8` on the NON-indexed path is healthy (the control arm above), so
resident Opus MoE is unaffected; it is specifically the indexed routed path that paging
requires.

So the question was mis-framed. It is not "why is paged Opus slow" — it is **"the OQ
indexed kernels are known-defective, and this stall is plausibly that defect."** Which
means the productive next step is not further bisection into
`gemv_oq4g256_moe_gate_up_k8_indexed_batched`, but establishing whether these kernels work
*at all* on a small case: run the OQ4 indexed path on the tiny qwen3_5_moe fixture, where a
wrong or hanging kernel is diagnosable in seconds instead of minutes.

One measurement caveat to carry forward: "GPU use 0%" came from single `rocm-smi` samples
on an integrated UMA part and was never sampled repeatedly under load. If the GPU is in
fact busy, 99% host CPU is consistent with HIP's default spin-wait sync, and the story is
simply "the kernel is pathologically slow" rather than "the host is computing". Do not
lean on that 0% without resampling.

**Established:** ~160 s per MoE layer, one core at ~99 %, GPU at 0 %. Execution reaches the
executor's `resolve` and stalls before the `after paged topk` device sync — so the window
is inside `run_moe_decode`, on the GPU path, between resolution and top-k.

**Disproven by measurement, in order:** the host repack (a pre-transformed qt-53 artifact
stalls identically); `MoePrefillDtypes::from_ffn` returning `None`; OQ4 not being indexable;
the pager admission phases (never reached — they are downstream); runtime hipcc JIT of the
OQ4 MoE kernels (present as `.hsaco`); and the CPU MoE fallback (branch not taken).

**Method note, which is the real lesson.** Six hypotheses, each plausible from reading code,
each wrong. The two things that produced actual information were both *bounding* moves
rather than guesses: `HIPFIRE_PAGED_MOE_DEBUG`'s device-sync brackets, which localised the
stall to a line range, and instrumenting the decision itself rather than reasoning about
what it would decide. **Next step is mechanical bisection** — timestamped prints at every
step between `resolve` and the top-k sync — not another candidate.

Also: a `ls <cache>/<fn>.hsaco` probe for "is this kernel precompiled" is unreliable, since
cache entries are module names, not function names. It reported `rotate_x_mq` missing while
that kernel demonstrably ran.

**(b) The refusal panics rather than returning an error.**
`generate.rs:2979` `unwrap()`s the dispatch result, so an unsupported-dtype *refusal* — the
correct, reject-rather-than-miscompute behaviour — takes the whole daemon down. Today that
loses one request. Under the v2 executor it would kill every co-scheduled stream, including
any realtime one, which makes it a Tier-1 correctness item rather than a papercut. Fixing
it is independent of (a) and worth doing regardless.

## 0.5 The residency work this builds on is NOT on master

`ResidencyPolicy` does not exist in `weight_pager.rs` on `origin/master`. Phases 1 and 2
of `docs/plans/moe-expert-residency-unification.md` (`ResidencyPolicy::{LazyLru, PinAll}`,
`ModuleRole` role-ordered layout, `ExpertModulePtrs`, the lfm2moe conversion) are in
flight on `fix/oq8-from-flag-and-rotation-guards` and unmerged. **M5 of this plan depends
on them landing first.** What *is* on master and can be relied on:

| symbol | site |
|---|---|
| `ExpertModuleKey { layer, expert }` | `hipfire-runtime/src/weight_pager.rs:130` |
| `trait Transport` | `weight_pager.rs:157` |
| `PagerConfig` | `weight_pager.rs:850` |
| `WeightPager` | `weight_pager.rs:880` |
| `would_fit_expert_module_set` | `weight_pager.rs:1083` |
| `ensure_expert_module_resident` | `weight_pager.rs:1191` |
| `patch_expert_module_ptr_table` | `weight_pager.rs:1309` |
| `evict_lru_until` | `weight_pager.rs:1351` |
| `touch_module_lru` | `weight_pager.rs:1471` |

---

# Part 1 — Target architecture

## 1.1 The realtime contract, and why it removes the largest work item

The constraint is **not** a per-frame hard deadline. As stated by the user: a realtime
stream has a critical delay only for the **first** preemption, up to **200 ms**; after
that first interruption, low-priority jobs may be **paused on the device** for the
duration of the realtime session.

That is a fundamentally cheaper contract than per-frame EDF co-scheduling, and it changes
three things:

- **The bound is entry latency, not steady-state jitter.** Once admitted, the realtime
  stream has the GPU essentially to itself, so its per-frame latency is its *solo*
  latency. Nothing has to interleave at frame granularity.
- **Module splitting drops from prerequisite to optimization.** Against 200 ms, an
  unsharded `lm_head` (~0.7–2.7 ms) and a whole 512-token MoE layer (~500 µs) both fit
  comfortably. **Split-K attention with cross-launch online-softmax combine — the single
  largest kernel item any version of this design would need — is out of v1 entirely.** It
  returns only if long-context bulk work is measured holding a module for a substantial
  fraction of 200 ms.
- **The binding constraint becomes VRAM, not time.** "Paused on the device" means a
  suspended job keeps its allocations. A suspended `LoraTrainSession` holds the base model
  in fp32 (4 B/param — ~28 GB for a 7B) alongside a live realtime model and the module
  cache. On nix1's 64 GB UMA that is what will actually fail.

So the contract is:

```
drain_to_suspend  =  max_module_wcet  +  park_cost(all running bulk streams)
admit(realtime)   ⟺  drain_to_suspend + realtime_model_residency_cost ≤ 200 ms
                     AND paused_bulk_vram + realtime_vram + cache_budget ≤ budget
```

Two properties must be **built** rather than measured:

- **Suspension is lossless** — a parked stream resumes from its cursor, never restarts.
  This is exactly what image generation fails to do today.
- **Suspension is bounded** — every workload declares its largest indivisible unit and
  its maximum tolerated yield granularity, both **defaulting to unbounded** so nothing
  silently claims co-schedulability with realtime work. Those fields go on `WorkloadSpec`
  (`hipfire-scheduler/src/lib.rs:347`); `SchedulerPriorityClass::Realtime` (`:708`,
  currently constructed nowhere) is the carrier.

**The stubs are the test instrument, not a placeholder.** A synthetic workload class that
demands the GPU every N ms and records the delay to its first dispatch is precisely the
apparatus needed to test admission control, and it can be built and trusted before any
audio model exists. `SpeechIn`, `SpeechOut`, `VideoIn`, `VideoOut` land as `WorkloadClass`
variants with declared periods and a null executor. The exit criterion for the whole
realtime story is measured against them.

## 1.2 The module model

New crate `hipfire-modules` — pure, no GPU, no IO, to the `hipfire-scheduler` genericity
standard.

- `ModuleId(u64)` packing `[kind:8][layer:16][ordinal:24][shard:16]` is the stable,
  serializable identity; `ModuleIndex(u32)` is the dense runtime handle assigned at load.
  The split is load-bearing: it turns `HashMap<ExpertModuleKey, _>` into
  `Box<[ModuleEntry]>`.
- `ExpertModuleKey` (`weight_pager.rs:130`) is **subsumed**, not extended:
  `ModuleId::routed_expert(layer, expert)`. Keep a `From` impl through the transition,
  delete after.
- `ModuleKind` is `HfqModuleKind` (`hipfire-runtime/src/hfq_modules.rs:18`) with its
  declared-but-unused variants finally used, plus shard-bearing kinds. `AlwaysResident`
  disappears — it was a residency *policy* smuggled into a kind, and becomes
  `ResidencyClass::Pinned`.
- `ModuleGraph { modules, ops, edges, layers, entry }` **replaces** `LoweredForward`. The
  `layer → ops` nesting loses the cut points; the graph keeps them. `LayerProgram` becomes
  the concatenation of one layer's module programs, and `run_layer_program` still works on
  a per-module `&[SuperOp]` slice.
- `SuperOp` / `OpBinding` / `ForwardBindings` are **reused verbatim** — POD, indices-only,
  no lifetimes, which is exactly what lets a module program live in a flat arena and be
  dispatched from a different loop shape. `ForwardBindings` stays the arch seam.
- **`BatchExecutor` (`hipfire-serving-core/src/batch_executor.rs:45`) is deleted.** Its own
  doc calls it "a seam with a known-thin proof" — one real impl and one degenerate one.
  The arch seam becomes: lower into a `ModuleGraph`, implement `ForwardBindings`. Narrower,
  testable per-op, and it removes the "arch has prefill but no batched decode" degenerate
  state by construction.

**HFQ container changes.** `canonical_tensor_order` (`hipfire-quantize/src/hfq_out.rs:431`)
already groups routed tensors by `(layer, expert)` into contiguous byte ranges — extend it
to group *every* module kind contiguously, and `metadata_with_routed_modules` (`:451`)
becomes `metadata_with_modules`. In `hfq_modules.rs`: `expert: Option<u16>` (`:46`) →
`ordinal: Option<u32>` + `shard: Option<u16>`, because the current field cannot represent
a vocab shard, a KV shard, or a fused multi-expert module; `placement_policy: Option<String>`
(`:48`, hardcoded `"lazy_lru"` at `:182` and **never read**) becomes a typed `residency`
that is actually consulted; and records get emitted for `Attention`/`Norm`/`Router`/
`LmHead*`/`Embedding`, which `classify_always_resident_tensor` already computes correctly
and then throws into one bucket.

## 1.3 The module-major executor

New crate `hipfire-exec`. The loop body is "run module M for every row waiting on M":

```
loop {
    front.absorb(inbound.try_drain());           // joins, aborts, control frames
    let work = front.pick(now, &cache, &sched);  // deadline → n_rows → resident
    cache.ensure(work.module, &mut gpu)?;
    taps.fire(PreForward, &mut ev)?;
    let elapsed = exec.run_module(&mut gpu, &work, &mut arena, &mut streams)?;
    taps.fire(PostForward, &mut ev)?;
    front.advance(&work, &graph);                // rows move to successor modules
    cache.hint(front.lookahead(2));
    timing.record(work.module, elapsed);
}
```

The suspension point is *between iterations*. `min_quantum()` (`batch_runner.rs:198`),
`preempt_max_depth()` (`:207`) and the `parked` LIFO in `batch_runner_loop` (`:464`) exist
only because each preemption decision is a process round trip; in-process at module
granularity all of it is deleted with the file. **That deletion is the clearest evidence
the design is correct.**

**The activation arena is the structure the design turns on.** In a layer-major loop `x`
is one buffer. Module-major, every in-flight row's residual stream must be simultaneously
live, because rows sit at different modules. `ActivationArena` holds `x: [max_rows × dim]`
plus per-row device tables `row_stream`, `row_pos`, `row_kv_cap`, `row_compact_offset`,
`row_seed`, a scratch pool and a retain table. `row_stream` / `row_pos` are literally
`row_session_indices` / `row_positions` from `prefill_batch.rs`, promoted from per-call
tables to daemon-lifetime state; the `kv_k_ptrs`/`kv_v_ptrs` `[n_streams × n_layers]`
table likewise becomes arena-lifetime, patched on stream join/leave rather than rebuilt
per batch. At `max_rows=1024`, `dim=2048`, f32 that is ~8 MB for `x` and ~100 MB with
scratch — negligible against a 20 GB model, which is why the design is affordable.

**No paged KV.** `KvCache` (`hipfire-runtime/src/kv.rs:149`) stays contiguous-per-stream,
reached by device pointer table — what the `attention_*_routed_batched` kernels already
consume. vLLM-style paged KV is a separate, larger project; conflating it with
module-major weights would sink both.

**Exactly two constraints bind streams together.** Everything else is free.

1. **Kernel-signature partitioning.** `DensePrefillSessionBatchStateSignature`
   (`prefill_batch.rs:212`) demands equality on nine fields. Two of them —
   `kv_physical_cap` and `kv_compact_offset` — **become per-row device tables**. That is a
   signature change to the three `attention_*_routed_batched.hip` kernels, which take them
   as scalar launch parameters today, and to their dispatchers in
   `hipfire-rdna/src/dispatch/attention.rs`. Mechanical, and **the highest leverage per
   unit of work in the plan** — it collapses the dominant source of batch fragmentation.
   The quant-mode fields cannot go per-row without a per-row kernel switch, so they stay a
   `PartitionKey { kv_mode, dn_quant }`, with fan-out bounded by the modes actually in use.
2. **Same-stream aliasing on advancing state.** A `Recurrent`/`Conv` module advances
   DeltaNet/conv state in place, so at most one row *per stream* may appear in any such
   invocation. `MarchFront::admit` enforces it with a per-stream bit.

Explicitly **not** constrained: streams sitting at different layers; prefill chunks mixed
with decode tokens in the same invocation (a decode stream contributes 1 row, a prefill
chunk C rows — the row tables already make the kernels indifferent, and this is the thing
today's daemon cannot do at all, since `prefill_batch.rs:195` collapses a ragged batch to
a `singleton_tail`); joining mid-march, which is backfill and is free; leaving mid-march,
handled by a generation counter on the row slot.

The scoring function **is** the priority order, as a lexicographic comparator: deadline
first, then maximize `n_rows` (throughput), then prefer resident (capacity).

## 1.4 Taps: one march, many workloads

A `Tap` is an observer attached to a `StreamState` — never a process global — with a
`TapInterest` bitmask over `ModuleKind` checked with a single `&` before dispatch, so a
stream with no taps pays one branch per module.

| workload | taps | what disappears |
|---|---|---|
| inference | `SampleTap` post-`LmHeadFine` | — |
| embedding | `PoolTap` post final `Norm`; `LmHead*` **not scheduled at all** | the separate embedding path — and it skips the entire lm_head weight read, a capacity win the current code cannot express |
| KLD eval | `LogitTap{reference}` | `hipfire-kld` as a separate driver |
| calibration / imatrix | `ActStatTap` pre-`Proj` | `CalibrateDaemonSession`, the one-layer-per-frame quantum, `handlers/calibrate.rs` |
| steering | `SteerTap` post-`Attend` | the `hipfire_steer` process globals |
| H-Neurons CETT | `CettTap` pre-`down_proj` | `DaemonState::cett_colnorms` |
| training | `GradTap` → `Retain` + backward march | `LoraTrainSession` and its micro-step quantum |

`TapAction::Retain` is what makes training possible in a module-major loop at all: it
refcounts a row block instead of recycling it, so a module's input survives until the
backward march reaches it.

## 1.5 The unified forward/backward — one graph, two numeric bodies

`hipfire-train/src/lib.rs:8-11` states a deliberate invariant: the crate does *not*
differentiate the fused inference kernels; it owns an un-fused fp32 forward built on
`gemm_f32_train`. **That invariant stands.** Differentiability becomes a per-module
property:

```rust
enum Differentiability {
    Frozen,
    ActivationVjp(&'static dyn ModuleVjp),          // dL/dx only — routes grad past
    Trainable { params: ParamSet, vjp: &'static dyn ModuleVjp },
}
```

**This is far cheaper than it sounds, and the reason is a finding: `hipfire-train` is
already decomposed at exactly the module boundaries v2 needs.**
`block_forward_attn_only` (`hipfire-train/src/block.rs:164`), `moe_block_backward_capture`
(`:646`, which returns per-expert adjoints) and `block_backward_from_dxn2` (`:673`, whose
entire contract is "the MLP was run by the CALLER — take its input gradient") are the
module seam already. Unification is a registration problem, not a rewrite.

Two further facts keep the surface small. First, what `hipfire-train` does in production
is LoRA/adapter/drafter training, where nearly every module is `Frozen` and modules below
the deepest adapter need no VJP at all. Second — the honest part — **there is no gradient
with respect to an MQ4/HFQ codebook-quantized weight**, only with respect to a dequantized
surrogate or an adapter. The un-fused fp32 path is not a workaround; it is an admission
that "train the served weights" is ill-posed.

So `Proj`/`Norm`/`Act`/`Residual` VJPs already exist in `hipfire-train/src/ops`. `Attend`
has no fused backward and is handled by re-executing that module with the existing
un-fused fp32 block **for retained rows only**. `hipfire-train` stops being a parallel
model and becomes a VJP provider crate. **Do not attempt to unify the drafter training
path** (`dspark_train.rs`, `ssm_drafter.rs`) — it trains small models from scratch and
legitimately wants its own graph. That is a different model, not a different tap.

Costs, stated plainly: full-parameter pretraining of a quantized base through fused
kernels stays impossible — it is impossible today too. The fp32 recompute and the fused
forward are not bit-identical, so the gradient is with respect to a slightly different
function than the one served. **That is already true today.** It needs a documented
tolerance and a regression test, not a fix.

## 1.6 Residency

`WeightPager` becomes a daemon-level service (`hipfire-residency`), inverting today's
`Option<RefCell<WeightPager>>` inside `Qwen35Weights` — per-model, `!Send`, and every
concurrent use contending on one `borrow_mut`. Three changes carry real weight:

- **Fixed-frame slabs, not an allocator.** Every routed expert in a layer is the same
  size, so one slab per distinct frame size: admission is `free.pop()`, eviction is
  `free.push()`. Zero allocator traffic, zero fragmentation — and it makes the
  `upload_raw`/`GpuPool` bug in §3 *structurally unreachable* rather than merely fixed.
- **SIEVE, not LRU.** `touch_module_lru` (`weight_pager.rs:1471`) does `iter().position()`
  then `remove()` on a `VecDeque`. At top-8 × 48 layers = 384 touches per token against a
  ~5000-entry deque, that is on the order of a million pointer comparisons per token, on
  the critical path. SIEVE gives O(1) touch — one byte store. The deeper reason is that
  **MoE expert reuse is frequency-skewed, not recency-skewed**, which is the access pattern
  where LRU underperforms, and multi-stream sharing *increases* the skew — so the algorithm
  choice matters more in v2 than it does today.
- **Real async prefetch.** Every `Transport::wait()` impl returns `Ok(())` and the fetch is
  a blocking pread plus a blocking H2D, inline in the MoE dispatch. Give `TransferHandle` a
  real HIP event, double-buffer pinned staging, and poll between modules. **`hipEventQuery`
  and `hipStreamQuery` are absent from `crates/hip-bridge/src/ffi.rs`** — without a
  non-blocking query you cannot poll, and `event_synchronize` would serialize the very
  pipeline the design depends on. ~15 lines, hard prerequisite.

**What to prefetch, and the unresolved part.** Routers, attention, norms and shared
experts of L+1 and L+2 are unconditional. Routed experts of L+1 are *predictable*: run
L+1's router on the CPU against L's output while the GPU still executes L's experts.
`hipfire-runtime/src/cpu_router.rs` exists with **zero callers** — only doc references at
`weight_pager.rs:10`, `:14` and `:879` — and this is its purpose. But note the residual
dependency this does not remove: decode does a **D2H readback of top-k** before
`ensure_expert_module_resident`, i.e. a device sync per MoE layer. **Measure the
CPU-router mispredict rate before committing**; this is the most likely place the design
underdelivers.

**SUPERSEDED — the measurement below was taken at a degenerate cache budget.**
16 GiB exceeds the model's entire 15.9 GiB expert set, so the cache filled and
never evicted; "zero evictions" was cited as evidence when it actually meant the
test could not exhibit the pressure prefetch exists to relieve. A budget sweep
(next block) reverses the conclusion in the constrained regime. Kept because the
*unconstrained* numbers are still correct and still bound the upside there.

**MEASURED 2026-08-10 — do not build the CPU-router prefetch. Both halves of the
case for it are false on real numbers.** Paged Opus MoE on
`Qwen3.6-35B-A3B--oq4.hfq` (40 layers, 256 experts, top-8, 16 GiB expert cache),
three 48-token generations in one daemon lifetime:

| generation | decode tok/s | ms/token | cumulative cold-loads | evictions |
|---|---|---|---|---|
| 1 (cold) | 5.4 | 185 | — | 0 |
| 2 | 12.0 | 83 | — | 0 |
| 3 (warm) | **13.9** | **72** | 6011 total | **0** |

1. **The sync is noise.** 40 layers × ~15 µs ≈ 0.6 ms against a 72 ms warm token
   — **0.8%**. Removing it perfectly returns under one percent, so the readback
   was never the thing worth engineering away.
2. **The misses it would prefetch are a warmup transient, not steady state.**
   Zero evictions across all three generations; peak 6011 modules resident
   (9.3 GiB) against a 16 GiB budget, out of a 10,240-module (15.9 GiB) total
   set. The working set *fits*, so once warm there is nothing left to prefetch
   and a long-running server pays the fill exactly once. Decode improving 2.6×
   (5.4 → 13.9 tok/s) with no evictions is the cache filling, not routing
   prediction succeeding.

Note the corollary that also softens §0.4's capacity argument on this box:
routing is skewed enough that 48 tokens × 3 generations touched only 6011 of
10,240 modules (59%), and the whole set fits in cache anyway — so expert
residency is not the binding constraint at this model size.

**BUDGET SWEEP 2026-08-10 — residency policy dominates once the cache is smaller
than the working set.** Same model and prompt, three 32-token generations per
budget, `decode_tok_s` per generation:

| budget | cold-loads | evictions | gen1 | gen2 | gen3 |
|---|---|---|---|---|---|
| 2 GiB | 18798 | **17475** | 4.1 | **0.9** | 2.2 |
| 4 GiB | 12159 | 9512 | 4.7 | 7.5 | 6.9 |
| 8 GiB | 5335 | **40** | 4.9 | 12.5 | 12.5 |
| 16 GiB | 5335 | **0** | 4.9 | 12.6 | 12.8 |

- **The touched working set is ~8.3 GiB** (5335 modules x 1.55 MiB). Cold-loads
  are *identical* at 8 and 16 GiB, so anything above ~8 GiB is pure headroom and
  measures nothing about residency.
- **Degradation is not graceful.** 2 GiB collapses to 0.9 tok/s — 14x slower than
  12.8 — with evictions (17475) running nearly 1:1 against cold-loads (18798).
  Almost every module fetched is evicted before it is reused: LRU thrash, not
  capacity shortfall. Note gen2 at 2 GiB is *worse than gen1*, i.e. warming the
  cache actively hurt.
- **This is where §1.5's SIEVE argument earns its place.** Expert reuse is
  frequency-skewed (144 tokens touched 59% of modules), and LRU is precisely the
  policy that lets one sweep of cold experts evict the hot set. The 2 GiB row is
  what that failure looks like.
- **It also partially revives the prefetch case**, but bounded: at 2 GiB the run
  fetches ~196 modules/token (~304 MiB/token). Prefetch can *hide* that latency
  behind compute; it cannot remove the traffic. Fix the policy first — thrash at
  1:1 evict:fetch is a policy failure, and prefetching into a cache that
  immediately evicts what it fetched buys nothing.

**Caveat on all of the above: batch size 1.** Every number here is single-stream
decode (`batch=1` in the dispatch dumps). That is the worst case for §0.4's
amortization curve — top-8 experts per layer, no sharing. A wide batch touches
far more distinct experts per step but amortizes each module's bytes across more
tokens, so both the working set and the crossover move. The sweep should be
repeated at batch > 1 before any residency policy is tuned on it.

**FIRST MULTI-STREAM DATA 2026-08-10 — and it supports the thesis this plan is
built on.** Concurrency on the 35B was blocked for most of this branch by two
KV-sizing defects (an operator cap silently overridden at load, then again
per-request; fixed in `21aca50bb` / `f2d59b442`) and is now reachable at widths
2-3 via `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=serial` with kvarn:

| batch | ok | aggregate tok/s | per-stream tok/s |
|---|---|---|---|
| 2 | 2/2 | 7.86 | 3.93 |
| 3 | 3/3 | 7.19 | 2.40 |
| 4 | 0/4 | — | fused grouped-MoE decode requires Q8 KV |

**Aggregate throughput is FLAT while per-stream falls proportionally.** 2 -> 3
streams moves aggregate 7.86 -> 7.19 tok/s (down, within noise of flat) while
per-stream drops 3.93 -> 2.40. That is the signature of sessions *serialising*
rather than sharing a pass over weights — precisely the deficiency §0 describes
and that module-major execution exists to remove. Until now that claim rested on
the architecture of `batch_runner.rs`; it now has a measurement on the target
model.

**CONFIRMED 2026-08-10 — the batch coalesces and the fused step does not
amortise.** The flat curve above had three candidate explanations; two are now
falsified and the third is measured:

| candidate | verdict |
|---|---|
| prefill serialisation (`serial` backend) | **no** — repeated at 128-token generations, decode-dominated; still flat (14.55 / 14.66 / 14.74 tok/s at batch 1/2/3) |
| the 10 ms coalescing gather window | **no** — `HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS=500` changed nothing (13.63 / 14.19 / 14.37) |
| sessions never coalescing | **no** — `HIPFIRE_BATCH_WIDTH_TRACE=1` shows `48 decode_step rows=3`, i.e. every step is full width |

So three sessions *are* fused into each decode step, and wall time is still exactly
linear in batch (7.0 / 14.0 / 20.9 s). **A width-3 fused decode step costs ~3x a
width-1 step: the rows share a step but not a pass over the weights.** That is
precisely the deficiency §0 asserts and §0.4's amortization curve is meant to
remove — previously argued from the structure of `batch_runner.rs`, now measured
on the target model.

Still not evidence FOR the curve: widths 2-3 sit far below the N where sharing is
predicted to pay (past ~`n_exp / k` token-slots), and batch >= 4 remains blocked on
the fused grouped-MoE path hard-requiring Q8 KV (see BUGS.md), so 16/64/128 —
the widths §0.4 actually argues about — are still unmeasured. What is established
is the *pathology*, at the exact place the design targets it.

**Where the 72 ms actually goes is the open question.** 3B active params at oq4
is ~1.5 GB/token, i.e. ~15 ms at gfx1103's ~100 GB/s — so warm decode is running
about 5× off roofline. The launch-overhead and LRU costs this plan already names
(384 per-expert dispatches ≈ 3 ms; `touch_module_lru`'s O(n) scan ≈ 2 ms) are
together under 10% of it. That points at kernel efficiency, consistent with the
prior finding that the warm step is ~92% GEMM at ~2% kernel efficiency —
a different investigation from residency, and the one with the actual headroom.

## 1.7 Process structure

One process. Main thread supervises and holds no GPU. An **executor thread owns `Gpu`**,
the module cache, the activation arena, the stream set, the march front and
`ContinuousWorkScheduler`. A prefetch/loader thread carries a second `Gpu` handle and its
own stream. A tokio runtime hosts `hipfire-server`'s axum surface as a library
(`hipfire-server` is already `axum` + `tokio` "full"; `hipfire-daemon` has zero async
deps, so tokio lands on sibling threads and never touches the executor). Per-connection
transport threads stay as they are.

`Gpu` moves off the main thread. `Gpu::bind_thread` is a `thread_local` + `set_device`, so
the affinity requirement is "one consistent thread," not "the main thread." **Audit every
`Gpu` field for `Send` before relying on this** — the buffer/stream/event types carry
explicit `unsafe impl Send`, but `Gpu` itself has none, and a single `Rc`/`RefCell` field
would block the move. Moving the loader off the executor thread also fixes an unstated
violator of the drain budget: `LoadModel` is currently a multi-second non-preemptible
frame.

**The executor stays serial, deliberately.** Nothing in shared-weight multi-stream
decoding needs two threads issuing GPU work — the parallelism is intra-kernel (many rows
per launch), not inter-thread. But the process globals must still die, because they were
never "one request at a time" state; they are *per-stream* state, and a serial executor
interleaves streams *within* a march. `RAW_OVERRIDE`
(`hipfire-serving-core/src/model.rs:204`) becomes silently wrong the instant two streams
interleave — not merely inelegant.

`hipfire-daemon/src/transport.rs` survives **verbatim** — its reader-thread + `ReplySink`
+ control-frame interception split is the right answer and the design does not disturb it.
`queue.rs`'s `PendingQueue` survives with a narrowed job: a `Generate` frame now *admits a
stream* and returns, rather than running one. `batch_runner.rs` is deleted, not moved.

---

# Part 2 — Migration

## Status, 2026-08-09

Branch `feat/v2-daemon-module-major`. `./tests/no-gpu-ci.sh` exits 0 throughout.

| stage | state | evidence |
|---|---|---|
| M0 executor trace | **landed** | 254 tokens → 253 gaps; dispatch span vs stopwatch worst 0.01%; tracing cost +0.23% |
| M1a `upload_raw` / `GpuPool` | **landed** | pooled 4000 cycles → +0 B VRAM, 0 HIP calls; unpooled 200 cycles → +400 MiB |
| M1b per-stream sampler RNG | **landed** | global deleted; greedy unchanged, temp>0 reproducible across an interleaved request |
| M1c lease reaper | **landed** | 4 unit tests; 41/41 scheduler tests |
| M1d `RAW_OVERRIDE` | **landed** | live cross-request leak found and fixed; regression gate committed |
| M1d `hipfire_steer`, `load_progress::SINK` | deferred, with cause | steer couples to M2b; `SINK` audited 2026-08-10 — correctly scoped today, becomes a bug only once §F moves loading to its own thread |
| M2 onward | not started | |

**Two corrections to the earlier M1b sizing, both from actually reading the callers.**
The count of "~12 call sites of `sampler::sample`/`sample_cpu`" conflated two different
functions: most `sample_top_p` hits were `gpu.sample_top_p`, a **method on `Gpu`** (the GPU
sampler), which already takes a seed per call and returns the advanced state. And
`sampler::sample` already took `rng_state: &mut u32`. So the GPU path was never on the
global at all — only the CPU primitives were, plus `sample`'s CPU *fallback*, which used
the global and then deliberately left `rng_state` untouched.

The second half of M1b (on-device `sample_rows` over `ActivationArena::row_seed`) is a
performance requirement, not a correctness one, and follows separately.

**A finding that makes the case sharper than the plan did.** Both `generate` and
`generate_vl` reset the global to the *same hardcoded constant* `0x13579BDF`. So
concurrent requests did not merely share a stream — they repeatedly reset each other onto
an identical one. The test module also carried a `lock_sampler_rng()` mutex added because
the global produced a real CI flake (`left: 95, right: 52` from two identically-seeded
`sample_top_k_top_p` calls, passing locally). That mutex is now deleted: with the RNG
owned, the race is gone by construction rather than by every test remembering to take a
lock. Being able to delete it is the clearest evidence the global was a correctness hazard
and not an aesthetic one.

**M1d is three independent sub-tasks, not one.** Sized 2026-08-09:

- **`RAW_OVERRIDE` — LANDED, and it was a live bug, not just a v2 hazard.** The
  thread-local was set only by the plain-generate handler and read by
  `qwen35_materialize_batch_prefill_prompt`; the batch path never wrote it and nothing
  cleared it. Its doc comment claimed "reset every generate request, so no cross-request
  leak" — true for `generate`, false for batch.

  Confirmed on gfx1103 with one identical `prefix_hash_preflight` session:

  | | prompt tokens | boundaries | full hash |
  |---|---|---|---|
  | fresh daemon | 15 | 3 (`message_end`, `assistant_turn_start`, `full`) | `c8427f59…` |
  | after one unrelated `generate` with `"raw": true` | **7** | 1 (`full`) | `12317032…` |

  Those hashes are the **KV-reuse cache keys**, so this did not merely reframe a prompt —
  it changed what a later request matched in the prefix cache. `GenerateBatchPrefillSession`
  also had no `raw` field at all, so the batch path could only inherit; it now carries one.
  Fixed by threading `Option<bool>` from the handler through the eleven generate entry
  points (`generate.rs` ×4, `generate_arch.rs` ×7) and reading `session.raw` in the batch
  materialiser. Gate: `scripts/v2_m1d_raw_override_gate.py`.
- **`hipfire_steer::{SESSION, ACTIVE, EPOCH}`** — 8 `is_active()` sites. Coupled to M2b,
  since an active steer session is one of the four escapes that force the hand path;
  worth doing together rather than twice.
- **`load_progress::SINK`** — 18 references. Least urgent: a load is not a stream, and the
  per-connection routing bug it once caused was already fixed.

  **Audited 2026-08-10 and deliberately deferred — there is no live bug.** The global is
  `Mutex<Option<Box<ProgressFn>>>` (`crates/hipfire-runtime/src/load_progress.rs:30`) with
  exactly one installer and two clears in the whole workspace: install at
  `handlers/lifecycle.rs:461`, clear at `:479` (the one early return in between) and `:500`
  (immediately after `load_model` returns, before the result is matched). Every path out of
  the installed window clears; the only leak is a panic inside `load_model`, which is fatal
  to the daemon anyway. So the sink is *scoped correctly today*, and an RAII guard here
  would be tidiness, not a fix.

  It becomes a real bug under exactly one condition, and that condition is created by this
  plan rather than existing now: §F moves loading to its **own thread**, at which point two
  overlapping loads clobber one another's sink and load A's progress frames route to load
  B's connection — the same misrouting class the stderr-scraping fix was meant to end. The
  right shape (thread-local, RAII guard, or a sink passed explicitly through the loader
  signatures) depends on whether that thread serialises loads, which M3/M5 decides.
  **Fixing it before then means guessing the answer, and a thread-local guessed wrong
  silently drops progress frames rather than failing loudly** — the chat UI's load bar is
  not covered by any test. Do it with the loader-thread move, not before.

**RETRACTED — the "first generation after a load differs" anomaly is not a bug.**
It was recorded across the M0 and M1b commits as unexplained and needing its own
investigation ("greedy decoding should be deterministic from the first token"). It is
explained, and it is intended behaviour: **successive `generate` calls on one worker are a
conversation.** The daemon accumulates `m.active.cursor.conversation_tokens` and reuses KV
by longest-common-prefix against it (`generate_arch.rs:684-690`, and the comment at `:576`
naming the `reset` handler as the thing that clears it). So request N legitimately starts
from request N-1's context.

Discriminated by experiment rather than argument — run a *different* prompt first and see
whether the probe's result changes. It did, which rules out warm-up and identifies content
contamination; and `reset` between requests restores byte-exact reproducibility, including
across an intervening unrelated generation.

**The consequence for this plan is a methodology rule, not a fix:** any gate asserting
byte-identity or comparing latency across repeated requests must send `reset` between them,
or it is measuring a growing conversation. Both v2 gates now do. It mattered — before the
fix the M0 latency arms drifted 4.6 s → 10.3 s across reps as prefill grew, and the
resulting tracing-overhead figure was noise against a moving baseline (reported −0.12 %).
With resets, every rep is 254 tokens in ~4.56 s and the overhead resolves to a real
**+0.23 %**.

Ordering principle: the three roles the module plays are separable, and they are sequenced
in the stated priority order — **preemption first, residency second, unification last.**
Building residency first would deliver the capacity win, the lowest-ranked goal, while
leaving the latency goal unproven.

Second principle: **nothing that cannot be measured from inside the executor ships.**

### M0 — The latency instrument, before any v2 code

Today's entire in-daemon observability is one `eprintln!` behind
`HIPFIRE_DAEMON_SCHED_DEBUG` plus `SchedulerStats`, which counts frames, not time. Add a
preallocated ring of `(monotonic_ns, event, stream_id, module)` records with no allocation
and no IO on the hot path, dumped via a new `executor_trace` frame. **Sample VRAM into
it** — risk 1 is invisible otherwise.

*Exit:* a 256-token greedy generation on `qwen3.5-0.8b--oq4++` produces exactly 255
inter-token gaps in the trace; wall time reconstructed from the trace matches externally
measured wall time within 2 %; tracing on versus off changes tok/s by < 1 %, A/B alternated
**within one daemon lifetime**.

*Breaks:* nothing. Purely additive.

### M1 — Correctness prerequisites a churning executor cannot survive

Four independent, individually revertible fixes. Each converts a latent bug into a certain
one under module churn.

**M1a — `Gpu::upload_raw` bypasses `GpuPool` while eviction frees into it. LANDED.**
`hipfire-rdna/src/dispatch/mod.rs:2018` calls `self.hip.malloc()` directly; `free_tensor`
(`:2062`) calls `pool.free()`. Every evicted expert's VRAM lands in a free list the next
cold load cannot reach — a monotonic leak at the rate of paging traffic.

**The sizing in the original draft was wrong and is retracted.** It called the fix "a
signature change across every `upload_raw` call site", implying something mechanical.
There are **693 `upload_raw` call sites**, many holding another borrow of `Gpu` across the
call, so `&self` → `&mut self` would have been a large and genuinely risky change.

It is not needed. The asymmetry only *bites* on a churning caller — a load-once /
free-at-unload caller is unaffected, because `pool.drain` returns everything at teardown.
The only churning callers are the pager's three transports, and `Transport::fetch` already
takes `&mut Gpu`. So the fix is `Gpu::upload_raw_pooled` / `Gpu::pool_alloc` plus **three
call sites**, with `upload_raw` left `&self` and documented as load-path-only.

*Exit (met), `cargo run --release -p hipfire-rdna --example pool_churn_upload_raw` on
gfx1103:*

| path | cycles | pool `total_new` | pool `total_reused` | VRAM growth |
|---|---|---|---|---|
| pooled (fixed) | 4000 | **0** | 4000 | **+0 B** |
| unpooled (pre-fix) | 200 | — | — | **+400 MiB** |

The contrast run leaks 251.6 modules' worth in 200 cycles rather than 200 — HIP rounds
allocations up, so the stranded bytes exceed the payload. The fixed-frame slabs of §1.6
remain the M5 design; they subsume this and make the bug structurally unreachable, but
they are not needed to close the leak.

**M1b — per-stream sampler RNG.** `static SAMPLER_STATE: AtomicU32`
(`hipfire-runtime/src/sampler.rs:686`) is why batch decode is greedy-only. With streams
interleaved at module granularity, stream *i*'s token depends on how many others sampled
before it: per-request seeds become meaningless and replay becomes impossible. Two steps,
**both required** — move the state into `StreamState`, then move sampling on-device via
`row_seed` plus a `sample_rows` kernel, or module-major lm_head becomes N D2H round trips.
*Exit:* two sessions with distinct seeds at `temperature=0.8`, decoded in one batched step,
byte-identical to each run alone.

**M1c — lease reaper. LANDED.** `complete(lease_id)` had no timeout and `next_batch`
consults `active`; a dropped exclusive lease wedged every workload forever, across 10 call
sites held together by discipline. With the scheduler in-process and leases taken per
module batch, this becomes a liveness bug on every panic.

Implemented as `granted_ms` on the lease plus `reap_expired_leases(now_ms)` called at the
top of `next_batch`, with `leases_reaped_total()` and a settable
`set_lease_timeout_ms(Option<u64>)`. **Not** an RAII `LeaseGuard` as the draft suggested:
`WorkloadBatchLease` is `Clone` and is handed across a channel to a runner, so a `Drop`
impl would fire on every clone that went out of scope and complete leases that were still
running. Reclaiming on a timeout observed by the scheduler is the version that cannot
misfire on a copy.

The timeout defaults to a deliberately generous 10 minutes, and the reason is the risk
direction: reaping a *live* lease releases its resources and can put a second batch on the
GPU beside it, which is worse than the wedge. So it must be far longer than any legitimate
quantum rather than tuned near one.

*Exit (met):* four unit tests — a dropped exclusive `Training` lease blocks while plausibly
alive then is reclaimed past the timeout with the counter at 1; a normally completed lease
never counts as reaped; the boundary is exact (`timeout` does not reap, `timeout + 1`
does); and `None` disables reclaiming entirely. 41/41 scheduler tests pass.

**M1d — the remaining process globals become per-stream:** `RAW_OVERRIDE`,
`hipfire_steer::{SESSION, ACTIVE, EPOCH}`, `load_progress::SINK`.

*Breaks:* M1b changes sampled output for every `temperature>0` request — deliberately;
today's output is not reproducible under batching anyway. Greedy tiny-quant baselines must
**not** move. If they do, M1b is wrong.

### M2 — One forward: the lowered substrate becomes the sole path

Anything not expressed as a `SuperOp` is invisible to the executor and runs to completion
holding the GPU.

**M2a — lower prefill.** Express the prefill forward as a `LayerProgram` with a row count
in `DispatchCtx`; the batched prefill's pointer-table machinery is already the binding
target. *Exit:* `HIPFIRE_FORWARD_ORACLE` dual-run reports **0** mismatched values on a
4-session × 64-token batched prefill. *Falsified by any nonzero mismatch* — this is the
accept-and-miscompute bug class and the oracle is the only thing that catches it.

**M2b — retire the four hand-path escapes** (`decode_layers.rs:42-49`). Each becomes either
a super-op — a capture hook as `Escape(EscapeKind)` is natural, since the executor already
has a per-op boundary — or an **explicit named refusal wired into backend selection**,
never an assertion inside a kernel. `hipfire_steer` is the hard one: it is a process
global, and the fact that it *forces the hand path today* is a tell that the subsystem
already knows it cannot survive the substrate.

*Breaks:* spec-decode GDN tape capture until ported, which takes down
`tests/tiny-spec-gate.sh` — **scheduled for repair in this stage, not deferred.**
RoughQuant corrections lose their dormant home; the hand path they are wired into is
already broken (bf16 self-KLD 13.89 versus lowered 0.000, recorded at
`decode_layers.rs:33-39`), so either port the correction into the super-op executor here
or delete it and say so.

*Revertible:* M2a yes; M2b no, it deletes the fallback. Separate commits.

### M3 — Executor v2: interleaved cursors and the suspension boundary

Replace the request `match` for forward-executing requests with `RunningStream` cursors and
the march loop. Frames still arrive through `PendingQueue` — its per-connection FIFO
invariant is still correct — but they now *admit* work rather than execute it.

*Exit — the headline numbers,* one realtime-class stream and one bulk stream, read from the
M0 trace:

1. **p99 and max module duration**, and which `SuperOpKind` owns the max. This is the
   achievable suspension floor and therefore the tightest drain budget the design can hold.
2. **Time from realtime admission to first dispatch** under saturating bulk load — the
   number the 200 ms contract is about.
3. **Bulk throughput, loaded versus solo — ≥ 0.6×.** Without this, (2) is trivially
   satisfiable by refusing to run the bulk job.

*Breaks:* everything that assumed a forward runs to completion. `hipGraph` capture is one
indivisible quantum by construction — **off on the v2 path** until its WCET is declared,
since a declared WCET that ignores an enabled graph is exactly the failure the contract
exists to prevent. Cancellation moves from three per-token hook sites into the executor's
pick step — one site instead of three-and-counting, and finer.

*Revertible:* yes, behind `HIPFIRE_DAEMON_EXECUTOR=v2`.

### M4 — Split `Moe` into `MoeRoute` / `MoeExpert(e)` / `MoeCombine`

Now the module is the quantum. Expect M3's exit measurement to name `Moe` as the max — and
**if it does not, say so**, because that means the workload never routed widely and the
measurement was too easy.

**`moe-expert-residency-unification.md`'s Phase 3 blocker applies here.** deepseek4 fuses
top-k selection into dispatch at `hipfire-dispatch/src/pipeline/mod.rs:851` — there is no
admission seam. It stays on a coarse `Escape` with a capability predicate that **names the
reason** ("deepseek4 fuses expert selection into dispatch; module quantum unavailable"),
checked at lowering time. qwen35's split must not restructure the shared hot-path code
deepseek4 also uses.

*Exit:* logits byte-identical to the fused `Moe` via `HIPFIRE_FORWARD_ORACLE`, **and** the
trace shows `MoeExpert` quanta per layer equal to the distinct experts in that pass's
routing table. *Falsified if* the counts differ — that is a silently dropped or double-run
expert.

### M5 — The cache becomes the executor's residency authority

**Depends on Phases 1–2 of `moe-expert-residency-unification.md` landing on master
first** (§0.5). **Correction (2026-08-10): that document is not on master and not on this
branch.** It exists only on the unmerged topic branch
`fix/oq8-from-flag-and-rotation-guards` (added by `691e7730e`, 2026-08-08), which was the
checked-out branch when this plan was written — hence the citation. Every reference to it
in this plan, including this dependency and the `ResidencyPolicy::{LazyLru, PinAll}`
discussion in §0.5, describes work that is **written but unmerged**. `PinAll` in
particular does not exist on master. So M5's stated prerequisite is not merely "not yet
done" — it is *done elsewhere and awaiting a merge decision*, which is a different kind
of blocker and a cheaper one. Resolve the merge before sequencing M5.

`MoeExpert(e)` lowers to `ensure_module_resident` → dispatch → touch. The
paths exist; what changes is that the *executor* calls them and the budget comes from
`ResourceReservationManager` rather than `qwen35_expert_cache_budget_bytes()`. Sub-items:
SIEVE (M5a), real async transport plus `hipEventQuery` (M5b), emitting and consuming the
non-`RoutedExpert` module kinds (M5c).

*Exit:* run a model whose expert set is 3× the pager budget to completion, greedy,
byte-identical to the same model run pinned. *Falsified by* any token difference, or by
VRAM growth — which would mean M1a regressed.

### M6 — Realtime admission and the stub classes

`SpeechIn`/`SpeechOut`/`VideoIn`/`VideoOut` land as `WorkloadClass` variants with declared
contracts and a synthetic periodic executor. `WorkloadSpec` gains declared
largest-indivisible-unit and max-yield-granularity fields, both defaulting to unbounded.
Admission implements the drain-budget and VRAM test of §1.1; bulk streams suspend
losslessly for the session's duration.

*Exit:* a synthetic realtime stream arriving against saturating training reaches its first
dispatch within 200 ms on ≥ 99.9 % of 10 000 arrivals, **and** every suspended bulk stream
resumes from its cursor with output byte-identical to an uninterrupted run.

### M7 — Cross-stream module coalescing (the capacity thesis)

*Exit:* the amortization ratio `distinct_experts_touched / (N × top_k)` for
N ∈ {1, 4, 16, 64, 128}, and the crossover N at which module-major beats layer-major on the
same box. *Falsified if* there is no crossover below the N whose KV fits in VRAM — in which
case report it rather than tune around it.

### M8 — Training onto the same substrate

Per §1.5. *Exit:* a LoRA training step and an interactive decode step interleaved at module
granularity in one executor, with the loss curve over 200 steps inside the run-to-run noise
band established by two solo runs.

*Residency arithmetic to note now:* a suspended `LoraTrainSession` holds the base model in
fp32, ~28 GB for a 7B. On nix1's 64 GB UMA, one fp32 7B base plus a served MoE plus the
cache budget does not fit. **Training as a co-resident bulk workload is size-limited on
this box regardless of scheduling.**

---

# Part 3 — The first demonstration

**It lands at the end of M3**, before any module splitting, because the top-priority goal
is latency and latency is provable with super-op suspension alone.

- **Interactive:** `~/.hipfire/models/qwen3.5-0.8b--oq4++.hfq` — on disk, arch 5 (dense).
  Dense is *correct* here: it isolates the scheduling claim from the MoE claim.
- **Bulk:** the same artifact as a second resident worker driving a LoRA session, so VRAM
  is not the confound.
- **MoE artifact for M4–M7:** built 2026-08-09 from
  `/srv/hipfire/models/Qwen3.6-35B-A3B.hfa`. Three things learned doing it, each a trap:

  1. **`.hfa` is not a quantizer input.** It is an `HFAR0002` HuggingFace *archive*
     produced by `hipfire-coexistence hub fetch`; `hipfire-quantize --input` accepts only a
     model dir, a `.gguf`, or a `.hfq`. Restore first with
     `hipfire-coexistence repack --input <a.hfa> --output <dir>` (lossless, byte-identical).
  2. **`--format oq4` is already the right layout, and `hipfire optimize` would destroy
     it.** The quantizer emits canonical `Oq4G256`, which is exactly what
     `WeightPager::register_expert_module` accepts (via `oq4_canonical_to_moe_blocks`). The
     dense `Oq4G256ArchPacked` that the pager *refuses* — "regenerate a canonical pageable
     artifact" — is produced only by the separate `hipfire optimize` tool
     (`hipfire-runtime/src/oq4_arch.rs:14-30`). So `MODEL-SUPPORT.md`'s "indexed OQ remains
     opt-in" is about the **load path**, not the storage format: quantize plainly, and do
     **not** optimize the artifact or it silently stops being pageable — a failure that
     would present as an M5 bug rather than an artifact mistake.
  3. **Disk placement is a measurement decision, not housekeeping.** The artifact must live
     on local NVMe rather than `/srv`: M5's paging numbers come from the pager's transports
     reading the `.hfq` directly, so an NFS-resident artifact would measure the network.

  **Built and verified 2026-08-09** — `~/.hipfire/models/Qwen3.6-35B-A3B--oq4.hfq`, 19.1 GB,
  20 min of quantize after a 71.9 GB restore (`.hfa` is compressed; the restored bf16
  checkpoint is ~72 GB, not the 47.8 GB archive size). `artifact inspect` confirms every
  property M4/M5 needs:

  | property | value |
  |---|---|
  | arch_id / format | 6 (qwen3_5_moe) / `oq4` |
  | layers × routed experts | 40 × 256 = **10,240 expert modules** |
  | top-k / hidden / moe_inter | 8 / 2048 / 512 |
  | tensors | 21,093 (from 1,045 source — the 3D stacked experts ARE split per-expert) |
  | routed-expert module records | **10,240**, `layers.0.experts.0` … `layers.39.experts.255` |
  | routed quant_type | **34 = `Oq4G256` canonical** (pageable; not 37 ArchPacked) |
  | module size | uniform **1,597,440 B**, exactly one distinct size |

  The uniform module size matters: §1.6's fixed-frame slab design applies directly — one
  slab, one frame size, `free.pop()`/`free.push()`, zero fragmentation. And 1.52 MiB is
  close enough to the 1.59 MiB used in §0.3's launch-overhead table that those figures hold.

  **A correction worth recording:** mid-build I inferred from "Found 1045 tensors" that the
  quantizer was *not* splitting the stacked `experts.gate_up_proj`, and concluded the
  artifact would carry no per-expert module records. That was wrong — 1,045 is the *source*
  count, and each stacked tensor expands into 256 per-expert outputs. The 21,093/10,240
  figures above are the refutation. The lesson is that a progress counter over inputs says
  nothing about outputs.

  **Blocker found, and it is two separate problems (see §0.6).**

  **Retracted:** this section previously warned off `--format qtip3`/`qtip4` because the
  qtip path self-locked under a parent holding the flock. That self-lock has been deleted —
  zero `lock_blocking` calls remain in the quantizer, and both
  `hipfire-quantize/src/main.rs:4992-4999` and the crate's `gpu` feature comment say so.
  Wrapping it in `hipfire lock`, as AGENTS.md asks, is now correct.

Procedure: ten alternating 30-second windows **in one daemon lifetime** (solo / loaded /
solo / …), reporting the paired difference — forced by gfx1103's ~8.6 % first-run position
effect. Identical prompts on the realtime stream so telemetry does not describe a ragged
tail. Everything read from the in-daemon trace. Coordinate non-daemon GPU work with
`hipfire lock`; **do not wrap `hipfire-eval`**, which deadlocks against the caller's own
holder.

**What would falsify it:** bulk throughput < 0.2× solo means you serialized rather than
suspended; `overtaken_total` = 0 while the other metrics look good means the workloads
never contended and the measurement proves nothing; solo latency already exceeding the
target means the model is too big for this box and the design cannot be evaluated here —
**report that, do not tune around it.**

---

# Part 4 — Prerequisites, ranked

**Tier 1 — blocks correctness.** `upload_raw`/`GpuPool` asymmetry
(`dispatch/mod.rs:2018` vs `:2062`); process-global sampler RNG (`sampler.rs:686`); the
missing lease reaper (`scheduler/lib.rs:542`, `:608`); `RAW_OVERRIDE`
(`serving-core/src/model.rs:204`); and a named capability predicate for the split-`Moe`
path, wired into *selection*. The accept-and-miscompute class has fired twice already —
awq_scale in the qwen35 fused dense body, expert dtype in deepseek4 prefill MoE — and a
third instance here would be silent wrong output on every MoE token.

**Tier 2 — blocks the latency goal.** Prefill not expressed as super-ops (14k lines of hand
control flow forming one indivisible quantum); the four hand-path escapes; `hipfire_steer`
globality; `Moe`'s unbounded data-dependent duration; the homogeneous-batch signature;
`load_progress::SINK`; `hipGraph` capture as an unsplittable quantum; and the missing
`hipEventQuery`/`hipStreamQuery` in `hip-bridge/src/ffi.rs`.

**Tier 3 — blocks scale only.** `WeightPager` being per-model; the O(n) LRU touch; no-op
`Transport::wait`; only `RoutedExpert` ever emitted and `placement_policy` never read; the
indexed expert layout being opt-in on arch 6; no MoE artifact quantized locally;
deepseek4's fused select.

---

# Part 5 — Top risks

1. **Memory, and a paging executor makes it acute rather than latent.** `GpuTensor` has no
   `Drop`; `serving-core/src/model.rs:266` documents ~2 GB/forward accumulation on a missed
   free. Module-major changes the shape three ways at once: allocations move *inside the
   scheduler*, at `ensure_module_resident`, whose free is a different control-flow event
   reached from a different stack; with M1a unfixed the free does not return memory to the
   allocator that will next request it; and the failure is not a crash but a slow VRAM
   slope that on 64 GB UMA reads as "the model got slower" long before it reads as OOM.
   *Mitigation:* every executor allocation is an `OwnedTensor` with `reclaim_pending()` at
   the pick step — one site — and VRAM is sampled into the M0 trace so the slope shows up
   in the same artifact as the latency numbers.
2. **Module-major loses at low concurrency — the regime the design prioritizes.** See §0.3.
   This needs a measured, declared coalescing policy, not an assumption.
3. **WCET is data-dependent and `Moe` is where it explodes.** A WCET measured on one
   workload is wrong on another, and it must be measured **on the serving path, not the
   induction path** — the calibration forward is not the serving forward.
4. **Five numeric paths collapse into one, and each collapse is a chance to
   accept-and-miscompute.** Non-negotiable mitigation: every collapse ships behind
   `HIPFIRE_FORWARD_ORACLE` dual-run with a zero-mismatch gate.
5. **Suspending bulk work on-device is a VRAM bet, not a scheduling one.** The 200 ms
   contract is easy on time and hard on space. If a paused training job plus a live realtime
   model does not fit, the contract fails for a reason no amount of scheduler work fixes.
6. **The cold-tier quality trade needs a number before it ships.** If a realtime stream's
   selected expert is cold on disk, the executor must either miss the budget or drop that
   expert from top-k and renormalize. The latter is correctness-visible: the KLD cost of a
   top-8 → top-7 substitution must be measured and the drops counted in telemetry.
7. **Scheduling is unobservable from outside, and that has already produced two false
   failures.** Every exit criterion here reads the in-daemon trace.

---

# Part 6 — What stays out

**Not in v1:** intra-daemon GPU parallelism (one executor thread, one HIP context); paged
KV; heterogeneous batches; the module quantum for deepseek4/minimax/zaya/nemotron, which
keep a coarse `Escape` and a named refusal; differentiating the fused inference kernels;
diffusion, where restart-from-seed is adequate for now and `routes/sdapi.rs` is 8866 lines;
real audio codecs, VAD and ASR/TTS models; `hipGraph` capture on the v2 path; multi-GPU and
EP; and **split-K attention with cross-launch online-softmax combine**, removed by the
200 ms reframing after being the largest kernel item in every earlier version of this
design.

**Stays out of the daemon** per AGENTS.md's format-conversion line: `hipfire-quantize`
stays standalone and CPU/rayon — emitting the new module records and WCET primitives into
`.hfq` is *format* work and belongs there, and it is the one thing v2 asks of the
quantizer. `hipfire-coexistence` keeps its entire index/bytes surface. The daemon-free
`calibrate` CLI keeps its self-lock. `artifact compare-calibration --atol 0 --rtol 0`
remains the byte-identity oracle.

---

# Part 7 — Verification

- Every stage: `./tests/no-gpu-ci.sh`, plus `./tests/tiny-affected-gate.sh
  --require-coverage` for runtime and quantization changes.
- Every numeric-path collapse (M2a, M2b, M4, M8): `HIPFIRE_FORWARD_ORACLE` dual-run, zero
  mismatches, no exceptions.
- M3 and M6: the in-daemon executor trace only. **Client-side reply order does not report
  service order** — that mistake has already produced two false failures.
- All A/B comparisons alternate **within one daemon lifetime** (gfx1103's ~8.6 % first-run
  position effect), with identical prompts on co-scheduled streams so telemetry does not
  describe a ragged tail rather than the batch.
- `tests/tiny-spec-gate.sh` is expected to break at M2b and must be repaired in that stage.

---

# Part 8 — Docs to update as part of the work

- **`crates/hipfire-dispatch/src/pipeline/superop.rs:34` and
  `crates/hipfire-arch-qwen35/src/qwen35/decode_layers.rs:31` — both state that the lowered
  path is off by default; both are false.** Fix these first; they have already misled one
  design pass (§0.1).
- `docs/plans/2026-07-25-daemon-merge-training-induction-scheduler.md` — mark M4d and M5–M8
  superseded; keep M0–M4c as landed history.
- `docs/plans/moe-expert-residency-unification.md` — **not on master; lives only on the
  unmerged branch `fix/oq8-from-flag-and-rotation-guards` (`691e7730e`).** Its Phase 0 also
  already root-caused the 14-cell tiny-quant breakage filed in `BUGS.md`: 9 cells are one
  regression from `8b9ee5392` (an unguarded `_` arm above the `oq8` literals in
  `HfqInputFormat::from_flag` made every OQ8 flag unreachable) and are fixed there; the
  other 5 are minimax and are separately pre-existing. Merging that branch is a decision,
  not a debugging task. Phase 3's deepseek4 blocker is now also
  M4's arch-coverage boundary, and Phases 1–2 are M5's dependency. Cross-link both.
- `MODEL-SUPPORT.md` — arch 6's indexed-OQ default changes at M5.
- `crates/hipfire-daemon/AGENTS.md` — still claims `/tmp/hipfire-gpu.lock`; the code
  resolves `~/.hipfire/locks/hip-gpu-0.lock`. Stale before this plan and still stale.
- `AGENTS.md` — add the declared-WCET / yield-granularity requirement, and the rule that a
  new fused path needs a named capability predicate in backend *selection*.
- `tests/AGENTS.md` — record the three measurement traps as standing verification warnings.
- `BUGS.md` — file the `upload_raw`/`GpuPool` asymmetry independently. It is wrong today,
  not only under v2.
