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

**CORRECTED 2026-08-11 — the earlier reading of this table was wrong, and the
correction matters.** On 2026-08-10 this section concluded "the batch coalesces
and the fused step does not amortise." The first half is right; the second was
never measured, because **at batch 1-3 the fused path does not run at all**:

```rust
pub fn qwen35_grouped_moe_decode_auto_latency_gate_passed(session_count: usize) -> bool {
    session_count >= 4          // crates/hipfire-generate/src/lib.rs:1522
}
```

`auto` deliberately refuses to fuse below four sessions and selects
`SerialReference` — a per-session loop. So the flat curve at batch 1-3 is
*designed behaviour*, not the amortization defect it was filed as. Confirmed
structurally: `HIPFIRE_LAUNCH_TRACE=1` at width 3 shows **exactly 3.00x the
launches of width 1, with every grid dimension unchanged** (`rmsnorm
grid=[1,1,1]`, `moe_gate_up grid=[1024,8,1]` at both widths) — 1322 launches per
row, three times over. The envelope carries 3 rows; the execution is a loop. That
is what `SerialReference` is defined to do.

What the three falsified candidates below still establish is that coalescing
itself works — the sessions really do arrive in one envelope. They were answering
the wrong question.

| candidate | verdict |
|---|---|
| prefill serialisation (`serial` backend) | **no** — repeated at 128-token generations, decode-dominated; still flat (14.55 / 14.66 / 14.74 tok/s at batch 1/2/3) |
| the 10 ms coalescing gather window | **no** — `HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS=500` changed nothing (13.63 / 14.19 / 14.37) |
| sessions never coalescing | **no** — `HIPFIRE_BATCH_WIDTH_TRACE=1` shows `48 decode_step rows=3`, every step full width |

**What fusion actually delivers, measured where it can run.** Q8 KV is the only
configuration in which the fused grouped-MoE decode executes (see the blocker
below), via `HIPFIRE_KV_ALLOW_DEPRECATED=1`. One daemon lifetime, 64-token
generations, `decode_step rows=4` confirmed on 32 consecutive steps:

| batch | aggregate tok/s | per-stream | vs batch 1 |
|---|---|---|---|
| 1 | 7.92 | 7.92 | — |
| 4 (genuinely fused) | 8.80 | 2.20 | **1.11x** |
| 8 | — | — | invalid: HTTP 429 from the `requests_per_minute` bucket in `api_auth.rs`, not a batching limit |

**Four rows buy 11%.** Read against §0.4 that is not a defect — it is roughly what
the curve predicts. The amortization table puts the knee near `n_exp / k` =
512/8 = **64** token-slots; at 4 slots it predicts only a few percent of weight-byte
saving, so 1.11x is if anything slightly ahead of theory. The honest conclusion is
therefore *not* "fusion is broken" but:

> every batch width reachable today sits far below the amortization knee, and both
> routes to a wider one are blocked.

That is evidence FOR the plan's premise, not against it, and it relocates the
work: the open question is no longer "why doesn't batching help" but "can we reach
N=64 at all on this box" — which is M7's exit criterion, and which §0.4 already
says should be *reported* rather than tuned around if VRAM cannot hold it.

**RESOLVED 2026-08-11 — aggregate throughput DOES scale; four caps were hiding
it.** Sweeping properly (Q8 KV, loopback bind, `HIPFIRE_SERVER_PREFILL_BATCH_MAX`
raised, `auto` prefill), one daemon lifetime, prefill and decode separated by
running each width at `max_tokens=1` and `max_tokens=64` and subtracting:

| width | prefill (t@1tok) | decode step | decode tok/s | vs width 1 |
|---|---|---|---|---|
| 1 | 0.93 s | 11.3 ms | 88.6 | 1.00x |
| 4 | 1.72 s | 33.1 ms | 120.7 | 1.36x |
| 8 | 3.45 s | 52.2 ms | 153.3 | 1.73x |
| 16 | 7.05 s | 87.3 ms | 183.3 | **2.07x** |

End-to-end at width 16: 12.55 s for ~207 tokens = **16.5 tok/s aggregate vs 7.9 at
width 1, 2.08x**. Sixteen rows cost 7.7x the time of one row. Both halves
amortize — decode 2.07x, and prefill 2.1x once `auto` is allowed (14.80 s -> 7.05 s
at width 16; the earlier BUGS.md note that "`auto` does NOT work, only `serial`"
was measured under kvarn and does not hold under Q8).

The four caps, each of which independently flattens the curve:

| cap | value | effect |
|---|---|---|
| `qwen35_grouped_moe_decode_auto_latency_gate_passed` | `n >= 4` | below 4, fused decode is refused by policy |
| `RatePolicy::default().max_in_flight_text` | `4` | non-loopback bind caps CONCURRENCY at 4 — this is what returned 429, not a rate bucket; `loopback_default()` sets 0 = unlimited |
| `BATCH_MAX_DEFAULT` (`batch_runner.rs:421`) | `8` | envelope never exceeds 8 rows regardless of demand |
| `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=serial` | harness | 16 sequential prefills = 73% of wall time at width 16 |

**Why 2.08x and not 16x — and why that is the plan's own answer.** §0.4 predicts
weight-byte savings of only ~1.05x at 8 slots and ~1.14x at 16, because distinct
experts touched grows nearly linearly with batch until N approaches `n_exp/k` =
512/8 = 64. Measured amortization is *well ahead* of that (1.73x and 2.07x), and
the reason is visible in the launch trace: **a width-1 decode step issues 1322
launches in 11.3 ms — ~8.5 us each, i.e. it is launch-bound, not
bandwidth-bound.** Batching amortizes those 1322 launches across N rows, which
pays far better at small N than sharing expert bytes does. At width 16 the step is
87.3 ms for the same 1322 launches (66 us each), so it has crossed into real work.

That sharpens F3 rather than contradicting it: launch overhead dominates
single-stream MoE decode on gfx1103, so the coalescing policy F3 demands is worth
more than its own arithmetic suggested — and it is an argument for grouping
modules into fewer launches, not more.

**The ceiling, and it is capacity not scheduling.** Raising `BATCH_MAX` to 64 does
not produce width 64: achieved width tops out near 18 and batch 64 collapses to
2.22 tok/s with 20/64 sessions surviving and
`generate_batch_prefill ... failed to create checkpoint`. So N=64 — the width at
which §0.4's curve finally pays 1.58x on expert bytes alone — is **not reachable on
nix1**. Per §0.4's own instruction that is to be reported, not tuned around: the
capacity half of the thesis cannot be evaluated on this box at 35B-A3B.

**The two blockers, both now named.**

1. **kvarn cannot use the fused path.** At batch >= 4 with the recommended KV mode
   the request does not fall back — it dies:
   `"grouped MoE session fused prefix row 0 must use Q8 KV state for the MQ4
   control path"` (`crates/hipfire-arch-qwen35/src/qwen35/prefill_batch.rs:1102`).
   The refusal is asserted at **execution**, after the backend was already
   selected, which is exactly the anti-pattern this plan's Tier-1 prerequisite
   list forbids ("a named capability predicate wired into *selection*, never an
   assertion inside a kernel"). Since q8 is now deprecated at load, the fused
   grouped-MoE decode is **unreachable on every supported KV mode** — batched MoE
   decode is effectively dead until the kvarn port lands. This raises that port
   from an optimization to a prerequisite for measuring §0.4 at all.
2. **The rate limiter caps the sweep** before VRAM does. Widths >= 8 need
   `requests_per_minute` raised in the server config; that is a harness fix, not a
   finding, but it must happen before any N=16/64/128 number is quotable.


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
`RAW_OVERRIDE` is retired as of 2026-08-25; the other two remain.

**Scoped 2026-08-25 — see `docs/plans/2026-08-25-m1-scope.md`.** Headline: M1b
is NOT as landed as this section reads. The `static SAMPLER_STATE` is gone, but
nothing samples through the per-stream `SamplerRng` (`daemon/stream.rs:181` says
so), no request seed reaches it (`SamplerRng::from_seed` has zero production
call sites), and the RNG state is still shared per MODEL — it round-trips
through `Qwen35Scratch::sample_buf`, a single `[2]` tensor on the one
`LoadedModel::q35_scratch`. The global was removed; the sharing was relocated,
so M1b's own failure mode is still live and its exit criterion is unmeetable
today.

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

**M3d measurement 3 — MEASURED 2026-08-23, PASSES.** nix1/gfx1103,
Qwen3.6-35B-A3B--oq4, kvarn, `HIPFIRE_DAEMON_EXECUTOR=v2`. Bulk = 96 tokens,
realtime = 32 tokens at `priority: 9`. Per-stream `tok_s` read off each `done`
frame, three reps:

| rep | bulk solo | bulk loaded | ratio | realtime |
|---|---|---|---|---|
| 1 | 18.400 | 13.300 | **0.723** | 9.200 |
| 2 | 18.600 | 13.300 | **0.715** | 9.200 |
| 3 | 18.600 | 13.300 | **0.715** | 9.200 |

Comfortably above the **0.6x** floor, and reproducible to three decimal places —
so priority admission is not buying realtime latency by starving bulk.

Read the exit's wording carefully before quoting this number: it specifies **one**
realtime stream and one bulk stream. A first attempt used FOUR realtime streams
and got 7.100 tok/s loaded, a ratio of 0.386 — a clear "fail" against a criterion
it was not measuring. That figure is worth keeping as the scaling datapoint it
actually is (bulk keeps ~39% against 4x priority-9 competition), but it is not
measurement 3.

**M3d measurement 2 — NOT obtainable from the trace as it stands. Measured
2026-08-23.** This plan says measurements 2 and 3 "are obtainable today", and
§M6 adds that measurement 2 "does not need [a realtime class] — it needs an
ordering lever, and there is one". Both are wrong, for two independent reasons
that a trace dump makes plain:

- **Admission is not stream-attributed.** Every `dispatch_begin` and
  `dispatch_end` record carries `stream: None`; only `token_emitted` carries a
  real id. So the generate frame's dispatch — which IS the admission — cannot be
  tied to the stream it admitted.
- **No per-quantum event exists.** `TraceEvent` has no variant the march loop
  emits when it hands a stream a quantum, so "first dispatch" has nothing to
  anchor to even if admission were attributed.

Measurement 2 therefore needs instrumentation before it needs a run: stream-scope
the generate frame's `DispatchBegin` (or add an `Admitted` variant), and add a
per-quantum event. That is a small change, but it is a change — not a
measurement someone forgot to take.

**A sampling trap worth recording, because it produced a confident wrong
reading.** Under executor v2 a `generate` frame only ADMITS a stream; the march
loop runs only once the pending queue is empty (`pop_next() -> None`). So an
`executor_trace` frame sent in the same stdin batch is serviced BEFORE a single
token exists, and reports `token_count: 0` with no stream ids. Under v1, where
generate executes inline during frame dispatch, the identical request order
reports all 128 tokens. Comparing the two naively says "the v2 executor records
no tokens" — a serious-sounding defect that does not exist. Moving the request
after `unload` (which drains the streams) gives the true v2 picture:

    record_count=142  token_count=128
    events={dispatch_begin: 5, vram_sample: 5, dispatch_end: 4, token_emitted: 128}
    inter_token_gap p50 45.68 ms / p99 47.71 ms / max 50.67 ms

For a MID-run snapshot, `--listen` works where stdin cannot: the frame loop
services one pending frame per iteration and marches only when none remain
(`main.rs`), so a socket client's request is handled between march rounds.

**M3d measurement 2 — INSTRUMENTED AND MEASURED 2026-08-23. The 200 ms contract
is NOT met.** Two stream-scoped `TraceEvent` variants were added: `Admitted`
(recorded where `admit_generate` inserts the stream, `aux` = priority) and
`QuantumBegin` (recorded in both march paths, round-robin and batched).
`QuantumBegin` is the `Yielded`-shaped variant the enum reserved in a comment and
declined to define because "today nothing can construct it" — M3's march loop
constructs it. Both key on `request_id`, which is what `events::emit_text_bytes`
already uses, so admission, quanta and tokens form one correlatable series.
`exec_trace::admission_to_first_quantum_ns` derives the measurement and
`snapshot_json` reports it under `admission`.

Measured, bulk (priority 0) admitted FIRST and realtime (priority 9) second,
three reps:

| rep | realtime (p9) | bulk (p0) |
|---|---|---|
| 1 | **659.44 ms** | 1474.20 ms |
| 2 | **652.94 ms** | 1463.86 ms |
| 3 | **652.97 ms** | 1462.27 ms |

**Read the gap correctly: it contains the stream's OWN prefill.** `admit_generate`
records `Admitted`, and the same frame handler then calls `generate_start`, which
"frames and prefills" (`handlers/generate.rs`) before the march ever runs. So
`Admitted -> first QuantumBegin` necessarily spans that stream's own prefill, and
the raw gap is not a scheduling latency. Subtracting each stream's `prefill_ms`
from its own `done` frame:

| stream | gap | own prefill | scheduling delay |
|---|---|---|---|
| realtime (p9) | 659.4 ms | 647.0 ms | **12.4 ms** |
| bulk (p0) | 1474.2 ms | 762.1 ms | 712.1 ms |

**So the 200 ms contract is met, at 12.4 ms** — priority admission puts the
realtime stream on the GPU one quantum after its own prefill finishes, ahead of a
bulk stream admitted before it. The bulk stream absorbs the interference (712 ms),
which is the trade the priority lever exists to make.

An earlier revision of this section claimed the 659 ms was the realtime stream
queueing behind the bulk stream's prefill and concluded the contract failed by
3.3x. That was wrong: the two streams' prefills are 647 ms and 762 ms, and each
gap matches its OWN prefill to within ~13 ms. The lesson is that this metric is
only a scheduling number after its prefill term is removed, so
`admission_to_first_quantum_ns` should be read alongside `prefill_ms`, never
alone.

The residual hazard is narrower than "the contract fails", and it is still §M2a:
a prefill is indivisible and runs in the frame handler, so a realtime request
that arrives while ANOTHER stream is mid-prefill waits out that prefill — up to
~762 ms here. Pausing or migrating low-priority streams does not help in that
window, because there is no quantum boundary to pause at. Prefill lowering is
what shrinks it.

**The mid-prefill arrival case, measured over `--listen` (2026-08-23).** Every
earlier number here was taken over the stdin protocol, which drains every frame
before the march runs — so a realtime request could never arrive *while* a bulk
prefill was in flight, and that is the case the 200 ms contract is actually
about. `--listen` can express it: a second connection injects the request 2 s
into a bulk prefill.

Client-observed send -> first token for the priority-9 request, two reps:

| config | client TTFT | trace admission -> first token |
|---|---|---|
| today (both flags off) | **8516.7 / 8505.5 ms** | 454.0 / 451.5 ms |
| `MARCH_PREFILL` + `STRICT_PRIORITY`, band 16 | **575.2 / 571.0 ms** | 489.1 / 487.1 ms |

**14.8x**, and the contract goes from missed by 42x to missed by ~2.9x.

The two columns are the finding. With the flags off they disagree by ~8 seconds:
the trace says 454 ms because ADMISSION ITSELF was delayed — the realtime frame
sat unread in the channel while the bulk prefill occupied the frame handler, and
`Admitted` is only recorded once the daemon gets to it. A metric anchored at
admission cannot see a queue it never entered. With the flags on the two columns
close to within ~86 ms, because prefill runs in the march and the frame loop
regains control between bands.

So `admission_to_first_quantum_ns` is the right instrument for scheduling INSIDE
the executor and the wrong one for end-to-end latency; the client stopwatch is
the honest number for the contract. Both are reported above deliberately.

**M3d measurement 1 — MEASURED 2026-08-24. M3d is complete.** §M0's module
dimension is built: `TraceEvent::ModuleEnd` carries the `SuperOpKind`
discriminant in `module` and the duration in `aux`, and
`exec_trace::module_duration_stats` reports per-module percentiles under
`modules` in the trace reply.

It could not be the passive field §M0 described. `dispatch_super_op` only
ENQUEUES, so wall-clock around it measures launch cost and every module reads as
a few microseconds; the number this exit wants is how long a module OWNS the
device, because that is the floor on how long a yield must wait for it. Getting
it honestly means bracketing each super-op with a device sync, so this is a
measurement MODE (`HIPFIRE_TRACE_MODULES`, off by default) rather than
always-on instrumentation, and its durations are per-module GPU time rather than
a throughput figure.

Qwen3.6-35B-A3B--oq4, kvarn, 12 tokens, `dropped=0` so the window is whole:

| module | count | p50 | p99 | max |
|---|---|---|---|---|
| **moe** | 1000 | 0.326 ms | **0.526 ms** | **5.476 ms** |
| attend | 1000 | 0.027 | 0.090 | 3.353 |
| proj | 1000 | 0.406 | 0.666 | 2.841 |
| recurrent | 750 | 0.104 | 0.148 | 0.745 |
| residual_gemv | 1000 | 0.122 | 0.141 | 0.404 |
| norm | 750 | 0.017 | 0.032 | 0.340 |

**`Moe` owns the max, and the suspension floor is ~5 ms.** Read p99 and max
differently: across three runs the p99s are stable to the third decimal (moe
0.526 / 0.527 / 0.529) while the maxima wander 4.8-5.5 ms, as a single-sample
statistic should. The drain budget should be sized on p99 plus a margin, not on a
max that moves 14% run to run.

So the tightest drain budget this design can hold is **sub-millisecond at p99**
(0.67 ms, owned by `proj`) with a multi-millisecond tail owned by `Moe`. That is
the achievable suspension floor, and it is three orders of magnitude below the
200 ms contract — the contract is not limited by module granularity, which is
what §M2a's prefill work already implied and this now confirms from the other
direction.

The observer is installed rather than called: `hipfire-runtime` owns the trace
and DEPENDS on `hipfire-dispatch`, so the super-op loop measures and
`exec_trace::install_dispatch_module_observer` says where the numbers land.
Escapes collapse to one bucket deliberately — they are the coarse model-owned
blocks that must stay coarse, so their payload does not subdivide into
separately yieldable units.

*Breaks:* everything that assumed a forward runs to completion. `hipGraph` capture is one
indivisible quantum by construction — **off on the v2 path** until its WCET is declared,
since a declared WCET that ignores an enabled graph is exactly the failure the contract
exists to prevent. Cancellation moves from three per-token hook sites into the executor's
pick step — one site instead of three-and-counting, and finer.

*Revertible:* yes, behind `HIPFIRE_DAEMON_EXECUTOR=v2`.

### M4 — measured 2026-08-25: on this box the unfused path will not even LOAD

The decision §M4 poses — "is module granularity worth unfusing the default MoE
decode path?" — was framed as a throughput trade: unfusing costs "a D2H plus a
kernel launch per expert per token". On gfx1103 with `Qwen3.6-35B-A3B--oq4` it is
not a throughput trade, because the unfused configuration does not load.

    HIPFIRE_QWEN35_MOE_OQ_INDEXED=1 (default, fused)   pp512 20.80  tg128 20.70 tok/s
    HIPFIRE_QWEN35_MOE_OQ_INDEXED=0 (unfused)          OOM at layer 26/40:
        hipMalloc(1.03 MiB), free=13.9 MiB of total=43008 MiB

Reproduced twice, the second time on a verified-idle GPU (no daemon, 82 MB VRAM
in use) so it is the configuration and not contention from the previous run.

**Read this with its caveat.** `HIPFIRE_QWEN35_MOE_OQ_INDEXED` gates two things at
once: whether routing and expert compute are one kernel, AND the resident expert
layout — indexed implies `oq_moe`-repacked experts, while off falls back to the
`oq4_arch` combined layout (`moe_decode.rs:639`). So this measures the flag, not
"unfusing" in isolation, and the OOM is most likely the layout half. It still
answers the practical question: there is no way to *try* the unfused path at this
model size on this hardware, so any M4 work that depends on it cannot be verified
here — the same stop-line §3.4 of the prefill-lowering plan invoked, and the same
one that turned out to be misdiagnosed there, so it is worth someone checking on
gfx1151 before treating it as settled.

**The gfx1151 cross-check was attempted and is blocked on halo's configuration,
not on hardware.** Two runs, both arms failing identically at LOAD with "DFlash
draft requested but target lm_head quant_type=36 is not supported" — before
either MoE path executes, so neither run is a datapoint. Cause: halo's global
`dflash_mode` is `on`, its per-model `off` overrides name artifacts that are not
on that box (`Qwen3.6-35B-A3B--mq4`, `--oq4++`), and the only 35B MoE artifacts
present are `bf16` and `oq4.25++` — whose lm_head quant the paired drafter
(`Qwen3.5-35B-A3B--dflash.oq4+.hfq`) cannot serve.

**Correction (2026-08-25).** The sentence that stood here — "`HIPFIRE_DFLASH_MODE=off`
did not override it, so the per-model config layer appears to win over env here"
— was wrong, and was a guess published as a finding. There is no config-layer-beats-env
precedence involved. What the code actually says:

- The env layer is fine. `env_var_name_for_key` maps `dflash_mode` to
  `HIPFIRE_DFLASH_MODE`, and `off` is a valid `ConfigType::Enum` value, so it
  parses and lands in `HipfireConfig::dflash_mode`.
- **The daemon never reads that env var.** `dflash_mode` reaches it only inside the
  load request params (`hipfire-daemon/src/handlers/lifecycle.rs:225`), and
  `.unwrap_or("auto")` when the field is absent.
- Only clients that call `load_params_from_config` forward it — `hipfire-cli`'s
  chat path and `hipfire-server` (which correctly omits `auto` and sends `off`).
  **`hipfire-eval` does not**: it builds params from its own `DflashMode`
  (`--dflash`, default `Off`, `hipfire-eval/src/config.rs:43`), which never
  consults `HipfireConfig` and so cannot see the env var at all.
- Anything other than `off` then lets `load.rs:857` auto-pair a sibling drafter by
  filename — which is how the drafter got requested without anyone naming it.

So the env var did not lose a precedence fight; depending on which client drove
that load it either never reached the request or was never consulted. Which of the
two it was on halo is still unverified — settling it means capturing the actual
load request, not re-reasoning about layering. The blocked-on-halo conclusion below
is unaffected: it rests on the artifacts present on that box, not on this mechanism.

Finishing it needs one of: a per-model `dflash_mode: off` override added to
halo's config (a shared box, actively in use — ask first), a 35B MoE artifact
copied there that has one, or a run on a third machine. Recorded rather than
retried because two attempts already failed the same way for a reason unrelated
to the question, and a third would too.

What that left for M4, as of the 2026-08-22 scoping: qwen35 joins deepseek4 on a
coarse `Escape` with a capability predicate that NAMES the reason, and
`MoeExpert(e)` exists only for arches whose MoE is not fused.

**Measured 2026-08-26 — the cost objection to option B does not hold at batch-1
decode.** See `docs/experiments/2026-08-26-m4-unfused-moe-decode.md`. The
per-expert arm already exists in `moe_decode.rs` as the non-all-MQ4 / k != 8
correctness fallback, so B was measurable behind a flag rather than a new
kernel. On gfx1103 with `Qwen3.6-35B-A3B--oq4.hfq`, A/B/A at 5 reps:
fused **11.52** t/s vs per-expert **11.46** t/s — **−0.52%**, and the per-expert
arm pays a CPU top-K D2H sync the fused path does not. Batch-1 decode is
bandwidth-bound on expert weights, so the fused kernel's launch amortization
buys nothing.

So B is viable on qwen35 decode and `Escape` is not forced by throughput there.
Still open before M4 is settled: batch > 1 (where fusion should start to pay),
gfx1151's grouped WMMA path, and deepseek4 — which has no existing per-expert
fallback, so B there is real kernel work rather than a flag. The defensible
middle is B where a per-expert arm already measures free, `Escape` for
deepseek4 until its per-expert path is written and measured.

⚠️ **That experiment also found a benchmarking hazard on nix1**: after sustained
load the DPM governor drops `mclk` to level 0 (1000 MHz) at only 46 °C — not
thermal — and batch-1 decode tracks it, so a fresh run reports ~1.8× the
sustained number. The first A/B here read "unfusing costs 31%" purely from
comparing a fresh run against a warm one. Any A/B on this box must be A/B/A or
pinned to one DPM state.

### M4 — scoped 2026-08-22. qwen35 has deepseek4's problem too.

**Measured before starting.** This section already concedes that deepseek4 fuses
top-k into dispatch and must stay on a coarse `Escape`. It assumes qwen35 is
clean. **It is not.**

`moe_ffn_decode_impl` (`qwen35/moe_decode.rs:751`) is **1,200 lines** of
dtype-conditional dispatch, and among its fast paths is an *indexed* routed-expert
GEMV gated by `hipfire_dispatch::families::moe::oq_indexed_decode_active`
(`moe_decode.rs:639`, `:717`), described in its own comment as "the device-side
top-K + indexed expert GEMV path" (`:797`). Routing and expert compute are **one
kernel** there. Indexed routed-OQ MoE decode is also the DEFAULT (`0d425bfbf`),
per this plan's own Tier-3 note.

Splitting `Moe` into route / expert / combine means **unfusing** that: materialise
per-expert intermediates the fused kernel exists to avoid, and pay a D2H plus a
kernel launch per expert per token. That is a throughput regression bought to
gain suspension granularity — the same trade this plan rejects elsewhere (copying
gemma4's per-token prefill onto qwen35).

What survives: the seam is real on the *unfused* paths. `MoeScratchRef` already
materialises `router_logits` / `topk_indices` / `topk_weights`
(`moe_decode.rs:51-64`), and the CPU-top-K path takes a D2H sync per layer
(`:34`) which is a natural quantum edge.

**So M4 needs a decision this plan does not currently pose:** is module
granularity worth unfusing the default MoE decode path? If not, qwen35 joins
deepseek4 on a coarse `Escape` with a capability predicate that NAMES the reason,
and `MoeExpert(e)` exists only for arches whose MoE is not fused. Decide before
implementing; the split is not free and the fused path is the one that ships.

Note also that `MoeExpert(e)` must be indexed by top-k SLOT, not expert identity:
`LayerProgram` is built once at lowering time and which experts fire is per-token.
Emit `k` slots and resolve the physical expert inside the binding.

### M4 — Split `Moe` into `MoeRoute` / `MoeExpert(e)` / `MoeCombine` (original)

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

**Paged-expert bring-up on DeepSeek-V4-Flash, measured 2026-08-22/23 (nix1,
gfx1103, 43008 MiB).** Artifact:
`/srv/hipfire/staging/DeepSeek-V4-Flash-w13--attnq8.oq4.hfq` (85.3 GB) — experts
oq4 with the w1/w3/w2 role ranks, attention re-encoded Q8_F16 from the FP8 .hfa.
`HIPFIRE_DEEPSEEK4_PAGED_EXPERTS=1` registers 11008 routed expert modules and the
82.8 GB predecessor loads in 58 s on a 43 GB device. Three findings:

1. **The fused MoE entry test asked for the wrong thing.** It gated on
   `expert_gate_up_blob`, which paging deliberately does not upload, so every
   paged layer fell into a fallback that no longer exists. Fixed by gating on the
   ptr tables the dispatch actually reads.
2. **Expert admission is incompatible with HIP graph capture — RESOLVED as a
   mutual exclusion, 2026-08-24.** Not a bug that could be reordered away: a
   captured graph is a fixed sequence of device work containing no host
   decisions, and paged experts need exactly such a decision per MoE layer (read
   the router's top-k back, admit, patch the pointer table). The decision depends
   on device work computed inside the region it would have to precede, so no
   ordering satisfies both. `decode_step_with_graph` now detects the combination,
   skips capture in favour of paging, and says why — where it previously died
   mid-decode with `hipMemcpy D2H: operation would make the legacy stream depend
   on a capturing blocking stream`, a HIP rule rather than the actual conflict.
   Verified: the previously-fatal combination now runs to a natural EOS with the
   same output as the graph-disabled path (7 tokens, `finish_reason: stop`).
   Capturing only the non-MoE spans was rejected — many tiny graphs, most of the
   benefit gone, on a model whose MoE layers dominate.

   *(original wording)* **Expert admission is incompatible with HIP graph capture.** The residency
   hook does a blocking D2H inside the captured region and dispatch fails with
   "would make the legacy stream depend on a capturing blocking stream".
   Admission is a host decision; it has to be resolved *before* capture, not
   inside it. With `HIPFIRE_DEEPSEEK4_GRAPH=0 HIPFIRE_GRAPH=0 HIPFIRE_GRAPH_MOE=0`
   decode runs to completion with no error — and emits degenerate output
   (BOS x13). **That degeneracy is NOT a pager defect.** Traced with
   `HIPFIRE_DEEPSEEK4_DUMP_STATE`, it originates in PREFILL at layer 0, before any
   expert is touched: `hc_compute_control_batched` turns finite HC streams
   (absmean 0.000439) and finite `hc_fn` into an all-NaN control vector for
   exactly the 9 real token rows (`hc_c` nan=216 = 9 x 24), while the zero padding
   rows come out clean. NaN then propagates to the router scores, and a top-k over
   NaN fails every comparison and returns index 0 six times — which is why the
   pager admitted exactly one expert, expert 0, on every layer. Fix that kernel
   before drawing any conclusion about paged decode correctness.

   Four hypotheses were tested and rejected on the way: a stream-ordering race on
   the residency D2H (adding `device_synchronize` changed nothing), a tensor-view
   offset (`GpuTensor` has no offset field), the FP8->Q8_F16 attention re-encode
   (`layers.0.attn.wo_a.weight` dequantizes to min=-0.23438 max=0.25000
   absmean=0.020271, 0 non-finite), and zero embeddings feeding a norm (the 9 real
   embedding rows are healthy; it is the zero rows that survive).

   **Fixed 2026-08-23.** The eight HC globals were uploaded with
   `upload_global_raw` (bytes verbatim) while all three HC kernels declare their
   weight parameters `const __half*`. DeepSeek-V4-Flash stores them F32, so the
   bytes were reinterpreted as pairs of halves — and since a 16-bit word lands on
   the all-ones exponent about 1 time in 32, a 16384-wide dot product escapes NaN
   with probability ~(31/32)^8192. `upload_global_as_f16` now converts.
   `hc_head_scale` is excluded: it is host-read into an f32 field and passed by
   value. Every NaN in the chain above is gone (`hc_c` nan=216 -> 0, absmean
   0.637; `hc_x_in` nan=36864 -> 0; `l3_end_streams` nan=147456 -> 0).

   **With that fixed, paged expert routing works.** Same artifact, same box:
   1342 admissions across 254 distinct experts (was 40 admissions, expert 0
   only), per-layer top-k indices varied (`l3=[154,185,13,140,80,159]`,
   `l42=[151,55,232,160,198,240]`), scores probability-shaped and weights
   descending (`[0.613,0.444,0.348,0.275,0.274,0.246]`). Layers 0-3 are
   numerically healthy (end_streams absmean 1.32 -> 1.85, max ~20, no NaN), and
   generation runs to a natural EOS.

   **The remaining mojibake is expected, not a defect.** The artifact's routed
   experts are `qt=19` = MQ2G256Lloyd — 2-bit — even though it was built with
   `--format oq4`. That is deliberate: `hipfire-quantize` routes DeepSeek V4
   per-expert tensors "through the MQ2-Lloyd path for every DeepSeek-specialized
   format, including OQ dense formats", because the deepseek4 routed-MoE kernels
   are MQ2-Lloyd-specific and an OQ4 expert payload would be raw-concatenated and
   then read through the wrong kernel family. `--format oq4` therefore applies to
   attention, shared experts and head (all `qt=34`) but never to routed experts.
   MQ2's documented failure mode in this repo is precisely mojibake — the
   `HIPFIRE_ALLOW_MQ2` gate exists because "the uniform 4-level codebook collapses
   at every model size validated locally". A 2-bit routed-expert MoE producing
   incoherent text is the format behaving as characterised, not the executor
   misbehaving.

   The Q8_F16 attention re-encode is also cleared on reasoning rather than
   measurement: MQ4/HFQ4 weights are STORED FWHT-rotated, which is why that arm
   feeds the rotated activation buffer; Q8 weights come from plain FP8 source, so
   the plain buffer is the correct pairing. There was never an Oq4 arm for `wo_a`
   at all — that is the error the re-encode was done to clear.

   **This does not block M5's exit.** The exit is a PARITY test — paged output
   byte-identical to pinned output — not a quality test. Mojibake is admissible
   evidence so long as it is the same mojibake. Coherent output would need routed
   experts above 2 bits, which needs a deepseek4 routed-MoE kernel family that
   speaks a wider format — a separate piece of work from the executor.

## M5 exit — RUN, root-caused, FIXED, and PASSING (2026-08-23, nix1)

**Result: the exit passes.** Output is byte-identical across a 4.3x swing in
eviction pressure and a 6x swing in budget:

| arm | budget | admissions | evictions | tokens | md5 |
|---|---|---|---|---|---|
| reference | 24 GB | 4474 | 1084 | 22 | `552280dbd78cad3df56d9acd48f71828` |
| 3x oversubscribed | 8 GB | 5846 | 4716 | 22 | `552280dbd78cad3df56d9acd48f71828` |
| 6x oversubscribed | 4 GB | — | — | 22 | `552280dbd78cad3df56d9acd48f71828` |

The 8 GB arm reproduces exactly on a repeat run. Touched set is ~23 GB, so 8 GB
is ~3x oversubscribed as the exit requires.

**Root cause of the first, falsified attempt: PREFILL COULD NOT ADMIT EXPERTS.**
`MoeBiasAwarePrefillParams` had no `expert_residency` field at all — the decode
twin has had one since the pager landed. So a paged model dispatched its entire
prompt pass against whatever the device pointer table happened to hold: entries
only for experts some *earlier decode* had admitted, and null for everything
else, since eviction nulls slots. Output therefore depended on eviction history,
which is exactly what the first A/B measured. The field, a union-over-rows
admission in `run_moe_prefill_bias_aware` (prefill routes B tokens
independently, so the set is up to B x k_top and needs dedup), and the wiring in
deepseek4's `ffn_batched` close it.

Two consequences worth keeping:

- The fix is confirmed by an OOM, not just by parity. With prefill admitting its
  real working set, `PinAll` now exhausts VRAM at layer 15
  (`hipMalloc 6.75 MiB, free=18.8 MiB of 43008`). Before the fix it fitted only
  because prefill silently used a fraction of the experts it needed. The pinned
  reference is genuinely unavailable for this model on a 43 GB box — the
  large-budget LazyLru arm stands in for it.
- Of the two pre-fix arms, the `PinAll` one (16 tokens, `88570cc71755...`) was
  the WRONG one. Every corrected run at every budget agrees on
  `552280dbd78c...`, the value the pre-fix LazyLru arm happened to produce.

### The falsified first attempt, and the hypotheses it cost

An earlier note here claimed the exit needed halo because 85.3 GB does not fit
pinned in 43 GB. **That was wrong**, and it cost a milestone. `ResidencyPolicy`
already names the reference: `PinAll` IS "pinned" (§M5 Phase 1 — "PinAll is a
name not a sentinel"), and it runs on this box. So both arms of the parity test
are executable here:

| arm | policy | admissions | evictions | tokens | md5 |
|---|---|---|---|---|---|
| A (reference) | `PinAll` | 3859 | 0 | 16 | `88570cc71755cfbf07171689d5183f8a` |
| B | `LazyLru` 9.1 GB | 5725 | 4440 | 22 | `552280dbd78cad3df56d9acd48f71828` |

Working set is 3859 x 7077888 B = 27.3 GB, so a 9.1 GB budget makes the touched
expert set exactly **3x the pager budget**, which is what the exit asks for. The
budget is respected (9088008192 bytes, 1284 modules resident) and eviction really
runs (4440 evictions).

**This attempt was falsified: the outputs differed.** Both arms were individually
reproducible — A twice at `88570cc7...`/16 tokens, B twice at `552280db...`/22
tokens — so this is systematic, not noise. Per the exit's own wording, this is
reported rather than tuned around.

Three eviction-path hypotheses tested and REJECTED, so they are not retried:

- *`patch_expert_ptr_table` is never called.* True — it has zero call sites — but
  it is a superseded pull-based API. Admission publishes through
  `write_expert_ptr_slot` ("push hook 1 of 3"), so the table is maintained.
- *Eviction leaves a stale pointer.* It does not; "push hook 2 of 3" nulls the
  slot BEFORE freeing the buffer, deliberately, with the reasoning in-line.
- *A selection evicts its own experts.* `ensure_resident` admits k_top one at a
  time with nothing pinning the earlier ones, so admitting expert 6 could evict
  expert 1 and null its live slot. Instrumented with a post-admission residency
  assertion: **0 hits** over the whole run. Newly admitted modules sit at the LRU
  back, so with 1284 resident the current six are never the eviction candidates.
  The assertion is kept as a cheap guard.

Two further suspects were then checked and cleared before the real cause turned
up: dropping `_handle` from `transport.fetch` (the transport is synchronous and
`TransferHandle` is an explicit forward-compat no-op) and a short
`copy_len < len` leaving an uninitialised tail in a recycled pool buffer
(`read_into_staging` returns `(rel, len)` and errors on short reads).

The clue that broke it open was per-prompt: with two prompts, e0 was IDENTICAL
across arms and only e1 diverged — pointing at table state carried between
requests rather than at the expert data itself.
3. **M5's exit cannot be run on nix1 for this model.** The exit asks for output
   "byte-identical to the same model run pinned"; pinned, this artifact OOMs at
   layer 19 (`hipMalloc 1152 MiB, free 521.9 MiB of 43008`). There is no resident
   reference on this box, so the A/B needs either halo (128 GB / ~120 GB GTT,
   where 85.3 GB fits pinned) or a smaller MoE whose pinned form fits in 43 GB.
   Pick one before treating M5 as measurable here.

Also landed for this: `--tensor-source` now accepts FP8 E4M3 archives, without
which the attention repair was impossible — DeepSeek ships FP8, and the surgical
re-encode path refused every tensor in the .hfa.

### M6 — priority admission landed 2026-08-22; classes still to come

**M6 was mis-scoped as blocked.** Its core dependency is lossless suspension,
which §M3c delivered — not M4 or M5. The ordering half is now in.

`WorkloadSpec` has carried a `priority: u8` all along and admission hardcoded 0,
so latency-sensitive work had no way to say so. Admission now reads `priority`
off the wire and `StreamTable::runnable()` returns highest-priority first (stable,
so equal priorities keep admission order). Measured, with bulk admitted FIRST:

```
dispatch order: R R b R b R b R b R b b b b b b ...
```

The `priority: 9` stream is dispatched ahead of a 60-token bulk stream that was
admitted before it, finishes early, and bulk runs throughout. Flag-off is
byte-identical and the §M3b1 exit is unchanged (default priority still fair-shares
`ABAB…`).

**Deliberately NOT done: the four `WorkloadClass` variants.**
`SpeechIn`/`SpeechOut`/`VideoIn`/`VideoOut` are worth adding when they carry
contracts and something consumes them; four variants whose only effect would be
ordering that `priority` already provides is scaffolding. The declared
largest-indivisible-unit and max-yield-granularity fields, the drain-budget and
the VRAM test are the substance of M6 and remain.

**Scoped 2026-08-22 — the remaining three, measured rather than assumed.**

*The drain budget's formula omits the term that dominates.* §1.1 states

```
admit(realtime) ⟺ drain_to_suspend + realtime_model_residency_cost ≤ 200 ms
```

§M3d measurement 2 measured the thing that inequality is supposed to bound:
**81.4 ms** admission→first RT token, of which **~71 ms is RT's own prefill** and
only ~10–17 ms is queueing. Prefill appears in neither term. So the inequality as
written would admit on a budget accounting for ~12% of the observed latency, and
would keep passing as prefill grew with context length. It is not implementable
as specified — it needs either a prefill term or a preemptible prefill, and
`2026-08-22-prefill-lowering.md` argues the second. **Blocked on that decision,
not on M4.**

*The VRAM test is real and cheaper than expected, but it is a load-path change.*
The assumption that one model is resident is **wrong**: `DaemonState` carries
`resident_models: HashMap<String, LoadedModel>` of parked workers, documented as
"there is no eviction policy" (`state.rs:40-42`). So unbounded residency growth
is a live hazard, not a hypothetical one, and the test guards something.
Accounting is available — `gpu.hip.get_vram_info() -> (free, total)`
(`hip-bridge/src/ffi.rs:1764`), reachable directly off `DaemonState.gpu`, with a
reserve-subtraction precedent in `vram_ceiling` (`layer_stream.rs:2846`).
What makes this more than a small patch is that the useful guard **refuses a
load**, which is a user-visible behaviour change to a path outside the executor.
Wants a decision before it lands.

*The declared granularity fields stay unbuilt, for the reason this section
already gives.* `largest_indivisible_unit` and `max_yield_granularity` have no
consumer until a forward loop can act on them. The yield point now exists —
`run_layer_program_from(.., start, budget)` — but nothing calls it with a real
budget, and that caller is M4's. Adding the fields first is the same scaffolding
this section declined for the four `WorkloadClass` variants.

**Net: M6's remainder is decision-blocked, not effort-blocked.**

**This also unblocks §M3d measurement 2** (admission→first dispatch under load),
which was previously recorded as unobtainable for want of a realtime class. It
does not need one — it needs an ordering lever, and there is one.

### M6 — Realtime admission and the stub classes (original)

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

**Measured 2026-08-22 (nix1, gfx1103, Qwen3.6-35B-A3B--oq4, kvarn, max_seq 1024).**
`HIPFIRE_DAEMON_EXECUTOR=v2` + `HIPFIRE_DAEMON_EXECUTOR_BATCHED=1`, one fused
`forward_prefill_grouped_moe_session_batch` per march over all runnable streams.
Decode-only throughput (wall minus a separately measured 32.3 s load), two reps:

| N | round-robin tok/s | batched tok/s | ratio |
|---|---|---|---|
| 2  | 26.95 / 22.32 | 19.46 / 22.91 | ~1.0 (noise) |
| 4  | 20.83 / 21.21 | 22.26 / 26.13 | 1.15x |
| 8  | 13.51 / 12.12 | 23.04 / 24.18 | 1.84x |
| 16 | 10.53 / 8.05  | 14.57 / 11.40 | 1.40x |
| 32 | 5.79 / 5.12   | 7.45 / 7.63   | 1.38x |
| 64 | 5.08 / 5.24   | 7.58 / 7.59   | 1.47x |

**Crossover N is ~4**, well below the N whose KV fits in VRAM (N=64 ran
comfortably), so the capacity thesis is **not falsified**. Round-robin falls off
steeply with N (26.95 -> 5.08 tok/s) while batched stays roughly flat above N=8
(23-24 tok/s at N=8, 7.6 at N=64) — the batch is absorbing the per-stream cost
the round-robin march pays serially.

Two measurement traps worth recording, because both produced a *plausible*
number:

- The N=4 entry in a first sweep showed batched LOSING (15.8 vs 17.6 tok/s).
  That sweep used 16-token streams, where decode is ~3.5 s against a ~32 s load;
  load-time variance alone is +/-60% at that scale. Re-measured with 96-token
  streams the sign flips. Do not measure this at small N with short streams.
- Batched output is bit-deterministic (identical md5 across reps at every N) and
  byte-identical to round-robin at N<=16 with short streams, but diverges from
  round-robin in 4/16 streams at 96 tokens and at N>=32. The divergent
  continuations are coherent near-tie swaps, and both modes are individually
  reproducible, which is consistent with float non-associativity in the batched
  GEMM amplified by autoregressive feedback rather than state corruption —
  *consistent with*, not proven. Bit-exact parity with sequential decode is not
  an M7 exit criterion, but it is not established either.

`HIPFIRE_BATCH_PROBE=1` reports the row count from inside the fused arm. Use it:
three separate defects in this path each produced byte-identical output while
the fused arm never ran, ran on half the tokens, or crashed after emitting a
plausible prefix (see the commits on `m7-batch-driver`).

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

- **Interactive:** ~~`~/.hipfire/models/qwen3.5-0.8b--oq4++.hfq` — on disk, arch 5
  (dense)~~ **INVALID, corrected 2026-08-11.** That artifact loads through the
  qwen3.5-VL text wrapper (`qwen3.5-vl text wrapper: mrope_interleaved=true` at
  load), so `is_qwen35_dense_arch_id` is false and it never reaches a fused dense
  backend. Verified by launch counts: width-4 decode issues 4.00x the launches of
  width 1, i.e. a per-row loop. Re-quantizing without AWQ does not help, and
  neither does the `-Base` variant — the quantizer reports
  `Architecture: qwen3_5 (id=5)` for it while the runtime still VL-wraps it (see
  BUGS.md). The reasoning below — "dense is correct here: it isolates the
  scheduling claim from the MoE claim" — still stands, but it needs a model that
  is actually dense at runtime, and no Qwen3.5 variant on hand is.
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

## Status, 2026-08-20 — prerequisites re-checked against master

The 2026-08-09 prerequisite list is stale. Re-checked item by item at
`20c02555b`. Method: grep plus the doc comments at each site, NOT a read of the
surrounding logic — several "cleared" marks below rest on a comment saying "this
used to be a static", which is strong evidence but not a proof of correctness
under interleaving.

### Tier 1 (blocks correctness) — 3 of 4 cleared

| item | state |
|---|---|
| process-global sampler RNG (`sampler.rs`) | **CLEARED** — "used to be `static SAMPLER_STATE: AtomicU32`" |
| missing lease reaper (`hipfire-scheduler`) | **CLEARED** — `reap_expired` / `LeaseGuard` present |
| `RAW_OVERRIDE` (`serving-core/src/model.rs`) | **CLEARED** — "used to be a `thread_local RAW_OVERRIDE`" |
| `upload_raw` / `GpuPool` asymmetry | **PARTIAL** — see below |

`Gpu::upload_raw` still calls `self.hip.malloc()` directly
(`dispatch/mod.rs:2169`) while `upload_raw_pooled` (`:2212`) uses the pool. The
paged-expert call sites that made this bite are fixed (PR #253 and the
independent `c6c06b27a`), and the M1a example passes — but **the asymmetric API
survives, so any new caller reintroduces the leak.** The plan's structural fix
(fixed-frame slabs) is what actually closes it; until then this is a live trap
rather than a cleared prerequisite.

### Tier 2 (blocks the latency goal) — mostly open, and one dominates

**Re-verified 2026-08-25 against the tree, item by item.** Two rows had gone
stale in the closing direction and are corrected below; the one that dominates
has barely moved.

| item | state |
|---|---|
| **prefill not lowered** | **OPEN — still the critical-path blocker.** `prefill_chunk.rs` is no longer zero: it now carries 11 `SuperOp`/`LayerProgram`/`Qwen35Prefill*Bindings` references, because the DeltaNetMoe and FullAttnMoe branches were folded onto the shared lowered super-ops (`0bbbfd08f`, -1834 lines). But **`prefill_batch.rs` is still zero across 6,426 lines**, and the hand-path total is 13,954 lines against the 14,200 this audit first cited — a ~2% dent. The shape improved; the blocker did not |
| `hipEventQuery` / `hipStreamQuery` | **CLOSED** — 5 occurrences in `hip-bridge/src/ffi.rs`. The async-prefetch prereq is in |
| `hipfire_steer` globals | **OPEN** — `static SESSIONS`, `static ACTIVE`, `static EPOCH` at `hipfire-steer/src/lib.rs:265/274/289` |
| `load_progress::SINK` | **OPEN** — still `static SINK: Mutex<Option<Box<ProgressFn>>>` at `load_progress.rs:34` |
| the four hand-path escapes | **REDUCED, unchanged since** — RoughQuant is dormant behind `HIPFIRE_RQ_HAND=1` (`loading.rs:3036` calls the hand path "broken"); GDN tape capture and a live steer session still force it |

**Prefill is the one that matters.** ~14.2k lines of hand-written control flow
across the two files is, by definition, one indivisible quantum — which defeats
the whole premise, since v2's claim is suspension *between modules* and an
unlowered prefill cannot be suspended at all. Everything else in this tier is
small by comparison; `hipEventQuery`/`hipStreamQuery` is ~15 lines.

### M0 (the instrument) — BUILT at event granularity, NOT at module granularity

**Corrected 2026-08-21.** This section previously read "not built. No
`executor_trace` in the daemon." That is no longer true, and the half that IS
true matters more than the half that changed.

Built: `hipfire-runtime/src/exec_trace.rs` exists and the daemon records into it
— `state.rs`'s `Responder::trace_frame` records `Completed`, `main.rs` records
dispatch, `serving-core::events` records token emissions — and it is readable
over the protocol via the `ExecutorTrace` request (`handlers/status.rs` →
`snapshot_json`). `TraceEvent` has five variants: `DispatchBegin`,
`DispatchEnd`, `TokenEmitted`, `Completed`, `VramSample`.

**The module dimension was BUILT 2026-08-23** (§M0 module dimension, PR #338).
This section previously said it was not, and that every `record()` call site
passed 0. Both are now false:

* `TraceEvent::ModuleEnd = 7` carries a real `SuperOpKind` discriminant, recorded
  at `exec_trace.rs:489` — the one call site that passes a non-zero `module`.
  Every other site is event-level and still passes 0, correctly.
* `module_duration_stats()` reads it back as per-module percentiles, and
  `module_name()` names the discriminant.
* The measurement lives in `hipfire-dispatch`'s super-op loop and reaches the
  trace through `install_dispatch_module_observer()`, called once from
  `daemon/main.rs:1609`. The indirection exists because `hipfire-dispatch`
  cannot depend on `hipfire-runtime` — the dependency runs the other way.

**Caveat that keeps this honest:** the observer is gated on `HIPFIRE_TRACE_MODULES`
and is **off by default**, because timing each module serializes the loop. So
§M3d's first exit measurement is now *obtainable*, but in a diagnostic mode
rather than from ambient production traffic. An always-on version wants
`hipEvent` timing instead of the serializing path — and `hipEventQuery` is now
in the bridge (see Tier 2), so that is no longer blocked either.

Measurements 2 and 3 (admission→first-dispatch, loaded-vs-solo bulk throughput)
were event-level all along and were taken in §M3d.

### Tier 3 — one item improved

Indexed routed-OQ MoE decode is now the DEFAULT (`0d425bfbf`), where the plan
listed "indexed expert layout opt-in on arch 6" as an M5 blocker.

### So: what actually blocks starting?

**Nothing blocks starting.** M0 is additive and is already the declared first
stage; Tier 1 is one API-shape trap away from clear. What blocks *finishing* —
specifically the latency claim that justifies the whole design — is prefill
lowering, and that is large enough to be its own project.

A defensible order, given the above: M0 (additive, and unblocks every other
exit criterion) → `hipEventQuery`/`hipStreamQuery` (~15 lines, hard prereq for
async prefetch) → the three remaining process globals (they become *wrong*, not
merely ugly, the moment two streams interleave) → `upload_raw` slabs → prefill
lowering.

**Where that order stands, 2026-08-25.** The first two steps are done: M0 has
both its event and module dimensions, and `hipEventQuery`/`hipStreamQuery` are
in the bridge. The next unclaimed step is therefore the process globals
(`hipfire_steer`'s `SESSIONS`/`ACTIVE`/`EPOCH`, `load_progress::SINK`), then the
`upload_raw` slabs, then prefill lowering — which remains large enough to be its
own project and is now concentrated almost entirely in `prefill_batch.rs`, the
fused multi-session path that the MoE consolidation did not touch.
