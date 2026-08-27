# Qwen3.8-27B on halo (gfx1151) — next phases

Scope is deliberately narrow: **this model, this GPU, greedy decode.** Nothing
here is claimed to generalise to another artifact or another card.

Where things stand after 2026-08-24/25: spec decode went **5.75 → 57.9 tok/s**,
past the 55 target, verified against a control with byte-identical output.
Landed: `910664f21` (DDTree Opus lm_head arm), `61c300992` + `c8b16753b`
(compact GDN tape), `bce8c84de` (multi-column drafter GEMV), `10beb8964`
(multicol widened 16→32), `0bb615d22` (batched Opus draft lm_head), `890d350c0`
(FA layers batch under a tape — the 2.08× step, plus the kernel-trace
instrumentation).

## Baseline — reproduce before trusting any number

```sh
M=~/.hipfire/models/Qwen3.8-27B--oq4.25++.hfq
D=~/.hipfire/drafts/Qwen3.8-27B--dflash2.oq4+.hfq.parked-slower-than-plain-decode
export HIPFIRE_OQ_COMPACT_MULTICOL_WIDE=1 HIPFIRE_LMHEAD_TWOSTAGE=q2 \
       HIPFIRE_GRAPH=1
# NB: do NOT set HIPFIRE_KV_MODE=asym3 here. The numbers below are Q8 KV, which
# is this harness's default AND the faster tier (see the correction at the end).
./target/release/examples/dflash_spec_demo --target $M --draft $D \
   --prompt "Write the numbers 1 through 30 as a comma-separated list." --max 96
```

Expected, 96 tokens:

| prompt | tau | tok/s |
|---|---|---|
| numbers 1..30 as a list | 5.733 | **57.9** |
| first 20 primes | 5.400 | 56.7 |
| JSON months | 4.556 | 47.0 |
| `def quicksort(arr):` | 2.654 | 28.6 |
| B-tree index explanation | 2.630 | 28.1 |
| MIT license header | 2.000 | 24.1 |

**Always bench a prompt MIX.** A single prompt has produced a wrong conclusion
here more than once — most memorably "55 is unreachable, the drafter is too
big", derived from one prose prompt at the worst tau in the set.

Control for any spec-decode change: `HIPFIRE_COMPACT_GDN_TAPE=0` gives 16.5
tok/s on the same binary with **identical generated text**. If a change moves
tok/s, diff the text against the control before believing it.

## Phase 1 — instrumentation sweep

Cheapest work here, and on the evidence the highest prior of another large win.

Four separate ~2× wins in one session were **the same defect**: a dtype ladder
or admission predicate that stopped at the MQ/HFQ families while the Opus
artifact (`OqCompactG256`) silently took a slow branch.

1. `run_dflash_draft_for_{logits,topk_gpu}` lm_head ladder — no Opus arm, so
   DDTree could not run at all (`910664f21`).
2. The lowered qkvza ladder tested `gdn_tape.is_some()` **above** its compact
   arm, so a tape-capturing forward read compact blocks as HFQ4G256
   (`61c300992`).
3. `spec_step_dflash`'s draft lm_head gate listed no Opus dtypes, so it looped
   B−1 full-vocab GEMVs per cycle (`0bb615d22`).
4. `prefill_chunk`'s FA admission passed `gdn_tape.is_none()` as
   `allow_compact` — a second copy of a guard already fixed elsewhere — so all
   16 FullAttn layers ran per-token (`890d350c0`).

**None of these showed up in a correctness gate, a phase timer, or a tok/s
number.** Each needed hand-instrumentation to find. #4 was ultimately found by
a diagnostic that had been in the tree the whole time behind a flag nobody set.

`hipfire_rdna::kernel_trace::record_fallback(site, detail)` now exists for
exactly this and prints a `SLOW PATHS TAKEN` section. **Only two sites are
instrumented.** The work:

- Annotate every remaining `_ =>` / `else` fallback arm across the dispatch
  families and the arch ladders. Cost is one relaxed atomic load when tracing
  is off, so be liberal.
- Run the prompt mix under `HIPFIRE_KERNEL_TRACE=1` and fix whatever announces
  itself.
- The `shaped traffic` section ranks by **weight bytes**, which is the ordering
  that finds these — call counts mislead badly (a `[248320, 5120]` lm_head at
  8 calls/cycle outweighs thousands of small projections). It did **not** fire
  in the last check; make it print. `record_shape` is wired only at the
  single-column compact GEMV in `hipfire-dispatch/src/families/gemv.rs`.
- `[pbs-gate]` now prints every term of the eligibility conjunction rather than
  just the verdict. Extend the same treatment to other multi-term gates.

## Phase 2 — tau is the binding axis

Cycle time is now flat across prompts, so throughput *is* tau: 57.9 at
tau 5.73 versus 24.1 at tau 2.0. Hard prompts are less than half the friendly
number purely because the drafter is wrong more often.

### (a) DDTree's redundant second verify

`spec_step_ddtree_batched` step 8 runs a **second** `verify_dflash_block` to
capture the GDN tape. At topk=1 `topk1_is_committed_prefix` holds and it
re-verifies the top-1 chain (byte-exact with baseline). At topk>1, an accepted
rank>0 branch is absent from the top-1 block, so it re-verifies the committed
path — **the second verify gets more expensive exactly when the tree does its
job**. Measured: topk=4/budget16 landed at 12.81 against chain's 25.61, almost
exactly half.

The tree verify already receives `gdn_tape` and captures rows for every tree
node, and `gather_accepted()` sits a few lines below. Gathering the committed
path's rows should replace the re-verify.

**Hazard the code documents:** same-depth siblings race in the KV cache and the
last write wins regardless of which sibling was committed, so the second verify
is also fixing KV slots — it is correctness-load-bearing, not just tape
plumbing. `HIPFIRE_DDTREE_PATH_B_CAPTURE=1` is a half-built attempt at exactly
this (gather + per-commit RoPE + quant-write on pre-RoPE K); read its
token-attractor warning before trusting any number from it.

Counters: `HIPFIRE_DDTREE_TAPE_DUMP=1` reports fast/slow per cycle. At topk=4
the slow path fired only 2 of 14 cycles, so the second verify is *not* the
whole story at that width — tree node count drives cost too.

### (b) Markov head / DSpark — not implemented, only scouted

`DsparkWeights` carries `markov_w1` / `markov_w2` `[vocab, rank]` F16 plus
`cfg.markov_rank`, with GPU-vs-CPU parity examples asserting token-identical
greedy output (`hipfire-arch-llama/examples/qwen3_dspark_parity.rs`).

Why it is not reachable here:

- The loader `hipfire-arch-llama/src/dspark_body.rs` targets the **Qwen3-8B**
  sidecar.
- The serving hook keys off `m.dspark` for **arch 0/1**; this target is
  qwen35-family **arch 5/6**.
- There is **no `.dspark.hfq` for Qwen3.8-27B on disk** — only `gemma3-4b` and
  `medgemma-27b` datasets, plus a `gemma3-4b-dspark.dsck` calib.

So it needs a trained sidecar *and* qwen35 wiring. Prior art worth reading
first: memory `project_dspark_lr_schedule` — 27b drafter non-convergence was
root-caused to a **missing LR schedule**, not a gradient bug; warmup+cosine took
loss 23.17 → 11.12 and acceptance to 0.613.

Why it is worth the lift: at rank *r* the head costs ~`r * vocab * 2` bytes —
tens of MB against the DFlash2 drafter's 1.18 GiB — so drafting becomes
near-free. That also makes **large B affordable**, and verify at B≤32 is now a
single weight sweep since multicol was widened. Cheap drafting plus wide verify
is the combination that raises tau without raising cycle cost.

`--ngram` / `--pld` do **not** substitute: they augment the DFlash drafter
rather than replacing it, so the 1.18 GiB sweep still happens. Measured
identical (25.59 / 25.54 / 25.76 friendly, 16.98 / 16.97 / 17.00 hard).

## Phase 3 — bandwidth tail

**~1.5× left, not 2×.** At tau 5.733 the cycle is 99ms moving ~15.6 GiB
(14.4 verify + 1.18 drafter) = **158 GB/s against a ~233 GB/s ceiling, 68%**.
Perfect bandwidth would be a 67ms cycle ≈ 85 tok/s.

`gemv_oq_compact_multicol` benched at **96.3% of ceiling in isolation** on real
27B shapes, so the GEMV itself is close to done. The target is the non-GEMV
tail the post-fix trace shows:

```
gemv_oq_compact_multicol_w8   52.00%   <- the verify; irreducible weight sweep
gemv_oq_compact_multicol_w23  20.47%   <- 23-token seed prefill, amortizes
gated_delta_net_f32            6.89%
__amd_rocclr_copyBuffer        3.59%
gemm_dflash_oq4_plain_...w8    3.55%   <- drafter, was 13.8%
oq_compact_x8_transpose        3.13%
fused_rmsnorm_mq_rotate_awq    1.28%
```

Also: **the verify lm_head is batched but not two-staged.** `q2` measures
**2.11× on this exact `[248320, 5120]` head at 323/323 recall@1** — see the
sweep in `lmhead_twostage_cfg`, and `examples/verify_lmhead_twostage_compact.rs`.
`q4` explicitly cannot pay for itself (a q4 tier is 637 MB against the head's
own 675 MB). `lmhead_twostage_applies` accepts `OqCompactG256` and
`lowered.rs:803` already calls it for single-token decode; the spec verify does
not. Worth ~6% of cycle bytes.

## Traps, all paid for

- **`HIPFIRE_SPEC_PHASES` timers are unusable for attribution.** They insert
  syncs, and once reported cycle total *rising* 218 → 306ms across a change
  that *raised* throughput 11.67 → 15.68 tok/s. Use `rocprofv3 --kernel-trace`
  or `HIPFIRE_KERNEL_TRACE=1`.
- **A drafter is silently ignored under KVarN** unless
  `HIPFIRE_KVARN_BATCHED_PREFILL=1`. The log warns, but tok/s is plain AR and
  looks like a real spec-decode number.
- The `tree-verify requires the batched-FA-eligible prefill path` assert
  **misnames its cause** — it blames KV tier and FA weight dtypes; the actual
  declining term is `gdn_tape.is_some()` via `allow_compact`.
- **Run both gates and read the exit code before committing**:
  `cargo test --workspace --lib` *and* `./tests/no-gpu-ci.sh`. Adding an env
  var requires `cargo run -p hipfire-cli -- gen-env-docs` or CI fails on stale
  env docs. Three commits went out on a red gate this session.
- `dflash_spec_demo` does **not** self-lock; coordinate with `hipfire lock`.
- `hipfire eval` caches on model+prompt+binary_hash, **not env** — env-only A/Bs
  replay arm 1. Use `--force`. Identical numbers are a cache hit, not a null
  result.


---

# MEASURED 2026-08-25 — two of the three phases were aimed wrong

Phases executed against this document. Recording what the measurements said,
because two of the targets it names are not where the time is.

## Phase 1 — done, and it found a real bug

117 `record_fallback` sites landed (five parallel agents, disjoint file sets).
On the full workload exactly TWO fired, and they were causally linked:

```
11x  dflash verify: no captured graph, direct dispatch  [embd=F32 ...]
 1x  qwen35 embedding: expanded to F32  [quant_type=49 (fast formats: 6/7/3)]
```

`embed_tokens` is quant_type 49 (`Bf16Lut3`) — a LOSSLESS bf16 recoding — and
both qwen35 load paths expanded it to F32: 4.74 GiB resident instead of 2.37,
12% of the model's footprint. `hipfire-arch-qwen2` had solved this since it was
written. Fixed (`f88863593`): VRAM 22207 -> 19783 MB, output identical, tok/s
flat (an embed lookup reads one row per token — always a footprint bug).

## Phase 2a — the named target is NOT the cost

This document says DDTree's redundant second verify is the Phase 2 target. It
is not. Instrumented and priced at topk=4: **2 firings in 12 cycles**, which
cannot explain a 2.5x gap against chain.

The actual cost was per-column scaling in `gemv_oq_compact_multicol`, and part
of it was self-inflicted: widening it 16->32 (`10beb8964`) dropped RW from 3 to
1 past B=16 on a REGISTER-PRESSURE ARGUMENT that was never benched. Measured,
weight bytes constant at 45.2 MiB:

```
B      RW=1      RW=2      RW=3
17    1.032ms   0.390ms   0.549ms
24    1.458     0.533     0.958
32    1.942     0.735     2.446
```

RW=1 was 2.6x slower. Fixed to RW=2 (`e1391d193`): tree verify 12.81 -> 33.35
(budget 16), 9.12 -> 26.29 (budget 24).

Trees STILL lose (chain 57.61) but now for an honest reason: tau moves only
5.733 -> 5.929 (+3.4%) against ~1.7x per-cycle cost. That is a property of this
drafter — its alternative branches rarely rescue a rejection — not a kernel
artifact. The earlier "tree verify loses here" verdict was measured on a kernel
I had crippled and should not be cited.

## Phase 2b — the Markov head is NET NEGATIVE here, do not build it

The premise in this document is "at rank r the head costs tens of MB vs the
drafter's 1.18 GiB, so drafting goes near-free". That premise was written when
the drafter was ~13.8% of GPU time and re-reading its weights B times. The
multicol drafter fix (`bce8c84de`) already collected that win. Measured now:

```
>>> all dflash (drafter) kernels combined: 5.10% of GPU time
```

So a Markov head that makes drafting ENTIRELY FREE wins at most 5.10%, and only
if tau holds — which it will not. A low-rank Markov head is a far weaker
predictor than a 5-layer DFlash2 head; tau falling from 5.733 toward 2-3 costs
40-50%. The "large B becomes affordable" half does not hold either: verify at
B=32 is 27.5% of ceiling even after the RW=2 fix.

The training stack is real and mostly arch-agnostic if someone wants it later:
`train_dspark_loop` touches the target for exactly `embed_tokens` and `lm_head`,
so `LlamaWeightsF32` is just a container; only LABEL GENERATION
(`examples/dspark_labels.rs`, "DENSE Qwen3 (LLaMA-family)") is arch-bound, and
qwen35 already captures the hidden states it needs via `hidden_rb`. But the
throughput case does not support doing it on this model.

## Phase 3 — the verify GEMV is nearly done; the tail is 16.65%

Post-fix trace, 64 tokens:

```
gemv_oq_compact_multicol_w8   57.86%   the verify — 87-90% of ceiling in isolation
gemv_oq_compact_multicol_w23  11.36%   23-token seed prefill, amortizes
gated_delta_net_f32            7.65%
__amd_rocclr_copyBuffer        4.02%   18226 calls, sub-us each = per-call overhead
gemm_dflash_oq4_plain_...w8    3.93%   drafter
oq_compact_x8_transpose        3.50%   ~280/cycle, one per projection
fused_rmsnorm_mq_rotate_awq    1.48%
```

The dominant kernel is the irreducible weight sweep and it is already at 87-90%
of the 233 GB/s ceiling (`bench_oq_compact_multicol`), so there is little left
there. The decode-time tail is **16.65%**; halving it is worth ~8%.

Ranked by size, with what is known:
1. `gated_delta_net_f32` 7.65% — core DeltaNet recurrence, a kernel project.
2. `copyBuffer` 4.02% — 1657 copies/cycle. The GDN tape writes are part of this
   and they BOUGHT +28-34%, so they are paid for; the rest is unattributed and
   worth tracing to a call site before optimizing.
3. `oq_compact_x8_transpose` 3.50% — feeds the compact GEMV. If multicol could
   read the untransposed layout this disappears.

NOT worth doing, checked: gating the tape's `x_in_bufs` writes. They are
genuinely diagnostic-only (every reader is a compare/repair/log behind
`dflash_serial_tape_rollback_replay_from_env`, and runs show
`replay_serial_tape=0`), but it is 48 of ~1657 copies/cycle, ~0.12% of GPU time.

## Also measured: verify-graph capture works with a bf16 embedding, and is worth ~0

`verify_graph_ok` admitted only HFQ4G256|Q8_0, but every arm of the batched embed
dispatch reads token ids from the same `pbs.tokens` device buffer — the list was
incomplete, not principled. `HIPFIRE_VERIFY_GRAPH_WIDE_EMBD=1` captures and
replays cleanly (`direct=0 replay=13`) with byte-identical output, and changes
throughput by nothing (57.72 -> 57.36). Graph capture removes CPU launch
overhead; this verify is GPU-bandwidth-bound. Left opt-in.

## Open, genuinely

- `embedding_lookup_bf16l3`: in-kernel decode exists for GEMM (`bf16l3_gemm_to_f32`,
  `bf16l3_wmma_coop`) which is how the lm_head stays packed, but the GATHER
  variant was never written — that is why qwen2's comment says the gather cannot
  read the packed form. Worth ~850 MB more (1.50x on the 2542 MB table) and a
  minor speedup. Needs a HIP kernel + parity test; `bf16_lut3::decode_block`
  is the host-side model to mirror.
- The Phase 3 tail above.


## Phase 3, worked: the tail is genuinely tail

Chased the three tail items to call sites. None is a bug; all are intrinsic.

**`oq_compact_x8_transpose` 3.50% (3072 calls) — NOT redundant.** It sits in the
iu4x2 GEMM path behind a reuse check ("skip when the hoisted quantize already
built XT"). Instrumented that miss: **it never fires**. The real callers are
`quant.rs:2283/2322`, inside the hoisted quantize itself — so the transpose IS
the hoist, done once per activation, and reuse works. 3072 calls is ~279/cycle,
one per projection-group activation, which is the floor. Removing it means
changing what layout the compact GEMV reads, i.e. a kernel redesign.

**`__amd_rocclr_copyBuffer` 4.02% (18226 calls) — per-call overhead, spread
thin.** Attributed by size (dim 5120, v_dim 6144, ffn 17408, n_v_heads 48, B=8):

```
557056 B = 8x17408x4   1584 calls  27.1%   FFN-sized
196608 B = 8x6144x4    3663 calls  22.1%   v_dim (attn_out / qkv tape)
163840 B = 8x5120x4    3993 calls  20.1%   hidden (x_in tape)
  1536 B = 8x48x4      2112 calls   0.1%   alpha+beta tape (exactly 48x11x2)
```

Total d2d traffic is ~3.0 GiB against the verify's 14.4 GiB/cycle — about 2% of
bytes. So it is ~2.7us of launch overhead per call, not bandwidth, and no single
site dominates. Merging the adjacent alpha/beta pair would remove 1056 of 18226
calls (~0.23% of GPU time). Not worth it individually; a batched-copy API would
be the real fix.

**`gated_delta_net_f32` 7.65%** is the core DeltaNet recurrence — the largest
single tail item and a kernel project, not a dispatch fix.

Conclusion: after the verify GEMV (57.86%, already 87-90% of ceiling), there is
no quick win left in this trace. The remaining levers are a DeltaNet kernel and
a batched-copy API.

## CORRECTION: the bf16 embedding fix was a PREFILL win too

Reported earlier as "tok/s flat". That was true of DECODE (an embed lookup reads
one row per token) and wrong overall — batched PREFILL gathers the whole table,
so halving it from F32 to bf16 nearly doubled prefill. Measured across the mix:

```
                  decode           prefill        VRAM
numbers  tau 5.733  57.87    106.69 tok/s   19784 MB
hard     tau 3.571  35.26    105.50         19786
MIT      tau 2.000  24.03    105.23         19784
```

prefill 63.4 -> ~106 tok/s (+65%), consistent on all three prompts. Prefill
should be in the baseline table above; it was not tracked and the win was nearly
missed.


## Phase 2b, MEASURED: a Markov head's tau ceiling is 2.077 — below what it must beat

Earlier I argued against building this from the drafter's 5.10% GPU share. That
was a prediction ("tau would fall toward 2-3"), so it has now been measured
instead. Harness: `scripts/markov_head_tau_ceiling.py`.

Corpus: 1280 tokens the TARGET itself generated across five prompts (that is
what a drafter has to predict). Method deliberately OPTIMISTIC — the n-gram
table is fit on the SAME stream it is scored on. That is train-on-test and
invalid for estimating accuracy; it is used only as a CEILING. Every eval
context is therefore in the table, so a miss is never sparsity — it is the
genuine limit that one context recurs with different successors and argmax can
only serve the majority.

```
order-1: tau=2.077  <- the Markov head's shape (bigram)
order-2: tau=4.206
order-3: tau=5.277
DFlash2, measured on these same prompts: 5.733 / 3.571 / 2.000  (mix mean ~3.77)
```

The DSpark head is `markov_w1`/`markov_w2` `[vocab, rank]` — a rank-r
factorization of the BIGRAM matrix, i.e. order-1. Its ceiling is 2.077, and a
low-rank approximation can only be WORSE than the full table measured here. The
order-2/3 rows are inflated by memorization (order-3 over 1280 tokens is close to
a lookup table of the eval set) and are not this head.

So the trade is: at best +5.10% from free drafting, against tau ~3.77 -> <=2.077,
about -45%. Net strongly negative, now on evidence rather than argument.

CAVEAT, stated because it cuts the other way: 1280 greedy-decoded tokens is a
small and repetitive corpus, which FLATTERS the n-gram (repetition is exactly
what it exploits). A larger or more diverse corpus would lower these numbers,
not raise them.

What would change the answer: a Markov head is not a drafter here, but it could
be a cheap FIRST-STAGE that a real drafter falls back on, or a component of a
full DSpark drafter (main projection + markov + confidence) rather than a
replacement for DFlash2. Neither is what the plan asked for, and neither is
worth building until the tail work above lands.


## Phase 2b COMPLETE: head trained, wired, and measured — it loses 2.6-2.9x

The plan said train + wire a Markov head. Done, end to end, rather than argued
about. Three artifacts:

* `scripts/train_markov_head.py` — fits a rank-r factorization of the bigram
  matrix (DSpark's `markov_w1`/`markov_w2` `[vocab, rank]` shape) and writes an
  MKV1 file. Only observed tokens get a row, which keeps the argmax O(n_obs*r);
  without that a single drafted token is 248320*rank host multiply-adds and the
  "free drafting" premise dies before it is measured.
* `dflash_spec_demo --markov-head <file.mkv>` — loads it and drafts the SPINE
  from it, filling the same slot PLD uses, so the DFlash model draft is skipped
  entirely. That is exactly the shape the plan wanted priced.
* `scripts/markov_head_tau_ceiling.py` — the ceiling harness.

Trained on 2560 tokens the target itself generated (10 prompts). rank=32,
n_obs=798. Its argmax agrees with the FULL bigram table only **45.2%** of the
time — a low-rank approximation is strictly worse than the table it approximates,
which is what the ceiling argument said and this quantifies.

Measured, 96 tokens, same binary, only the drafter differing:

```
                tau      decode tok/s
hd  DFlash2    3.571     35.42
hd  Markov     0.397     13.67      -61%
fr  DFlash2    5.733     57.82
fr  Markov     1.159     20.23      -65%
```

Output stays coherent under the Markov head — the target verifies every token,
so this costs speed, never correctness.

Measured tau is BELOW the 2.028 full-bigram ceiling, as it must be: the rank-32
approximation loses 55% of the table's argmaxes, and the spine chains 7
predictions so per-token error compounds.

This closes the question the plan opened. The arithmetic was: at best +5.10% from
free drafting (the drafter's whole GPU share) against a tau collapse. Measured:
-61 to -65%. The head is real, trained, wired, opt-in behind `--markov-head`, and
should not be turned on for this model.

Where it could still earn its place: as a cheap FIRST STAGE behind a real
drafter, or as one component of a full DSpark drafter (main projection + markov +
confidence) — not as a replacement for DFlash2.


# PREFILL — two crashes and a 20% overlay pass

Prefill was never exercised past ~500 tokens this session, because the bench
harness aborts above that. Fixed, then measured.

## Bug 1+2: prompt length is never checked against `--ctx` (harness only)

`--ctx` defaults to **512**, and both the draft's `target_hidden`
(`[ctx_capacity x num_extract x hidden]`) and the KV cache (`ctx_capacity +
block + 16`) are sized from it. Nothing validated the prompt against either, so
past 512 it failed two different ways:

```
  69 tokens   ok, 164 tok/s
 609 tokens   thread 'main' panicked: assertion failed: dst_offset + size <= dst.size
              (hip-bridge ffi.rs:1027, via scatter_hidden_block_to_interleaved)
2408 tokens   Memory access fault by GPU node-1 ... kernel: kv_cache_write_q8_0_batched
              Reason: Page not present or supervisor privilege
```

The second is the serious shape: an UNCHECKED OUT-OF-BOUNDS GPU WRITE, not a
caught assert.

**The daemon is NOT affected** — it guards this properly at
`generate.rs:223 / 679 / 698` and returns "request exceeds loaded KV budget".
This was `dflash_spec_demo` only. But it silently capped every prefill
measurement in this document at <=512 tokens.

Fixed with an explicit guard naming the offending number and the flag to raise.

## Prefill scaling, now measurable

```
  69 tokens   160.38 tok/s
 609 tokens   303.54 tok/s
2408 tokens   274.50 tok/s   (--ctx 4096)
```

Roughly 32x off the bandwidth bound: a batched prefill reads the 14.4 GiB of
weights ONCE for the whole prompt, which at 233 GB/s is 62 ms, i.e. ~9800 tok/s
at n=609. So prefill is nowhere near memory-bound and the ceiling is elsewhere.

## Where prefill time goes (2408 tokens, --max 1)

```
gemm_oq_compact_iu4x2_w64      4960 calls  46.77%
attention_q8_0_kv_batched       320 calls  20.51%   O(n^2), inherent
oq_compact_overlay_correct_tr  4960 calls  19.85%   <-- the anomaly
gated_delta_net_f32             480 calls   5.03%
quantize_act_oq8               2560 calls   1.56%
oq_compact_x8_transpose        2560 calls   1.44%
```

**The sparse overlay is a separate 19.85% pass in prefill** — 42% of the GEMM's
own cost, once per GEMM (identical 4960 call counts), for a correction that
touches 3 entries per block. In DECODE the same overlay is folded inline and
branchless into `gemv_oq_compact_multicol` and costs ~3%.

The mechanism is activation traffic, not weight traffic. Per (row, group,
overlay entry) the kernel gathers a B-wide contiguous slice of the TRANSPOSED
activation:

```c
const uint32_t w4 = *(const uint32_t*)(XT + (long long)(kbase + idx) * B + b0);
```

At M=17408, n_groups=20, n_ov=3, B=256 that is ~267 MB of activation reads per
projection against the GEMM's 44.6 MB of weights — the activation is re-read
three times per group. The side plane it also reads is only ~2.8 MB, so this is
not the weights.

The fix is to fold the overlay into `gemm_oq_compact_iu4x2_w64`, which already
has the activation tile staged — exactly what the decode multicol kernel does.
That is a kernel project and is the single largest prefill lever: ~20% of
prefill, and it would delete the `oq_compact_x8_transpose` pass (1.44%) that
exists only to feed it.


# CORRECTION: every number in this document was Q8 KV, not asym3

`dflash_spec_demo` never read `HIPFIRE_KV_MODE`. It hard-defaulted to
`kv_mode_str = "q8"` and honoured only `--kv-mode`, so the
`HIPFIRE_KV_MODE=asym3` in the baseline recipe above was a NO-OP for this
binary — every run in this document used **Q8** KV and printed `kv_mode: Q8`
where nobody was looking.

It produced no error because there was nothing to error on: the demo validates
`--kv-mode` strictly (unknown value exits 1) but had no notion the env var
existed. Every other qwen35 example honours it — `infer_qwen35`,
`bench_qwen35_speed`, `probe_argmax_agreement`, `speed_bench`. An env var that
one binary honours and its sibling silently ignores is worse than one nothing
reads, because it buys false confidence.

Fixed: precedence is now `--kv-mode` > `HIPFIRE_KV_MODE` > `q8`, and the line
reads `kv_mode: Q8 (from default)` / `(from HIPFIRE_KV_MODE)` so the source is
never in doubt.

**The accident was in our favour, and the baseline should stay Q8.** Measured
now that the flag works:

```
                        tau     decode tok/s
numbers   Q8           5.733       57.86
numbers   Asym3        5.733       56.91     -1.6%
hard      Q8           3.571       35.21
hard      Asym3        2.840       29.25     -17%
```

Asym3 costs 20% of tau on the hard prompt — coarser KV degrades the hidden
states the drafter reads, so acceptance falls. Committed text is identical
either way (the target verifies every token); only speed moves.

So the reported 57.9 / 35.2 are not inflated by the mix-up — they are the BETTER
configuration. But the recipe at the top of this document was wrong, and after
this fix it would have produced asym3 and an apparent failure to reproduce. It
has been corrected.


## SUPERSEDES the Q8 correction above: KVarN is the default, Q8/asym are DEPRECATED

Q8 and the asym*/fwht* tiers are deprecated; new KV work goes to KVarN, which is
meant to be the default path. Two things were wrong here, not one:

1. `dflash_spec_demo` ignored `HIPFIRE_KV_MODE` and hard-defaulted to `q8`.
2. Its `--kv-mode` match did not list `kvarn` AT ALL — only q8/asym4/asym3/asym2/
   fwht4/fwht3/fwht2 — so the harness could not select the intended default path
   even deliberately, despite `KvMode::Kvarn` existing and
   `speculative.rs:881` constructing it via `KvCache::new_gpu_kvarn_filtered`.

So every measurement in this document ran on a DEPRECATED KV tier, chosen by a
default nobody stated, with no way to opt into the supported one.

Fixed: `kvarn` is accepted and is now the default; precedence is `--kv-mode` >
`HIPFIRE_KV_MODE` > `kvarn`; the source is printed (`kv_mode: Kvarn (from
default)`); and selecting any deprecated tier prints a WARNING naming it.

Re-baselined on KVarN — it costs nothing:

```
                     tau    prefill tok/s   decode tok/s
numbers 1..30       5.733      109.64          57.53
merge two lists     3.571      105.76          35.32
MIT header          2.000      107.03          24.14
```

Identical to the Q8 numbers within noise (57.68 / 35.21 / 24.03), and tau is
unchanged, so nothing in this document's conclusions moves. Asym3 remains the
one that actually hurt (tau 3.571 -> 2.840 on the hard prompt).

NOT warned about, deliberately: the daemon warns that KVarN without
`HIPFIRE_KVARN_BATCHED_PREFILL=1` drops decode to plain AR. That is a
serving-path condition and is FALSE here — measured, this harness reaches tau
5.733 with the flag unset (57.39) and 57.54 with it set, so the drafter engages
either way. A warning that is false where it prints is worse than none.


## Prefill fidelity, re-measured: batched is 2.3x worse than per-token (not 57x)

With KVarN now the default it was worth re-checking the recorded claim that
"batched prefill + KVarN is 57x less faithful than per-token". It does not hold.
`compare_prefill_hidden_paths --n 48` on this model:

```
batched vs per-token:   worst |rel| 1.58e-2, first diverging layer 0
against the fp32-KV reference (lower is more faithful):
  batched    2.766e-2
  per-token  1.203e-2        -> 2.3x, not 57x
```

And it is **not KV-tier-specific**: `kvarn` and `q8` return byte-identical
numbers. Both are 8-bit KV, and with only 16 FA layers over 48 tokens the tier
barely moves the residual stream — the divergence is dominated by the
BATCHED-vs-PER-TOKEN difference itself, which is the real defect and applies to
every dtype. (`asym3` is unsupported by that tool and panics cleanly, which is
the correct behaviour and worth contrasting with the demo's silent q8 default.)

This is the mechanism behind the ~±22% acceptance swing already documented for
`HIPFIRE_COMPACT_BATCHED_CAPTURE`: the drafter is sensitive to which capture it
receives.

No fallback bugs remain in prefill: a 2408-token run under
`HIPFIRE_KERNEL_TRACE=1` takes the batched path (`verdict=true ... n=2408`) and
fires only the decode-side verify-graph site.


## KVarN bit widths: 4 is available and IS the default; 2 was a GPU memory fault

`KvMode::Kvarn` reads `KvCache::kvarn_bits_from_env()` — `HIPFIRE_KVARN_BITS` in
{2,4,8}, **defaulting to 4**, invalid values warned and coerced. So the KVarN
runs in this document are KVarN-**4**.

Note `compare_prefill_hidden_paths` hardcodes `new_gpu_kvarn(..., 8)`, so the
fidelity tool tests KVarN-8 while serving defaults to 4 — which is why its kvarn
and q8 rows came out byte-identical (both 8-bit).

At short context the width is immaterial — tau 3.571 and ~35.4 tok/s for 2/4/8
and deprecated q8 alike, VRAM within 10 MB, because `--ctx 512` makes the KV
cache tiny.

### The bug: KVarN-2 faulted on any real prompt

```
kvarn2   69 tokens  ok
kvarn2  609 tokens  Memory access fault ... kernel: attention_flash_kvarn_tile_batched
kvarn2 2408 tokens  Memory access fault (same kernel)
kvarn4/8 at 2408    fine
```

Root cause in `attention_flash_kvarn_tile_batched.hip`:

```c
const int TPW = 32 / bits;                    // 16 at bits=2
const int nt  = min(TPW, tile_len - t_base);  // can reach 16
float part[8];                                // <-- only 8
for (int tt = 0; tt < nt; tt++) part[tt] += ...;   // writes part[8..15]
```

A stack overflow of 8 floats on every 2-bit tile. The old comment
("TPW <= 8 for bits >= 4") stated the precondition correctly, but nothing
enforced it while the kernel signature accepts bits in {2,4,8}. 69 tokens
happened to fit; 609 did not.

Fixed by sizing `part[16]` for the worst case. After:

```
kvarn2   609 / 2408   rc=0   13.44 / 6.40 tok/s
kvarn4   609 / 2408   rc=0   13.25 / 5.82   (unchanged)
kvarn8   609 / 2408   rc=0   13.20 / 5.80   (unchanged)
```

KVarN-2 is now slightly the fastest of the three, as less KV bandwidth predicts.
Short-prompt baseline unmoved: 57.49 / 35.04 against 57.53 / 35.32.

Worth doing next: point `compare_prefill_hidden_paths` at
`kvarn_bits_from_env()` instead of a hardcoded 8, so the fidelity tool measures
the tier that actually ships.


## compare_prefill_hidden_paths was measuring nothing about KV — fixed

Two defects, and the second invalidated every KV comparison in this document.

1. It hardcoded `new_gpu_kvarn(.., 8)`, so it tested KVarN-**8** while serving
   defaults to **4**. Now reads `KvCache::kvarn_bits_from_env()` and PRINTS the
   width (`kv=kvarn bits=4`).
2. Its default `--n` was **48**, below the prefill chunk size
   (`PREFILL_MAX_BATCH = 256`). With one chunk, attention reads the in-flight
   f32 K/V and NEVER reads the quantised cache — so every tier returned
   bit-identical numbers (2.766e-2 / 1.203e-2 for kvarn2, kvarn8 AND q8 alike).
   That is what "the KV tier does not matter" earlier in this document actually
   meant: the tool was not testing the KV tier at all.

Default is now 512, and an `--n <= 256` warns that KV will not be exercised.

### The real numbers (n=512, so later chunks read the quantised cache)

```
                  worst |rel|   vs fp32-KV: batched   per-token
kvarn bits=2       9.83e-2         2.842e-1           2.591e-1
kvarn bits=4       3.93e-2         1.317e-1           9.622e-2   <- SHIPPING DEFAULT
kvarn bits=8       3.56e-2         3.443e-2           1.203e-2
q8 (deprecated)    5.09e-2         6.392e-2           1.240e-2
```

**The default KVarN-4 is ~8x less faithful than KVarN-8, and ~8x worse than the
deprecated q8 it supersedes.** KVarN-8 and q8 agree closely (1.203e-2 vs
1.240e-2), which is what two 8-bit schemes should do and is a good check that the
tool is now measuring the right thing.

This matches the recorded finding in memory `project_light_qat_recovery` —
"KVarN-4 loss NON-recoverable => deploy KVarN-8 not KV4" — so the shipping
default contradicts the project's own conclusion.

NOT changed here, because it is a product decision: at the lengths benchmarked
in this document KVarN-4 buys nothing measurable (decode 35.40 vs 35.40 vs 35.34
for 4/8/2; VRAM within 10 MB) while costing 8x fidelity. Its case is KV MEMORY at
long context, which is not what these prompts test. Whoever owns that tradeoff
should decide whether `HIPFIRE_KVARN_BITS` should default to 8.

Also visible: batched prefill is worse than per-token at EVERY tier (2.842e-1 vs
2.591e-1, 1.317e-1 vs 9.622e-2, 3.443e-2 vs 1.203e-2), so the batched-vs-per-token
defect noted earlier is real and independent of the KV tier — but it is far
smaller than the tier choice itself at 4 bits.


# KVarN does NOT faithfully reproduce the paper — the Hadamard rotation is missing

Paper: `/srv/hipfire/references/Quant/2606.03458v1-KVarN/` (2606.03458v1), with a
vLLM reference implementation under `git/`.

The method, stated four times in the paper and confirmed in the reference code, is
**"a Hadamard rotation followed by a dual-scaling variance normalization across
both axes of the K and V matrices"**, calibration-free, headline result at
**2-bit**. Its thesis is that decode-time error accumulation is driven by
*incorrect token scales*, which the rotation + dual scaling fix.

| paper component | hipfire KVarN | |
|---|---|---|
| Hadamard rotation (decoherence) | **ABSENT** | zero mentions across all 5 kvarn kernels |
| Sinkhorn dual-scaling, both axes | present | `KVARN_SINKHORN_ITERS 10`, per-row + per-col |
| applied to K **and** V | **K only** | "V stays Q8_0 (reuses the asym4 V layout)" |

The quantiser itself is faithful. `kvarn_mla_tilepack.py` dequants as
`(q*scale_abs + zp_abs) * s_row`; hipfire's record is
`(q*scale_abs[r] + zp_abs[r]) * s_col[c]` — the same form, same per-row/per-col
scales, same affine codes. What differs is the FRAME: the reference packs values
that are ALREADY Hadamard-rotated (`qH = q @ H`, keys stored rotated,
`acc_rot @ H.t()` to un-rotate V's contribution), while hipfire quantises raw.

That is the paper's actual contribution, and it is exactly what 2-bit needs:
rotation spreads outliers so a 2-bit grid can represent them. Our 2-bit being
catastrophic (KLD 9.6e-3 against fp32 KV, ~100x the 4-bit number) is the expected
symptom of quantising raw at 2 bits.

**The irony:** the DEPRECATED `Fwht4` tier DOES carry the Hadamard rotation
("signed-FWHT-rotated 4-bit K ... matches MQ4's weight-quant trick"), and
`attention_flash_asym4_tile.hip` already implements the required
rotate-Q-at-attention pattern ("Load + rotate Q for each half" ... "Q·K dot
products (K in rotated 4-bit space)"). So the machinery exists in-repo; the tier
named after the paper is the one not using it.

## What a faithful fix requires

Because H is orthogonal, `q·k = (qH)·(kH)`, so scores are preserved and K needs
no inverse:

1. Rotate K into the Hadamard frame at the KVarN write path, BEFORE the Sinkhorn
   normalise + affine quantise.
2. Rotate Q identically on entry to `attention_flash_kvarn_tile_batched.hip`,
   mirroring `attention_flash_asym4_tile.hip`. The f32 window must be rotated
   too, since attention reads it for the trailing partial tile.
3. For full fidelity, extend to V: store rotated V and un-rotate the attention
   output (`acc_rot @ H.t()`). hipfire's V is Q8 today, so this is a second,
   separate deviation.

This changes the DEFAULT KV path for all serving, so it wants the KLD and
coherence batteries behind it, not a spot check.

## Note on measuring before fixing

KLD numbers taken against the current implementation characterise
hipfire's-KVarN-minus-the-rotation, not KVarN. For the record, measured against
an fp32-KV reference at one position (prefill 512): kvarn8 4.5e-5, kvarn4 1.0e-4,
kvarn2 9.6e-3, q8 4.7e-5, top-1 unchanged for all. Those say the 4-bit tier is
cheap in output terms TODAY, but they cannot say anything about KVarN as
published until the rotation is in.

`dump_logits_qwen35` could not measure this at all before — it handled only
q8/asym{4,3,2} and panicked otherwise, so neither the shipping tier (kvarn) nor
the only fixed point to compare against (fp32) were reachable. Both added here.


## STOP: do not port the Hadamard rotation on current evidence

Measured properly, the rotation makes single-shot reconstruction WORSE, and the
earlier "1.17-1.41x gain" numbers were noise I should not have reported.

200 random [128 x 32] tiles with ~1/8 outlier channels, gain = plain/rotated:

```
 bits      mean       std       min       max     n>1
    2    0.7721    0.0126    0.7385    0.8084    0/200
    4    0.7755    0.0122    0.7486    0.8186    0/200
    8    0.7776    0.0130    0.7482    0.8236    0/200
```

Rotated error is ~1.3x plain, 0 wins out of 200 at every width, tight variance.

The mechanism is clear once stated: **hipfire's Sinkhorn already normalises
per-row = per-channel.** The Hadamard rotation deliberately DESTROYS per-channel
structure — that is its purpose, spreading outlier energy across all channels —
which is exactly the structure the per-row scales exploit. Applied together they
are antagonistic on this data.

### Why this does NOT refute the paper

The paper's claim is about ERROR ACCUMULATION over autoregressive decoding
("error accumulation" appears 21 times), and it says explicitly that prior work
is "evaluated under prefill-like settings and errors behave differently under
autoregressive decoding". A single-shot tile reconstruction is a prefill-like
measurement — the exact regime it says the benefit does not appear in. My test
cannot see the claimed advantage by construction, while it can and does see a
cost.

### What that means for us

The earlier framing in this document — "the Hadamard rotation is missing, that is
the fidelity gap, port it" — was too confident. It is a fidelity DIFFERENCE whose
value cannot be judged by anything measured here. Porting it into the default KV
path on the strength of "the paper says so", against local evidence that it costs
~1.3x reconstruction error, would be unjustified.

To decide it properly one needs the paper's own instrument: a decode-accumulation
proxy (it calls this "pseudo-decode"), measuring drift over many autoregressive
steps rather than one tile. That is the prerequisite for any port, and it does not
exist here.

Kept: `hadamard_channels` / `hadamard_rows` / `quantize_tile_rotated` in the
oracle, with tests. They are correct, they cost nothing when unused, and they are
what a decode-accumulation study would need. Nothing is wired into serving.


# Pseudo-decode proxy: built, and the rotation accumulates WORSE

`crates/hipfire-kvquant/examples/kvarn_pseudo_decode.rs` implements the setting
the paper defines (Fig. "pseudo-decode"):

> "We split the sequence into blocks of size b. After every block, the freshly
> produced K, V are quantized before being written back to the KV-cache.
> Subsequent blocks access a quantized cache, so quantization error accumulates
> over time."

The feedback is the point, and it is what the earlier single-shot test could not
see: the quantised run's FUTURE keys are generated from its OWN dequantised
history, so error compounds. Both runs share one driving process; only the cache
differs.

Two frame bugs had to be fixed before the numbers meant anything, and both
pinned drift at exactly 1.0 (a collapse, not a measurement):
  * a rotated cache must be queried with a ROTATED query — H orthonormal gives
    `q.k == (qH).(kH)`, which is the entire reason the rotation is free at
    attention time. Querying it raw produces meaningless scores.
  * only the CACHE lives in the rotated frame. A real implementation un-rotates
    the attention output (`acc_rot @ H.t()`) before the network sees it, so the
    model STATE stays in the natural basis. Feeding a rotated state forward makes
    the two runs different processes rather than one process plus quant error.

## Result (128 channels, b=32, 24 blocks, ~1/8 outlier channels)

Relative drift of the attention readout vs the unquantised run:

```
2-bit    step 1:  plain 0.0696   rotated 0.0729   1.05x
         step 24: plain 0.5556   rotated 1.2901   2.32x
4-bit    step 1:  plain 0.0179   rotated 0.0141   0.79x   <- rotation better
         step 24: plain 0.1491   rotated 0.4770   3.20x
```

The rotation starts neutral-to-better at step 1 and accumulates WORSE, with the
gap widening as steps go on — the opposite of the paper's claim.

The paper's own decomposition explains the shape without rescuing the outcome:

```
E_M/E_T (magnitude share of total error)
  2-bit   plain 0.239   rotated 0.017
  4-bit   plain 0.029   rotated 0.000
```

The rotation DOES do what the paper says — it nearly eliminates magnitude error,
which the paper identifies as the dominant outlier failure. But in this proxy the
residual DIRECTIONAL error accumulates worse than the magnitude error it removed.

A coherent reading: hipfire's Sinkhorn already normalises per-row = per-channel,
so magnitude error is ALREADY small on the plain path (E_M/E_T 0.029 at 4-bit).
The rotation's magnitude fix is therefore largely redundant here, while its
directional cost is not. The paper pairs rotation WITH dual-scaling and reports
SOTA, so either real K statistics differ from this synthetic process, or the
benefit lives somewhere this proxy does not reach.

## Honest limits

The driving process (`make_block`) is synthetic — my construction, not a model
forward. Accumulation dynamics depend on it, so this bounds what can be claimed:
it is evidence that the rotation is not obviously beneficial for OUR quantiser,
NOT a refutation of the paper. The decisive version replaces `make_block` with a
real qwen35 forward and measures drift in the actual residual stream.

## Where that leaves the port

Both instruments now agree the rotation costs us: single-shot reconstruction
~1.3x worse (0/200 tiles), accumulated drift 2.3-3.2x worse over 24 blocks.
Porting it into the default KV path is not justified on this evidence. The oracle
functions stay (correct, tested, free when unused) for whoever runs the
real-forward version.


# Real-forward decode accumulation: the error PLATEAUS, it does not accumulate

`crates/hipfire-runtime/examples/kvarn_decode_accumulation.rs` runs the
pseudo-decode experiment on an actual qwen35 forward: two identical decode runs
over one model differing ONLY in the KV tier, measuring logit KLD against an
fp32-KV reference as steps accumulate. Both runs are TEACHER FORCED on the same
token stream — left to sample their own they diverge in token space and the drift
stops being attributable to the KV tier.

Qwen3.8-27B, prompt 64 + 384 decode steps, KLD(fp32-KV ref || tier):

```
             steps 1-64     after the first block flush (step 128+)
bits=8        ~8e-6          ~5e-6        flat
bits=4        ~8e-6          ~6-9e-5      flat
bits=2        ~8e-6          ~2-9e-3      oscillating, no trend
```

Two things to read off this.

**The step-128 jump is the first flush, not accumulation.** KVarN keeps the
trailing GROUP=128 tokens in an f32 window; below that nothing is quantised at
all. Prompt 64 + 64 steps = 128 is exactly where the first block leaves the
window. Before it, all three bit widths are bit-identical.

**After the flush the error PLATEAUS.** The 2-bit series across the plateau is
2.4, 2.0, 2.0, 4.5, 3.6, 8.8, 2.9, 6.3, 3.7 (x1e-3) — high step-to-step variance,
no monotonic growth over 256 further steps. Same for 4-bit and 8-bit. There is no
compounding here to fix.

That CONTRADICTS the synthetic proxy, which showed drift growing to 0.55/1.29
relative. The synthetic driving process had far stronger state feedback than the
real model does, which is exactly the limitation flagged when it was committed —
and the reason the real-forward version was worth building.

**Why this model probably does not accumulate: it is a HYBRID.** 16 of 64 layers
are FullAttention and carry KV; the other 48 are LinearAttention (DeltaNet) with
no KV cache at all. Only a quarter of the depth is exposed to KV error, and
attention is a weighted average, which averages perturbations rather than
compounding them. The paper targets standard full-attention models where every
layer reads the quantised cache; accumulation may well be real there and simply
absent here.

## So: is there anything to do about the accumulating error?

On this model, no — because there is not any. Concretely:

* Accumulation-targeted mitigations have nothing to fix here. That covers the
  Hadamard rotation (measured 1.3x worse single-shot, 2.3-3.2x worse on the
  synthetic proxy), and also the usual suspects — error feedback across blocks,
  periodic f32 re-anchoring, attention-sink preservation. They all buy reduced
  COMPOUNDING, and compounding is not what is happening.
* The real cost is a ONE-TIME step at flush, sized by bit width: 8-bit 5e-6,
  4-bit 8e-5, 2-bit 3e-3. If KV error matters to you, the lever is bit width, and
  the shipping default (4) sits three orders of magnitude below 2-bit for KLD.
* The f32 window already does the one thing that clearly helps: it keeps the
  most recent GROUP tokens exact, so the freshest context — the part attention
  weights most heavily — never carries quantisation error at all.
* If accumulation ever does appear (a full-attention model, or much longer
  horizons than 384 steps), this harness is the instrument to catch it, and the
  GROUP guard in it stops the "everything is identical" failure mode.

## A recurring trap, now guarded in three places

Three separate diagnostics silently reported "the KV tier does not matter"
because none of them exercised the quantised path:

| tool | threshold | symptom |
|---|---|---|
| `compare_prefill_hidden_paths` | prefill chunk = 256 | every tier bit-identical at n=48 |
| same | hardcoded `bits=8` | kvarn read as q8 |
| `kvarn_decode_accumulation` | KVarN GROUP = 128 | every `--bits` identical at 96 positions |

All three now warn. The general shape is worth remembering: a KV diagnostic that
does not cross the quantiser's own block boundary is measuring f32.


# Does the KVarN f32 window need to be f32? No — but the win is small

## How much math actually touches the window

All of it, in `attention_flash_kvarn_tile_batched.hip`:

```c
partial += mq[i] * kt[d0 + i];      // Q.K, one MAC per element
partial = wave32_sum_dpp(partial);  // wave reduction
s = partial * scale_attn;           // one scale
```

That is the entire lifetime of a window value: read once per decode step, dotted
with Q, done. No in-place transform, no iterative refinement, nothing that
compounds precision error across steps. The accumulator is f32 regardless of how
the value is STORED. V is not in the window at all (it is Q8 separately), so this
is K-only.

The window is also read by the block flush (gather -> quantise), which turns
these very values into 4-bit KVarN codes.

## So the storage precision only has to beat 4 bits

Measured on K-like vectors with the outlier-channel structure KVarN targets
(20k vectors, head_dim 128, every 8th channel x9), relative error of Q.K:

```
    window storage      rel err of Q.K
    f32 (today)         0
    f16                 2.07e-4
    bf16                1.67e-3
    4-bit (post-flush)  1.88e-1     <- what these same values BECOME
```

f16 is ~900x more accurate than the quantisation these values receive 128 tokens
later; bf16 ~113x. Holding a 24-bit mantissa for data whose destiny is 4 bits is
not buying anything. **f16 over bf16** here: K is bounded and roughly normalised,
so f16's larger mantissa beats bf16's larger exponent range by ~8x.

## What it would actually save

window = GROUP(128) x kv_dim x 4 B per KV layer. On this model (16 of 64 layers
carry KV — the rest are LinearAttention with no cache):

```
    n_kv_heads=8  ->  512 KiB/layer  ->  8.0 MiB f32  ->  4.0 MiB f16
```

Footprint saving ~4 MiB against a ~20 GB resident model: 0.02%. The window is
re-read every decode step but only `tile_len` of it (0..128, ~64 average), so
per-step traffic is ~2-4 MiB — roughly 0.1-0.2% of the 233 GB/s budget at 57
tok/s. Halving it is not measurable.

## Recommendation: correct, safe, and not worth doing here

f32 is unnecessary and f16 is provably sufficient, but the payoff on THIS model
is ~4 MiB and ~0.1% bandwidth, against touching three places (allocation, the
window writer, and the attention kernel's window read) on the default KV path.
That trade is not worth it on the evidence.

It becomes worth revisiting if the window grows: a FULL-ATTENTION model where all
64 layers carry KV puts it at 32 MiB f32 and ~0.5-0.9% of per-step bandwidth, and
a larger GROUP scales it linearly. The measurement above is the justification
whenever someone wants to make that change — the precision question is settled,
only the size question is open.

## KVarN window precision — the knob is config, not env

`kv_window_precision` (`auto` | `f16` | `f32`, load-time, global/model/runtime)
is the user-facing control for the KVarN recent window's storage dtype. It is a
real config field in `hipfire-config`, not an env var: **the env system here is
for debugging.** `HIPFIRE_KVARN_WINDOW_F16` still exists and still outranks the
config, but only as a documented debug override for developing the remaining
consumers.

`auto` resolves to **f32 today** and says so loudly, naming what blocks it.
Precision is not what blocks it — f16 is ~900x tighter than the 4-bit these
values become on flush (Q·K rel err 2.07e-4 vs 1.88e-1). What blocks it is
consumer coverage, and both survivors are inside `Gpu::kvarn_attend`:

1. it stages K into the window with a raw dtod blit hardcoding 4 bytes/element,
   so flipping the dtype under it overruns the buffer (measured: `assert!` at
   hip-bridge `ffi.rs:1619`, first decode step);
2. `kvarn_gather_k_tiles` reads the window as f32 to build the quantiser's
   tiles — and this one corrupts KV records silently rather than crashing.

That second point is why an explicit `kv_window_precision=f16` is a **request,
not a force**: it falls through to the same honest probe and the same loud
fallback. Forcing is the debug override's job. `window_f16_precedence` is a pure
function with a unit test pinning exactly this.

Converted consumers derive the flag from `kvarn_window_is_f16()` or from the
tensor's own dtype, so a kernel can never disagree with the memory it reads —
that invariant is the whole reason the routed kernels take a `window_f16`
kernarg instead of assuming.

Regression after the change, warm, `--max 96`: **57.61 / 57.53 tok/s, τ=5.733**
on "numbers 1 through 30", unchanged from baseline.

⚠️ **Bench trap:** the FIRST run of a fresh binary reads 34.4 tok/s on that same
prompt — cold page cache, not a regression. It cost a round of chasing. Discard
run 1 of any newly built binary, or you will bisect a phantom.

## The headline number depended on knowing an env var

Measured properly (each knob isolated with `env -u`; the three were previously
exported together, which made every partial run look like "both on"):

| knob | tok/s alone | verdict |
|---|---|---|
| `HIPFIRE_OQ_COMPACT_MULTICOL_WIDE=1` | **55.53** | the entire 2.3x |
| `HIPFIRE_LMHEAD_TWOSTAGE=q2` | 24.27 | inert on this model |
| `HIPFIRE_GRAPH=1` | 57.07 vs 57.57 unset | no-op — AR-forward hipGraph is hard-disabled |
| all off | 24.31 | baseline |

Two of the three incantations in the bench recipe above do nothing here. The
whole gap between 24 and 56 tok/s was ONE opt-in env var that defaults off, read
by a raw `env::var` on every dispatch — in the crate whose `FeatureFlags` module
exists precisely to stop that ("read exactly once at `Gpu::init()`... instead of
hitting `std::env::var`'s global lock on every call").

It is now `oq_compact_multicol_wide` in the config file, verified end to end:
**55.98 tok/s from config alone with no env set**, and `HIPFIRE_...=0` still pins
it off at 24.36 as the debug override.

Default stays OFF pending cross-shape proof against the narrow kernel — it needs
`K % 1024 == 0` and falls back where that fails. On this model it is a 2.3x
default waiting to be flipped.

⚠️ **Measurement trap, second instance this session:** `export A=1 B=2` then
`B=2 cmd` does NOT unset A. Every "A off" row was really "both on". Use
`env -u A -u B` per run. The first instance was cold-cache; both produced
confident wrong numbers that survived until something looked impossible.

## Verify is nearly free — up to B=8, then it falls off a cliff

`bench_oq_compact_multicol`, gate/up [17408, 5120], weight bytes CONSTANT at
45.2 MiB across every B, so GB/s *is* the amortisation efficiency:

| B | LDS | VGPRs | occupancy (min of VGPR/LDS) | GB/s | % of 233 | GB/s per wave |
|---|---|---|---|---|---|---|
| 1 | — | 79 | 16 | 235 | 96.6% | — |
| **8** | 8 KB | 135 | **10** | **216** | **95.2%** | 21.6 |
| 12 | 12 KB | 163 | 9 | 184 | 79.2% | 20.4 |
| 16 | 16 KB | 191 | 8 | 148 | 63.9% | 18.5 |
| 17 | 17 KB | 148 | 6 | 124 | 51.1% | 20.6 |
| 24 | 24 KB | — | 4 | 91 | 38.4% | 22.7 |
| 32 | 32 KB | 223 | 4 | 64 | 27.5% | 16.0 |

At B=8 eight tokens cost 0.213 ms against one token's 0.210 ms — the weight
sweep is amortised essentially perfectly. **That is why B=8 is the sweet spot:
it is the largest B where occupancy still sits at 10.**

ROOT CAUSE: the kernel is LATENCY-bound, and throughput is ~20 GB/s per
wave/SIMD across B=8..24. Occupancy is capped by `facc[RW][BC]` registers for
B=9..16 and by `lds_x[4 * BC * 256]` above 16 — both linear in batch width.

The decisive evidence is B=17: it has FEWER VGPRs (148 vs 191) and higher
compiler-reported occupancy (9 vs 8) than B=16, yet is SLOWER (124 vs 148
GB/s). Only the LDS-limited occupancy (6 vs 8) explains that ordering, which is
why the compiler's VGPR-only "Occupancy" line must not be read as the answer.

Two hypotheses TESTED AND KILLED, so nobody re-runs them:

- **LDS read traffic.** The activation tile is re-read per column, so LDS
  traffic scales with BC while global traffic stays flat. Raising RW 3 -> 4 for
  B=9..16 (more row reuse per LDS read) changed nothing: B=12 0.258 vs 0.257 ms,
  B=16 0.321 vs 0.325 ms, three runs each.
- **The overlay's scattered per-(row,column) LDS gather.** `BENCH_NOUT=0`
  removes it entirely and the cliff is untouched: B=16 147.5 vs 147.7 GB/s. It
  costs ~4%, flat in B, which is just its extra bytes.

Also note the wide kernel is what makes verify viable at all: at B=8 it is
**3.5x** the narrow kernel (0.213 vs 0.748 ms). Narrow never amortises — 27% of
ceiling at B=8, falling to 16% at B=16.

## DDTree does not pay here, and a smaller budget does not rescue it

Matched-proposal comparison, 96 tokens:

| mode | proposals | tau | node acceptance | tok/s |
|---|---|---|---|---|
| linear DFlash2 | 7 | **5.733** | 81.9% | **57.55** |
| tree budget=6 | 6 | 5.125 | 85.4% | 51.87 |
| tree budget=8 | 8 | 5.667 | 70.8% | 47.43 |
| tree budget=12 | 12 | 5.786 | 48.2% | 43.16 |
| tree budget=16 | 16 | 5.857 | 36.6% | 33.19 |

The verify cliff is real but it is NOT the whole story: at equal proposal count
the tree has LOWER tau than the linear chain, and tau rises only 5.125 -> 5.857
for 2.7x the nodes. DFlash2's sequential chain already accepts 82%, so tree
diversity buys almost nothing on this drafter. Tree build itself is free
(0.01 ms); verify is 90% of the cycle.

Conclusion: the lever is NOT tree topology. It is either the drafter (raise tau
at B<=8, where verify is free) or the kernel's occupancy past B=8 (which would
let B=12 spend its tau=7.1 at B=8 prices).

## Prefill throughput: measured curve and a fitted cost model

Qwen3.8-27B oq4.25++ on gfx1151, `dflash_spec_demo --max 4`, 20 prompt lengths
from n=37 to n=4489. **Prefill is not a single number** — it ranges from 95 to
306 tok/s over that span, so quoting one figure without its n is meaningless.

    t(n) = 0.174 + 2.854e-3 n + 0.275e-6 n^2      seconds     (fit on n >= 250)

worst error 8.5%, mean 3.3%. Equivalently, per-token cost is

    2.85 ms + 0.275 us * n

which is the more useful form: a flat compute term plus an attention term
linear in context. Reading the coefficients:

- `0.174 s` fixed setup, which is why short prompts look slow (n=37 measures
  95 tok/s and is ~all overhead).
- `2.85 ms/token` -> a **350 tok/s ceiling** from the linear term alone.
- `0.275 us * n` attention. Negligible under n~1000; it **doubles** the
  per-token cost at n = 10379.

Measured, and predicted past the data:

| n | tok/s |
|---|---|
| 37 | 95 (measured, overhead-dominated) |
| 121 | 247 (measured) |
| 849 | 302 (measured) |
| 1269 | **306 (measured peak)** |
| 4489 | 242 (measured) |
| 8192 | 195 (predicted) |
| 16384 | 136 (predicted) |
| 32768 | 84 (predicted) |

⚠️ **A better-fitting model was WRONG.** `a*ceil(n/256) + b n + c n^2` scored
better (worst 10.0% / mean 2.8% vs 18.8% / 5.0%) and 256 is exactly
`PREFILL_MAX_BATCH`, so it looked physically motivated — cost per chunk. But
its distinctive prediction is a ~285 ms sawtooth at each chunk boundary, and
that does not exist: crossing n=256 measures 249 -> 0.79 s, 259 -> 0.88 s
(+0.09 s for 10 tokens, not +0.285 s), and crossing 512 is equally smooth. The
chunk term was fitting noise in the small-n scatter. Prefill cost is SMOOTH in
n. Fit quality alone did not distinguish the true model from the false one —
only testing the prediction did.

## Why DDTree fails: tau SATURATES with width, and climbs with depth

The verify cliff is real, but it is not the reason the tree loses. Measured on
the same prompt, 96 tokens:

| tree budget | proposals | tree tau | | linear B | proposals | linear tau | tok/s |
|---|---|---|---|---|---|---|---|
| 8  | 120 | 5.667 | | 8  | 105 | 5.733 | 57.76 |
| 16 | 224 | 5.857 | | 12 | 121 | 7.818 | 50.08 |
| 24 | 336 | 5.929 | | 16 | 150 | 8.500 | 40.22 |
| 32 | 448 | 5.929 | | 24 | 253 | 8.818 | 23.92 |
| 48 | 672 | **5.929** | | | | | |

**Tree tau saturates at 5.929.** Tripling the nodes from 16 to 48 — 224 to 672
proposals — buys +0.07 tau. Linear tau keeps climbing to 8.818.

The reason is structural, and it means no amount of cheap verify rescues the
tree: a tree built on a B=8 drafter explores ALTERNATIVES at 8 positions. Its
tau ceiling is the draft DEPTH, and it saturates well below even that. Width
cannot substitute for depth. Raising B raises depth directly, which is why
linear tau goes to 8.8 while the tree sits at 5.9.

## But the verify curve IS what caps spec decode — via linear B, not the tree

From B=8 to B=24 tau rises 54% (5.733 -> 8.818) while throughput falls 59%
(57.76 -> 23.92), because verify cost grows with proposals. The DDTree phase
timer gives `verify(N) ~ 35 + 8*N ms`: a ~35 ms fixed sweep plus **8 ms per
proposed token**. Plain decode is 68 ms/token, so verify is already 8.5x
cheaper per token — that is the spec-decode win — but it is NOT flat, and flat
is what the amortisation argument assumes.

Projected throughput if the marginal per-token verify cost were reduced
(draft ~1.6 ms/token + 35 ms + marginal*N):

| B | tau | now (8 ms) | 4 ms | 1 ms |
|---|---|---|---|---|
| 8 | 5.733 | 52 | 73 | 106 |
| 12 | 7.818 | 53 | 78 | **121** |
| 16 | 8.500 | 46 | 69 | 113 |
| 24 | 8.818 | 33 | 53 | 92 |

(model tracks the measured 57.8 / 40.2 / 23.9 within ~15%)

So the prize is ~2x — 57.8 -> ~110-120 tok/s — and it comes from letting B=12..16
run at their tau of 7.8-8.5. **The lever is the 8 ms/token marginal verify
cost, not tree topology and not the drafter.**

Note this refines the microbench section above: `gemv_oq_compact_multicol`
amortises 8 tokens for the price of 1, but WHOLE-MODEL verify does not inherit
that — it still costs 8 ms per extra token. The GEMV is not where the marginal
cost lives. Attention is inherently O(N) here, so that is the first suspect.

Context for the ceiling: prefill peaks at 306 tok/s = 14.9 TOPS effective over
24.35 B matmul params, which is 26.6% of the 56 TOPS int8 peak (3.8x headroom)
or 13.4% of the 110.9 TOPS iu4 ceiling.

## The drafter's width ladder stopped at 8 — 57.8 -> 63.9 tok/s

rocprofv3 on the demo (B=8 vs B=24, per cycle, kernel time in ms) finally named
where the marginal cost of raising B goes. **Attention is not it** — 1.14 ->
1.77 ms/cycle, ~1% of the cycle, essentially flat. My "attention is the first
suspect" guess was wrong.

| phase | B=8 | B=24 | delta |
|---|---|---|---|
| multicol GEMV (verify weights) | 81.96 | 149.67 | +67.7 |
| **draft GEMM** | **5.44** | **73.04** | **+67.6** |
| DeltaNet/GDN | 7.99 | 50.43 | +42.4 |
| other GEMM | 1.24 | 41.19 | +40.0 |
| attention | 1.14 | 1.77 | +0.6 |
| TOTAL GPU | 113.07 | 341.42 | +228.4 |

`gemm_dflash_oq4_plain_dp4a_staged_8w` went from 2 calls / 0.6 ms to **41 calls
/ 72.8 ms** — the call count, not just the work, so the draft path changed
SHAPE. Root cause in `dflash_plain_multicol_kernel`:

    if !(2..=8).contains(&batch) { return None; }   // 9+ -> per-column fallback

The drafter's multi-column ladder stopped at 8 while the TARGET's
`gemv_oq_compact_multicol` goes to w32. So every B>8 fell back to the
per-column kernel and paid one weight sweep per drafted position. Same defect
shape as the four dtype ladders in Phase 1 — a ladder that stops short while
the artifact silently takes a slow branch — except the axis is WIDTH.

The macro body is fully parametric in NB (`acc[NB]` in registers, no LDS), so
extending it is additive. Added sets 9..16 plus the dispatch arms.

Warm results, tau IDENTICAL at every B (behaviour unchanged, just faster):

| B | tau | before | after |
|---|---|---|---|
| 8 | 5.733 | 57.8 | 57.4 |
| 9 | 6.000 | — | 58.2 |
| **10** | **6.917** | — | **63.9** |
| 11 | 7.500 | — | 63.0 |
| 12 | 7.818 | 50.1 | 62.1 |
| 13 | 8.273 | — | 55.8 |
| 16 | 8.500 | 40.2 | 49.4 |

**New optimum B=10 at 63.9 tok/s, up from 57.8 at B=8 (+10.6%).** B=12 gains
+24% against its own baseline. Past ~12 the verify GEMV occupancy cliff
reasserts itself and throughput falls again, so this does not remove that
ceiling — it removes a second, independent one that was hiding underneath it.

⚠️ **Cold-cache trap, third instance this session.** Adding 24 kernel
instantiations makes the first run of each width pay JIT inside the measured
window: B=8 first read 8.33 tok/s and B=10 read 28.50, against warm 57.4 and
63.9. Always warm every width before believing a sweep.

## DeltaNet perf pass — where the time actually is

Real geometry from the artifact: `linear_num_value_heads=48`, `linear_key/value_head_dim=128`,
48 of 64 layers DeltaNet. Bench at that shape (`bench_gated_delta_net`, GDN_HEADS=48):

    t=1   23.6 us        t=10  51.7 us       => fixed 23.6 us + 3.12 us/token

**The fixed half is already optimal and cannot be tuned.** State is
48 x 128 x 128 x 4 = 3.15 MB, read AND written every call = 6.29 MB in 23.6 us
= **267 GB/s, at the DRAM ceiling** (measured achievable 250-256). The only way
down is fewer state BYTES, which is exactly what FP16 state does.

**The marginal half runs at ~10% of f32 FMA peak** and resisted every structural
lever tried:

| lever | result |
|---|---|
| `#pragma unroll` the row loop | null — compiler already unrolls a constant bound |
| ablate BOTH cross-lane reductions (wrong results, timing only) | only −4% — NOT shuffle-bound |
| hoist state slice LDS -> registers | **−3.7%, kept** (lands on the ablated floor) |
| TILE_ROWS 4 -> 8 -> 16 | null: 0.0517 / 0.0522 / 0.0518 ms, despite occupancy 16 -> 8 -> 4 |
| chunkwise-parallel (`HIPFIRE_GDN_CHUNK=1`) | null at B=10: 63.87 vs 63.75 tok/s |

TILE_ROWS being flat while occupancy drops 4x is the strongest single result
here: this kernel is **not occupancy-limited**, so the usual RDNA levers
(registers, LDS, waves) have nothing to give. That is why the LDS hoist bought
3.7% and nothing else bought anything.

The chunkwise kernel DOES run when enabled (`gdn_chunk_pairs` 336 calls/run
under HIPFIRE_KERNEL_TRACE, alongside 720 sequential calls for the single-token
steps) — it simply has nothing to amortize at 10 tokens. It is a long-prefill
optimization, and it additionally requires `StateQuant::FP32`, so it is
MUTUALLY EXCLUSIVE with the FP16 state that actually wins here.

### The lever that works is FP16 state

Measured end-to-end at B=10 on Qwen3.8-27B, tau identical (6.917) in all three:

| config | tok/s |
|---|---|
| fp32 sequential (DEFAULT) | 63.75 |
| fp32 chunkwise | 63.87 |
| **fp16 sequential** | **68.02** |

FP16 halves the DRAM-bound term that dominates the call, and halves the
spec-decode state snapshot alongside it. It is the only thing that moved the
number, it is opt-in, and the env doc claimed for months that it was already
the default. See the `HIPFIRE_DN_STATE_FP16` entry.

## ⚠️ RETRACTION: most of the DeltaNet ablation section above is INVALID

`kernels.rs` does `include_str!` on the .hip files, so **kernel source is
embedded in the Rust binary at build time**. Editing a .hip and deleting
`~/.hipfire/kernels/gfx1151/*` does NOT change what runs — the cache is
recompiled from the same stale embedded string. Every ablation run without a
`cargo build` in between measured the unmodified kernel.

Caught by an impossible result: with the ENTIRE token loop ablated, runtime
still scaled with n_tokens (0.0238 / 0.0516 / 0.0971 ms at t=1/10/24). A kernel
doing no per-token work cannot do that.

Corrected, with a rebuild between every edit:

| claim (as published above) | corrected |
|---|---|
| reductions cost only 4% — "NOT shuffle-bound" | **WRONG. They are 57% of the marginal**: ablating them takes t=24 from 0.0965 to 0.0533 ms (−45%) |
| LDS -> register hoist worth −3.7% | **null** (0.0523 vs 0.0524) |
| TILE_ROWS 4/8/16 null | untested; that sweep never changed the binary |
| output store / q-k loads null | untested |

Only the env-var-driven results (FP16 state, chunkwise) were valid, because
those needed no kernel edit.

Ablation with a correct build, 48 heads:

    baseline            t=1 0.0239  t=10 0.0518  t=24 0.0979
    no token loop       t=1 0.0180  t=10 0.0201  t=24 0.0212   <- flat, as it must be
    loop, no row work   t=1 0.0181  t=10 0.0193  t=24 0.0212
    no reductions       t=1   -     t=10 0.0368  t=24 0.0533
    no out dot          t=1   -     t=10 0.0516  t=24 0.0985   <- null

So: fixed ~18-21 us (state I/O, at the DRAM ceiling), and the per-token cost is
**dominated by the two 32-lane shuffle reductions**, 40+ shuffle steps per token
on the critical path.

## The fix works in isolation and LOSES end-to-end — deferred, not abandoned

Restructured the wave as 4 groups of 8 lanes: group `g` owns row `row_start+g`,
lane `l` owns columns [16l, 16l+16), so the 4 rows resolve CONCURRENTLY and each
reduction spans 8 lanes (`__shfl_down(v, o, 8)`) instead of 32. Per token:
2x3 shuffle steps instead of 4x(5+5). Same FMA work, 4x the ILP.
Kept at `docs/experiments/gated_delta_net_8lane_groups.hip.txt`.

    kernel   t=10 0.0513 -> 0.0377 (−27%)   t=24 0.0965 -> 0.0574 (−41%)
    VGPRs 46 -> 72, no spills, occupancy still 16
    parity vs f64 CPU reference: PASS (3.4e-7)

But `test_gated_delta_net_tree_f32` went **FAIL: f32 tree vs f32 linear 782/2560
byte-exact, max|diff| 2.794e-9**, and end-to-end **tau fell 6.917 -> 6.308 and
throughput 63.75 -> 59.73 tok/s**.

Changing the summation order changes the rounding, and the f32 GDN kernels are
required to agree BIT-FOR-BIT: `gated_delta_net_f32_tree.hip` says so in its own
header ("Lane mapping is `col = tid * 4` (contiguous), matching the FP32 LINEAR
kernel"). When they disagree, replay commits a state sequential decode never
reaches and acceptance collapses — the exact failure the f16 dither commit
(074e28503) fixed the tree kernel to avoid.

REVERTED. To land it, the identical layout has to go into every f32 GDN kernel
at once — `gated_delta_net_f32_tree.hip`,
`gated_delta_net_f32_routed_batch_seq.hip`, and the batch-seq variant — then
re-verify byte-exactness across all of them. Worth ~2-3% end-to-end (DeltaNet is
~7% of the cycle and this halves its marginal), which is why it is recorded
rather than dropped.

Note the FP16 state path is untouched by all of this and still wins outright at
67.5 tok/s, because it attacks the DRAM-bound fixed half instead.

## DeltaNet, corrected: where the per-token cost really is

With a rebuild between every edit (48 heads, ms at t=10 / t=24):

| ablation | t=10 | t=24 | reading |
|---|---|---|---|
| baseline | 0.0519 | 0.0970 | — |
| no token loop | 0.0201 | 0.0212 | fixed cost ~20 us, flat — state I/O at the DRAM ceiling |
| **no reductions** | **0.0368** | **0.0533** | **the dominant per-token term** |
| no output store | 0.0365 | 0.0584 | ⚠️ CONFOUNDED — see below |
| no q/k global loads | 0.0487 | 0.0943 | 3-6%, minor |
| no out dot | 0.0516 | 0.0985 | null |

⚠️ **"no output store" is not a store measurement.** Removing the store makes
`out_v` dead, so the compiler also eliminates the out-dot AND the second
reduction. That is why it reads 40% — it is measuring the reduction a second
time. Coalescing the store for real (below) is worth 2%, not 40%. An ablation
that deletes a *consumer* deletes everything upstream of it; only ablations that
keep the value live measure what they name.

So the per-token cost is dominated by the two 32-lane shuffle reductions, and
essentially nothing else is worth attacking.

### Landed: coalesced output store (byte-exact, ~2%)

The per-row store was a single-lane 4-byte scattered write, four per token per
block. Now the four contiguous rows are stashed and written together. Pure store
shaping — arithmetic and its ORDER untouched — so all four parity tests stay
green and tau is unchanged.

    kernel  t=10 0.0519 -> 0.0509   t=24 0.0970 -> 0.0949
    e2e     64.18 -> 64.3 tok/s at B=10, tau 6.917

### On "drop the whole kernel to f16"

Worth separating what is already f16 from what is not. `HIPFIRE_DN_STATE_FP16`
narrows the GLOBAL STATE only: `gated_delta_net_f16.hip` still stages an
**FP32 working copy in LDS** and does every operation — q/k/v, both dots, both
reductions, the update — in `float`. That is deliberate, and the dither exists
because f16 *storage alone* is already biased on a recurrent accumulator.

Narrowing the arithmetic would attack the FMA term, which the table above shows
is the MINORITY. The dominant term is the reductions, and those are 32-bit
shuffles whose count does not fall just because the values are narrower. Packing
two rows' partials into one 32-bit shuffle WOULD halve them — but that changes
the reduction's precision and order, which is the byte-exactness constraint
again.

Which is the real conclusion: **every remaining DeltaNet lever changes summation
order, and therefore has to be applied to the whole f32 GDN kernel family in one
change** (`gated_delta_net.hip`, `_f32_tree`, `_f32_routed_batch_seq`, the
batch-seq variant), with byte-exactness re-verified across all of them. The
8-lane-group experiment is worth −27%/−41% on the kernel and is kept in
`docs/experiments/`; it is blocked on exactly that, not on the idea.

## LANDED: the 8-lane rewrite, family-wide

All three f32 GDN kernels now carry the IDENTICAL step block —
`gated_delta_net.hip`, `gated_delta_net_f32_tree.hip`,
`gated_delta_net_f32_routed_batch_seq.hip`. (`gated_delta_net_f32_batch_seq` is
not a fourth kernel; it launches `gated_delta_net_f32` with a batch grid. The
f64acc oracle and the chunk kernel are compared on tolerance, not bytes, and
are deliberately excluded.)

The wave is 4 groups of 8 lanes: group `g` owns row `row_start+g`, lane `l`
owns columns [16l, 16l+16). The 4 rows resolve CONCURRENTLY and each reduction
spans 8 lanes, so a token costs 2x3 shuffle steps instead of 4x(5+5). The `r`
loop disappears entirely.

    kernel (48 heads)  t=10 0.0509 -> 0.0367 (−28%)   t=24 0.0949 -> 0.0563 (−41%)
    f32 tree vs f32 linear: 2560/2560 byte-exact
    all four parity tests PASS

### The accuracy trap that cost two rounds

The first version summed the 16 per-lane products SEQUENTIALLY. That is less
accurate than the layout it replaced — which summed 4 per lane and then reduced
across 32 lanes as a tree — and the f64 reference caught it:
**state rel L2 err 2.997e-7 -> 3.412e-7**. Small, but spec-decode acceptance is
sensitive to it: tau moved 6.917 -> 6.308 on one prompt and 2.414 -> 1.857 on
another, and the mix went 3.5% SLOWER despite a 38% faster kernel.

Tree-summing the 16 (`part[i] += part[i+w]`, w = 8,4,2,1) restores the depth and
is **more accurate than the original**: 2.9927e-7 vs 2.9973e-7. It is also
FASTER than the sequential version (0.0367 vs 0.0396 at t=10) because the tree
carries more ILP. Accuracy and speed moved together, not against each other.

### End-to-end: small, and smaller than the kernel win

6 prompts, 256 tokens each (17-94 cycles), fp32, B=10:

| | OLD | NEW |
|---|---|---|
| mix mean tok/s | 39.32 | 39.20 (−0.3%) |
| excluding p0 | 33.91 | **34.41 (+1.5%)** |

p0 is the shortest run (17 cycles) and the only one with a material tau shift
(7.118 -> 6.667). Five of six prompts favour the new kernel.

⚠️ **tau is noisy at short generation lengths and it swamps kernel wins.** At 96
tokens (~13 cycles) the same comparison read 3.5% SLOWER and looked like a
systematic regression; at 256 tokens it is +1.5% on 5 of 6 prompts. Any
rounding change reshuffles which drafted tokens are accepted, and that is worth
several percent either way on a short run. Do not judge a numerics-touching
change on <30 cycles.

The honest ceiling: DeltaNet is ~7% of the cycle, so even a 41% kernel win is
~2.5% end-to-end at best. The f16 state family (`gated_delta_net_f16.hip`,
`_f16_tree`, `_f16_routed_batch_seq`) is a SEPARATE byte-exact group and still
has the old layout — the same rewrite there would give the fp16 path (the
fastest config, 67.8 tok/s) the same treatment.

## f16 family: NOT landed — routed byte-exactness fails and I cannot explain it

Applied the identical 8-lane + pairwise transform to `gated_delta_net_f16.hip`,
`_f16_tree` and `_f16_routed_batch_seq`. Two of the three are fine:

    f16 tree vs f16 linear: 2560/2560 byte-exact   (dither ON *and* OFF)
    f32 tree vs f32 linear: 2560/2560 byte-exact

but `test_gated_delta_net_routed_f16` fails:

    routed f16 vs per-session f16 linear: 1713/3072 byte-exact, max|diff|=1.863e-9

REVERTED. What was ruled out, so nobody repeats it:

- **Not the step block.** A comment-stripped diff of the routed and linear
  kernels shows the two step blocks are character-identical; the only
  differences are the routing scaffolding and the dither index.
- **Not the dither.** Same 1713/3072 with `HIPFIRE_DN_STATE_FP16_DITHER=0`.
- **Not FP contraction.** `alpha*sreg[i] + kk[i]*delta` is contractible and the
  compiler may fuse it in one kernel and not another, which would give exactly
  this size of divergence. Pinning it with `__builtin_fmaf` in all six kernels
  changed nothing — the count stayed 1713 exactly.
- **Not geometry.** All nine GDN kernels are `#define HD 128`, `TILE_ROWS 4`,
  `__launch_bounds__(32, 8)`, launched `[32,1,1]`.

The count being bit-stable at 1713 across all of those says it is structural,
not a rounding lottery. Unresolved.

⚠️ **There is no `test_gated_delta_net_routed_f32`.** The f16 routed test is the
ONLY byte-exact cover for a routed kernel, which means the f32 routed kernel
that DID land is verified only indirectly (its tree sibling, and the
tolerance-based f64 oracle). If the routed transform has a defect, the f32 one
has it too and nothing currently catches it. Writing that test is the next step
and is worth more than the remaining 1-2%.

## f16 family LANDED — and the routed failure was FP CONTRACTION

The previous section said the f16 routed byte-exactness failure was
unidentified. It is identified, and the method that found it is the point:
**stop reading the kernels and localise the failure.**

Instrumenting the test to report WHICH elements differ took one run:

    mismatches by ROW: [(0,s0,216),(1,s1,189),(2,s0,261),(3,s1,208),(4,s0,253),(5,s1,232)]
    mismatches by HEAD: [330, 355, 339, 335]
    mismatching COLS: 128 of 128 — all 8 lane groups

Uniform across every row, head and column, and **row 0 already differs**, so it
is not accumulation, not routing, not the lane mapping. Identical inputs +
character-identical source + different results = the two kernels are COMPILED
differently.

Cause: HIP defaults to `-ffp-contract=fast`, which fuses `x += y*z` ACROSS
statements wherever scheduling suits. The pairwise reduction creates many such
sites and the compiler resolved them differently in the routed kernel than in
the linear one. `#pragma clang fp contract(on)` — contraction confined to a
single expression, so identical source gives identical code — fixes it:

    routed f16 vs per-session f16 linear: 3072/3072 byte-exact

Pinned in ALL SIX kernels, f32 included. The f32 trio landed WITHOUT it and was
carrying the same latent nondeterminism, invisible only because no routed-f32
test exists. Contraction pinning costs nothing measurable (t=10 0.0367 ->
0.0370, t=24 0.0563 -> 0.0560).

⚠️ **Byte-exactness between kernels is not a property of the source alone.**
Two kernels with character-identical bodies can disagree because the compiler
contracts differently under scheduling pressure. Any kernel family required to
agree bit-for-bit must pin contraction explicitly.

### fp16 end-to-end, 6 prompts x 256 tokens, B=10

| p | OLD tau / tok/s | NEW tau / tok/s | |
|---|---|---|---|
| 0 | 6.667 / 65.92 | 7.118 / 69.95 | +6.1% |
| 1 | 7.323 / 69.07 | 7.323 / 69.01 | −0.1% |
| 2 | 2.228 / 27.24 | 2.355 / 28.24 | +3.7% |
| 3 | 3.962 / 39.45 | 3.796 / 39.81 | +0.9% |
| 4 | 1.723 / 23.44 | 1.763 / 23.78 | +1.5% |
| 5 | 2.507 / 28.52 | 2.012 / 25.51 | −10.6% |

mix mean **42.27 -> 42.72 (+1.0%)**; on the three tau-stable prompts,
39.92 -> 40.34 (+1.1%). p5's tau swing dominates its row, in the opposite
direction from p0's — which is the tau-noise point again, now visible in both
directions.

Best single config measured: **fp16 state, B=10, 69.95 tok/s.**

## The missing cover, closed: `test_gated_delta_net_routed_f32` + a gate

Two gaps, both of which let the contraction defect reach master:

1. **No routed-f32 byte-exactness test.** The f16 one was the ONLY routed cover,
   so the f32 routed kernel was verified only through its tree sibling and a
   tolerance-based f64 oracle. Written now, mirroring the f16 fixture
   (interleaved sessions, per-session linear replay, byte-exact comparison).

   Proven to have teeth rather than merely passing: with
   `#pragma clang fp contract(on)` stripped from the f32 trio it fails at
   **exactly 1713/3072 byte-exact, max|diff| 1.863e-9** — the identical
   signature to the f16 failure. So the f32 trio *did* carry the same latent
   defect, and this test is what would have caught it.

2. **No gate ran ANY of them.** `parity_gated_delta_net_f64acc{,_routed}`,
   `test_gated_delta_net_tree_f32` and the two routed tests were manual
   examples. `tests/tiny-deltanet-gate.sh` now runs all five, and
   `tiny-affected-gate.sh` selects it on `kernels/src/gated_delta_net*.hip`,
   `dispatch/gated.rs`, `gdn_chunk.rs` and `qwen35/state.rs`. Verified:
   `--base HEAD~1 --dry-run` on the commit that touched all six kernels reports
   `deltanet=1`, and the gate itself passes 5/5 in ~8s with no model artifacts.

A byte-exactness invariant that nothing runs is not an invariant. Both f16 and
f32 routed kernels had the same defect; only the half with a test — a test
nobody ran automatically — showed it.

## 2026-08-27 — drafter quality: it was the adaptive-B controller, not the draft

Question asked: is the DFlash2 drafter functioning correctly, and can its
quality be improved? Answer: the drafter is fine. **The adaptive-B controller
was pinning it at B=4 on every prompt**, and that is what a whole session of
"the drafter saturates at tau 2.25 on code" was actually measuring.

`--block-size` is silently overridden per cycle by the controller (`adaptive_b`
defaults to **true**, range 2..16 clamped to the draft's trained block). Every
value of `--block-size` from 6 to 20 therefore produced a **bit-identical**
trace — same cycles, same accepted count, tau=2.2532 to four decimals. That
identity reads as "drafter saturation" and is not: it is one operating point
sampled fifteen times.

Qwen3.8-27B--oq4.25++ + its dflash2.oq4+ draft, 256 tokens, kvarn, Python BST
prompt (a background download was contending for UMA bandwidth, so absolute
tok/s is a floor; tau is exact and contention-free):

| config | tau | decode tok/s |
|---|---|---|
| adaptive (default) | 2.253 | 19.19 |
| `--no-adaptive-b --block-size 2` | 0.889 | 14.88 |
| `--no-adaptive-b --block-size 4` | **2.253** | 19.04 |
| `--no-adaptive-b --block-size 6` | 3.483 | **21.40** |
| `--no-adaptive-b --block-size 8` | **3.849** | 21.26 |

The adaptive row is byte-identical to the fixed B=4 row, which is what pins the
diagnosis: the controller settles on 4 and stays there. Fixed B=8 is **+71%
tau**; throughput peaks at B=6, same shape the 35B comment block already
documented ("tau rises monotonically with B; THROUGHPUT peaks at 6 and falls
... picking B to maximise tau maximises the wrong quantity").

So this is the SECOND model on which the shipped controller default loses to a
fixed B, and the failure mode is identical.

⚠️ **The attribution in the paragraph above is WRONG and is retracted.** The
controller does NOT maximise tau — it has optimised measured ms/committed-token
since 2026-04-24, and the EWMA/utilisation description at the top of
`dflash_spec_demo.rs` is a stale comment that outlived the code it described.
Root cause below.

Against the literature this drafter is **not** underperforming: EAGLE-3 reports
tau 3.21 on Qwen3-30B-A3B/HumanEval, and driven at its trained block this draft
does 3.85 on code. The earlier "30% below the published comparable" reading was
an artifact of the clamp.

Two corrections to earlier notes in this file:

- The baseline recipe above says "do NOT set HIPFIRE_KV_MODE=asym3; the numbers
  below are Q8 KV, which is this harness's default". Both halves are now stale:
  Q8 KV is deprecated, and the default is KVarN. kvarn4 and kvarn8 produce
  **bit-identical** traces here (tau 2.253, same cycles/accepted), so the KV
  tier is not a tau lever on this model either way.
- DFlash1 (`Qwen3.8-27B--dflash.oq4+`) carries `block_size: 16`, not 8. Re-run
  at its trained block it is still far behind DFlash2: tau 1.361 at both B=8 and
  B=16, with `anypos_match=0.019` — its proposals match the target at any
  position 2% of the time. DFlash2 is the only viable draft for this target.

### Open, and it gates the headline number

Spec-decode output is **not** byte-identical to `--ar-baseline` on this model —
and this is pre-existing, not caused by disabling the controller. Control run:
AR reproduces byte-for-byte across runs, while BOTH the default adaptive path
and fixed B=8 diverge from it. This is the same verify-forward divergence
already recorded for the 35B (`project_dflash_35b_verify_divergence`), now
confirmed on the 27B. tau remains a valid measure of draft-vs-verifier
agreement, but any "max tok/s via spec decode" figure for this model carries
that asterisk until the verify path is fixed.


### Root cause and fix — a biased estimator, not a wrong objective

The controller sweeps each candidate B and commits to the lowest measured
ms/committed-token. Its objective was already right. What was wrong is how it
estimated that cost.

It accumulated, per cycle, `elapsed / gained` and divided by the sample count —
a **mean of ratios**. That is a biased estimator of a rate: by Jensen it
overestimates ms/token, and the bias grows with the variance of `gained`. Since
`gained` ranges 1..B, **the variance, and therefore the penalty, grows with B.**

    B=8, one cycle gaining 1 token (60 ms) and one gaining 6 (60 ms):
      mean-of-ratios = (60/1 + 60/6)/2 = 35 ms/token
      true rate      = 120/7           = 17.1 ms/token
    At B=2, where `gained` is 1 or 2, the same bias is negligible.

So every comparison was rigged in favour of small B, and the controller walked
to the small end and stayed. Two compounding factors: `need = 2` samples, and an
ascending probe that abandons the rest of the range as soon as one candidate
scores worse than its predecessor — so a single unlucky pair at B=6 permanently
excluded B=8. Observed on the Python prompt before the fix:

    adaptive-b: range=2..=8 mean_B=3.97 changes=1 dist=[B=2:3.8% B=4:93.7% B=6:2.5%]

B=8 was never measured once.

Fix (`dflash_spec_demo.rs`): accumulate ms and gained separately and divide once
— cost is now a **ratio of sums** — and require a 5% margin before the ascending
probe abandons the range. The margin is a noise guard, not a thumb on the scale:
a candidate genuinely past the peak loses by far more than 5%.

    adaptive-b: range=2..=8 mean_B=5.28 changes=3 dist=[B=2:4.7% B=4:34.4% B=6:53.1% B=8:7.8%]

Python prompt: **19.12 -> 20.97 tok/s (+9.7%)**, closing the gap to the fixed-B
optimum (21.75) from 12% to 3.6%; the residue is exploration cost.

Validated on a prompt MIX, per this file's own standing warning, 192 tokens:

| prompt | adaptive | B=4 | B=6 | B=8 | vs best fixed |
|---|---|---|---|---|---|
| numbers 1..30 | 21.52 | 18.22 | 20.92 | **22.23** | −3.2% |
| B-tree prose | 13.75 | 13.91 | 12.88 | **13.92** | −1.2% |
| Python BST | 17.59 | 16.06 | **17.88** | 17.23 | −1.6% |
| MIT license | **12.91** | 12.86 | 12.84 | 12.51 | +0.4% |

The controller now picks a genuinely different B per prompt (B=8 on numbers,
B=6 on code and license), which is the only reason to have one. Summed across
the mix it ties the best global fixed choice (65.77 vs 65.89 for a fixed B=8)
while avoiding B=8's 3.6% loss on code. The prose and license spreads are inside
run-to-run noise; the real wins are `numbers` and `code`.

`./tests/no-gpu-ci.sh` green. Note `hipfire-eval`'s
`run_eval_reuses_cached_battery_rows` flakes under the parallel test run (shared
result-cache dir) and passes 3/3 in isolation — unrelated to this change.
