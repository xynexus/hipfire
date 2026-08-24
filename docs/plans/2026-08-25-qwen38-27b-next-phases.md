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
       HIPFIRE_GRAPH=1 HIPFIRE_KV_MODE=asym3
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
