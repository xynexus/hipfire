# DDTree beats chain DFlash on an Opus target

> ⚠️ **CORRECTED 2026-09-01.** The first version of this file reported chain
> DFlash at 10.26 tok/s and concluded it "loses to AR" by 0.67x, making the
> chain-vs-tree gap 3.2x. **That chain number was a cold-JIT-kernel-cache
> artifact.** Re-measured warm and repeated: chain is **22.1 tok/s**, which BEATS
> AR by 1.45x. The real tree-vs-chain gap is **1.47x**, a tuning difference
> explained by tau, not a defect. See "The cold-cache trap" below — the effect is
> 3.45x and it invalidates any first-run benchmark in this repo.

**Measured 2026-08-31, halo (gfx1151, 128 GB UMA).**
Target `Qwen3.6-27B--oq4.25++.hfq`, drafter converted from
`z-lab/Qwen3.6-27B-DFlash` (block_size 16), `examples/dflash_spec_demo`,
128 tokens, greedy.

All arms warm, same session, repeated twice:

| arm | pass 1 | pass 2 | tau | vs AR |
|---|---|---|---|---|
| AR | 15.25 | 15.25 | 1.00 | 1.00x |
| chain DFlash B=16 | 22.31 | 21.98 | 2.49 | **1.45x** |
| DDTree budget 12, topk 1 | 32.69 | 32.76 | 5.79 | **2.14x** |

Both speculative paths beat AR. Tree beats chain by **1.47x**, and the whole of
that is acceptance: tau 5.79 vs 2.49 at the same block size (chain at B=12 and
B=16 both give tau 2.4865 — the extra width is pure waste, accept_rate falls
0.226 -> 0.166 while accepted tokens stay at 92).

## Why this needed measuring at all

`speculative.rs:145` records that `spec_step_ddtree_batched` died at the first
cycle with "unsupported target.output dtype" on Opus targets, because the two
DDTree draft helpers carried their own dtype ladder that stopped at the MQ/HFQ
families. In its own words: *"Chain mode was not chosen over tree mode on this
family; tree mode was never reachable to be measured."*

So the standing "tree verify loses" verdict came from families where tree mode
ran. It does not transfer to Opus, and on Opus it is **backwards**.

## The cold-cache trap — read this before benchmarking anything here

Kernels are JIT-compiled and cached under `~/.hipfire/kernels/<arch>/`. The
compile happens **inside the timed window**, so the first run of any new kernel
mix measures compile time, not throughput. Measured directly by moving the cache
aside:

| kernel cache | chain B=16 | tau |
|---|---|---|
| cold (1846 entries removed) | **6.41 tok/s** | 2.4865 |
| warm (566 regenerated) | **22.13 tok/s** | 2.4865 |

**3.45x, with tau bit-identical.** That identity is the tell: the computation is
the same, only the wall-clock differs. It is what makes the trap dangerous —
acceptance metrics look perfectly healthy while throughput is a third of real.

The original 10.26 for chain was this, partially. AR was unaffected because it
was not the first run and shares fewer kernels.

**Rule: discard the first run after any binary rebuild, kernel edit, or new
arm.** A benchmark's control must be re-run in the same warm session as its
treatment, never quoted from an earlier one.

## The winning config is topk=1 — a linear spine, not a tree

Sweeping both axes changes the conclusion. First topk at budget 8, with
`HIPFIRE_DDTREE_TAPE_DUMP=1` reporting how often the LA fixup slow path fires:

| topk | fast/slow cycles | tok/s | tau |
|---|---|---|---|
| 1 | 25 / 0 | **28.11** | 4.28 |
| 2 | 14 / 9 (39% slow) | 25.90 | 4.61 |
| 4 | 12 / 12 (50% slow) | 23.71 | 4.50 |

Widening buys +0.33 tau and costs 39-50% slow cycles. Throughput falls
monotonically. Then budget at topk=1, where no cycle ever hits the slow path:

| budget | tok/s | tau | slow |
|---|---|---|---|
| 4 | 23.64 | 3.03 | 0 |
| 6 | 28.08 | 4.00 | 0 |
| 8 | 28.13 | 4.28 | 0 |
| **12** | **32.89** | **5.79** | **0** |
| 16 | 30.51 | 5.95 | 0 |

**So the win was never tree-ness.** topk=1 is a linear spine with no branching.
What wins is verifying that spine in one batched forward, and getting a longer
accepted run out of it: tau 5.79 against chain's 2.49 at the same block size.
That is a real 1.47x, but it is an ACCEPTANCE difference, not two
implementations of one operation differing by 3.2x — that reading came from the
cold-cache number and is withdrawn.

Two things worth keeping about the tree axis:

* **Tuning on tau would pick the wrong config.** At topk=1 tau still rises from
  5.79 to 5.95 between budget 12 and 16 while throughput FALLS. Any auto-tuner
  must optimise wall-clock, not acceptance.
* **Tree width is not free on a hybrid arch.** `spec_step_ddtree_batched` runs one
  forward over the linearized tree; the attention layers get correct per-branch
  views from the mask, the LA layers cannot be masked, so `gdn_tape` captures
  per-position innovations and only the committed path is replayed. When the
  committed path is not a contiguous prefix (`spine_accept = false`) it pays a
  second exact verify plus a `kv_compact_gather` tape rearrangement. That is what
  the 39-50% slow-cycle rate above is measuring.

### Would duplicating GDN state remove that penalty?

Cheap to answer: one full GDN state for this model is **149.6 MiB** (48 LA layers
x [48 heads x 128 x 128 f32 S-matrix + 30720 f32 conv state]). Sixteen branches is
2.34 GiB — trivial on a 128 GB box, so capacity is not the obstacle.

But it would buy only the elimination of the slow path, and the best config never
hits it. Duplication also has its own bandwidth bill (B x 149.6 MiB read+write per
cycle, ~9.4 ms at B=8 against a ~65 ms step) plus a batch dimension through the LA
recurrence kernel. Netting out, it lands within noise of where topk=1 already is.
Not worth building.

## Caveats

* One target, one drafter, two prompts, 128 tokens, greedy only. Enough to
  overturn a verdict, not enough to set a default.
* Qwen3.6-27B is dense in the MoE sense only. It has **48 linear-attention
  layers of 64**, so the hybrid-recurrent path WAS exercised — an earlier draft of
  this file claimed otherwise and was wrong. What was not exercised is a MoE
  target, where expert routing adds its own per-cycle cost. `qwen4_exp` remains
  out of scope for a different reason: it has no batched forward at all.
* Every `*-DFlash--bf16.hfq` on this box is **arch 1**, not arch 20, so
  `DflashConfig::from_hfq` rejects all of them and no DFlash path can run from
  them. They carry `config.dflash_config` (mask_token_id + target_layer_ids)
  rather than the top-level `dflash` key the loader wants. Rebuild drafters from
  the HF checkpoints with `dflash_convert --input <snapshot> --output <x>.hfq`;
  the sources are on `/srv/huggingface/models--z-lab--*DFlash*`.

## Next

* **Why does the same drafter yield tau 5.79 through the tree path and 2.49
  through chain?** Block size is NOT the confound — chain gives tau 2.4865 at
  both B=12 and B=16, so the extra proposals are pure waste. The tree path
  truncates on confidence where chain proposes B-1 unconditionally. Quantifying
  that is the next real question.
* Re-run on a MoE target (`Qwen3.6-35B-A3B--oq4.25++`) — expert routing adds a
  per-cycle cost this dense target does not pay.
* DDTree is the only path with `supports_temp_verify() == true`, so this is also
  the only speculation available above temp 0. Worth measuring there.
* Budget was swept 4-16 at topk=1; the optimum (12) is interior, so the range is
  no longer the binding constraint.
