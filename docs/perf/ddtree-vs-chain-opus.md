# DDTree beats chain DFlash on an Opus target (and chain can lose to AR)

**Measured 2026-08-31, halo (gfx1151, 128 GB UMA).**
Target `Qwen3.6-27B--oq4.25++.hfq`, drafter converted from
`z-lab/Qwen3.6-27B-DFlash` (block_size 16), `examples/dflash_spec_demo`,
128 tokens, greedy.

| prompt | AR | chain DFlash | DDTree (budget 12, topk 1) |
|---|---|---|---|
| code (RFC 3339 parser) | 15.23 tok/s | **10.26** (0.67x AR) | **32.89** (2.16x AR, 3.21x chain) |
| prose (coral reefs) | 15.25 tok/s | 19.02 (1.25x AR) | **21.87** (1.43x AR, 1.15x chain) |

Reproducible: the code run repeated at 32.92 tok/s (~0.1% variance).

## Why this needed measuring at all

`speculative.rs:145` records that `spec_step_ddtree_batched` died at the first
cycle with "unsupported target.output dtype" on Opus targets, because the two
DDTree draft helpers carried their own dtype ladder that stopped at the MQ/HFQ
families. In its own words: *"Chain mode was not chosen over tree mode on this
family; tree mode was never reachable to be measured."*

So the standing "tree verify loses" verdict came from families where tree mode
ran. It does not transfer to Opus, and on Opus it is **backwards**.

## Chain DFlash is the one that loses, and only on some prompts

On the code prompt chain runs at **0.67x AR** — speculation actively costs
throughput. Its acceptance explains it: tau = 2.49 but accept_rate = 0.166, i.e.
it proposes 15 tokens per cycle (B=16) and keeps ~1.5. Every rejected token is a
verify slot paid for and thrown away.

On prose the same drafter manages 1.25x. The drafter is not uniformly bad; the
fixed B=16 chain is simply the wrong shape when acceptance is low, because a
chain cannot spend its width on alternatives — only on depth it will not reach.

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
What wins is verifying that spine in ONE batched forward through the DDTree
machinery — and `spec_step_dflash` at B=16, doing nominally the same work, gets
10.26. That 3.2x gap between two implementations of one operation is the most
interesting number here and is not a tuning result.

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

* **Explain the 3.2x chain-vs-spine gap.** Two implementations of one operation,
  and the slow one is below AR. Equalise block size first (chain at B=12 against
  budget 16) so it is not a confound, then trace. Leading hypothesis: chain issues
  per-token verify launches where the tree path issues one batched forward.
* Re-run on a MoE target (`Qwen3.6-35B-A3B--oq4.25++`) — expert routing adds a
  per-cycle cost this dense target does not pay.
* DDTree is the only path with `supports_temp_verify() == true`, so this is also
  the only speculation available above temp 0. Worth measuring there.
* Budget was swept 4-16 at topk=1; the optimum (12) is interior, so the range is
  no longer the binding constraint.
