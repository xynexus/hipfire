# §M4's premise is false — route/combine are already separate dispatches

Status: 2026-08-22, read from the code on nix1. Amends
`2026-08-22-m4-decided.md` and §M4 of the parent plan.

## The claim

§M4 blocks itself on this:

> among its fast paths is an *indexed* routed-expert GEMV ... described in its
> own comment as "the device-side top-K + indexed expert GEMV path". Routing and
> expert compute are **one kernel** there.
>
> Splitting `Moe` into route / expert / combine means **unfusing** that:
> materialise per-expert intermediates the fused kernel exists to avoid, and pay
> a D2H plus a kernel launch per expert per token.

## They are not one kernel

The indexed decode path is a sequence of distinct dispatches that hand off
through scratch tensors (`qwen35/moe_decode.rs`):

| # | dispatch | line | produces |
|---|---|---|---|
| 1 | `softmax_f32(router_logits)` | 1234 | router probs |
| 2 | `moe_topk_renorm_k8(...)` | 1235 | `s.topk_indices`, `s.topk_weights` |
| 3 | `rotate_x_mq_awq_indexed_batched(...)` | 1483 | `s.rot_batch` |
| 4 | `gemv_oq4g256_moe_gate_up_k8_indexed[_batched]` | 1497/1509 | `s.gate_batch` … |
| 5 | `moe_down_combine_k8_batched(s.down_expanded, s.topk_weights, …)` | 1694 | residual |

Every intermediate is a `MoeScratchRef` field (`:51-64`) — a real device tensor,
not a register kept inside one kernel. §M4 already notices this
("`MoeScratchRef` already materialises `router_logits` / `topk_indices` /
`topk_weights`") but files it under *"the seam is real on the unfused paths"*.
It is the **indexed** path that writes `s.topk_indices` at line 1235.

The comment §M4 quotes says "device-side top-K **+** indexed expert GEMV
**path**" — a path with two stages, which is what it is.

## What is actually fused, and what that costs

The fusion is **across the k expert slots**, not across route/compute/combine:
one `*_k8_indexed*` kernel handles all 8 slots. So:

- `MoeRoute` / `MoeExpert(all-k)` / `MoeCombine` as three super-ops needs **no
  unfusing at all**. The dispatch boundaries already exist, the intermediates are
  already materialised, and the cost is zero.
- `MoeExpert(e)` **per slot** is the part that needs unfusing, and that is where
  §M4's "a kernel launch per expert per token" objection actually bites.

§M4 conflated the two. The three-way split it names is free; the per-slot split
it costs out is a different, more expensive change.

## Why this matters

**A coarse `Escape` is not required.** `2026-08-22-m4-decided.md` concluded that
qwen35 keeps a single indivisible `Moe` quantum because the unfused path will not
load. That conclusion stands for *per-slot* granularity. It does not hold for the
three-way split, which is reachable on the default indexed path with no residency
change, no format change, and no throughput cost.

Against §M6's drain budget that is the difference between one yield point per MoE
layer and three — 120 rather than 40 across a 40-layer forward — without touching
the kernel contract that `2026-08-22-unfused-oom-mechanism.md` shows is the real
blocker for the per-slot version.

## Not claimed

That the three-way split is *implemented* — it is not. What is established is
that its stated blocker does not exist: the boundaries are already dispatch
boundaries. Lowering `Moe` into three `SuperOp`s and teaching `ForwardBindings`
the three entry points is ordinary work, and `run_layer_program_from` (#303)
already provides the yield point that would consume it.
