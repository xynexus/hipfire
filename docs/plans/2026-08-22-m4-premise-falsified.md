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

## Where the super-op boundary can actually go

It is tempting to read the above as "so binding `MoeRoute` is a small change".
**It is not, and the reason sits one line above the seam.**

`router_logits` is produced at `moe_decode.rs:1213` by the **fused 4-way GEMV** —
router + `shared_expert_gate` + `shared.gate` + `shared.up` — whose own comment
records that the four "read the SAME rotated `x_rot_local` with the SAME K" and
that fusing them "saves 3 launch submits per MoE layer and lets underused tails
(shared_expert_gate_m=1, router_m=256) co-schedule".

So routing cannot be hoisted *above* the expert-side work into a leading
`MoeRoute` op: it depends on a kernel deliberately fused with three
shared-expert GEMVs. Hoisting means unfusing that GEMV — the same class of cost
§M4 objected to, at a boundary the sections above had not checked.

What is reachable is a two-way split at a **different** point:

```
stage A: norm/rotate → fused 4-way GEMV → top-K      (ends at moe_route_topk)
stage B: routed experts → combine
```

That needs no unfusing, and still gives two quanta per MoE layer rather than
one. But it means splitting `moe_ffn_decode_impl` into two functions sharing the
locals that live across the seam — the shared-expert block at `:1283-1347` sits
between them and uses stage-A state. A restructure, not a binding.

The three-way `MoeRoute` / `MoeExpert` / `MoeCombine` shape §M4 names is
reachable only after the 4-way GEMV is unfused, or with `MoeRoute` redefined to
include it.

## What the stage split can actually be verified against here

A restructure of `moe_ffn_decode_impl` is only as safe as the arms you can run,
and the arms are better covered on this box than they look. Checked, not assumed:

| arm | exercised by | here? |
|---|---|---|
| indexed k8, OQ4 | `Qwen3.6-35B-A3B--oq4` (k=8) | **yes** |
| CPU top-K fallback | every tiny MoE fixture — all emit `num_experts_per_tok=2`, and `use_gpu_topk` needs k=8 | **yes** |
| OQ8 / OQ4+ / OQ4++ / OQ4.25++ | tiny-quant `qwen3_5_moe`, `lfm2_moe` | **yes** |
| MQ3 / MQ4 / MQ6 / q8f16 | tiny-quant | **yes** |
| MQ2-Lloyd | `deepseek4` fixture (`--allow-mq2-lloyd`, `tiny-state-gate.sh:143`) | **yes** |
| paged experts | `HIPFIRE_QWEN35_PAGED_EXPERTS=1` — both artifacts carry a routed-expert module table (tiny: 32 modules, 35B: 10 240); verified to register and load | **yes** |
| ParoQuant (`ParoQ4G128`) | no fixture emits it | no |
| EP sharding | needs multi-GPU; medusa unreachable, halo single-GPU | no |

Six of eight, and crucially **both** top-K regimes: the tiny fixtures are all
k=2 so they run the CPU fallback, while the 35B is k=8 so it runs the indexed
path. A split verified against both, plus paged and MQ2-Lloyd, leaves only paro
and EP unexercised — and those two should be stated as the residual risk rather
than used as a reason not to start.

## Not claimed

That the three-way split is *implemented* — it is not. What is established is
that its stated blocker does not exist: the boundaries are already dispatch
boundaries. Lowering `Moe` into three `SuperOp`s and teaching `ForwardBindings`
the three entry points is ordinary work, and `run_layer_program_from` (#303)
already provides the yield point that would consume it.
