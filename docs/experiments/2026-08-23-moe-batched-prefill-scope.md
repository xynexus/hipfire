# MoE batched prefill: the gate was dead code, the body is the real work

2026-08-23, halo/gfx1151, Qwen3.6-35B-A3B--oq4.25++ (K=8, E=256), kvarn KV.

## Two bugs in the gate, both fixed here

`prefill_batch_pbs_eligible` carried a blanket

    all(|lw| matches!(lw, DeltaNet | FullAttn))

term that rejected every MoE model **before** the per-layer term below it could
be consulted -- and that per-layer term has complete `DeltaNetMoe` /
`FullAttnMoe` arms (moe_topk_ok, moe_router_logits_present, projection dtypes,
moe_ffn_batched_admissible). Those arms were unreachable, permanently. So was
`prefill_chunk.rs`'s matching `(LayerWeights::DeltaNetMoe(..), LinearAttention)`
body. Removed.

The comment beside it was also wrong: it claimed a MoE model "REQUIRES
dn_state.quant == Q8", while the code requires `FP32 | FP16`. Both cannot hold;
MoE could never pass. Corrected.

Measured before removal -- every MoE-specific input already passed:

    all_layers_dense_la=false   <- the only decline
    moe_topk_ok=true (K=8, E=256)
    router_logits=true
    dn_quant=FP32   n=720

## And a diagnostic, because the aggregate could not name the failure

The per-layer term is one `all(..)` over a five-arm match, so a false verdict
said nothing about which arm or which dtype. Added a trace (under
`HIPFIRE_KERNEL_TRACE=1`) that names the FIRST declining layer and dumps its MoE
dtype profile. It immediately produced the real blocker:

    per-layer DECLINE at layer 0 (DeltaNetMoe): wqkv=OqCompactG256 ffn_admissible=false
      dtypes router=Q8_0 shared_gate=OqCompactG256 shared_up=OqCompactG256
             shared_down=OqCompactG256 expert_gate_up=Oq8G256 expert_down=Oq8G256

## What is actually left: the BODY predates the Opus quants

This model is **shared experts compact, routed experts Oq8**. Three gaps:

1. **`moe_prefill_quant_family_supported_for_arch`** lists `Oq4G256 | Oq8G256`
   and not `OqCompactG256`, so `moe_ffn_batched_admissible` returns false. That
   is the literal decline -- but admitting it alone would be WRONG, see 2 and 3.

2. **`prefill_moe_ffn_body_batched`** (prefill_chunk.rs:65-1872) has 5 `Oq8G256`
   and 6 `Oq4G256` references and **zero `OqCompactG256`**. Shared-expert gate_up
   dispatches through `KernelKey::FusedGateUpOq4G256` / `FusedGateUpOq8G256`;
   there is no compact equivalent. Routed experts (Oq8) are already covered.

3. **The MoE attention arms predate Opus entirely.** Both
   `(DeltaNetMoe, LinearAttention)` and `(FullAttnMoe, FullAttention)` gate on
   `let is_mq = matches!(.., MQ4G256 | MQ6G256)` with no Opus dtypes and no
   compact dispatch arm -- 0 Opus references across 1870 lines. `is_mq` is what
   triggers the FWHT(+AWQ) activation rotation, so admitting a compact MoE layer
   without adding it there would feed an UNROTATED activation to an Opus GEMM.
   The dense path records what that looks like: "garbage: PPL 3.5e6".

So the remaining work is ~4 attention dispatch regions (2 matchers + ~4 arms,
transplantable from the dense arms in prefill_lowered.rs -- the MoE layer uses
the same `layer.wqkv/wz/w_beta/w_alpha` and `pbs.dn_*_batch` field names) plus
compact shared-expert support in the FFN body, then the admission in 1.

**Not done here, and deliberately not half-done**: adding the admission without
2 and 3 produces silently wrong numbers rather than an error.

## Current state is safe

With the veto removed, Qwen3.6-35B-A3B still declines -- correctly, on
`ffn_admissible=false`. No model on this box changes behaviour. What changed is
that the MoE arms are now REACHABLE, so finishing the body is all that remains.

Payoff, from `2026-08-23-pflash-blocked-on-moe-batched-prefill.md`: MoE prefill
is 54.8 tok/s against dense 179.8, and this gap also blocks PFlash entirely.
