# TriAttention eviction under KVarN: decline the sidecar, don't panic

State box: halo, Strix Halo gfx1151, 128 GB UMA. Qwen3.8-27B oq4.25++,
`kv_mode=kvarn`, TriAttention sibling sidecar present.

## The bug

`EvictionCtx::maybe_evict` matched the KV quant flags and, on no match, ran

    panic!("TriAttention eviction only supports Q8, asym2, asym3, asym4 KV modes for now");

`maybe_evict` fires once the physical cache reaches `budget + beta`. With a
TriAttention sidecar the KV is deliberately allocated at `physical_cap =
budget + beta + 256` (896 at budget 512 / beta 128) rather than `max_seq`,
precisely so eviction bounds VRAM. So under `kv_mode=kvarn` **any prompt past
896 tokens killed the daemon**, mid-generation, with a panic and no fallback.

## Why eviction cannot simply learn KVarN

Every supported mode compacts by gathering retained positions — a per-position
copy into `k_compact`/`v_compact`. KVarN's K is not addressable that way: it is
4-bit var-norm records covering **128-token blocks**, so dropping arbitrary
positions means re-encoding every surviving block. That is new kernel work, and
worse, it would re-quantize already-quantized values on every eviction cycle,
compounding error the longer the context runs. V is Q8 and would have been fine;
K is the blocker.

## The fix

Decline the sidecar instead of half-supporting it. The load path now filters `resolved_triattn`
to `None` for any `kv_mode` that is not known-evictable, before `cask_requested`
is derived, so `physical_cap` falls through to `max_seq` and no eviction context
is built. It says so out loud:

    TriAttention component (sibling) declined: eviction cannot compact
    kv_mode=fp32 — KV sized for the full context window instead

Nothing is given up. Eviction exists to bound KV cost, and KVarN already does
that — ~8x on K, **without dropping a single token**. Stacking a lossy
token-dropping policy on top of a lossy-compression one was never the better
trade. `HIPFIRE_KV_PHYSICAL_CAP` still caps the allocation for operators who
want a smaller one.

**This is not a KVarN-only bug.** `maybe_evict` dispatches on `quant_asym3`,
`quant_asym4`, `quant_asym2` and `quant_q8`; the `kv_mode` tokens that set one of
those are `asym2/3/4`, `fwht2/3/4` (asym plus a rotation) and `q8`. **`fp32`
reaches no arm either**, so unquantized KV + a sidecar panicked exactly the same
way — that was the first crash seen this session, before KVarN was even
suspected. `triattn_can_evict_kv_mode` is therefore a WHITELIST: an unrecognised
or newly added mode declines the sidecar rather than failing a request the moment
the cache reaches `budget + beta`. Verified `asym4` still builds eviction at
`physical_cap=896`, unchanged.

Separately, `maybe_evict`'s `panic!` is now a returned `HipError`. The load-path
filter means it should be unreachable for KVarN, but a panic takes down the
whole daemon and a request error does not; defence in depth for any mode that
reaches it another way.

## Measured

The 2059-token prompt that previously killed the daemon now completes and is
coherent.

| | before | after |
|---|---|---|
| 2059-token prompt | **daemon panic** | 209.4 tok/s prefill, coherent |
| `physical_cap` | 896 | 8192 (= max_seq) |
| `runtime_state_bytes` | — | 1.04 GB |

1.04 GB of KV state for a full 8192-token window, against a 15.5 GB model on a
128 GB box. That is the point of KVarN: the full-window cache is cheap enough
that token-dropping eviction is not needed to afford it.

no-gpu-ci PASS.

## Where this leaves prefill throughput

Worth recording, because the fix exposes it: with the sidecar declined, prefill
on the default config is **14.4 tok/s** (kvarn) and **15.0** (fp32) — the
per-token path. The 210 tok/s figure requires `HIPFIRE_KVARN_BATCHED_PREFILL=1`.
Batched prefill is not reachable any other way on this model: fp32 KV fails
`fa_kv_ok`, so it stays per-token whether or not `HIPFIRE_PREFILL_BATCHED` is
set, and under KVarN the batched path is gated off because
`forward_prefill_batch` runs its own batched attention and never populates the
KVarN window/records.

So the single gate between the default config and a **~14x** prefill speedup is
the KVarN batched-prefill faithfulness bug, not anything in the GEMM.
