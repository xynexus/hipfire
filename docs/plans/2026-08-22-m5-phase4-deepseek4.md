# §M5 Phase 4 — deepseek4's need for eviction, measured

Status: 2026-08-22, nix1 / gfx1103 (43 008 MiB). Artifact built this session.

Phase 4 says *"enable eviction where it pays. deepseek4 is the arch that wants
it most... Gate on a measured decode-latency delta, not on the mechanism
working."* This records the half that is now measured and the exact reason the
gate itself cannot yet be run.

## The artifact

`/srv/hipfire/models/models--deepseek-ai--DeepSeek-V4-Flash.hfa` (158.7 GB) →
`hipfire-coexistence repack` on carbon (159.6 GB of safetensors, lossless) →
`hipfire-quantize --format oq4` → **`DeepSeek-V4-Flash--oq4.hfq`, 82.78 GB**.

    arch      9 (deepseek4)          layers 43     experts 256    top-6
    hidden    4096                   heads  64     kv heads 1
    modules   11 008 RoutedExpert    (= 43 x 256)

Worth noting: despite `--format oq4` the bulk landed at **MQ2G256Lloyd**
(77.91 GB across 33 024 tensors); only 3.18 GB is Oq4G256. deepseek4's routed
experts take the Lloyd path.

## The measurement

It does not load. With the F32-globals fix (`ds4-accept-f32-globals`) it gets
past `attn_sink` and dies here:

    upload gate_up layers.19: hipMalloc(1207959552 bytes = 1152.00 MiB),
    free=521.9 MiB of total=43008.0 MiB

**Layer 19 of 43.** So roughly 42.5 GB of the 43 GB device is consumed by
nineteen layers' experts, and the remaining twenty-four cannot be placed.

That is Phase 4's premise as an observation rather than a claim: 82.78 GB of
weights against 43 GB, and deepseek4 uploads **every expert resident**. Its
loader builds one `combined` Vec per layer — all owned experts' `w1`‖`w3`
concatenated — and uploads it as a single tensor
(`hipfire-arch-deepseek4/src/arch.rs:298-330`), deriving expert pointers as
`base + local * stride`.

## Why the gate cannot be run yet

The gate wants a decode-latency delta from *enabling* eviction. There is nothing
to enable: **deepseek4 has zero `WeightPager` usage** — no `register_expert_module`,
no `ensure_expert_module_resident`, nothing. Grep the crate and the pager does
not appear.

So Phase 4 is gated on the Phase 3b migration, and the container side is already
done — the artifact ships all 11 008 routed-expert modules, so
`register_expert_modules` has something to consume.

## What the migration actually involves

From how qwen35 does it:

1. **Load time** — one bulk call,
   `pager.register_expert_modules(hfq.modules(), ..)`
   (`qwen35/loading.rs:4738`), instead of building the per-layer `combined`
   upload.
2. **Decode time** — `pager.ensure_expert_module_resident(key, gpu)` for the
   selected experts before dispatch (`qwen35/moe_decode.rs:363`, `:488`).
3. **The pointer table** — and this is the real work. deepseek4's forward assumes
   experts are contiguous at `base + local * stride`. Paged experts live at
   arbitrary pager addresses, so the table must be built from actual module
   addresses.

Note on (3): `patch_expert_ptr_table` looks like the answer and is not.
`qwen35/layout.rs:92-93` records that it "has zero call sites workspace-wide" —
qwen35 does not patch per token, it makes the selected experts resident and the
table follows by construction. Whatever deepseek4 does must be chosen
deliberately rather than by copying a function that nothing calls.

## What does not block it

The EP smoke on medusa is the *validation* Phase 3b asks for, and medusa is
unreachable. But the migration has a stronger local test than an EP smoke: **does
this 82.78 GB artifact load and decode on a 43 GB box?** Today it dies at layer
19. That is a sharper pass/fail than any tiny fixture, and it needs no second
GPU. The EP predicate can stay untouched and unexercised while single-GPU paging
is proven.
