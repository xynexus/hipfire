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

## The migration, fully specified

Traced to the bottom. Every primitive already exists and is arch-generic; what
is missing is one struct field and one call.

**The pager already owns the pointer tables.** An arch registers them with
`register_expert_ptr_tables(layer, ..)` (`weight_pager.rs:1499`), and
`ensure_expert_module_resident` patches the registered table on admission
(`:1056` comment). That is why the trait doc for `ExpertResidency::ensure_resident`
can promise that a resident expert "hence" has a live slot — the arch never
patches anything itself, which is also why the manual `patch_expert_ptr_table`
has no callers.

**The gap is that deepseek4's decode path cannot carry a residency provider.**
deepseek4 is top-6 and routes through `run_moe_decode_bias_aware`, which takes
`MoeBiasAwareParams`. That struct carries `expert_gate_up_ptrs` and
`expert_down_ptrs` — the tables — but has **no** `expert_residency` field.
`expert_residency` lives on `MoeParams` (`families/moe.rs:367`) and is consumed
only by `run_moe_decode` (`pipeline/mod.rs:492`). So the bias-aware sibling has
the tables and not the hook.

The work, in order:

1. Add `expert_residency: Option<&'a dyn ExpertResidency>` to
   `MoeBiasAwareParams`, mirroring `MoeParams:367`.
2. Call `ensure_resident` for the selected experts in
   `run_moe_decode_bias_aware`, mirroring `run_moe_decode`'s use at
   `pipeline/mod.rs:492`.
3. In deepseek4's loader, when paged: `register_expert_modules(hfq.modules())`
   and `register_expert_ptr_tables` per layer, pass a residency provider into
   the MoE params, and **stop building the per-layer `combined` upload**
   (`arch.rs:298-330`) — that upload is the thing that dies at layer 19.

Steps 1 and 2 are mechanical mirrors of code that already exists in the sibling
function. Step 3 is the arch work, and its test is the artifact above.

### Step 3's shape, and its fast test loop

Steps 1 and 2 are done (`ds4-biasaware-residency-hook`): `MoeBiasAwareParams`
now carries `layer_idx` and `expert_residency`, and `run_moe_decode_bias_aware`
consults the provider between the top-K and the indexed GEMV. Every caller
passes `None`, so it is inert.

Step 3 is four parts, and it is more than "register the modules":

1. **Somewhere to hold the pager.** qwen35 threads
   `Option<&RefCell<WeightPager>>` as a parameter (`qwen35/mod.rs:3332`).
   deepseek4's `ffn_routed(cfg, weights, state, gpu, layer_idx, routed_out)` has
   no such parameter, and there are three call sites
   (`forward.rs:1962`, `:2101`, `:2650`) plus their callers.
2. **A residency provider.** qwen35 has a local `PagerExpertResidency { pager }`
   implementing the generic trait; deepseek4 needs the equivalent, or that type
   moves somewhere shared.
3. **Loader registration.** `register_expert_modules(hfq.modules())` plus
   `register_expert_ptr_tables` per layer. deepseek4's loader has **no paged
   branch at all** today — grep finds nothing.
4. **Drop the `combined` upload** (`arch.rs:298-330`) on the paged path. This is
   the 1152 MiB-per-layer allocation that dies at layer 19.

**The test loop is fast, which matters more than the size.** The tiny deepseek4
fixture quantizes to an artifact carrying **16 RoutedExpert modules** (2 layers ×
8 experts), so paging can be exercised in seconds against `tiny-quant-gate` /
`tiny-state-gate` rather than a ten-minute 82.78 GB load. Develop against the
fixture; use the big artifact once, as the acceptance test.

## What does not block it

The EP smoke on medusa is the *validation* Phase 3b asks for, and medusa is
unreachable. But the migration has a stronger local test than an EP smoke: **does
this 82.78 GB artifact load and decode on a 43 GB box?** Today it dies at layer
19. That is a sharper pass/fail than any tiny fixture, and it needs no second
GPU. The EP predicate can stay untouched and unexercised while single-GPU paging
is proven.
