# distinct_experts(B): how a MoE target's activated expert set grows with tree size

Date: 2026-07-21
Task: scope gate #1 of `docs/plans/2026-07-21-disaggregated-draft-target-protocol-scope.md`
(§9, §11 — the load-bearing measurement for streaming-MoE tree-verify).

## The question

For a streaming / expert-offloaded MoE target, one tree-verify pass costs ≈
streaming the **union of activated experts** once. A DDTree draft tree verifies
`B` candidate tokens together. So the useful tree size is bounded not by compute
but by how fast that expert union grows with `B` — `distinct_experts(B)`. The
sharp sub-question that decides whether wide trees are cheap:

- **DEPTH axis** — union over `B` *consecutive* positions of a real sequence
  (models a linear accepted chain).
- **WIDTH axis** — at a fixed prefix, the target's own top-`W` next-token
  candidates (siblings sharing a parent), union of *their* routing (models
  branching at one tree position).

Hypothesis under test: same-prefix branches route alike, so **width adds few new
experts** → wide trees are ~free (depth-bounded). Baseline: random-uniform
`E·(1−(1−k/E)^B)` and an empirical random-token union.

## Verdict (short)

**Width is the cheaper axis, and its advantage grows with the expert pool — but
width is NOT free.** Branching at one position does add experts (different
next-token choices route somewhat differently), just at ~0.6–0.7× the rate of an
extra depth position, and both axes stay well under "most experts" until large
`B`. The strong "width ≈ depth-bounded / free" hypothesis is **refuted**; the
weaker, still-decisive claim — *tree width is materially cheaper than depth and
the gap widens as `E` grows* — is **confirmed** across a 16× expert-pool ladder.

| model | E | k | at B | depth (%E) | width (%E) | width/depth |
|---|---|---|---|---|---|---|
| LFM2.5-8B-A1B | 32 | 4 | 512 | 89% | 78% | 0.87 |
| Qwen3.6-35B-A3B | 256 | 8 | 256 | 65% | 43% | 0.66 |
| **397B (projected)** | 512 | 10 | 256 | ~65% | ~43% | ~0.66 → lower |

The width/depth ratio falls from 0.87 (E=32) to 0.66 (E=256); extrapolated to
E=512 it keeps improving. **More experts ⇒ wider-trees-cheaper.**

## Method

- **Capture.** HF `transformers` 5.10.2 forward with the router-selection method
  of each sparse-MoE block monkeypatched to record the per-token top-k expert
  ids (`benchmarks/scripts/distinct_experts_moe.py`, class `Capture`). Works for
  LFM2-MoE (`route_tokens_to_experts`) and the Qwen3.5-MoE family
  (`Qwen3_5MoeTopKRouter.forward`, selected-experts auto-detected as the int
  `[N,k]` tensor). No sampling of the model — exact router argtopk.
- **Corpus.** 30 English Wikipedia articles (≥4000 chars) from
  `wikimedia/structured-wikipedia`, pre-extracted to
  `benchmarks/corpora/wiki_moe_corpus.jsonl` (md5 `4485ada1f1a3f277f71963b84a5a7f2f`)
  so both models see byte-identical text.
- **Depth axis.** Real sequences (len 512–640); union over every sliding window
  of size `B`, averaged over windows and docs, per layer.
- **Width axis.** For each of several prefixes, take the target's top-`Wmax`
  next tokens; forward `prefix+candidate` for each; read routing at the
  candidate position; union over random `B`-subsets of the candidates.
- **Baselines.** Analytic `E·(1−(1−k/E)^B)`; empirical union of `B` randomly
  chosen real tokens (shuffled across positions/docs).
- **Routed experts only.** Qwen3.5-MoE also has 1 always-on *shared* expert per
  layer — streamed every cycle regardless of `B`; it is a fixed additive cost,
  not part of the `E`-pool union measured here.

### Where it ran
- **LFM2.5-8B-A1B** — nix1 CPU, bf16 (E=32, k=4, 22 MoE layers of 24; layers 0–1
  dense). Full run: depth-len 640, 12 docs; width Wmax 512, 8 prefixes; B≤512.
- **Qwen3.6-35B-A3B** — halo iGPU (gfx1151) under `hipfire lock run`,
  `HSA_OVERRIDE_GFX_VERSION=11.5.1`, bf16, `device_map="auto"` (E=256, k=8).
  **Loaded the first 20 of 40 layers** — see caveat 1. depth-len 512, 12 docs;
  width Wmax 256, 6 prefixes; B≤256.

## Results — aggregate distinct_experts(B)

### LFM2.5-8B-A1B (E=32, k=4, 22 MoE layers)

| B | depth | %E | width | %E | rand-emp | analytic |
|---|---|---|---|---|---|---|
| 1 | 4.00 | 12% | 4.00 | 12% | 4.00 | 4.00 |
| 2 | 6.07 | 19% | 5.92 | 18% | 6.88 | 7.50 |
| 4 | 9.14 | 29% | 8.28 | 26% | 10.94 | 13.24 |
| 8 | 13.14 | 41% | 11.04 | 34% | 15.94 | 21.00 |
| 16 | 17.50 | 55% | 14.00 | 44% | 20.96 | 28.22 |
| 32 | 21.28 | 67% | 16.90 | 53% | 24.72 | 31.55 |
| 64 | 24.09 | 75% | 19.49 | 61% | 27.19 | 31.99 |
| 128 | 25.99 | 81% | 21.69 | 68% | 28.65 | 32.00 |
| 256 | 27.46 | 86% | 23.47 | 73% | 29.58 | 32.00 |
| 512 | 28.61 | 89% | 24.93 | 78% | 30.17 | 32.00 |

### Qwen3.6-35B-A3B (E=256, k=8, first 20 MoE layers) — the 397B family proxy

| B | depth | %E | width | %E | rand-emp | analytic |
|---|---|---|---|---|---|---|
| 1 | 8.00 | 3% | 8.00 | 3% | 8.00 | 8.00 |
| 2 | 12.89 | 5% | 12.88 | 5% | 15.01 | 15.75 |
| 4 | 20.94 | 8% | 19.75 | 8% | 27.15 | 30.53 |
| 8 | 33.92 | 13% | 29.07 | 11% | 46.58 | 57.42 |
| 16 | 53.34 | 21% | 41.25 | 16% | 75.03 | 101.96 |
| 32 | 79.19 | 31% | 56.37 | 22% | 111.62 | 163.31 |
| 64 | 109.18 | 43% | 73.88 | 29% | 151.35 | 222.44 |
| 128 | 139.18 | 54% | 92.53 | 36% | 185.90 | 251.60 |
| 256 | 167.57 | 65% | 111.30 | 43% | 211.69 | 255.92 |

Ordering at every `B`: **width < depth < random-empirical < analytic-uniform.**
Real context-sharing tokens are far less expert-diverse than random tokens
(discount below), and same-prefix *width* branches are less diverse still.

### Per-layer spread (large — the aggregate hides it)

35B, over the 20 measured MoE layers:

| B | depth min/med/max | width min/med/max |
|---|---|---|
| 8 | 25 / 30 / 52 | 21 / 29 / 40 |
| 16 | 36 / 46 / 88 | 26 / 41 / 63 |
| 32 | 51 / 67 / 132 | 34 / 55 / 93 |
| 64 | 70 / 93 / 179 | 44 / 71 / 126 |
| 128 | 93 / 123 / 220 | 55 / 89 / 157 |

Some layers (max column) route much more diffusely than the median — the
verify-cost bound is set by the *diffuse* layers, not the mean. LFM shows the
same shape (min/med/max at B=32: depth 14/22/28, width 12/17/22).

## Real vs random: the diversity discount

`empirical / analytic-uniform`, i.e. how much *less* of the pool real routing
touches than random selection would:

| B | LFM depth | LFM width | 35B depth | 35B width |
|---|---|---|---|---|
| 4 | 0.69 | 0.63 | 0.69 | 0.65 |
| 8 | 0.63 | 0.53 | 0.59 | 0.51 |
| 16 | 0.62 | 0.50 | 0.52 | 0.40 |
| 32 | 0.67 | 0.54 | 0.48 | 0.35 |
| 64 | 0.75 | 0.61 | 0.49 | 0.33 |
| 128 | 0.81 | 0.68 | 0.55 | 0.37 |
| 256 | 0.86 | 0.73 | 0.65 | 0.43 |

- The discount is transferable at small B (both models ≈0.69 depth / 0.63–0.65
  width at B=4).
- LFM's discount *rebounds* toward 1 past B≈16 — an E=32 **saturation artifact**
  (everything hits the 32-expert ceiling). The 35B, with 256 experts, keeps
  dropping to a real minimum (**width ≈ 0.33 at B=64**) before its own, later
  saturation. A 512-expert target saturates later still, so its discount stays
  low over a wider `B` range than the 35B — making the projection below
  conservative (it over-counts) at large `B`.

## Extrapolation to Qwen3.5-397B-A17B (E=512, k=10)

Applying the 35B (nearest-E) empirical/analytic discounts to the exact 397B
random-uniform curve:

| B | analytic (E=512,k=10) | depth proj | width proj | depth %E | width %E |
|---|---|---|---|---|---|
| 4 | 38.8 | 26.6 | 25.1 | 5% | 5% |
| 8 | 74.7 | 44.1 | 37.8 | 9% | 7% |
| 16 | 138.6 | 72.5 | 56.1 | 14% | 11% |
| 32 | 239.6 | 116.2 | 82.7 | 23% | 16% |
| 64 | 367.1 | 180.2 | 121.9 | 35% | 24% |
| 128 | 471.0 | 260.5 | 173.2 | 51% | 34% |
| 256 | 508.7 | 333.1 | 221.2 | 65% | 43% |

### Free tree-width budget for the 397B

Reading "cheap" as *verify streams ≤ ¼ of the routed experts per cycle* (≤128 of
512):

- **Pure width** stays ≤25%E up to **B ≈ 64 sibling branches**; ≤34%E at B=128.
- **Pure depth** stays ≤25%E only to **B ≈ 32 positions**; hits 51%E at B=128.
- A realistic combined tree (e.g. depth 8 × width 8 = 64 nodes) lands **between**
  the two curves — order **24–35% of experts** touched per verify.

So on the 397B you can afford trees of **~64 nodes touching ≈¼–⅓ of the expert
pool per cycle**, and the width dimension is the one to spend the budget on
(cheaper per node, and it gets cheaper as `E` grows). This is the opposite of the
H200 papers' 256–512 compute-optimum: here the bound is expert-set growth, and it
bites earlier — but wide-over-deep is the right shape.

### What this means for the design (§9/§11)

- Wide trees are **worth building** and are cheaper than the depth budget would
  suggest — but they are **not free**; the BASTION cost model must price width at
  a real (sub-linear, ~0.6–0.7× depth) marginal-expert rate, not zero.
- The per-cycle streamed set is **the routed union + the always-on shared expert
  per layer + the diffuse-layer tail** (per-layer max, not mean). Size the
  streaming engine to the tail.
- Risk in §11 ("if the expert set grows fast, wide trees stream most of the
  397B") is **not realized**: at B=64, width touches ~24% of experts, not most.

## Caveats / uncertainty

1. **35B used the first 20 of 40 layers** (halo is a 128 GB unified-memory APU;
   the full 70 GB bf16 model + framework thrashed swap — `device_map="cuda"`
   double-buffers CPU+GTT on an APU, and even lean `device_map="auto"` streaming
   peaked at the RAM limit around 65% load). Routing for layers 0–19 is *exact*
   (each layer's router depends only on prior layers). The aggregate is over the
   first half; per-layer spread is wide, so the aggregate carries real
   uncertainty. Later layers were not measured.
2. **The extrapolation crosses E 32→256→512, k 4→8→10, and different
   layer-counts / hybrid (GatedDeltaNet) structure.** The discount-transfer
   assumption is the main source of error; it is well-supported at small B, and
   conservative (over-counting) at large B. Treat the 397B numbers as a shape,
   not a promise — run the measurement on the real 397B before committing a tree
   width (medusa was unreachable this session).
3. **Width proxy.** The width axis uses the *target's* own top-W next tokens as
   siblings. A real tree's siblings are the *drafter's* proposals (correlated but
   not identical); and real branches occur at multiple depths, not one. The
   full-stack DFlash→DDTree→verify run (the task's bonus) was **not** done — it
   needs the tree-mask masked-SDPA verify path that does not yet exist.

## Reproduce

```bash
# corpus (committed): benchmarks/corpora/wiki_moe_corpus.jsonl
# LFM on a CPU/GPU box:
python benchmarks/scripts/distinct_experts_moe.py \
  --model /srv/huggingface/models--LiquidAI--LFM2.5-8B-A1B/snapshots/<hash> \
  --device cpu --dtype bfloat16 \
  --corpus-file benchmarks/corpora/wiki_moe_corpus.jsonl \
  --out lfm.json --depth-docs 12 --depth-len 640 \
  --width-prefixes 8 --width-prefix-len 40 --width-max 512 --width-batch 64 --max-B 512

# Qwen3.6-35B on halo iGPU (gfx1151), under the GPU lock:
HSA_OVERRIDE_GFX_VERSION=11.5.1 hipfire lock run de35b -- python \
  benchmarks/scripts/distinct_experts_moe.py \
  --model /srv/huggingface/models--Qwen--Qwen3.6-35B-A3B/snapshots/<hash> \
  --device cuda --dtype bfloat16 --max-layers 20 \
  --corpus-file benchmarks/corpora/wiki_moe_corpus.jsonl \
  --out q35b.json --depth-docs 12 --depth-len 512 \
  --width-prefixes 6 --width-prefix-len 40 --width-max 256 --width-batch 64 --max-B 256

# render tables:
python benchmarks/scripts/distinct_experts_render.py "LFM2.5-8B-A1B=lfm.json" "Qwen3.6-35B-A3B=q35b.json"
```

Raw result JSONs (per-layer included):
`benchmarks/results/distinct-experts-lfm2.5-8b-a1b-20260721.json`,
`benchmarks/results/distinct-experts-qwen3.6-35b-a3b-20260721.json`.

Notes on the toolchain (nix1/halo): HF `transformers` here trips over a broken
`torchvision` device-plugin stub (`benchmarks/scripts/tv_shim.py` shims it); on
the halo APU use `device_map="auto"` (never a bare `"cuda"`) and expect the full
40-layer bf16 model not to fit alongside a co-tenant.
