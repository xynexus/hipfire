In Hipfire, **Opus is a family of neural-network weight quantization codecs**, unrelated to the audio codec. It combines a fixed G256 transform, symmetric integer quantization, activation-aware calibration, and optional sparse higher-precision corrections.

## Core design

For a matrix (W), each output row is divided along (K) into groups of 256 weights. Each group is encoded roughly as:

[
w_g \xrightarrow{\text{signed FWHT}_{256}} \tilde w_g
\xrightarrow{\text{scale search}} q_g
]

The quantizer:

1. Zero-pads the final group to 256 elements.
2. Applies a fixed signed FWHT-256 rotation.
3. Searches for a symmetric clipping scale.
4. Quantizes the rotated values.
5. Stores an f16 scale followed by packed integer codes.

The activation path applies the corresponding transform at runtime, so computation remains in the rotated basis. The low-level codec already performs per-group clip search. In the artifact naming, `+` denotes additional activation-aware calibration such as AWQ/SmoothQuant, while `++` adds Hessian/LDLQ error feedback. Those suffixes change the offline values and optional sidecars, not the fundamental runtime encoding.

The main formats are:

| Format                 | Block contents per 256 weights   | Effective storage | Runtime intent            |
| ---------------------- | -------------------------------- | ----------------: | ------------------------- |
| `Oq4G256`, qt 34       | f16 scale + 128 signed nibbles   |        4.0625 b/w | W4A4                      |
| `OqPlusG256`, qt 33    | same fundamental W4 values       |        4.0625 b/w | W4A8 via int8 expansion   |
| `Oq8G256`, qt 35       | f16 scale + 256 signed int8      |        8.0625 b/w | W8A8                      |
| `OqPlusCompact`, qt 36 | W4 base + sparse W8 replacements |          variable | Mixed W4/W8, usually W4A8 |

There are also W3, W2 and W6 Opus IDs, plus row-padded and non-rotated variants for XDNA and DFLASH, although not every one has complete production runtime coverage.

## The interesting part: `oq4.25++`

This is not “some tensors at four bits and some at eight bits.” Precision mixing happens **inside every 256-weight group**.

For each transformed group, the encoder first calculates a W4 representation:

[
q^{(4)}_i =
\operatorname{clip}\left(
\operatorname{round}\frac{\tilde w_i}{s},
-7,7
\right)
]

It also calculates the hypothetical W8 representation:

[
q^{(8)}_i =
\operatorname{clip}\left(
\operatorname{round}\frac{\tilde w_i}{s},
-127,127
\right)
]

Each position receives an upgrade score:

[
G_i =
\left(\tilde w_i-sq^{(4)}_i\right)^2
------------------------------------

\left(\tilde w_i-sq^{(8)}_i\right)^2
]

So the codec promotes the weights for which int8 removes the most quantization error. It does **not** simply select the largest magnitudes. The implementation then refits the common scale, recomputes the best upgrade positions, and refits once more.

The block layout is:

```text
[f16 scale]
[128 bytes containing all 256 W4 values]
[N × {u8 index, i8 replacement}]
```

At a nominal 1% upgrade fraction:

[
N=\operatorname{round}(0.01 \times 256)=3
]

Therefore:

[
130 + 2(3)=136\text{ bytes}
]

and:

[
\frac{136 \times 8}{256}=4.25\text{ bits/weight}
]

Every position retains a valid W4 value. The three selected positions receive sparse int8 replacements, making the stream robust even when a runtime ignores the overlay.

Conceptually, mixed Opus evaluates:

[
W_{\text{mixed}} = W_4 + \Delta_{\text{sparse}}
]

and therefore:

[
XW_{\text{mixed}}
=================

XW_4 + X\Delta_{\text{sparse}}
]

That decomposition is explicit in the XDNA implementation.

## Activation and output reconstruction

The XDNA implementation makes the runtime contract particularly easy to see.

For every activation row and G256 block it:

1. Optionally divides the activation by the tensor’s AWQ scale.
2. Applies the matched signed FWHT.
3. Finds the transformed activation absmax.
4. Dynamically quantizes the activation to signed int8.
5. Executes integer matrix multiplication.
6. Reconstructs floating output using both activation and weight scales.

The reconstructed output is effectively:

[
y_{m,n}
=======

\sum_g
\left(
\sum_{k \in g}
q^x_{m,k}q^w_{n,k}
\right)
s^x_{m,g}s^w_{n,g}
]

For a compact mixed block, the decoder sign-extends the nibbles, checks that sparse indices are unique, and stores each correction as:

[
\Delta_i = q^{(8)}_i-q^{(4)}_i
]

The NPU can either execute the W4 base plus sparse residuals or expand the block once into dense int8 resident weights and reuse its ordinary W8 schedule.

The NPU runtime has several increasingly fused execution levels:

* Per-G256 W4/W8 plus sparse-residual kernels
* Full-(K) projection kernels
* Whole-array kernels
* Whole-array kernels with scaling inside the device path
* Staged full-(K) resident execution

The executor chooses the best available representation during matrix packing, then falls back toward groupwise execution when a fused cache is unavailable.

## Why the transform helps

The FWHT is doing more than making the distribution vaguely Gaussian. It spreads isolated large channels across the 256-dimensional group, which makes:

* A single symmetric scale more representative
* Activation energy more uniform across positions
* Sparse precision upgrades selectable largely from weight-side error
* W4 and W8 share the same zero-point-free arithmetic
* Integer dot products simpler than affine MQ-style formats

This is also why the upgrade score can be based on the reduction in weight reconstruction error rather than maintaining a large per-channel activation-importance structure.

Opus therefore differs fundamentally from Magnum/MQ:

* Magnum uses an affine W4 representation with a zero/min term.
* Opus uses signed symmetric integers around zero.
* Opus’s compact mixed stream is a W4 base plus sparse signed replacement values.
* The symmetric representation maps much more naturally onto signed integer matrix instructions.

## Measured quality

A recorded Qwen3.5-0.8B experiment used a Wikitext2 slice at context 2048, with fp32 KV for every candidate to isolate weight quantization:

| Weights   |         Size |    PPL |  KLD/token vs BF16 |
| --------- | -----------: | -----: | -----------------: |
| BF16      |      1519 MB | 24.029 |                  0 |
| `oq8++`   |       787 MB | 24.029 | (3.0\times10^{-4}) |
| `oq4.5++` |       566 MB | 24.523 |              0.048 |
| `mq4+`    | not recorded | 26.462 |              0.082 |

In that specific test, `oq8++` was effectively lossless, while the mixed Opus format reduced KLD by about 41% relative to `mq4+`. These are model-and-corpus-specific results rather than universal ratios.

## Where implementation reality diverges from the clean design

The codec itself is well-defined. Most of the remaining sharp edges are in **runtime routing**.

The ordinary expanded qt-36 GPU path reconstructs dense int8 weights and can use existing fused OQ8 kernels. The optional compact-resident path leaves the W4-plus-overlay blocks compressed in memory. On Qwen3.5, those two representations currently take different execution routes:

```text
Expanded:
  fused_qkvza_oq8_gemv
  fused_gate_up_oq8_gemv
  gemv_oq8_grouped

Compact resident:
  gemm_oq_compact_grouped_wmma
  separate unfused operations
```

Thus the mathematical weights are the same, but accumulation order and fusion differ. The compact path was approximately 31% slower in the recorded probe, despite reducing calculated Opus tensor allocation by roughly 48%. More importantly, the numerical difference can flip argmax when logits are close. It is not merely a harmless last-bit discrepancy.

The current master commit, merged on August 6, 2026, adds a dedicated compact-residency drift gate. It records separate expected hashes for expanded and compact operation instead of pretending they are currently identical.

The proper fix is one of:

* Native compact versions of the fused QKVZA, gate/up and grouped GEMV kernels
* An expand-on-use mechanism at fused call sites
* A unified fused kernel that reads compact blocks directly

Until then, compact residency remains opt-in.

True W4A4 is in a similar state. The int4 activation quantizer and W4A4 kernels exist and pass isolated parity checks, and an environment-gated full-model path produced coherent text. But several serving and fused paths still route OQ4 through W4A16 or W4A8/F16 alternatives, or reject it from fused execution. Consequently the current support matrix correctly describes OQ4 and the other Opus variants as opt-in or partially admitted rather than universally production-complete.

## Model-family coverage

The August 5 plan began with 18 of 20 model-family variants baselined across:

```text
oq4
oq4+
oq4++
oq4.25++
oq8
oq8+
oq8++
```

Its later phase update says `deepseek4_compressed` was completed, leaving `deepseek4_mtp` as the explicit architectural blocker. MTP is not just missing a quantizer branch: packaged MTP tensors, loader integration and the decision to preserve training precision for acceptance-rate quality all remain entangled.

The same plan records a separate current MiniMax regression, where its seven Opus cells fail on master. So the honest state is broad format correctness, with several family-specific admission and routing problems still active.

## My reading of the design

The most distinctive idea is not the FWHT or symmetric W4 alone. It is that `oq4.25++` treats precision as a **sparse correction field**:

[
\text{compact model}
====================

\text{cheap dense approximation}
+
\text{small exacting correction}
]

That gives one common scale, one W8A8 compute contract, graceful W4 fallback, and a directly tunable storage-quality knob through (N_{\text{out}}). It is much cleaner than arbitrary mixed-bit blocks.

The codec is already coherent. The unfinished work is the plumbing around it: making compact storage remain compact through every fused projection, ensuring every architecture admits the correct dtype, and proving that test oracles exercise the kernels the actual model dispatches. The repository’s recent debugging history demonstrates that the largest hazards have been route mismatches, not quantization mathematics.
















Short answer: yes… and the interesting ones are *not* in the obvious “better clipsearch / better rotation” bucket. The OpusQuant you’ve built is already pretty sharp; most remaining gains are about **how the correction layer behaves as a signal**, not just how you pick it.

Let me throw a few ideas at you that feel *on-theme* with what you’re doing.

---

## 1) Turn the correction into a tiny learned codebook (not just raw int8)

Right now your overlay is:
[
\Delta_i = q^{(8)}_i - q^{(4)}_i
]

That’s elegant, but also slightly… wasteful. You’re spending a full int8 per outlier even though:

* The dynamic range is heavily structured
* Most corrections fall into a small set of patterns

### Twist:

Instead of storing raw `i8`, store:

* a **4–6 bit index into a per-layer (or per-tensor) codebook**
* plus the position

So the block becomes:

```text
[f16 scale]
[W4 bulk]
[N × (u8 index, u4 code)]
```

You’ve effectively turned the residual into a **vector-quantized error field**.

Why this works:

* FWHT already flattens distribution → residuals cluster
* Gain-based selection already isolates “important error modes”
* You compress entropy *and* preserve structure

This pushes you toward ~4.1–4.2 b/w without losing fidelity.

---

## 2) Don’t store Δ — store **directional correction**

Right now Δ is scalar per weight. But after FWHT, correlation still exists across nearby indices.

Instead, think:

[
\Delta_{i} \approx \alpha \cdot v_{k}
]

Where:

* (v_k) is a small basis vector (size maybe 4–8)
* (\alpha) is scalar per selected index group

So instead of N independent corrections, you store:

* a few **basis IDs**
* a few **coefficients**

That turns your sparse correction into a **low-rank residual patch**.

This is basically:

> “LoRA, but inside a 256-weight block”

And it fits your design philosophy weirdly well.

---

## 3) Joint optimize scale + mask (you’re *almost* there already)

You already do:

* initial scale
* pick indices
* refit scale
* re-pick indices

But your objective is still:
[
\sum_i (e_i^2)
]

The missing piece:

### Make the mask part of the optimization

Solve:
[
\min_{s, S} \sum_i |w_i - \hat w_i(s, S)|^2
]

Where (S) is the selected outlier set.

Right now you approximate it greedily via gain. That’s good, but:

👉 You can do **one-step lookahead**:

* evaluate scale impact *after* masking
* not just per-index gain at fixed scale

This matters because:

* scale shrinkage benefits W4 bulk
* but hurts promoted values

You’re balancing two competing regimes… but only partially.

---

## 4) Use activation-aware gain, not just weight error

You hinted at this already in your comments, but it’s worth pushing further.

Current gain:
[
G_i = e4_i^2 - e8_i^2
]

Better:
[
G_i = \mathbb{E}_x \left[ (x_i \cdot e4_i)^2 - (x_i \cdot e8_i)^2 \right]
]

Which simplifies to:
[
G_i \propto \sigma_x^2(i) \cdot (e4_i^2 - e8_i^2)
]

Where (\sigma_x^2(i)) is activation variance per rotated dimension.

FWHT helps flatten, but:

* real models still have structure
* especially post-attention projections

This turns OpusQuant from:

> “best weight reconstruction”

into:

> “best output reconstruction”

---

## 5) Bitplane-aware packing for the overlay

Right now your overlay is:

```text
(index, value)
```

But your W4 base is nibble-packed, and your kernels are bitplane-friendly.

So… go full symmetry:

* Store overlay as **bitplanes aligned with W4 layout**
* Or even reuse the same unpack path

Why it matters:

* lets you fuse decode paths
* avoids divergent memory access for sparse indices
* improves SIMD/WMMa utilization

This directly attacks your current pain point:

> compact vs expanded path divergence

---

## 6) “Soft promotion” instead of binary W4/W8

Right now:

* either W4 or W8

But you could do:

* W4 baseline
* * **1 extra bitplane** for selected indices

So promoted weights become:

* 5-bit or 6-bit effective precision

That gives you:

* smoother tradeoff
* less overkill than full int8

And crucially:

* smaller overlay payload

Think of it as:

> progressive refinement rather than discrete jump

---

## 7) Cross-group promotion pooling

Right now selection is per-group (256).

But in practice:

* some groups are “easy”
* some are “disaster zones”

Instead:

* allocate a **budget per row or per tensor**
* let groups borrow from each other

So:
[
\sum_g N_g = \text{fixed}
]

Instead of:
[
N_g = \text{constant}
]

That’s basically:

> entropy coding across groups

And it improves worst-case error significantly.

---

## 8) Make compact path numerically identical (big one)

Not glamorous, but honestly the most impactful:

Right now:

* expanded → fused kernels
* compact → unfused path
* → different accumulation → different logits

Your own test shows:

> differences can flip argmax when logits are close

This is *the* blocker to treating OpusQuant as a first-class format.

Two options:

* fused kernels that read compact directly
* or guaranteed expand-at-boundary

Until that’s fixed, everything else is slightly undermined.

---

## 9) Use OpusQuant as a training target (this one’s spicy)

You’re treating it as post-training quantization.

But imagine:

* training a model to **minimize OpusQuant reconstruction error directly**

Loss:
[
\mathcal{L} = \text{task loss} + \lambda \cdot |W - \text{OpusQuant}(W)|^2
]

That gives you:

* weights that *want* to live in your quant space
* fewer extreme outliers
* lower overlay usage

Basically:

> you stop fighting the distribution

---

## My honest take

You’ve already nailed the core idea:

> “Dense cheap approximation + sparse high-precision correction”

That’s the right abstraction. Everything above is just different ways of making the correction:

* more compressible
* more aligned with compute
* more aware of activations
* more globally optimal

If I had to rank impact:

1. **Kernel parity (compact = expanded)**
2. **Activation-aware gain**
3. **Joint scale+mask optimization**
4. **Codebook residuals**
5. **Cross-group budgeting**

Everything else is bonus power-ups.

---

> “Can I make the model *want* to be Opus-shaped so W4A4 stops feeling like a constraint and starts feeling like home?”

Short answer: yes. And if you do it right, OpusQuant stops being a clever compression trick and becomes the *native parameterization*.

Let’s walk through a clean mental model.

---

## 1) What W4A4 is really fighting

Right now W4A4 fails not because 4 bits is too small, but because:

* weights have heavy-tailed distributions
* activations have uneven energy across channels
* quantization error is *structured*, not noise

Your FWHT already fixes a big chunk of this by flattening energy. That’s huge.

But training is still optimizing:
[
\min \mathcal{L}_{task}(W)
]

while inference uses:
[
\hat W = \text{OpusQuant}(W)
]

So the model is constantly drifting *away* from the manifold you actually execute on.

---

## 2) Flip the objective: train **inside the codec**

Instead of:

```text
train → quantize → run
```

You do:

```text
train → quantize → run → backprop through quant
```

Formally:

[
\mathcal{L} =
\mathcal{L}_{task}(\hat W, \hat x)
\quad\text{where}\quad
\hat W = Q(W),;
\hat x = Q(x)
]

And (Q(\cdot)) is your full Opus pipeline:

* FWHT
* scale
* rounding
* (optional OpusQuant overlay)

---

## 3) The key trick: straight-through but *structured*

Naive STE would just do:
[
\frac{\partial \hat W}{\partial W} \approx 1
]

That works, but it wastes your structure.

You actually want something more *aligned with Opus geometry*:

### Replace gradient with projection:

[
\nabla_W \approx \text{FWHT}^{-1}(\text{clip}(\nabla_{\tilde W}))
]

So:

1. rotate gradients into FWHT space
2. clip to quant range
3. rotate back

Now the model learns:

> “Don’t put energy where quantization will kill it.”

---

## 4) Train the scale as a parameter (big win)

Right now scale comes from clipsearch.

During training:

[
s_g = \text{learned parameter per group}
]

or slightly constrained:

[
s_g = \alpha_g \cdot \text{running absmax}
]

Then you optimize:

[
q_i = \text{round}\left(\frac{\tilde w_i}{s_g}\right)
]

The network learns:

* tighter distributions
* fewer saturations
* better use of 4-bit dynamic range

This alone usually gives a *huge* W4 boost.

---

## 5) Make activations obey W4A4 too

Weights alone aren’t enough. W4A4 dies if activations spike.

So you inject:

[
\hat x = \text{round}\left(\frac{x}{s_x}\right)
]

during training with:

* per-channel or per-group activation scales
* optional EMA tracking

And add a small penalty:

[
\lambda \cdot \mathbb{E}[|x| > 7 s_x]
]

Now activations *learn* to stay inside 4-bit range.

---

## 6) OpusQuant becomes a **learned residual budget**

Here’s where it gets interesting.

Instead of selecting outliers post-hoc, define:

* a binary mask (m_i \in {0,1})
* constraint: (\sum m_i \le N)

Then:

[
\hat w_i =
\begin{cases}
\text{int8}(w_i) & m_i = 1 \
\text{int4}(w_i) & m_i = 0
\end{cases}
]

Training objective includes:

[
\mathcal{L}_{budget} = \lambda \left(\sum m_i - N\right)^2
]

You can relax (m_i) with sigmoid / Gumbel.

Now the model learns:

* which weights deserve promotion
* not just which look bad locally

It becomes a **resource allocation problem learned by the network**.

---

## 7) Better: continuous promotion instead of binary

Even cleaner:

[
\hat w_i =
\text{round}\left(\frac{\tilde w_i}{s}\right)
+
\alpha_i \cdot \delta_i
]

Where:

* (\alpha_i \in [0,1]) is learned importance
* (\delta_i) is higher-precision residual

Then you sparsify (\alpha) with L1 or top-k.

This avoids brittle discrete jumps during training.

---

## 8) Regularize the distribution directly

You can push the model toward quant-friendly structure:

### Flatten rotated weights:

[
\mathcal{L}_{flat} = \text{Var}(|\tilde w|)
]

### Penalize outliers:

[
\mathcal{L}_{tail} = \mathbb{E}[|\tilde w|^p],\quad p>2
]

### Encourage bin alignment:

[
\mathcal{L}_{snap} = | \tilde w - s \cdot \text{round}(\tilde w / s) |^2
]

That last one is basically:

> “snap weights to quant grid during training”

---

## 9) What this buys you (the real prize)

If you do all of the above:

### W4A4 stops being lossy

Because:

* weights are already near 4-bit bins
* activations are bounded
* residuals are minimal

So:
[
W \approx \hat W_{4bit}
]

And OpusQuant becomes:

* tiny correction
* or even unnecessary in many layers

---

## 10) Practical training recipe

If I were implementing this in your stack:

1. Start from FP16 checkpoint
2. Enable:

   * FWHT in forward
   * fake quant (W4 + A4)
3. Use STE + FWHT-aware gradient
4. Learn per-group weight scales
5. Track per-channel activation scales
6. Add mild clipping penalties
7. After convergence:

   * freeze
   * optionally train OpusQuant mask for last 1–2 epochs

---

## 11) The deeper insight

Right now OpusQuant is:

> “Fix the model after it misbehaves”

Training flips it into:

> “Make the model behave so it doesn’t need fixing”

And that’s what unlocks W4A4 properly.

---

If you want, next step we can design a *minimal training loop* that plugs directly into Hipfire’s pipeline, including how to reuse your existing FWHT and codec code without rewriting everything.



You’ve hit the exact pressure point where math elegance meets silicon reality.

Right now the situation is basically:

* W4A4 = 🚀 fast, clean, bandwidth-light
* W8A8 = 🐢 slower, but tolerant of outliers
* PUDIC = 🧠 clever, but forces you into awkward execution paths

And the core question hiding underneath what you said is:

> “Can I keep the **W4A4 datapath hot** and still handle outliers without falling off a cliff?”

Let’s walk through the viable strategies, from “pragmatic” to “slightly evil genius”.

---

## 1) The trap: switching kernels per block

If you literally do:

```text
if has_outliers:
    run W8A8 kernel
else:
    run W4A4 kernel
```

that *will* nuke performance:

* warp divergence
* pipeline flushes
* memory layout mismatch
* scheduler thrash

So yeah… don’t do that. That path is a tar pit.

---

## 2) The correct mental model

Think of your matmul as:

[
Y = X W = X(W_4 + \Delta)
= XW_4 + X\Delta
]

Where:

* (W_4) = dense int4 weights
* (\Delta) = sparse corrections

The trick is:
👉 **don’t change the main kernel at all**

Instead:

* run W4A4 for everything
* *add a second, tiny correction pass*

---

## 3) Two-pass execution (this is the baseline you want)

### Pass 1 (fast path)

* pure W4A4 WMMA
* no branching
* full throughput

### Pass 2 (correction)

For each promoted weight:

[
y += x_k \cdot \Delta_{k}
]

Implementation options:

* small GEMV per row
* warp-level sparse accumulate
* even scalar FMA loop (it’s that small)

Because:

* N_out ~ 2–4 per 256
* that’s like ~1–2% density

So cost ≈ negligible.

👉 This keeps:

* **100% W4A4 utilization**
* no kernel switching
* no divergence

---

## 4) Even better: fuse correction into epilogue

Instead of a second kernel:

* keep a tiny buffer of `(index, delta)` per block
* after WMMA accumulate, apply:

[
acc += x[index] * delta
]

This happens:

* in registers
* inside the same threadblock

No extra memory roundtrip.

This is *chef’s kiss* level efficiency.

---

## 5) The real bottleneck: activation format

You said:

> “we use W4A8 and promote weights to W8 anyway”

That’s actually the hidden tax.

Because:

* bi4 WMMA wants **int4 activations**
* bi8 MMA wants **int8 activations**

If you’re feeding int8 activations:

* you’re already paying bandwidth + conversion cost
* and losing half the speed advantage

---

## 6) So the real unlock is:

### 👉 commit to true W4A4

Meaning:

* activations quantized to int4
* weights int4
* everything aligned to bi4 WMMA

Then:

* PUDIC corrections operate in int8 or fp16 **on top**

---

## 7) How to handle mixed precision without kernel switch

You’ve got three viable patterns:

---

### A) “Shadow accumulation” (cleanest)

Keep two accumulators:

```text
acc4 = W4A4 result
accΔ = correction result
```

Final:
[
Y = acc4 + accΔ
]

* acc4 via WMMA
* accΔ via sparse ops

No mixing inside the tensor core path.

---

### B) “Upcast on the fly” (what you’re half doing now)

* expand W4 → W8
* run W8A8

Problem:

* doubles bandwidth
* kills your main advantage

So… yeah, you already know this is suboptimal.

---

### C) “Bitplane augmentation” (spicy option)

Instead of storing full int8 corrections:

* store extra bitplanes
* reconstruct higher precision in registers

So:

[
q = q_{4bit} + (b_4 << 4) + (b_5 << 5) + ...
]

Then:

* still run W4A4 core
* add extra bits via cheap logic

This avoids:

* switching to bi8 MMA
* full int8 path

This is very hardware-friendly.

---

## 8) Warp-level scheduling trick (underused)

You can assign:

* most lanes → W4A4 WMMA
* a few lanes → correction work

Because correction density is tiny.

So instead of:

* separate kernel

you do:

* **intra-warp specialization**

Modern schedulers handle this surprisingly well if structured cleanly.

---

## 9) Training ties directly into this

Here’s the payoff from your earlier question:

If you train properly:

* W4A4 handles ~99% of energy
* PUDIC handles the last 1%

So:

* correction pass becomes almost free
* sometimes removable entirely

Meaning:

> you asymptotically approach pure W4A4 speed

---

## 10) The “endgame” architecture

If you push this fully, you end up with:

```text
Main path:
    W4A4 WMMA (dominant compute)

Side channel:
    Sparse correction (tiny, irregular)

Optional:
    occasional W8 fallback (rare, maybe layer-level)
```

No kernel switching per block.
No full upcasting.
No bandwidth explosion.

---

## 11) One subtle but important improvement

Right now your correction is:

[
\Delta_i = q8_i - q4_i
]

But for compute efficiency, you want:

[
\Delta_i' = (q8_i - q4_i) \cdot s
]

Pre-scaled.

So correction becomes:

[
y += x_i \cdot \Delta_i'
]

No extra scale multiply in the correction path.

That matters when you’re doing thousands of tiny ops.

---

## My blunt take

You do **not** want to switch kernels.
You want:

> W4A4 as the highway
> PUDIC as a few motorcycles weaving through traffic

If you instead turn the highway into a mix of trucks and bikes, everything slows down.

---
