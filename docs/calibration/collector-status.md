# Native Tier-1 calibration collector — status & roadmap

Status as of 2026-07-22. The resident collector and family-neutral layer-stream
engine are built and mechanism-verified. The full Qwen3.5-397B teacher artifact
and its `oq4.25++` second pass are complete; the controlled expert sweeps,
second-family resident parity, and final quality/channel admission evidence
remain pending.

## Update (2026-07-25) — stratum lens + a multilingual calibration corpus

### The stratum lens (semantic routing, where token identity says nothing)

Token-identity profiling is the right lens where routing is lexical, but ZAYA's
middle third is language-*universal* (max 4.2% CJK concentration in blocks 16-23
versus 100% in block 26) so a token histogram there is uninformative. Matching
[Multilingual Routing in MoE](https://arxiv.org/pdf/2510.04694), which finds
language-specific routing in early and late layers with middle layers acting as
language-universal machinery.

`LayerRouterStats.stratum_counts` therefore counts, per expert, the *label of the
sample* each routed token came from. `CalibrationSample::stratum` already existed
and was hashed into the sample fingerprint but nothing ever set it beyond the
literal `"plain-text"`; a `.jsonl` corpus now populates it. Both fields ride a
single [`RoutedRowContext`] so the grouped-MoE dispatch seam — which sees neither
token nor stratum — keeps passing `unknown()`.

The report scores **enrichment** (expert's stratum share ÷ the layer's stratum
share), not raw share: an expert taking 40% code from a 40%-code corpus has
learned nothing about code. A single-label corpus sets
`stratum_profile_present: false` rather than reporting a tautological 1.0x.

Labelled corpora are JSONL, one record per sample, text from `text`/`content`/
`messages` (chat-style arrays are flattened over `content`) and label from
`stratum`/`label`/`source`/`domain`/`language`. That makes
[`Moe-lab/DBES`](https://huggingface.co/datasets/Moe-lab/DBES) (7 domains) and
`allenai/tulu-3-sft-mixture` usable as-is. A record is truncated to the sample
context, never split across samples — a sample straddling two labels would make
the profile a lie.

### The starved set is not stable under reweighting — re-measure lift every time

`stratum_guidance` reports each stratum's share of *starved-expert* traffic over
its share of all traffic. Measured on `calib-multi-labelled.jsonl` (32,138
tokens/layer, 5 strata):

| stratum | starved share | corpus share | lift |
|---------|---------------|--------------|------|
| math | 50.4% | 9.6% | **5.27x** |
| code | 18.0% | 19.1% | 0.94x |
| english | 25.8% | 41.4% | 0.62x |
| japanese | 4.1% | 14.0% | 0.29x |
| chinese | 1.8% | 15.9% | 0.11x |

Math holds 4.50-6.25x lift in *every* depth band (early 0-9, mid 10-19, mid
20-29, late 30-39), so it is not a shallow-layer artifact. The conclusion
**reverses the CJK finding above**: CJK starvation was real for the English-only
corpus, but once CJK reaches ~17% of tokens it over-serves already-covered
experts (lift 0.04-0.24x) and the binding constraint becomes math. Pushing CJK
further actively hurts — the 44%-CJK overshoot build raised mean Gini to 0.535
against 0.323 for the corrected mix.

Tail health with the labelled corpus: **never-routed 0** (against 29 on
English-only), starved-below-400 21/640, absolute counts worst 3 / 5th pct 491 /
10th pct 692 / median 1684. Scaled to 262k tokens that puts the whole
distribution above the 2048 floor bar a handful of points.

The operational lesson: expert-targeted calibration is a **loop**, not a one-shot
fix. Reweighting the corpus moves the starved set, so lift must be re-measured
after every change; a fixed prior about which material the tail needs will be
wrong within one iteration.

### Building a calibration corpus: two traps

Both bit during construction of `benchmarks/calib/calib-multi-8m.txt` (English +
zh + ja + code + math from
[`eaddario/imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration),
`text_{en,cn,jp}_large` + `code_large` + `math_large`, cached under
`/srv/huggingface/datasets--eaddario--imatrix-calibration/manual`):

1. **`load_corpus_samples` consumes the corpus in order and stops** once it has
   `--sequences` samples. A concatenated corpus is therefore sampled only from its
   head: a first attempt with 100 KB interleave rounds put 64 KB of English first,
   and an 8192-token pass measured **0.00% CJK** from a corpus that is 6.3% CJK.
   Sources must be interleaved so that *every prefix* carries the target mix —
   greedy largest-deficit-first over documents does this exactly.
2. **A sample cannot span a paragraph.** `pending` resets per `\n\n` paragraph, so
   any document shorter than `--context` tokens yields a short sample. Four-line
   documents silently produced 311-token samples against a 1024-token context
   (2489 routed tokens per layer instead of 8192). Documents must exceed the
   sample context — size them in *chars* via each source's chars/token (~4.0
   ASCII, ~1.2 CJK).

Getting both right moved the measured token mix from cjk 0.62% / digit 6.18% /
punct 12.79% / word 75.76% to **cjk 8.36% / digit 10.13% / punct 26.87% / word
48.94%**. A labelled twin lives at `benchmarks/calib/calib-multi-labelled.jsonl`
(2713 records, english 38% / code 21% / chinese 15.5% / japanese 15.5% / math
10%), trimmed to the prefix over which its stated mix holds within 2 points.

Caveat worth carrying: `eaddario/imatrix-calibration` covers 18 languages but
**not Korean**, so hangul experts stay starved by this mix.

## Update (2026-07-25) — MoE router specialisation profiler

`artifact moe-router-profile` answers *what* an expert specialises in, not just
how much load it takes. Router load alone cannot distinguish a weak expert from a
specialist the corpus never triggers, and only the second case is fixed by corpus
material rather than a quantization policy.

Producer: `LayerRouterStats` gained `token_counts` — a per-expert routed-token
histogram — populated when the family adapter passes the corpus token to
`record_router_selection`. It is truncated to the top
`TOKEN_PROFILE_KEEP` (256) ids per expert at snapshot time with
`token_profile_dropped` recording the discarded distinct-id count, so a report
never implies it saw the whole tail. The ZAYA streamed adapter supplies tokens;
the grouped-MoE dispatch seam (qwen3.5's batched routed capture) does not see
corpus tokens, so those layers report `token_profile: absent` instead of an empty
profile. Threading tokens through that callback is a follow-on.

Consumer: `hipfire-coexistence artifact moe-router-profile --input <calib.hfq>
--tokenizer <hf-dir|model.hfq> [--layer N] [--top N] [--min-activations N]
[--json]` decodes the retained ids and buckets them into coarse, tokenizer-neutral
character classes (word / digit / punct / whitespace / CJK / other-non-ascii /
byte-fallback), then reports per expert: load share, mean±σ of the winning gate,
top-10 concentration, dominant classes, and top decoded tokens — plus per-layer
imbalance and Gini, and a starved-expert summary against `--min-activations`.

### Finding: ZAYA1-8B strict coverage fails because the corpus is English-only

The 262144-token `benchmarks/calib/calib-5m.txt` run aborted at block 7 under the
default `strict` policy (4 of 32 capture points below the 2048-activation floor).
An 8192-token profiling pass explains why. Comparing starved (<200 of 8192) with
well-covered capture points across all 40 blocks:

| class | starved | well-covered | enrichment |
|-------|---------|--------------|-----------|
| cjk | 5.66% | 0.44% | **13.0x** |
| whitespace | 11.93% | 5.46% | 2.2x |
| digit | 7.12% | 6.91% | 1.0x |
| word | 45.24% | 72.26% | 0.63x |

The corpus itself is word 76% / punct 13% / digit 6% / whitespace 5% / **cjk 0.6%**.
The mean winning gate is *identical* for starved and well-covered experts (0.458
vs 0.461), so starvation is a coverage problem, not weak experts — and 17 starved
experts are >10% CJK, several of them 83-100% CJK in the deep blocks with the
highest gates in their layer (e.g. block 28 expert 3: 40 tokens, cjk 90%, gate
0.666 — the most confident expert in that block). Their tokens are `ヴァ ル キュ 戦`,
i.e. "Valkyria" in katakana: the corpus is WikiText-style English prose whose only
CJK is one article's incidental Japanese, so ZAYA's CJK experts can never reach
the floor. 29 experts were never routed at all.

Block 28 shows the router has learned an interpretable functional decomposition —
separate experts for determiners (`·the ·a ·The`), prepositions (`·of ·in ·to`),
copulas (`·was ·were ·be ·is`), punctuation (96% punct, concentration 1.00),
numerals (56% digit + 42% space), formatting/newlines, single-capital initials,
and CJK. The narrow ones carry the highest gates and the least traffic.

Consequence for admission: calibrating a multilingual MoE on an English-only
corpus yields trustworthy Hessians for the prose experts and thin statistics for
the rest. Either extend the corpus with the scripts the model was trained on, or
run `--expert-coverage-policy preserve-undercovered` so the thin experts are held
at high precision instead of being quantized off <2048 samples. Also worth noting:
the Mixture-of-Depths skip route never won in any block across either run, so
`zaya_use_mod` is effectively inert on this corpus.

## Update (2026-07-25) — ZAYA1 streamed adapter + resume is now the default

`hipfire-coexistence calibrate` gained a third family adapter,
`zaya-stream-v1` (`crates/hipfire-arch-zaya/src/calibration_stream.rs`, arch 16),
so ZAYA1 no longer has to go through the single-load resident collector.

Two design points are ZAYA-specific and worth carrying forward to any family
with cross-layer state:

- **The boundary row is wider than the residual.** ZAYA's EDA router carries
  `router_states [router_hidden_size]` from block `l-1` into block `l`, which
  has exactly the residual's lifetime. It rides in the same boundary row —
  `ModelInspection::hidden_width` is `hidden_size + router_hidden_size` (2304
  for ZAYA1-8B), i.e. the *boundary row width*, not the model's hidden size.
  This is what makes a resumed run pick the EDA state back up instead of
  silently restarting it at zero. A family whose only cross-layer quantity is
  the residual needs none of this.
- **The adapter reads the raw Megatron alternating half-layer checkpoint**
  (even `2l` = block `l`'s CCA attention, odd `2l+1` = its EDA/MoD MoE, residual
  scales one half-layer ahead of the weights they scale). A unit test asserts
  those raw names canonicalize through `ingest::canonical_name` to exactly the
  names `gpu::build_capture_names` uses, so both calibration paths emit
  artifacts the quantizer reads identically.

**Resume is now the default** for `calibrate`; `--no-resume` opts out. The two
modes that cannot checkpoint — `--boundary-ram` and residual probes — quietly
turn the default off, while an explicit `--resume` with either is still an
error. An interrupted run being continuable is the common case, and the old
default silently restarted from layer 0.

### Resident vs streamed parity (ZAYA1-8B, 256 tokens, 1 sample)

Measured by running the streamed path first, replaying the identical token
sequence through the resident collector with `collect_artifacts --job-from`, and
diffing with `artifact compare-calibration` at `--atol 1e-3 --rtol 1e-2`.
Corpus and sample fingerprints matched (`provenance_complete: true`).

**Block 0 is numerically identical**: every layer-0 dense Hessian/imatrix
(q/k/v_current/v_delayed, o_proj, router down_proj, router MLP fc1/fc2/out_proj)
and every layer-0 routed-expert imatrix matched. That covers the embedding +
input affine, the whole CCA mixer (conv pair, delayed value, L2/temp qk-norm,
partial RoPE, GQA attention), both residual affines, and the EDA router prep.

Downstream the two paths are **equivalent but not bit-identical**:

| depth | max abs error, `q_proj.hessian` | top-1 routings differing |
|-------|--------------------------------|--------------------------|
| 0–1   | match                          | 0 / 256 (block 0)        |
| 2     | 0.0078 (one bf16 ULP at 1.0)   | 0                        |
| 10    | 0.14                           | ~1                       |
| 20–39 | 1.2–20                         | ~3 per block             |

Total: **77 of 10240 (0.75%) token-block routings flip**, and the MoD skip
decision matched at every block (all 40 blocks routed exactly 256 tokens on both
paths). The seed is the attention kernel: the resident path runs full-sequence
prefill (`zaya_gqa_attn_f32`), the streamed path runs per-token flash-decode
(`attention_f32`) against per-sequence KV/conv/delayed-value state. Those are
mathematically equivalent but differ in reduction order at ~1e-7, and a top-1
router amplifies a near-tie into a different expert. Hessian off-diagonals are
stored bf16 (`quant_type` 130), so the shallow-depth deltas are literally one
storage ULP.

**Known gap:** the streamed path does not capture the tied `model.embed_tokens`
lm-head input — the engine has no capture seam in the finalizer phase, and
neither the Gemma3 nor Qwen3.5 adapter captures it either. A streamed artifact
carries 360 Hessians where the resident one carries 361, and that projection
falls back to RTN. ZAYA's embed is best left bf16 regardless (see the embed
quant residual-sensitivity finding), but use the resident path if you need it.

**Also found:** the resident `collect-artifacts` path cannot read a raw ZAYA
safetensors directory even though `HfqFile::from_safetensors` opens it —
`ZayaGpuWeights::load` wants the canonical hybrid-block names, and
`from_safetensors` passes the raw Megatron names through unchanged. Convert with
`hipfire-coexistence import safetensors` first, or use the streamed path, which
reads the raw layout directly. (`derive_arch_id` also had to learn `zaya`; a
stale binary silently fell back to arch 5 / Qwen3.5.)

## Update (2026-07-20) — family-neutral native safetensors engine

`hipfire-coexistence calibrate` now emits the canonical HFQM v2
`<model>.calib.hfq` contract directly from Hugging Face BF16 or F16
safetensors. The Rust engine owns sampling, source planning, layer execution,
capture reduction, KLDREF, read accounting, and crash-safe resume;
`scripts/depreciated/collect_hessian.py` is retained only as a parity/debug oracle.

Large checkpoints can use original-shard reference offload. Disk-owned layers
are loaded from the existing safetensors files on demand instead of first
copying hundreds of GiB into an Accelerate `.dat` offload directory. Dense
projections known to consume the same activation (Q/K/V, linear-attention input
projections, and gate/up pairs) share one accumulator while retaining separate
HFQM entries. This reduces the 397B model's estimated dense accumulator memory
from about 36.8 GiB to roughly 23 GiB.

The 397B path uses the native layer stream with KLDREF: embedding and host boundary
activations are materialized once, each Qwen3.5 layer consumes every corpus
microbatch while resident, and finalized layer statistics are spooled before
the weights are released. Routed-expert capture is quota- and tile-aware while
teacher routes always execute. Final norm/lm-head tensors are read once to
append KLDREF. The native read ledger rejects a duplicate teacher read, so
calibration is one source-checkpoint pass. `scripts/two_pass_quantize.py`
composes it with the existing safetensors quantizer pass and records the ledger
and artifact fingerprints in an atomic resume manifest.

Layer telemetry also persists the cost shape behind routed capture: grouped
microbatch count, active-expert sum/maximum, padding rows, gather launches,
full reduction tiles, final partial tiles, and the routed-token point where all
expert roles reached the capture limit. Historical schema-1 records remain
explicitly distinguishable with `launch_telemetry_recorded=false`, but new
schema-2 checkpoints additionally bind the calibration engine executable
identity into every layer and record it in the final artifact separately from
the semantic run fingerprint. A schema-2 binary refuses to continue schema-1
progress rather than silently mixing execution semantics or instrumentation
across binaries, while a completed compatible artifact remains reusable.
The boundary manifest stores a composite of executable and semantic run
identity at job creation, so the same guarantee also covers a crash after
embedding materialization but before the first layer checkpoint.

Qwen3.5 and Gemma3 provide thin adapters to the same engine. Mechanism tests,
including grouped expert capture and mixed OQ4 plus BF16/F16 execution, pass on
gfx1151. The production Qwen3.5-397B teacher and quant artifacts are complete,
and dense Qwen3.5 resident/streamed calibration and residual probes now match;
Gemma3 has the same family-owned resident residual seam but still needs its live
comparison. Python remains parity/debug tooling rather than the production
calibration path.

Production preflight now estimates compact or dense Hessian payloads including
activation aliases, KLDREF bytes, the mmap boundary spool, simultaneous layer
parts plus final assembly, fixed container overhead, and a safety margin. It
reports filesystem availability during `--dry-run` and refuses an insufficient
fresh run. `--pause-after-layers N` provides a durable bounded-layer check that
can be continued with `--resume`; no partial artifact is published.

Live gfx1151 evidence on 2026-07-20:

- Qwen3.5-397B-A17B layer 0 streamed from the real 94-shard checkpoint with
  K=10 routing: 2 tokens produced 20 routed slots at both gate-up and down,
  zero dropped indices, a 203,066,368-byte capture part, 19 canonical logical
  reads totaling 15,184,552,832 bytes, and zero duplicate reads. The deliberately
  undercovered smoke correctly preserved all 512 experts.
- Gemma3-text (`medgemma-27b-text-it`) committed layer 0 and then resumed through
  layer 1. The ledger grew monotonically from 14 to 27 canonical reads and from
  3,644,369,408 to 4,470,166,528 bytes, with zero duplicates and no embedding or
  layer-0 reread.
- The same Gemma job subsequently completed all 62 layers and published a
  38,692,740,100-byte artifact with 434 Hessians, 434 imatrices, one KLDREF row,
  and a complete 809-logical/808-canonical, 54,018,004,480-byte ledger. A bounded
  `oq4++` second pass joined all seven layer-0 Hessians through AWQ+LDLQ with no
  missing/mismatched records and wrote a 249,080,134-byte HFQ.
- Qwen layer-0 fresh-process sequence-batch 1/4/8/16/32 wall times on the same
  32x8-token sample set were 6.06/4.89/4.48/4.31/4.18 seconds. All geometries
  recorded exactly 2,560 K=10 routes at gate-up and down, zero drops/duplicates,
  identical part sizes, and zero diagonal consistency error. Batch 32 is the
  bounded-layer winner, not yet the declared full-run optimum.
- A longer Qwen batch-32 layer-0 run processed 4,096 tokens and observed all
  40,960 K=10 routes at both capture roles without invalid, duplicate, or
  quota-dropped rows. It took 138.88 seconds cold with 15.1 GB maximum RSS, but
  remained deliberately undercovered: 31 experts had zero routes, only one met
  the 2,048-row floor, and 511 were preserved at high precision.
- Resume checkpoints and final artifacts now retain per-layer phase timings. A
  fresh two-token Gemma layer-0 check attributed 64.6 ms to load/upload,
  118.1 ms to execution, 1.226 s to capture serialization, 0.5 ms to finish,
  and 1.437 s to part sync/hash (2.846 s before checkpoint commit).
- Network-backed source lookahead is now family-neutral and ledger-safe. The
  engine reads the next owner's canonical physical ranges through one 8 MiB
  worker chunk into resident staging while the current layer executes, bounded
  to 16 GiB with a 32 GiB live host-memory reserve plus the next layer's upload
  footprint. Any recent full-memory PSI or less than 25% free swap disables the
  transition, and only complete tensor ranges are retained. Complete tensor
  views are consumed directly from staging and released after GPU upload. Checkpoints
  record read/staged/consumed bytes plus background, view, decode, upload,
  release, foreground-wait, pressure-disable reason, and error telemetry;
  matched staged/page-cache/off
  production timings remain to be collected. The first 397B production layer
  using resident staging consumed all 13.124 GB across 15 tensors directly,
  waited 3 microseconds, uploaded in 1.027 seconds, and completed layer
  construction in 1.540 seconds before 232.463 seconds of teacher execution.
  After resident staging reproduced a second swap/SVM stall at layer 27/60,
  the same run resumed with lookahead disabled and committed layer 28 in 336
  seconds: 115 seconds of foreground load/upload and 220 seconds of teacher
  execution. This proves the recovery path and motivates the pressure gate; it
  is not a same-layer controlled performance comparison.
- On the identical 4,096-token Qwen sample set, 256/512/1,024/2,048/4,096-row
  geometries took 7.56/3.76/2.55/2.16/1.89 seconds of layer execution and
  produced identical normalized descriptors and expert telemetry. Total
  pre-checkpoint time was best at 2,048 rows due capture-write variance. At that
  row count, sequence batches 32/64/128 took 2.55/2.15/2.27 seconds; batch 64 is
  the bounded-layer target. The native CLI now uses 2,048 as its auto-tuning
  ceiling while retaining live memory estimates and allocation fallback.

The full Gemma run found a unified-memory residency hazard: mmap-backed source
pages accumulated to about 57 GB RSS and blocked ROCm SVM setup in the finalizer.
Planned safetensor views release completed ranges while canonical tied-weight
pages remain until their declared alias is consumed. Gemma completed after an
initial `posix_fadvise(DONTNEED)` fix, but Qwen's larger shard mappings proved
that file advice alone did not evict mapped PTEs: RSS reached 44.8 GB after two
layers. Adding mapping-level `MADV_DONTNEED` bounded the next production layer
to a 21.9 GB peak; a refault test verifies that released read-only bytes remain
available if a declared alias needs them later.

The tiny-corpus artifacts are mechanism/read-accounting evidence, and the batch
figures are bounded-layer throughput evidence. None establish production expert
coverage, matched KLD/PPL quality, or model admission.

## Done + verified (committed)

- **Reduction kernels** (`kernels/src/calib_reduce.hip`): `calib_sumsq_reduce_f32`
  (imatrix Σx²) + `calib_hessian_outer_f32` (Hessian Σxxᵀ, tiled). CPU-verified
  (`hipfire-rdna/examples/test_calib_reduce`).
- **Capture wiring** (`hipfire-rdna` `ActivationCapture`): `Gpu.active_capture`
  (Arc) + `capture_names` (weight-ptr→name); fired from the BF16/F16 chokepoints
  the lowered super-op path actually uses (`gemm_bf16_x_bf16_wmma_labeled`,
  `gemm_f16_batched_lmhead`). Gated on `is_none()` ⇒ non-calibration forwards are
  byte-identical. `n`/`k` passed by the gemm (the input is a shared scratch buffer
  whose shape ≠ the linear's input width). Verified: `test_capture_hook`.
- **Lib-ified collector** (`hipfire_runtime::calibration::CalibCollector`): generic
  Hessian + imatrix accumulator + `drain()` → HFQ tensors + consistency. Reusable
  by the CLI and the (future) daemon op without an arch-crate cycle.
- **Single-load CLI** (`hipfire-runtime/examples/collect_artifacts`): loads a bf16
  `.hfq` once, arms the collector, forwards over the corpus, writes a unified
  `<model>.calib.hfq`.
- **Artifacts in the `.calib.hfq`** (the unify-on-HFQ decision):
  - `<name>.hessian` [K,K] + `<name>.imatrix` [K] — verified vs the Python Hessian
    on 0.8B: 186 tensors, all K match, byte-identical size, `diag(Σxxᵀ)==Σx²`.
  - `moe_router_histogram` metadata (top1/topk per expert, per-layer, top-64
    co-occurrence = the scheduler-affinity signal) — verified on
    `qwen3.5-35b-a3b-mq4` (256 experts, top-8, all experts hit).
  - `lm_head.kldref_{idx,logit,logz}` (`--kldref`) — verified on 0.8B.
  - AWQ: derived at quant time from the captured imatrix + weights; not stored
    separately (avoids a stale-prone artifact).
- **`hfq` tool** (`hipfire-runtime` bin): `list` / `extract` / `meta-set` /
  `meta-get` — split a Hessian out, embed a jinja2 template, query provenance.
  Bundle-vs-separate is a runtime choice.

## Remaining (needs design/review — paused for the user)

1. **Daemon `Collect` op** — DONE + VERIFIED E2E (loop session 5). Additive
   `{"type":"collect",...}` handler calibrates the resident model in place (no
   reload) and writes the `.calib.hfq`, returning `{"type":"collected", output,
   n_hessian, n_calib_tokens, max_consistency}`. Data plane stays daemon-internal
   (only request + summary cross JSONL). Single-GPU (pp==1) qwen3.5-family bf16
   only; additive/gated, never on the decode hot path. Pieces: typed
   `CollectRequest`/`CollectResponse` + `DaemonRequest::Collect`/`Collected`
   (hipfire-daemon-protocol), `DaemonProcess::collect` adapter method
   (hipfire-daemon-adapter), and the daemon handler (parses msg fields directly,
   calls `qwen35::collect_calibration_artifacts` on `LoadedModel.q35_weights`,
   writes via the shared `qwen35::write_calib_artifacts`). Verified on resident
   `qwen3.5-0.8b-bf16`: `n_hessian=186, max_consistency=0.0`. The CLI subcommand
   (above) remains the standalone in-process path.
2. **eval `calibrate` battery** — DONE (loop session 4). Additive
   `BatteryId::Calibrate` (opt-in via `--battery calibrate`, not in any default
   tier) spawns the `collect_artifacts` example via the examples executor and
   asserts `[CONSISTENT]` + non-zero `n_hessian`. bf16-only (skips otherwise).
   Verified on `qwen3.5-0.8b-bf16`: pass, n_hessian=186, consistent=true.
   Also: **`hipfire collect-artifacts` CLI subcommand DONE** (forwards to the
   example, mirroring `eval`/`host-profile`).
3. **Runtime per-session MoE histogram → microbatch scheduler** — the daemon
   already calls `reset`/`take_moe_router_histogram` (`hipfire-daemon/src/main.rs`).
   Wire the per-session histogram (esp. the co-occurrence pairs) into the
   scheduler's expert-affinity grouping so the paged-expert (`WeightPager`) path
   sees fewer page-ins. Scheduler hot-path — do with review.
4. **MoE-expert capture for A3B Hessians** — DONE + VERIFIED E2E (loop session 3,
   see Update below): `build_capture_names` maps MoE dense projections (full
   Hessian) + resident routed experts (imatrix-only); the bf16 MoE forward gap
   that blocked E2E is fixed (`moe_gemv_plain`). Verified on `qwen3.6-35b-a3b-bf16`:
   350 dense Hessians CONSISTENT + 3014 routed-expert imatrices. Remaining:
   paged-mode experts (WeightPager-owned buffers — needs pager-side capture).
5. **#11 cross-model** — once (4) lands, generate full `.calib.hfq` (Hessian +
   histogram) for the two `qwen3.5/3.6-35b-a3b` models and re-run the
   importance/KLD sweep to confirm generality.

## Update (2026-06-18, loop session 2)

- **Driver lib-ified** into `qwen35::collect_calibration_artifacts` (+ `CalibOpts`/
  `CalibArtifacts`); `collect_artifacts` is now a thin CLI. The daemon op + a CLI
  subcommand reuse this driver (no duplication).
- **#11 cross-model VALIDATED on `qwen3.5-9b-bf16`** (dense, same hybrid arch):
  248 hessian tensors, `diag(Σxxᵀ)==Σx²` CONSISTENT, `[4096,4096]` Hessians. The
  collector generalizes beyond 0.8B. (Run used 16 tokens — mechanism check, not a
  full-quality Hessian.)
- **Scaling finding (RESOLVED — streaming writer shipped).** Originally the
  collector materialized ALL Hessians in RAM (`drain() -> Vec<HfqMemTensor>`,
  then `write_hfqm_package_mem`) — ~32 GB for a 9B (down-proj Hessians are
  `[mi², ]`), filling the 63 GB RAM-backed `/tmp`. Fixed with a **streaming
  writer** (`hfq::write_hfqm_package_streaming` + `CalibCollector::write_streaming`):
  the HFQM index/metadata are written first (payload sizes are deterministic from
  `k`), then each Hessian/imatrix is downloaded → normalized → written → dropped,
  one at a time. Peak host RAM is now a single tensor (~hundreds of MB) instead of
  the whole package. `collect_calibration_artifacts` writes directly to the output
  path (no `CalibArtifacts`/`write_calib_artifacts` round-trip — removed).
  Verified byte-correct on 0.8B (quantizer reads the streamed file, LDLQ engages
  on all 8 layer-0 tensors). The imatrix-default / fp16 alternatives are
  unnecessary now (no precision loss, no RAM ceiling). Still: write big-model
  artifacts to a real disk path, not the RAM-backed `/tmp`.

- **Disk-size finding (2026-06-26).** New `.calib.hfq` Hessians default to compact
  storage: exact F32 diagonal plus BF16 lower strict triangle (`quant_type=130`),
  still exposed to the quantizer as a logical symmetric `[K,K]` Hessian. This
  targets ~4× smaller Hessian payloads while preserving the diagonal exactly.
  Legacy dense F32 Hessian output remains available with
  `HIPFIRE_CALIB_HESSIAN_STORAGE=full-f32`, and the reader accepts both formats.

- **Daemon `Collect` op — design (review-gated, NOT done autonomously):** the daemon
  (`hipfire-daemon/src/main.rs`, ~9k lines) dispatches via a custom JSON
  message-parser loop (`parse_*_request` at ~8865+), not a clean `match
  DaemonRequest`; the `DaemonRequest` enum is the *client/adapter* send-side. So a
  `Collect` op spans: (a) a `DaemonRequest::Collect(CollectRequest)` variant
  (`hipfire-daemon-protocol`), (b) an adapter send method
  (`hipfire-daemon-adapter`), (c) a server-side message parser + handler in the
  daemon loop that calls `qwen35::collect_calibration_artifacts` on the resident
  `LoadedModel.q35_weights` (+ `q35_config`, the daemon's main `Gpu`, tokenizer),
  writes the `.calib.hfq`, returns the path. Data plane stays daemon-internal
  (only request + path cross JSONL). Touches the flagged-unstable daemon
  interface ⇒ do with review. The `collect_artifacts` CLI already provides the
  standalone (in-process, daemon-free) path.

## Update (2026-06-18, loop session 6) — build COMPLETE; cross-model artifact landed

The full single-load calibration-artifact collector (Phases 1–5) is **done and
verified**. Final-session work:

- **Genuine cross-model artifact (item 4 / #11).** Generated a real (128-token,
  not 8-token mechanism) `.calib.hfq` for the focus MoE model
  `qwen3.6-35b-a3b-bf16` to **real disk** (`~/.hipfire/calib/`, not tmpfs):
  **7.3 GiB**, `n_hessian=350` (dense projections, `diag(Σxxᵀ)==Σx²` CONSISTENT,
  rel-err 0.000e0), `n_imatrix=11568` (350 dense + 11218 routed-expert
  imatrix-only = 5609 distinct expert-projections covered, up from 3014 at 8
  tokens — coverage scales with tokens as expected). 35.9 s capture / ~60 s
  total on gfx1151. **Collector cross-model generality confirmed**: 0.8B (dense,
  byte-identical to the Python Hessian), 9B (`[4096,4096]` CONSISTENT), and A3B
  (MoE, dense-Hessian + per-expert-imatrix) all produce correct artifacts.
  Correctness is token-count-independent (the consistency check holds at any N),
  so this is a full-size, real-disk validation of the focus model.

- **Importance/KLD sweep bridge — RESOLVED.** The quantizer reads canonical
  `<name>.hessian` tensors directly from `.calib.hfq`; no HFHS extraction shim
  is needed. `scripts/roughquant_ablation_oracle.sh` was parameterized and the
  9B rerun is recorded in `docs/roughquant/9b-importance-generality.md`. Legacy
  `.hessian.bin` inputs remain historical fixtures only and must not be produced
  by new workflows.

## f32 vs f64 Hessian accumulation (measured — f32 is sufficient)

The GPU accumulates `Σxxᵀ` in **f32** (`calib_hessian_outer_f32`). Question: does
f32 summation error over many tokens hurt LDLQ? Measured with an opt-in CPU f64
reference (`HIPFIRE_CALIB_F64_AUDIT=1` — `CalibCollector` accumulates the same
staged rows in f64 on the CPU and `drain` reports the max element-wise f32-vs-f64
rel-diff). RDNA has no f64 matrix units and only ~1:16 scalar f64, so the
reference is computed CPU-side, not on-GPU.

qwen3.5-0.8b-bf16 (gfx1151):

| tokens | max f32-vs-f64 Σxxᵀ rel-diff | collect time |
|-------:|----------------------------:|-------------:|
| 256    | 2.46e-4 | 40 s  |
| 4096   | 6.07e-4 | 570 s |

**Verdict: f32 accumulation is sufficient — f64 is NOT made the default.**
- Divergence grows sub-linearly with N (16× tokens → 2.4× error ≈ N^0.32, not the
  N¹ of catastrophic cancellation); extrapolated to a 262k-token corpus ≈ 2e-3.
- LDLQ adds a 1% diagonal ridge (`damp = 0.01·mean_diag`) before Cholesky, so even
  the extrapolated full-corpus perturbation is ~5× below the regularization the
  quantizer already applies — no quant-quality impact.
- The CPU f64 path is also 14× slower (570 s vs 40 s at 4096 tok), confirming a
  GPU f64 kernel would be the wrong trade. The audit stays as an opt-in
  diagnostic (e.g. for a future arch, or to re-check at very large N).

## Perf note
Per-token AR forward + per-token K×K outer-product is slow (~35 s / 256 tok on
gfx1151). A full 262k-token calibration wants **batched-prefill capture** (process
many tokens per forward, batch the outer-product) — the throughput follow-up.
Always state the box for perf numbers (gfx1151/Strix Halo here).

## Update (2026-06-18, loop session 3)

- **Buffer-and-flush capture landed (perf).** `CalibCollector` now stages
  activation rows into a `[FLUSH_BATCH=256, K]` buffer and runs a SINGLE batched
  `calib_hessian_outer_f32` / `calib_sumsq_reduce_f32` per 256 rows (the tiled
  GEMM is built for N≥16, so per-token N=1 wasted ~256×). Verified on
  `qwen3.5-0.8b-bf16` (gfx1151): 186 hessians, `diag(Σxxᵀ)==Σx²` CONSISTENT,
  **4.5–7.8 s** vs the prior ~35 s/256-tok. Commit 6084ade2.

- **MoE-expert capture (A3B) — capture side built + verified-by-construction.**
  `build_capture_names` now maps the MoE-layer dense projections (attention
  q/k/v/o or qkv/z/a/b/o, the router `mlp.gate`, and the shared expert
  gate/up/down) → **full Hessian** (same gemm chokepoint as the dense layers),
  and the resident routed experts (`mlp.experts.{x}.{gate_up,down}_proj`) →
  **imatrix-only**. The collector gained `CalibCollector::with_imatrix_only(substr)`:
  tensors whose name contains a substring (here `".experts."`) accumulate only
  Σx² (no [K,K] Hessian alloc / outer-product). Rationale: a full per-expert
  Hessian for A3B is **256 experts × 40 layers × [K,K] ≈ 196 GB** — does not fit
  on-GPU; the imatrix is a K-vector (~100 MB total) and is the importance signal
  AWQ-style quant needs. Dense path re-verified unchanged (0.8B: 186 hessians,
  CONSISTENT). Routed-expert names are emitted only when experts are **resident**
  (`HIPFIRE_QWEN35_PAGED_EXPERTS` unset = the default; paged mode owns buffers in
  the WeightPager and patches ptrs per-token, so capture-by-buf-ptr can't key them).

- **bf16 A3B MoE forward — RESOLVED, and A3B capture VERIFIED E2E.** The bf16 MoE
  FFN forward routes through the Ship 4.1 MoE dispatch family
  (`hipfire_runtime::llama::moe_family().run()` → `pipeline::run_moe_decode`),
  whose inner gemvs resolved via `for_gemv` — which has **no `(BF16, Plain)` entry**
  (BF16 has no scalar GEMV kernel; it uses WMMA), so the first forward token failed
  `unsupported gemv.unknown for /`. Fix (commit pending): added `moe_gemv_plain`
  (`pipeline/mod.rs`), a `weight_gemv`-style helper that short-circuits BF16 →
  `gemm_bf16_x_bf16_wmma`, applied at the 7 bf16-reachable gemv sites (router +
  3 shared on the gate side; shared-down + per-expert gate_up + down in the
  CPU-top-K fallback). Non-bf16 dtypes still go through `run_auto` unchanged
  (byte-identical for mq*/paro production — the change only touches the arm that
  previously hard-errored). **Bonus:** routing bf16 through `gemm_bf16_x_bf16_wmma`
  is exactly the capture chokepoint, so this simultaneously unblocked the forward
  AND fired the MoE capture. Verified on `qwen3.6-35b-a3b-bf16` (gfx1151, 8 tokens):
  **350 dense Hessians (CONSISTENT, rel-err 0.000e0)** + 3364 imatrices (350 dense
  + 3014 routed-expert imatrix-only), 3714 tensors. Structure confirmed via `hfq
  list`: dense projections (linear_attn in_proj_*/out_proj, full_attn q/k/v/o,
  router `mlp.gate`, shared_expert gate/up/down) carry both `.hessian [K,K]` +
  `.imatrix [K]`; routed `mlp.experts.{x}.{gate_up,down}_proj` carry `.imatrix`
  only (shapes [2048]/[512]). A full-corpus run (more tokens) covers more of the
  256×40 experts; 8 tokens hit 1507 distinct (expert,layer,proj) imatrices. Paged
  mode still uncovered (WeightPager owns buffers).
