# `<embed>.coarse.weight` is emitted by default and read by nothing

Status: found and **FIXED** 2026-09-04 — emission is now opt-in behind
`--coarse-lmhead`. Existing artifacts keep the dead tensor until rebuilt; it is
inert, so they still serve correctly. A second, latent defect in how the tier is
built on UNTIED models is documented below and deliberately NOT fixed, because
nothing consumes the tier today and fixing it blind would be untested code.

## Symptom

Every `oq4.25++` artifact carries an `<embed>.coarse.weight` tensor
(`QuantType::CoarseQ4Row`, code 48) that no loader reads. Measured on the five
artifacts on halo:

| artifact | coarse tier | share of file |
|---|---|---|
| `Qwen3.5-27B--oq4.25++.hfq` | 636.20 MB | **4.0%** of 16.10 GB |
| `Qwen3.6-27B--oq4.25++.hfq` | 636.20 MB | 4.0% |
| `Qwen3.8-27B--oq4.25++.hfq` | 636.20 MB | 4.0% |
| `Qwen3.5-35B-A3B--oq4.25++.hfq` | 254.78 MB | 1.3% of 19.20 GB |
| `Qwen3.6-35B-A3B--oq4.25++.hfq` | 254.78 MB | 1.3% |

~2.4 GB across those five files, plus the write bandwidth on every build.

## Cause

The tier is a real technique (`docs/kernel_work/two-stage-lmhead.md`): a coarse
Q4 row-normalised copy of the output projection shortlists top-K rows, then an
exact pass rescores only those, which is greedy-exact and ~4x cheaper on
bandwidth-bound decode. The quantizer emits it. **The serving path does not read
it.**

`llama::lmhead_project` builds its coarse tier at load, from the head it will
actually shortlist for:

```rust
let coarse = build_coarse_from_compact(gpu, &w.buf, vocab, hidden, bits)
```

so the artifact's sidecar is never consulted. Grepping every reference to
`CoarseQ4Row` / `.coarse.weight` outside `hipfire-quantize` finds only the
`hipfire-quant-format` enum plumbing, an env-doc string, and a comment in
`qwen35/decode_layers.rs:2959` that says so in passing — *"the 35B's CoarseQ4Row
tensor was dead weight for qwen35 serving"*. That comment is about a routing fix
for the runtime-built tier; the sidecar stayed unread either way.

The runtime path is additionally gated on `HIPFIRE_LMHEAD_TWOSTAGE` **and** a
BF16 head. These heads are `OqPlusCompact`, so it does not engage at all.

Emission was on by default, so nobody had to ask for the bytes.

## Second defect: the tier is built from the wrong matrix on untied models

Not fixed. Recorded so it is not rediscovered by whoever wires the loader.

`quantize_hfq_source_tensor` builds the tier inside the `is_embed` arm, and
`is_embedding_table_name` matches `embed_tokens` / `embeddings.weight` /
`embedding.weight` — **not** `lm_head.weight`. Its comment claims:

> Built from the SAME f32 the fine tier is quantized from, so the coarse ranks
> rows consistently with the fine weights it shortlists for.

That holds only when the model is TIED. All five artifacts above are untied
(`tie_word_embeddings: false`, with a separate trained `lm_head.weight`), so the
sidecar ranks rows of the *embedding* to shortlist for a *different* matrix.
Recall@K — the one quantity `two-stage-lmhead.md` says the method's correctness
rests on — would collapse, and the failure mode is a silently wrong argmax
rather than an error.

Same class as `1b707c3db` on `origin/quant/qwen35-0.8b-sub4bit`: *"the pre-pass
derived lm_head from the EMBEDDING even on untied models, replacing a trained
output head with the embedding matrix."*

Harmless today only because nothing reads the sidecar.

## Fix

`--coarse-lmhead` opts in; the default is off. `HIPFIRE_NO_COARSE_LMHEAD` is
still honoured and beats the flag, so a script that sets it is unaffected.

`--no-coarse-lmhead` is now **parsed**. It was documented in the
`coarse_lmhead_enabled` doc-comment and in the `hipfire-env` registry but never
read on master — only the env var ever turned emission off. The flag exists on
`origin/quant/qwen35-0.8b-sub4bit` (`db1d683b9`) and never landed here.

Wire a loader and this can become a default again — after fixing the untied case.

## Test

`coarse_lmhead_is_opt_in_and_the_negative_wins` (`hipfire-quantize::cli`) pins
the off default and makes a contradictory command line resolve to off. An
artifact that silently regrows a 636 MB unread sidecar is the regression it
exists to catch.

## Related

- `docs/kernel_work/two-stage-lmhead.md` — the technique, and its porting checklist.
- `docs/todo/2026-08-08-quant-benchmark-queue-handoff.md` — the queue that
  produced the five artifacts above.
