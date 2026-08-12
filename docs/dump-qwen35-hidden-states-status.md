# dump_qwen35_hidden_states: repaired, but its capture semantics are unresolved

## What was wrong and is now fixed

The tool looped `forward_scratch_with_hidden` per token — the DECODE path from a
cold state at position 0. Generation never does that: it prefills the prompt and
only then decodes, so decode-from-cold is not a supported entry.

It now calls `forward_prefill_batch`, which takes the ring buffer directly, so
the trusted path performs the capture itself.

Two further traps, both recorded in the source:

- **Do not call `commit_staging_to_ring` from the caller.** Prefill commits each
  chunk internally (`prefill_batch.rs:5484`) and advances the head by n. A
  second commit re-advances the head and writes empty staging over the captured
  data — producing an all-zero dump that passes every structural check.
- `HiddenStateRingBuffer::new` cannot express "all layers": it derives ids from
  `dflash_extract_layer_ids`, which spaces picks across `1..n_layers-3`, so the
  rounding collides and the constructor rejects duplicates. Use
  `new_for_layers`.

## What is still wrong

The captured states do not match a verified forward, and the mismatch is NOT
explained by anything checked so far.

On Qwen3.5-0.8B, whose implementation matches `dump_logits_qwen35` at cos 0.9996
end to end:

| check | result |
|---|---|
| dumped layer i vs verified walk's layer i output | 0.62 at layer 0, decaying with depth |
| dumped FINAL state -> final norm -> tied lm_head, vs `dump_logits` | 0.776 |
| same, per position at n_ctx 4 | 0.57, 0.52, 0.59, 0.47 |

Checked and eliminated:

- **Layer mapping.** A best-match scan over all pairs maps dumped slot i to walk
  layer i+1 monotonically for i <= 14 — the identity mapping, no offset.
- **Position ordering.** Position 0's state is bit-identical between an n_ctx 1
  run and an n_ctx 4 run, which is what causality requires. The ring is not
  rotated.
- **The capture point.** `prefill_chunk.rs:3934` writes `pbs.x_batch` under the
  comment "Post-layer hidden extract for the DFlash draft path" — the same
  post-layer residual the decode path captures at `decode_layers.rs:949`.
- **KV quantisation.** Default `--kv-mode` is `q8`; forcing `fp32` changes
  nothing at layer 0 (a linear_attn layer, which uses no KV cache).
- **The head convention.** The same final norm (`1 + w`, GemmaRMSNorm) and tied
  embedding reproduce the runtime's logits from THIS implementation's final
  state at 0.9996, so the head math is not the variable.

So the forward is right (its logits agree), the capture point looks right, the
mapping is right, and the states are still wrong. The most likely remaining
explanation is that `pbs.x_batch` at capture time is not the value the name
implies on this path — for instance a chunk-local buffer that has already been
advanced, or one holding a pre-residual quantity.

## Why this matters

This is the tool that would give per-layer references for the 35B, whose forward
is orthogonal to the runtime at ONE token. Every cheaper hypothesis for that bug
has been eliminated (see `linear-attn-real-model-status.md`), and a trustworthy
per-layer oracle is the remaining instrument.

Until the semantics are settled, these dumps must not be used as a reference —
which is exactly the mistake made earlier in this session, when the old
per-token states were read as evidence about layer 0 and produced a confident
wrong conclusion.
