# Idefics3 bring-up: IBM Granite-Docling-258M (arch 27)

gfx1151 (Strix Halo, 128 GB UMA). Source: `ibm-granite/granite-docling-258M`
(`Idefics3ForConditionalGeneration`, transformers 4.56 checkpoint).
Artifact: `Granite-Docling-258M--oq8.hfq` (328 MB).

## Shape of the model

| part | tensors | notes |
|---|---|---|
| vision | `model.vision_model.*` | SigLIP, 768 × 12 layers, patch 16, 512² tiles → 1024 patches |
| connector | `model.connector.modality_projection.proj` | pixel shuffle ×4 (1024×768 → 64×12288), then `[576, 12288]`, no bias |
| text | `model.text_model.*` | plain LLaMA, hidden 576, 30 layers, 9 q / 3 kv heads, head_dim 64, tied embeddings |

The processor resizes the long side to 2048 (`size.longest_edge`), rounds both
sides up to multiples of 512, cuts row-major 512² tiles, and adds one global
view resized to 512². Every frame is exactly 512², so the NaViT patch masks and
position bucketing in `Idefics3VisionEmbeddings` reduce to positions `0..1023`.
Each `<image>` expands to `<fake_token_around_image><row_r_col_c>` + 64
`<image>` per tile, then the global block.

## What was reused

- SigLIP tower and forward: `hipfire-arch-gemma3-vl` (`SigLipWeights::load_prefixed`,
  `vision_forward`, `patchify_rgb8`). Same LayerNorm+bias / gelu-tanh / learned
  position layout; only the tensor prefix and tile size differ.
- Text decoder: the shared LLaMA loader. `HfqFile::resolve_idx` now also tries
  `model.text_model.<rest>`, and `config_from_hfq` reads a nested `text_config`.
- Splice: `llama::prefill_forward_spliced` copies the projected rows into the
  batched prefill at the `<image>` positions; decode is the shared `decode_loop`.
- Serving: a registered `ServingFactory` (no load.rs ladder arm). Backends whose
  `caps().vision` is set now receive decoded frames from the daemon.

## Quantization: why "oq8" is not uniform here

GPU Opus kernels require `K % 256 == 0`. Hidden 576 is not, so q/k/v/o/gate/up
(K = 576, 180 tensors) cannot use Oq8G256 on the GPU:

- Default `--format oq8` writes them as `Oq8G256RowPadded` (qt 43). That layout
  is NPU-only; the GPU LLaMA loader panics on it (`unsupported quant_type 43`).
- `HIPFIRE_OQ_RAGGED_Q8=1` (used for the shipped artifact) writes them as Q8F16.
  down_proj (K = 1536, 30 tensors) is real Oq8G256.
- Vision tower and connector stay BF16 (`should_quantize`); the embedding/tied
  head stays BF16 (Lut3-compressed on disk).

Making Oq8 real for K = 576 needs zero-padded activations before the FWHT on
every Oq8 route (48 rotate call sites plus fused/batched kernels in the LLaMA
path). Not attempted here.

## Results

Prompt "Convert this page to docling.", greedy, `Granite-Docling-258M--oq8.hfq`.
The oracle is HF transformers 5.2, fp32, teacher-forced on hipfire's own output
(per-position argmax of HF vs the token hipfire emitted):

| image | frames | prompt tok | hipfire tok | HF argmax agreement |
|---|---|---|---|---|
| `assets/new_arxiv.png` 1190×1540 | 4×4 + 1 | 1142 (= HF) | 400 | 100.0% (CPU eager and GPU sdpa) |
| `assets/granite_docling_split_page.png` 3128×2256 | 3×4 + 1 | 877 | 300 | 100.0% |
| 400×300 crop of the arXiv page | 3×4 + 1 (upscaled to 2048) | 877 | 171 (EOS) | 99.4% (one near-tie at token 17) |

- SigLIP global-frame features vs HF: cosine 0.99976 (BF16 tower vs fp32).
- Decode 118-166 tok/s; text-only requests serve through the plain LLaMA path.
- HF's own `generate()` (transformers 5.2) returned a degenerate decode
  (`<loc_0><loc_0><loc_500><loc_25>` then EOS) on the arXiv page, while its
  teacher-forced forward agrees with hipfire on every token. Use the forward
  pass, not `generate()`, as the oracle for this family on 5.2.

Two bugs found on the way:

- Image placeholders in user content were escaped by the chat frame's
  special-token hardening (`LITERAL_ESC`). After the split, the leftover mark
  turned the first `<fake_token_around_image>` into 7 text tokens (prompt 1148
  vs HF 1142). Placeholders are now inserted and un-escaped in exactly one
  place (`generate_registered_backend`); user-typed `<image>` stays text.
- HF composites RGBA onto white before resizing; the tiler does the same.

## Known limits

- RDNA3/RDNA4 only for images: the SigLIP tower and connector use
  `gemm_bf16_x_bf16_wmma`, which has no non-WMMA fallback (same as gemma3-vl).
  Text-only requests use the plain LLaMA path.
- No calibration adapter, so no `oq*+` / `oq*++`.
- Video input is not routed to idefics3.
