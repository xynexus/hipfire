# HOWTO: add a model family to the tiny-quant matrix

The **tiny-quant matrix** (`hipfire-eval` battery `tiny_quant` + `tests/tiny-quant-gate.sh`)
is a tokenizer-free, daemon-free, no-real-checkpoint test of the whole quant
pipeline — **quantizer → loader → dequant kernels → output** — for each model
family. Per family it: emits a seeded tiny random-init bf16 fixture, quantizes it
to that family's loader-supported formats, generates a tiny Hessian/imatrix
(`collect`), and scores each quant's **KL divergence vs a near-full-precision
anchor** over a fixed synthetic token stream, drift-checking against committed
per-GPU baselines.

This guide walks through adding a new family, using **llama (arch 0)** as the
worked example (it was added this way). Budget ~1–2 hours for a dense family.

## The 4 files you touch

| File | What you add |
|------|--------------|
| `crates/hipfire-quantize/src/fixture.rs` | A `<Family>Tiny` preset: `config_json()` + `manifest()`, an `emit_fixture` match arm, a unit test |
| `crates/hipfire-serving-core/src/tiny_harness.rs` | A `TinyArch` variant + `TinyModel` variant: load / `forward_logits` / `capture_names` dispatch |
| `crates/hipfire-eval/src/executor_tinyquant.rs` | A `FamilyPlan` (anchor + candidate formats) |
| `tests/fixture-roundtrip-nogpu.sh` | The family in `ARCHS`/`EXPECT_ID`/`ARCH_FLAGS` |

Then **record baselines** and commit.

## Step 0 — recon the loader (the part that bites)

Before writing anything, read the family's **HFQ loader** and answer four questions.
Most of the work is discovering these constraints; the code is mechanical.

1. **Exact tensor names + shapes.** Find `load_weights*` for the arch (e.g.
   `hfq::load_weights_hfq` for llama, `qwen2::load_weights`, `gemma3::weights::load_weights`,
   `MiniMaxWeights::load`). Your `manifest()` must match these names byte-for-byte —
   the quantizer's name-mapper and the loader both key off them.
2. **Which weight quant_types the loader supports.** Grep the loader's
   `match info.quant_type`. This bounds your candidate formats. Examples found in
   practice:
   - llama linears: q8f16(3)/hfq4(6)/mq4(13)/mq3(17)/… but **NOT** F16(1)/BF16(16)/MQ6(15).
   - qwen2/gemma3 linears: F16(1)/Q8(3)/HFQ4G256(6)/HFQ4G128(7) — **NOT** mq4.
   - minimax MoE experts: MQ4/MQ6 only (indexed-GEMV); F16 experts **panic**.
3. **What it rejects.** e.g. the llama loader **errors if a `q_proj.bias` tensor
   exists** (bias ⇒ use qwen2/arch 7). gemma3/qwen2 loaders **reject bf16 norms**.
4. **Tied vs untied lm_head**, and any required config fields (`config_from_hfq`).

### Two rules that fall out of step 0

- **1-D tensors (norms, biases) must be emitted as F16, not bf16.** The per-arch
  loaders reject bf16 for norms/biases (real checkpoints keep them F16/F32). Use
  `TensorSpec::f16(...)` for norms/biases, `TensorSpec::new(...)` (bf16) for weight
  matrices. (qwen3.5 is the exception — it stays all-bf16 so its committed golden
  hashes don't drift; don't change it.)
- **Keep every quant-fed dim a multiple of 256** (the g256 quant group): `hidden`,
  `intermediate`, `moe_intermediate`, `vocab`. head_dim 128 is the safe default.

## Step 1 — the fixture preset (`fixture.rs`)

Mirror an existing `*Tiny` struct (llama copied the qwen2 shape minus biases/qk-norm).
Keep it tiny (<10M params): e.g. `hidden=256, inter=512, vocab=4096, layers=2,
n_heads=2, n_kv_heads=1, head_dim=128`.

```rust
struct LlamaTiny { hidden, inter, vocab, layers, n_heads, n_kv_heads, head_dim }
impl LlamaTiny {
    fn preset() -> Self { /* the tiny dims above */ }
    fn config_json(&self) -> serde_json::Value { /* model_type + HF config fields config_from_hfq reads */ }
    fn manifest(&self) -> Vec<TensorSpec> { /* exact loader tensor names; norms via ::f16, weights via ::new */ }
}
```

Wire it into `emit_fixture`'s match (accept the `model_type` aliases), extend the
unsupported-arch error string, and add a unit test (assert key tensors present,
forbidden ones absent — e.g. llama asserts no `.bias`/`q_norm` — and `<10M` params).
Add the family to the `emit_new_families_are_deterministic` test loop.

Validate CPU-only:
```
cargo test -p hipfire-quantize fixture
./tests/fixture-roundtrip-nogpu.sh        # after step 4
```

## Step 2 — the harness (`tiny_harness.rs`)

Add a `TinyArch` enum variant (+ `parse`/`arch_id`/`as_str`) and a `TinyModel`
variant holding the loaded config/weights/state. Implement three dispatch arms:

- **`load`**: `config_from_hfq` → `load_weights` → state/scratch + KV cache. Pick
  the closest existing template by *forward shape*: `forward_step`+State (qwen2,
  gemma3) vs `forward_scratch`+scratch+KV (qwen35, **llama**) vs `decode_step`
  returning `Vec<f32>` (minimax).
- **`forward_logits`**: run one token, return host logits. If the forward also
  **samples** (llama's `forward_scratch` does), pass greedy/no-op sampling params
  and read the pre-sample `scratch.logits` — the sampled token is discarded.
- **`capture_names`**: map each linear's `wt.buf.buf.as_ptr() as usize` → its
  checkpoint name **minus `.weight`** (so the Hessian sidecar keys match the
  quantizer's `--hessian`/`HIPFIRE_QTIP_HESSIAN` lookup). Walk the weight struct's
  public fields. MoE routed experts go through indexed-GEMV (not `weight_gemv`), so
  leave them out. (qwen35 needs a `model.language_model.`→`model.` prefix fixup;
  most families use the short prefix directly.)

The capture hook itself is already universal: `weight_gemv` calls
`gpu.maybe_capture_activation`, a no-op unless a collector is armed.

## Step 3 — the battery `FamilyPlan` (`executor_tinyquant.rs`)

```rust
FamilyPlan {
    arch: "llama",
    anchor: "q8f16",            // highest-fidelity *loadable* format (NOT fp16 — llama can't load F16 weights)
    candidates: &["hfq4", "mq4", "mq3"],
    quant_flags: &[],            // e.g. &["--arch-id", "7"] for qwen2 (its model_type auto-detects to arch 1)
    calibrated: &[],             // qtip3-sim emits bf16; only families whose loader loads bf16 (qwen3.5) get a calibrated cell
}
```

Anchor = the highest-fidelity format the loader accepts (fp16 for qwen2/gemma3;
**q8f16** for llama since it rejects F16 weights; **mq6** for minimax since its MoE
kernels need MQ4/MQ6 experts). Candidates = other loadable formats.

If a specific format **GPU-faults on some arch** (minimax topk faults on gfx1151),
exclude that *family* per-host with `HIPFIRE_TINYQUANT_FAMILIES=qwen2,gemma3,...`
rather than dropping it from the matrix.

## Step 4 — no-GPU roundtrip (`tests/fixture-roundtrip-nogpu.sh`)

Add the family to `ARCHS`, `EXPECT_ID` (the quantize arch-detect line — `id=N`, or
`to N` if you pass `--arch-id`), and `ARCH_FLAGS`.

## Step 5 — build, validate on GPU, record baselines

```
RUSTFLAGS="-D warnings" cargo build --release \
  -p hipfire-quantize --bin hipfire-quantize \
  -p hipfire-serving-core --example tiny_quant_probe \
  -p hipfire-eval --bin hipfire-eval

# manual smoke (gfx1103 / any iGPU needs HIPFIRE_GPU_SLAB_LOAD=0):
Q=./target/release/hipfire-quantize; P=./target/release/examples/tiny_quant_probe
$Q --emit-fixture llama --out /tmp/ll --seed 42
$Q --input /tmp/ll --output /tmp/ll-q8.hfq --format q8f16
LD_LIBRARY_PATH=/opt/rocm/lib HIPFIRE_GPU_SLAB_LOAD=0 \
  $P kld --arch llama --ref /tmp/ll-q8.hfq --cand /tmp/ll-q8.hfq --len 16 --warmup 2   # self-KLD must be 0
$P collect --arch llama --model /tmp/ll-q8.hfq --out /tmp/ll-calib-q8.hfq --len 16     # n_tensors>0, consistency~0

# record + check:
./tests/tiny-quant-gate.sh --record        # writes tests/tiny-quant-baselines.txt for THIS gpu_arch
./tests/tiny-quant-gate.sh                  # all cells PASS
```

Expect KLD **monotonic in bit-width** (q8f16 < mq6 < mq4 < mq3). Self-KLD (a model
vs itself) must be exactly 0 — if not, the harness dispatch is wrong.

## Recording baselines on another GPU (e.g. halo / gfx1151)

Baselines are per-`gpu_arch`; record on each validation box. Gotchas learned the hard way:

- **`.bashrc` toolchain.** A non-interactive SSH shell (`ssh halo 'cmd'`, even
  `bash -lc`) hits the `.bashrc` `*) return` non-interactive guard, so the ROCm
  build toolchain isn't on PATH and any **uncached kernel fails to JIT** (`failed to
  run hipcc` / `clang-offload-bundler`). Export it explicitly:
  `export LD_LIBRARY_PATH=/opt/rocm/lib; export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$HOME/.hipfire/bin:$HOME/.cargo/bin:$PATH`
  (use `/opt/rocm/lib/llvm/bin`, **not** `/opt/rocm/llvm/bin` — wrong LLVM ⇒ bad codegen).
- **Isolate the build.** Use a `git worktree` at the target branch (its own
  `./target`); don't rebuild against another agent's checkout. Use the prebuilt
  `~/.hipfire/bin/hipfire` for the gpu-lock CLI.
- **Wrap SSH** in `timeout -k 3 N ssh -o BatchMode=yes ...` so it can't hang on an
  auth/agent prompt.
- A family whose kernel **GPU-faults** on that arch must be excluded with
  `HIPFIRE_TINYQUANT_FAMILIES=` — a fault can wedge a shared GPU.

`--record` merges: it preserves other GPUs' rows and re-recorded cells keep any
hand-tuned `rel_tol` (5th column).

## Known family-specific constraints (quick reference)

| family | anchor | candidates | notes |
|--------|--------|------------|-------|
| llama (0) | q8f16 | hfq4, mq4, mq3 | no bias/qk-norm; loader rejects F16/BF16/MQ6 weights |
| qwen2 (7) | fp16 | q8f16, hfq4 | needs `--arch-id 7` (model_type auto-detects to arch 1); has QKV bias |
| dots_ocr (8) | fp16 | hfq4 | complete Qwen2 text + Dots vision artifact load; synthetic RGB preprocessing feeds `vision_forward`, then visual rows splice through `forward_step_with_embed`; vision tensors included with `--include-vision --vision-quant hfq4` |
| deepseek4 (9) | deepseek4-source-precision | deepseek4-source-precision | text core: Q/O-LoRA, Hyper-Connections, score-routed MoE, shared experts, native MQ2-Lloyd routed experts; needs `--allow-mq2-lloyd` |
| deepseek4_compressed (9) | deepseek4-source-precision | deepseek4-source-precision | compressed-KV/indexer variant: ratio-4 compressor/indexer tensors plus long state hashing; needs `--allow-mq2-lloyd` |
| deepseek4_mtp (9) | deepseek4-source-precision | deepseek4-source-precision | MTP variant: loads `mtp.0.*`, seeds `mtp_last_hidden` with main decode, then drives `mtp_forward`; needs `--allow-mq2-lloyd` |
| gemma3 (12) | fp16 | q8f16, hfq4 | (1+w) norm baked at ingest; GeGLU |
| gemma3_vl (13) | fp16 | q8f16, hfq4 | complete multimodal artifact load; embedded-PNG preprocessing feeds `vision_forward` + `project`, then image-token embeddings splice through `forward_step_with_embed`; vision tensors included with `--include-vision --vision-quant q8f16` |
| gemma4_dense (24) | fp16 | q8f16, hfq4 | dense text only |
| gemma4_ple (24) | fp16 | q8f16, hfq4 | PLE/KV-sharing text coverage through the reference forward path |
| gemma4_moe (24) | fp16 | q8f16, hfq4 | dense-MoE text coverage through the reference forward path |
| minimax (10) | mq6 | mq4 | MoE experts MQ4/MQ6 only; **topk GPU-faults on gfx1151** (excluded there) |
| lfm2_moe (11) | mq6 | mq4 | hybrid short-conv + attention + MoE; experts MQ4/MQ6 only |
| mamba2 (15) | fp16 | q8f16 | State Spaces naming, loaded through the Mamba-capable Nemotron backend |
| qwen3_5 (5) | fp16 | q8f16, mq6, mq4, mq3 | + calibrated `qtip3-sim` cell (loads bf16) |
| qwen3_5_vl (5) | fp16 | q8f16, hfq4 | composite Qwen35 text + SigLIP vision artifact; synthetic vision forward feeds `forward_scratch_embed`; vision tensors included with `--include-vision --vision-quant hfq4` |
| qwen3_5_moe (6) | fp16 | q8f16, mq6, mq4, mq3 | 3D-stacked experts; was the cross-arch MoE NaN fix |

## Follow-ups / not yet added

The remaining dots.ocr gap is full decoded-image + prompt-template e2e coverage;
minimax has no gfx1151 baseline pending the topk fault fix.
