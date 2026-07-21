# DFlash Phase C — single-core attention numerics debug plan

Self-contained plan to get the single-core drafter non-causal cross-attention
kernel numerically correct on nix1's NPU (Gate C). Everything referenced here is
committed on `chaingun` (commits `ff48d840a`, `ae5f7f715`, `48a5c8e20`).

## Current state (what works, what doesn't)

- **Algorithm is correct** — `tools/npu/test_dflash_attention_npu.py --algo-only`
  reproduces the golden l0 attention at **cos = 1.000000** in numpy. So the math
  (per-head GQA, non-causal, full softmax, scale=1/√128) is not in question.
- **Kernel builds + RUNS** on nix1 (RyzenAI-npu1 / aie2). Dataflow executes; with
  `TensorTiler2D` taps on fill/drain, the output is **input-dependent** (different
  q-heads → different output), so real Q/K/V reach the core.
- **But the on-device output is wrong**: `--heads 4` gives cos ≈ 0.03–0.21 vs the
  golden (want cos > 0.99, max_abs < 0.1).

## Files

- Kernel: `tools/npu/dflash_attention_sc_bf16.cc`
  (plain bf16 row-major; one q-head/dispatch; `KV = [K(kv_len,128) | V(kv_len,128)]`
  in one buffer; `HIPFIRE_Q_LEN` / `HIPFIRE_KV_LEN` baked via `-D`).
- IRON build: `tools/npu/build_dflash_attention_sc.py`
  (`@iron.jit dflash_attn_head`, ObjectFifo depth=1, memtile `.forward()`,
  `TensorTiler2D` taps; `run_attn_head()` runs one head).
- Test: `tools/npu/test_dflash_attention_npu.py`
  (`--algo-only` = numpy vs golden; default = per-head on-device vs golden).

## Environment (nix1 fork toolchain — REQUIRED for @iron.jit)

```bash
export PATH=/opt/xilinx/xrt/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.cache/hipfire-npu-deps/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export PEANO_INSTALL_DIR=$HOME/mlir-aie-312/venv312/lib/python3.12/site-packages/llvm-aie
export PYTHONPATH=$HOME/mlir-aie-312/install/python:/opt/xilinx/xrt/python:$PYTHONPATH
PY=$HOME/mlir-aie-312/venv312/bin/python
GDIR=<golden>/rust   # rust_l0_{q_roped,k_roped,v,attn_out}.npy — see "Golden" below
$PY tools/npu/test_dflash_attention_npu.py --golden-dir "$GDIR" --heads 4
```

NPU JIT cache: `~/.npu/cache/<hash>/`. Each build ~2 min.

## Golden regeneration (if the scratch dir is gone)

The golden l0 tensors come from the F16 sidecar run of `dflash_ref_dump` with the
per-op dump env var set:

```bash
# build once (features deltanet):
cargo build --release -p hipfire-runtime --features deltanet --example dflash_ref_dump
DRAFT=~/.hipfire/drafts/Qwen3.5-9B.dflash.f16.hfq   # exists; else dflash_convert (no flag)
OUT=<somedir>
HIPFIRE_DFLASH_GOLDEN_DIR="$OUT/rust" ./target/release/examples/dflash_ref_dump \
    "$DRAFT" --block 16 --ctx 32 --out "$OUT"
# golden l0 tensors land in $OUT/rust/rust_l0_*.npy  → GDIR=$OUT/rust
```

Shapes: `rust_l0_q_roped [16, 32*128]`, `rust_l0_k_roped`/`rust_l0_v [48, 8*128]`,
`rust_l0_attn_out [16, 32*128]`. ctx=32, block=16, tot=48, 32 q / 8 kv heads.

## Hypothesis 1 (do FIRST): @iron.jit does not rebuild on `.cc` edits

Symptom that flagged this: head-0 output stayed **bit-identical** (cos 0.06506,
max_abs 2.237e+01) after rewriting the dot-product in the `.cc` AND `rm -rf
~/.npu/cache`. If the JIT design hash keys only on the Python design + CompileTime
args (not the ExternalFunction `.cc` bytes), every `.cc` edit silently reuses a
stale object — which would make all compute-side debugging meaningless.

**Confirm it:**
1. Add an obvious wrong-answer change to the `.cc` (e.g. `O[...] = 0` at the end),
   rebuild WITHOUT clearing cache, run `--heads 1`. If output is unchanged → cache
   is stale; if it goes to zero → cache is fine (skip to Hypothesis 2).
2. Inspect `~/.npu/cache/*/` for the compiled `.o`/`.cc` and whether its mtime/hash
   tracks edits. Check `aie.iron.ExternalFunction` / `compile_external_kernel` and
   the jit `compilabledesign.py` hashing (under `~/mlir-aie-312/install/python/aie/
   utils/compile/jit/`) for whether `source_file` content is hashed.

**Fix options (pick the cleanest that works):**
- Force-rebuild each run: `rm -rf ~/.npu/cache` (already done — if that DOESN'T
  invalidate, the cache dir isn't the only cache; look for an in-package or
  `~/.cache` JIT cache too).
- Put a content hash of `dflash_attention_sc_bf16.cc` into a CompileTime arg or the
  design name so edits change the design hash.
- Pass the `.cc` via `source_string=` (read the file in Python) instead of
  `source_file=`, if the string is hashed but the path isn't.

Do not trust any compute fix until a deliberately-wrong `.cc` change is observed to
change the on-device output.

## Hypothesis 2: compute / layout bug in the `.cc`

Once edits reliably rebuild, bisect with a passthrough then narrow:

1. **Layout passthrough**: make the kernel `O[i] = Q[i]` (copy Q→O), run, and in the
   test compare `npu` to the fed `q[:,h,:]` (NOT attn_out). cos≈1 ⇒ Q layout + O
   drain correct. Repeat `O[qi] = K[qi]` / a V element to confirm the `KV=[K|V]`
   split and offsets (`V = KV + kv_len*HEAD_DIM`).
2. **Scores only**: write `O[qi][0..kv_len] = scores[ki]` (pre-softmax, ×scale) and
   compare to numpy `q@k.T*scale` per head. Validates the dot (`aie::mul` +
   `reduce_add` over the 8 lane-vectors) and the K layout.
3. **Softmax**: dump the post-softmax weights; compare to numpy. Check `exp_f32`
   (exp2-domain polynomial) and the max/sum bookkeeping.
4. **AV**: with correct weights, check the `aie::mac(accum<accfloat>, V_bf, wv_bf)`
   accumulate + `accum.to_vector<bfloat16>()` store. Suspect the bf16 weight
   broadcast precision and the accfloat accum tag.

Likely culprits (in order): the dot reduction (`aie::reduce_add` semantics over a
`vector<float,16>` — confirm it sums all 16 lanes to a scalar), the `KV` split
offset, and bf16 `mac` accumulate. Compare each stage to the numpy reference which
is already exact (`test_dflash_attention_npu.py --algo-only`).

## Success criteria (Gate C)

- `test_dflash_attention_npu.py --golden-dir $GDIR` (all 32 heads): every head
  cos > 0.99, worst max_abs < 0.1 vs `rust_l0_attn_out`.
- Then generalize `kv_len` for ctx ∈ {block-only=16, short, 512} (rebuild per
  kv_len; kv_len is a `-D` compile-time constant) and re-validate — Gate C wants
  ctx ∈ {block-only, short, 512}.

## After Gate C

- Loop all 32 heads in `run_attn_head` (or one dispatch over heads) and wire the
  attention into the block body (Phase D). Attention stays its own dispatch.
- The 8-col npu2 path (`build_qwen3_dflash_attention.py`, compiles for aie2p) is an
  alternative for halo; it needs a host runner staging the bucketed Q/KV/O layout.

## Fast resume checklist

1. `--algo-only` still cos=1.0 (sanity).                                  [~1s]
2. Hypothesis 1: prove edits rebuild (wrong-answer test).                 [~2min]
3. Passthrough Q→O, then scores, then softmax, then AV.                   [~2min each]
4. Full `--heads 0` PASS, then kv_len sweep. Update the plan's Gate C.
