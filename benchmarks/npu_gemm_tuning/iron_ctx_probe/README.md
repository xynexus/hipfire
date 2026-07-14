<!-- SPDX-License-Identifier: Apache-2.0 -->
# IRON multi-context stability probe

**One-day go/no-go for the top-down (compiler-managed graph) NPU arm.**

## Why this exists

The resident hipfire path (R119–R121 in `../../../docs/npu/NPU-RESULTS.md`) is
blocked by a *stochastic* failure: some **fresh hardware contexts** return the
whole projection output as zeros — R120: *"six other fresh contexts return the
known whole-output zero symptom."* That is the one thing keeping the fast
N128/N1280 schedule from becoming the resident default.

hipfire creates and programs those contexts by hand, through direct **amdxdna DRM
ioctls** with hand-rolled BD/DMA scheduling. The strategic question for the
"attack from both directions" plan is:

> Is the zero symptom caused by *our* hand plumbing, or is it a property of a
> fresh amdxdna context that any stack would hit?

This probe answers it by creating the fresh contexts through the **IRON /
MLIR-AIE + XRT** stack instead of by hand, and measuring the zero rate.

## What it does

Builds one trivial multi-column *add-one* kernel at the real projection footprint
(default `256 x 768` i32, fanned across `--cols` columns, streamed as many small
`8 x 256` tiles so multiple shim/mem DMA BDs and many output-DMA tasks are
programmed per context — the R67/R120 regime). Each column writes
`out = in + (col+1)`, a distinct nonzero constant, so a stuck column is
identifiable and a zero output is unambiguous.

It then runs that kernel in **N fresh hardware contexts** and tallies
`{correct, zeros_whole, zeros_partial, mismatch, error}`.

- `--mode process` (default): each trial is a fresh OS process ⇒ fresh
  `pyxrt.device` + fresh `pyxrt.hw_context`. Closest match to how the resident
  path sees "fresh contexts."
- `--mode inproc`: N fresh `hw_context`s inside one process (cheaper; isolates
  context-fresh from process-fresh).

## Run it

```bash
# from this directory, with the mlir_aie venv active (python3.14 / ~/.venv)
python3 ctx_stability_probe.py build                 # compile once (~tens of s)
python3 ctx_stability_probe.py run --trials 64       # 64 fresh-process contexts
python3 ctx_stability_probe.py run --trials 300 --mode inproc
```

Knobs: `--cols 8 --rows 256 --width 768 --tile-h 8 --tile-w 256 --fill 1000`.
Push `--cols` up (more columns ⇒ more per-context BD programming) if the default
does not surface the symptom — R120 zeros appeared on wide multi-column output.

`pyxrt` is picked up from `/opt/xilinx/xrt/python` automatically. Uses the **NPU
only**, so it does not take the hipfire GPU lock.

## Reading the result

Exit code `0` = stable (GO), `2` = unstable (investigate), `1` = harness error.

| Outcome | Meaning | Action |
|---|---|---|
| **0 zero contexts** | XRT-managed fresh contexts are stable; the R120 zeros are self-inflicted by our direct-ioctl BD plumbing. | **GO** on the top-down arm: let IRON own context setup / BD allocation / program-fit. The manual R60→R121 pain is exactly what disappears. |
| **Nonzero zero rate** | The instability follows the amdxdna context, not our hand plumbing. | Top-down *multi-context* won't rescue it. Pivot to a **single-context whole-model graph** (see Phase 2) and/or file the fresh-context race against amdxdna. |

Either way it is the single most decision-relevant fact for the strategy, bought
for ~one day.

## Preliminary result (2026-07-14, halo)

Runs at the default shape (8 columns × 8×256 i32, one 8 KiB object per column,
two shared BOs, ~16 shim BDs):

| mode | contexts | correct | zero/partial |
|---|---:|---:|---:|
| inproc  | 200 + 1000 | 1200 (100%) | 0 |
| process |  40 |  40 (100%) | 0 |

**1240/1240 fresh contexts correct — the XRT/IRON context-creation mechanism does
NOT reproduce the R120 zero symptom at this load.** That is a genuine negative
result (rules out any zero rate above ~0.3% at 95% confidence): the bare act of
registering an xclbin, creating a fresh `hw_context`, and dispatching is not
stochastically flaky.

**Caveat — not yet conclusive.** This kernel is deliberately *light*. Two aie2p
limits were hit while building it, and both are the exact walls the resident path
fights, surfaced here at **compile** time instead of as runtime zeros:

- **Static BD-ID pool:** the first design enqueued one DMA per tile; at ~24
  tiles/column it failed to build with `aie.dma_bd … Free called on BD chain with
  unassigned IDs`. The R120 zeros appear on a much heavier N128/N1280, 40-block
  schedule — i.e. deep in this BD-pressure regime the light probe stays under.
- **5-data-argument DPU regmap (R59):** per-column BOs failed at `group_id(8)`;
  the probe had to adopt the production two-shared-BO ABI to run at all.

So the probe clears the *mechanism* but not the *heavy schedule*. To decide
whether the zeros are a runtime manifestation of the BD-ID limit, push toward the
R120 regime (memtile chaining / larger effective tiles that stay compilable) or
go straight to Phase 2, which exercises a real heavy whole-model graph.

## Phase 2 — escalate to a real compiler-generated whole-model graph

The synthetic kernel isolates context init. To test the *real* regime — and the
single-context escape hatch — use the vendored IRON Llama-3.2-1B app, whose
**decode path runs the entire model as one fused full-ELF in a single context**
(`self.decode.fused()` = 16 blocks + final norm + final linear):

    ../../../third_party/IRON/iron/applications/llama_3.2_1b/

1. Point it at local weights (HF snapshot is on `/srv/huggingface`):
   ```bash
   # model.safetensors + tokenizer.model expected under /srv/llama3.2-1b
   ls /srv/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/*/
   ```
2. Wrap `llama_npu.py` in the same fresh-process loop this probe uses: run it K
   times with a fixed prompt/seed and **diff the generated token stream** across
   runs. Stable identical output across many fresh processes ⇒ the single-context
   fused-ELF graph is context-stable, which is the concrete evidence that the
   top-down single-context path escapes the R120 zeros. Divergence/zeros ⇒ the
   race reaches even a compiler-managed single context.

Phase 1 tells you whether to go top-down at all; Phase 2 tells you whether
multi-context or single-context is the right top-down shape.

### Phase 2 attempt status (2026-07-14) — env-blocked, not concept-blocked

Getting `llama_npu.py` running surfaced two environment issues, one fixed and one
open:

1. **FIXED — duelling LLVMs.** torch's ROCm `libLLVM.so` and mlir_aie's
   `libAIEAggregateCAPI.so` both register the same `llvm::cl` options and segfault
   when torch imports first (llama_npu.py imports torch at line 14, the aie stack
   at line 26). Workaround proven: import the aie/MLIR stack *before* torch — see
   `scratchpad/run_llama.py` (imports `aie.dialects.aie` first, then runpy's the
   app). With that, it gets past imports into per-operator MLIR generation.
2. **OPEN — mlir_aie version skew.** The vendored `third_party/IRON` operators do
   `from aie.iron.placers import SequentialPlacer`, but the py3.14 venv's mlir_aie
   (`~/.venv`, the one with a 3.14 `pyxrt`) has no `aie.iron.placers`. That module
   exists only in older installs (`/srv/Public/sadara/.venv` py3.12 tree, and the
   uv archive caches) whose interpreters lack a matching `pyxrt`. So no single
   environment currently has {IRON-compatible mlir_aie *with* placers} + {matching
   pyxrt} + {torch/tiktoken}. Reconciling that (pin the placers-era mlir_aie
   against a 3.14 pyxrt, or rebuild pyxrt for 3.12) is a real but bounded task.

### Phase 2 update (2026-07-14) — env RECONCILED; fused-ELF blocked by a toolchain codegen bug

The environment mismatch is solved, non-destructively (`~/.venv` untouched). The
vendored IRON llama app was pinned to older toolchain versions than `~/.venv` has:

- IRON `requirements.txt` pins `mlir_aie==0.0.1.2026033104+e4f35d6` (March) and
  `llvm-aie==21.0.0.2026062201+7820460e` (June Peano). `~/.venv` had the May
  mlir_aie (dropped `aie.iron.placers`, changed `resolve_program` to take
  `device_name` instead of a `Placer`) and the July Peano.
- Both pinned versions are already extracted in the uv cache with cp314 ABIs:
  - mlir_aie March → `~/.cache/uv/archive-v0/g8BAAQbkdDCqINSV/mlir_aie` (has placers)
  - llvm-aie June  → `~/.cache/uv/archive-v0/WW__PzgdMSz4Uz7c/llvm-aie`

**Working recipe** — durable launcher: `phase2_llama/run_llama.sh` (sets the
overlay env and imports `aie.iron` before torch for LLVM `cl::opt` ordering). It
prepends the pinned March mlir_aie + points `PEANO_INSTALL_DIR` at the pinned June
Peano, both from the uv cache, over an untouched `~/.venv`.

With that, **all 13 real aie2p llama operators compile** — GEMMs (M2048×K2048×{N2048,
N512,N8192,N32256}), RMSNorm, SiLU, RoPE, ElementwiseAdd/Mul — and the app runs
end-to-end up to the final fused-decode ELF. Those xclbins+insts live in
`third_party/IRON/iron/applications/llama_3.2_1b/build/` and are a ready, heavier
stability payload than the synthetic add-one kernel here.

**Remaining blocker (NOT env, NOT Peano):** the fused whole-model single-context
ELF (`aiecc --generate-full-elf`) fails NPU lowering with
`'aie.dma_bd' op Buffer argument must be a constant aie.buffer, a runtime sequence
input argument, or a (chain of) subview(s)/cast(s) … with constant offsets and
strides equal to one` on a `memref<2048xbf16>` transfer. Identical with the July
and the pinned-June Peano, so it is a codegen limitation in the March mlir_aie's
`--generate-full-elf` lowering path, not a version mismatch.

**Strategic upshot — both top-down shapes hit a compile-time wall, not runtime zeros:**
- multi-context heavy schedule → static BD-ID pool exhaustion (Phase-1 finding);
- single-context whole-model fused ELF → `aie.dma_bd` lowering-verifier failure.
Neither silently zeros; the compiler stops you loudly.

### Real-operator stability (Phase-2 salvage) — `phase2_llama/real_op_probe.py`

Dispatching a compiled **real** llama operator (no fused ELF needed) in N fresh
contexts and checking self-consistency (bit-identical + non-zero output):

| operator | shape | contexts | correct | distinct hashes | verdict |
|---|---|---:|---:|---:|---|
| `GEMM_M2048_K2048_N2048` bf16 | 2048×2048×2048, 8 col, 3×8 MB BOs | 200 | 200 (100%) | 1 | **STABLE** |

Output is a uniform computed value (0.125 for all-1.0 inputs — a real scaled
transform, not a no-op or zeros), bit-identical across all 200 fresh contexts.
This is the heavy-real-operator stability number the synthetic add-one kernel
couldn't reach (it hit the BD-ID compile wall first). Combined with Phase-1's
1240/1240 synthetic contexts: **XRT-managed fresh contexts are stable across both
light-synthetic and heavy-real operators — the R120 zeros are self-inflicted.**

### Non-fused per-op decode — END-TO-END WHOLE-MODEL RUN WORKS (2026-07-14)

Since the fused ELF is the only broken piece and every individual operator
compiles + dispatches, the whole model can run through the per-op path instead.
`LLAMA_SKIP_FUSED_DECODE=1` (patch `phase2_llama/llama_npu_skip_fused_decode.patch`,
3 env-gated guards; default unset = original behavior) skips the fused-decode
build and routes decode through re-prefill (`use_kv_cache=False`; prefill pads to
`max_seq_len=2048`, so a growing seq_len ≤2048 reuses the same M2048 operators).

Run (via `phase2_llama/run_llama.sh`, which passes through `LLAMA_SKIP_FUSED_DECODE`):

    LLAMA_SKIP_FUSED_DECODE=1 phase2_llama/run_llama.sh --prompt-len 13 --num-tokens 5

Result — the full Llama-3.2-1B (16 layers + final norm + output head) runs
end-to-end on the aie2p NPU and generates **coherent** text:

    prompt "SCENE I. King"  ->  "SCENE I. King John's Palace.\nEnter"

(King John = a real Shakespeare play; "…Palace. Enter" is correct stage-direction
form.) Prefill-only (`--num-tokens 1`) → "SCENE I. King John". **The
compiler-generated whole-model graph runs coherently via per-op dispatch,
bypassing the fused-ELF codegen bug entirely.** This is the Phase-2 signal: the
single-context *fused* ELF is blocked, but the compiler-managed per-op whole-model
path is real and works.

#### Making it faster — `LLAMA_MAX_SEQ_LEN`, and the dispatch-bound floor

Re-prefill decode reprocesses the whole *padded* sequence each step, so sizing the
prefill operators to just cover the sequence is a direct win. `LLAMA_MAX_SEQ_LEN`
overrides the hardcoded 2048 (must be a multiple of 512 — the prefill attention
GEMM `attn_scores` needs M/N multiples of 256/512):

| max_seq_len | prefill (s) | decode (tok/s) | output |
|---:|---:|---:|---|
| 2048 | 2.475 | 0.408 | coherent |
| 512  | 1.147 | 0.977 | coherent (identical) |

    LLAMA_SKIP_FUSED_DECODE=1 LLAMA_MAX_SEQ_LEN=512 phase2_llama/run_llama.sh --prompt-len 13 --num-tokens 5

**2.4× faster decode, same coherent output.** But the scaling is sub-linear (4× less
compute → 2.4× faster), which pins the ceiling: re-prefill fires **~160 NPU
dispatches per token** (16 layers × ~10 ops) and is **dispatch-overhead-bound, not
compute-bound** — matching this repo's headline NPU finding (feeding/overhead, not
compute). 512 is the minimum valid `max_seq_len`, so this is the re-prefill floor
(~1 tok/s).

**Consequence for a "real" M1 KV-cache decode:** the decode operators exist as real
per-op M1 building blocks (`GEMV`, `RoPE(angle_rows=1)`, `StridedCopy` cache-append)
— but a hand-orchestrated M1 decode still fires ~the same ~160 dispatches/token, so
being dispatch-bound it would *not* dramatically beat the ~1 tok/s re-prefill floor.
The only path to fast decode (≫ few tok/s) is **fewer dispatches per token — i.e.
fusion** (partial or the full ELF). That is exactly why the app fuses decode, and
exactly what the `--generate-full-elf` codegen bug blocks. So "faster non-fused
decode" has a hard ceiling at the dispatch floor; real speed needs the fusion path
fixed (upstream) or a partial-fusion executor.

### ✅ FUSED-ELF BUG FIXED FROM SOURCE — fused decode at 8 tok/s (2026-07-14)

Built the toolchain from source and fixed it. Root cause (found with a source
`aie-opt`): the full-ELF path materializes every fused-`main`-device DMA buffer as
a `memref.view` slice of one flat byte-arena block argument
(`%v = memref.view %arena[%byte_off][] : memref<Nxi8> to memref<2048xbf16>`), but
`AIEX::traceSubviewToBlockArgument` (in `lib/Dialect/AIEX/Utils/AIEUtils.cpp`) only
followed `memref.cast`/`reinterpret_cast`/`subview` — **not `memref.view`** — so
the buffer failed to trace to its block arg and `aie-dma-tasks-to-npu` rejected it.

**Fix** (`phase2_llama/mlir-aie-memref-view-fullelf-fix.patch`, ~18 lines): add a
`memref::ViewOp` case to the tracer — accumulate the constant byte offset and
follow the source. That's the whole fix.

Build (from source, latest mlir-aie `d098819482` + LLVM `068c6c5c`):
- LLVM+MLIR: `utils/build-llvm.sh` (Release, RTTI-off, ccache); mlir-aie:
  `utils/build-mlir-aie.sh llvm/build`. Both under `/home/sadara/build/mlir-aie`.
- Rebuild after the patch: `ninja aiecc` (aiecc.py delegates to the compiled
  `build/bin/aiecc`, so that binary — not just `_aie.so` — must be relinked).

Wire-in: `LLAMA_FULLELF_AIECC=/home/sadara/build/mlir-aie/build/bin/aiecc` routes
ONLY the `--generate-full-elf` step to the fixed aiecc (via
`AieccFullElfCompilationRule`, env-gated, default off); per-op xclbins stay on the
pinned March aiecc. The source-built latest aiecc lowers the March-generated MLIR
fine.

Result — full Llama-3.2-1B with the **real fused decode ELF**:

| decode path | tok/s | note |
|---|---:|---|
| re-prefill, max_seq_len=2048 | 0.41 | dispatch-bound |
| re-prefill, max_seq_len=512  | 0.98 | dispatch-bound floor |
| **fused ELF (bug fixed)**    | **8.05** | 1 dispatch/token, coherent |

    LLAMA_FULLELF_AIECC=/home/sadara/build/mlir-aie/build/bin/aiecc LLAMA_MAX_SEQ_LEN=512 \
      phase2_llama/run_llama.sh --prompt-len 13 --num-tokens 5

**~8× over the best re-prefill, ~20× over the original**, same coherent output
("SCENE I. King John's Palace. Enter"). The dispatch-bound wall is gone — the whole
model runs in one dispatch/token. Confirms the earlier analysis: fusion was the
only path past the dispatch floor, and the blocker really was this one toolchain
bug. Fix committed in the mlir-aie tree (`ba6a9bebbc9`).

### (b) fused-ELF root cause (historical) — fusion path, not dma_bd shape, not env, not Peano

The failing `aie.dma_bd` in `fused_op_fused.mlir` uses a runtime_sequence block-arg
buffer with dims `[<1,0>,<1,0>,<1,0>,<N,1>]`. The **standalone** `ElementwiseMul`
operator emits the *identical* dma_bd pattern on a runtime_sequence block arg and
**compiles cleanly**; the same pattern only fails once `--generate-full-elf` fuses
the per-op devices into one module. So the defect is isolated to the March
mlir_aie's fused-full-ELF lowering (buffer-provenance verification across the
fused multi-device module), not the operators, the dma_bd shape, the Peano
version, or the env overlay. It is a native-pass bug — not patchable from our
layer. Workarounds: (i) an mlir_aie version whose full-ELF lowering is fixed
(the newer May build changed the IRON-facing API, so the app would need porting),
or (ii) run decode as separate per-op dispatches instead of one fused ELF (an
app-level change). The individual operators all work, as the table above shows.

#### Deeper root cause + dead-ends (2026-07-14)

Traced further. The failing op is in a special fused device named **`main`** that
aiecc reports as having **"no cores to compile"** — it is a pure DMA-orchestration
device whose `aie.runtime_sequence` stitches the per-op sub-ELFs together. Its
`aie.dma_bd(%arg388)` where `%arg388` is a runtime_sequence block arg is rejected
by the NPU-lowering verifier — even though the verifier's own rule explicitly
permits "a runtime sequence input argument". So this is a genuine verifier/lowering
bug in the full-elf `main`-device path, not malformed input.

Dead-ends tried (all reproduce the identical `dma_bd` error at `main`):
- **Newer aiecc** — routed only the `--generate-full-elf` step to the May mlir_aie
  aiecc (via `AieccFullElfCompilationRule` + clean-PYTHONPATH subprocess). Same
  bug; May did **not** fix it. (An earlier standalone "success" was a false
  positive — it died on a missing `.o` *before* reaching the verifier.)
- **`--expand-load-pdis`** is not the cause: aiecc runs both a normal and an
  "expanded" NPU lowering and **both** fail on the same `dma_bd`.
- **Peano version** (June vs July) is irrelevant — the verifier runs pre-Peano.

The verifier is compiled C++ inside the mlir_aie wheel (no source), so it cannot
be patched here, and the bug spans every locally-available mlir_aie version. Fixing
it properly is an **upstream mlir-aie** task (source build + report). The tractable
local path to fast decode is therefore **partial fusion** — fuse per *layer* (~10
ops → 1 dispatch) via the normal xclbin path, giving ~16 dispatches/token (~10×
headroom over the ~160 of re-prefill) **without** invoking the broken full-model
`--generate-full-elf` `main`-device lowering.
```
