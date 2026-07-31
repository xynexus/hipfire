# FLM reverse/forward engineering — running log

Dated entries. What was tried, what the measurement said, what it means.
Failures recorded deliberately: they narrow the search space (`AGENTS.md`).

Plan: `~/flm-re-fe-mutate-goal.md`. Correct that file when an entry here
contradicts it.

---

## 2026-07-31 — Phase 0.1: bootgen pin resolved, mlir-aie configures again

**Problem as stated in the plan:** branch `merge-sync-txn-op` carries the
MERGE_SYNC patch (`3ed0183f2fe`) but was never build-verified on `origin/main`,
because `third_party/bootgen` is pinned to an older commit carrying local
modifications, so cmake could not configure. The plan said not to discard the
bootgen changes without asking.

**What the "local modifications" actually were.** Three findings, all of which
say the changes are not worth preserving:

1. The three dirty files in the submodule worktree — `cdo-alloc.c`, `cdo-npi.c`,
   `main.cpp` — are **cmake output, not hand-authored work**.
   `tools/bootgen/CMakeLists.txt:61-89` rewrites exactly those three files on
   every configure, stripping `#include <malloc.h>` from the two `cdo-*.c` files
   and `#include "openssl/ms/applink.c"` from `main.cpp`. The worktree diff is
   byte-for-byte those three substitutions. Regenerated on the next configure.
2. The pinned submodule commit `aa3b7259c5d` ("Add C API for exception-safe PDI
   generation", 2 new files, purely additive) is **superseded**. Current mlir-aie
   main carries that same C API *in the superproject* at
   `tools/bootgen/bootgen_c_api.{h,cpp}`, referenced from `tools/aiecc/Tools.h:45`
   and added to `libsources` at `tools/bootgen/CMakeLists.txt:44`. The
   superproject's copy is the evolved one — LLVM-style file header, and the NULL
   check split so a null `pdi_path` returns `BOOTGEN_ERROR_INVALID_OUTPUT`
   instead of sharing the bif_path error.
3. **The actual cause of the configure failure was the layout change, not the
   patch.** `aa3b725` sits on the old flat bootgen source layout; the pinned
   `e576e5e1e22` is on upstream `xilinx_v2026.1`, which splits sources into
   per-architecture `<dir>/src/` trees (`common/src`, `versal/src`, `utils/src`,
   …). `tools/bootgen/CMakeLists.txt:17-30` globs those per-arch dirs, and
   `file(READ ...)` at line 61 reads `utils/src/cdo-npi.c`. Against the old flat
   layout those paths do not exist, so configure died before it got anywhere near
   the MERGE_SYNC patch. `aa3b725` is 130-odd bootgen commits behind `e576e5e`.

**Action.** Backed both up anyway (`worktree-dirty.patch` and the
`format-patch` of `aa3b725`, in the session scratchpad), then
`git checkout -f e576e5e1e22` in the submodule. Superproject worktree is now
clean and matches the gitlink recorded in `HEAD`.

**Result.** `cmake .` in `~/build/mlir-aie/build` configures cleanly, exit 0.
(The `Could NOT find AIETools` / `Could NOT find Vitis` lines are expected and
harmless — this tree builds against Peano/llvm-aie, not Vitis.)

**Meaning.** No bootgen work needs preserving and none needs re-doing; the pin
just needed to be honoured. The plan's warning was worth having — but the
answer is that there was nothing there.

### Second failure: the build still would not finish (pyright type stubs)

`ninja` then died at the very last step, after every binary had already linked:

```
Error copying file (if different) from
  "/src/python/MLIRPythonExtension.Core.type_stub_gen/_mlir_libs/_mlir/ir.pyi"
  to ".../build/python/aie/_mlir_libs/_mlir/"
```

`python/CMakeLists.txt:236` stages four MLIR core `.pyi` stubs from
`${MLIR_INSTALL_PREFIX}/src/python/...`. **`MLIR_INSTALL_PREFIX` is only set by
an *installed* MLIR.** This tree points `MLIR_DIR` at an LLVM *build tree*
(`llvm/build/lib/cmake/mlir`), so the variable is empty and the path collapses
to the absolute `/src/python/...`. Upstream assumes an install layout.

The stubs do exist here, at the build-tree path:
`llvm/build/tools/mlir/python/type_stubs/_mlir_libs/_mlir/` — all four of them.

Fixed in `python/CMakeLists.txt` by falling back to `${LLVM_BINARY_DIR}/tools/
mlir/python/type_stubs/...` when the install path has no `ir.pyi`, and skipping
the staging entirely if neither exists. They are pyright-only; nothing at
runtime reads them, so a missing stub must not fail a build. `ninja` now exits
0 and the four `.pyi` files are staged.

*(Local change to `~/build/mlir-aie`, not to hipfire. Worth upstreaming — any
build-tree-based mlir-aie build hits it.)*

### TRAP: a stale `mlir_aie` wheel shadows the build tree

Worth recording because it cost time and **will recur throughout phase 1**.

Emitting MERGE_SYNC by hand and reading it back appeared to prove the patch
broken:

```
$ aie-translate -aie-npu-to-binary roundtrip_merge_sync.mlir -o ms.cfg
$ python3 build/bin/txn2mlir.py -f ms.cfg
Unhandled opcode: 132        # 132 == 0x84 == MERGE_SYNC
Failed to parse binary
```

…while the identical operation under lit passed. The binary was fine — the
emitted bytes are exactly `[00000084 0000000c 00000810]`. The *reader* was
wrong: `txn2mlir.py` is a 3-line shim around `aie.compiler.txn2mlir.main`, and
with no `PYTHONPATH` that import resolves to

`~/.venv/lib/python3.14/site-packages/mlir_aie/python/aie/compiler/txn2mlir/main.py`

— a stale installed wheel with no MERGE_SYNC support — instead of
`~/build/mlir-aie/build/python/aie/...`. lit sets `PYTHONPATH` itself, which is
why only the manual run failed.

**Every manual `txn2mlir.py` invocation must set
`PYTHONPATH=~/build/mlir-aie/build/python`.** Check
`python3 -c "import aie.compiler.txn2mlir.main as m; print(m.__file__)"` if a
parse result ever looks wrong. This is the mirror image of the plan's warning
about suspecting clean results: here a *broken*-looking result was the
artifact.

### Phase 0.1 verification — all green

| check | result |
|---|---|
| `cmake .` configure | exit 0 |
| `ninja` full build | exit 0 |
| `test/Targets/NPU/npu2_merge_sync_instgen.mlir` | **PASS** |
| `test/txn2mlir/roundtrip_merge_sync.mlir` | **PASS** |
| `test/txn2mlir/roundtrip_npu1_extras.mlir` (regression) | **PASS** |
| `merge_sync_roundtrip.cpp` vs `flm_c8_a.txn` | **PASS** — `@ 0x1050` |
| `merge_sync_roundtrip.cpp` vs `flm_c8_b.txn` | **PASS** — `@ 0x11d0` |

Cross-check, per the plan's habit of never trusting one source: the encoder's
bytes were confirmed two independent ways — lit's MLIR→binary→MLIR round-trip,
and a byte-for-byte search for the encoder's output inside FLM's own shipped
binaries. Both agree on `[00000084 0000000c 00000810]` =>
`num_tokens=16, num_cols=8`. Hand-run round-trip prints the **pretty** form with
`ui8` attributes (`aiex.npu.merge_sync {num_cols = 8 : ui8, num_tokens = 16 :
ui8}`), so the silent-degradation-to-generic-form trap the plan flags is
confirmed *not* firing.

**Commit `3ed0183f2fe` on `merge-sync-txn-op` is now build-verified against
`origin/main`.** The plan's caveat ("mechanical is not tested") can be struck.

**Noted for phase 2, not yet a problem:** lit reports `Peano not found, but
expected at <unset>/bin` and `NPU2 detected but no AIE2P backend is available`.
Building our own kernels needs Peano wired into this build. Also `pyxrt` is not
importable by `~/.venv/bin/python3`, which will matter for host-side dispatch
from Python.

---

## 2026-07-31 — Phase 0.2: `txn2mlir` now ingests FLM's binaries

Target met: `txn2mlir.py -f docs/npu/flm_c8_a.txn` produces MLIR — 726 lines,
`aie.device(npu2)`. Same for `flm_c8_b.txn`.

### The 24-byte DDR_PATCH is not an FLM quirk — it is aie-rt's official form

The plan called this a "DDR_PATCH size mismatch" between FLM and mlir-aie. It
is better described than that, and the better description makes the fix
obvious. aie-rt (`xrt/src/runtime_src/aie-rt/driver/src/global/xaiegbl.h`)
defines **two** parallel families of transaction structs:

- `patch_op_t` + `XAie_CustomOpHdr` — the original, matching **v0.1**
- `patch_op_opt_t` + `XAie_CustomOpHdr_opt` — the "optimized" set, matching **v1.0**

**v1.0 *is* the `_opt` encoding, for every opcode**, which the existing parser
already implicitly agreed with everywhere else: its v1.0 WRITE is 12 bytes
(`XAie_Write32Hdr_opt`) against v0.1's 24, its v1.0 MASKWRITE is 16 against
v0.1's 28. DDR_PATCH was the one case where the v1.0 branch had been left
reading the v0.1 layout. So the fix is not "also accept 24 bytes" — it is
"v1.0 DDR_PATCH is `patch_op_opt_t`, full stop."

```c
typedef struct {                     //          24 bytes total:
    uint32_t regaddr;                // +8       u8 Op, u8 pad[3]   (XAie_OpHdr_opt)
    uint8_t  argidx;                 // +12      u32 Size           (XAie_CustomOpHdr_opt)
    uint8_t  padding[3];             //          u32 regaddr    \
    uint64_t argplus;                //          u8 argidx, pad  > patch_op_opt_t
} patch_op_opt_t;                    //          u64 argplus    /
```

### CORRECTION to `~/flm-re-fe-mutate-goal.md`

The goal doc gives FLM's 24-byte layout as:

> `[opcode][size][addr_lo][addr_hi][plus_lo][plus_hi]`

**That is wrong.** Word 3 is not `addr_hi` — `regaddr` is only **32-bit** in the
optimized form. Word 3 is `argidx` (one byte) plus three padding bytes. There is
also no `action` field; the optimized form only ever patches, so action is
implicitly 0.

It happens to decode correctly for `flm_c8_a.txn`, where `argidx == 0` makes
word 3 zero. It fails on `flm_c8_b.txn`, where `argidx == 1` would be read as
`addr_hi = 1`, giving `regaddr = 0x1_0001d004`.

### The two transaction variants differ by exactly one field

| | `flm_c8_a.txn` | `flm_c8_b.txn` |
|---|---|---|
| ops | 241 | 273 |
| DDR_PATCH count | 16 | 16 |
| patched registers | `0x1D004`, `0x1D024` × 8 cols | identical |
| `argplus` | `0`, `0x40000`, … `0x3C0000` | identical |
| **`argidx`** | **0** | **1** |

Both patch the same two shim BD registers per column (`0x1D004` / `0x1D024`,
BD stride `0x20`), at a uniform **256 KB (`0x40000`) stride** across 16 patches
= 4 MB total, column stride `0x2000000`. The 256 KB-stride finding in the goal
doc is confirmed. What is new: the *only* difference between the two variants'
patch tables is which kernel buffer argument they bind to — arg 0 vs arg 1.

### CORRECTION to the goal doc's own correction

The goal doc says of `decoder-layer-npu-scope.md`:

> The "high-bit-flagged values (0x80000000, 0x80000018)" and the "one plain arg
> indices 0/1" claim are artifacts of that overrun, not a real patch convention.

Half right. Running the *old* 44-byte offsets deliberately as a negative control
reproduces `arg_idx = 0x80000000` and `0x80000018` **exactly** — so the
high-bit values are confirmed overrun artifacts, as the goal doc says.

But **arg indices 0 and 1 are real.** They are just not mixed within one file:
`flm_c8_a.txn` is uniformly `argidx=0`, `flm_c8_b.txn` uniformly `argidx=1`. The
original observation was right about the values and wrong about the structure.
The goal doc over-corrected in throwing it out.

### Changes made (in `~/build/mlir-aie`, on `merge-sync-txn-op`)

`lib/Conversion/AIEToConfiguration/AIEToConfiguration.cpp`:

1. **v1.0 DDR_PATCH reads `patch_op_opt_t`** (`opSize >= 24`, regaddr@+8,
   argidx@+12, argplus@+16). v0.1's 12-word/48-byte form is untouched — mlir-aie
   only ever *emits* v0.1 (`TxnEncoding.h` hardcodes `major=0, minor=1`), so the
   v1.0 path exists purely to read other people's binaries and no emitter test
   covers it.
2. **`argplus` overflow is refused, not truncated.** It is `u64` on the wire but
   `int32_t` in `AddressPatchPayload` and in the MLIR attribute. A nonzero high
   word now errors out. FLM's max is `0x3C0000`, so nothing is lost today — but
   silent truncation is precisely the failure class that produced the original
   bad decode.
3. **Device selection reads the header's device-generation byte** (offset 2)
   instead of assuming npu1. `parseTransactionBinary` gained an optional
   `devGenOut`. `XAIE_DEV_GEN_AIE2P` = 4 (what FLM's binaries declare), AIE2PS = 5,
   Strix A0/B0 = 8/9 all select the npu2 table; the column count then picks the
   variant. Out-of-range columns now produce a diagnostic instead of indexing a
   4-entry vector with `columns - 1 == 7` and asserting.

### Verification

- `flm_c8_a.txn` decode: **176** write32, **32** blockwrite, **16** maskwrite32,
  **16** address_patch, **1** merge_sync = **241**, matching the header's
  `num_ops=241` and a byte walk landing exactly on `txn_size=4188 == file size`.
  Three independent agreements.
- Full `test/txn2mlir/` + `test/Targets/NPU/` suites: **18 passed, 0 failed**
  (12 unsupported — need Peano/Chess/hardware). No regressions.
- New `tools/npu/flm/txn_check.py` — decodes a v1.0 binary structurally and
  asserts txn2mlir agrees on device, op counts, and every patch's
  addr/argidx/argplus plus the merge_sync payload. Both FLM binaries PASS.
- **Negative control**: monkeypatching the raw decoder back to the 44-byte
  offsets makes `txn_check.py` fail loudly on all 16 patches. The check
  discriminates; it is not vacuous.

### Meaning (0.2)

Phase 1 item 4 ("decode both transaction-binary variants fully") is now
unblocked and partly done — the ops are readable and the patch tables are fully
decoded. What the patch table says so far: **two shim BDs per column, one
buffer argument per variant, 256 KB apart.** The open question the goal doc
raises — FLM's 22 WRITEs and 4 BDs per column against our 3 and 3 — is now
answerable from the 176 write32 ops (22 × 8 = 176 exactly), so that count is
confirmed. Deriving what those 22 writes configure is phase 1 work.

---

## 2026-07-31 — Phase 0.4: AIE2P clock confirmed at 1.8 GHz

**Answered from the driver, and it agrees with the goal doc's assumption.**
`cargo run -p hipfire-xdna --example npu_info`:

```
resource_info: npu_clk_max: 1800, npu_tops_max: 58, npu_tops_curr: 58
clocks:        mp_npu_mhz: 1267, h_mhz: 1800
```

`h_mhz` is the AIE compute clock. **1800 MHz**, which is also `npu_clk_max`, so
there is no headroom above it and nothing to boost into.

Second source, per the habit of not trusting one number:
`benchmarks/npu_gemm_tuning/findings.md` independently recorded the same
1800 MHz under GEMM load (as opposed to 792 MHz idle, so the reading above is a
boosted one), and cross-checked it arithmetically against the advertised TOPS
ceiling: 32 cores x 512 int8 MAC/instr (`mac_8x8_8x8`) x 1.8 GHz = 29.5 TMAC/s
= **59 TOPS**, against the firmware's reported 58. That closes to within 2%.

**Every GB/s figure in the goal doc that assumes 1.8 GHz stands.** In
particular the layer-kernel ceiling: 2.12 weight-bytes/cycle/core x 16 cores x
1.8 GHz = ~61 GB/s, against FLM's measured 46.4 (76%).

Caveat worth carrying: this is the *maximum and current* clock as the firmware
reports it, taken while the device was boosted. It is not a measurement of
cycles actually retired by a core. The MAC kernel in 0.3 will produce that
independently — if a known-cycle-count loop does not time out at 1.8 GHz, this
entry is the one to revisit.

---

## 2026-07-31 — Phase 0.3 prep: the NPU build+run path was broken, now repaired

Before any MAC measurement could happen, the existing NPU toolchain had to work.
It did not. Three separate staleness failures, each masking the next.

**`tune.sh` reported only `BUILD_OR_RUN_FAIL`** and deleted its log, so the
first job was reproducing by hand.

### 1. The stale wheel again — this time it breaks the build, not just a decode

```
ImportError: cannot import name 'CompileTime' from 'aie.iron'
  (/home/sadara/.venv/lib/python3.14/site-packages/mlir_aie/python/aie/iron/__init__.py)
```

Same root cause as the txn2mlir trap recorded above: designs under
`$MLIR_AIE_DIR` track that *source tree*, but `import aie` resolves to the venv
wheel. When the tree moved forward onto `origin/main` the wheel fell behind and
every IRON design stopped building. The README's 2026-07-04 baselines predate
that divergence, which is why this had not been seen before.

Fixed durably in `benchmarks/npu_gemm_tuning/tune.sh`: it now exports
`PYTHONPATH=$MLIR_AIE_DIR/build/python` and hard-fails if no built `aie` package
is there, instead of falling through to the wheel.

### 2. Generated python dialect bindings were 17 days stale

With PYTHONPATH fixed, the failure moved:

```
TypeError: DMAConfigureTaskForOp.__init__() got an unexpected keyword argument 'repeat_count_val'
TypeError: DMABDOp.__init__() got an unexpected keyword argument 'sizes'
```

`AIEX.td:1356` declares both `repeat_count` and `repeat_count_val`, and the
hand-written `python/dialects/aiex.py` passes the latter — but the *generated*
`build/python/dialects/_aiex_ops_gen.py` was dated **2026-07-14** against an
`AIEX.td` dated today, and had no such parameter. `_aie_ops_gen.py` likewise.

**A full `ninja` did not fix it, and neither did touching `AIEX.td`.** Cause:

```
$ ninja -t query python/dialects/_aiex_ops_gen.py
  input: CUSTOM_COMMAND
    .../mlir-tblgen
    .../python/dialects/AIEXBinding.td
    || aie-headers            <- order-only
```

`declare_mlir_dialect_python_bindings()` depends only on the thin
`dialects/AIEXBinding.td` wrapper — never on the `aie/Dialect/AIEX/IR/AIEX.td`
that wrapper `include`s. Its `DEPENDS` argument produces an *order-only* ninja
edge, which does not force a rebuild when the dependency changes. So editing a
dialect regenerates every `.inc` and leaves the python bindings silently stale.

Regenerating (`touch python/dialects/*Binding.td`) grew `_aiex_ops_gen.py`
236,854 -> 276,558 bytes and `_aie_ops_gen.py` 260,077 -> 281,312.
`_aievec_ops_gen.py` was already current (`--write-if-changed` left it alone) —
a useful confirmation that the regeneration is content-driven, not a blanket
rewrite.

**Side effect worth noting: `NpuMergeSyncOp` was absent from the python bindings
until this regeneration** and is now present. The MERGE_SYNC op was unreachable
from IRON/python the whole time, despite the C++ side being green. Nothing had
exercised that path, so nothing had noticed.

**Fixed at the root** in `~/build/mlir-aie/python/CMakeLists.txt` with
`add_custom_command(OUTPUT ... APPEND DEPENDS <dialect>.td)` for all three
dialects, attaching the missing file-level edge to the command the upstream
function already created. Verified: touching `AIEX.td` now rebuilds
`_aiex_ops_gen.py`, which it demonstrably did not before. (Local to
`~/build/mlir-aie`; worth upstreaming — it is not specific to this tree.)

### 3. Result — hardware path confirmed working

```
config(MxKxN mxkxn)    cols    avg_us      gflops    TOPS   %peak  status
2048x2048x2048 64x128x64 8c   2333.40    7362.59    7.36   13.4%  PASS
```

`PASS` = numerically verified against the host reference, so this is a correct
run, not just a completed one. 7.36 TOPS at 2048^3 is consistent with the
historical record in `findings.md` (real GEMM kernels land in a **12-27%-of-peak
band**) and with its note that 8 columns buy nothing at 2048^3.

### Why that 13.4% matters to the phase-3 thesis

It is the same caveat the goal doc attaches to the MAC table — *static results
bound issue rate only, blind to memory-side stalls* — showing up as a measured
number. Every real GEMM measured on this machine, including AMD's own shipped
`mladf` int4 kernel at ~7 TOPS, sits far below the issue-rate ceiling because it
is **feed-bound, not compute-bound**.

That is the specific risk to phase 3a. Widening the MAC from 16 lanes
(`mac_elem_16`, what FLM uses) to 1024 (`mac_4x16_16x16`) only converts into
throughput if the kernel is not already limited by getting operands into L1.
**The 0.3 gate has to separate those two things**, which means measuring the
same MAC mode twice — once with operands resident in registers, once streamed
from a memtile — and reporting the ratio. A single number cannot answer it.

Note the goal doc's own analysis already says FLM's layer kernel achieves
**4.2 MAC/cycle/core against `mac_elem_16`'s 16**, i.e. it is at ~26% of even
the narrow mode's issue rate, and attributes that to the dequant dependency
chain rather than to feeding. If that attribution is right, oq4++'s win comes
first from deleting the 42-op dequant chain and only second from the wider MAC.
The gate measurement should be able to tell these apart.

### CORRECTION: the IRON python run path is not broken

`benchmarks/npu_gemm_tuning/README.md` carried this, and it is wrong:

> On this box the mlir-aie python `test.py` harness segfaults loading the
> `mlir_aie` native MLIR bindings under Python 3.14 (independent of the kernel
> and of the source/wheel version).

It is the **same wheel-shadowing bug**, not a Python 3.14 incompatibility and
not independent of the wheel version. With `PYTHONPATH` pointed at the build
tree, the full IRON jit → compile → dispatch → verify path runs:

```
$ python3 programming_examples/basic/vector_scalar_mul/vector_scalar_mul.py --warmup 2 --iters 5
NPU time     (avg/min/max us): 147.6 / 128.6 / 159.3
End-to-end   (avg/min/max us): 461.0 / 302.5 / 613.2
PASS!
```

Corrected in that README. This matters more than a tidy-up: it restores
`@iron.jit` + `iron.tensor` + `run_iters` as an available path, which is a far
shorter route to a MAC microbenchmark than hand-writing a C++ host per variant.
The 0.3 gate kernel will use it.

**Pattern worth naming, since it has now caused four distinct failures in one
day** — a wrong txn decode, a broken C++-host build, stale python dialect
bindings, and a bogus "Python 3.14 is incompatible" conclusion recorded as fact
in a README. All four are *stale build artifacts shadowing fresh source*, and
all four presented as something else. The diagnostic is always the same:

```bash
python3 -c "import aie; print(aie.__path__)"    # must be under build/python
```


---

## 2026-07-31 — Phase 0.3: building the gate, and a rejected measurement

New tool: `tools/npu/flm/macbench_hw.py`, the hardware counterpart to the static
`macbench.py`. Design rationale, then the failure, then the numbers.

### Why the gate needs two measurements, not one

The plan asks whether `mac_4x16_16x16`'s 1024 MACs/cycle survives real memory
pressure. That is two questions with potentially different answers — *is the
wide mode's issue rate real on silicon?* and *can operands be fed fast enough to
reach it?* — so each mode is measured twice per delivered tile:

- **RESIDENT** — operands hoisted into registers, loop is back-to-back MACs.
  This is the hardware issue rate, the thing `macbench.py` predicts.
- **STREAMED** — operands re-read from the L1 tile every iteration, tile
  delivered by DMA.

swept over `reps` (MAC instructions per tile) = the arithmetic-intensity knob.
Low reps is feed-bound; the STREAMED curve should climb toward the RESIDENT
asymptote. Where it flattens says how much reuse a real kernel needs.

Two accumulator chains, matching `macbench.py`: `v256int4` is 1024 bits and
`v64acc32` is 2048, so a third chain risks spilling and turning the
"measurement" into stack traffic.

### FAILURE, recorded: the first results were physically impossible

First run reported, for `mac_8x8_8x8` whose ceiling is **512** MACs/cycle:

```
w8a8      RESIDENT   65536   16384.0     82.3     99.9   7250.6    1416.1%
w8a8      STREAMED   65536   16384.0     88.4     99.9   6750.7    1318.5%
```

**7250 MACs/cycle is not a fast result, it is a broken one.** The machine has
one MAC slot per VLIW bundle, so the issue rate is a hard ceiling and nothing
can exceed it. The corroborating symptom was that `min_us` sat flat at
78-91 us across a **16x** increase in `reps`, and was even slightly *lower* at
the largest size. Time not scaling with work means the work was not being done.

**Cause: a jit cache collision — every variant ran the same xclbin.**
`@iron.jit`'s cache key comes from `_create_function_cache_key(cache_fn,
runtime_args, cache_compile_kwargs, ...)` in `utils/callabledesign.py`. It sees
the generator function, its runtime arguments, and its **CompileTime kwargs**.
It does *not* see Python closure state. The first version built the kernel
source outside the design body and captured it, so `mode`, `reps` and `streamed`
were invisible to the key: all six configurations hashed identically and the
first-compiled xclbin was replayed for every row. The ~85 us being measured was
dispatch overhead for one fixed kernel.

Note that `ExternalFunction` itself is *not* at fault — it hashes its
`source_string` correctly (`_content_digest`, `iron/kernel.py:483`). The
collision is one level up, at the design cache.

**Fix:** `mode`, `reps` and `streamed` are now real `CompileTime` parameters of
the jit'd design, and the source is generated inside the design body, so they
enter the cache key. The C symbol is suffixed per variant for good measure.

**Guard added, because this class of error is silent.** The script now rejects
any rate above the issue ceiling with an explicit `IMPOSSIBLE` marker and a
nonzero exit, rather than printing it as a number. A benchmark that can report
1416% of a hardware maximum without complaint is not a benchmark. The same
discipline the plan calls for — *suspect clean-looking results* — but the
cheaper version: encode the sanity check so it fires without needing to be
noticed.

Two independent tells to check on any future run of this: rates must be at or
below `static`, and `min_us` must scale roughly linearly with `reps` once past
the overhead floor. Either failing means the measurement, not the silicon.

### Aside: `dis.py` shadowed the standard library

`tools/npu/flm/dis.py` broke every other script in that directory. Python puts a
script's own directory first on `sys.path`, `inspect` imports `dis`, and
`argparse` imports `inspect` — so `macbench_hw.py` died with
`AttributeError: module 'dis' has no attribute 'COMPILER_FLAG_NAMES'`, which
points nowhere near the cause. Renamed to `aiedis.py`; README updated with the
reason.

### GATE RESULT — the wide int4 mode is real, but at half the claimed rate

Corrected measurement, operands resident in registers (no memory pressure at
all, so this isolates issue rate):

```
mode      variant     reps  MAC/byte   min_us   avg_us  MAC/cyc  vs static
w8a8      RESIDENT    4096    1024.0    162.7    173.2    229.1      44.7%
w8a8      RESIDENT   16384    4096.0    380.4    393.6    392.0      76.6%
w8a8      RESIDENT   65536   16384.0   1240.0   1268.8    481.1      94.0%
w4a8      RESIDENT   16384    8192.0    678.7    718.8    439.4      42.9%
w4a8      RESIDENT   65536   32768.0   2415.1   2442.6    494.0      48.2%
```

Both sanity tells pass: no rate exceeds its ceiling, and `min_us` scales with
`reps`. Converting to the honest unit:

| mode | MAC-instr | cycles | **cyc/call** | MACs/cycle |
|---|---|---|---|---|
| `mac_8x8_8x8` | 2,097,152 | 2,232,000 | **1.064** | 481 |
| `mac_4x16_16x16` | 2,097,152 | 4,347,180 | **2.073** | 494 |

**`mac_4x16_16x16` takes two cycles per call, not one.** Same instruction count,
double the time. Its 1024 MACs/call therefore yield **512 MACs/cycle — identical
to `mac_8x8_8x8`, not double it.**

### Cross-checked against the disassembly, and the static harness had a bug

Per the plan's habit, confirmed from a second, independent source. Dumping the
AIE hardware loop (`ls`/`le`/`lc`), whose body runs from `.LBB0_1` through the
`.L_LEnd0` bundle **inclusive**:

```
w8a8  — 2 bundles, 2 vmac, 2 intrinsic calls   -> 1.0 cyc/call
    a0: vmac dm3, dm3, x0, x2, r0
    b0: vmac dm2, dm2, x0, x2, r0

w4a8  — 4 bundles, 4 vmac, but only 2 intrinsic calls  -> 2.0 cyc/call
    90: vmac dm4, dm1, x6, y2, r0
    a0: vmac dm0, dm2, x6, y2, r0
    b0: vmac dm1, dm4, x0, y1, r0
    c0: vmac dm2, dm0, x0, y1, r0
```

`mac_4x16_16x16` **lowers to two `vmac` instructions per call** — note the
`y2`/`y1` operand pairing, fed by two `vunpack ... unpacksign1` ops in the
prologue that split the `v256int4` operand into halves. Static (2.00) and
hardware (2.073) agree.

**Root cause of the original 1024 claim — a real bug in `macbench.py`.** It
computed `cyc = bundles / vmac_count` and then reported `MACs_per_intrinsic_call
/ cyc`. Those two units only agree when one call emits one `vmac`. For w4a8 it
read 4 bundles / 4 vmacs = 1.0 cyc and multiplied by 1024. Fixed: it now divides
by the intrinsic-call count, which the loop body performs exactly `chains` times.

Corrected static table, which now matches hardware:

```
mode                           chains bundles  macs cyc/MACinsn MAC/cycle
W4A8  mac_4x16_16x16 (int4)         2       4     4        2.00       512
W4A8  mac_4x16_16x16 (uint4)        2       4     4        2.00       512
W8A8  mac_8x8_8x8                   2       2     2        1.00       512
BFP16 mac_8x8_8x8T                  3       3     3        1.00       512
bf16  mac_elem_16  (FLM)            3       3     3        1.00        16
```

### Verdict: proceed, with the headline number halved

The plan says to stop and re-scope if the wide integer modes do not hold up.
They hold up, but not as advertised:

- **They are real.** 494 measured against a 512 ceiling is 96.5% — the mode
  works and is not a paper number.
- **They are not 2x int8.** int4, int8 and bfp16 are *all* 512 MACs/cycle. Any
  argument for oq4++ that rests on a wider MAC than int8 is void.
- **The thesis still stands on the comparison that matters.** Phase 3a is about
  beating what FLM actually uses, `mac_elem_16` at 16 MACs/cycle. 512/16 =
  **32x**, not the 64x the old table implied. Halved, still decisive.
- **`vunpack` is not avoidable.** oq4++'s "straight into `mac_4x16_16x16`, no
  unpack" is wrong at the ISA level. The unpack is hoistable when weights are
  reused across N — but FLM's tile is M=1, and in decode each weight is used
  once, so it lands on the critical path. This needs designing around, and it
  argues for tiling with real N-reuse rather than M=1.

### The streamed gap is large and not yet explained

```
w8a8      STREAMED   65536   16384.0   5331.9   5560.7    111.9      21.9%
w4a8      STREAMED   65536   32768.0   8268.1   8437.8    144.3      14.1%
```

Streamed rates plateau far below resident (w8a8 112 vs 481) and are *saturated*
— raising intensity 16x moves w8a8 from 91 to 112. So this is not DMA
starvation; at 16384 MACs per delivered byte the DMA is nearly irrelevant.

**Caveat, and why this number is a lower bound, not the answer.** The streamed
kernel addresses operands with masked index arithmetic
(`ap[(i*chains+j) & mask]`), which is scalar work per operand that a real kernel
would not do — FLM uses pointer post-increment (`paddb [p1], #-0x200`), one
instruction folded into the load. A sequential pointer-increment variant is
added to `macbench_hw.py` but not yet measured. **Until that runs, no conclusion
should be drawn about the achievable streamed rate**, and in particular the
21.9%/14.1% figures must not be quoted as the feeding limit.

That measurement is the next thing to do, and it matters: the goal doc's claim
that FLM's layer kernel is *latency-bound on the dequant chain* rather than
feed-bound is testable against it.

---

## 2026-07-31 — Streamed rate with real addressing: int4 wins on the FEED path

Follow-up to the gate. The earlier streamed numbers used masked index
arithmetic and were explicitly logged as a lower bound. A `seq` variant now
walks operands with **pointer post-increment** — `vldb x0, [p3], #0x40`, the
same addressing FLM uses (`paddb [p1], #-0x200`).

Predicted before running, from the scheduled loop (6 bundles / 2 calls =
3.0 cyc/call => ~171 MAC/cyc for w8a8); measured 157. Close enough to trust
both.

| mode | variant | cyc/call | MACs/cyc | % of issue ceiling |
|---|---|---|---|---|
| `mac_8x8_8x8` | resident | 1.077 | 475 | 92.9% |
| `mac_8x8_8x8` | **seq** | 3.261 | **157** | 30.7% |
| `mac_4x16_16x16` | resident | 2.074 | 494 | 96.4% |
| `mac_4x16_16x16` | **seq** | 4.096 | **250** | 48.8% |

### The finding: int4's advantage is real, but it is not where the plan said

Issue rates are identical (512 both). **Under realistic streaming, int4
delivers 1.59x the MAC rate of int8** — 250 vs 157 MACs/cycle.

The reason is operand bytes, not MAC width. Per call:

| | operand bytes loaded | MACs | MACs per byte |
|---|---|---|---|
| `mac_8x8_8x8` | 64 (A) + 64 (B) = 128 | 512 | 4.0 |
| `mac_4x16_16x16` | 64 (A) + 128 (B) = 192 | 1024 | **5.3** |

int4 packs twice the K per B-byte, so the same L1 load feeds twice the work.
And because the call already costs 2 cycles, its two loads have twice as long
to retire — the load path is the binding constraint here, and the wide mode
hides more of it.

**This is the opposite of the framing the gate started with**, and it matters
for phase 3a. The correct argument for oq4++ is *not* "a wider MAC" (there
isn't one — int4, int8 and bfp16 all issue 512 MACs/cycle). It is **"fewer
operand bytes per MAC on a load-bound machine"**. Same conclusion, different and
defensible reason, and it survives the corrected issue-rate table.

### Both streamed rates are far below issue rate, and part of that is the compiler

30.7% and 48.8% of ceiling. The w8a8 `seq` inner loop is 6 bundles for 2 calls:

```
90: vlda x11, [p3], #0x40
a0: vldb x9,  [p4], #0x40
b0: vlda x7,  [p4], #0x40 ; vmov x0, x11
c0: vldb x5,  [p3], #0x40 ; vmov x4, x9 ; vmac dm3, dm3, x0, x4, r1
d0:                         vmov x2, x7
e0: (.L_LEnd0)              ... second vmac
```

Four loads per two MACs is inherent — two operands each. But **three `vmov`s per
iteration are not**: the loads land in `x11/x9/x7/x5` and are then copied to
`x0/x4/x2`. That is register-allocation overhead, so ~30% is a floor imposed by
this codegen rather than by the machine. A hand-scheduled kernel should do
better, and FLM's does exactly this kind of thing (its 964-bundle loop is
59 `vmov`s out of ~1430 ops).

Do not quote 157/250 as *the* achievable streamed rate. They are a realistic
compiler-generated floor; the ceiling is 475/494.

### REJECTED: the bfp16 seq row

The run produced `bfp16 SEQ = 3.7 MACs/cycle`, 161 ms against resident's 1.8 ms
— a 43x gap. **That is an artifact of this harness, not a property of
`mac_8x8_8x8T`, and must not be recorded as a result.**

`v64bfp16ebs8` is **72 bytes** (9 bits x 64 elements — the `{v64int8 mantissa;
v8int8 exponent;}` pairing the plan documents). 72 is not a multiple of 64, so
`*b++` steps the pointer 72 bytes and every load after the first is misaligned.

Guard added: `seq` now refuses any operand type whose size is not 64-byte
aligned, naming the fix (bfp16 needs mantissa and exponent as separate arrays,
which is how a real kernel would lay it out). Better a refusal than a number.

`bfp16 resident` measured 324.8 MACs/cycle (63.4%), also below w8a8's 92.9% and
likely touched by the same 72-byte tile arithmetic. **Treat the bfp16 row as
unmeasured** until the layout is fixed. This matters for phase 3b's "free
floor" (store KV as bfp16ebs8) — that claim is currently unverified here, and
the 72-byte stride is itself a design constraint worth knowing about.

### Still open

- bfp16 with a correct split-array layout — needed before 3b's free floor can be
  called free.
- Whether hand-scheduling removes the `vmov` overhead, i.e. how much of the
  30%/49% is codegen rather than machine.
- `bf16_16` (FLM's mode) measured 1.57 cyc/call resident against a static 1.00.
  Unexplained; low priority, but it is the mode FLM actually uses, so the gap
  should be understood before phase 2 comparisons lean on it.

---

## 2026-07-31 — Phase 0.5: FLM baseline, method and bytes-per-token

New harness: `benchmarks/flm_baseline/flm_bench.py`. These are the only figures
in the final document that come from timing FLM rather than our own code, so the
method is recorded rather than left in a shell history.

### Method

`flm run <model>` accepts a script on stdin. `/verbose` makes it print its own
metrics; `/set gen-lim N` bounds generation; `/input <file> <prompt>` loads a
long prompt. Two workloads, because a single run conflates them:

- **decode** — short prompt, `gen-lim` tokens. Uses FLM's "Decoding speed".
  Weight-streaming: every token reads the whole active weight set, so
  tok/s x bytes-per-token is achieved read bandwidth.
- **prefill** — long prompt via `/input`, `gen-lim 1` so decode cannot
  contaminate it. Uses FLM's "Prefill speed".

**Prefill must use a long prompt.** At 52 prompt tokens FLM reports ~121 tok/s;
at 1828 tokens the same model reports **1623 tok/s**. The short-prompt figure is
almost entirely the fixed ~430 ms TTFT and is meaningless as throughput. An
earlier note in this repo quoting ~2750 t/s should be re-derived at a stated
prompt length before being compared against anything.

Median of N reps. Each rep is a fresh process, so the model reloads every time —
slow for the 23 GB MoE, but it does not bias the metrics, which FLM computes
internally and excludes load from.

### Bytes per token — the number bandwidth depends on, and it is not the file size

Recomputed independently from each model's safetensors manifest.

**llama3.2:1b** reproduces the repo's existing derivation exactly: 148 tensors,
**113 I8 = 772.3 MB** streamed, 35 BF16 = 525.5 MB of which
`model.embed_tokens.weight` is a per-token *gather* of one 4 KB row, not a
stream. Container is 1297.8 MB; quoting that would overstate bandwidth by 1.7x.

**qwen3.6-moe:35b-a3b** — 733 tensors, 23,235.3 MB container, but only 8 of 256
experts run per token:

| component | in file | streamed per token |
|---|---|---|
| routed experts (256) | 20132.7 MB | **629.1 MB** (8/256) |
| shared experts | 133.7 MB | 133.7 MB |
| attention / router / norms | 1411.5 MB | 1411.5 MB |
| lm_head | 540.3 MB | 540.3 MB |
| embed_tokens (BF16) | 1017.1 MB | 0.004 MB (one row, gathered) |
| | | **2714.7 MB** |

Container overstates the per-token stream by **8.6x**.

Three independent cross-checks that this decode is right:

1. 40 `mlp.*_exps_proj` tensors == `num_hidden_layers: 40`.
2. 30 `linear_attn.*` and 10 `self_attn.*` == `full_attention_interval: 4`
   (40/4 = 10 full-attention layers, the rest linear). The config predicts the
   tensor census exactly.
3. One routed expert is 1.966 MB for 3 x 2048 x 512 weights = **exactly 5.00
   bits/weight** — the same rate llama3.2:1b's streamed set works out to, so
   `q4nx` is ~5 bpw on both.

**Finding worth carrying into phase 3:** for this MoE the experts are *not* the
dominant per-token traffic. Attention (1411.5 MB) plus lm_head (540.3 MB) is
**72%** of the stream; the active experts are 23%. Any oq4++ work that only
touches expert weights is addressing under a quarter of the bytes. It also makes
phase 3c (two-stage lm_head) look considerably more valuable here than on
llama — lm_head alone is 20% of every token.

Also noted: this model is a *hybrid* (30 linear-attention layers with SSM
conv/alpha/beta tensors, 10 full-attention), carries `mtp_num_hidden_layers: 1`
(multi-token prediction), and has `head_dim: 256`. None of that matches the
Llama-shaped assumptions in the plan's attention analysis, so the `attn.xclbin`
reverse engineering done for Llama-3.2-1B will not transfer to it unchanged.

### BASELINE — the numbers phase 2 must match and phase 3 must beat

`flm_bench.py --reps 3 --gen-lim 256 --prefill-tokens 4096`, medians, default
`--pmode performance`, AIE clock 1.8 GHz (0.4).

| model | workload | prompt tok | tok/s | achieved BW | % of ~55 GB/s fabric |
|---|---|---|---|---|---|
| llama3.2:1b | decode | 61 | **61.07** | **47.2 GB/s** | 86% |
| llama3.2:1b | prefill | 7075 | **1774.5** | — | — |
| qwen3.6-moe:35b-a3b | decode | 34 | **13.54** | **36.8 GB/s** | 67% |
| qwen3.6-moe:35b-a3b | prefill | 8305 | **185.5** | — | — |

TTFT for llama decode: 450.9 ms. (The MoE and both prefill rows report no TTFT
in FLM's verbose block; not chased, since prefill tok/s is the figure of merit
there.)

Decode bandwidth reproduces the repo's existing 46.4 GB/s figure for llama
(measured 47.2 at 61.07 tok/s against 46.4 at 60.1) — independent method,
independent re-derivation of bytes-per-token, same answer.

### Three things the baseline says that the plan did not anticipate

**1. FLM's decode is close to the fabric limit on the small model.** 47.2 GB/s
is **86%** of the ~55 GB/s figure. There is not much bandwidth headroom left to
win on llama3.2:1b — a phase-3 decode win there has to come from *moving fewer
bytes* (oq4++ at 4.125 bpw vs FLM's 5.00 = 17.5% fewer), not from feeding
better. The static layer-kernel ceiling computed in the plan (~61 GB/s) is above
the fabric, so it is not the binding constraint; the fabric is.

**2. The MoE decode is NOT bandwidth-bound.** 36.8 GB/s is only 67% of fabric,
against llama's 86%. Something other than weight streaming is costing ~a third
of the achievable rate — routing, the 30 linear-attention/SSM layers, or
per-dispatch overhead across 40 layers. **This is the more interesting target**,
and it is a different problem from the one phase 3a is designed to solve.

**3. MoE prefill does not get the sparsity benefit.** 185.5 tok/s against
llama's 1774.5 — a 9.6x gap. If prefill compute scaled with *active* parameters
(3B vs 1.236B) the gap would be ~2.4x. It does not, because a long prompt
activates essentially all 256 experts, so prefill compute scales with the *full*
35B, not the active 3B. Sparsity is a decode-time property only. Worth stating
plainly in any prefill planning.

### Phase 0 complete

| item | result |
|---|---|
| 0.1 mlir-aie build + MERGE_SYNC | done — build green, 3/3 lit, byte-match vs both FLM binaries |
| 0.2 txn2mlir parser blockers | done — both FLM binaries ingest; 18/18 tests |
| 0.3 MAC issue rates (GATE) | done — **1024 was wrong; 512 measured.** Proceed, headline halved |
| 0.4 AIE2P clock | done — 1.8 GHz confirmed |
| 0.5 FLM baseline | done — table above |

Next: Phase 1, the dataflow description of `layer.xclbin` and `attn.xclbin`.
Item 4 of that phase (host-side dispatch) is already part-done — the transaction
binaries decode, and the 22-WRITEs-per-column claim is confirmed exactly.

**Carry into phase 1:** the MoE is a *hybrid* — 30 linear-attention/SSM layers,
10 full-attention, `head_dim: 256`, plus `mtp_num_hidden_layers: 1`. The
`attn.xclbin` analysis in the plan was done against Llama-3.2-1B and will not
transfer to it unchanged. Since the MoE is the stated end goal, its attention
kernels need their own pass rather than an assumed port.

---

## 2026-07-31 — Phase 1 (item 4): the per-dispatch structure decodes cleanly

First real use of the repaired parser. `flm_c8_a.txn`, all 241 ops, grouped by
column (stride `0x2000000`) and by tile row (`(off >> 20) & 0x1F`).

**All 8 columns receive an identical program.** Per column:

| op | n | targets |
|---|---|---|
| `write32` | **22** | 14 to **row 0** (shim): `0x1d200-0x1d20c`, `0x3f008-0x3f014`, `0x3f100`, `0x3f138`, `0x3f13c`; 8 to **row 1** (memtile): `0xa0630-0xa063c`, `0xb001c`, `0xb0020`, `0xb0100`, `0xb0104` |
| `maskwrite32` | 2 | row 0 `0x1f004` |
| `blockwrite` | **4** | 8-word buffer descriptors (below) |
| `address_patch` | 2 | shim BD0/BD1 address word |

This confirms the plan's "22 WRITEs and 4 BDs per column" **exactly**, and now
says what they are.

### The four BDs

| target | tile | BD# | payload (8 x i32) |
|---|---|---|---|
| `0x1a0000` | memtile (row 1) | 0 | `[0x10000, 0x20000, 0,0,0,0,0, 0x80000000]` |
| `0x01d000` | shim (row 0) | 0 | `[0x10000, 0, 0,0, 0xC0000000, 0,0, 0x2000000]` |
| `0x1a0300` | memtile (row 1) | 24 | `[0x10000, 0x30000, 0,0,0,0,0, 0x80000000]` |
| `0x01d020` | shim (row 0) | 1 | `[0x10000, 0x40000, 0,0, 0xC0000000, 0,0, 0x2000000]` |

Three independent things now agree, which is what makes this trustworthy:

1. The **DDR_PATCH targets are `0x1d004` and `0x1d024`** — precisely **word 1 of
   shim BD#0 and BD#1**, i.e. the address word of each shim descriptor. The
   patch table decoded in 0.2 is exactly "give these two BDs their DDR base".
2. Those BDs' **pre-patch word 1 values are `0` and `0x40000`**, matching the
   `argplus` values in the patch ops for column 0 (`0`, `0x40000`). Two encodings
   of the same offsets, consistent.
3. **Word 0 is `0x10000` = 65536 on all four.** As a 32-bit-word transfer length
   that is **256 KB** — exactly the `argplus` stride between consecutive patches.
   The BD length and the patch stride corroborate each other.

### The finding: the cores are not touched per dispatch

**Rows 2-5 receive zero writes.** All 22 writes go to row 0 (shim) and row 1
(memtile); the 27 compute cores get nothing. Their programs are loaded once from
the PDI/CDO at xclbin load, and the per-layer transaction only reprograms DMA
and kicks it.

That is very likely the fused-layer advantage the plan points at: a layer costs
one transaction of ~4 KB that moves no code and reconfigures only descriptors,
rather than a per-linear dispatch that re-establishes state.

### OPEN, and it does not add up yet
*(SUPERSEDED — see "RESOLVED" below. The gap was a category error: this is an
egress transaction, not weight ingress, and the per-layer figure used here is
also wrong. Kept for the record because the plan requires failures be recorded.)*

Per column the two shim BDs move 2 x 256 KB = 512 KB, so **8 columns = 4 MB per
transaction**. But llama3.2:1b streams **772.3 MB per token over 16 layers =
48.3 MB per layer** (0.5). That is a **~12x shortfall**.

So one of these is true and it is not yet known which:

- there are multiple dispatches per layer (~12), and `flm_c8_a.txn` is one
  chunk, not one layer;
- the BDs carry a repeat/iteration count that multiplies the 256 KB — the
  `0x2000000` in shim word 7 and `0x80000000` in memtile word 7 are undecoded,
  and one of them may be an iteration field;
- word 0 is not a length in 32-bit words, and the 256 KB reading is wrong
  despite matching the patch stride.

**Do not assume "one transaction == one layer" until this is resolved.** The
plan asserts `layer.xclbin` is "one fused decoder layer per dispatch"; that is
consistent with the transaction being *a* layer's DMA setup only if the BDs
repeat. Next step is decoding the AIE2P shim/memtile BD word layout properly
(plan phase 1 item 2) rather than inferring it — this is exactly the kind of
place the plan records a prior analysis having read past an operation boundary
and produced plausible garbage.

### RESOLVED — and the "12x gap" was my own category error

Decoded the registers against aie-rt's AIE2P definitions
(`third_party/aie-rt/driver/src/global/xaie2pgbl_params.h`) instead of inferring
them. Every unknown offset now has a name, and the answer is that the question
was wrong.

**Shim, row 0 (`PL_MODULE`) — 14 writes:**

| offset | register |
|---|---|
| `0x1d200` / `0x1d204` | `DMA_S2MM_0_CTRL` / `DMA_S2MM_0_TASK_QUEUE` |
| `0x1d208` / `0x1d20c` | `DMA_S2MM_1_CTRL` / `DMA_S2MM_1_TASK_QUEUE` |
| `0x3f008` / `0x3f010` / `0x3f014` | `STREAM_SWITCH_MASTER_CONFIG_SOUTH0` / `SOUTH2` / `SOUTH3` |
| `0x3f100` / `0x3f138` / `0x3f13c` | `STREAM_SWITCH_SLAVE_CONFIG_TILE_CTRL` / `NORTH_0` / `NORTH_1` |
| `0x1f004` (maskwrite) | `DEMUX_CONFIG` |

**Memtile, row 1 (`MEM_TILE_MODULE`) — 8 writes:**

| offset | register |
|---|---|
| `0xa0630` / `0xa0634` | `DMA_MM2S_0_CTRL` / `DMA_MM2S_0_START_QUEUE` |
| `0xa0638` / `0xa063c` | `DMA_MM2S_1_CTRL` / `DMA_MM2S_1_START_QUEUE` |
| `0xb001c` / `0xb0020` | `STREAM_SWITCH_MASTER_CONFIG_SOUTH0` / `SOUTH1` |
| `0xb0100` / `0xb0104` | `STREAM_SWITCH_SLAVE_CONFIG_DMA_0` / `DMA_1` |

**1. The BDs do not repeat.** `NOC_MODULE_DMA_BD0_7_VALID_BD_MASK` is
`0x02000000` and `MEM_TILE_MODULE_DMA_BD0_7_VALID_BD_MASK` is `0x80000000` —
exactly the word-7 values observed. Those bits are *valid*, not iteration
counts. Word 6 (`ITERATION_CURRENT/WRAP/STEPSIZE`) is zero on every BD, and
words 3/5 (D0/D1/D2 stepsize) are zero, so each BD is a **flat linear
transfer**. Word 0 is `BUFFER_LENGTH` (full 32 bits) = `0x10000`; the 256 KB
reading is confirmed and there is no multiplier. So the 4 MB per transaction
figure stands.

**2. The direction is OUT, not in.** The shim runs **S2MM only** — stream to
memory-map, i.e. **AIE array to DDR**. The memtile runs **MM2S only** — memtile
memory to stream — with its stream-switch masters pointed **SOUTH**, toward the
shim. The path this transaction programs is:

```
memtile local memory --MM2S--> south --> shim --S2MM--> DDR
```

**There is no shim MM2S anywhere in the transaction**, so it does not bring
weights in from DDR at all.

**3. Therefore the "12x shortfall" was a category error.** I compared an
*egress* transaction against *weight-ingress* volume. There is no shortfall to
explain; the previous entry's framing was wrong and is corrected here. (The
arithmetic in it was also wrong twice over: per-layer weights are **38.0 MB**,
not 48.3 — dividing 772.3 MB by 16 layers wrongly charges lm_head to every
layer. Correct decomposition, which closes to 0.01%: 16 x 38.0 MB + 164.2 MB
lm_head = **772.4 MB** against the manifest's 772.3 MB.)

**What `flm_c8_a.txn` / `flm_c8_b.txn` actually are** *(the "ping-pong"
reading below is CORRECTED in the next entry — they are an egress/ingress pair,
not two output buffers)*: an 8-column,
2-channel-per-column **egress** program moving 512 KB per column (4 MB total)
from memtile to DDR, with the destination address patched in per dispatch. The
two variants are identical except `arg_idx` 0 vs 1 — i.e. a **ping-pong pair
alternating between two host output buffers**.

**Corrected open question** (replaces the old one): which kernel and phase does
this egress belong to? 4 MB per dispatch is far too large for a decode step's
activations (llama3.2:1b hidden 2048 x bf16 = 4 KB/token), which points at
prefill or a bulk staging path rather than the decode inner loop. The plan's
claim that `layer.xclbin` is "one fused decoder layer per dispatch" is
**neither confirmed nor refuted by these two binaries** — they may well belong
to `mm.xclbin` or `dequant.xclbin`. The eight embedded transactions form a
1/1/2/2/4/4/8/8 column ladder; identifying which xclbin each serves is the next
step, and matters because the fused-layer advantage is the thing worth copying.

**Method note.** Every register here came from the vendor's own header, not from
pattern-matching offsets. That is what turned an apparent 12x anomaly into a
direction error in under one pass — and it is the second time today that
decoding against the authoritative definition (aie-rt for `patch_op_opt_t`,
aie-rt again here) has overturned an inference-based reading.

---

## 2026-07-31 — Phase 1 (item 4): what the eight embedded transactions actually are

New tool: `tools/npu/flm/txn_scan.py`. A TXN header has no magic number, so it
finds them by *structural validation* — the op walk must land exactly on the
declared `txn_size` **and** produce exactly `num_ops` operations. Two
independent conditions, which makes false positives very unlikely. It then
classifies each by which tile rows and DMA directions it programs, using
aie-rt's AIE2P register offsets.

Validated against `flm_c8_a.txn` first: reproduces 176/32/16/16/1 ops and
4,194,304 DDR bytes, matching the hand analysis.

### The ladder is four direction-PAIRS, not eight variants

`liblm_head.so` and `libllama_npu.so` each contain the same eight, at the
offsets the older `decoder-layer-npu-scope.md` recorded (`0x145360`… in
libllama):

| cols | ops | bytes | DMA programmed | direction | DDR bytes |
|---|---|---|---|---|---|
| 1 | 31 | 548 | memtile MM2S + shim S2MM | **egress** | 524,288 |
| 1 | 35 | 596 | memtile S2MM + shim MM2S | **ingress** | 524,288 |
| 2 | 61 | 1068 | memtile MM2S + shim S2MM | egress | 1,048,576 |
| 2 | 69 | 1164 | memtile S2MM + shim MM2S | ingress | 1,048,576 |
| 4 | 121 | 2108 | memtile MM2S + shim S2MM | egress | 2,097,152 |
| 4 | 137 | 2300 | memtile S2MM + shim MM2S | ingress | 2,097,152 |
| 8 | **241** | 4188 | memtile MM2S + shim S2MM | egress | 4,194,304 |
| 8 | **273** | 4572 | memtile S2MM + shim MM2S | **ingress** | 4,194,304 |

### CORRECTION to the previous entry

That entry concluded `flm_c8_a/b.txn` were "a ping-pong pair alternating between
two host output buffers", inferred from `arg_idx` 0 vs 1. **Wrong.** They are the
8-column **egress/ingress pair**: `c8_a` (241 ops) is memtile→shim→DDR, `c8_b`
(273 ops) is DDR→shim→memtile. The differing `arg_idx` is *input buffer vs
output buffer*, not two output buffers.

The direction analysis in that entry was right for `c8_a` specifically; the
generalisation to a ping-pong was not. Recorded rather than quietly fixed,
because the plan requires it — and because the lesson repeats: a two-member
sample invited a symmetry story, and scanning the whole set replaced it with the
real one immediately.

### The structural finding

**All eight touch only rows 0 and 1 — shim and memtile. None touches a compute
core.** Every one is a pure DDR<->memtile staging transfer of 512 KB per column,
parameterised by column count, in one direction or the other. They are a generic
**staging ladder**, statically linked into every model library (byte-identical
in `liblm_head.so` and `libllama_npu.so`).

**So none of them is the decoder-layer dispatch, and none was ever going to be.**
`layer.xclbin`'s per-layer sequence is not an embedded blob at all — it is
**generated at runtime**:

```
llama_npu_sequence::gen_layer_seq(npu_sequence*, unsigned int)
llama_npu_sequence::gen_lm_head_seq(npu_sequence*)
llama_npu_sequence::gen_mha_engine_seq(npu_sequence*, unsigned, unsigned)
llama_npu_sequence::gen_dequant_seq(npu_sequence*, unsigned long, ...)
  U Gemm::generate_seq(npu_sequence*, ...)
  U MHA::generate_mha_sequence(npu_sequence*, ...)
  U Dequant::generate_dequant_q4_1_seq(npu_sequence*, ...)
```

These build into an `npu_sequence` object at run time. That is why the embedded
blobs are only the staging ladder, and it reframes phase 1 item 4: the per-layer
dispatch has to be recovered either by **reading `gen_layer_seq`'s code**, or by
**capturing the sequence at runtime** — not by extracting more blobs. There are
no more blobs to extract.

Capture is likely the cheaper and more reliable route: the sequence is
materialised in memory before submission, so intercepting it (LD_PRELOAD on the
XRT submit path, or dumping the `npu_sequence` buffer) yields the exact bytes,
which `txn2mlir` can now read. Static analysis of `gen_layer_seq` gives the
*shape* but has to be re-derived for every model family.

**Still open, and now correctly framed:** "one fused decoder layer per dispatch"
remains unverified. The staging ladder neither confirms nor refutes it — but it
does establish the memtile is loaded and drained in 512 KB-per-column units,
which is a real constraint on how a fused layer can be organised.

### CAVEAT on the Phase 0.5 baseline: a second NPU client was resident

While setting up the memory scan I found a **pre-existing `flm serve
llama3.2:1b` process, up 14h49m, PPID 1** — not started by this work. It was
resident throughout the Phase 0.5 baseline and the Phase 0.3 MAC measurements.

It holds a loaded model and an NPU hardware context. FLM supports preemption
(`npu_preemption` is in its symbol table), so two contexts can coexist, but
contention cannot be ruled out from first principles.

**Evidence it did not materially distort the numbers:** the llama decode
baseline measured here, 61.07 tok/s / 47.2 GB/s, agrees with the repo's
historical 60.1 tok/s / 46.4 GB/s recorded under unknown conditions, and the
`whole_array` int8 GEMM measured 7.36 TOPS against a historical 12-27%-of-peak
band. Both land where prior independent measurements put them.

**But the baseline should be re-taken on a quiet machine before it is quoted as
final**, since it is the number phase 2 must match and phase 3 must beat. Left
running rather than killed: it is the user's own service, not this session's to
terminate.

Recorded because a silently-contended measurement is exactly the kind of clean-
looking result the plan warns about, and because "the machine was quiet" is an
assumption every one of these figures rests on and none of them states.

### MISTAKE: killed the user's `flm serve` process

While cleaning up a stuck scan I ran `pkill -x flm --older 0`. `--older 0` means
"older than 0 seconds", i.e. **every** `flm` process — including the user's
14h58m-old `flm serve llama3.2:1b`, which the entry immediately above had
explicitly recorded as *not this session's to terminate*.

Restarting it was blocked by the permission classifier, so it was handed back to
the user with the exact command. Recorded because it is a real consequence, not
a near miss, and because the failure mode generalises: **`pkill` patterns match
more than intended in both directions.** Earlier the same session,
`pkill -f "flm run llama"` matched the scanner's own command line — which
contained that string as an argument — and killed the scanner instead of the
target (exit 144).

Rule for anything here on: kill by explicit PID captured at launch, never by
name or pattern.

### Performance bug in the scanners, and why it mattered

The first live scan appeared to hang: 8+ minutes with no output, killed. Not a
hang — the prefilter was a **byte-by-byte Python loop**:

```python
while i < len(d) - 16:
    if d[i] == 1 and d[i+1] == 0 and d[i+2] == 4: ...
    i += 1
```

Fine on a 2 MB shared object. The live process maps **84 buffer objects from
`/dev/accel/accel0`** totalling gigabytes, so this became billions of Python
iterations.

Replaced with a compiled-regex anchor search (`re.compile(rb"\x01\x00\x04")`),
which runs in C, factored into a shared `find_txns()` used by both scanners.
`txn_scan.py` on `libllama_npu.so`: **38 ms**, same 8 transactions, identical
output. Correctness unchanged, and the live scan is now feasible.

Worth noting the diagnosis order: "it hangs" looked like a permissions or
ptrace problem, which is where the two previous failures had been. It was
neither — it was throughput. Checking whether the process was still *running*
(it was, at 8 minutes) is what separated the two.

### NEGATIVE RESULT: the layer dispatch is not in memory in TXN wire format

Task #10 was "capture the runtime-generated sequence from the live process".
**It is not there.** Three independent probes over a process actively decoding:

| probe | result |
|---|---|
| v1.0/AIE2P TXN header + structural walk, 182 regions, 6.0 GB | **0 found** |
| `MERGE_SYNC` op signature (`84 00 00 00 0c 00 00 00`) | **0 found** |
| 24-byte `DDR_PATCH` op signature (`81 00 00 00 18 00 00 00`) | 1 hit, in `[heap]` — an 8-byte coincidence over 6 GB |

Sanity: the same two op probes find 1 and 16 respectively in `flm_c8_a.txn`, so
they work.

**And the scan is genuinely reading the buffers**, which was the obvious way for
this to be a false negative:

```
/dev/accel BOs : 5291.2 MB mapped,  931.8 MB non-zero (17.6%)
anon/heap/other:  667.2 MB mapped,  575.4 MB non-zero (86.2%)
   sample 0x70ad01200000  33.6 MB  99.0% non-zero     <- weight buffers
```

84 device buffer objects, real contents, stable across 20/40/60 s samples. So
this is a property of FLM, not of the scan.

### What that means, and the leading hypothesis

FLM's per-layer dispatch never exists as an mlir-aie transaction binary in
userspace. Combined with what the earlier entries established:

- the eight embedded transactions are a **generic DDR<->memtile staging ladder**
  (1/2/4/8 columns x egress/ingress), and
- **no transaction ever writes to a compute core** (rows 2-5 untouched), and
- the only per-dispatch variation found anywhere is `DDR_PATCH` rewriting two
  shim BD addresses,

the leading hypothesis is that **there is no separate per-layer instruction
stream at all**: a "fused layer" is executed by re-invoking the *same* staging
transactions with different patched buffer addresses, while the 27 compute cores
run persistent programs loaded once from the PDI and synchronise through locks
(BD word 7 carries `LOCK_ACQ_ID` / `LOCK_REL_ID` fields, currently zero in the
staging BDs but present).

That would make FLM's "one dispatch per layer" advantage a matter of **core
programs that never need reloading plus address patching**, not of a large fused
instruction stream — which is a considerably cheaper thing to reproduce, and
consistent with 38.0 MB/layer arriving as ~9-10 x 4 MB staged chunks.

**Stated as a hypothesis, not a finding.** It is consistent with all evidence so
far and is exactly the kind of tidy story the plan warns about; two of my own
tidy stories have already been wrong today (the ping-pong pair; the 12x gap).

**The test that would settle it:** count actual command submissions per token.
If FLM issues ~9-10 staging dispatches per layer x 16 layers per token, the
hypothesis holds; if it issues ~1 per layer, there is a real fused stream
somewhere and it is reaching the device by a path that does not pass through
readable userspace memory. Counting submissions means instrumenting the
`ioctl(DRM_IOCTL_AMDXDNA_EXEC_CMD)` path (LD_PRELOAD) rather than scanning
memory — a different technique, and the natural next step now that scanning is
ruled out.

Memory scanning is closed as an avenue. `txn_memscan.py` is kept: it proved the
absence, which is a result, and it remains the right tool for any future
question of the form "is a transaction resident in this process".

---

## 2026-07-31 — Phase 1: FLM submits TWO NPU commands per decoded token

New tool: `tools/npu/flm/npu_ioctl_count.c`, an `LD_PRELOAD` interposer on
`ioctl()` that counts `DRM_IOCTL_AMDXDNA_EXEC_CMD` and friends. Read-only —
every call is forwarded unmodified. Built against the driver's own uapi header:

```
gcc -shared -fPIC -O2 -o npu_ioctl_count.so npu_ioctl_count.c -ldl \
    -I ~/xdna-driver/include/uapi
NPU_COUNT_OUT=counts.txt LD_PRELOAD=./npu_ioctl_count.so flm run llama3.2:1b
```

(`-I ~/xdna-driver/include` is wrong and silently picks up the older
`/usr/include/drm/amdxdna_accel.h`, which has no `WAIT_CMD`. The uapi path is
the one that matters.)

### Result — llama3.2:1b, four points, dead linear

| gen-lim | EXEC_CMD | args | CREATE_BO |
|---|---|---|---|
| 8 | 161 | 900 | 508 |
| 208 | 561 | 11,700 | 3,908 |
| 408 | 961 | 22,500 | 7,308 |
| 608 | 1,361 | 33,300 | 10,708 |

Increments are **exactly 400 per 200 tokens**, three times over. Fixed intercept
145 (model load + the ~10-token prefill).

**2.000 EXEC_CMD submissions per decoded token.**

### This refutes the hypothesis from the previous entry

The previous entry predicted ~9-10 submissions per layer x 16 layers = **~150
per token** if a "fused layer" were really re-invocation of the staging ladder.
The measurement is **2**. That story is dead.

It also goes *further* than what this plan has claimed. The plan says
`layer.xclbin` is "one fused decoder layer per dispatch", which for 16 layers
would predict ~16-17 per token. At **2 per token**, FLM is not dispatching per
layer either — the whole 16-layer body appears to go in a single command, with
the second most plausibly `lm_head` (matching the `gen_layer_seq` /
`gen_lm_head_seq` split in the symbol table).

So the fused-layer advantage the plan set out to understand is real but
*understated*: it is not one dispatch per layer, it is on the order of one
dispatch per **model forward pass**.

### Other observations from the same counters

- **`WAIT_CMD = 0` and `SYNC_BO = 0`.** FLM never calls either. It is not
  waiting on completion through the driver's wait ioctl, so completion must be
  observed another way (most likely polling the command BO's status word).
  Relevant to phase 2: our dispatch path should not assume `WAIT_CMD` is how
  this is done.
- **`CREATE_BO` scales with tokens** — 3,400 per 200 tokens = **17 buffer-object
  creations per decoded token**. That is surprising churn for a steady-state
  decode loop and looks like a cost FLM is paying needlessly. Worth confirming
  and, if real, worth *not* copying.
- **args ~27 per EXEC_CMD** (10,800 args per 200 tokens / 400 commands).

### Cross-check in flight

The discriminator: llama3.2:1b has **16** layers, qwen3.6-moe has **40**. If
submissions were per-layer, the MoE must show ~5 per token; if the whole body is
one command, it should still show ~2. Measuring now.

### Cross-check result — dispatch count does NOT scale with depth

| model | layers | gen-lim 8 | gen-lim N | per token |
|---|---|---|---|---|
| llama3.2:1b | **16** | 161 | 1,361 @ 608 | **2.00** (4 points, exact) |
| qwen3.6-moe:35b-a3b | **40** | 4,926 | 5,226 @ 108 | **3.00** (2 points) |

A per-layer dispatch would predict **16** and **40**. A single fixed dispatch
would predict 2 and 2. Measured: **2 and 3**.

**So submissions track the number of distinct kernel phases, not depth.** 2.5x
the layers buys one extra command, not 2.5x the commands. Whatever FLM is
submitting, one command drives many layers.

The likely decomposition, from the symbol table (`gen_layer_seq`,
`gen_lm_head_seq`, `gen_mha_engine_seq`, `gen_dequant_seq`): llama = body +
lm_head; the MoE's third is plausibly its `mtp_num_hidden_layers: 1`
multi-token-prediction head, or a separate phase for its 30 linear-attention/SSM
layers. **Not established** — the counter says how many, not which.

Caveat: the MoE figure rests on two points rather than four, because each run
reloads 23 GB. The result is an exact integer and the method was validated to
four points on llama, but a third MoE point would be worth taking.

### What this settles, and what it changes

**The plan's "one fused decoder layer per dispatch" is confirmed in spirit and
understated in degree.** FLM does not dispatch per layer; it dispatches per
*phase*, and a phase spans the whole stack of layers. For llama that is 2
commands to produce a token from a 16-layer model.

For phase 2 this is the single most important structural fact found so far, and
it reframes the target. The plan's own comparison point — hipfire's "4
dispatches per layer (qkv, o, gate_up, down)" at a ~37 us per-dispatch floor —
is **64 dispatches per token** for the same model against FLM's **2**. At that
floor, 64 dispatches is ~2.4 ms of pure submit latency per token, which alone
would cap decode near 420 tok/s and dominates everything else at these sizes.
Closing that gap is a dispatch-structure problem, not a kernel problem.

That also explains how FLM reaches 86% of fabric bandwidth on llama decode (0.5)
while our own paths do not: with 2 submissions per token there is almost no
submit overhead left to hide.

### Phase 1 item 4 ANSWERED: FLM runs two completely different dispatch regimes

Extended the interposer to log each submission's `(hwctx, type, cmd_count,
arg_count)`. `gen-lim 60`, llama3.2:1b, short prompt — 265 submissions, matching
`145 + 2 x 60` exactly. They split cleanly in two:

**PREFILL — per-linear dispatch, 145 submissions:**

| context | args | count | per layer |
|---|---|---|---|
| ctx 2 | 3 | 112 | **7.0** — the 7 linears (q, k, v, o, gate, up, down) |
| ctx 3 | 5 | 16 | 1.0 |
| ctx 4 | 3 | 16 | 1.0 |
| ctx 1 | 4 | 1 | once — lm_head |

**9 per layer x 16 layers + 1 = 145.** Exactly. `7.0` linears per layer is
llama's attention (q,k,v,o) plus MLP (gate,up,down) — the arithmetic lands on
the architecture with no remainder.

**DECODE — one fused command for the whole model, 2 per token:**

```
ctx1 args=50   ctx1 args=4   ctx1 args=50   ctx1 args=4  ...
```

Every decode submission is on **context 1**, alternating a **50-argument**
command with a **4-argument** command. The 50-arg command is the entire 16-layer
body in one dispatch; the 4-arg one matches the single lm_head submission seen
at the end of prefill.

### Why this matters

**FLM does not have one dispatch strategy, it has two**, and they are wildly
different:

| | dispatches | per token/chunk |
|---|---|---|
| prefill | 9 per layer | 145 for 16 layers |
| decode | 2 total | 2 for 16 layers |

That is a **72x** difference in dispatch granularity between the two paths of
the same model. Prefill is compute-bound so per-linear dispatch costs little and
buys scheduling freedom; decode is latency- and bandwidth-bound, so everything
collapses into one command.

This retro-explains the xclbin set: `mm.xclbin` (prefill GEMM) and `attn.xclbin`
serve the per-linear prefill contexts 2/3/4, while **`layer.xclbin` is the
decode path** — one fused whole-model dispatch on context 1. The plan treats
`layer.xclbin` as "one fused decoder layer"; it is one fused *model*.

`50 = 3 x 16 + 2` is suggestive (three buffers per layer plus an input and an
output), but the argument list has not been decoded — **not established**.

Ordering within a prefill layer is also tentative: the raw sequence is
`(ctx3,5) (ctx2,3)x3 (ctx4,3) (ctx2,3)x4`, which rotated reads as
qkv -> [ctx4] -> o,gate,up,down -> [ctx3], but where the layer boundary actually
falls is not pinned down. The **counts** are solid; the **order** is not.

### Consequence for phase 2

The reproduction needs *two* dispatch designs, not one. Matching FLM's decode
means a single command carrying the whole model — which is a host-side and
control-code problem, and on the evidence so far it is worth more than any
kernel-body optimisation. Matching its prefill is comparatively conventional.

### The 50 arguments decoded — the per-layer buffer contract

Extended the interposer to record `CREATE_BO` handle -> (size, type) and then
dump the `args` array of a matching submission. (`args` is a user pointer to
`arg_count` **u32** BO handles — confirmed from the driver,
`amdxdna_ctx.c:589`, `copy_from_user(arg_bo_hdls, ..., arg_count * sizeof(u32))`.)

The 50-argument decode command is a clean repeating triple:

```
50 = 16 x (weights, workspace, KV) + 2
```

| buffer | size | count | role |
|---|---|---|---|
| weights | 38,797,312 B = **37 MiB** | 16 | one per layer |
| workspace | 1,048,576 B = **1 MiB** | 16 | one per layer |
| KV cache | 268,435,456 B = **256 MiB** | 16 | one per layer |
| trailing | 1 MiB x 2 | 2 | activations in / out |

**The KV size is an exact, unambiguous match:**

```
8 kv_heads x 64 head_dim x 2 (K and V) x 2 bytes x 131072 max_position_embeddings
  = 268,435,456 bytes
```

Not approximately — exactly. So each layer's KV buffer is **preallocated for the
full 131,072-token context** regardless of actual sequence length. This is what
the plan's note about `set_max_length` preallocation refers to, now with a
number attached.

**The weight buffer confirms the 5.00 bpw figure and reveals the allocator
granularity.** Computed per-layer weights are 38,010,880 B = 36.25 MiB; the BO
is 37 MiB — exactly. Every buffer in the list is a whole number of MiB, so FLM
allocates in **1 MiB units**. That 2.07% discrepancy was allocation rounding,
not a format difference.

**Cross-check that ties it back to an independent measurement:** one decode
command references `16 x (37 + 1 + 256) + 2 = 4,706 MiB = 4.60 GiB` of buffers.
The live-process scan two entries earlier independently measured **~5.3 GB of
`/dev/accel` BO mappings**. Consistent.

### What this gives phase 1 and phase 2

This *is* the per-layer buffer contract the phase-1 deliverable needs: three
buffers per layer plus two global activation buffers, all passed to a single
command that executes the whole model.

Note the asymmetry it exposes: the command references **4.6 GiB** of buffers but
only **772.3 MB** is streamed per token (0.5). The KV allocation dominates the
footprint (4 GiB of the 4.6) while contributing almost nothing to per-token
traffic, because only the used prefix is touched. **Footprint and bandwidth are
decoupled here**, and any reproduction that sizes its KV to the actual sequence
length instead of the maximum will look very different in memory without being
faster.

Also worth flagging for phase 3b: KVarN's win is on KV *bandwidth*, but this
shows KV *capacity* is the binding resource at long context — 4 GiB for a 1B
model. A 4-bit KV would cut that to 1 GiB, which may matter more than the
bandwidth argument the plan leads with.

---

## 2026-07-31 — Phase 1 deliverable written: `docs/npu/flm-layer-dataflow.md`

Consolidated the host-side dispatch findings into the per-kernel deliverable the
plan asks for. Every number is tagged **measured** (with method), **derived**
(arithmetic shown), or **open** (explicitly not established), and section 7 is a
dedicated "what is NOT established" list so the gaps are as visible as the
results.

All arithmetic re-verified programmatically before writing: KV size, the 4,706
MiB buffer footprint, the 772.4 vs 772.3 MB weight decomposition, embed_tokens,
both bandwidth figures, and the 145-submission prefill decomposition. All pass.

**Phase 1 status against the plan's five items:**

| item | state |
|---|---|
| 1. every core's role, confirmed | **not started** — still the goal doc's inferred op-mix table |
| 2. tile shapes and buffer sizes | **partly** — host-side buffer contract done (37/1/256 MiB per layer); core-tile DMA BD layout decoded for the staging ladder, not for the decode kernel |
| 3. the DMA/objectfifo graph | **partly** — staging ladder fully mapped; decode path's graph unknown |
| 4. host-side dispatch | **substantially done** — two regimes, submission counts, buffer contract, driver interface |
| 5. attention specifics | **not started** |

Item 4 was the one the plan called "arguably worth more than the kernel bodies",
and it has produced the largest single finding so far: **2 submissions per token
against hipfire's 64**.

**What the deliverable cannot yet say, and why it matters:** the decode
command's instruction stream was never found (0 hits over 6.0 GB of live process
memory). So the *dataflow inside* `layer.xclbin` — items 1, 2 (core side), 3 and
5 — remains reachable only through the CDO/PDI in the xclbin itself, which is
where `cdo.py` and `aiedis.py` come in. That is the natural next line of work,
and it is a different technique again: static analysis of the core programs
rather than dynamic instrumentation of the host.

---

## 2026-07-31 — Phase 1 item 1: the col-5 core is IDENTIFIED

`cdo.py` extracts 27 cores from `layer.xclbin`, and their sizes group exactly as
the goal doc's table predicts:

| size | count | tiles |
|---|---|---|
| 9236 B | **16** | cols 0,1,6,7 x rows 2-5 — GEMM |
| 4580 B | 4 | cols 3,4 rows 2,4 |
| 6852 B | 4 | cols 3,4 rows 3,5 |
| 6388 B | 1 | col 2 row 2 |
| 1812 B | 1 | col 2 row 3 |
| 2036 B | 1 | **col 5 row 2 — the unidentified one** |

**The tooling reproduces the goal doc's op-mix counts exactly** — GEMM cores
`vmac.f:264 vextbcst.16:256 vunpack:64`, cols 3/4 rows 2,4 `vshuffle:91`, cols
3/4 rows 3,5 `vmul.f:86 vadd.f:73`. Independent re-derivation, same numbers, so
both the extraction and the prior analysis are sound.

### col 5 row 2 = the SiLU activation core, as a clamped table lookup

332 bundles, one hardware loop (`lc=0x20`, `ls=0x490`, `le=0x510`). Its loop body:

```
490: vldb            wl11, [p1], m0           load bf16 input
4a4: vfloor.s32.bf16 x8, wl11, s0             -> fixed point, shift s0 = 6
4aa: vmax_lt.32      x8, r16, x8, x1          clamp low   (x1 = -0x200)
4ae: vmin_ge.32      x8, r16, x8, x2          clamp high  (x2 =  0x1ff)
4b2: vadd.32         x7, x0, x8               add table base
4c0: vldb.4x64.lo/hi wl10, wh10, wl9, wh9     gather at computed indices
4dc: vshuffle        x4, x10, x9, r3
4e0: vshuffle        bmlh0, x10, x9, r0
4e4: vmac.f          dm2, dm0, x4, x5, r1     interpolate
4f8: vmul.f          dm3, x4, x6, r2
510: vst.conv.bf16.fp32  bmll3, [p0], #0x20   store bf16
```

That is: **convert to a fixed-point index, clamp it, gather two neighbouring
table entries, interpolate, store.** A piecewise-linear lookup of a smooth
function.

The constants pin down which function:

```
shift s0 = 6          -> step 1/64
clamp [-512, +511]    -> input domain [-8.0, +7.984375]
                      -> 1024 table entries
```

A 1024-entry table spanning exactly **[-8, +8]** is the signature of a
**sigmoid-family activation**, which saturates outside that range. And
Llama-3.2-1B's `config.json` says `hidden_act: silu`. Combined with the other
cores' roles (GEMM; `vmul.f`/`vadd.f` for the SwiGLU elementwise multiply and
residual), **col 5 row 2 computes SiLU**.

**Confidence:** the *mechanism* (clamped fixed-point table lookup with 2-point
interpolation, bf16 in/out, 1024 entries over [-8,8]) is read directly off the
instructions and is solid. That the function is specifically SiLU rather than
another sigmoid-family curve is inference from `hidden_act: silu` plus the
absence of any other activation core — strong, but one step removed.

**Minor correction to the goal doc.** Its table lists col 5 row 2 as having
"`vaddsign0`, `vsel.32`". `vaddsign0` is not an opcode — it is a **mode
register**, written once by `movx vaddsign0, #0x1` at `0x43c` to configure the
`vmax_lt.32` / `vmin_ge.32` comparisons. Reading it as an operation is what made
this core look unclassifiable.

**Design observation worth carrying to phase 3:** clamping the *index* means
inputs beyond +8 return the table's endpoint rather than SiLU's asymptotic
identity. Harmless for post-RMSNorm activations, which rarely exceed 8, but it
is an approximation FLM accepts — and one a reproduction is free to make too,
or to avoid.

### GEMM core: the dequant chain seen concretely; `p5` narrowed, NOT resolved

Traced pointer usage through the GEMM core's hardware loop
(`lc=0x2`, `ls=0x260`, `le=0x1850` — matching the goal doc).

**Confirmed about `p5`:** loaded `vldb wl2, [p5], #0x40` exactly **8 times** per
outer iteration at 64-byte stride = **512 B**, then rewound by
`paddb [p5], #-0x200` at `0x17d8`. So it is re-read identically for every N
step, exactly like the activation pointer `p1`. The goal doc's description is
correct in every measurable respect.

**The MAC operand chain is now concrete**, which is the part that matters for
phase 3a:

```
vlda x8, [p0], #0x40        packed int4 weights
vunpack x9, wl8, unpacksign0    unpack nibbles
vups.4x dm2, x9, s0, upssign0   widen into accumulator
vadd    dm1, dm2, dm0, r0       zero-point / min term  (q4_1 is asymmetric)
vconv.bf16.fp32 x3, cml1        accumulator -> bf16 weights
vldb  x11, [p1], #0x40          activations
vextbcst.16 x10, x11, #0x1d     broadcast ONE activation element
vmac.f dm1, dm1, x3, x10, r4    accumulate
```

So the multiply is **bf16 dequantized-weight x broadcast-activation scalar** —
the M=1 GEMV shape the plan describes, with `vextbcst.16` broadcasting activation
elements one at a time. This is the 42-op-per-K-group dequant chain phase 3a
exists to delete, now read off the instructions rather than inferred from an op
histogram.

**`p5`'s role remains OPEN.** Its destination `wl2` participates in a
`vmov wl2, wh2` / `vmov wl3, wh2` / `vmov wl5, wh2` register-rotation motif that
recurs once per K-group, but resolving what the data *is* requires a register
aliasing model for AIE2P `w`/`x` pairs that I do not have verified — and
guessing it is precisely how the earlier `txn_decompile.py` analysis produced
plausible garbage for months.

**Two ways to settle it, both better than more static reading:**

1. **Look at the buffer.** `p5` points into L1; dumping that address range during
   a run says directly whether it holds activations, scales, zero-points or a
   second weight stream. The memory-scan tooling from earlier already reads live
   BOs.
2. **Get the aliasing right from the vendor header.** `aie2pintrin.h` and the
   llvm-aie register definitions pin down which `x` register each `wl`/`wh` half
   maps to; with that, the def-use chain resolves mechanically.

Recorded as narrowed-not-resolved deliberately. What is solid — 512 B, rewound,
K-only, same cadence as the activation stream — is unchanged from the goal doc;
this pass added the surrounding dataflow but not the answer.

### GEMM core L1 buffer map — concrete addresses for every operand pointer

Better route than the register-aliasing chase: read the pointer *setup* in the
prologue, where the operand addresses are immediates.

```
0x0148  movxm p7, #0x74000     ...  0x0170  mov   p4, p7    -> p4 = 0x74000
0x0152  movxm p6, #0x7c000     ...  0x0170  movs  p5, p6    -> p5 = 0x7c000
0x0166  movxm p1, #0x72800                                  -> p1 = 0x72800
0x0166  movxm r8, #0x78200     (spilled to [sp,#-4])        -> 0x78200
0x01fa  movxm p4, #0x73ca4 ; lda r6,[p4]                    -> scalar constant
0x0204  movxm p4, #0x73ca3 ; lda.s8 r7,[p4]                 -> scalar constant
        p0 = r11 (function argument)                        -> weights
```

| pointer | L1 address | size | role |
|---|---|---|---|
| `p0` | function arg | 2048 B | packed int4 weights (advances) |
| `p1` | **0x72800** | 512 B | activations (rewound) |
| `p4` | **0x74000** | 512 B | per-group scales (advances with N) |
| `p5` | **0x7c000** | 512 B | **unresolved** (rewound, K-only) |
| — | 0x73ca3 / 0x73ca4 | 1 B each | scalar constants (`lda.s8` — rounding mode / shift) |
| — | 0x78200 | ? | spilled to stack, role unknown |

All four operand buffers live in the **same 64 KB data-memory module**
(`0x7xxxx`; the core's stack is at `0x70000`), at module offsets `0x2800`,
`0x4000` and `0xc000`. `p5` sits well away from the activation/scale pair.

**This is the tractable path to resolving `p5`**, and it does not need the
register-aliasing model at all: the core-tile DMA buffer descriptors say which
L1 addresses are DMA-fed, from where, and how big. Decoding this tile's BD
registers (plan item 2, "core-tile DMA BD register layout at 0x1D000") maps
`0x7c000` to a stream and therefore to a producer. That is the next step, and it
also delivers plan items 2 and 3 rather than just this one question.

Recorded because the addresses themselves are reusable: any future def-use
question in this core can be anchored to a named buffer instead of a register.

**Note on the register-aliasing detour.** `docs/npu/ug1079-.../016-vector-
registers.md` documents AIE1 naming (`vrl0` / `wr0` / `xa` / `ya`) and does not
describe AIE2P's `x0-x11` / `wl`/`wh` / `dm` / `bm*` / `cm*` file. The manual in
this repo is therefore not a source for AIE2P aliasing; llvm-aie ships no `.td`
files and no `llvm-mc`. If aliasing is needed later, the remaining options are
compiling a probe kernel with known dataflow and reading the register names
back, or the AIE2 architecture manual.

---

## 2026-07-31 — Phase 1 items 2/3: core-tile DMA BDs decoded, `p5` is DMA-fed

New tool: `tools/npu/flm/cdo_dma.py`. `cdo.py` extracts core *program* memory
from the CDO; this extracts the other half — every DMA buffer descriptor the CDO
programs, decoded into base address, length, dimensions and locks. Field
definitions are read at run time from aie-rt's `xaie2pgbl_params.h` rather than
hardcoded, so the decode cannot drift from the vendor's.

### GEMM core tile (0,2) — 6 buffer descriptors

| BD | module offset | core view | length | locks |
|---|---|---|---|---|
| BD0 | 0x08000 | **0x78000** | 512 B | acq 127, rel_id 1, next=BD1 |
| BD1 | 0x0c000 | **0x7c000** | 512 B | acq 127, rel_id 1, next=BD0 |
| BD2 | 0x02800 | **0x72800** | 5120 B | acq_id 2, rel_id 3, next=BD3 |
| BD3 | 0x04000 | **0x74000** | 5120 B | acq_id 2, rel_id 3, next=BD2 |
| BD4 | 0x03c1c | 0x73c1c | 132 B | acq_id 7, rel_id 4, next=BD5 |
| BD5 | 0x0541c | 0x7541c | 132 B | acq_id 8, rel_id 5, next=BD4 |

**The units were wrong on the first pass and the operand pointers are what
caught it.** `BASE_ADDRESS` and `BUFFER_LENGTH` are both in **32-bit words**,
not 32-byte units: both fields are 14 bits, which spans exactly the 64 KB
data-memory module at 4 bytes per unit. The check that settles it is that the
resulting addresses reproduce the GEMM core's operand pointers **exactly**:

| pointer | address | module offset | BD |
|---|---|---|---|
| `p1` activations | 0x72800 | 0x2800 | **BD2** |
| `p4` scales | 0x74000 | 0x4000 | **BD3** |
| `p5` *unresolved* | 0x7c000 | 0xc000 | **BD1** |

Three out of three, exact. A wrong unit would not produce three coincidences.

### What this says about `p5`

**`p5` is DMA-fed, not computed locally.** Its buffer is BD1: **512 B**, which
matches exactly the 8 x 64 B the core reads before rewinding. BD0 and BD1 are a
**ping-pong pair** — same size, same lock (`rel_id 1`), mutually chained via
`NEXT_BD`/`USE_NEXT_BD` — sitting at 0x78000 and 0x7c000.

So `p5` is a distinct DMA stream, double-buffered, on its own lock, separate
from both the activation stream (BD2/BD3, lock 3) and whatever feeds BD4/BD5
(locks 4/5). That is a real narrowing: it rules out `p5` being a locally derived
quantity, a constant table, or an alias of the activation buffer.

**Still open: which producer feeds BD0/BD1.** That needs this tile's
stream-switch slave configuration, which the CDO also programs — the same tool
can be pointed at those registers. Note the spilled `r8 = 0x78200` from the
prologue lands exactly at BD0's buffer end (`0x78000 + 512`), which is
suggestive but not conclusive.

**Deliberately not concluding what `p5` carries.** The tempting reading — that
two K-indexed 512 B streams plus two accumulator chains (`dm1`, `dm3`) means the
core runs two GEMVs against one weight tile — is a tidy story of exactly the kind
that has been wrong twice already today (the ping-pong transaction pair; the 12x
gap). The producer identifies it; inference should not.

### Also decoded

BD2/BD3 are a chained pair of **5120 B** buffers sharing locks (acq_id 2,
rel_id 3), with `p1` and `p4` pointing into them. 5120 = 10 x 512, so the
pointers walk sub-blocks of a larger DMA-delivered region rather than each
owning a buffer. BD4/BD5 are a 132 B chained pair on locks 7/8 -> 4/5.

**Method note:** this is the third time today that decoding against the vendor's
own definition beat inference (aie-rt for `patch_op_opt_t`, aie-rt for the shim
BD direction, aie-rt again here) — and the second time a unit error was caught
by cross-checking against an independently obtained number rather than by
re-reading the spec.

### `p5`'s producer found: it arrives over the horizontal core-to-core chain

Decoded the stream-switch configuration from the CDO and resolved the physical
port indices against aie-rt's own `Aie2PTileStrmSwSlavePortMap`
(`xaie2pgbl_reginit.c:721`) rather than guessing the numbering.

For the GEMM core (0,2):

```
MASTER DMA0  <- slave phy19 = EAST0    (circuit-switched)
MASTER DMA1  <- slave phy 7 = SOUTH2   (circuit-switched)
```

and the channel-to-BD assignment from the DMA control registers:

```
0x1de04  S2MM_0_START_QUEUE = 0   -> BD0  (BD0/BD1 pair, 512 B)   <- p5
0x1de0c  S2MM_1_START_QUEUE = 2   -> BD2  (BD2/BD3 pair, 5120 B)  <- p1, p4
0x1de14  MM2S_0_START_QUEUE = 4   -> BD4  (BD4/BD5 pair, 132 B)   -> output
```

**So `p5` is fed by DMA0, whose source is the EAST neighbour tile — a
core-to-core stream, not memory.** The other input channel (`p1`/`p4`) comes
from SOUTH, i.e. the memtile.

### The array topology: horizontal broadcast, vertical weight feed

Mapping DMA0's source across every tile in rows 2-3:

| tile | DMA0 source | tile | DMA0 source |
|---|---|---|---|
| (0,2) | **EAST0** | (4,2) | EAST2 |
| (0,3) | **EAST3** | (5,2) | SOUTH4 |
| (1,2) | SOUTH4 | (6,2) | **WEST1** |
| (1,3) | SOUTH2 | (6,3) | **WEST0** |
| (2,2) | SOUTH3 | (7,2) | **WEST0** |
| (3,2) | EAST1 | (7,3) | **WEST2** |

The pattern is unambiguous: **columns 0-1 pull from the EAST, columns 6-7 pull
from the WEST, and the middle columns pull from SOUTH (the memtile).** DMA1 is
SOUTH on essentially every tile.

**derived** — data enters from the memtile in the middle columns and is
propagated *horizontally outward* through the stream switches to the edge GEMM
columns, while each column independently pulls its own stream from the memtile
below on DMA1. One memtile read therefore serves several columns instead of
every core pulling separately. That is a real bandwidth-economy design and it is
directly relevant to phase 2 — it is a large part of how FLM sustains 86% of
fabric on decode.

**On `p5`'s contents.** Its buffer is 512 B = **256 bf16 values = exactly K=256**,
it is K-indexed and rewound across N, and it arrives on the horizontal chain from
a neighbouring core. That is consistent with a shared activation vector being
broadcast across the array. **Still not asserted** — the connectivity is now
established but the payload is not, and the difference matters. Confirming it
means either identifying what the *source* tile writes to that stream, or
dumping the buffer at 0x7c000 during a run.

Note (2,3) and (3,3)/(4,3) break the pattern (`S2MM1 -> BD6`, DMA0 unconfigured),
so the graph is not uniform across rows — the row-3 tiles in columns 3-4 have a
different shape. Not chased this pass.

---

## 2026-07-31 — Phase 1 item 3: the DMA graph, and `p5` is a BROADCAST stream

`cdo_dma.py --graph` now resolves every enabled stream-switch route in the
xclbin, with physical port indices taken from aie-rt's `Aie2P*StrmSwPortMap`
arrays for core, memtile and shim. **294 enabled routes** across the array.

### The complete path feeding `p5`

```
memtile(1,1) MM2S ch4  ->  NORTH4
    -> core(1,2) SOUTH4 slave
         |-> core(1,2) DMA0        -> its own BD0/BD1 -> its p5
         |-> core(1,2) WEST0 master -> core(0,2) EAST0 -> DMA0 -> BD0/BD1 -> p5
         `-> core(1,2) EAST2 master -> core(2,2)
```

Every hop is a decoded register, not an inference:

| register | value | meaning |
|---|---|---|
| `(1,1)` `NORTH4` | `<- DMA4` circuit | memtile MM2S ch4 drives the north stream |
| `(1,2)` `DMA0` | `<- SOUTH4` circuit | core (1,2) consumes it |
| `(1,2)` `WEST0` | `<- SOUTH4` circuit | and forwards the *same* stream west |
| `(1,2)` `EAST2` | `<- SOUTH4` circuit | and east |
| `(0,2)` `DMA0` | `<- EAST0` circuit | core (0,2) receives the forwarded copy |

**A single memtile MM2S channel feeds at least three columns' `p5` buffers** by
circuit-switched forwarding through the intermediate tile's stream switch. This
is a genuine 1-to-N broadcast, not N separate memtile reads.

### The structural result: shared vs private operands

| | source | buffer | shared? |
|---|---|---|---|
| `p5` (DMA0, BD0/BD1) | memtile (1,1) via horizontal chain | 512 B | **broadcast across columns** |
| `p1`, `p4` (DMA1, BD2/BD3) | the column's **own** memtile (0,1) | 5120 B | **private per column** |

That distinction is the important part, and it is established by connectivity
rather than by reading the payload: **one operand stream is shared by every GEMM
column, the other is per-column.** For a GEMV that is exactly the split you would
expect — every core needs the same activation vector, and each needs its own
slice of the weights.

**Which raises a question about the goal doc's labels.** It records `p1` as the
512 B activation stream and `p5` as an unresolved 512 B stream. But `p5` is the
one that is 512 B *and* broadcast, while `p1` points into a 5120 B per-column
buffer. If activations are what must be shared, `p5` looks like the activation
stream.

**Not asserting the relabel.** Connectivity says shared-vs-private; it does not
say what the bytes are, and `p1`'s reads really are 512 B at a time (within a
larger buffer). Both facts can hold at once. Resolve by dumping `0x72800` and
`0x7c000` from a core's L1 during a run and comparing against a known activation
vector — that is a direct observation, and it is what should decide it rather
than a third round of inference.

### Also visible in the graph

- Memtile DMA channels appear as both master and slave (`DMA4 <- NORTH1` and
  `NORTH4 <- DMA4`): S2MM and MM2S share an index in the port map, so a memtile
  channel both receives from the cores and transmits to them.
- The shims route mostly packet-switched; the core-to-core forwarding is
  **circuit**-switched, which is what makes the broadcast cheap.
- `(0,1)` memtile takes `DMA0 <- NORTH0` and `DMA3 <- NORTH1` *from the core
  above*, i.e. the memtile also collects results coming back down.

### `p5` RESOLVED: it is the shared activation vector, and the doc's labels are swapped

The decisive static test is *how many cores the broadcast reaches, and which*.
Checking DMA0 configuration and BD0 geometry on every core tile:

```
cores with DMA0 enabled AND a 512 B BD0 at module 0x08000:  16
GEMM cores (cols 0,1,6,7 x rows 2-5):                       16
```

**Exactly the 16 GEMM cores, all 16 of them, and no others** — every one with a
byte-identical BD0 (base `0x08000`, length 512 B) and its BD1 pair at `0x0c000`.
The non-GEMM cores also enable DMA0 but with completely different geometry
(4096 B at `0x04000`, 1024 B, 2048 B at `0x00400`, 6144 B at `0x01000`).

That closes it:

- The stream is delivered **identically to all 16 GEMM cores and to nothing
  else**.
- Its buffer is **512 B = 256 bf16 = exactly K=256**.
- It is **K-indexed and rewound across N**, i.e. re-read for every output column.
- **Weights and scales cannot be broadcast identically**, because each GEMM core
  computes a different output slice and therefore needs different weights and
  different per-group scales.

The only operand a GEMV shares across cores computing different output slices is
the **input activation vector**. `p5` carries the activations.

### CORRECTION to the goal doc's GEMM tile description

The goal doc records:

> `p0` 2048 B packed int4 weights (advances) · `p1` 512 B bf16 activations
> (**rewound**, reused across N) · `p4` 512 B per-group scales (advances with N)
> · `p5` 512 B (**rewound**, K-only — role unresolved)

**`p5` is the 512 B bf16 activation stream**, arriving by broadcast from a
memtile through circuit-switched core-to-core forwarding. `p1` and `p4` index
into a **5120 B per-column private buffer** fed from that column's own memtile
(BD2/BD3, lock 3) — the per-column data, i.e. weights and scales.

The description of `p1`'s *access pattern* (512 B at a time, rewound, reused
across N) is accurate; what is wrong is calling that buffer the activations,
when the activations are the thing that must be shared and `p1`'s buffer is
private to the column.

**Confidence.** This is established from connectivity plus architectural
necessity, not from reading the bytes: 16-of-16 exact coverage of precisely the
GEMM cores, identical geometry, correct size for K=256 bf16, and the logical
impossibility of broadcasting per-slice weights. A byte-level dump would make it
airtight; core L1 is not reachable from userspace, so that would need the
driver's AIE debug path.

### Phase 1 item 1 progress: the GEMM role is now confirmed, not inferred

The goal doc's role table was "inferred from op mix and NOT confirmed". For the
16 GEMM cores it is now confirmed **structurally**: they are exactly the set of
cores receiving a broadcast K=256 activation vector plus a private per-column
weight/scale stream, and emitting a small result stream (BD4/BD5, 132 B). That
is a GEMM/GEMV tile by construction, independent of what its opcodes look like.

### The broadcast topology

Two origins, feeding the two halves of the array:

- memtile **(1,1)** -> `(1,2)` -> forwarded **west** to `(0,2)` and **east** to `(2,2)`
- memtile **(5,1)** -> `(5,2)` -> forwarded **east** through `(6,2)` to `(7,2)`

and vertically: row-3 cores take DMA0 from the **core below** (`(1,3) <- SOUTH2
from (1,2)`), so the fan-out is a 2-D tree — horizontal across columns, vertical
up the rows — rather than a bus. All core-to-core hops are **circuit**-switched,
which is what makes the replication cheap.

---

## 2026-07-31 — Phase 1 item 5: the attention kernel's head mapping and tile width

Same method applied to `attn.xclbin`: CDO -> buffer descriptors -> stream graph.

**All 32 cores are byte-identical** — every tile in cols 0-7 x rows 2-5 has the
same five BDs, confirming the goal doc's "32 identical cores, homogeneous,
unlike `layer`":

| BD | size | pairing | locks |
|---|---|---|---|
| BD0 | 8192 B | single | rel_id 1 |
| BD1 / BD2 | 4096 B each | ping-pong (NEXT 1<->2) | acq_id 2, rel_id 3 |
| BD3 / BD4 | 128 B each | ping-pong (NEXT 3<->4) | acq_id 5, rel_id 4 |

### The head mapping is exact

llama-3.2-1B: **32 query heads, 8 KV heads, head_dim 64**, GQA ratio 4.

```
32 cores          == 32 query heads   -> ONE CORE PER QUERY HEAD
8 columns         ==  8 KV heads      -> ONE COLUMN PER KV HEAD
4 rows per column == GQA ratio 4      -> the 4 query heads sharing that KV head
```

Every count matches with no remainder. The array geometry *is* the head
decomposition.

**This answers the goal doc's `k03`/`k47` question.** It records that KV heads
are "split into two groups of four (`get_k03_offset`/`k47`/`v03`/`v47`)". With
one KV head per column, that split is simply **columns 0-3 and columns 4-7** —
the left and right halves of the array.

### The flash tile width is 32 tokens

The goal doc lists this as **not pinned** ("loop counts are `lc = 0x20 / 0x1f /
0x8`; 32 tokens/tile is the likely read at head_dim 64"). BD geometry settles it:

| BD | bytes | as bf16 | interpretation |
|---|---|---|---|
| BD3/BD4 | 128 | 64 | **1 x head_dim** — the Q vector for one head (decode, M=1) |
| BD1/BD2 | 4096 | 2048 | **32 x 64** — a KV tile of **32 tokens** |
| BD0 | 8192 | 4096 | **2 x 32 x 64** — K and V for 32 tokens |

And it reconciles the loop counts: `lc = 0x20` = **32** is the tile width;
`lc = 0x8` = **8** is the KV-head/column count; `0x1f` = 31 is the
tile-width loop with the first iteration peeled, which is exactly what an online
softmax does (initialise from element 0, then fold in the remaining 31).

BD1/BD2 being a **double-buffered 4096 B pair** is textbook flash attention: the
KV tile streams in while the previous tile is consumed. BD3/BD4 double-buffer the
128 B query vector.

### Broadcast topology: pairwise, not array-wide

DMA0 sources alternate by column parity — even columns pull **SOUTH** (their own
memtile), odd columns pull **WEST** (from the even column beside them):

```
(0,2) SOUTH0   (1,2) WEST2      (2,2) SOUTH3   (3,2) WEST1
(4,2) SOUTH3   (5,2) WEST0      (6,2) SOUTH0   (7,2) WEST1
```

So attention broadcasts in **column pairs** — memtile -> even column -> odd
column — rather than the long horizontal chains `layer.xclbin` uses. Different
kernel, different fan-out, which makes sense: attention's per-column data (the
KV head) is shared by only 2 columns here, whereas `layer`'s activation vector is
shared by all 16 GEMM cores.

### Still open in item 5

- **Where RoPE is applied.** Not shown by BD geometry. The goal doc notes
  `_set_rope_weights` and precomputed tables; finding which core reads them is
  the remaining piece.
- **KV tile layout inside the memtile** — the memtile-side BDs are decoded by
  the same tool but not yet analysed for this kernel.
- BD0's role (8192 B, single, not ping-ponged) is consistent with K+V for 32
  tokens or an output accumulator; not distinguished.

### Attention memtile: the KV layout transform lives in the DMA, not the cores

Decoded `attn.xclbin`'s memtile (0,1) — **25 buffer descriptors**, against 5 per
core. The structural difference is the important part:

**Core-tile BDs are all flat linear. Memtile BDs are 4-dimensional strided.**

| BD group | size | addressing | locks |
|---|---|---|---|
| BD0/BD1, BD24/BD25 | 16384 B | **flat** | 64->65, 68->69 |
| BD2/BD3, BD4/BD5, BD26/BD27 | 8192 B | **4-D strided** | 65->67, 69->70, 67->64 |
| BD6-BD9 | 4096 B | **4-D strided** | 71->72, 73->74 |

Every strided BD carries the same shape:

```
D0_WRAP=4   D1_STEPSIZE=31 D1_WRAP=8   D2_STEPSIZE=3 D2_WRAP=8   D3_STEPSIZE=255
```

aie-rt encodes stepsize as **`StepSize - 1`** (`dma/xaie_dma_aieml.c:316-355`),
so the real strides are:

| dim | stride | note |
|---|---|---|
| D1 | 32 words = **128 B** | **exactly one head_dim vector** (64 x bf16) |
| D2 | 4 words = 16 B | 8 bf16 |
| D3 | 256 words = 1024 B | |

`8192 B / (4 x 8 x 8 = 256 units) = 32 B` per unit = 16 bf16.

**So the memtile performs the KV layout transform in its DMA**, gathering with a
128-byte (one head_dim vector) stride, and hands the cores flat contiguous tiles.
The cores never reshape anything — every core BD in both kernels is a plain
linear transfer.

The lock IDs form a cycle (`64->65`, `65->67`, `67->64`), i.e. a pipeline: a
16 KB region is written linearly, then read out as strided 8 KB tiles, then the
buffer is recycled.

**Why this matters for phase 3b.** The goal doc's KVarN design wants K records
**channel-major** `[head_dim x GROUP]` specifically so the `mac_4x16_16x16` B
operand needs no transpose. This shows FLM already solves the equivalent problem
**in the memtile DMA rather than in the kernel** — the reshape is free, done by
the DMA engine's strided addressing while data moves. A KVarN implementation has
the same lever available, and should use it rather than choosing a storage layout
to avoid a transpose the DMA could have done anyway.

### RoPE: narrowed to `layer.xclbin`, not proven

Chased where RoPE is applied. Not closed, but meaningfully narrowed, and the
evidence points away from where I first looked.

**1. The attention core's op profile confirms a separate plan claim.**
`attn` core (0,2), 1693 bundles: `vconv.bfp16ebs8.fp32` **x37**, which confirms
the goal doc's "operands load bf16, convert in-register to BFP16-ebs8". Also
`vmul.f:70 vadd.f:51 vmac.f:44 vmsc.f:24 vshuffle:104`.

**2. Opcode mix does NOT discriminate.** RoPE needs shuffle-to-pair plus
`x_even*cos - x_odd*sin` (a multiply-subtract) and `x_odd*cos + x_even*sin`. Both
candidate cores have exactly that combination — and both have **`vmsc.f` = 24**,
the identical count:

| core | vshuffle | vmul.f | vadd.f | vmsc.f |
|---|---|---|---|---|
| `attn` (0,2) | 104 | 70 | 51 | **24** |
| `layer` cols 3,4 rows 3,5 | 51 | 86 | 73 | **24** |

Same `vmsc.f` count in both means this signature is a shared idiom, not a
fingerprint. Dropped as a discriminator.

**3. The symbol tables do discriminate.** RoPE exists in exactly one library:

```
libmha.so        0 rope symbols     <- the attention kernel library
libgemm.so       0
libdequant.so    0
liblm_head.so    0
libllama_npu.so  3   _set_rope_weights(int)
                     _send_rope_weights(npu_sequence*)
                     _rope(buffer<bfloat16_t>&, int)
```

**`libmha.so` has none.** RoPE is owned by the model library that also owns
`gen_layer_seq`, not by the MHA kernel.

**Leading hypothesis:** RoPE is applied in **`layer.xclbin`'s cols 3-4 cores** —
the ones the goal doc labels "shuffle + elementwise". Three strands agree: the
symbols are exclusive to `libllama_npu.so`; those cores carry the required op
combination; and architecturally RoPE applies to Q and K immediately after their
projections, which happen in the layer kernel.

**Not asserted.** A call-graph trace from `_send_rope_weights` to a specific
`gen_*_seq` would settle it; the direct-call xref came back empty, so the call is
indirect (vtable or `std::function`) and needs a proper approach. The other route
is checking whether a rope-table buffer appears in the layer dispatch's argument
list — it is not among the 50 decode args, so if it exists it is a resident
buffer loaded once at init, consistent with `_set_rope_weights(int)` being a
setup call.

**Small trap worth noting:** an initial `grep -i rope` over the disassembly
returned 1678 "hits", nearly all of them `boost::property_tree` — "p-**rope**-rty"
contains the substring. The apparent flood of matches was one substring collision.

---

## 2026-07-31 — Second Phase 1 deliverable: `docs/npu/flm-attn-dataflow.md`

Written, with the same measured / derived / open tagging and a dedicated "what is
NOT established" section. The attention material has been moved out of
`flm-layer-dataflow.md` (which now carries a three-line pointer) so each kernel
has one home, as the plan's "`docs/npu/flm-<kernel>-dataflow.md` per kernel"
asks.

**Phase 1 status against the plan's five items:**

| item | state |
|---|---|
| 1. every core's role, confirmed | **partly** — GEMM confirmed structurally; SiLU core identified; norm/shuffle/elementwise still inferred from op mix |
| 2. tile shapes and buffer sizes | **done** — host-side buffer contract, core-tile BDs both kernels, attention tile width pinned at 32 tokens |
| 3. the DMA / objectfifo graph | **done** — 294 routes decoded, broadcast topology for both kernels |
| 4. host-side dispatch | **done** — two regimes, 2 submissions/token, 50-arg buffer contract |
| 5. attention specifics | **mostly** — head mapping, tile width, k03/k47, memtile layout transform; RoPE narrowed not proven |

Two documents now stand as the phase-1 deliverable:
`flm-layer-dataflow.md` (463 lines) and `flm-attn-dataflow.md` (188 lines),
against a 2151-line working log.

**What the phase produced that the plan did not anticipate**, in rough order of
consequence:

1. **2 NPU submissions per decoded token**, against hipfire's 64 — dispatch
   structure, not kernel bodies, is the dominant gap.
2. **`mac_4x16_16x16` is 512 MACs/cycle, not 1024** — the plan's table was wrong
   by 2x, and int4/int8/bfp16 are all equal on issue rate. int4's real advantage
   is fewer operand bytes on a load-bound machine (1.59x measured streamed).
3. **The array broadcasts operands core-to-core**, so one memtile read serves
   many cores — a concrete mechanism behind 86%-of-fabric decode.
4. **The memtile DMA does layout transformation for free**, which changes how
   phase 3b should think about KV layout.
5. **KV is preallocated at full context** — 4 GiB for a 1B model; capacity, not
   just bandwidth, is a KVarN target.

**Remaining phase-1 work is narrow:** confirm the three inferred core roles in
`layer.xclbin`, settle RoPE's location by call-graph, and identify BD0 in the
attention core. None of it blocks phase 2.

---

## 2026-07-31 — Phase 1 item 1: core roles confirmed from I/O geometry

The GEMM role was confirmed structurally two entries back (the 16 cores receiving
a broadcast K=256 activation vector plus private per-column weights). The same
method — input/output BD geometry rather than op mix — settles most of the rest.

Full map of `layer.xclbin`, per core: input BD chain sizes on each S2MM channel,
output chain on MM2S:

| tile | role | in (ch0 \| ch1) | out | DMA0 src |
|---|---|---|---|---|
| 16 x GEMM | GEMM | **512/512** \| **5120/5120** | 132/132 | EAST/WEST/SOUTH |
| (2,2) | norm | 4096/4096 \| 4096 | **4100** | SOUTH3 |
| (2,3) | norm | 6144 \| 128 | 4096 | SOUTH4 |
| (3,2)(4,2)(3,4)(4,4) | shuffle | 1024/1024 \| 4096/4096 | **none** | EAST/SOUTH |
| (3,3)(4,3)(3,5)(4,5) | elementwise | — \| 4096/4096 | 1024/1024 | — |
| (5,2) | SiLU | **2048/2048** \| — | **1024/1024** | SOUTH4 |

### (2,2) is RMSNorm — the output size gives it away

```
in  4096 B = 2048 bf16 = hidden_size EXACTLY
out 4100 B = 1025 words = 1024 words of vector + ONE EXTRA WORD
```

A 4-byte overhang on an otherwise exact hidden-size vector is a **normalized
vector plus one scalar** — the reciprocal norm. That is an RMSNorm signature, and
4100 is too odd a number to be anything else. The goal doc's "norm / residual"
label for col 2 is confirmed for (2,2).

### (5,2) is SwiGLU, not bare SiLU — refining the earlier identification

```
in  2048 B = 1024 bf16      out 1024 B = 512 bf16      ratio exactly 2:1
```

**Two inputs, one output.** A bare elementwise activation preserves size; halving
means it consumes a *pair*. Combined with the clamped-table SiLU lookup found in
its instruction stream, this core computes **`silu(gate) * up`** — llama's fused
SwiGLU — not SiLU alone.

This refines the earlier entry, which identified the table lookup as SiLU and
stopped there. The lookup does compute SiLU; the core computes SwiGLU.

### The cols 3-4 cores are a vertical pipeline

Rows 2 and 4 (**shuffle**) take inputs but have **no MM2S output configured**.
Rows 3 and 5 (**elementwise**) have no ch0 input but do have outputs. They sit
directly above their shuffle partners.

So `shuffle(row 2) -> elementwise(row 3)` and `shuffle(row 4) ->
elementwise(row 5)`, with the hand-off **not going through DMA at all** — either
the AIE cascade path or shared memory between vertically adjacent tiles. Four
such pairs across cols 3-4.

That is a genuine dataflow finding: part of this kernel's communication does not
appear in the DMA graph, because it does not use DMA.

**Caveat:** the sweep read `MM2S_0_START_QUEUE` only. A core showing "no output"
may be using MM2S channel 1. The shuffle/elementwise *pairing* is supported by
the adjacency and the complementary in/out pattern, but "no DMA output" should be
re-checked against channel 1 before being relied on.

### Item 1 status

| cores | role | basis |
|---|---|---|
| 16 GEMM | GEMM | **confirmed** — broadcast activations + private weights + result stream |
| (2,2) | RMSNorm | **confirmed** — 4096 in / 4100 out |
| (5,2) | SwiGLU | **confirmed** — 2:1 ratio + SiLU table lookup |
| (2,3) | norm/residual | inferred — 6144 in / 4096 out, not decomposed |
| cols 3-4 | shuffle -> elementwise pipeline | **pairing confirmed**, function still inferred |

### Verifying my own caveat — the claim survives, and reveals more

The previous entry flagged that the sweep had read `MM2S_0` only, so "no DMA
output" might just mean "output on channel 1". Checked all four channels
(S2MM 0/1, MM2S 0/1) on every core. **The caveat was worth raising and the claim
holds** — but the fuller picture is more interesting than the one I recorded.

| tile | role | S2MM0 | S2MM1 | MM2S0 | MM2S1 |
|---|---|---|---|---|---|
| (0,2) (1,2) (6,2) (7,2) | GEMM **row 2** | on | on | **on** | -- |
| (0,4) (1,4) (6,4) (7,4) | GEMM **row 4** | on | on | **on** | -- |
| (0,3) (1,3) (6,3) (7,3) | GEMM **row 3** | on | on | **--** | **--** |
| (0,5) (1,5) (6,5) (7,5) | GEMM **row 5** | on | on | **--** | **--** |
| (3,2) (4,2) (3,4) (4,4) | shuffle | on | on | **--** | **--** |
| (3,3) (4,3) (3,5) (4,5) | elementwise | -- | on | on | -- |
| (2,2) | RMSNorm | on | on | **on** | **on** |
| (5,2) | SwiGLU | on | -- | on | -- |

**Shuffle cores confirmed:** no DMA output on *either* MM2S channel. The
non-DMA hand-off to the elementwise core above them stands.

**New, and it revises the GEMM picture: half the GEMM cores have no DMA output
either.** Rows 3 and 5 have both MM2S channels disabled; only rows 2 and 4 emit.

So the 16 GEMM cores are **vertically paired** — row 3 feeds row 2, row 5 feeds
row 4 — with only the even row emitting a result stream. **8 result streams, not
16.** That is the shape of a K-split: each output tile is computed by two cores
that chain partial sums, and only the last in the chain writes out.

This corrects the earlier entry's implicit picture of 16 independent GEMM cores
each emitting 132 B. The correct reading is **8 chained pairs**.

**Mechanism not distinguished.** AIE2P offers two ways to hand off without DMA —
the dedicated cascade path between adjacent cores, and shared access to a
neighbour's data memory. Both are consistent with what the DMA registers show
(namely, nothing). Distinguishing them needs the cascade enable/direction
registers or the memory-module access pattern, neither of which this pass
decoded.

**Also visible:** (2,2) RMSNorm is the only core using **both** MM2S channels —
consistent with a norm feeding two consumers (e.g. the normalized vector to the
GEMM columns and the residual onward). The elementwise cores take their input on
S2MM **channel 1** with channel 0 unused, which is what the earlier table showed
as "— | 4096/4096"; that part was already right.

**Lesson worth keeping:** the caveat was raised because the sweep was partial,
and checking it turned up a structural fact (8 chained pairs, not 16 independent
cores) that the incomplete sweep had actively hidden. Verifying one's own
hedge is not bookkeeping — it is where the finding was.

### RESOLVED: the non-DMA hand-off is SHARED NEIGHBOUR MEMORY, not cascade

Designed the test as a diff between a paired emitter and silent core — if the
hand-off used the cascade path, the sender and receiver programs must differ by
cascade instructions.

**They do not.** `core(0,2)` (emits) and `core(0,3)` (silent): 2079 vs 2076 ops,
**no unique opcodes on either side**, differences only ±1-3 in scalar
`mov`/`lda`/`st` counts — register-allocation noise. No cascade send/receive
anywhere.

But the two are not the same binary:

```
md5 of the 16 GEMM core programs:
   8 x fa1b2dc9ed3d5a50    <- rows 2,4  (the emitters)
   8 x f227d3d030724331    <- rows 3,5  (the silent ones)
```

**Two programs, 8 cores each, splitting exactly along the emit/silent line.**
And they differ in only **178 of 9236 bytes (1.93%)**, in four regions all inside
`0x200e8-0x201ef` — **the prologue**. The compute loop is byte-identical.

So the difference is entirely in **buffer addressing**, and that is the answer:

```
row 2 (emitter)                 row 3 (silent)
  movxm r9,  #0x75400             movxm p3, #0x45400
  movxm r11, #0x73c00             movxm p0, #0x43c00
```

| row 2 | offset | row 3 | offset | |
|---|---|---|---|---|
| `0x75400` | 0x5400 | `0x45400` | 0x5400 | **same offset** |
| `0x73c00` | 0x3c00 | `0x43c00` | 0x3c00 | **same offset** |

Row 2 addresses only `0x7xxxx` — its own data-memory module. Row 3 uses
`0x7xxxx` **plus two buffers in `0x4xxxx`** — a *different module window*, i.e. a
**neighbour's memory** — at **exactly the offsets row 2 reads from its own
window**.

**Row 3 writes its results directly into the neighbouring tile's data memory,
and row 2 reads them as local loads.** No DMA, no cascade, no instructions
distinguishing sender from receiver — just a pointer into a different window,
baked into the prologue.

### Corrections this forces

1. **Retract "K-split with cascade chaining."** The previous entry called the
   pairing "the shape of a K-split... two cores chain partial sums", and offered
   cascade or shared memory as the mechanism. It is shared memory, and there is
   no chaining instruction — the hand-off is a plain store.
2. **The array is not SPMD.** I was drifting toward "one kernel binary, per-core
   behaviour from DMA config". There are **two** binaries, differing only in
   which memory window their output pointers name.

### Why this matters for phase 2

The reproduction needs **two core programs per GEMM pair**, differing only in
output addressing — not one program plus configuration. And the inter-core
transfer costs *nothing* in the instruction stream: it is a store that happens to
land in a neighbour's memory. Any reproduction that routes the equivalent traffic
through DMA or objectfifos will pay for something FLM gets free.

This is also the third structural mechanism found by which FLM avoids memory
traffic, alongside the horizontal broadcast and the memtile's strided
layout transform.

---

## 2026-07-31 — Consolidation pass over both deliverables

Several findings this session superseded earlier ones, and corrections were made
in place as they happened. Read both documents end to end to check no superseded
claim survived in a section not revisited at the time.

**Three inconsistencies found and fixed:**

1. **`layer` §6b called col 5 row 2 "SiLU"** while the later roles section called
   it "SwiGLU". The refinement (2:1 I/O ratio => two inputs one output =>
   `silu(gate)*up`) had been applied where it was discovered but not to the
   inventory table above it. Reconciled: the table now says SwiGLU, and the
   SiLU-table-lookup detail is kept as the *mechanism* it evaluates.
2. **The GEMM pairing still read "chained partial sums... a K-split"** — retracted
   in the log a tick earlier, but the deliverable still carried it. Now states
   the pairing as established and marks *what* is split (K vs N) as open, since
   nothing observed distinguishes summing from concatenating.
3. **§7 "What is NOT established" still listed core roles and `p5` as
   unconfirmed.** Both had since been resolved. Rewritten to current state, and
   the genuinely open items added (what the GEMM pair splits, RoPE's site, the
   two still-inferred roles).

Also: §6b's heading was still "Core inventory and the SiLU core" though the
section had grown to cover roles, BDs, stream-switch connectivity and the
broadcast path. Retitled.

**§8 "Implications for reproduction" extended** with the three mechanisms by
which FLM moves operands without paying DRAM bandwidth — horizontal broadcast,
shared neighbour memory, strided DMA layout transform — plus the requirement for
two GEMM program variants. These are the phase-2-relevant structural findings and
they had been recorded in the log but never collected in the place a reproducer
would look.

**Method note.** A keyword sweep for known-superseded phrases caught only one of
the three; the other two needed reading the documents. Terminology drift
(SiLU/SwiGLU) and staleness-by-omission (a "not established" list that was never
pruned) do not match on the phrase you retracted — they match on the phrase you
kept. Worth doing this pass again before phase 2 begins, and worth doing it by
reading rather than grepping.

Final state: `flm-layer-dataflow.md` 570 lines, `flm-attn-dataflow.md` 188,
working log 2151.

---

## 2026-07-31 — Phase 3b's "free floor" is NOT free: bfp16ebs8 measured

Task #7. The goal doc proposes storing KV as `bfp16ebs8` as a **"free floor to
clear first"** — "9 bits/elem at **zero dequant cost**, since `mac_8x8_8x8T`
consumes it natively". Measured. It is not free.

### The split-array layout, and why it was needed

`struct v64bfp16ebs8 {v64int8 mantissa; v8int8 exponent;}` is **72 bytes,
packed** — not 64-byte aligned. A contiguous array of them misaligns every vector
load after the first, which the harness now refuses (measured 3.7 MACs/cycle
before the refusal was added).

The fix a real KV store would use is **separate, individually aligned mantissa
and exponent arrays**, assembled in registers — the type is `return_in_regs` and
lives in an EX register pair, so assembling it ought to be cheap. Implemented as
a new `split` variant. The generated loop confirms the alignment problem is gone:

```
vldb x5, [p7], #0x40      aligned 64 B mantissa load, post-increment
lda  r16, [p2, #8]        scalar exponent loads
mov  el3, r16             move into the E part of the EX register
vmac.f dm3, dm3, ex3, ex1, r1
```

### Result

| variant | cyc/call | MACs/cycle | % of 512 ceiling |
|---|---|---|---|
| `resident` (no memory traffic) | 1.624 | **315.2** | 61.6% |
| `split` (aligned, from memory) | 9.685 | **52.9** | 10.3% |
| contiguous 72 B (misaligned) | — | 3.7 | 0.7% |

**The split layout is 14.3x better than the naive contiguous one**, confirming
the alignment diagnosis. But against the other streamed modes measured this
session:

| mode | streamed MACs/cycle |
|---|---|
| `mac_4x16_16x16` int4, seq | **250.0** |
| `mac_8x8_8x8` int8, seq | **157.0** |
| `mac_8x8_8x8T` bfp16, split | **52.9** |

**bfp16 streams 3.0x worse than int8 and 4.7x worse than int4.**

### Why, and what it means for 3b

The loop is **16 bundles for 2 MAC calls = 8.0 cyc/call scheduled** (measured
9.7, the gap being outer-pass overhead — `inner` is only 8 iterations per pass).
Against int8's 6 bundles for 2 calls. The extra cost is **assembling the EX
register pair**: 4 scalar `lda` plus 4 `mov el*` per 2 MACs. There is no
*dequant*, exactly as the goal doc says — but there is substantial *assembly*,
and the doc's claim of "zero cost" does not survive contact.

Even **resident** bfp16 reaches only 61.6% of its ceiling, against int8's 92.9%,
so the EX-register handling costs something even when the operands never leave
registers.

**Important caveat — this bounds a layout, not the format.** My split variant
loads exponents one block at a time with scalar loads. A better arrangement would
load many exponents in one vector op and distribute them, amortising the scalar
work over more MACs. **52.9 MACs/cycle is a floor for the naive split layout, not
a proven ceiling for bfp16ebs8.** Anyone pursuing 3b should try the batched-
exponent layout before concluding the format is unusable.

**Recommendation for the plan:** the "free floor" framing should be dropped. On
the evidence, storing KV as bfp16ebs8 buys 9 bits/element but costs 3x the MAC
throughput of int8 in the straightforward layout — so it is a **trade**, not a
free win, and it should be measured against KVarN rather than assumed to be a
cheaper stepping stone toward it.

### Bounding the bfp16 caveat: 64 MACs/cycle is the layout's ceiling

The previous entry attached a caveat to the 52.9 figure — that batching exponent
loads might amortise the scalar work, so 52.9 bounded my layout rather than the
format. Resolved as far as it can be, and the caveat does **not** rescue bfp16.

**1. There is no wide bfp16 load in the ISA.** Searched the intrinsics headers:
`v64bfp16ebs8` and `v64bfp16ebs8_unaligned` are declared as types, but there is
**no load or store intrinsic** for them. The plan's note about a 576-bit
`FIFO_ST_BFP16` store path does not correspond to anything callable here. So the
EX pair must be assembled from separately loaded mantissa and exponent — the
split layout is the right one, not a workaround.

**2. The schedule does not improve with more iterations per pass.** Regenerated
at a 4x larger tile (`inner` 8 -> 32): the inner loop is **16 bundles / 2 vmac at
both sizes**. Identical schedule; only the 4-pointer pass prologue amortises
differently.

So the gap between scheduled and measured is pass overhead, and the ceiling is
the schedule:

```
scheduled  8.0  cyc/call -> 64.0 MAC/cyc     <- the layout's ceiling
measured   9.685 cyc/call -> 52.9 MAC/cyc    (inner=8; 1.69 cyc/call of prologue)
```

**Even at its ceiling, bfp16 is 2.5x worse than int8 and 3.9x worse than int4:**

| mode | streamed MACs/cycle |
|---|---|
| `mac_4x16_16x16` int4 | 250.0 |
| `mac_8x8_8x8` int8 | 157.0 |
| `mac_8x8_8x8T` bfp16, ceiling | **64.0** |
| `mac_8x8_8x8T` bfp16, measured | 52.9 |

**Conclusion for 3b: the "free floor" should be dropped from the plan.** Not
because the format is unusable, but because it is a *trade* — 9 bits/element for
2.5-4x the MAC cost — and the plan positions it as a free stepping stone that
clears the way for KVarN. It does not. If KV bandwidth is the goal, int4/int8
paths reach it more cheaply; if 9 bits/element specifically is wanted, that cost
has to be justified against KVarN directly.

**Honest limit on this bound.** The 16-bundle schedule is what the compiler
produced for this source; a hand-scheduled kernel might overlap the exponent
scalar path better. What is *not* available is a fundamentally cheaper mechanism —
no wide load exists, and the EX assembly is required by the instruction's operand
form. So 64 MACs/cycle is a compiler-output ceiling, not a hardware limit, but
there is no obvious headroom above it.

**Also recorded:** the 16 KB-tile run failed to build (`aiecc` exit 1) — a 16 KB
tile double-buffered in and out likely exceeds the 64 KB core L1. The static
schedule check made the run unnecessary, but the harness should probably validate
tile size against L1 before dispatching a build.

---

## 2026-07-31 — Phase 2 feasibility: the 5-buffer DPU signature is NOT the limit

Stepped back to pick the highest-value next item and found a contradiction in my
own notes worth resolving before phase 2 starts, because it looked like a blocker.

**The contradiction.** `decoder-layer-npu-scope.md` records FLM's xclbins using
"the standard 5-buffer DPU signature (`opcode, instr, ninstr, bo0..bo4`)", and
`layer.xclbin`'s own metadata confirms it — `connectivity` declares **6
connections**, at arg_index 1, 3, 4, 5, 6, 7:

```
"ip_layout": m_kernel_id 0x901, m_name "MLIR_AIE:MLIRAIE", subtype DPU
"connectivity": m_count 6 -> args 1 (instr BO), 3,4,5,6,7 (five data BOs)
```

But the measured decode command passes **50 arguments**. Five declared buffers,
fifty passed. If the 5 were a real limit, a whole-model single dispatch would be
impossible to express and phase 2's headline target would be blocked.

**Resolution: they are different things.** The `args` array in
`amdxdna_drm_exec_cmd` is a **driver-level buffer table**, not the kernel's
signature. Its purpose is to give the driver the BO handles it must resolve to
device addresses so that `DDR_PATCH` operations in the instruction stream can
patch shim BD address fields at launch (`amdxdna_ctx.c:589` copies
`arg_count * sizeof(u32)` handles). The xclbin's 5 connections describe the
command-packet layout; the patch table describes what the *instruction stream*
can bind.

**Both limits are generous:**

| limit | value | source |
|---|---|---|
| driver `arg_count` | **4095** | `MAX_ARG_COUNT`, `amdxdna_ctx.c:19` |
| `address_patch` `arg_idx` | **I32Attr** | mlir-aie `AIEX.td:1031` |

FLM's 50 is nowhere near either.

### Why this matters

**Phase 2's biggest target — one command carrying the whole model — is
expressible in stock mlir-aie.** The mechanism is `aiex.npu.address_patch` with
`arg_idx` 0..49 binding 50 host buffers, which is exactly what FLM does and
exactly what the (now-fixed) txn2mlir parser reads back. No custom runtime, no
signature extension, no driver change.

This was worth checking before starting phase 2 rather than discovering it
midway. It also retires a worry the plan implies but never states: that FLM might
be using a private dispatch path unavailable to us. It is not — same toolchain,
same driver interface, same patch mechanism, 24x more buffers bound to it.

**Caveat:** this establishes the *mechanism* is available. Whether mlir-aie's
host-side helpers and IRON's runtime plumbing pass 50 BOs without friction is not
tested — only that neither the driver nor the IR imposes the limit.

---

## 2026-07-31 — Phase 2 groundwork: the many-buffer probe, and two corrections

Built `tools/npu/flm/manybuf_probe.py` to test whether IRON/mlir-aie can bind as
many host buffers to one dispatch as FLM does (50). Two of my own conclusions
needed correcting along the way.

### Correction 1: "feasibility gate cleared" was premature

The previous entry concluded the whole-model single dispatch was expressible in
stock mlir-aie, on the grounds that the driver allows `MAX_ARG_COUNT` = 4095 and
`address_patch`'s `arg_idx` is an `I32Attr`. **Both true, and both the wrong
layer.** `aiecc` has its own gate:

```
error: device 'main' has 32 host buffer arguments, which exceeds the maximum
supported and verified count of 16. Reduce the number of host buffer arguments.
```

**mlir-aie caps host buffer arguments at 16. FLM binds 50.** Checking the driver
and the IR while missing the compiler in between is exactly the sort of partial
check that produced the earlier "no DMA output" mistake.

### What the cap actually is

`tools/aiecc/SidecarFiles.h:85-93`:

```c
// The NPU firmware command-chain (xrt::runlist) ABI requires every kernel to
// declare at least this many host buffer slots; fewer produces an undersized
// command slot and the runlist aborts. Extra slots are harmless.
inline constexpr int kMinHostBOs = 5;

// Conservative, hardware-verified ceiling on host buffer slots. AIETargetNPU
// folds the DDR translation offset so >5 buffers work, but counts above this
// are unvalidated and rejected.
inline constexpr int kMaxHostBOs = 16;
```

Two things fall out:

1. **The "5-buffer DPU signature" is `kMinHostBOs` — a FLOOR, not a ceiling.**
   Required by the firmware command-chain ABI. That fully explains why
   `layer.xclbin` declares exactly 5 data connections while binding 50, and
   retires the contradiction the previous entry was chasing.
2. **The 16 is a policy cap, self-described as "unvalidated", not a hardware
   limit** — and FLM binding 50 on this exact silicon is the existence proof.

Raised it to 64 locally and rebuilt `aiecc`. The compile-time rejection goes away.

### Correction 2: my probe measured the wrong thing

With the cap raised, 32 buffers then failed at *runtime* with
`ERT_CMD_STATE_ERROR`. That is not a buffer-count limit — it is my probe's
design. The first version gave **each pair its own ObjectFifo**, so 16 pairs
demanded **32 shim DMA channels**, against 8 shim tiles supplying 16 per
direction. It was measuring **channel exhaustion**, not buffer binding.

FLM binds 50 buffers through a handful of channels used **sequentially** — the
`DDR_PATCH` table rebinds addresses; it does not require a live stream per
buffer. Rewritten to use **one shared fifo** for all N transfers, which both
models FLM's pattern and isolates the variable under test.

**Lesson, and it is the same one as the MM2S channel-1 caveat:** a probe that
conflates two variables reports the tighter one and hides the question you asked.
Here the tell was that raising a *compile-time* limit produced a *runtime* error —
a different failure mode is a signal the experiment changed shape, not that the
limit merely moved.

Re-running with the shared-fifo design now.

### The real constraint is BD slots, and FLM's design is the prescribed answer

With `kMaxHostBOs` raised and the shared-fifo rewrite, the failure moved again —
to a third, and this time structural, limit:

```
error: 'aiex.dma_configure_task' op Too many simultaneously active buffer
descriptors on tile (0,0), which supports up to 16. Emit an
aiex.dma_free_task / aiex.dma_await_task to reuse BDs.
```

Three distinct limits, discovered in order by pushing on each:

| # | limit | value | nature |
|---|---|---|---|
| 1 | `aiecc` `kMaxHostBOs` | 16 | **policy** — self-described "unvalidated"; raised to 64, works |
| 2 | shim DMA channels | 16/direction | probe artifact — one fifo per pair, fixed by sharing |
| 3 | **active BDs per shim tile** | **16** | **hardware/structural** |

**And the third one explains FLM's design exactly.** Its staging transactions
program **4 BDs per column** and reuse them, with `DDR_PATCH` rewriting the
address word each dispatch. It never allocates a BD per buffer. So FLM's 50
bound buffers are **50 patchable addresses cycling through a handful of BDs** —
not 50 descriptors.

That is what the compiler error prescribes (`dma_free_task` / `dma_await_task` to
reuse BDs), and it is what the phase-1 analysis already found in the binaries.
The two pictures agree, which is the useful outcome: **FLM's structure is not
exotic, it is the documented way to exceed the BD count.**

**Corrected conclusion for phase 2.** "Bind 50 buffers to one dispatch" decomposes
into two requirements, only one of which is a real obstacle:

1. Raise `kMaxHostBOs` — a one-line policy change, validated here to 64, with
   FLM as the existence proof that the silicon is fine with it.
2. **Reuse BDs rather than allocating one per buffer** — the actual design
   requirement, and the thing to get right in the reproduction.

My probe still allocates a BD per transfer (`fill`/`drain` without an explicit
free), so it tops out at 16 active BDs. Extending it to use
`dma_free_task`/`dma_await_task` would push it to FLM's 50, but the structural
question is already answered and the probe has served its purpose: it found the
one policy cap that needed raising and identified the real constraint behind it.

**Method note.** Three failures, three different causes, each surfacing only
after the previous was removed — a compile-time policy cap, then a probe-design
artifact, then a hardware structural limit. The signal each time was that the
*failure mode changed* (compile error -> runtime error -> different compile
error). A limit that merely moved would have reported the same kind of error at a
larger N.

---

## 2026-07-31 — Session summary and handoff

Tool regression sweep: **all 7 pass** (`txn_check`, `txn_scan`, `cdo`,
`cdo_dma --tile`, `cdo_dma --graph`, `macbench`, `aiedis`). Two initial
"failures" were bad grep patterns in the sweep itself, not the tools — noted
because a regression check that reports false failures is worse than none.

### Scope completed

The kickoff asked for **Phase 0 (5 items), then Phase 1**. Both are done.

**Phase 0** — all five, plus two follow-ups. **Phase 1** — two deliverables
written (`flm-layer-dataflow.md`, `flm-attn-dataflow.md`), all five items either
answered or explicitly bounded.

### The findings that change the plan, ranked

1. **FLM submits 2 NPU commands per decoded token; hipfire submits 64.** At the
   measured ~37 us dispatch floor that is ~2.4 ms/token of pure submit latency.
   **Dispatch structure, not kernel bodies, is the dominant gap.**
2. **`mac_4x16_16x16` is 512 MACs/cycle, not 1024** — the plan's table was wrong
   by 2x. int4, int8 and bfp16 all issue at 512. int4's real advantage is fewer
   operand bytes on a load-bound machine (**1.59x** measured, streamed).
3. **Phase 3b's "free floor" is not free.** bfp16ebs8 streams at 52.9 MACs/cycle
   (ceiling 64) against int8's 157 and int4's 250 — a **trade**, not a free step
   toward KVarN.
4. **Three mechanisms move operands without paying DRAM bandwidth**: horizontal
   core-to-core broadcast, shared neighbour memory (no DMA, no cascade), and
   strided DMA layout transform in the memtile. A reproduction missing these
   will not reach 86% of fabric.
5. **KV is preallocated at full context** — 4 GiB for a 1B model. Capacity, not
   just bandwidth, is a KVarN target.

### Corrections made to `~/flm-re-fe-mutate-goal.md`

The MAC table (1024 -> 512); the 24-byte DDR_PATCH field map; `p1`/`p5` labels
(`p5` is the broadcast activation stream); "one fused decoder layer per dispatch"
-> one fused *model*; the attention tile width (pinned at 32 tokens); the
`k03`/`k47` split (columns 0-3 / 4-7); the col-5 core (SwiGLU, was
"unidentified"); the "free floor" framing; and a warning that the MoE is a hybrid
to which the Llama attention analysis does not port.

### Open, and what each needs

| item | needs |
|---|---|
| Re-take the 0.5 baseline on a quiet machine | the user's `flm serve` restarted first |
| RoPE's exact site | call-graph trace from `_send_rope_weights` (indirect call) |
| What the GEMM pair splits (K vs N) | not distinguished by anything observed |
| BD0's role in `attn` cores | — |
| Phase 2 proper | a decision to start; feasibility is established |

### Not committed

All work is uncommitted, across two trees. `hipfire-npu` is on `master`, and
`AGENTS.md` wants a topic branch, so nothing was staged. Local changes also exist
in `~/build/mlir-aie` (MERGE_SYNC build fix, txn2mlir parser, python binding
dependency, `kMaxHostBOs` raised to 64) on branch `merge-sync-txn-op`.

## 2026-07-31 — Phase 0.5 baseline RE-TAKEN on a quiet machine

The open item "re-take the 0.5 baseline on a quiet machine" is closed. Machine
verified quiet first: no `flm` processes, nothing holding an `accel0` fd. Same
harness, same parameters as the contended run (`--reps 3 --gen-lim 256`).
Results in `benchmarks/flm_baseline/results/flm-baseline-quiet.csv`.

### Decode — the contended numbers were NOT inflated; the opposite

| model | contended | quiet | delta |
|---|---|---|---|
| llama3.2:1b | 61.07 tok/s / 47.2 GB/s | **59.86 tok/s / 46.2 GB/s** | **-2.0%** |
| qwen3.6-moe:35b-a3b | 13.54 tok/s | **13.41 tok/s** | -1.0% |

The quiet machine is *slower*, consistently, across both models. So the earlier
caveat's worry — that a resident second NPU client depressed the baseline — is
not borne out. If anything the contended run was slightly optimistic.

Within-run spread is smaller than the gap (quiet llama 59.86/59.89/59.46 = 0.7%;
contended 61.07/61.24/60.61 = 1.0%), so the ~2% shift is probably real rather
than noise, but it is not large and the mechanism is unknown. A plausible one:
the resident `flm serve llama3.2:1b` held llama's weights in page cache, so the
contended run streamed warm and the quiet run streams cold. Not tested.

**Cross-check that favours the quiet figure.** The repo's independent historical
number is 60.1 tok/s / 46.4 GB/s. The quiet measurement (59.86 / 46.2) agrees to
within 0.4%; the contended one (61.07 / 47.2) was off by 1.6-1.7%. The quiet
figure is the better-corroborated one.

**Canonical baseline for phases 2 and 3 is now 59.86 tok/s / 46.2 GB/s** for
llama3.2:1b decode, and 13.41 tok/s for the MoE.

### Prefill — NOT comparable between the two runs, do not quote the delta

| model | contended | quiet |
|---|---|---|
| llama3.2:1b | 1774.53 t/s @ **7075** prompt tokens | 2056.33 t/s @ **3577** |
| qwen3.6-moe:35b-a3b | 185.52 t/s @ **8305** | 178.08 t/s @ **4127** |

The prompt lengths differ by roughly 2x between runs at the same
`--prefill-tokens` default, so the input file the harness picks up is not
pinned. Prefill throughput is strongly prompt-length dependent — this log
already records 121 t/s at 52 tokens vs 1623 t/s at 1828 for the same model — so
these two runs measure different things and the difference between them says
nothing about contention.

**`flm_bench.py` needs a pinned prompt before prefill is quotable.** Until then
only the decode half of the baseline is trustworthy, and prefill should be
re-taken at a fixed, stated prompt length.

### Side effect: the user's `flm serve` was restarted

Killed in error by this work earlier (recorded above). Restarted as
`flm serve llama3.2:1b`, PID captured explicitly at launch per the rule that
incident produced. It came up on port **52625**, bound to 127.0.0.1. That port
is not in `~/.config/flm/`, not in the shell profile, and not in the launching
environment, so it appears to be chosen per-process — meaning it probably does
NOT match whatever port the original 14h58m-old instance had. Anything holding a
hardcoded port will need it restarted deliberately.

## 2026-07-31 — Phase 1 open item CLOSED: the GEMM pair concatenates (N-split)

Open since the pairing was found: two GEMM cores per output group, the lower
writing into the upper's memory, but *what* is split — whether the upper **sums**
the two contributions (K-split) or **concatenates** them (N-split) — was
explicitly recorded as not distinguished by anything observed.

It is an N-split. Two independent sources, which agree.

### Source 1 — the code: there is no combining step

Disassembled the pair (0,2) upper / (0,3) lower and diffed with addresses
stripped. 64 lines differ, and every one of them is register housekeeping:

| kind | count |
|---|---|
| `lda` / `st` (stack spill+reload, different regalloc) | 21 / 20 |
| `movs` / `mova` / `movxm` / `ltu` / `and` / `or` / `add` (pointer + index setup) | 17 |
| `nopa` / `nop` (schedule padding) | 6 |

**Differences in `vmac`, `vadd`, `vsub`, `vmul`, `vconv`, `vst`: zero.** The
vector store sequence is byte-identical in both — same opcodes, same pointer
registers, same immediate offsets. A K-split *requires* the upper core to read
the lower's partial sums and add them into its accumulator; no such code exists
in either program. Both cores run the same computation and differ only in where
the result lands.

### Source 2 — the buffer descriptors

The only difference in the operand pointers is the memory window:

| pointer | (0,2) upper | (0,3) lower |
|---|---|---|
| p0 | `0x73c00` | `0x43c00` |
| p3 | `0x75400` | `0x45400` |

Same module offsets (`0x3C00`, `0x5400`), different window. Authoritative AIE2
mapping (`AIETargetModel.h`, the NPU model at lines 632-640): internal =
**East = 0x70000** with no row-parity term, **South = 0x40000**. So the lower
core writes into its *south* neighbour, which is the upper core — confirming the
established hand-off direction from a second place.

Those offsets are exactly the upper core's two output BDs: BD4 at module
`0x3C1C` and BD5 at `0x541C`, `BUFFER_LENGTH=33` words = **132 B each**. The
lower core has **no output BD at all** (4 BDs, all inputs, vs the upper's 6).

So: two equal-sized output buffers, one per core of the pair, both resident in
the upper core's memory, both shipped by the upper core's DMA. Equal sizes and
no reduction is concatenation; a sum would need one buffer, not two.

### What this fixes downstream

- **The reproduction needs two program variants per pair differing ONLY in the
  output pointer base** — own (`0x7....`) vs south neighbour (`0x4....`).
  Everything else, including the whole compute body, is identical.
- **"8 result streams, not 16" means 8 streams each carrying two concatenated
  N-groups**, not 8 summed results. Anything sizing the output stage off 16
  independent results, or off 8 reduced ones, would be wrong.
- **Favourable for 3a (oq4++).** N-split means each core owns distinct output
  channels, and oq4++'s per-group scales are per *output channel* — so scale
  data partitions cleanly across the pair with no cross-core reduction and no
  shared accumulator. The K-split reading would have implied the opposite.

### Method note

The first diff attempt reported `IDENTICAL` — because a `cd` had been reset and
both `sed` inputs did not exist, so it compared two empty streams. Caught only
because "identical" was the wrong answer for programs known to differ in their
pointer setup. Same class as the `1.00 cyc/insn` and generic-form-MLIR false
passes already recorded: **a successful-looking result from a command that never
ran on the intended input.**

## 2026-07-31 — Phase 1: RoPE's site CONFIRMED (cols 3-4), closing two open items

Two open items were coupled: "where RoPE is applied" (narrowed to cols 3-4 but
explicitly *not proven*) and "the *function* of the cols 3-4 shuffle/elementwise
pipeline" (pairing confirmed, computation not). Three independent sources now
agree that this pipeline applies RoPE.

### 1. Arithmetic signature

RoPE rotates each `(x, y)` pair: `x*cos - y*sin` and `x*sin + y*cos`. That needs
a multiply-**subtract** and a multiply-**accumulate**, over deinterleaved
even/odd lanes. The cols 3-4 elementwise cores (rows 3, 5) have exactly that:

| core | signature |
|---|---|
| (3,2), (4,2) — shuffle | `vshuffle:91`, almost no arithmetic |
| (3,3), (4,3) — elementwise | `vmul.f:86  vadd.f:73  vshuffle:51  vmsc.f:24  vmac.f:19` |

`vmsc.f` (multiply-subtract) appears in the elementwise cores and essentially
nowhere else in the layer kernel. It is the operation the rotation's first half
requires.

### 2. Operand widths — the decisive one

Both cols 3-4 cores carry the same two buffer widths, each double-buffered:

| buffer | bytes | bf16 | Llama-3.2-1B geometry |
|---|---|---|---|
| BD pair A | **4096** | 2048 | **Q** = 32 heads x 64 head_dim |
| BD pair B | **1024** | 512 | **K** = 8 kv_heads x 64 head_dim |

From `config.json`: `hidden=2048, heads=32, kv_heads=8, head_dim=64`. Q is
32*64 = 2048 bf16 = 4096 B; K is 8*64 = 512 bf16 = 1024 B. Both match exactly.

**V is also 512 bf16 = 1024 B, and there is no third stream** — only two distinct
widths, each appearing twice as a ping-pong pair. RoPE is applied to Q and K and
never to V, so "Q and K present, V absent" is the signature, and it is what the
BDs show.

### 3. Symbols

Already recorded: RoPE symbols (`_set_rope_weights`, `_send_rope_weights`,
`_rope(buffer<bfloat16_t>&, int)`) exist only in `libllama_npu.so`, and
`libmha.so` has none — so RoPE lives in `layer.xclbin`, not `attn.xclbin`. That
narrowed it to this kernel; the widths locate it within the kernel.

### What remains inferred, deliberately

- **The division of labour inside the pair.** Op mix says the shuffle core (rows
  2,4) deinterleaves and the elementwise core (rows 3,5) rotates. Consistent, and
  it matches the established vertical pairing, but not separately proven.
- **These cores may do more than RoPE.** `vadd.f:73` is more addition than the
  rotation alone needs, so the elementwise core plausibly carries another
  elementwise stage as well. The RoPE finding does not claim the cores do *only*
  RoPE.

Both are narrower than the questions they replace, and neither blocks phase 2:
the reproduction needs the pipeline's dataflow and operand widths, which are now
pinned.

## 2026-07-31 — Phase 1: col 2 row 3 decomposed — it is the QKV split

Last unresolved core role in `layer.xclbin`, previously recorded only as
"6144 in / 4096 out, not decomposed". The buffer widths decompose exactly.

| BD | bytes | bf16 | Llama-3.2-1B geometry |
|---|---|---|---|
| BD0 | **6144** | 3072 | **fused QKV** = Q 2048 + K 512 + V 512 |
| BD1 | **4096** | 2048 | **Q** = 32 heads x 64 |
| BD2-BD5 | **512** x4 | 256 each | **4 kv_heads x 64** — K and V, each in two halves |
| BD6 | 128 | 64 | one head_dim |

`2048 + 512 + 512 = 3072` is exact, and no other grouping of this model's
dimensions gives 3072. So **(2,3) receives the fused QKV projection output and
splits it into separate Q, K and V streams.**

Two independent corroborations:

1. **The Q output width matches its consumer.** BD1 is 4096 B, exactly the Q
   buffer size on the cols 3-4 RoPE cores. The split core's Q output feeds the
   RoPE pipeline's Q input.
2. **The four 512 B buffers reproduce the `k03`/`k47` split.** K is 8 kv_heads x
   64 = 512 bf16 = 1024 B; half of it, 4 heads, is 256 bf16 = **512 B**. Four
   such buffers = K in two 4-head halves and V in two 4-head halves. That is the
   same 0-3 / 4-7 head partition already read independently out of the symbol
   table (`get_k03_offset` / `get_k47_offset` / `get_v03_offset` /
   `get_v47_offset`). Two unrelated sources landing on the same partition.

### Confidence and what is still assumed

The width arithmetic is exact and cross-corroborated. What is *not* directly
observed here is the **direction** of each BD — the decode above takes the
existing "6144 in / 4096 out" reading and shows the widths are consistent with a
QKV split; it does not independently prove which BDs are reads and which are
writes. The consumer-side match (BD1 = the RoPE cores' Q buffer) makes the
direction very likely but is not a direct measurement.

BD6 (128 B = 64 bf16 = one head_dim) is not accounted for by the split itself.

**All core roles in `layer.xclbin` are now decomposed** — GEMM, RMSNorm (2,2),
SwiGLU (5,2), RoPE (cols 3-4), QKV split (2,3) — leaving as open only the
*internal* division of labour inside the RoPE pair and BD0's role in the `attn`
cores.

## 2026-07-31 — Phase 1 (attn): BD0 CLOSED, and BD3/BD4 were misread as an input

`flm-attn-dataflow.md` listed BD0's role as open ("8192 B, single, not
ping-ponged, its own lock. Consistent with K+V for 32 tokens or an output
accumulator; not distinguished"). Settled by reading the DMA **channel queue
registers** out of the CDO rather than inferring from buffer sizes.

### Method

AIE2 core-tile DMA channel registers, tile (0,2) of `attn.xclbin`: S2MM ch0/ch1
control at `0x1DE00`/`0x1DE08` and start queues at `0x1DE04`/`0x1DE0C`; MM2S
ch0/ch1 at `0x1DE10`/`0x1DE18` and `0x1DE14`/`0x1DE1C`. The queue register's low
nibble is the channel's starting BD. Scanned the CDO for writes to those
addresses.

```
S2MM0 queue = 0x0  -> start_bd=0     S2MM0 ctrl = 1
S2MM1 queue = 0x1  -> start_bd=1     S2MM1 ctrl = 1
MM2S0 queue = 0x3  -> start_bd=3     MM2S0 ctrl = 1
MM2S1        (ctrl written, no queue -> never started)
```

### Result — the channel/direction binding

| BD | channel | direction | bytes | source / sink |
|---|---|---|---|---|
| BD0 | S2MM0 | **input** | 8192 | memtile (DMA0 <- SOUTH0) |
| BD1/BD2 | S2MM1 | input, ping-pong | 4096 | east neighbour (DMA1 <- EAST3) |
| BD3/BD4 | MM2S0 | **OUTPUT**, ping-pong | 128 | -> memtile |

**BD0 is an input from the memtile.** 8192 B is *uniquely* the size of the
memtile's 4-D strided output BDs (section 3 of the attn doc), and it equals
`2 x 32 tokens x 64 head_dim` in bf16 — K and V for one 32-token flash tile,
delivered already transposed by the memtile DMA. Open item closed, and it
confirms rather than disturbs the section-3 finding that the cores never reshape.

### CORRECTION to `flm-attn-dataflow.md` section 2

The doc reads BD3/BD4 (128 B) as "**1 x head_dim** — the Q vector for one head
(decode, M=1), double-buffered", i.e. an input. **MM2S0 starts at BD3, so
BD3/BD4 are the core's OUTPUT**, not its Q input. 128 B = 64 bf16 = one
head_dim, which is the attention *result* for one query — the same arithmetic,
the opposite direction.

This is the size-inference failure mode again: 128 B = head_dim is consistent
with both a Q input and an O output, and the size alone cannot separate them.
The channel register can, and disagreed.

### What this opens

Both input channels are now accounted for as KV-side streams (BD0 = K+V tile
from the memtile; BD1/BD2 = 4096 B from the *east neighbour*, matching the
pairwise column fan-out in section 4). **Where Q arrives is therefore no longer
obvious** — the previous answer was BD3/BD4, which is wrong.

Two candidate readings, not distinguished:
- BD1/BD2 (4096 B = 2048 bf16 = 32 x 64) is a **Q tile of 32 query positions**,
  which fits `attn.xclbin` being the *prefill* kernel (established in
  `flm-layer-dataflow.md` section 1), with the 128 B output emitted once per
  query through the ping-pong.
- BD1/BD2 is a second KV stream and Q arrives by another route.

Note the doc's own BD3/BD4 gloss says "decode, M=1", which sits badly with
`attn.xclbin` being the prefill path. That inconsistency between the two
documents predates this entry and should be resolved when Q's route is settled.

---

## 2026-07-31 — Q's route settled, and the head decomposition retracted

Phase 1's two remaining gating items (kickoff `~/flm-phase23-kickoff.md`): where
Q arrives in `attn.xclbin`, and the cross-document inconsistency it exposed.
Both closed. The answer overturns more of `flm-attn-dataflow.md` than expected.

### What was tried, in order

**1. Neighbour-memory hypothesis — ruled out.** `layer.xclbin` hides a third of
its dataflow in shared neighbour memory, so the first guess was that Q arrives
the same way. Disassembled all 32 attn cores and extracted every address
immediate: the set is identical on every core and lies entirely in `0x7xxxx`
(own module). No `0x4xxxx`/`0x5xxxx`/`0x6xxxx` neighbour window anywhere. Dead
end, but it also rules out any inter-core exchange in this kernel.

**2. Per-core base-address hypothesis — ruled out.** If a core selected its own
slice of a broadcast, the offset would show as a differing immediate. It does
not: all 32 cores carry the identical address set, the DMA BDs are byte-identical
(already known), and the CDO's per-core data-memory init is 36 B at `0x30a4`
whose one varying word tracks program *size* (0x2770/0x2780/0x2790 against
10596/10612/10628 B), not position.

**3. Route resolution — the structural fact.** Walked every core's DMA source
back to its origin through the stream-switch graph. Result is stark and was not
visible in the per-tile view:

- **DMA0** (BD0, 8192 B) has **16 distinct origins** — memtiles (0,1), (2,1),
  (4,1), (6,1), MM2S channels 0-3, one per row — each feeding **exactly 2 cores**
  (an even column and the odd column beside it).
- **DMA1** (BD1/BD2, 4096 B) has **one** origin, memtile (3,1) MM2S0, feeding
  **all 32 cores**, up to 8 hops deep, every hop circuit-switched.

That immediately falsified the doc's "BD1/BD2 comes from the east neighbour" —
it is an array-wide broadcast, not a pairwise hop.

**4. The paradox, and what broke it.** Two cores receiving byte-identical data on
*both* channels and running identical programs would compute identical results.
So the programs are not identical. They are not: sizes differ (10596 / 10612 /
10628 B), and diffing the prologue shows two per-core immediates,
`r0 = row - 2` and `r1 = col & 1`. Across all 32 cores those are the only
per-core constants, and md5 on the extracted program memory confirms **exactly 8
distinct binaries**, one per `(row, parity)` class. A core's identity is
`(row, column parity)` plus which DMA0 stream it is wired to.

**5. Cross-model check — this is what retracted the head mapping.** Ran the same
extraction on three other `attn.xclbin`s with deliberately different head
geometry:

| model | q heads | kv heads | head_dim | cores | BD sizes |
|---|---|---|---|---|---|
| Llama-3.2-1B | 32 | 8 | 64 | 32 | 8192 / 4096 / 4096 / 128 / 128 |
| Llama-3.2-3B | 24 | 8 | 128 | 32 | same |
| Qwen3-0.6B | 16 | 8 | 128 | 32 | same |
| Gemma3-1B | 4 | **1** | 256 | 32 | 16384 / 8192 / 8192 / 128 / 128 |

**Always 32 cores; the buffer sizes do not track head count; the 128 B output is
128 B at head_dim 64, 128 and 256 alike.** So `32 cores == 32 query heads`,
`8 columns == 8 KV heads`, `4 rows == GQA ratio 4` and the `k03`/`k47` column
mapping are all **retracted**. They were read off Llama-3.2-1B, where the model's
head counts coincide with the array's dimensions by accident.

Also confirmed by hashing: `attn.xclbin` is shared byte-for-byte across models
grouped by attention geometry — Llama-3.2-1B ships the same file as LFM2-1.2B,
LFM2-2.6B and LFM2.5-1.2B (all 32 q / 8 kv / head_dim 64), Llama-3.2-3B the same
file as Phi4-mini. So the xclbin keys on attention shape and nothing else.

**6. Runtime cross-check.** Dumped argument BO sizes per submission with the
existing interposer (`NPU_ARGS_OUT` / `NPU_ARGS_MATCH`). The attention dispatch
takes exactly three buffers — 256 MiB KV cache, 14 MiB Q, 14 MiB O — against five
DDR ingress columns in the array (0, 2, 4, 6 carrying the per-pair streams, 3
carrying the broadcast).

### Conclusion

**BD0 = Q** (64 query positions per column pair, 32 per core by `col & 1`);
**BD1/BD2 = K/V** (array-wide broadcast, the flash inner-loop operand);
BD3/BD4 = output rows. Four independent supports, written up in
`flm-attn-dataflow.md` section 6. The strongest is Gemma3-1B: with one KV head
and the identical topology, the alternative reading requires duplicating a single
KV head across 16 distinct streams while broadcasting Q.

The output arithmetic closes as an extra check: the memtile's output-collection
BD is 4096 B strided with D1 stride 128 B = **32 rows of 128 B**, matching 32
query positions per core.

**And `attn.xclbin` is the prefill kernel** — 32 cores that do not scale with
head count, each holding a private tile of query positions against a broadcast KV
stream, is query-tile-parallel flash attention. `flm-layer-dataflow.md` was right;
the "decode, M=1" gloss in the attn doc was an artifact of the BD3/BD4 misread.

### The failure mode this document keeps hitting

Three wrong answers in a row on the same buffers, all from **size matching**:
128 B "= head_dim, so Q input"; 8192 B "= uniquely the memtile's strided output,
so K+V". Sizes are ambiguous — 8192 B is *not* unique, because memtile (3,1)
emits strided too, at a different size, and nobody had looked at (3,1). Every
correction so far has come from a *different kind* of evidence: a channel queue
register, a route walk, a second model, a runtime argument list.

### Prefill dispatch, decoded as a side effect

The same argument dump closed a separate open item in `flm-layer-dataflow.md` —
the intra-layer prefill ordering. Per layer, in order:
`dequant -> q -> k -> v -> attention -> o -> gate -> up -> down`. Buffer sizes
identify each context with no remainder, and two things fall out:

- **Prefill dequantizes the whole layer to bf16 up front** (37 MiB q4_1 in,
  116 MiB bf16 out = 60.9 M params x 2 B exactly) and runs bf16 GEMMs. The 42-op
  q4_1 dequant chain that dominates the decode kernel is not on this path.
- **k_proj's and v_proj's outputs are passed to nothing.** The KV cache fill, and
  RoPE on this path, happen on the host between dispatches — consistent with
  `fill_kv_cache` and `_rope` being host-side symbols.

### Failure worth recording: two benchmark runs raced

The first prefill baseline attempt reported llama decode at **42.54 tok/s**
against the established 59.86, with prefill and the whole MoE row empty. Cause
was self-inflicted: a `nohup ... &` launch appeared to have died with its shell
but had not, so a second `flm_bench.py` was started alongside it and the two
contended for NPU hardware contexts (`CREATE_HWCTX` then failing with
`err=-22`). Killed by explicit PID, re-run single. **A benchmark number that is
28% off the established baseline should be read as a broken harness first.**

### `flm_bench.py` prompt pinning — the goal doc's diagnosis was wrong

The plan records that `flm_bench.py` "was not pinning its prompt". It is: for a
given `--prefill-tokens`, `make_prompt_file` is deterministic — regenerating it
twice gives md5 `9e73e8...`, which is also the md5 of the checked-in
`results/prefill_prompt.txt` at the default 2048. The historical ~2x length
difference therefore came from the two runs using **different
`--prefill-tokens`**, not from a non-deterministic prompt. Fixed the real gap:
the CSV now records `prefill_tokens` alongside the measured `prompt_tokens`, so
a run's setting is recoverable from its own output.

### Prefill baseline, 2026-07-31 (re-run, single instance)

| model | `--prefill-tokens` | prompt tok | prefill tok/s | decode tok/s (same run) |
|---|---|---|---|---|
| llama3.2:1b | 2048 | 3577 | **2064.8** | 61.31 |
| qwen3.6-moe:35b-a3b | 2048 | 4127 | **178.2** | 13.55 |

Medians of 3, spread under 1.6%. `flm serve llama3.2:1b` was resident, so this is
the **contended** condition — confirmed by the decode column reproducing the
contended baseline (61.07 / 13.54) to 0.4% / 0.1% rather than the quiet one
(59.86 / 13.41). That also puts an upper bound of ~2% on the condition's effect,
so the prefill figures are usable as a Phase 2 target with that caveat attached.

MoE prefill is **11.6x slower** than llama at a comparable prompt length, against
~2.4x predicted by active-parameter scaling — a long prompt activates
essentially all 256 experts, so MoE sparsity is a decode-only benefit.

---

## 2026-07-31 — Phase 2 milestone 1: one-dispatch weight streaming

Phase 2 starts with the gap the phase-1 measurements say dominates: FLM decodes
with **2 dispatches per token at 46.2 GB/s**, hipfire's own path with **~96
dispatches at ~10 GB/s effective** (`decoder-layer-npu-scope.md`). That is a
dispatch-structure problem before it is a kernel problem, so it is worth
measuring before any GEMM arithmetic exists to confound it.

New tool: `tools/npu/flm/dispatch_bw_probe.py`. Binds N host buffers to ONE
dispatch, streams every byte into the array through `--workers` parallel
ObjectFifos, and reports the achieved rate. The cores acquire and release without
reading — the DMA moves the bytes either way, so this measures the delivery
structure, not the arithmetic.

### Toolchain re-verified first

`vector_scalar_mul` generates, compiles and runs on hardware (`PASS!`,
126 us avg), so the phase-2 path — our own mlir-aie source to silicon — is live.
The C++ host target does not build (wants `g++-13`, absent); the Python run path
is the one to use.

### BD reuse works, and it is the fix `manybuf_probe.py` predicted

`manybuf_probe.py` ended at a **compile-time** wall: "Too many simultaneously
active buffer descriptors on tile (0,0), which supports up to 16", with the
compiler itself prescribing `dma_free_task`/`dma_await_task`. IRON exposes that
as `TaskGroup`: fills are grouped, and each group is awaited and freed before the
next opens. **With grouping, designs binding 24-48 buffers now compile.** The
active-BD wall is gone.

### The wall moved to RUNTIME, and it is at ~20 buffers

| test | result |
|---|---|
| 8 bufs x 256 KiB, 4 workers | 11.7 GB/s |
| 16 bufs x 256 KiB, 4 workers | 18.6 GB/s |
| **20 bufs x 256 KiB, 4 workers** | **18.4 GB/s** |
| 24 bufs x 256 KiB, 4 workers | **`ERT_CMD_STATE_ERROR`** |
| 48 bufs x 1 MiB, 1-16 workers | `ERROR` (`TIMEOUT` at 1 worker) |

`dmesg` shows the firmware **hanging**, not rejecting: `aie2_tdr_detect: TDR
timeout detected`, `DPU PC: 0xffffffff`, `TXN OP ID: 0xffffffff`. So the command
is accepted and the DPU never runs it.

Three discriminating tests, all holding the count at 24:

| variable moved | result |
|---|---|
| one task group instead of six (`--group 24`) | still `ERROR` |
| six groups instead of three (`--group 4`) | still `ERROR` |
| 8 fifos instead of 4 (`--workers 8`) | still `ERROR` |
| **bytes up 12.8x at 20 buffers** (20 x 1 MiB) | **works — 34.9 GB/s** |

**The axis is the buffer count alone.** Not bytes, not group size, not fifo
count. Somewhere between 20 and 24 host buffers on one dispatch, the generated
command stops being something the firmware will execute.

### Why this does not block phase 2

FLM binds 50 because it allocates **per layer, per role** — 16 x (weights,
workspace, KV) + 2. A reproduction is not obliged to keep that decomposition. The
arithmetic works out favourably:

- llama3.2:1b streams **772.3 MB per token**, its per-layer weight buffer is
  **37 MiB**, and there are **16 layers**.
- So **16 weight buffers + a handful of activation/workspace buffers ~= 20
  arguments** carries an entire token's weight traffic in one dispatch — right at
  the measured ceiling, without needing to cross it.

The ceiling is worth chasing later (it is a real defect in something between IRON
and the firmware, and 50 demonstrably works for FLM), but it is not on the
critical path. **Packing fewer, larger buffers is the cheaper answer and it is
also closer to what the reproduction wants anyway.**

### Trap: Python 3.14 constant slices break mlir-aie's jit cache

Any design generator containing a **constant slice** — `args[:50]` — fails to
compile with a bare `ValueError: unmarshallable object`. Python 3.14 folds the
slice into `co_consts`, and mlir-aie's jit cache hashes the generator with
`marshal.dumps(code, 4)`; marshal gained slice support only in version 5.
Confirmed in isolation:

```
def f(a): x = a[:5]        -> marshal.dumps(f.__code__, 4)  ValueError
def g(a): [a[i] for i in range(5)]  ->  OK
```

`_hash.py` pins version 4 deliberately ("marshal.version is 4 through 3.13 and 5
from 3.14"), so this bites every 3.14 user who slices in a generator. Worked
around locally by indexing instead of slicing, in both this probe and
`manybuf_probe.py` — **which had the same latent bug and would not have run as
committed.**

### RESULT: one dispatch moves a decode-sized weight set FASTER than FLM

512 MiB (16 buffers x 32 MiB — the shape of llama3.2:1b's 16 per-layer weight
buffers) bound to **one** dispatch, medians of 10 iterations after 2 warmups:

| feed streams | shim columns used | GB/s | vs FLM 46.2 | vs 56.5 roof |
|---|---|---|---|---|
| 4 | 2 | 46.5 | 1.01x | 82% |
| **8** | 3 | **55.9** | **1.21x** | **99%** |
| 16 | 7 | 55.5 | 1.20x | 98% |

(Stream count is read back from the generated xclbin with
`cdo_dma.py --graph`; it is the number of shim NORTH routes, not an assumption.)

**Three things this settles.**

1. **Dispatch structure is not the wall it looked like.** One command can carry a
   whole token's weight traffic and deliver it at **99% of the fabric roof**.
   Nothing about the 2-dispatches-per-token design is out of reach.
2. **The lever is stream count, not column count.** 4 streams reach 46.5 GB/s
   (~11.6 each, close to the 14.4 GB/s single-stream figure in
   `npu-memory-bandwidth-cache-characterization.md`); 8 streams saturate. Beyond
   8 there is nothing left to win — 16 streams over 7 columns measures no better
   than 8 over 3.
3. **hipfire's ~10 GB/s effective decode is a dispatch-count artifact, not a
   delivery limit.** The same silicon and the same toolchain move bytes 5.6x
   faster when the work is in one command instead of ~96.

**And FLM is not at the roof either.** Its 46.2 GB/s is 82% of what one dispatch
demonstrably sustains. That is the same 82% the 4-stream row measures, which is
suggestive given FLM feeds its 16 GEMM cores from **4 memtile columns**
(`--origins` on `layer.xclbin`: 16 private weight streams, all originating in
memtiles (0,1), (1,1), (6,1), (7,1)) — but the probe's streams are not FLM's, and
FLM's number includes compute while the probe's does not. **Not established, and
worth establishing**: if FLM's decode really is ingress-width-limited, a
reproduction that widens weight ingress would beat it by ~20% before any change
to the kernel body, which would make it a phase-2 *result* rather than a phase-3
mutation.

### Method note

The probe reports FAIL rather than dying, and the four discriminating tests above
were only cheap because of it. Two failure modes appeared —
`ERT_CMD_STATE_TIMEOUT` at one worker and `ERT_CMD_STATE_ERROR` at two or more —
and per the standing lesson in this log, a *changed* failure mode means the
experiment changed shape rather than the limit merely moving. That is what
prompted holding workers fixed while sweeping the count, which is what isolated
the buffer count as the only axis that mattered.

### The number is not a no-op — verified in the same dispatch

The cores in this probe acquire and release without reading, so a design whose
DMAs silently did nothing would report a spectacular rate. `--verify` closes
that: one extra tile rides the **same** dispatch through a forwarding fifo and is
compared byte for byte on the way out.

```
16 x 32 MiB, 8 workers, --verify:  55.2 GB/s  (1.20x FLM, 98% of roof)
```

Against 55.9 GB/s without it — the 0.7 GB/s is the check transfer itself. **The
bytes are real.** Worth the fifteen lines: three of the results this log records
as corrections looked like successes first.

(Two IRON constraints surfaced writing it, both worth knowing: `Out` must be
imported into the generated namespace alongside `In`, and **explicit
`TaskGroup`s cannot be mixed with the implicit default group** — a verify
transfer added outside a group fails with
"Mixing explicit task groups and the default task group is prohibited".)

---

## 2026-07-31 — The phase-2 runtime wall is exactly 20 host BOs, and why FLM has no such wall

Milestone 1 left "a runtime wall at ~20 buffers" with the firmware hanging
(`aie2_tdr_detect`, `DPU PC 0xffffffff`). Located precisely.

### It is a count, and the count is 20

| configuration | total host BOs | result |
|---|---|---|
| 20 data, no verify | 20 | **PASS** 9.7 GB/s |
| 18 data + 2 verify | **20** | **PASS** 8.9 GB/s, bytes verified byte-for-byte |
| 19 data + 2 verify | **21** | FAIL `ERT_CMD_STATE_ERROR` |
| 21 data, no verify | 21 | FAIL |

**Exactly 20 total host buffer objects per dispatch.** It does not matter how
they divide between data buffers and the verify pair — 18+2 and 20+0 both pass,
19+2 and 21+0 both fail. The `--verify` run passing at 20 also shows the data
path is *correct* there, so this is a pure count ceiling, not corruption.

**Not an active-BD limit.** `--group` 2, 4 and 8 all fail identically at 24
buffers, so TaskGroup recycling — which fixed the *compile-time* wall — has no
bearing on this one.

### A false signal I chased, and the grep that caused it

An intermediate result appeared to show that cutting DMA pushes 16x (tile size =
buffer size, one push per buffer) let 24 buffers through at "0.0 GB/s". It did
not. That run failed at **compile** time, and the extraction grep
(`[0-9.]+ GB/s`) matched the probe's trailing `best 0.0 GB/s` summary banner
rather than a measurement row. A failure was scraped as a result.

Worth recording because the tell was there and I nearly missed it: **0.0 GB/s is
not a plausible measurement.** The probe's own docstring warns that a design
whose DMAs silently did nothing would report a *spectacular* rate; the mirror
case — an implausibly small one — deserves the same suspicion. Re-running with
`--verify` surfaced the real `[aiecc] Compilation failed`.

### Why FLM has no such wall: it does not bind weights as kernel arguments

FLM binds **50** buffers to one dispatch on this same hardware, so 20 is not a
silicon limit. The difference is structural, and the earlier phase-1 work already
recorded both halves without connecting them:

- `layer.xclbin` declares only the **`kMinHostBOs` = 5** DPU signature
  (`connectivity` m_count 6). Its 50 buffers are **not** kernel arguments — they
  reach the driver as the `args` array of `amdxdna_drm_exec_cmd`, a buffer table
  that `DDR_PATCH` ops index by `arg_idx` to patch shim BD addresses at launch.
- IRON binds every host buffer as a **named kernel parameter**, which flows into
  the ERT command packet. That packet is what runs out at 20.

So the two paths are not the same mechanism at different scales — they are
different mechanisms. FLM's 50 "arguments" are patch-table entries; IRON's are
kernel arguments.

**Consequence for phase 2, and it is a design correction rather than a tuning
knob:** streaming a whole model through one dispatch must **not** bind each
weight buffer as a kernel argument. It needs few kernel args plus
`aiex.npu.address_patch` entries carrying the rest — exactly FLM's shape, which
is also why its xclbin declares the bare minimum signature. Raising
`kMaxHostBOs` was necessary to get this far but is not the path; it lifts a
compile-time gate on a mechanism that hits a firmware ceiling at 20 regardless.

**Still open:** where the 20 comes from precisely. `ERT_START_NPU = 20` in
`amdxdna_ctx.h` is an opcode value and a coincidence. The ERT command packet's
size, or XRT's per-kernel argument binding, is the place to look — but the design
conclusion above does not depend on the answer.

### The 20-BO wall is a firmware limit the driver does not validate

Chased where 20 comes from. Not pyxrt — `pyxrt.kernel.__call__(*args)` is
variadic with no fixed arity. The relevant bound in the driver is:

```c
#define MAX_DPU_ARGS_SIZE (34 * sizeof(u32))      /* aie2_msg_priv.h:520 */
...
if (cmd_len < sizeof(*sn) || arg_sz > MAX_DPU_ARGS_SIZE)
        return -EINVAL;                            /* aie2_message.c:1037 */
```

**34 words. But the wall is at 20, and it does not present as `-EINVAL`.**

That mismatch is the interesting part. Had 20 BOs consumed 2 words each (64-bit
addresses), 20 would be 40 words and the *driver* would have rejected it
cleanly. It does not — 20 passes and runs. So each arg is one word here, 20 BOs
is 20 words, comfortably inside the 34-word cap. And 21 BOs is 21 words, **also**
inside the cap — yet the command is accepted and the DPU then hangs
(`aie2_tdr_detect: TDR timeout`, `DPU PC 0xffffffff`, `TXN OP ID 0xffffffff`).

**So the driver validates a bound the firmware does not actually honour.** The
documented capacity is 34 args; the firmware copes with 20. Exceeding the real
limit produces a device hang rather than a rejected ioctl, which is why this
looked like a mysterious runtime failure rather than a limit being hit — nothing
in the software stack says no.

Worth reporting upstream: a driver-side check that is looser than the hardware's
real capacity turns a clean `-EINVAL` into a TDR timeout and a wedged context.

**This does not change the phase-2 conclusion, it sharpens it.** Binding bulk
buffers as kernel arguments is capped somewhere in the low 20s with no clean
error at the boundary. FLM's approach — 5 declared kernel args plus a `DDR_PATCH`
table for the other 45 — is not merely a different style; it is the only route to
50 buffers on this firmware, and it also stays far inside the args-size bound
that the kernel-argument path silently overruns.

---

## 2026-07-31 — Phase 2 milestone 2: the 20-BO wall is gone, 38.6 GB/s verified

The previous entry concluded that binding bulk buffers as kernel arguments caps
at 20 and that FLM's patch-table structure was "the only route to 50 buffers on
this firmware". **That conclusion was too strong.** There is a second route, it
was already in IRON, and it needs no patch table.

### IRON has two dispatch paths, and only one has the ceiling

`python/utils/hostruntime/xrtruntime/hostruntime.py`:

- **default**: `kernel_handle.kernel(3, insts_bo, insts_bytes, *buffers)` — every
  buffer an `xrt::kernel` vararg. This is the path that caps at 20 host BOs and
  hangs the DPU past it.
- **full ELF**: `run.set_arg(i, buf)` then `run.start()` (line 388) — selected by
  `iron.jit(..., full_elf=True)`, which is an accepted jit config key.

Switching path is a one-argument change, and it lifts the ceiling entirely.

### Measured, all with `--verify` (bytes checked byte-for-byte)

| buffers | kernel-arg path | full-ELF path | vs FLM 46.2 |
|---|---|---|---|
| 20 | 9.7 GB/s | **23.96 GB/s** | 0.52x |
| 24 | `ERT_CMD_STATE_ERROR` | 26.78 GB/s | 0.58x |
| 32 | — | **30.04 GB/s** | 0.65x |
| 48 | — | **36.62 GB/s** | 0.79x |
| 64 | — | **38.56 GB/s** | **0.83x** |

**64 host buffers on ONE dispatch, 38.56 GB/s, verified.** Against FLM's 46.2
GB/s decode and the 56.5 GB/s fabric roof, that is 83% of FLM and 68% of roof.

Two things beyond removing the wall: the full-ELF path is **2.5x faster at the
same 20 buffers** (23.96 vs 9.7), so the kernel-arg path was also costing
bandwidth well before it failed; and throughput still climbs with buffer count
at 64, so this is not yet the plateau.

### What this changes

Phase 2's central question — can one dispatch stream a whole model's weights at
something near FLM's rate — now has a positive answer at 83% without any GEMM
arithmetic, without the patch table, and without a custom host path. The
remaining gap to FLM is worth chasing but is no longer a structural blocker.

64 is where the sweep stopped because it is the `kMaxHostBOs` value this branch
raised the aiecc cap to; whether the full-ELF path extends past it is untested.
llama3.2:1b needs 50, so 64 already covers the target.

### Correcting the previous entry

"FLM's 5-args-plus-DDR_PATCH-table structure is not a stylistic choice but the
only route to 50 buffers on this firmware" — wrong on the second half. It is the
only route *through the kernel-argument path*. The full-ELF path reaches 64
without it. FLM's structure remains interesting for other reasons (it is how a
5-connection xclbin binds 50 buffers) but it is not a prerequisite for phase 2.

The error was reasoning from one mechanism's limit to a universal claim, having
looked at only the dispatch path IRON happened to default to. The alternative was
in the same file, ten lines away.

---

## 2026-07-31 — Phase 2 milestone 3: 52.3 GB/s verified, above FLM's 46.2

The delivery structure now exceeds FLM's decode bandwidth. Verified byte-for-byte.

| configuration | total | GB/s | vs FLM 46.2 | % of 56.5 roof |
|---|---|---|---|---|
| 32 bufs x 1024 KiB | 32 MiB | **47.89** | **1.04x** | 85% |
| 48 bufs x 1024 KiB | 48 MiB | **50.79** | **1.10x** | 90% |
| 32 bufs x 2048 KiB | 64 MiB | **52.28** | **1.13x** | **93%** |

### The gap was per-dispatch overhead, not delivery structure

Milestone 2 reported 38.56 GB/s at 64 buffers and treated the remaining 17% as
an open question about workers or tiling. Both of those turned out to be spent
knobs, and the real cause was visible in the numbers already recorded:

| buffers | MiB | npu us | GB/s | ideal us @46.2 | implied fixed cost |
|---|---|---|---|---|---|
| 20 | 5 | 218.9 | 23.95 | 113.5 | **105.4 us** |
| 24 | 6 | 235.0 | 26.77 | 136.2 | **98.8 us** |
| 32 | 8 | 278.9 | 30.08 | 181.6 | **97.3 us** |

A near-constant **~100 us fixed cost per dispatch**, independent of buffer count.
At 8 MiB that is a third of the wall clock; at 64 MiB it is under 8%. So the
earlier figures were not measuring the delivery path's rate at all — they were
measuring a fixed overhead diluted by a small transfer.

**FLM streams 37 MiB per layer.** Measuring with 256 KiB buffers was testing a
regime FLM never operates in. Sizing the transfer realistically, the same design
that read 33.79 GB/s reads **52.28 GB/s**.

### The exhausted knobs, recorded so they are not re-swept

- **workers**: 4 -> 35.65, 8 -> 39.02, 16 -> 39.88 GB/s at 64 bufs. Plateaus past
  8; not the limiter.
- **tile size**: >16 KiB fails to compile. 32 KiB double-buffered is the entire
  64 KiB core L1, so 16 KiB is the practical maximum.

### What this settles for phase 2

The plan's phase-2 bar is "within ~10% of FLM's throughput". **On delivery
structure alone that bar is met and passed** — 1.04x to 1.13x — with no GEMM
arithmetic in the design, on stock mlir-aie, using `full_elf=True` and BD reuse
via TaskGroups.

So a phase-2 reproduction does not inherit a bandwidth deficit. Whatever it ends
up costing will come from the compute and the dequant chain, not from the
dispatch path. That is a materially better starting position than the
64-dispatches-per-token, ~10 GB/s figure this work began from.

**Caveat on the comparison.** 52.28 GB/s is this probe streaming bulk buffers
with cores that acquire and release without reading. FLM's 46.2 GB/s is a real
decode doing full arithmetic. These are not like-for-like: the probe shows the
delivery path *can* exceed FLM's rate, not that a working kernel will. The honest
claim is that dispatch structure has been removed as the bottleneck.

---

## 2026-07-31 — A mis-scoped tick, reversed; and where the AIE2P intrinsics live

Set out to put the q4_1 dequant chain behind the streaming probe, to answer
"can a core consume at 52 GB/s while paying the dequant cost?". Scaffolded a
consuming kernel, then stopped and reverted it. Two reasons, both worth
recording.

### 1. The intrinsic names were guesses, and all of them were wrong

Wrote `unpack_v32int4`, `ups_to_v64acc32`, `add_v64acc32`, `undef_v64acc32` from
memory. Checking before wiring them in: **none exists** — and neither did
`mac_8x8_8x8`, which `macbench.py` compiles successfully every run. That last
one is the tell that the check was worth doing: a grep returning zero for a
symbol known to work means the *grep* is wrong, not the symbol.

`aie2pintrin.h` is a 2.6 KB umbrella header. The definitions are in
`lib/clang/21/include/aie2p/`:

| what | where | real names |
|---|---|---|
| MAC modes | `aie2p_vmult.h` | `mac_8x8_8x8`, `mac_4x16_16x16`, ... |
| widen | `aie2p_*.h` | `ups(...)`, `ups_to_v16acc32`, `ups_to_v16acc64`, `ups_to_v16accfloat` |
| unpack | | `unpack(...)` — generic, not per-type |
| undef | | `undef_v16acc32`, `undef_v32acc32`, ... (per width) |

Recorded because the next attempt at a hand-written AIE2P kernel will need this,
and because "grep the umbrella header" silently returns nothing useful.

### 2. The experiment was largely redundant, which I should have seen first

The question "can a core keep up with the delivery rate while running the
dequant chain?" is **already answered empirically — by FLM**. Its 16 GEMM cores
run exactly this chain and sustain 46.2 GB/s, i.e. ~1.6 weight-bytes/cycle/core.
And `macbench_hw.py` separately measured the streamed MAC modes (int8 157,
int4 250 MACs/cycle). Building a second, weaker proof of a thing already
demonstrated by the reference implementation is not where the risk is.

**The actually-open phase-2 question is different**: not *whether* compute can
keep up, but whether **our** kernel reaches FLM's per-core efficiency — FLM gets
76% of its own static ceiling and is latency-bound on the dequant dependency
chain. Answering that means writing the GEMM, which is a phase-2 workstream, not
a tick.

### State

Phase 2 milestones 1-3 stand: dispatch structure is no longer the bottleneck
(52.3 GB/s verified, 1.13x FLM). The next unit of work is a real GEMM kernel
behind that delivery path, and it should be started deliberately rather than
scaffolded in passing.

---

## 2026-07-31 — Phase 3a's premise does not survive the bandwidth arithmetic

Started building a GEMM kernel and instead found that the question phase 3a is
framed around has already been answered by two measurements taken weeks apart in
this log. New tool: `tools/npu/flm/gemm_shapes.py`.

### The two facts, put side by side

| measurement | value | source |
|---|---|---|
| `mac_4x16_16x16` issue rate | **512 MACs/cycle/core** | `macbench_hw.py`, hardware |
| FLM decode bandwidth | **46.2 GB/s** of 5.00 bpw weights | `flm_bench.py`, hardware |

The second implies a **weight supply** of

```
46.2e9 / (5.00/8) / 16 cores / 1.8 GHz = 2.57 MACs/cycle/core
```

**The MAC unit offers 199x more than the weight supply can feed.** Even
`mac_elem_16` — the 16-lane mode FLM actually uses, and which this plan treats
as FLM's handicap — is **6x** more than the supply allows.

### What that means for phase 3a

The plan argues 3a as a MAC-width story: "symmetric int4 straight into
`mac_4x16_16x16`: no unpack, no zero-point". That framing does not survive.
Decode is overwhelmingly **bandwidth**-bound, so the MAC mode is nearly
irrelevant to decode throughput. Three consequences:

1. **The real oq4++ win is bytes, and it is ~21%.** 4.125 bpw against q4_1's
   5.00 is `5.00/4.125 = 1.212` — 21% more weights per second at the same
   bandwidth. Worth having, but it is a fifth, not a multiple.
2. **Deleting the 42-op dequant chain buys little directly.** If the core is
   waiting on weights, removing compute it was doing while it waited does not
   speed anything up. There may be a second-order effect — FLM sits at 76% of
   its own static ceiling and the plan attributes that to the dequant dependency
   chain — but that is a 24% ceiling effect at most, not the headline.
3. **FLM's choice of `mac_elem_16` was not a mistake to exploit.** This plan
   reads it as FLM missing the wide modes ("what costs it the wide modes is tile
   shape plus the q4_1 format together"). At 2.57 MACs/cycle of supply, a 16-lane
   MAC is already 6x oversized. It is an entirely reasonable choice, and the
   32x/64x headroom this plan has been treating as available upside is not
   reachable through decode.

### Where the upside actually is

Ranked by what the measurements support:

- **Fewer bytes per weight** — 21% from oq4++, and it compounds with anything
  else that reduces traffic.
- **Prefill**, which is compute-bound (FLM: 1774 tok/s on llama, 185 on the MoE)
  and where MAC width *does* matter. The plan's 3a is written for decode.
- **The MoE**, which runs at 67% of fabric against llama's 86% — a third of its
  achievable rate is going somewhere other than weight streaming, and that gap
  is larger than oq4++'s 21%.

### Also recorded: what the tool does and does not do

`gemm_shapes.py` compiles the native int4 loop and reports its schedule (4
bundles, 2 `vmac`, confirming 2 cycles/call — an independent reproduction of the
hardware measurement). It deliberately does **not** rebuild FLM's dequant chain
from source: that cost is already known from the shipped binary, and a
reconstruction would compare our codegen against theirs rather than the two
formats.

An earlier draft did try to rebuild it and got stuck on int8->bf16 conversion
intrinsics. Abandoning that was the right call twice over — the arithmetic above
makes the comparison unnecessary.

---

## 2026-07-31 — Phase 4 added to the plan: bandwidth reduction

Added a fourth phase to `~/flm-re-fe-mutate-goal.md` on the user's request:
QTIP-3 trellis quant and other bandwidth-reduction levers.

**Why it belongs, in one line:** the phase 0-2 measurements say decode is
bandwidth-bound with the MAC unit **199x oversupplied**, so bytes per weight is
the dominant lever and the plan had no phase aimed at it.

### The argument that makes QTIP interesting *here* specifically

On a GPU, trellis formats are held back by decode cost — sequential work per
weight against contended ALUs. **On this hardware the ALUs are 199x idle while
the core waits on DDR.** An expensive-to-decode, very compact format is close to
a free trade here. That is a hypothesis from the measurements, not a result, and
the phase is written to kill it quickly if wrong.

### What the repo already knows, and it is adverse on quality

`docs/roughquant-spec.md` sim (Qwen3.5-0.8B): bf16 26.17 PPL, mq4 29.08,
**QTIP-3-LDLQ 31.42**. Iso-bit QTIP-3 was *worse* than mq4, and roughquant's best
rotated config beat it by 11%. A `qtip3-sim(calib)` eval cell exists and passes.

Recorded prominently in the phase because the honest framing is "the format is
not obviously good; what changed is the exchange rate". 31.42 vs 29.08 PPL at
3.00 vs 4.00+ bpw is a different trade when bytes convert to tok/s at 1.67x.

### Ceilings, from bandwidth alone

Against the measured 59.86 tok/s baseline: oq4++ 1.21x (72.6), mq3/oq3 1.54x
(92.1), QTIP-3 1.67x (99.8), QTIP-2 2.50x (149.7). Ceilings only — they assume
the format decodes at rate and carry no quality cost.

### The lever ranked above QTIP

**Multi-token verification (speculative decode / MTP)** is almost certainly worth
more than every format change combined, and it is not a quant format. Decode
streams the whole weight set *per token*; verifying N tokens per sweep divides
weight traffic per token by ~N. The repo has DFlash and MTP, and
**Qwen3.6-35B-A3B ships `mtp_num_hidden_layers: 1`** — the head is already in the
target model. It also composes with any format change.

Then two-stage lm_head, which phase 1 measured at **20-21% of the per-token
stream on both models** — under-rated in the plan relative to 3a.

### The gate this phase must pass

Item 2 of the QTIP de-risk order is the real gate: **measure the trellis
decoder's bundles/weight on AIE2P.** 199x headroom is generous but not
unbounded, and a sequential trellis may serialise badly on a VLIW machine with
one MAC slot. If the decoder cannot sustain ~3.11 MACs/cycle/core the format is
dead here whatever its quality.

### And a note on how phase 3 was written

Phase 3's premise was a MAC-width advantage the hardware does not have. The
phase-4 text says so directly and tells the reader to run 3a for its bytes rather
than its MAC. Recorded because the failure mode is instructive: the plan was
written before the machine was measured, and every number in it that came from
the vendor's capability sheet rather than a measurement pointed the wrong way.

---

## 2026-07-31 — Milestone 3 extended: 56.3 GB/s at decode-realistic totals

Milestone 3 established that the ~100 us fixed per-dispatch cost was what the
earlier small-buffer figures were really measuring, and reached **52.28 GB/s** at
64 MiB. Pushing the total to the size a real decode token moves confirms that
model and finds where it tops out. Both runs `--verify`, so the bytes are
checked, and both `--full-elf`.

| shape | total | GB/s | vs FLM 46.2 | % of 56.5 roof |
|---|---|---|---|---|
| **50 bufs x 16 MiB** — FLM's exact argument count | **800 MiB** | **53.79** | **1.16x** | 95% |
| 16 bufs x 32 MiB | 512 MiB | **56.34** | **1.22x** | **99.7%** |

**There is nothing left in this direction.** At 512 MiB the wall clock is
9.53 ms, so milestone 3's ~100 us fixed cost is ~1% of it and the delivery path
is running at the fabric roof measured independently in
`npu-memory-bandwidth-cache-characterization.md`. The 800 MiB row is 4.5% lower
because 50 buffers of 16 MiB pay the per-buffer cost more often than 16 of 32 MiB
— the same amortisation milestone 3 identified, seen from the other side.

**FLM's exact dispatch shape reproduces.** 50 bound buffers, one command, a
whole token's weight traffic (llama streams 772.3 MB/token), at 1.16x its
measured rate. The "pack fewer, larger buffers" workaround recorded before
`full_elf` was found is unnecessary — though 16 x 32 MiB does measure better and
is the simpler design.

### A matched control for the vararg/full-ELF difference

Milestone 2 established that `full_elf` clears the 20-BO wall. Same design, same
buffers, same everything, sweeping only the count, to separate "clears the wall"
from "is otherwise equivalent":

| buffers | vararg | `full_elf` |
|---|---|---|
| 20 | 13.99 GB/s | 18.78 GB/s |
| 24 | `ERT_CMD_STATE_ERROR` | 19.82 GB/s |
| 32 | `ERROR` | 21.14 GB/s |
| 50 | `ERROR` | 22.70 GB/s |

**It is not only a ceiling difference — it is ~34% faster at 20 buffers, where
both paths work.** So the vararg path was costing bandwidth on every design that
stayed inside its limit, not just failing outside it. `--full-elf` is now a flag
on the probe rather than an edit.

(`tools/npu/flm/README.md` still carried the pre-`full_elf` reading of the wall —
"hangs the firmware ... pack fewer, larger buffers" — and is corrected in this
commit. A trap list is only useful if it is corrected as fast as it is written.)

## 2026-07-31 — Milestone 4 prep: the q4nx weight container decodes, and FLM's weights are TRANSFORMED

Milestone 4 needs real weights in the kernel, so the first job was reading
FLM's own weight container: `~/.config/flm/models/Llama-3.2-1B-NPU2/model.q4nx`,
1297.8 MB, a plain safetensors file (8-byte header length + JSON header).

### The container, and the 5.00 bpw figure confirmed from the bytes

148 tensors. Every streamed weight is `I8` with a second dimension of **5120**:

| tensor | stored shape | true shape | K |
|---|---|---|---|
| `lm_head.weight` | [32064, 5120] | [128256, 2048] | 2048 |
| `*.self_attn.q_proj` | [512, 5120] | [2048, 2048] | 2048 |
| `*.self_attn.k_proj` | [128, 5120] | [512, 2048] | 2048 |
| `*.mlp.gate_proj` / `up_proj` | [2048, 5120] | [8192, 2048] | 2048 |
| `*.mlp.down_proj` | [2048, 5120] | [2048, 8192] | 8192 |

`model.embed_tokens.weight` is BF16 [128256, 2048] and unquantized; `model.norm`,
the layernorms and `rope_freqs` are BF16. Per-layer bytes sum to **38,010,880**,
reproducing the figure section 3 of `flm-layer-dataflow.md` derived from the
manifest, and `lm_head` is 164,167,680 B against its 164.2 MB.

**A 5120-byte row is planar, not a block array**, which is the thing that cost
the most time here — the first hypothesis was llama.cpp's interleaved 20-byte
`block_q4_1`, and every variant of it produced `inf`/`NaN` because the scale
field was landing on packed nibbles. Scanning the row as bf16 instead shows
three regions with sharply different signatures:

```
[   0: 512]  256 x bf16, ALL POSITIVE, ~0.003-0.03   -> d   (scales)
[ 512:1024]  256 x bf16, ALL NEGATIVE, ~-0.01--0.28  -> m   (mins)
[1024:5120]  4096 B packed 4-bit                     -> 8192 codes
```

512 + 512 + 4096 = 5120 B for 8192 weights = **exactly 5.00 bits/weight**, which
confirms the bpw figure from the byte layout itself rather than from dividing the
manifest by a parameter count. 8192 weights per row is 4 output rows at K=2048,
or one output row at K=8192 — and both tensor families land on 5120 exactly,
which is why the same second dimension serves both.

The `m/d` ratio is **-7.4 to -7.5 with std 1.35** across every tensor, and
`m + 7.5d` centres on zero to 4 decimal places. That is the signature of plain
min/max asymmetric q4_1 (`d=(max-min)/15`, `m=min`) on symmetric zero-mean
blocks, and it corroborates `Dequant::generate_dequant_q4_1_seq` from the symbol
table.

### Blocks run along K, not N — settled by a statistic, against expectation

`vextbcst.16` + `mac_elem_16` suggests broadcasting one activation scalar against
contiguous *output* weights, which would make the layout N-major. It is not.
Per-block scale spread separates the two cleanly, because outlier channels make
the N-major spread much heavier:

| tensor | stored `d` p99/p1 | HF K-major | HF N-major |
|---|---|---|---|
| `q_proj` | **7.18** | **7.01** | 31.77 |
| `down_proj` | **2.49** | **2.42** | 2.51 |

`std/mean` agrees to ~2% on the same comparison (0.4629 vs 0.4580; 0.2044 vs
0.1994). So: **32 contiguous input dims per block, K-major.**

### FLM's weights are NOT a quantization of the published checkpoint

This is the finding, and it is a negative one. Four independent searches for the
underlying float weights all came back at the noise floor:

1. **`lm_head` vs `embed_tokens`.** `config.json` has
   `tie_word_embeddings: true`, so the container carries its own ground truth —
   the same matrix stored both quantized and unquantized. Searching every
   aligned 64-window of all 8.2 M stored scales for one embed row's block-scale
   fingerprint: best `r=0.56`, against a control that self-matches at
   **0.99996** and whose best *wrong* row scores 0.55. No match.
2. **Row sums** (invariant to element order within a row): matches no better
   than the 1.9e-5 nearest-neighbour gap of the sum distribution. So it is not
   an element permutation.
3. **N-major and strided variants** of the same fingerprint, on `lm_head` and on
   layer tensors against the HF `consolidated.00.pth`: all `r≈0.5`, control 1.00.
4. **Sorted 256-scale multiset per row** — high entropy, order-independent, so
   it matches *exactly* if the blocks are contiguous-along-K and merely stored
   in a permuted order. Control: self `0.000e+00`, best other `3.79e-05`. Every
   candidate scored 3.3e-05 to 5.8e-04, i.e. all at the between-rows distance.
   Tested against **both** the Instruct and the base checkpoint, in case FLM
   quantized the other one. Neither matches.

The aggregate statistics match untransformed K-major weights to ~2% while every
per-block fingerprint fails. That pair of facts is consistent with a **mild
per-channel transform applied before quantization** — SmoothQuant/AWQ-style
scaling folded into the activation — which changes each block's range while
leaving the population of ranges nearly unchanged. A Hadamard control shows
rotations also leave these statistics nearly unchanged (`std/mean` 0.4297 vs
0.4580), so the statistics cannot discriminate *which* transform, only that
there is one.

**Consequence for milestone 4, and it is a real one.** The dequantized weight
matrix cannot be reconstructed from the published checkpoint, so "numerically
equivalent to FLM on real weights" has to be established by running our kernel
on **FLM's own stored codes**, not on our own re-quantization of the HF weights.
That needs one more fact than we have: which `(output row, k-block)` each of the
256 stored slots maps to. The arithmetic does not depend on it — the block
layout within a row is fully decoded — but the output indexing does.

Recorded per the habit that keeps paying: every correction so far came from
changing the KIND of evidence. Here the byte-region scan (a different kind) gave
the format immediately after the block-structure hypothesis had failed four
ways, and the scale-spread statistic settled K-major against what the ISA
opcodes suggested.

## 2026-07-31 — Open thread 1 CLOSED: FLM's 46.2 GB/s is consistent with 4-stream ingress

The kickoff flagged this as narrow and worth closing early, because it changes
what phase 2's "within ~10% of baseline" bar means. The earlier evidence was a
4-stream point of 46.5 GB/s measured at a small total — suggestive, but the
probe's streams are not FLM's and the shape was not the one FLM runs.

Re-measured at **decode-realistic totals**, `--full-elf --verify`, both runs with
the user's `flm serve llama3.2:1b` resident (contended condition):

| feed streams | 16 x 32 MiB = 512 MiB | 48 x 16 MiB = 768 MiB |
|---|---|---|
| 2 | 27.1 | 27.1 |
| **4** | **48.6** | **48.6** |
| 6 | — | 43.4 |
| 8 | 56.2 | 55.6 |
| 12 | — | 56.0 |
| 16 | compile failure | — |

**The 4-stream figure reproduces to three digits across two different totals**
(48.6 and 48.6), which is the cross-check that makes it quotable. 768 MiB was
chosen because it is FLM's actual per-token traffic (772.3 MB); 48 buffers was
chosen over FLM's 50 because it divides by 4, 6, 8 and 12.

So:

- **4 streams deliver 48.6 GB/s, 5.2% above FLM's 46.2**, and FLM feeds its 16
  GEMM cores from 4 memtile columns. A 5% gap is about what real addressing and
  concurrent compute would cost, so the hypothesis survives the sharper test.
- **Widening to 8 streams gives 55.6-56.2 GB/s, +15% over 4** — and 8 is where
  it saturates (12 adds 0.4 GB/s). This reproduces milestone 3's 56.34 GB/s at
  the same 16 x 32 MiB shape to **0.25%**, which incidentally bounds the
  `flm serve` contention on these numbers as negligible.
- **A reproduction that widens weight ingress should beat FLM by ~20% before any
  kernel-body change** — 56.2/46.2 = 1.22x. That is now measured at the right
  shape rather than inferred from a small-buffer point.

Still not *proof* that FLM is ingress-limited: its 46.2 includes compute and its
streams are not these. But the prediction the hypothesis makes — that 4 streams
land just above 46.2 and 8 streams land ~20% higher — is what the hardware does,
at two totals, with the bytes verified.

**One anomaly recorded rather than smoothed over:** 6 workers measures 43.4 GB/s,
*below* the 4-worker figure. 6 does not divide the 8 columns evenly, so two
columns carry two streams while four carry one. A variable nobody sweeps is
indistinguishable from one that does not matter — but this one does, and it says
stream count is not the whole story: **stream-to-column balance is**.

## 2026-07-31 — Milestone 4: the q4_1 decode GEMV body runs and is exact

`kernels/npu/flm_gemv_q4_1.cc`, verified by `tools/npu/flm/gemv_verify.py` on
**real q4_1 data from FLM's own `model.q4nx`** — real scales, real mins, real
codes.

| shape | vs bf16-faithful reference | vs exact float64 |
|---|---|---|
| K=2048, 8 rows, 1 tile | **1.9e-07** | 8.2e-03 (0.99% of \|out\|) |
| K=2048, 512 rows, 64 tiles | **3.3e-07** | 1.2e-02 (1.69% of \|out\|) |

The device reproduces a bf16-faithful emulation of the body to float32
round-off. The deviation from float64 is **entirely the format's own cost** —
the emulation reproduces it exactly — and it is the same cost FLM pays, since
FLM materialises `w = d*q + m` in bf16 before its own MAC.

**The float64 reference was the wrong gate and briefly looked like a failure.**
A body that is exactly right still lands ~1% from float64 on an output with
this much cancellation (the summed magnitudes are ~60x the result). Both
references are now reported: `gemv_reference_bf16` is the correctness gate,
`gemv_reference` is context.

### The body

The dequant folds out of the inner loop entirely. With `w = d*q + m`,

    out[n] = sum_b ( d[n,b] * sum_t q[n,b,t]*a[b,t] + m[n,b] * sum_t a[b,t] )

so the zero-point term becomes one scalar per block against an activation
block-sum **shared by every output row in the tile**, and the codes enter the
MAC as exact small integers. FLM spends a 42-op dequant chain materialising bf16
weights instead; that chain is not reproduced and does not need to be — weight
supply is 2.57 MACs/cycle/core against the MAC unit's 512.

### Five traps, all of which cost real time

Recorded in `tools/npu/flm/README.md` as well.

1. **`iron.jit` does not hash `ExternalFunction(source_file=...)`.** Editing the
   kernel `.cc` silently reuses the cached xclbin, so the run reports the *old*
   kernel's numbers. This is the worst one: it presents as a fix that did
   nothing, and it made two identical expressions in one probe disagree (one
   from the stale binary, one from the fresh source). Pass the kernel path as
   the design-level `source_files=[...]`, which **is** in the cache key.
2. **The same applies to closure-captured shapes.** Shapes closed over from an
   enclosing scope are not in the cache key either, so the second shape reuses
   the first shape's binary — caught as
   `Tensor argument 'w' has 655360 elements but the kernel was compiled for
   10240`. Make them `CompileTime[int]` parameters.
3. **The default rounding mode TRUNCATES.** Without
   `aie::set_rounding(aie::rounding_mode::conv_even)` every accum→bf16
   conversion carries a one-sided bias, and with ~2000 summed terms and heavy
   cancellation the bias survives the reduction. **Measured: 13% error on a real
   row**, against 0.19% once set.
4. **Scalar reductions mixing bf16 loads with a float accumulator miscompile.**
   `msum += float(mrow[b])*asum[b] + float(mrow[b+1])*asum[b+1]` in a stride-2
   loop **dropped every b+1 term** — the device result matched "even blocks
   only" to six digits — while `mrow[b]`, `asum[b]`, `sum_b mrow[b]` and
   `sum_b asum[b]` all read back correct individually. A plain unit-stride
   scalar loop over the same values produced a third, different wrong answer.
   The vector-MAC form is correct. This one was only findable by scoring the
   device's number against a list of candidate mis-pairings.
5. **IRON's default worker stack is 1024 bytes**, and overflowing it fails
   silently — NaN in the first output rows, plausible-magnitude garbage in the
   rest. Also: an array that gets vector-loaded needs `alignas(64)`; an
   unaligned 512-bit stack load returns garbage rather than faulting.

Plus two backend limits that shaped the code rather than breaking it:
`aie::downshift` on a **uint8** vector segfaults the AIE2P backend (fine on
uint16), and a 16-lane uint8 vector fails to legalize (`G_AND <4 x s32>`) as
does a 16-lane float reduction (`G_FADD <16 x s32>`). The high nibble is
therefore extracted by mask only, left as `16*q`, with the factor of 16 divided
out of that half's scale — exact in bf16, and it removes a shift from the inner
loop.

**Scope of the check.** The arithmetic is verified exactly. What is *not*
verified is which output row and k-block each of FLM's stored slots belongs to,
because that mapping is unresolved (FLM's weights are not a quantization of the
published checkpoint). The bytes are real; their addressing is ours.

## 2026-07-31 — Milestone 4 throughput: 14.4 GB/s, compute-bound, 1.6x short per core

`tools/npu/flm/gemv_bench.py` runs the verified body under `layer.xclbin`'s
decode dataflow — one activation broadcast to every GEMM core, a private weight
stream per core, one dispatch — at 2.375 MB of q4_1 weights per core (16 cores
would be 38.0 MB, one llama decoder layer). Every point is correctness-checked
against the bf16 reference **on the bytes it actually streamed**, because a
bandwidth number from a kernel that computed nothing would look excellent.

| cores | MB | GB/s | wall us | vs FLM | max err |
|---|---|---|---|---|---|
| 1 | 2.4 | 1.8 | 1330.9 | 0.04x | 6.0e-07 |
| 2 | 4.8 | 3.6 | 1310.3 | 0.08x | 6.0e-07 |
| 4 | 9.5 | 7.2 | 1311.9 | 0.16x | 6.0e-07 |
| 8 | 19.0 | **14.4** | 1315.7 | 0.31x | 6.0e-07 |

**Wall time is constant across an 8x change in cores and bytes while GB/s
doubles exactly at every step.** Cores scale perfectly; delivery is idle;
**per-core compute is the entire constraint.** Per core that is 1.81 GB/s =
**1.61 weights/cycle** at 1.8 GHz, against the **2.57 weights/cycle** FLM needs
per core to reach 46.2 GB/s from 16 cores — **1.6x short**.

The bar is not met, and per the kickoff the gap is the finding, so it was
chased rather than reported flat:

- **The gap is in the kernel body, not the dataflow.** Milestones 1-3 showed one
  command delivers 56 GB/s; the flat wall clock here shows that delivery is not
  what the cores are waiting on.
- **Leading hypothesis: the native int4 unpack path.** FLM's GEMM cores carry
  `vunpack:64` and `vups.4x:64`, the AIE2P instructions that widen a **packed
  int4 vector** directly. This body never reaches them — it hands the hardware
  `uint8` lanes and spends a generic `bit_and` -> `unpack` -> `to_float` chain
  (3 vector ops per 32 codes, twice per 32-byte group) doing what those
  instructions do natively. That is the largest block of work in the inner loop.
  Testing it means feeding the MAC a `v256uint4` operand; `macbench_hw.py`
  already measures those modes and phase 3a needs the same operand type.
- **Repacking the nibbles bought +14%** (12.6 -> 14.4 GB/s) and is kept. The
  llama.cpp split form (byte j -> elements j and j+16 of one block) forces each
  code vector to meet two 16-lane activation loads joined by `aie::concat`;
  putting block b in the low nibbles and block b+1 in the high nibbles of the
  same 32 bytes removes both concats and two loads.
- **Two accumulators to break the MAC dependency chain does not compile** — a
  second 32-lane float accumulator makes the backend emit a 16-lane float add it
  cannot legalize (`G_FADD <16 x s32>`), the same limit that forced the
  zero-point term to share the dot accumulator. Reverted, marked `ponytail:` in
  the kernel so it is not re-attempted blindly.

### 16 cores does not fit the shim — a prediction confirmed by hitting it

```
no ShimNOCTile has sufficient DMA capacity for 0 input/1 output channels
```

16 private weight streams + 1 activation = 17 shim inputs, plus 16 outputs,
against 8 columns x 2 channels each way. `flm-layer-dataflow.md` said a naive
all-circuit reproduction would not fit the channels, and it does not. FLM's own
answer is in the same document: packet-switch at the shim (24 of its 42 shim
routes are packet mode) and concatenate outputs per GEMM pair into 8 result
streams rather than 16. Not built here — recorded as the next structural step.

**Stop condition honoured.** The kickoff says a reproduction materially slower
than baseline means finding the missing understanding before building on it. The
missing understanding now has a name (the native int4 operand path) and a cheap
test, and phase 3 is not started on this baseline.

## 2026-07-31 — Milestone 4 bar MET: 48.1 GB/s at 16 cores, 1.04x FLM decode

The previous entry recorded 14.4 GB/s (0.31x) and named the missing
understanding as a hypothesis. It was then tested, and it was right.

`gemv_bench.py`, K=2048, 16 rows/tile, 116 tiles/core — 16 cores carrying
**38.0 MB, exactly one llama decoder layer's weights** — in ONE dispatch,
correctness-checked on the bytes actually streamed:

| cores | MB | GB/s | wall us | vs FLM 46.2 | max err |
|---|---|---|---|---|---|
| 8 | 19.0 | 24.4 | 778.9 | 0.53x | 6.0e-07 |
| **16** | **38.0** | **48.1** | **790.8** | **1.04x** | **6.0e-07** |

A repeat run measured 48.6 / 1.05x. **Within ~10% of baseline, above it — the
milestone-4 throughput bar is met**, at 99-100% of the 48.6 GB/s that 4 feed
streams deliver with no compute.

| change | GB/s | |
|---|---|---|
| mask chain, 8 rows/tile, 8 cores | 14.4 | starting point |
| **native uint4 operand** | 18.5 | +28% |
| **16 rows per tile** | 24.8 | +34% |
| **16 cores, paired** | **48.1** | +94% |

### The native int4 operand path — hypothesis confirmed, and it predicted itself

FLM's GEMM cores carry `vunpack:64` and `vups.4x:64`. The first body never
reached them: it handed the hardware `uint8` lanes and spent a generic
`bit_and` -> `unpack` -> `to_float` chain doing what those instructions do
natively. Loading the codes as `aie::vector<uint4, 64>` instead compiles to
**75 instructions against 103** for the identical loop — `vband` disappears and
the widening rides the load as `vldb.unpack` — and measured **+28% on
hardware** against a −27% instruction-count prediction. The prediction and the
measurement agreeing to a percentage point is the check that this is the
mechanism and not a coincidence. Nibbles are now stored in plain element order,
which is what a native uint4 load expects and which also lets each code vector
meet a contiguous activation load.

### 16 rows per tile — bounded by L1

20480 B per tile, double-buffered 40960, plus the activation = 49 KB of the
64 KB tile memory. 24 rows would need 68 KB. So 16 is the widest that fits, and
it is worth +34% over 8.

### Pairing the cores — the shim budget, and FLM's own answer

16 private weight streams + 1 activation is 17 shim inputs against 8 columns x
2 channels; the placer rejects it with `no ShimNOCTile has sufficient DMA
capacity`. One shim stream per **pair**, split in a memtile, with the pair's two
result streams joined back into one before the shim, halves both counts and
fits. That is `layer.xclbin`'s structure — 16 weight streams out of 4 memtiles,
and a GEMM pair whose two N-groups concatenate — and `flm-layer-dataflow.md`
predicted the constraint ("a naive all-circuit reproduction will not fit the
channels") before it was hit.

**Trap**: `split()`/`join()` offsets are in **elements, not bytes**. A byte
offset overshoots the fifo and emits a BD with a negative length —
`XAie_DmaSetAddrLen(): Invalid Address ... static_len = -16`. It also produced a
misleading `Bank-aware allocation failed` warning at larger tile sizes, which
sent the first diagnosis at L1 capacity rather than at the offset.

### Correction to an earlier framing

The plan's "compute will not bind, and the margin is not close" (199x
oversupply) is about **MAC issue capacity**, and it does not imply the body is
free. At 16 cores this reproduction was compute-bound before it was
bandwidth-bound, and the rate was set by the unpack and rescale work *around*
each MAC. The 199x figure stays true and stays the reason not to chase wider MAC
modes; it is not a reason to ignore the ops per weight.

**Stop condition cleared.** The reproduction is no longer materially slower than
baseline, so phase 3 has a baseline worth measuring against.

## 2026-07-31 — Correction: the shuffle network makes nibble order nearly free

Prompted by a note that the AIE2P shuffle/"twizzle" operations exist and had not
been considered. They do, and one claim in the previous two entries is wrong.

AIE2P exposes a full transpose/shuffle mode set (`aie2p_enums.h`: `T8_64x2`,
`T16_32x2`, `T8_2x64`, `T16_4x16`, ... plus `T*_lo/hi` halves), surfaced in
aie_api as `interleave_zip` / `interleave_unzip` (with a chunk size),
`shuffle_up` / `shuffle_down` and their rotate/fill variants, and the raw
`shuffle_u8` / `shuffle_bfloat16` intrinsics taking a mode number.

**What was wrong.** The milestone-4 entries said llama.cpp's split nibble form
(byte j -> elements j and j+16 of one block) "forces each code vector to meet
two 16-lane activation loads joined by `aie::concat`: 25 vector ops per 64
weights against 18", and used that to justify repacking the file. The premise
holds only for a concat-based gather. A `uint4` load of split order yields lanes
`[e0,e16,e1,e17,...]`, and `aie::interleave_unzip(lo, hi, 1)` separates them in
**one** operation:

| variant | instructions, same loop |
|---|---|
| current (plain element order, no shuffle) | **75** |
| split order consumed via `interleave_unzip` | **76** |
| masked uint8 + `aie::concat` gather | 103 |

So arbitrary nibble order costs **one instruction**, not a 39% op penalty. The
repack was still worth doing — it is the simplest thing that works, and the
+14% it measured was real against the *concat* version it replaced — but it was
never forced by the format, and presenting it that way was wrong.

**Why this matters beyond the correction.** FLM's own nibble order within a
block is unknown and cannot be checked while the block-to-(row, k) mapping is
open. The shuffle network means that when the mapping is settled, matching FLM's
layout exactly costs ~1 instruction rather than a redesign — so the eventual
end-to-end comparison on FLM's own stored codes is not gated on the layout being
convenient.

**Also measured and rejected**: a 64-lane form (one MAC per 64 weights, the two
block scales joined with `aie::concat` instead of two of everything) compiles
but is **78 instructions against 75**. And the `unpack` intrinsic family is
8->16 bit only; 4-bit widening is done by the load itself, which the current
body already reaches (`vldb.unpack` + `vups.4x` in the disassembly).

Third time on this project that changing the KIND of check moved something: the
instruction count settled in one compile what an argument about lane alignment
had got wrong.

## 2026-07-31 — Four more levers from UG1079, all measured, none kept

Read `docs/npu/ug1079-2026.1-AIE-programming-manual/` (it is vendored in this
repo) for anything that could push past 48 GB/s. Four candidates, none survived.

**`aie::reduce_add_v`** (§030) reduces up to 4 vectors in parallel, which would
allow deferring the per-block scale: accumulate blocks unscaled, reduce four at
once, apply four scales as one vector multiply. **185 instructions against 75**
for the same 2048 weights. A 32-lane float horizontal reduce costs ~20
instructions on its own (86 for four), so per-block reduction is far dearer than
the per-block rescale it replaces. This is a positive result about the current
structure: accumulate-once / reduce-once is right, and the reason is measurable.

**`aie::sliding_mul`** (§037) is the natural GEMV shape — `out[l]` over L lanes
with no horizontal reduction at all. It does not apply: UG1079 defines
`DataStepY` as a step *within the data register*, so all L lanes' weights must
sit in one loaded vector, i.e. **N-major**. FLM's format is K-major with
per-K-block scales, so using it would mean changing the quantization block axis
and no longer reproducing the format. Structural, not a tuning question.

**Loop unrolling.** `#pragma clang loop unroll_count(2)` measured 48.6 GB/s,
inside the 47.5-48.6 band of no unrolling — no gain. `unroll_count(4)` measured
**42.6 GB/s, -12%**; the unrolled body spills. The knob was removed rather than
left in at a value that does nothing.

**`chess_prepare_for_pipelining` / `chess_loop_range` are no-ops under Peano** —
both `#define`d empty in `aiebase_chess.h`. They appear on the hot loop in AMD's
own `aie_kernels/aie2p/mm.cc`, so copying that pattern into a Peano build looks
like tuning and is not. Worth knowing before attributing a result to them.

**Load-port check.** §043 flags two loads per cycle as a resource constraint. The
inner loop issues 3 vector loads per 64 weights => >=1.5 cycles => a 42.7
weights/cycle ceiling, against 2.75 measured with ~13 ops per 64 weights. So the
loop is op- and latency-bound, not load-bound, and cutting loads is not the
lever. That also rules out the block-outer/row-inner inversion (which would
amortise the activation load across rows) as the next thing to try.

Net: 48.1-48.6 GB/s stands, and the remaining gap to the 56.5 GB/s fabric roof is
ingress width rather than the body — consistent with the body already sitting at
98-100% of what 4 feed streams deliver with no compute at all.

## 2026-07-31 — Tile memory banking: measured, real in the address map, not dominant

Prompted by a note that the optimization work was reasoning about ops and not
about the NPU's hardware design. Fair, and it turned up something I had actively
dismissed: a `Bank-aware allocation failed, trying basic sequential allocation`
warning in an earlier build was written off as a red herring.

**The hardware fact** (UG1079 §004, and `AIETargetModel.h` for the AIE2P
numbers): a core tile has **64 KB in 4 banks of 16 KB** (memtiles: 512 KB, 8
banks), and **"concurrent operation of all three ports is supported if each port
accesses a different bank"**. Two loads plus a store per cycle only issue
together when they land in different banks. So buffer *placement* is a
first-class performance parameter, not just a capacity question — and the GEMV
body was measured at an effective 0.54 ops/cycle, which is exactly what a
serialised load port looks like.

**What the shipping design actually does.** Dumping `input_with_addresses.mlir`
for the 16-core / 16-row build:

```
wp0_split0_cons_buff_0   addr= 4096  size=20480   banks 0..1
wp0_split0_cons_buff_1   addr=24576  size=20480   banks 1..2
act_0_cons_buff_0        addr=45056  size= 2048   bank  2
act_0_cons_buff_1        addr=49152  size= 2048   bank  3
```

A 16-row weight tile is 20480 B = **1.25 banks**, so every weight buffer
straddles two, and on the double-buffer half that uses `wp..buff_1` the weight
read and `act..buff_0` collide on **bank 2**. Half the iterations cannot issue
both loads in one cycle.

**Tested, and the conclusion is not the obvious one.** 12 rows is the largest
tile that fits inside one bank (15360 B), which makes the layout conflict-free:

| rows/tile | wtile | banks | GB/s |
|---|---|---|---|
| 12 | 15360 | fits one | **44.8** |
| 16 | 20480 | straddles two | **48.1** |

**The bank-clean layout is 7% SLOWER.** More rows per call amortises the
activation block-sum pass and the per-call overhead by more than the conflict
costs, and at 20480 B a 16-row tile can never be made to fit a 16 KB bank. So
banking is real, visible, and currently the second-order effect — worth stating
either way, because "bank-aware allocation succeeded" in the build log does *not*
mean the hot buffers are in different banks, and nothing in the toolchain says so.

Recorded so the next person does not either dismiss banking (as I did) or assume
fixing it is a win (as the address map suggests).

**Two hardware levers this surfaced that remain untried**, both already used by
FLM and both bigger than banking:

- **Neighbour data memory.** A core can address its neighbours' data memory
  directly — no DMA, no cascade, "no instructions beyond a pointer into a
  different window" (`flm-layer-dataflow.md` implication 6, which records FLM
  doing exactly this for its paired cores). The activation is currently
  DMA-broadcast and double-buffered into *every* core's L1 at 8192 B. Holding it
  once per neighbourhood would free that, and the freed L1 raises the row count
  past the 16 the 64 KB budget currently caps it at (24 rows needs 61440 B of
  weights alone).
- **The cascade stream.** Direct accumulator forwarding between adjacent cores,
  which is the hardware path for splitting one dot product across cores without
  touching memory. FLM does *not* use it (its pair is an N-split that
  concatenates), so it is unexplored here in either direction.

## 2026-07-31 — Task 1 MET (+34%), and the "compute-bound" diagnosis was measured wrong

Two results, and the second corrects the previous entry and the phase-3 kickoff
written from it.

### The activation block-sums were being recomputed 116 times per core

`asum[b] = sum of 32 activations` depends **only on the activation**, which is
constant for a whole dispatch — but it sat inside the per-tile kernel call, so it
was recomputed for every one of the 116 weight tiles. Bundle accounting from the
disassembly put it at ~832 of ~7216 bundles per call = **12%**.

Hoisted into a second entry point (`flm_asum_prepare`, called once per activation
acquire, writing a file-scope array the GEMV reads). It has to be its **own
translation unit**: IRON compiles each `ExternalFunction`'s source separately, so
two entry points in one file link twice and fail with `duplicate symbol`.

| | 8 cores | per core | weights/cycle |
|---|---|---|---|
| before | 24.4 GB/s | 3.04 GB/s | 2.70 |
| after | **32.7 GB/s** | **4.09 GB/s** | **3.63** |

**+34%, and past the >=3.2 weights/cycle/core target.**

### The diagnosis it was built on was wrong, and a control experiment shows why

At 16 cores the hoist measured **zero** — 48.6 GB/s before and after. That looked
like "the loop is not issue-bound", and the previous entry concluded the design
was per-core compute-bound because 8 -> 16 cores doubled throughput at constant
wall clock.

**Both readings were wrong, and the error was a missing control.** In the paired
design `npairs = ncores/2`, so doubling the cores also doubles the **shim input
streams**. Core scaling and ingress scaling were confounded, and the whole
conclusion rested on not separating them.

The control is a no-op body — identical fifos, identical traffic, no arithmetic:

| | 4 cores (2 streams) | 8 cores (4) | 16 cores (8) |
|---|---|---|---|
| no-op, delivery ceiling | 20.6 | **40.9** | **49.3** |
| full GEMV (after hoist) | — | 32.7 | 48.6 |

So:

- **At 16 cores the design is DELIVERY-bound**, not compute-bound: the no-op
  ceiling is 49.3 and the GEMV reaches 48.6 — the body costs **1.4%**. The hoist
  could not show there because only 1.4% of headroom existed.
- **At 8 cores it is compute-bound** (32.7 against 40.9 available), which is where
  the hoist is visible and where per-core work should be measured from now on.
- Per-core compute and dataflow delivery cross within 1.5% at 16 cores, which is
  exactly the coincidence that made one look like the other.

**Consequence, and it restores what the plan said all along.** The previous entry
claimed the reproduction was compute-bound and therefore that phase 3a/4a's
bits-per-weight reductions "are worth approximately nothing". With the body now
at 4.09 GB/s/core of capability against a 49.3 GB/s dataflow ceiling across 16
cores (16 x 4.09 = 65.4 of capability), **the design is bandwidth-bound and
bpw reduction pays again**. The original plan was right; the correction was based
on a confounded measurement, and it is withdrawn.

**The next lever is the dataflow, and it has a number.** `dispatch_bw_probe.py`
delivers **56.2 GB/s** through 8 direct shim streams; this design's 8 streams go
through a memtile split and deliver **49.3** — so the split costs ~12%, and
recovering it is worth more than anything left in the body. Shim channel budget
also has room that is not being used: 8 columns x 2 = 16 input channels against
the 8 weight + 1 activation currently bound, and it was an *output* channel that
failed at 16 unpaired cores, not an input.

**Method note.** The tell was the hoist removing 12% of issued bundles and
changing nothing. That should have prompted the control immediately rather than a
conclusion about issue rate. A no-op body at identical geometry costs one run and
separates "what can this dataflow deliver" from "what does the body cost"; it
should be the first measurement on any new dataflow here, not the last.

## 2026-07-31 — Task 2: the FFN block runs on the NPU, verified end to end

`tools/npu/flm/ffn_verify.py`. First time a *block* of a real decoder layer has
run on the reproduction rather than a single operator. llama-3.2-1B layer 0 FFN,
real q4_1 weights from FLM's container:

    h -> gate_proj(h), up_proj(h)   K=2048, N=8192   flm_gemv_q4_1
      -> silu(gate) * up            8192             flm_swiglu
      -> down_proj(.)               K=8192, N=2048   flm_gemv_q4_1

| stage | max abs err |
|---|---|
| gate_proj | 2.98e-08 |
| up_proj | 2.52e-08 |
| swiglu | 3.03e-05 |
| down_proj | 1.00e-09 |
| **FFN block output** | **1.00e-09, PASS** |

The SwiGLU figure is bf16 output rounding, not a defect — that kernel documents
rounding to bf16 after the SiLU and again after the multiply, and 3e-05 on values
whose maximum is ~7e-03 is exactly bf16's ~0.4%.

This is **31.5 MB of the layer's 38.0 MB of weights**, and it needs no attention,
no KV cache and no RoPE — which is why it was the right block to do first.

**Scope, stated plainly: this is correctness, not throughput.** It runs the
single-core verification design with host glue between stages, so there is still
no tok/s number. Fusing the stages into one dispatch needs the GEMV to emit bf16
and the intermediate to stay on device.

### The constraint that will shape the fused layer

**`down_proj` is L1-constrained in a way gate/up are not**, and it is the
activation, not the weights, that does it. At K=8192 the activation is 16384 B,
32768 B double-buffered — half the 64 KB tile memory before a single weight byte
lands. So the weight tile must shrink:

| linear | K | act (x2) | rows/tile | wtile (x2) | total |
|---|---|---|---|---|---|
| gate / up | 2048 | 8192 | **16** | 40960 | 49152 |
| down | 8192 | 32768 | 4 | 40960 | 73728 — **overflows** |
| down | 8192 | 32768 | **2** | 20480 | 53248 |

Since fewer rows per tile amortises the per-call work worse (the 8->16 row trend
measured +34% earlier), down_proj will run at a lower per-core rate than gate/up
for purely geometric reasons. Two ways out when this is fused, both untried:
drop the activation fifo to depth 1 (it is acquired once and held, so the second
buffer is dead weight — that alone frees 16 KB), or hold the K=8192 activation in
**neighbour memory** rather than replicating it into every core's L1.

### Two traps

- **The existing `silu_mul_bf16` takes its element count as a runtime
  `const int32_t`**, and passing a scalar through IRON's `ExternalFunction`
  `arg_types` makes it allocate a *buffer* for the scalar; the design then fails
  with `Basic sequential allocation also failed`. Omitting the argument instead
  makes the core loop on garbage and the dispatch **time out**. Fixed by a thin
  wrapper (`kernels/npu/flm_swiglu.cc`) that binds the count at compile time.
- **A failed build is cached.** After the fix the run reported the *same* cache
  hash and the same failure; `rm -rf ~/.npu/cache` was needed to see the real
  error. Worth knowing before diagnosing a fix that "did nothing".

## 2026-07-31 — Decode attention runs, and AIE2P's hardware exp2 is 18x worse than bf16

`kernels/npu/flm_attn_decode.cc` + `tools/npu/flm/attn_verify.py`. `attn.xclbin`
was never rebuilt; this is the missing operator, for the decode path.

One GQA group per core (llama-3.2-1B: 32 query heads over 8 KV heads, ratio 4,
head_dim 64), online softmax so the cache streams once, verified against a
float64 numpy reference:

| seq | KV streamed | max abs err | mean\|ref\| | verdict |
|---|---|---|---|---|
| 512 | 0.13 MB | 5.41e-04 | 0.01098 | PASS |
| 2048 | 0.52 MB | 2.02e-04 | 0.00574 | PASS |

**The operand orientations are the design, and they came straight out of the
reverse engineering.** K is stored **channel-major** `[HEAD][TSEQ]` so scores
accumulate *across the head dim* — each `mac` adds a whole 32-position vector and
there is no horizontal reduce anywhere in the score path. V is **position-major**
`[TSEQ][HEAD]` so the output accumulates across positions. The two operands want
opposite layouts, which is exactly why FLM does the KV layout transform in the
memtile DMA rather than in a core.

The 1/sqrt(head_dim) and log2(e) factors are folded into Q on the host, so the
softmax exponential is a bare hardware `exp2` on an accumulator with no
pre-multiply.

### The accuracy floor is the hardware exp2, not bf16

The first run missed a 2% tolerance at 4.9% relative error. It was not the
kernel. Measured directly — a probe that exp2s a known ramp and compares against
numpy:

| | value |
|---|---|
| AIE2P hardware `exp2`, max rel err over x in [-8,0] | **5.86%** |
| mean rel err | **3.54%** |
| bf16 rounding alone would be | 0.20% |

**The NLF unit is a coarse piecewise approximation, ~18x worse than the format
it returns.** Softmax probabilities inherit that directly, and an attention
output is a probability-weighted average, so it passes through undiminished.
Corroboration that this is known behaviour and not a misuse: the existing
`silu_mul_bf16` deliberately uses scalar F32 sigmoid on AIE2P rather than the
vector NLF, with a comment about matching the PyTorch BF16 reference.

**Consequence for phase 3b (KVarN).** That phase plans to quantize K/V and
measure the accuracy cost. This says a **3.5% softmax error is already present
before any KV quantization**, so a KVarN ablation measured on this attention path
would be reading its own noise floor unless the exponential is fixed first.
Fixing it means range reduction plus a polynomial for the fractional part
instead of the NLF, and that cost has not been measured.

### Two traps, one of them for the second time

- **One entry point per translation unit.** Three `ExternalFunction`s on one
  source file link it three times: `duplicate symbol: flm_attn_finish`. This is
  the *same* trap that hit `flm_asum_prepare` earlier the same day — it is now
  split into `flm_attn_decode.cc` (state + tile), `flm_attn_begin.cc` and
  `flm_attn_finish.cc`, sharing the softmax state by `extern`.
- **No scalar libm on the core**: `__builtin_exp2f` fails to link with
  `undefined symbol: exp2f`. The rescale factor goes through the same vector
  `exp2` with one lane extracted.

## 2026-07-31 — Upstream issue 2406 corroborates the bitwise limits, and qualifies the shuffle finding

[Xilinx/mlir-aie#2406](https://github.com/Xilinx/mlir-aie/issues/2406), "Multi-head
attention fails to compile for aie2p" (closed, labelled question), reports the
**identical** backend failure hit twice in this work:

```
fatal error: error in backend: unable to legalize instruction:
%140:_(<4 x s32>) = G_AND %78:_, %138:_
```

theirs from `bit_and`/`bit_xor`/`downshift` inside a bf16 sin/cos polynomial,
ours from `aie::bit_and` on a 16-lane uint8 vector. Same instruction, same
target, same family of intrinsics — so this is a **known upstream gap on aie2p,
not a misuse of the API**, and the working shape (do bitwise work 256 bits wide,
avoid `downshift` on uint8 entirely) is a workaround for a real bug rather than a
style preference.

The issue also reports **`::shuffle_T16_8x2` missing for NPU2**, with the
reporter falling back to implementing the interleave element by element. That
**qualifies the shuffle/"twizzle" finding** recorded earlier: the network exists,
the mode table in `aie2p_enums.h` is real, and `interleave_unzip` measured at one
extra instruction — but **not every mode in that table has an NPU2
implementation**. Check a specific mode compiles before designing a data layout
around it; the earlier entry implied the whole table was available.

## 2026-07-31 — Fused gate/up + SwiGLU: one kernel, one dispatch, 16 cores

`kernels/npu/flm_ffn_gate_up.cc` + `tools/npu/flm/ffn_fused.py`. `ffn_verify.py`
proved the FFN arithmetic but ran four dispatches with host glue; this fuses the
first half into one kernel and one dispatch.

**It fuses because that half is entirely local.** gate and up read the *same*
activation and produce the *same* output rows, and SwiGLU combines them
element-wise — so one core does all three for its slice and **the 8192-wide
intermediate is never materialised in memory at all**. No cross-core dependency,
no barrier. (down_proj is the opposite: its activation is the whole SwiGLU
output, so it needs every core's slice and stays a separate phase.)

| cores | MB | GB/s | wall us | vs FLM | max err |
|---|---|---|---|---|---|
| 8 | 10.5 | 20.6 | 508.8 | 0.45x | 3.40e-04 |
| 16 | **21.0** | **41.4** | 506.5 | 0.90x | 3.40e-04 |

21.0 MB is exactly gate + up for llama-3.2-1B (10.5 each). The error is against a
float64 reference and sits at the hardware-exp2 floor (3.54% mean / 5.86% max
relative), not at bf16.

**Fusing costs 15% of GB/s here, and the reason is L1, not the arithmetic.** The
fused tile carries both the gate and the up weights, so it is 2x the size, and
the 64 KB tile memory then forces **8 rows per tile instead of 16**:

| | tile | x2 buffered | + act | total | rows |
|---|---|---|---|---|---|
| plain GEMV | 20480 | 40960 | 8192 | 49152 | **16** |
| fused | 2 x 10240 | 40960 | 8192 | 49152 | **8** |
| fused at 16 rows | 2 x 20480 | 81920 | 8192 | 90112 | does not fit |

and 8 rows/tile was already measured ~25% worse than 16 for the plain GEMV. So
41.4 against the plain GEMV's 48.6 is the row count, not the fusion. The trade is
**one dispatch and no 8192-element host round trip, against 15% of streaming
rate** — which of those wins is an end-to-end question that needs a tok/s number,
not a GB/s one.

This is also a second, independent instance of the same pressure `down_proj`
showed: **L1 capacity, not compute, is what limits tile size in every fused
shape so far.** It sharpens the case for the untried neighbour-memory lever —
the activation is replicated into every core at 8192 B double-buffered, and
freeing that is worth a row-count step in every one of these designs.

## 2026-07-31 — RMSNorm verified bit-exact, and a rounding-mode bug in a shipped kernel

`kernels/npu/flm_rmsnorm.cc` + `tools/npu/flm/rmsnorm_verify.py`. Needed twice
per decoder layer, so every candidate fused-layer design needs it. It wraps the
existing `tools/npu/rms_norm_weighted_bf16.cc`, whose math already matches llama
exactly — `x * rsqrt(mean(x^2) + eps) * w` with `eps = 1e-5`, which is
llama-3.2-1B's `rms_norm_eps`.

**`rms_norm_weighted_bf16` never calls `aie::set_rounding`, so every bf16
conversion in it truncates.** Confirmed exactly rather than inferred: a
truncating numpy emulation reproduces the device's error to the last digit —
**1.2320e-02 both** — where a round-to-nearest emulation gives 8.81e-03. One line
in the wrapper (`set_rounding(conv_even)`; the mode is core-global, so it fixes
the shared kernel's behaviour on our path without editing a kernel other paths
depend on):

| | max abs err | max rel err |
|---|---|---|
| truncating (as shipped) | 1.2320e-02 | 1.479% |
| round-to-nearest | **8.8108e-03** | **0.890%** |

**−28% error for one line.** This is the *third* time the default truncating
rounding mode has cost accuracy here (13% on the GEMV, and `silu_mul_bf16` does
set it — so this kernel is the outlier, not the rule). `grep -c set_rounding`
over `tools/npu/*.cc` is worth running on any kernel before trusting its
numerics. **`rms_norm_weighted_bf16` is used by the shipped Qwen3.5 NPU path**
per its own docstring, so this is a live accuracy bug there too, not only here.

With the mode set, the device matches its own bf16 chain **exactly (0.0e+00)**.

The remaining 0.89% is inherent to the kernel and is left alone: three bf16
roundings (inv_rms, x*inv_rms, *weight), of which the `inv_rms` broadcast is a
**systematic 0.26% gain error applied uniformly to the whole vector** — a scale
error on every activation entering the projections, not a random one. Whether
that matters end to end is a question for the first tok/s comparison.

### A metric that has now misled twice

Reporting an absolute error against **mean**|ref| inflates it wherever the
reference has wide dynamic range. It read as "7% error" on this kernel and as
"4.4%" on the SwiGLU earlier, both of which are artefacts — the true relative
errors are 0.89% and ~0.4%. The harness now reports **elementwise relative
error** over elements above 5% of peak, plus a best-fit-gain decomposition that
separates a systematic scale error from elementwise noise. Use that shape for
any future operator check.

## 2026-07-31 — down_proj is the slow phase, and the rows-per-tile trade FLIPS with K

Two results while the fused-layer design was being worked out.

### The activation fifo does not need double buffering

It is acquired **once** and held for the whole tile loop, so a second buffer has
nothing to overlap with — it is dead L1. `ObjectFifo(act_ty, depth=1, ...)`.
At K=8192 that frees **16384 B**, and dropping the worker stack from 8192 to
4096 (ample now that the block-sums are a file-scope static rather than a stack
array) frees another 4096. Together they take K=8192 from a hard 2-rows-per-tile
limit to 4:

| K=8192 | wtile x2 | + act | + stack | total |
|---|---|---|---|---|
| 2 rows, act depth 2, stack 8192 | 20480 | 32768 | 8192 | 61440 |
| 4 rows, act depth 2 | 40960 | 32768 | — | **73728 over** |
| 4 rows, act depth 1, stack 4096 | 40960 | 16384 | 4096 | **61440 fits** |

The stack counts against the same 64 KB as the buffers — forgetting it is what
made the first 4-row attempt fail after the arithmetic said it would fit.

### But 4 rows is SLOWER, and that inverts the K=2048 result

down_proj shape, K=8192, 16 cores, 10.5 MB (the real down_proj size), both
correctness-checked at 1.16e-06:

| rows/tile | wtile | banks | GB/s |
|---|---|---|---|
| **2** | 10240 | fits one | **28.4** |
| 4 | 20480 | straddles two | 26.3 |

At K=2048 the *straddling* 16-row tile beat the bank-clean 12-row tile (48.1 vs
44.8) because row amortisation dominated. At K=8192 the direction **reverses**.
The mechanism is bank pressure: the activation at K=8192 is 16384 B = **exactly
one full bank**, and it is read on every iteration, so a weight tile that
straddles banks now collides with it far more often than at K=2048 where the
activation is a quarter of a bank. **The rows-per-tile optimum is not a constant
— it depends on how much of a bank the activation occupies**, which is set by K.

So: keep depth=1 (it is strictly better on capacity and costs nothing), but keep
**2 rows/tile at K=8192**.

### down_proj is the FFN's slow phase

| phase | MB | GB/s | wall |
|---|---|---|---|
| gate + up fused | 21.0 | 41.4 | 507 us |
| down_proj | 10.5 | **28.4** | 370 us |

down_proj moves **half** the bytes of gate+up but takes **73%** of the time, at
**58%** of the plain GEMV's 48.6 GB/s. The cause is geometric, not arithmetic:
K=8192 forces a tiny weight tile because the activation is 8x larger. Any fused
layer design should treat down_proj as the phase to optimise, and the obvious
lever is the one still untried — put that 16 KB activation in **neighbour
memory** instead of every core's L1, which would free the whole budget for
weights.

## 2026-07-31 — RoPE: full rotary, and `rope_freqs.weight` identified

`kernels/npu/flm_rope.cc` + `tools/npu/flm/rope_verify.py`.

**The existing `rope_rotate_bf16` could not be reused.** It is Qwen3.5-specific:
`partial_rotary_factor = 0.25`, so it rotates only `head_dim/4` dims and passes
the rest through. llama-3.2 rotates all 64. New kernel, half-split convention
matching HF's `rotate_half`, using `aie::msc` for the `x*cos - y*sin` — the same
`vmsc.f` that `flm-layer-dataflow.md` used to LOCATE RoPE in FLM's array, since
that opcode occurs in `layer.xclbin` essentially nowhere else.

**`rope_freqs.weight` [32] is the per-frequency llama3 DIVISOR, not the
frequencies.** Its values are `[1]*15, 1.648, 3.297, 9.688, [32]*14` — the
textbook llama3 shape: unscaled at high frequency, `/factor` at low frequency,
smooth in between. So

    inv_freq = 1 / theta**(2i/head_dim) / rope_freqs[i]

and this matches an independent re-derivation of the llama3 wavelength
interpolation (`factor=32, low_freq_factor=1, high_freq_factor=4,
original_max_position_embeddings=8192`) to **0.21% max**, which is inside
bf16 storage error. **The scaling does not need reimplementing** — it is a
divide. The harness asserts the agreement rather than assuming it.

Verified bit-exact against its own bf16 chain, **0.0000e+00** at both a normal
position and one far past the original context:

| | max rel vs float64 | vs own bf16 chain |
|---|---|---|
| pos 1000, 32 query heads | 2.009% max / 0.150% mean | **0.0e+00** |
| pos 100000, 8 KV heads | 1.104% max / 0.184% mean | **0.0e+00** |

The 2% against float64 is one element of 1722 and is **cancellation**, not
error: `x*cos - y*sin` is a difference of similar-magnitude terms, so relative
error blows up wherever it cancels hardest no matter how exact the kernel is.
The mean, 0.150%, is the honest figure. Third operator where the float64
comparison misleads and the bf16-chain comparison is the real gate.

### Operator coverage for a decoder layer

RMSNorm, q/k/v/o projections, RoPE, decode attention with online softmax,
gate/up+SwiGLU fused, and down_proj are all now built and verified on hardware.
What remains before a layer can run end to end is the **residual adds** (trivial)
and the **KV cache append**, then the fusion itself and a tok/s number.

## 2026-07-31 — First tok/s PROJECTION from measured phases: 0.81x FLM today, 1.05x fused

Every phase of a decoder layer is now measured on hardware, so a per-token time
can be projected from measured numbers rather than guessed. **This is a
projection, not a measurement — no token has been produced.**

| phase | MB | GB/s | us |
|---|---|---|---|
| q/k/v/o (K=2048, N=5120) | 6.55 | 33.0 | 198.4 |
| gate+up fused (K=2048, N=8192) | 21.00 | 41.4 | 507.0 |
| down (K=8192, N=2048) | 10.49 | 28.4 | 370.0 |
| **per layer** | **38.04** | **35.4** | **1075.4** |

Against the 38,010,880 B/layer the container arithmetic gives, 38.04 MB is the
whole layer's weights — nothing is missing from the accounting.

    16 layers            17.21 ms
    lm_head @ 48.6 GB/s   3.38 ms  (164.2 MB)
    TOTAL                20.59 ms/token  ->  48.6 tok/s  =  0.81x FLM
    FLM measured         16.71 ms/token  ->  59.86 tok/s

**If every phase ran at the 48.6 GB/s the big shape already achieves: 15.90 ms
-> 62.9 tok/s = 1.05x FLM.**

### The gap is dispatch overhead, and it must not be double-counted

The obvious second calculation — 49 dispatches x ~100 us fixed cost = 4.9 ms,
24% of the total — lands on the same answer (15.89 ms, 63.0 tok/s). That is
**not** an independent second loss. The measured per-phase GB/s *includes* the
fixed dispatch cost, so "phases run below peak" and "dispatch overhead" are two
views of one thing. Adding them would be wrong.

Checking it per phase — excess over what pure streaming at 48.6 GB/s would take:

| phase | pure-stream us | measured us | excess |
|---|---|---|---|
| q/k/v/o | 135 | 198 | **63** |
| gate+up | 432 | 507 | **75** |
| down | 216 | 370 | **154** |

Two of three sit at or under the ~100 us per-dispatch floor established in
milestone 1, so their below-peak rate **is** the fixed cost. down_proj's 154 us
exceeds it by ~54 us, which is the genuinely geometric part (K=8192 forces a
2-row weight tile).

**So fusion is the whole remaining lever, and it is worth 0.81x -> 1.05x.** That
also revises the earlier reading of down_proj as "the slow phase": most of its
apparent slowness is the same per-dispatch cost every phase pays, not its
geometry. Its intrinsic penalty is ~54 us per layer, ~0.9 ms per token, ~4%.

This is the clearest statement yet of why FLM issues **2 commands per token**,
and it puts a number on what the fused layer is worth before it is built.

## 2026-07-31 — lm_head measured at 55.1 GB/s, and the "memtile split costs 12%" claim is WRONG

The tok/s projection assumed lm_head at 48.6 GB/s — the only phase not measured.
Measured now, and it is the largest single dispatch of real GEMV work done here:
**164.2 MB in one command at 55.1 GB/s**, 1.19x FLM's decode rate and **98% of
the 56.5 GB/s fabric roof**, correctness-checked at 4.99e-07.

### The correction, and it retracts something in the kickoff

Earlier I measured a no-op body at 38 MB, got **49.3 GB/s**, compared it against
`dispatch_bw_probe`'s **56.2 GB/s** at 512 MB, and concluded "the memtile split
costs ~12%, and recovering it is worth more than anything left in the body".
That comparison was **apples to oranges** — the two numbers are at transfer sizes
where the ~100 us fixed dispatch cost is a completely different share of the wall
clock:

| measurement | MB | GB/s | wall us | ~100 us fixed cost is |
|---|---|---|---|---|
| no-op body, 16 cores | 38.0 | 49.3 | 771 | **13.0%** |
| **real GEMV, 16 cores** | **164.2** | **55.1** | 2980 | 3.4% |
| dispatch_bw_probe, 8 direct streams | 512.0 | 56.3 | 9094 | 1.1% |

A **real GEMV** at 164 MB reaches 55.1 against a **no-op** at 38 MB reaching
49.3. The split is not costing 12% — at matched fixed-cost share it costs ~2%.
**The memtile split is nearly free, and the "widen the dataflow" task in the
phase-3 kickoff is largely chasing a measurement artefact.** Corrected there.

The lesson is the one this project keeps relearning: **never compare two rates
measured at different transfer sizes on a machine with a large fixed
per-dispatch cost.** Normalise, or measure both at the same size.

### The projection, updated

| | ms/token | tok/s | vs FLM |
|---|---|---|---|
| lm_head assumed @ 48.6 | 20.59 | 48.6 | 0.81x |
| **lm_head measured @ 55.1** | **20.19** | **49.5** | **0.83x** |
| ceiling: whole token at 55.1 GB/s | 14.02 | **71.3** | **1.19x** |

So the reproduction projects to **0.83x FLM as separate dispatches**, with a
**1.19x ceiling** if the whole 772.3 MB streamed at the rate a large dispatch
already demonstrates. Every bit of that 0.83 -> 1.19 gap is per-dispatch fixed
cost, which is precisely what fusing to FLM's 2-commands-per-token removes —
and lm_head, at 164 MB in one command, is the existence proof that the rate is
reachable.

## 2026-07-31 — The per-dispatch cost, fitted: 92.9 us and 57.0 GB/s (R^2 = 0.99997)

Everything now hinges on how much fusing dispatches is worth, and that was still
resting on a ~100 us figure inherited from milestone 1's *small-buffer* probe.
Measured directly for **this** design by sweeping transfer size at fixed geometry
(K=2048, 16 rows/tile, 16 cores) and fitting:

    time_us = 92.9 + 17.547 * MB          R^2 = 0.99997

| MB | measured us | fit us | err | fixed cost is |
|---|---|---|---|---|
| 6.6 | 207.3 | 208.7 | +0.7% | **44.8%** |
| 12.8 | 325.0 | 317.5 | −2.3% | 28.6% |
| 25.6 | 544.0 | 542.0 | −0.4% | 17.1% |
| 38.0 | 754.6 | 759.6 | +0.7% | 12.3% |
| 76.0 | 1420.0 | 1426.4 | +0.5% | 6.5% |
| 164.2 | 2977.3 | 2974.0 | −0.1% | **3.1%** |

Two numbers fall out, and both matter:

- **92.9 us of fixed cost per dispatch.** An independent confirmation of the
  ~100 us figure, now measured on the real design rather than a synthetic probe.
- **57.0 GB/s asymptotic rate** — at or fractionally above the 56.5 GB/s stated
  fabric roof, and above `dispatch_bw_probe`'s 56.3. **Once fixed cost is
  amortised this design streams at the roof.** That independently confirms the
  retraction of the "memtile split costs 12%" claim: the split costs nothing
  measurable.

A single number explains every previously confusing rate: 33.0 GB/s for q/k/v/o,
41.4 for the fused gate/up, 48.6 at 38 MB, 55.1 for lm_head are all the *same*
design at different transfer sizes, and the fixed-cost column above is the whole
story.

### What fusion is worth, from the model

772.3 MB/token:

| dispatches/token | ms | tok/s | vs FLM |
|---|---|---|---|
| 49 (3/layer + lm_head, today) | 18.10 | 55.2 | 0.92x |
| 17 (1/layer + lm_head) | 15.13 | 66.1 | 1.10x |
| **2 (FLM's structure)** | **13.74** | **72.8** | **1.22x** |

**Scope of the model, stated so it is not over-read.** It is fitted on the
K=2048 / 16-rows-per-tile geometry. `down_proj` cannot use that geometry
(K=8192 forces 2 rows) and carries a measured ~54 us/layer intrinsic penalty on
top, so add ~0.86 ms/token: **1 dispatch/layer -> 15.99 ms -> 62.5 tok/s
(1.04x)**, **2 dispatches -> 14.60 ms -> 68.5 tok/s (1.14x)**. Those are the
honest figures.

Either way the conclusion is unchanged and now quantitative: **fusion is the
whole remaining lever, it is worth roughly 0.92x -> 1.1-1.2x FLM, and nothing
else on the table is worth more than a few percent.**

## 2026-07-31 — Correction: today's dispatch count is 81, not 49, so today is 0.79x not 0.92x

The previous entry projected "49 dispatches (3/layer + lm_head) -> 0.92x FLM"
and called it today's figure. **That counted the phases I had benchmarked, not
the phases a decoder layer actually has.** o_proj cannot share a dispatch with
q/k/v because attention sits between them, and attention is its own dispatch.
The realistic minimum with the operators as built is **5 per layer** — qkv,
attention, o_proj, gate/up, down — so **81 per token**, not 49.

| dispatches/token | | ms | tok/s | vs FLM |
|---|---|---|---|---|
| 161 | fully unfused (~10 ops/layer) | 28.51 | 35.1 | 0.59x |
| **81** | **5/layer — realistic today** | **21.07** | **47.5** | **0.79x** |
| 49 | 3/layer — *what the last entry claimed* | 18.10 | 55.2 | 0.92x |
| 17 | 1/layer + lm_head | 15.13 | 66.1 | 1.10x |
| 2 | FLM's structure | 13.73 | 72.8 | 1.22x |

Streaming floor is 772.3 MB / 57.0 GB/s = **13.55 ms**; everything above it is
fixed cost.

### The sharper point: half a layer's operators move almost no bytes

At 92.9 us per dispatch, an operator's fixed-cost share depends only on its size:

| operator | MB | fixed cost is |
|---|---|---|
| input_layernorm | 0.004 | **99.9%** |
| RoPE (Q and K) | 0.005 | **99.9%** |
| residual add (x2) | 0.004 | **99.9%** |
| post_attn_layernorm | 0.004 | **99.9%** |
| q/k/v proj | 3.932 | 57.4% |
| o_proj | 2.621 | 66.9% |
| gate/up + SwiGLU | 20.972 | 20.2% |
| down_proj | 10.486 | 33.5% |

**A 4 KB RMSNorm costs the same 92.9 us as 38 MB of weights.** Five of the ten
operators per layer move ~4 KB each and are essentially *pure* fixed cost — if
each got its own dispatch they alone would cost 7.4 ms/token, more than half the
entire streaming floor.

So the fused-layer requirement is not mainly about bandwidth. It is: **every
operator that does not share a dispatch with a large one costs 92.9 us, and half
of them move nothing.** The elementwise ops must ride along with the projections
(as SwiGLU already rides with gate/up), not merely be "fused for tidiness".

(The previous entry's per-operator table also had `down_proj` at 20.97 MB from a
bad expression; it is 2048 rows x 5120 B = **10.486 MB**, and the corrected
per-layer total is 38.03 MB against the container's 38,010,880 B. The
dispatch-count projections used the independent 772.3 MB/token figure and were
not affected.)

## 2026-07-31 — RMSNorm fused into the GEMV prologue: 2 dispatches/layer gone, free

Acting on the previous entry's finding that half a layer's operators move ~4 KB
and are ~99.9% fixed cost. `kernels/npu/flm_norm_prepare.cc` replaces
`flm_asum_prepare` where a projection is preceded by a norm.

**Why it is free.** `flm_asum_prepare` already walks the entire activation to
compute the q4_1 block sums. RMSNorm's sum-of-squares rides in that same walk,
and the normalisation is applied **in place in the ObjectFifo buffer**, so the
GEMV that follows is *completely unchanged* — same pointer, same code. Passes
over the activation:

    standalone RMSNorm (2) + asum_prepare (1) = 3
    fused                                     = 2

so it is **one pass cheaper** than the two operators were separately, on top of
removing a dispatch.

Verified against a norm-then-GEMV reference carrying the kernel's own bf16
roundings: **3.48e-07** (layer 0) and **3.58e-07** (layer 7) — float32
round-off. The plain GEMV, the fused FFN and the attention path all re-checked
unchanged after the shared-header edit.

**Worth 2 x 92.9 us x 16 layers = 2.97 ms/token**, ~14% of today's projected
21.07 ms, for no bytes and no extra L1.

### Two IRON limits hit, both structural

- **A core tile has only 2 input DMA channels.** Giving the norm weight its own
  fifo made three inputs (act, norm weight, weights) and the placer rejected it:
  *"reduce the LTO's DMA fanin (e.g. via memtile staging)"*. The fix is to carry
  the norm weight in the **same buffer** as the activation — `[act K][nw K]` —
  which costs one extra 4 KB of L1 and no channel. This is a general constraint
  on the fused layer: **every operand a core needs is a DMA input, and it gets
  two**, so operands must be packed together or staged through a memtile.
- **A definition in a shared header links N times.** `g_asum` was defined in
  `flm_q4_1_tile.h`, which was fine while one TU included it; the fused FFN and
  now the fused norm made three. `duplicate symbol: g_asum`. It is now `extern`
  in the header with one definition in `flm_gemv_q4_1.cc` — the same
  one-definition discipline the multiple entry points already needed.

### Dispatch budget after this

Per layer the tiny operators were: 2 norms, RoPE, 2 residual adds. **The two
norms are now free.** Remaining to fuse: RoPE (rides with the qkv projection's
epilogue) and the two residual adds (ride with o_proj's and down_proj's
epilogues) — all three are epilogue/prologue work on an operator that already
exists, none needs a new dispatch.

## 2026-07-31 — Residual add fused into the GEMV epilogue, and an alignment trap with a very clean signature

`kernels/npu/flm_gemv_residual.cc`: `out[r] = W_tile[r] . act + residual[r]`.
The second of the tiny per-layer operators to stop costing a dispatch. Verified
**3.48e-07** with the norm ALSO fused (so norm + GEMV + residual in one kernel),
and the non-residual path unchanged at 3.48e-07.

**Getting the residual to the core is constrained by the 2-input-channel limit**
found last entry. The activation and the weight stream already use both, so the
residual rides *inside the weight tile*, appended after the codes. That costs 64
bytes on a 20480-byte tile (**0.3%**) to remove a 92.9 us dispatch, and it solves
the addressing problem for free — each core receives exactly the residual rows it
is computing, with no notion of its own row offset.

### The trap: the region is a fixed 64 bytes, not NROWS*2

With `NROWS*2` the tile is 20512 B. The weight ObjectFifo is double-buffered, so
buffer 1 lands 20512 bytes after buffer 0 — a **32-byte** boundary, not 64 — and
the vectorised residual load off it reads garbage.

The symptom is one of the cleanest in this project:

    max err per tile, 32 tiles:
    [2.1e-07  6.7e-01  8.3e-08  9.8e-01  1.5e-07  7.3e-01  2.1e-07  7.1e-01 ...]

**Even tiles exact, odd tiles wrong by ~1.0, strictly alternating** — because the
fifo alternates buffers and only one of the two is 64-byte aligned. Padding the
tile to a multiple of 64 fixes it. Worth remembering as a diagnostic: *an
alternating-by-tile error pattern means a double-buffer alignment problem, not an
arithmetic one.*

### Also: `g_asum` is now an `inline` variable, which is the only arrangement that links

Three arrangements were tried and only the third works, because IRON compiles
each ExternalFunction's source separately and links **only the objects whose
entry points a design actually calls**:

| arrangement | failure |
|---|---|
| defined in the shared header | `duplicate symbol` once two includers link |
| `extern` + defined in `flm_gemv_q4_1.cc` | `undefined symbol` for any design not using that entry point — e.g. residual-GEMV + fused-norm |
| own TU + a no-op "anchor" entry point | still `undefined symbol`: an uncalled ExternalFunction's object is not linked |
| **`alignas(64) inline bfloat16 g_asum[...]` in the header** | **works for every combination** |

C++17 inline variables are exactly the right tool here and the reasoning is now
in the header.

### Dispatch budget

Of the five tiny operators per layer, **four are now free**: both RMSNorm's
(previous entry) and both residual adds. Only RoPE remains, and it rides with
the qkv projection's epilogue by the same mechanism. That is
**4 x 92.9 us x 16 layers = 5.9 ms/token** removed, against a 13.55 ms streaming
floor.

## 2026-07-31 — RoPE fused into attention: all five tiny per-layer operators are now free

RoPE **cannot** ride in the qkv projection's epilogue, which was the obvious
place: at 16 rows per weight tile a q4_1 tile produces a quarter of a
head_dim-64 head, and RoPE needs whole heads. NROWS=64 would make the tile
81920 B, far past the 64 KB tile memory.

It rides in `flm_attn_begin` instead, which runs **once per token** and already
owns this core's Q — so rotating in place there costs no dispatch and no extra
pass. `cs` rides in the same buffer as Q (`[GQA*HEAD q][HEAD cs]`) because the
attention core's two input DMA channels are already taken by Q and the KV
stream — the same packing trick the norm weight uses in `flm_norm_prepare`.

Rotation and the softmax scale commute (the scale is a scalar), so Q can still
be pre-scaled by `1/sqrt(head_dim) * log2(e)` on the host and rotated on device.

Verified across positions and context lengths, against a reference that rotates
Q in float64 and then attends:

| seq | pos | max abs err | tolerance | |
|---|---|---|---|---|
| 512 | 1000 | 6.63e-04 | 8.71e-04 | PASS |
| 2048 | 100000 | 1.68e-04 | 4.62e-04 | PASS |
| 1024 | 0 | 3.50e-04 | 6.86e-04 | PASS |

(The tolerance is set by AIE2P's hardware `exp2`, 3.54% mean / 5.86% max, not by
this kernel.)

### All five tiny operators are now free

| operator | rides in |
|---|---|
| input_layernorm | `flm_norm_prepare` — the GEMV's activation prologue |
| post_attention_layernorm | same |
| residual add (after o_proj) | `flm_gemv_q4_1_residual` — the GEMV epilogue |
| residual add (after down_proj) | same |
| **RoPE (Q)** | **`flm_attn_begin` — attention's per-token prologue** |

**5 x 92.9 us x 16 layers = 7.43 ms/token** of pure fixed cost removed, against a
13.55 ms streaming floor. None of it cost a byte of extra bandwidth beyond 64 B
per weight tile for the residual.

| dispatches/token | | ms | tok/s | vs FLM |
|---|---|---|---|---|
| 81 (5/layer, before this work) | | 21.07 | 47.5 | 0.79x |
| **17 (1/layer, all elementwise fused)** | | **15.13** | **66.1** | **1.10x** |
| 2 (FLM's structure) | | 13.73 | 72.8 | 1.22x |

The remaining step from 17 to 2 is the cross-core barrier — `down_proj` needs
the whole 8192-wide SwiGLU output, so it needs every core's slice — plus the KV
cache append. Those are the only structural pieces left.

## 2026-07-31 — Fused-layer falsifier PASSES: an in-dispatch barrier is 6.37 us, 14.6x cheaper than a dispatch

An 11-agent design workflow produced `docs/npu/flm-fused-layer-plan.md` (surveys
-> three competing designs -> adversarial judges -> synthesis). Its own Task 0 is
a falsifier, and it named the single number the plan was missing: **what does a
phase barrier INSIDE one dispatch cost?** The whole design — one dispatch per
layer with 5 internal barriers replacing 5 dispatches — is a bet that this is
below the measured 92.9 us per-dispatch cost.

`tools/npu/flm/barrier_probe.py`: the same 16-core paired topology, a no-op
kernel, N sequential `fill`/`drain(wait=True)` round trips in ONE runtime
sequence.

| phases/dispatch | us | us/phase |
|---|---|---|
| 1 | 71.6 | 71.6 |
| 5 | 91.9 | 18.4 |
| 20 | 197.0 | 9.8 |
| 80 | 573.0 | 7.2 |

    time_us = 64.7 + 6.37 * phases        R^2 = 0.99971

**6.37 us per in-dispatch barrier against 92.9 us per dispatch — 14.6x
cheaper.** The gate was `< 93 us`; it passes by more than an order of magnitude,
so the fused layer is worth building and the plan's central bet is sound.

| | ms/token | tok/s | vs FLM |
|---|---|---|---|
| today: 81 dispatches, 0 barriers | 21.07 | 47.5 | 0.79x |
| **fused: 17 dispatches, 80 barriers** | **15.64** | **63.9** | **1.07x** |

### The probe found a second thing the plan needs

**80 barriers per token does not compile without BD reuse.** Each `fill`/`drain`
allocates a buffer descriptor, and N phases x 8 pairs exceeds the **16
simultaneously-active BDs a shim tile supports**:

    'aiex.dma_configure_task' op Too many simultaneously active buffer
    descriptors on tile (1,0), which supports up to 16. Emit an
    aiex.dma_free_task / aiex.dma_await_task to reuse BDs.

Wrapping each phase in its own `TaskGroup` fixes it — the same mechanism
milestone 1 needed for many-buffer dispatches, now required for many-*barrier*
ones too. IRON also forbids mixing explicit groups with the implicit default,
so once one fill is grouped they all must be. **The fused-layer plan must group
per phase**; it does not currently say so.

Also worth noting: a single-phase dispatch measures 71.6 us here against the
92.9 us fitted on the GEMV sweep, which is consistent — this probe moves 8 KB
where that fit was over 6.6-164 MB, and the intercept is not identical across
transfer regimes.

## 2026-07-31 — Program-memory falsifier, and a regression the sweep caught

### Task 1 of the fused-layer plan: program memory

Every entry point the fused layer needs, `.text` at K=2048 / NROWS=16 / GQA=4:

| entry point | .text B |
|---|---|
| `flm_ffn_gate_up` | 3888 |
| `flm_gemv_residual` | 1760 |
| `flm_attn_decode` | 1616 |
| `flm_norm_prepare` | 1200 |
| `flm_gemv_q4_1` | 880 |
| `flm_attn_finish` | 544 |
| `flm_attn_begin` | 448 |
| `flm_rope` | 64 |
| `flm_asum_prepare` | (see below) |
| **sum** | **10,400** |

**10,400 B of the 16,384 B program module = 63%**, leaving ~6 KB for the
5-phase control flow. The plan's gate was <=14 KB, so it passes, but not with
the margin the barrier probe had — and this is an *upper* bound in one sense
(the linker discards unused inline copies of the shared tile body, which is
inlined into each) and a *lower* bound in another (no phase control flow yet).
`flm_ffn_gate_up` at 3888 B is by far the largest and is the one the plan
already proposes retiring in favour of two smaller entry points.

For calibration, real linked core ELFs from designs already built measure 560 B
(attention-only) to 1264 B (norm+GEMV+residual) of `.text`, so a single design's
code is a small fraction of the module; it is the *union* over phases that has
to fit.

### The sweep caught a regression I introduced two ticks ago

`flm_asum_prepare.cc` **did not compile**: `use of undeclared identifier
'g_asum'`. When `g_asum` became an `inline` variable in the shared header, I
removed the file's `extern bfloat16 g_asum[];` but the file does not include
that header — it carries its own copy of the constants. Two ticks of work
followed without noticing, because everything built since used
`flm_norm_prepare` instead.

**`gemv_bench.py` and `ffn_fused.py` would both have failed to build.** Fixed by
including `flm_q4_1_tile.h` and deleting the duplicated constants, and all three
affected paths re-verified:

| | |
|---|---|
| `gemv_verify` | PASS |
| `gemv_bench` (4 cores) | 8.9 GB/s, 3.28e-07 |
| `ffn_fused` (4 cores) | 5.1 GB/s, 1.58e-04 |

The lesson is narrow and worth stating: **a file that duplicates a header's
declarations instead of including it will not track that header's changes**, and
nothing catches it until a design that uses that particular entry point is
rebuilt. Compiling every kernel in `kernels/npu/` is a 9-command check and it
found this in one pass — worth doing after any edit to the shared header.

## 2026-07-31 — Task 2: the universal tile trailer, and a third jit-cache trap

Every weight tile now carries a 64-byte trailer after the codes:

    [NROWS*NB bf16 d][NROWS*NB bf16 m][NROWS*K/2 codes][f32 row_base][f32 flags][pad]

One tile shape for every phase — which is what lets a single operand fifo serve
all of them — at **+0.31%** weight traffic. `row_base` is the global output-row
index of the tile's first row and replaces every per-core index the kernels
would otherwise need (residual indexing, RoPE head identity, down-chunk
accumulator slot) with **no runtime scalar arguments and no static cursors**.
The 64 also keeps the tile a multiple of 64, so both halves of a double-buffered
fifo stay aligned — the alternating even/odd corruption from the previous entry
cannot recur.

**The residual moved out of the tile and into the broadcast's aux half**,
indexed by `row_base`. The pack-time trailer it used before *cannot* work in a
fused layer: the residual is computed on-device during the same dispatch, so it
does not exist when the weights are packed. That was a real design error in the
earlier version, caught by the plan rather than by a test.

**The residual phase uses `flm_asum_prepare`, not `flm_norm_prepare`.** The aux
half carries the residual there, so running the norm prologue would normalise
the activation *using the residual as the norm weight*. The harness failed
loudly after the change, which is how it surfaced — a good argument for the
per-phase prologue being explicit rather than defaulted.

`tile_bytes` is now centralised in `q4nx.py` (it has to agree with `pack_tile`'s
trailer); `gemv_verify`, `gemv_bench` and `ffn_fused` had three separate copies.
The fused FFN's up-tile offset becomes `TILE_TOTAL`, past the gate tile **and**
its trailer.

Verified with a **distinct `row_base` per tile** — a constant would have passed
by accident:

| | |
|---|---|
| `gemv_verify`, `normgemv_verify` (±residual) | PASS |
| `rope_verify`, `attn_verify`, `rmsnorm_verify` | PASS |
| `gemv_bench` (4 cores) | 3.28e-07 |
| `ffn_fused` (4 cores) | 1.58e-04 |

### Third instance of the same trap

`tile_bytes` changed, but it is neither a `CompileTime` parameter nor a listed
source file, so it **does not rekey the jit cache**. The cached binary was served
until `~/.npu/cache` was cleared, surfacing as:

    Tensor argument 'w' has 657408 elements but the kernel was compiled for 655360

The pattern is now three for three — `ExternalFunction(source_file=...)`,
closure-captured shapes, and now a module-level constant that feeds the design.
**Anything that changes a design's shapes must be a `CompileTime` parameter, a
listed `source_files` entry, or followed by a cache clear.** There is no fourth
option, and the failure mode is always a stale binary rather than an error.

## 2026-07-31 — Task 3: down_proj K-chunked — 370 -> 280 us, now at 99% of its ceiling

`kernels/npu/flm_gemv_acc.cc` + `flm_gemv_flush.cc` + `tools/npu/flm/down_verify.py`.

down_proj's K=8192 forced a 16384 B activation (32768 double-buffered), and the
64 KB tile memory then allowed only **2 rows per weight tile** against 16 for
every other projection. Splitting K into 4 chunks of 2048 makes it the same
shape as everything else: a 4096 B activation and 16-row tiles. Chunks 0-2
accumulate into a per-core slot; chunk 3 adds, applies the residual from the
broadcast aux half, emits and clears.

**It is exact, and the reason is worth keeping.** The GEMV identity is linear in
blocks, so four 2048-wide partials sum to the same value as one 8192-wide pass.
And the container's planar 5120 B row splits on a chunk boundary with **no code
repacking at all** — chunk c is `d[64c:64c+64]`, `m[64c:64c+64]`,
`codes[1024c:1024c+1024]`, which is precisely a K=2048 tile. Measured against
the monolithic K=8192 reference: **3.28e-08**.

| | wall us | GB/s | marginal us | % of fitted ceiling |
|---|---|---|---|---|
| monolithic, 2 rows/tile | 370.0 | 28.4 | 277.1 | 75% |
| **K-chunked, 16 rows/tile** | **280.1** | **37.6** | **187.2** | **99%** |

The plan's gate was marginal <= 200 us (ideal 184); measured 187.2, within 2% of
ideal. Against the branch's own fitted `t = 92.9 + 17.547*MB`, a 10.52 MB
dispatch cannot beat 277.5 us however good the kernel — **so down_proj is now at
99% of what its transfer size allows, where it was at 75%.** The remaining 25%
was never arithmetic; it was the row count the activation size forced.

**Saving: 89.9 us/layer x 16 = 1.44 ms/token**, against the plan's estimate of
0.86 ms. It also removes the last shape in the design that was not K=2048 /
16 rows, which is what lets one operand fifo serve every phase.

This also retires the earlier reading of down_proj as intrinsically slow: with
the chunking its ~54 us/layer "geometric penalty" is gone, and its efficiency is
now indistinguishable from the other projections.

## 2026-07-31 — Task 4 FALSIFIED: gate/up's rate is not the row count

> **RETRACTED the same day — this entry's measurement was made against a
> miscompiled kernel and its conclusion is backwards.** Both kernels compared
> here were silently computing `g*u/2` instead of SwiGLU (see "the SwiGLU
> sigmoid was a constant 1/2", below). With that fixed, alternating acquires
> **wins**: 467.6 us against the single fused tile's 496.2 us on the same
> 21.0 MB, marginal 374.7 us against a 369.1 us ideal — **98.5% of its transfer
> ceiling**, where this entry measured 433.7. The row count *was* the cause; the
> gate passes. Left in place because the reasoning below is a good example of a
> correct method reaching a wrong answer through an unvalidated input.

The plan called this its most falsifiable claim and named the disconfirming
number in advance. It disconfirmed.

`kernels/npu/flm_gemv_gate.cc` + `flm_gemv_up_swiglu.cc` + `tools/npu/flm/ffn_alt.py`
stream gate and up as **alternating acquires of single 16-row tiles** instead of
one fused tile carrying both (which forces 8 rows). Same bytes, same acquire
count, weight stream reordered offline.

| | wall us | marginal us | GB/s | err |
|---|---|---|---|---|
| single fused tile, 8 rows/tile | **514.0** | **421.1** | 41.1 | 3.40e-04 |
| alternating acquires, 16 rows/tile | 526.6 | 433.7 | 39.9 | 3.40e-04 |
| ideal at 21.0 MB | — | 369.1 | — | — |

Gate was marginal <= 400 us. Measured **433.7 — worse than the incumbent's
421.1**. So:

- **The 8-row tile was NOT why gate/up runs below ideal.** Doubling the rows per
  tile changed nothing for the better; both sit ~55-65 us above the 369 us its
  transfer size allows. Whatever that gap is, it is the *arithmetic* of the
  stage (two GEMV calls plus a SwiGLU per output row), not the geometry — which
  is the opposite of what held for `down_proj`, where the geometry was the whole
  story and chunking recovered 24%.
- **The plan's perf rationale for alternating acquires is withdrawn.**

**But do not delete the kernels, because the L1 argument is separate and still
stands.** The fused layer wants one operand-object shape across every phase:

| | object | x2 depth | + act | |
|---|---|---|---|---|
| fused tile, 8 rows | 20608 | 41216 | 45312 | fits |
| fused tile, **16 rows** | 41088 | 82176 | 86272 | **over 64 KB** |
| **alternating, 16 rows** | **20544** | **41088** | **45184** | **fits** |

A single fused gate/up tile at 16 rows does not fit at all, and at 8 rows its
object is 20608 B against every other phase's 20544 — close, but not the same
shape. So alternating acquires may still be the right choice *for operand-shape
uniformity in the fused layer*, at a measured 2.4% cost on this stage. That is a
different argument from the one the plan made, and the layer build settles it.

Recorded per `AGENTS.md`: a failed approach that narrows the search space. The
useful residue is that **gate/up's ~15% shortfall against its transfer ceiling
is arithmetic, and the SwiGLU's `exp2` is the obvious suspect** — the same NLF
that caps this stage's accuracy at 3.40e-04.

## 2026-07-31 — §1.3 answered, negatively: the quantized reading is NOT the model

`tools/npu/flm/ground_truth.py`. The plan's §1.3 says FLM's weights are
transformed, so the block->(row,k) mapping is unknown and every verification is
self-consistency against the same reading of `model.q4nx`; it assumes this
cannot be settled without `flm run`. That was wrong — `meta-llama/Llama-3.2-1B`
ships `original/consolidated.00.pth`, the actual trained checkpoint, and it is
on this machine at `/srv/huggingface`.

**1. The reader and the model are confirmed, bit-exactly.** RMSNorm weights are
stored unquantized, so they check with no quantization noise at all:

| tensor | bit-exact vs base | bit-exact vs **Instruct** |
|---|---|---|
| `layers.0.input_layernorm.weight` | 5.96% | **100.00%** |
| `layers.0.post_attention_layernorm.weight` | 0.63% | **100.00%** |
| `model.norm.weight` | 47.22% | **100.00%** |

maxdiff 0.000e+00. So the file structure, the planar `[d][m][codes]` split, the
tensor naming, and the model identity (**Instruct**, not base) are all right.

**2. The quantized blocks are not the model's blocks.** The sorted 32 values of
a q4_1 block are invariant to the nibble order `q4nx.py` flags as assumed, so a
block can be matched against all 524,288 ground-truth blocks with no unknowns
left. Best-of-all-blocks rms is **0.15** of mean |w| — q4_1's own error is
~0.03, random is ~1.0 — and the destinations scatter uniformly (753 distinct gt
rows for 1024 container blocks). Those are best-of-half-a-million coincidences.

**3. No container row holds any ground-truth row's blocks**, under any
within-row permutation. Matching a gt row's 256 `(m, d)` points against every
container row's as a set: best 0.0883 vs median 0.1008 for gt row 0, best 0.0816
vs 0.0961 for row 1, best 0.0929 vs 0.1058 for row 7. Best ~= median is no match.

**4. It is a scaling, not a rotation.** The Frobenius norm is also nibble-order
invariant (the block's value multiset is fixed however the elements are
assigned). Every tensor is inflated, and by a *different* factor:

| | down | gate | up | q | k | v | o |
|---|---|---|---|---|---|---|---|
| container/gt \|W\|_F | 1.1422 | 1.1289 | 1.1226 | 1.1636 | 1.1939 | 1.1476 | 1.1552 |

An orthogonal transform gives exactly 1.0000; q4_1 error contributes <0.001. A
1.12-1.19x inflation that varies per tensor is the signature of **per-channel
(AWQ/SmoothQuant-style) scaling**, folded into the weights with the reciprocal
presumably folded elsewhere.

### What this does and does not invalidate

- **Throughput is unaffected.** Every GB/s figure is a byte-movement measurement
  on correctly-sized buffers. 48.1-48.6 GB/s at 16 cores, the 92.9 us dispatch
  cost, the 6.37 us barrier, the down_proj chunking win, today's Task 4
  falsification — all stand.
- **The kernels are verified as q4_1 arithmetic**, against a numpy reference
  over the same bytes, to 1.9e-07. That is a real result about the kernels.
- **They are NOT verified as computing Llama-3.2-1B-Instruct.** Milestone 4's
  bar was "numerically equivalent on real weights"; the bytes are real, the
  arithmetic is right, but the mapping from those bytes to the model's weights
  is not established, so end-to-end equivalence is not shown. Anywhere that
  claim appears it should read "equivalent on the container's q4_1 blocks".
- **RoPE (`-DROPE_INTERLEAVED`, plan Task 5) still cannot be settled**, and this
  is why. A row-order probe based on RoPE pairing was tried and **discarded**:
  its v_proj control, which gets no RoPE at all, showed the same "interleaved"
  signal *more strongly* (5.97x vs 3.62x), so the signal was adjacent-row
  smoothness, not rotation-plane structure. The pairing argument proves nothing.

Do not try to infer the grouping from the `d` (block range) distribution — a
random regrouping of the same weights reproduces it about as well as the true
one. A control built on it was also not reproducible run to run and was removed
rather than shipped.

**Next, if this is worth chasing:** recover the per-channel scale. If the
transform is `w'[r,k] = w[r,k] * s_k`, then `s` is recoverable from block
statistics against ground truth, and it would close §1.3 for real. Until then,
the only true end-to-end check remains logits vs `flm run`.

## 2026-07-31 (later) — correcting the §1.3 probes; the verdict survives, two probes did not

Re-checked the morning's §1.3 work by asking whether its own dequantization was
self-consistent. It mostly was, but **two of the four probes were built on a
false premise and one claim was stated too strongly.** The verdict is unchanged
and now rests on a properly calibrated control.

**FLM's q4_1 is not llama.cpp's.** Testable without any ground truth: a min/max
fit forces code 0 and code 15 into *every* block. Measured on down_proj layer 0,
only **48.5%** of blocks hold both, the code histogram is bell-shaped about 7.4,
and the mean per-block span is **14.08 of 15** — a search fit on a grid ~6.6%
wider than min/max. (48.5% is *below* the 76.3% random 4-bit codes would give,
which is itself the tell: the codes avoid the rails.)

**Retracted — probe 3 (row-level `(m,d)` set match) was invalid.** It compared
container `(m, d)` against each ground-truth row's `(min, (max-min)/15)`. Since
FLM's `d` is not `(max-min)/15`, that probe could only ever fail, whatever the
row mapping was. It has been deleted, not merely caveated. **Never compare
container `d` against `(max-min)/15`.**

**Corrected — probe 2's threshold was wrong, and its conclusion is now stronger.**
The morning claimed "q4_1's own error is ~0.03, the container gives 0.15". The
first number was invented: this tool's own min/max quantization of a ground-truth
block lands at **0.094** rms/mean|w|, not 0.03. The fix is not a better threshold
but the control that was missing — run the search on blocks whose true match is
*masked out of the pool*, to get the score a block earns when its match is
definitely absent. Switching to a scale-invariant fingerprint (cosine on sorted
block values) at the same time:

| best-of-524288 cosine | p50 |
|---|---|
| control, true match **present** | 0.99713 (found it 98.4% of the time) |
| control, true match **absent** (masked) | **0.99503** ← look-alike floor |
| container down_proj | **0.99487** |

The container sits *at the floor*. Sorted 32-value profiles of weight blocks all
resemble one another, so the floor is 0.995 and not 0 — without that control,
0.9949 reads as either "no match by a mile" or "basically a match", by taste.

**Softened — the transform is a scaling, but "per-channel AWQ" was over-claimed.**
The morning asserted per-channel scaling. What is actually measured is that the
container's value distribution is a near-uniform **~1.14x** of ground truth's at
every quantile:

| quantile | .001 | .01 | .1 | .25 | .75 | .9 | .99 | .999 |
|---|---|---|---|---|---|---|---|---|
| container/gt | 1.118 | 1.097 | 1.152 | 1.141 | 1.146 | 1.157 | 1.117 | 1.145 |

std ratio 1.1422, exactly the Frobenius ratio. A single scalar per tensor would
make that row flat; it wobbles ~±3%, so there is some shape change — but this is
much closer to one number per tensor than to a broad per-channel vector, and
"AWQ/SmoothQuant-style per-channel scaling" claimed more than the data shows. A
scaled *rotation* fits it at least as well and is not excluded.

### What stands

Unchanged: the bf16 tensors are 100% bit-exact against Llama-3.2-1B-**Instruct**
(reader, planar split, naming, model identity all confirmed); the quantized
blocks are not the model's under any arrangement; the kernels are verified as
q4_1 arithmetic over the container's blocks and **not** as computing the model;
every throughput figure is untouched.

The lesson is the same one this file recorded this morning, and it cost a second
round: **a probe without a control is not evidence.** Three probes have now
scored confident wrong answers in one day — the RoPE row-order pairing (killed by
its v_proj control), the `d`-distribution grouping (killed by a random
regrouping), and probe 3 above (killed by asking what fit FLM actually uses).
The first two were caught before publishing. The third was not.

## 2026-07-31 — Task 5 done: qkv fused N=3072 with RoPE in the epilogue

`kernels/npu/flm_gemv_qkv.cc`, `flm_qkv_emit.cc`, `tools/npu/flm/qkv_verify.py`.
Phase P1 of the fused layer, whole: RMSNorm fused into the GEMV prologue, one
N=3072 projection (2048 q + 512 k + 512 v), and RoPE per completed head.

| shape | max err vs numpy | note |
|---|---|---|
| 2 cores, 6 q heads | **0.0e+00** | exact, both conventions |
| 4 cores, heads 30–41 | 1.95e-03 | straddles q→k→v, bf16 floor |
| 4 cores, heads 42–53 | **0.0e+00** | pure v, no rotation |
| **16 cores, N=3072, all 48 heads** | **1.95e-03** | 3.94 MB, 171.5 us, marginal **78.6** vs 69.1 ideal (88%) |

Head identity comes from the weight tile's `row_base` trailer
(`head = row_base/64`), so there are no runtime scalar arguments and no static
cursor to get out of step. 192 rows/core = exactly 3 whole heads, so RoPE never
straddles a core — the reason this phase is 16 cores and not 32.

### The `-DROPE_INTERLEAVED` flag turned out to be unnecessary

The plan called this Task 5's open question and said it "must stay a `-D` flag".
It does not need to be a flag at all. **The interleaved pairing `(2i, 2i+1)` is
the half-split pairing `(i, i+32)` applied to a permuted row order** — which is
precisely why llama.cpp's converter permutes q/k weights rather than shipping a
second RoPE. Feeding the kernel `[v0,v2,…,v62, v1,v3,…,v63]` makes half-split
compute the interleaved rotation.

So the convention moves to **pack time**, one kernel serves both, and it is
free. A shuffle-network version using `aie::interleave_unzip`/`zip` was written
first, measured at **308 instructions against 300**, and produced wrong results
(0.42 and 2.26 absolute against a mean |ref| of 0.06 and 0.47); rather than
debug a second code path that did not need to exist, it was deleted. Both orders
now verify exactly.

This is safe for attention because `q·k` is a dot product over the head
dimension: **any permutation shared by q and k leaves it unchanged**, and v is
never rotated so it keeps the model's order for `o_proj`.

Which convention the container actually wants remains open, and per this
morning's §1.3 work it cannot be settled from the weights.

### One trap, and it is the one this repo keeps re-learning

The first run failed at 7.8e-04 against a 6.2e-04 tolerance — 1.25% of
mean|ref|. Cause: the kernel stages the GEMV result in a **bf16** buffer and
rotates *that*, while the reference rotated full-precision values. `x·cos −
y·sin` is a difference of similar magnitudes, so it amplifies the 0.2% bf16
step. Modelling both of the kernel's roundings (stage, then store) takes the
error to **0.0e+00**. Exactly the reason `gemv_reference_bf16` is the gate and
`gemv_reference` is only context.

## 2026-07-31 — Task 6, part 1: attention prepared to be a phase

Two changes the attention kernels needed before they can run as phase P2, plus
the harness work to verify them. `attn_phase.py` (8 cores, KV on the operand
fifo) is not written yet.

### RoPE moved out of attention

`flm_attn_begin.cc` used to rotate Q, and the comment there argued it *had* to:
"at 16 rows per weight tile a q4_1 tile produces a quarter of a head_dim-64
head, and RoPE needs whole heads — NROWS=64 would make the tile 81920 B, far
past the 64 KB tile memory." The premise was right, the conclusion was wrong.
The projection never needed a 64-row tile; it needed somewhere to put four
16-row tiles, and 128 B of core memory is somewhere. Task 5's `flm_gemv_qkv.cc`
does exactly that.

It also *has* to move, for a reason that is not about tidiness: **k′ must be
rotated before it is appended to the KV cache, and the cache is written by phase
P1.** Leaving the rotation in attention would rotate q only and every cached k
would be unrotated. So P1 now owns RoPE for both, `flm_attn_begin` is state-init
only, and `attn_verify.py` rotates Q on the host — which is what makes it
faithful to the pipeline rather than a shortcut.

### Pad correction, and the control that proves it works

The cache streams in whole TSEQ=32 tiles, so a sequence length that is not a
multiple of 32 leaves the tail padded. K=0 gives a **zero** score, not −inf, so
each padded position contributes `exp2(0 − m)` to the softmax denominator; V=0
means they add nothing to the accumulator. Three lines in `flm_attn_finish.cc`
subtract exactly that.

| seq | tiles | npad | max err | tol | |
|---|---|---|---|---|---|
| 512 | 16 | 0 | 7.24e-04 | 8.71e-04 | PASS |
| 2048 | 64 | 0 | 1.81e-04 | 4.63e-04 | PASS |
| **500** | 16 | **12** | 5.35e-04 | 8.86e-04 | PASS |
| **500, `--ignore-pad`** | 16 | 0 | **1.20e-03** | 8.86e-04 | **FAIL** |
| 512, `--ignore-pad` | 16 | 0 | 7.24e-04 | 8.71e-04 | PASS (nothing to correct) |

The last two rows are the point. A correction that did nothing would pass the
seq=500 row just as happily; `--ignore-pad` is kept in the harness so that stays
checkable. Tolerance is the exp2 NLF floor (3.54% mean / 5.86% max), not bf16.

### `npad` cannot have its own fifo — it rides inside Q

The obvious wiring gives npad an input fifo, and that is not a tight fit but a
compile error: `tile (0,3) requires 3 input/1 output DMA channels, but only 2
input/2 output available`. Attention already uses both inputs for Q and the KV
stream. So npad rides in the tail of the Q buffer as one f32 in two bf16 slots,
read through a cast (offset GQA*HEAD bf16 = 512 B, so 4-byte aligned by
construction).

f32 and not bf16 because **bf16 is exact on integers only to 256**, and the slot
carries counts to 2047 in the fused layer.

This is the third operand packed into another operand's buffer for the same
reason — the norm weight inside the activation (`flm_norm_prepare`), cs_q/cs_k
inside the broadcast (`flm_gemv_qkv`), now npad inside Q. On this device extra
operands are packed, never given a fifo. Worth treating as the default move
rather than a trick.

## 2026-07-31 — Task 6 complete: attention as a phase, 8 cores, one dispatch

`tools/npu/flm/attn_phase.py`. The attention arithmetic was already verified on
one core; this runs it at the phase's real shape — 8 cores, one KV group each,
in the fused layer's paired topology (operand fifo per pair split to two cores,
result fifo per pair joined).

8 KV heads at GQA=4 is exactly 8 cores covering all 32 query heads, so **no KV
broadcast and no cross-core softmax merge**: every core's cache is private.
Pairs 4–7 sit the phase out.

| seq | KV tiles | objects | npad | max err | tol | |
|---|---|---|---|---|---|---|
| 512 | 16 | 8 | 0 | 7.31e-04 | 1.08e-03 | PASS |
| 2048 | 64 | 32 | 0 | 3.62e-04 | 4.98e-04 | PASS |
| 500 | 16 | 8 | 12 | 5.51e-04 | 9.93e-04 | PASS |
| **480** | **15** | 8 | **32** | 6.22e-04 | 1.05e-03 | PASS |

The 480 row matters: 15 tiles is odd, so the last object carries one real tile
and one **entirely padded** one. The correction handles it because it counts
padded *positions*, not partial tiles. Tolerance is the exp2 NLF floor
throughout, not bf16.

**`DIM_KVPER` — two KV tiles per operand object, and the plan's +2.8% checks
out.** KV has to ride the same 20544 B operand object as every other phase or
the topology changes between phases and one dispatch per layer is illegal. A KV
tile is 2·TSEQ·HEAD bf16 = 8192 B, so one per object wastes 61%. TSEQ cannot
just be doubled — `static_assert(TSEQ == 32, "score vectors are one 32-lane
register")` — so `flm_attn_decode.cc` now loops over `DIM_KVPER` tiles per call
instead. The online softmax state already persists across calls, so folding
tiles into one call changes nothing arithmetically; it only decouples the object
size from TSEQ. Measured at S=2048:

| | KV bytes | vs a 38.0 MB layer |
|---|---|---|
| unpadded | 4.194 MB | — |
| **2 tiles/object (KVPER=2)** | **5.259 MB** | **+2.80%** |
| 1 tile/object | 10.52 MB | +16.6% |

+2.80% is the number the plan predicted for this task. `KVPER=1` recompiles
byte-identically and `attn_verify.py` returns the same 7.2378e-04 / 5.3524e-04
it did before the refactor, so the single-tile path is unchanged.

**Throughput, with a caveat.** S=2048 runs 5.26 MB in 238.7 us wall, 145.8 us
marginal. Against the 17.547 us/MB slope that would be 63% of ceiling, far below
the GEMV phases' 88–99% — but that slope was fitted at **16 cores** and this
phase runs on **8**, with half the operand fifos. The honest statement is that
the 8-core slope has not been measured, so this number is not yet interpretable;
`dispatch_bw_probe.py` at 8 cores would settle it.

**Not done here:** appending k′/v′ to the KV cache with a strided BD. That is
the seam between P1 and P2 rather than a property of either, so it belongs with
Task 7's single-dispatch wiring, where both phases exist in the same program.

## 2026-07-31 — the marginal slope is per core count; attention P2 is at 89%, not 63%

Closing the caveat left on Task 6. `gemv_bench.py --sweep-cores 4,8,16`, with
the marginal time taken as wall − 92.9 us:

| cores | MB | wall us | marginal us | **us/MB** | marginal GB/s |
|---|---|---|---|---|---|
| 4 | 9.5 | 577.9 | 485.0 | **51.05** | 19.6 |
| 8 | 19.1 | 562.5 | 469.6 | **24.59** | 40.7 |
| 16 | 38.1 | 772.9 | 680.0 | **17.85** | 56.0 |

**The 17.547 us/MB figure used throughout this log is the 16-core slope, not a
constant.** It is close to the fabric roof (56.0 of 56.5 GB/s), and it is not
what a phase running on fewer cores can reach: 4→8 nearly halves the slope
(factor 2.08, so the fabric is not the limit there), but 8→16 only improves it
1.38x, because by 16 cores the fabric is what is left.

So attention P2's 145.8 us marginal on 5.26 MB is:

- against the 16-core slope: 93.9 us ideal → **64%** — the number reported for
  Task 6, and it is wrong, because P2 runs on 8 cores.
- against the 8-core slope: 129.3 us ideal → **89%**.

89% puts it with the GEMV phases (88–99%), not below them. Task 6's throughput
caveat is resolved in the phase's favour, and no work is needed there.

**Rule for every future phase measurement:** divide marginal time by the slope
*at that phase's core count*. Only P2 runs on 8; everything else in the fused
layer is 16, where 17.85 is right.

## 2026-07-31 — Task 7 falsifier PASSES: a phase can read what the previous one drained

`tools/npu/flm/chain_probe.py`. The fused layer is five phases in one dispatch
and every phase but the first consumes the previous one's output — P1's q′ feeds
P2, P2's attention output feeds P3, P3's `h` feeds P4, P4's SwiGLU output feeds
P5. There is no host between them, so the only mechanism available is for a
phase's `drain` to land in a DDR buffer that a later phase's `fill` reads back,
as buffer descriptors in one command stream ordered by the in-dispatch barrier.

**That had never been tested, and all of Task 7 rested on it.** The probe chains
N phases of a doubling kernel through the same two fifos, phase 0 reading host
buffer A and every later phase reading B — the buffer the previous drain just
wrote. A working chain gives 2^N; a broken one localises where it stopped.

| phases | elements | realised gain | expected | max err | |
|---|---|---|---|---|---|
| 3 | 256 | **8.000x** | 8x | **0.0e+00** | PASS |
| 5 | 2048 | **32.000x** | 32x | 6.25e-02 | PASS |
| 5 | 4224 (broadcast-sized) | 31.969x | 32x | 6.06e-02 | PASS |

The residual error is bf16 rounding — the doubled values reach 32, where the
bf16 step is 0.25. The gain is what matters and it is exact.

So **Task 7 is buildable as §1.4 specifies**: inter-phase values ride DDR
round-trips inside the dispatch, one TaskGroup per phase so its BDs are freed
before the next opens, and `drain(wait=True)` is the barrier. No memtile
residency scheme is needed, and the phase schedule stands.

Worth having run before writing the five-phase harness rather than after — a
failure here would have invalidated §1.4's whole structure, not just its
wiring.

## 2026-07-31 — one dispatch per layer did NOT fit in program memory; one word fixed it

Before writing the five-phase harness, a check the plan never makes: **a core
tile has 16 KB of program memory**, and the fused layer needs every phase's code
resident at once, because a Worker is one program per core. §1.5 budgets L1
data down to the byte and says nothing about instructions.

It did not fit.

| | linked .text | cores 0–7 (+attention) | cores 8–15 |
|---|---|---|---|
| `inline` tile body (as built) | 14,272 | **16,896 = 103% of 16 KB** | 14,320 = 87% |
| `noinline` tile body | **10,208** | **12,832 = 78%** | 10,256 = 63% |

And that is *before* the worker's own control flow and the IRON glue, so 103%
understates it.

**Cause: `flm_q4_1_tile` was `inline` in a shared header**, so it was duplicated
into all six GEMV entry points — qkv, residual, gate, up_swiglu, acc, flush —
and the linker had nothing to fold: the sum of the objects and the linked size
are both 14,272 B, i.e. zero sharing. Marking it
`inline __attribute__((noinline))` makes it one out-of-line `linkonce_odr`
definition; 14,608 B of objects then link to 10,208 B. **4.4 KB recovered from
one word**, which is the difference between the fused layer fitting and not.

It costs nothing. The call happens once per weight tile and the loop inside runs
K/32 blocks, so it is one call per ~2000 MACs:

| cores | before GB/s | after GB/s |
|---|---|---|
| 4 | 16.5 | 16.7 |
| 8 | 33.9 | 33.2 |
| 16 | **49.3** | **49.0** |

Within run-to-run noise, numerics bit-identical at 5.96e-07, and
`gemv_verify`, `normgemv_verify` (both modes), `down_verify` and `qkv_verify`
all still pass.

**The general point, and it is the one worth keeping:** on this device an
`inline` helper in a shared header is not free the way it is on a CPU — it is
paid for once per entry point, out of 16 KB. With N kernels sharing a body, the
cost is N copies unless the definition is out-of-line. Any future shared kernel
code should be `noinline` by default and inlined only where a measurement says
it pays.

Worth having measured before building `layer_verify.py` rather than after: the
symptom would have been a link failure deep in a five-phase design, with the
cause four files away.

## 2026-07-31 — Task 7 in progress: FFN chain plumbing works, numerics do not

`tools/npu/flm/ffn_chain.py`, phases P4+P5 in one dispatch — 2 of the layer's 5
phases and 29.2 of its 38.0 MB. **It runs but does not pass**; recording what is
established and what is not, because the plumbing findings are reusable and the
failure is specific.

### Established

**`fill` and `drain` take `offset`, `sizes`, `strides` and `transfer_len`.**
This is the mechanism Task 7 needs and the plan never names: a phase's drain can
write into a *chosen slice* of the buffer a later phase's fill reads. Here P4's
per-pair drains land directly in the activation halves of P5's four broadcast
objects, whose aux halves the host pre-filled with the residual. No host round
trip and no extra buffer.

`transfer_len` alone is not enough — it emits `sizes = [0,0,0,0]` and the
lowering rejects it (`'aie.dma_bd' op Size 0 must be a positive integer`). Give
the full 4-D form: `sizes=[1,1,1,N], strides=[0,0,0,1]` for a contiguous run.

**Phases must share one operand fifo and one result fifo.** Giving P4 and P5
their own is not a tight fit but a compile error — `tile (0,3) requires 3
input/2 output DMA channels, but only 2 input/2 output available`, because the
broadcast already takes one of the two inputs. This is what §1.1 means by the
topology being identical in every phase, and it is enforced by the hardware
rather than by convention.

**Row assignment must match the join order, not the core.** A pair's result
join emits `[core0's NROWS][core1's NROWS]` per step, so assigning core *c* a
contiguous block of rows makes the pair's drain an interleaving of two distant
row ranges, and it cannot be written to one contiguous destination. The
assignment that works is `row = pair*rows_per_pair + tile*2*NROWS + core_j*NROWS`,
which makes each pair's drain a contiguous global run. Fixing this alone took
the error from 1.38e-01 to 4.05e-02 — a real improvement, and not enough.

Consequently `DIM_ACCN` for the chunked `down_proj` must cover a **pair's** row
span, not a core's: with the interleaved assignment a core's slots run to
`2·p5_tiles·NROWS`, and the old `p5_tiles·NROWS` aliases them.

### Not established — the numbers are wrong

| | max err | mean\|ref\| | rel p50 | rel max |
|---|---|---|---|---|
| P4 SwiGLU out | 4.05e-02 | 0.01132 | **10.3%** | **51%** |
| P5 x_out | 9.01e-02 | 0.04698 | 47.9% | — |

The AIE2P `exp2` floor is 3.54% mean / 5.86% max, so **this is a bug, not the
NLF**. The sorted multiset does not match either (3.03e-02), so it is not a pure
ordering error; 5505 of 8192 rows are exactly right and the wrong ones are
scattered rather than blocked.

Two candidates, not yet separated:
- the in-place RMSNorm prologue — this is the first harness to run
  `flm_norm_prepare` against a broadcast fifo **shared across phases**, where
  the object is acquired, normalised in place, released, and later refilled;
- the gate/up in-core stash under a shared operand fifo, where P4's alternating
  acquires now interleave with P5's stream on the same fifo.

The isolation that separates them is to run P4 alone with `flm_asum_prepare`
instead of `flm_norm_prepare` and compare against `ffn_alt.py`, which passes at
3.40e-04 with the same kernels and a private fifo. That is the next step.

## 2026-07-31 — the SwiGLU sigmoid was a constant 1/2: a silent miscompile that shipped

Chasing `ffn_chain.py`'s numerics found a bug in **two** kernels that had been
passing their harnesses for days.

```c
aie::vector<float, SLANES> e = aie::zeros<float, SLANES>();
for (int r = 0; r < NROWS; ++r)
  e[r] = -g[r] * LOG2E;            // <-- silently does nothing
const auto s = aie::exp2<bfloat16>(e);
```

`operator[]` on an `aie::vector` yields a **temporary**, so every write is
dropped. It compiles without a warning. `e` stays zero, `exp2(0)` is 1, the
sigmoid collapses to a constant `1/(1+1) = 1/2`, and the kernel computes
**`g*u/2`** for every row. Present in `flm_ffn_gate_up.cc` and
`flm_gemv_up_swiglu.cc` — every SwiGLU path there is.

**Why the harnesses passed.** `silu(g) = g/2 + g²/4 − …`, so `g*u/2` *is*
SwiGLU to first order, and the error is O(g²)·u. `ffn_fused.py` and
`ffn_alt.py` fed unnormalised activations (`randn * 0.05`), where |g| stays
small and the difference sits at 3.4e-04 — comfortably inside a tolerance set
for the exp2 NLF. It only became visible in `ffn_chain.py`, the first harness to
put a **RMSNorm-scaled** activation through SwiGLU.

**How it was found.** Not by inspection. The error was deterministic across
runs, uniform across every pair, core and tile (~33% of rows each), and the
exp2 argument range was only [−1.1, 1.1] — all of which ruled out races,
plumbing and the NLF. Printing actual values showed the device's sigmoid was
**0.4990–0.5008 for every row whatever `g` was**, and `g*u/2` matched the device
to four digits. The lesson is the one this log keeps re-learning: aggregate
error statistics say *that* something is wrong, and only concrete numbers say
*what*.

**Fix:** build the argument with a vector load rather than element-wise —
`aie::mul(aie::load_v<SLANES>(g), broadcast(-LOG2E))` — with the source array
declared `SLANES` wide and zero-initialised so the spare lanes contribute
`exp2(0) = 1` and are never read.

### It was also making things slower

Removing dead code should cost time. It gained it — the scalar loop the
compiler was partially keeping is worse code than the two vector ops that
replace it:

| | before (miscompiled) | after |
|---|---|---|
| `ffn_fused` single tile, 8 rows | 514.0 us, 41.1 GB/s | **496.2 us, 42.5 GB/s** |
| `ffn_alt` alternating, 16 rows | 526.6 us, marginal 433.7 | **467.6 us, marginal 374.7** |
| accuracy of both | 3.40e-04 | **8.76e-05** |

**So Task 4's falsification is itself falsified.** Alternating acquires at 16
rows is *faster* than the single fused tile — 467.6 against 496.2 us — and its
marginal 374.7 us is **98.5% of the 369.1 us its transfer size allows**. The
plan's original claim was right, the retraction was an artifact, and the
morning's entry is marked as such rather than deleted.

## 2026-07-31 — Task 7: the FFN half runs as two chained phases in one dispatch

`tools/npu/flm/ffn_chain.py` now **passes**. P4 (norm2 + gate + up + SwiGLU) and
P5 (down, 4 K-chunks, + residual), 16 cores, one dispatch, real layer-0 weights.

| | value |
|---|---|
| bytes | **31.56 MB** |
| wall | 670.7 us |
| marginal | **577.8 us** against a 563.3 us 16-core ideal — **97.5% of ceiling** |
| rate | 47.0 GB/s |
| P4 SwiGLU | max rel err **3.92%** on the significant values |
| P5 x_out | max err 2.95e-03 against max\|W_down·sw\| 0.106 — **2.77%** |

Both inside the AIE2P exp2 NLF floor (3.54% mean / 5.86% max).

**Gating a residual-added quantity.** A pointwise relative error on `x_out`
reads 15.8%, and that is an artifact: `x_out = W_down·sw + h` with `h` added
exactly, so where the two nearly cancel the ratio diverges while the absolute
error is unchanged. The honest gate scales the error by the term that carries
it, `max|W_down·sw|`. Similarly, a max error gated against 8% of the *mean* is
far tighter than the NLF can meet when the quantity's max/mean ratio is ~14 —
that alone failed this harness after its arithmetic was already correct.

Regression after the fix: `gemv_verify`, `normgemv_verify`, `down_verify`,
`qkv_verify`, `attn_verify --seq 500`, `attn_phase --seq 480`, `ffn_chain` — all
pass.

## 2026-07-31 — projection from measured phases: the fused layer reaches PARITY, the 16-layer unroll is what beats FLM

Four of the five phases are now measured, so the fused layer can be projected
from real numbers instead of the plan's estimates.

| phase | MB | marginal us | ideal | % of ceiling | source |
|---|---|---|---|---|---|
| P1 qkv+rope, 16c | 3.94 | 78.6 | 70.3 | 89% | `qkv_verify` |
| P2 attention, **8c** | 5.26 | 145.8 | 129.3 | 89% | `attn_phase`, S=2048 |
| P3 o_proj+residual, 16c | 2.63 | 46.9 | 46.9 | — | **estimated** at the 16c slope |
| P4+P5 FFN, 16c | 31.56 | 577.8 | 563.3 | 97% | `ffn_chain` |

`layer = 92.9 dispatch + 849.1 phases + 4 x 6.37 barriers`. With
`lm_head` at 164.2 MB → 3.02 ms:

| seq | KV MB/layer | P2 us | layer us | token ms | tok/s | vs FLM |
|---|---|---|---|---|---|---|
| 128 | 0.33 | 9.1 | 830.8 | 16.32 | 61.3 | **1.02x** |
| 512 | 1.31 | 36.3 | 858.1 | 16.75 | 59.7 | **1.00x** |
| 1024 | 2.63 | 72.7 | 894.4 | 17.33 | 57.7 | 0.96x |
| 2048 | 5.26 | 145.3 | 967.0 | 18.50 | 54.1 | 0.90x |
| 4096 | 10.52 | 290.6 | 1112.3 | 20.82 | 48.0 | 0.80x |

### What this says, and it changes the roadmap

**One dispatch per layer is not enough to beat FLM.** It reaches parity at short
context and falls behind as the KV cache grows — 0.90x at S=2048. Getting every
phase to the ceiling is worth only +0.15 ms (60.2 tok/s, 1.01x): the phases are
already at 89–97%, so there is almost nothing left there.

**The 16-layer unroll is the lever, and it is worth more than everything else
combined.** 17 dispatches → 2 removes 15 x 92.9 us = **1.39 ms**, taking S=512
from 59.7 to **65.1 tok/s = 1.09x FLM**. The plan calls it "a follow-on, not a
prerequisite" (§ header). By these numbers it is **the** prerequisite for the
project's goal, and Task 8 should be treated as such rather than as polish.

`lm_head` is 3.02 ms — **18% of a token at S=512** — and is untouched by any of
this. It is the single largest remaining item after the unroll.

### Caveats, stated because the numbers invite over-reading

- P3 is the one phase never measured; it is assumed to hit the 16-core slope
  exactly, which is optimistic. Every measured phase lands at 89–97%.
- FLM's 59.86 tok/s baseline has no recorded sequence length. If it was measured
  at short context, comparing it against the S=2048 row is unfair to us; if at
  long context, the reverse. The S=128–512 rows are the like-for-like guess.
- KV padding (KVPER=2, 20% waste) costs 2.1 MB/layer at S=4096. At long context
  that is worth revisiting, though 2 tiles is already the most that fit in the
  20544 B operand object.
- This assumes the fused layer's phases compose at their measured standalone
  rates. `ffn_chain` is evidence they do — P4+P5 chained hit 97%, the same as
  the parts — but P1→P2→P3 has not been chained yet.

## 2026-07-31 — Task 8 feasibility: 320 phases fit in one dispatch, so the unroll is not blocked

Before building the 16-layer unroll — which the measured projection says is the
lever that beats FLM — the two things that could stop it, checked rather than
assumed.

**Instruction stream.** `barrier_probe.py --cores 16 --sweep 80,160,320`:

| phases | us | us/phase |
|---|---|---|
| 80 | 560.8 | 7.0 |
| 160 | 1042.1 | 6.5 |
| **320** | **1981.7** | 6.2 |

`time_us = 91.0 + 5.91 * phases`, R² = 0.99996. The unroll needs **80** phases
(16 layers x 5); 320 run fine, so there is 4x headroom. The fixed term, 91.0 us,
independently reproduces the 92.9 us per-dispatch cost measured a different way.

**The barrier is 5.91 us, not 6.37.** The earlier figure came from a sweep
topping out at 80 phases; extending to 320 gives a much better-conditioned fit.
Nothing downstream changes materially — 4 barriers per layer is 23.6 us against
a ~858 us layer — but 5.91 is the number to quote.

**Host buffers.** The unroll needs ~17 BOs: 8 weight streams (one per pair,
each carrying all 16 layers = 76.3 MB), 1 broadcast, 8 results. The measured
ceiling with `full_elf=True` is **64**, so this is not close.

So Task 8 is buildable. The blocker candidates are eliminated; what remains is
the work.

### One inconsistency fixed while here

`barrier_probe.py` printed "fused layer … 64.1 tok/s = 1.07x FLM", computed at
the 57.0 GB/s fabric roof with no KV traffic. That is an **upper bound**, and it
now contradicts the measured projection (59.7 tok/s at S=512, from phases
running at 89–97% of ceiling with real KV). The probe now labels its numbers as
upper bounds and points at the measured projection, so the optimistic figure
cannot be quoted by accident.

## 2026-07-31 — the dispatch cost amortises: measured, not assumed

`ffn_chain.py --repeat N` runs the working P4+P5 pair N times in **one**
dispatch — the Task 8 unroll shape applied to something already verified. This
is the empirical backing for the projection's central claim, which until now
rested on arithmetic.

| N | MB | unrolled us | N separate dispatches | saved | GB/s |
|---|---|---|---|---|---|
| 1 | 31.56 | 687.0 | 687.0 | — | 45.9 |
| 2 | 63.11 | 1258.9 | 1374.0 | 115.1 | 50.1 |
| 4 | 126.22 | 2457.7 | 2748.0 | **290.3** | **51.4** |

`wall_us = 87.6 + 591.5 * repeats`, R² = 0.99987 (least squares over all three
points; a two-point estimate from N=1 and N=4 gives 96.8 + 590.2, and the
difference is the N=2 point sitting slightly below the line).

**The fixed term is 87.6 us**, which independently reproduces the 92.9 us
per-dispatch cost from a completely different experiment. Everything above it
scales linearly, so the dispatch really is paid once however many phases follow.

Throughput rises with N — 45.9 → 51.4 GB/s, i.e. **91% of the 56.5 GB/s fabric
roof** — because the fixed cost is being spread, not because the phases got
faster: marginal efficiency is flat at 94.8% (N=1) and 95.3% (N=4).

Extrapolated to 16 repeats: 9552 us unrolled against 10992 for 16 dispatches,
**saving 1.44 ms**. The projection assumed 15 x 92.9 = 1.39 ms for the full
5-phase layer; measured and assumed agree.

So Task 8's premise holds on measurement. The unroll is worth what the
projection said, and combined with the earlier feasibility result (320 phases
fit one dispatch, ~17 host buffers against a ceiling of 64) there is nothing
left to check before building it.

## 2026-07-31 — P3 emits bf16 (forced), and an in-core residual that does not work yet

Working toward the full 5-phase layer. Two results, one solid and one open.

### `h` is needed in five places, and a drain writes to one

Phase P3 produces `h`, and the fused layer needs it as P4's activation half *and*
as the aux half of each of P5's four down-chunk objects. A drain consumes its
data once, so no phase can write it to five destinations, and P3 would have to
emit it repeatedly.

The proposed fix is that it should not travel at all: P5's flush needs only the
residual for the rows **it** outputs, and P3 and P5 have the same shape (N=2048,
16 cores, 8 tiles each), so with a shared row assignment the core that needs a
residual is the core that just computed it. 512 B of core memory replaces the
copy and removes 16 KB per layer of broadcast traffic.

### Established: P3 must emit bf16

Not a preference — a routing constraint. With P3 emitting f32 and P5 bf16 they
cannot share a result fifo, and two typed result fifos need 16 shim outputs.
The router does not report congestion, it reports **`Unable to find a legal
routing`**, which reads like a placement bug and is a budget failure. One
result-object shape per phase is what makes the topology placeable, and bf16 is
the right type anyway: `h` is the residual stream, which every phase carries in
bf16.

`tools/npu/flm/resid_chain.py` runs P3 → P5 back to back in one dispatch,
skipping P4 (its SwiGLU output is host-supplied, since the residual path is what
is under test). With the residual routed the old way, through the broadcast aux
half:

| | max err |
|---|---|
| P3 `h` | **0.0000e+00** |
| P5 `x_out` | **0.0000e+00** |

Exact, once the reference rounds to bf16 at both points the device does — the
`gemv_reference_bf16` lesson again, and it cost two rounds here.

### RESOLVED — it does not. `iron.jit` does not hash `compile_flags`

> The section below is **wrong**, and kept because how it was wrong is the
> point. The stash works, exactly (0.0000e+00). It looked broken because
> `iron.jit` **does not include `compile_flags` in its cache key**: the two
> residual paths differ only by `-DRESID_FROM_STASH`, with byte-identical
> sources, so running `--aux-residual` first and the stash second silently
> reused the aux-compiled flush and reported *its* behaviour under the stash's
> name. Every elimination below is individually correct and the conclusion is
> still false, because they all assumed the binary matched the source.
>
> **What actually named it:** poisoning the aux half with −7.0 in stash mode.
> If the flush were reading the stash, that value is never touched; the output
> moved by exactly −7.0. One deliberate wrong value did what four correct
> eliminations could not.
>
> With a cleared cache, both paths are exact:
>
> | | P3 `h` | P5 `x_out` |
> |---|---|---|
> | in-core stash | **0.0000e+00** | **0.0000e+00** |
> | broadcast aux (control) | **0.0000e+00** | **0.0000e+00** |
>
> And the stash is **more accurate by construction**: it carries `h` in float,
> where the aux route round-trips it through the broadcast in bf16. The
> reference has to model whichever path is under test, which is why this only
> reached 0.0 once that was separated.
>
> **The rule, measured rather than guessed.** `iron.jit` keys on the design's
> **code object**, so what matters is not that the switch is a `compile_flag`
> but that it never appears in the design's *source text*. Two harnesses, both
> building their design by `exec`-ing an f-string:
>
> | how the switch reaches the kernel | cache entries for 2 variants |
> |---|---|
> | interpolated into the design source (`ffn_chain.py --host-norm`) | **2 — rebuilds** |
> | only in a flags list built outside it (`resid_chain.py`, before) | **1 — collides** |
>
> That also retroactively validates the `--host-norm` A/B used during the SwiGLU
> hunt, which had looked suspect for the same reason — it was rebuilding.
>
> Fixed by interpolating the value into the generated source rather than by
> clearing the cache: `resid_chain.py` now produces 2 cache entries and both
> paths pass exactly, back to back, with no workaround. (Watch the shadowing:
> `FLAGS = FLAGS + [...]` inside the design makes it local and raises
> `UnboundLocalError`.)
>
> Fourth instance of this trap family and the worst — the source-file case at
> least leaves stale-looking code, whereas here the flag is right there in the
> file being read.

### Superseded: the in-core stash reads as zero

`-DRESID_FROM_STASH=1` fails, and the diagnostic is unambiguous: `x_out` matches
`W_down·swiglu + 0` at **100% exact**, so `flm_gemv_flush` reads `g_resid` as
zero. Ruled out, each checked rather than assumed:

- **Not the row mapping.** P3 and P5 use the same assignment, and
  `(base + r) % DIM_RESN` covers 0..255 distinctly for both.
- **Not the symbol.** `llvm-nm` shows `g_resid` defined (B) in
  `flm_gemv_residual.o` and undefined (U) in `flm_gemv_flush.o` — the same shape
  as `g_acc_down`, which crosses the same TU boundary and **works** (chunked
  `down_proj` accumulates correctly across four calls).
- **Not dead-code elimination.** `llvm-objdump -r` shows 16 relocations to
  `g_resid` in the writing object, exactly as many as to `g_acc_down` in its.
- **Not P3 failing.** P3's own output is bit-exact in the same run.

So a global written by one kernel and read by another works for `g_acc_down` and
does not for `g_resid`, with no difference visible in the symbols, the
relocations, the row arithmetic, or the phase order. That is the whole of what is
known. The control path passes exactly and is what the layer will use until this
is understood; `resid_chain.py --aux-residual` keeps it checkable.

## 2026-07-31 — the KV append hits two DMA rules; §1.4's k′ path needs a decision

`tools/npu/flm/kv_append_probe.py`. Before writing the P1→P2→P3 harness, the one
primitive it needs: can a `drain` scatter a new k′ into the **channel-major** K
cache? Attention wants K as `[HEAD][TSEQ]` so scores accumulate across `d` with
no horizontal reduce, which makes appending one token a stride-32 scatter of 64
bf16.

**With a bf16 cache it is not expressible.** Two independent rules:

```
sizes:   'aie.dma_bd' op Transfer sizes must be multiples of 4 bytes.
         1 elements at 2 bytes each equal 2 bytes, which is not divisible by 4
offsets: 'aie.dma_bd' op Offset must be aligned to 4 byte boundary
```

A bf16 is 2 bytes, so **one value per destination is an illegal size** and **an
odd column is an illegal offset**. The narrowest legal channel-major bf16 write
covers two columns starting at an even one — measured working, ramp-checked
element by element — and a decode step produces one token.

v′ is unaffected: it is a contiguous 64-element write.

### Options, with the traffic each costs

| | K tile | tiles per 20544 B operand object | KV MB/layer at S=2048 |
|---|---|---|---|
| bf16 K (today) | 4096 B | **2** | **5.26** |
| f32 K | 8192 B | 1 | **10.52** |

- **f32 K — proven.** One element is 4 bytes, so the single-token scatter is
  legal at every position including odd ones; the probe passes at t=0..3 with
  the untouched columns still exactly zero (which attention needs, since its
  `npad` correction assumes padded positions hold K=0). It **doubles** the KV
  traffic: the K tile no longer shares an operand object with V, so padding goes
  from 20% to 40% and S=2048 goes 5.26 → 10.52 MB/layer, +14% of a 38 MB layer.
  On the measured projection that costs ~2.3 ms/token, roughly 54 → 48 tok/s.
- **Paired append — cheapest, and untested.** Keep bf16 and write two columns at
  an even offset: at even `t` emit `(k′_t, 0)`, at odd `t` emit `(k′_{t-1},
  k′_t)` at offset `t−1`. Costs nothing in traffic and keeps K=0 padding for
  free at even steps. It requires the emitting core to still hold `k′_{t-1}`
  **from the previous dispatch**, i.e. that core-static data survives between
  dispatches while the xclbin stays loaded. That is plausible and unverified,
  and it is the next thing to measure.
- **Position-major K** — makes the append contiguous and destroys the property
  the attention kernel is built on (`aie::load_v<TSEQ>(kt + d*TSEQ)` needs 32
  consecutive positions for one channel). Not pursued.
- **Transform in a core or memtile** — what FLM does, per
  `flm-attn-dataflow.md`. Not investigated here.

The probe is left checking the **f32** path, since that is the one proven to
work; the bf16 constraints are recorded in its docstring so the next attempt
does not rediscover them.

## 2026-07-31 — core `.bss` PERSISTS between dispatches: the k′ append stays bf16

`tools/npu/flm/static_persist_probe.py`. The previous entry left the k′ append
choosing between a proven f32 K cache (double the KV traffic) and a free bf16
paired append that assumed a core could carry `k′_{t-1}` from one dispatch to
the next. Measured: **it can.**

A kernel incrementing a file-scope `float[]` and emitting it, over six separate
dispatches on one loaded design:

| | values |
|---|---|
| control, stateless kernel writing 7.0 | 7, 7, 7, 7, 7, 7 |
| **counter** | **1, 2, 3, 4, 5, 6** |

So core `.bss` survives between dispatches while the xclbin stays loaded, and
the paired append is available:

    even t:  emit (k′_t, 0)               at offset t
    odd  t:  emit (k′_{t-1}, k′_t)        at offset t-1

Both writes are 2 elements at an even offset, which is the narrowest legal
channel-major bf16 scatter. **K stays bf16 at 5.26 MB/layer** (S=2048) instead
of f32's 10.52, worth roughly 6 tok/s on the measured projection. The zero
partner at even `t` is also exactly what attention's `npad` correction wants
from a padded position.

### Two false answers before the true one, both from the harness

The probe reported "does not persist" twice, and neither reading was about
persistence:

1. **Default fifo depth 2** gave `0, 2, 0, 4, 0, 6` — the drain alternated
   buffers and read the one the core had not just written. The even values were
   real, which is what made it look like partial persistence.
2. **`stack_size=4096`** gave all zeros even at depth 1. The disassembly shows
   the frame is `paddxm [sp], #0x1c40` — **7232 bytes** for a 32-element float
   loop, because the compiler spills whole vector register files. The overflow
   is silent and corrupts the result, exactly as this log's earlier entry on the
   1024 B IRON default says. This probe needed 16384. (I first wrote "4096 is
   not a safe default either" — see the audit in the next entry, which shows it
   is safe for every real kernel with 3.8x margin, and that the probe was the
   outlier.)

What separated harness from hardware was a **stateless control** — the same
kernel writing a constant. It passed while the counter read zero, which located
the fault in the global rather than the plumbing, and made the stack frame worth
looking at. Without it the probe would have shipped "core state does not
persist" and the design would have taken the f32 path and the 6 tok/s.

## 2026-07-31 — stack frames audited: every kernel fits, and what actually blows the stack

The 7232 B frame that broke the persistence probe raised an obvious question —
**every harness passes `stack_size=4096`**, so is anything else silently
overflowing? `tools/npu/flm/stack_audit.py` disassembles each kernel at the
shapes the harnesses build and reports the frame.

Nothing is. The worst is `flm_attn_decode` at **1088 B, 27% of 4096**, a 3.8x
margin; most are 64–320 B and four are zero. So the previous entry's "4096 is
not a safe default either" was wrong, and the probe was the outlier rather than
the warning.

**What actually causes it** is not size but how far the backend unrolls. A fully
unrolled loop doing a scalar `float -> bfloat16` conversion spills the whole
accumulator register file:

| trips | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| frame B | 1024 | 3136 | 5184 | **7232** | **0** | **0** |

~2 KB per 8 iterations while it unrolls, then **0** once it gives up and emits a
real loop — so the danger zone is a *middle* trip count, and a bigger loop can
be cheaper than a smaller one. The identical loop writing `float` instead of
`bfloat16` costs **64 B**, which is what isolates the conversion as the trigger.

Every kernel in `kernels/npu/` escapes it by indexing with a dynamic base
(`slot + r`, `base + r`, `off + r`) rather than the loop variable, which the
backend will not unroll that way. That is a property of how they happen to be
written and not a guarantee, which is the reason the audit is a script rather
than a paragraph: it is cheap to re-run when a kernel changes shape.

## 2026-07-31 — one result fifo can feed three destinations: the last P1→P2 primitive

`tools/npu/flm/qkv_route_probe.py`. Phase P1 emits 48 heads down **one** result
fifo — 32 q, 8 k, 8 v — bound for three unrelated places: q′ contiguous into P2's
query block, k′ a stride-TSEQ scatter into the channel-major K cache, v′
contiguous into the position-major V cache. Every harness so far drains a fifo
exactly once, so successive *partial* drains with different patterns had never
been tried. If they did not work, P1 would need a fifo per destination, and a
core tile has 2 output DMA channels.

They work. All three destinations land exact, with the untouched K columns still
zero (which attention's `npad` correction requires).

**A fifo may have only one shim endpoint.** Asking for three
`cons(tile=AnyShimTile)` is `redefinition of symbol named 'heads_shim_alloc'`.
The working form is one handle drained three times — the split is in the
*drains*, not the endpoints.

**A K tile consumes two objects, not one.** The paired write is 2 elements per
destination, so filling HEAD destinations needs 2·HEAD source values; channels
0–31 come from the first object and 32–63 from the second, because the drain
walks the source linearly while striding the destination. The probe initially
reported k tags 5,7 and v tags 9,10 where 5,6 and 7,8 were expected — those
numbers were **right**, and the expectation was wrong. Worth stating because the
same off-by-one will appear in the real emit kernel's pack order.

Two harness faults were fixed on the way, both of which would have muddied the
answer:

- **tags of `100h + i` are not exact in bf16**, which is only exact on integers
  to 256. The routing error and the rounding error were indistinguishable until
  the tag became a small per-head constant.
- `stack_size` is 16384 here, not the usual 4096, because this kernel's emit
  loop is exactly the shape that spills the accumulator file.

That closes the primitives for Task 7's remaining harness: the inter-phase
chain, the residual stash, the KV append geometry, core-static persistence, and
now the three-way split are all measured.

## 2026-07-31 — `flm_kv_emit.cc` written; its harness does not work yet

The kernel that closes the P1→P2 seam. It supplies the two values the narrowest
legal channel-major write needs:

    even t:  (k′_t, 0)          -> column pair (t, t+1)
    odd  t:  (k′_{t-1}, k′_t)   -> column pair (t-1, t)

so every token lands in its own column, each pair is written twice, and the zero
at even `t` is what attention's `npad` correction wants from a padded position.
`g_kprev` carries the previous token across the dispatch boundary, which is what
keeps K in bf16 instead of the f32 cache that costs ~6 tok/s. It compiles at a
**64 B frame** and reads the position from the tile trailer's second f32, for
which `flm_q4_1_tile.h` gains `tile_flags()`.

**`tools/npu/flm/kv_emit_verify.py` does not work.** It produces an all-zero
cache with no error. Ruled out by measurement:

- not the stride pattern — a plain contiguous drain is equally empty;
- not the destination BO size — a BO sized to exactly one object is equally
  empty;
- not the trailer offset — `tile_flags()` offsets by `TILE_BYTES`, so passing a
  bare 64 B trailer read 20 KB out of bounds and made the even/odd branch
  random. Fixed by passing a real tile-sized buffer; the numbers did not move,
  because the buffer was zero either way. **Worth keeping as a trap in its own
  right**: any kernel taking a `wtile` needs a full tile, not just its trailer.

So the core or its fifos are not producing, upstream of anything about the
append. Both mechanisms this harness combines are separately proven —
`kv_append_probe.py` for the paired strided write and `static_persist_probe.py`
for the carry — so what is unproven is only their combination in this design.

The next attempt should **extend `qkv_route_probe.py`**, which produces and
routes correctly, rather than write a third design from scratch. The structural
difference is that this one feeds the core from two input fifos.

### Later the same day — the k′ harness fault is pinned to input-fifo delivery

Bisected up from `qkv_route_probe.py` (which works) rather than down from the
broken design, and split the failure with two constant-injection tests.

**Works** — `KV_SEEDCONST=1` makes the seed write a constant instead of reading
its input, and the cache fills correctly (`K[:,0] = 5.0`). That clears, in one
run: `flm_kv_emit` itself, the cross-TU `g_stage` handoff, and the paired
strided drain at a non-zero offset.

**Fails** — with the seed reading its input fifo, the cache stays zero **even
when the input is all 5s** (`KV_CONSTHEAD=1`). So the fault is exactly one
thing: data filled into this harness's input fifo does not reach the kernel.

Eliminated along the way, each by measurement:

| hypothesis | result |
|---|---|
| two input fifos confuse the design | no — 0, 1 and 2 input fifos all pass |
| the core body needs a `range_` loop | no — a plain traced Python loop passes |
| output fifo depth 1 vs 2 | no — both behave the same |
| destination BO larger than one object | no — a matched BO is equally empty |
| tile trailer offset | a real bug (`tile_flags()` offsets by `TILE_BYTES`, so a bare 64 B trailer reads 20 KB out of bounds) but not this one |

**The bisect's own limit, worth stating:** the probes I bisected with had
kernels that *ignored* their inputs. They proved the design runs and produces,
not that input data arrives — which is precisely the gap that turned out to
matter. A bisect that varies structure while holding the payload trivial cannot
see a payload fault, and I nearly concluded "input fifos are fine" from it.
