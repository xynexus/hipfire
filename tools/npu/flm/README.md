# FLM NPU inspection tools

Tools for reverse engineering FastFlowLM's AIE2P kernels
(`/opt/fastflowlm/share/flm/xclbins/<Model>-NPU2/`) on Strix Halo.

Findings: `docs/npu/flm-layer-dataflow.md`, `docs/npu/flm-attn-dataflow.md`.
Working log including failures: `docs/npu/flm-refe-log.md`. Plan:
`~/flm-re-fe-mutate-goal.md`.

## Static analysis — what is in the xclbin

```bash
python3 cdo.py       <file.xclbin|.pdi> [dump_dir]   # per-core program memory
python3 aiedis.py    <dump_dir>/core_C_R.bin         # AIE2P disassembly
python3 cdo_dma.py   <file.xclbin> [--tile C,R]      # DMA buffer descriptors
python3 cdo_dma.py   <file.xclbin> --graph           # stream-switch connectivity
python3 txn_scan.py  <file.so|.txn> [--dump DIR]     # embedded transaction binaries
```

`cdo.py` finds the CDO blob by magic, so it works on an `.xclbin` directly — no
`xclbinutil` step. `cdo_dma.py` reads its register and port definitions from
aie-rt's own headers at run time (`AIE_RT_PARAMS`, `AIE_RT_REGINIT`), so the
decode cannot drift from the vendor's.

## Dynamic analysis — what the runtime does

```bash
# count NPU command submissions (and trace their shape / argument buffers)
gcc -shared -fPIC -O2 -o npu_ioctl_count.so npu_ioctl_count.c -ldl \
    -I ~/xdna-driver/include/uapi
NPU_COUNT_OUT=c.txt LD_PRELOAD=./npu_ioctl_count.so flm run llama3.2:1b

python3 txn_memscan.py --launch "flm run llama3.2:1b" --dump captured/
```

## Measurement

```bash
python3 macbench.py                        # static MAC issue rates (bundle counts)
python3 macbench_hw.py                     # the same modes on hardware
python3 manybuf_probe.py                   # max host buffers per dispatch
python3 txn_check.py [file.txn ...]        # cross-check txn2mlir against a raw decode
```

`macbench_hw.py` measures each MAC mode three ways — `resident` (issue rate),
`seq` (pointer post-increment from L1), `strided` (masked-index control) — plus
`split` for the block-float types. `macbench.py` is the compile-time counterpart;
the two agree to within 4%.

## Traps these encode

Each is a thing that cost real time; module docstrings have the detail.

- **A stale `mlir_aie` wheel in the venv shadows the build tree.** Every tool that
  imports `aie` needs `PYTHONPATH=<mlir-aie>/build/python`. Symptom is anything
  from a bad decode to `ImportError: cannot import name 'CompileTime'`. Check with
  `python3 -c "import aie; print(aie.__path__)"`.
- **CDO command word** is `cmd | len<<16` with an **8-bit** length, `0xFF`
  escaping to a following 32-bit length. `0x102`/`0x103` are 32-bit addressed,
  `0x105` is 64-bit — the mixed widths are the trap.
- **Transaction binaries have no magic number.** `txn_scan.py` finds them by
  structural validation: the op walk must land exactly on `txn_size` *and*
  produce exactly `num_ops`. Two independent conditions, so false positives are
  very unlikely.
- **v1.0 `DDR_PATCH` is aie-rt's 24-byte `patch_op_opt_t`**, not the 44-byte
  `patch_op_t`. `regaddr` is 32-bit and there is no `action` field; reading the
  44-byte offsets walks into the following operation.
- **Core-tile BD `BASE_ADDRESS`/`BUFFER_LENGTH` are in 32-bit words**, not bytes
  or 32-byte units. The check that fixes it is that the decoded addresses
  reproduce a core's operand pointers exactly.
- **Disassembly** wraps raw bytes via `.incbin` into an aie2p object because this
  `llvm-objdump` has no `-b binary`, and resumes past both undecodable words and
  disassembler aborts. Without the resume, cores decode to 78 instructions
  instead of 1568.
- **`aiedis.py` is not named `dis.py`** because Python puts a script's own
  directory first on `sys.path`, so a `dis.py` here shadows the **standard
  library** `dis` module. `inspect` imports it and `argparse` imports `inspect`,
  so any other script in this directory dies with
  `AttributeError: module 'dis' has no attribute 'COMPILER_FLAG_NAMES'`.
- Compile for AIE2P with `--target=aie2p-none-unknown-elf` — the bare `aie2p`
  triple fails on a missing libc++ `__config_site`.
