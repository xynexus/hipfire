# DFlash Phase 0 — working brief

Self-contained execution brief for Phase 0 of
`docs/plans/2026-07-19-hybrid-gpu-npu-cpu-spec-decode.md`. Carries the measured
state so a fresh session does not re-derive it, and the traps that cost the most
on the previous pass.

Full measurement detail: `docs/npu/dflash-native-driver-plan.md`.

**Goal.** Get the DFlash NPU draft block inside the GPU verify budget, measured
end-to-end — not projected. Then re-run Phase F to decide the weight format.

Repo `/home/sadara/hipfire`, branch `chaingun`, machine nix1 (gfx1103 GPU +
npu1/aie2 NPU, 4 columns).

## Measured state — do not re-derive

NPU DFlash block wall: **726 ms** (native driver,
`crates/hipfire-xdna/examples/dflash_body_native.rs`). Verify budgets: 9B 57 ms,
27B 155 ms, 31B 345 ms. Attribution, each term measured with the kernel pinned:

| term | original | NOW (measured) |
|---|---|---|
| GEMM (weight-bandwidth-bound) | 317 ms | **139.5 ms** (98bbce9b6, r14 W4A8) |
| attention | 236 ms | **61.4 ms** (326971468, 4-core, 3.83×) |
| host glue (quant/bf16/packing) | 143 ms | **67.9 ms** |
| primitives (norm/rope/swiglu) | 24 ms | 22.3 ms |
| **warm block wall** | **726 ms** | **291.1 ms** |

**291.1 ms FITS the 31B budget (345 ms) with margin. Still 1.88× over 27B
(155 ms)**, which is Phase 0's stated gate — so the gate is not met, but the
architecture is viable end-to-end on a 31B-class target today.

**Items 1 and 2 are DONE.** Remaining budget gap to 27B is ~136 ms, and the two
largest remaining terms are GEMM (139.5) and host glue (67.9).

**Of the GEMM's 139.5 ms, ~62 ms is hardware-context contention, not bandwidth** —
npu1 admits 6 contexts, the r14 array pins one, leaving 4 LRU slots for 8 primitive
kernels → 36 context misses/block. Isolated probes hit 0.80–0.84 ms/dispatch vs
1.24–3.30 ms in-body. ~60 ms is the genuine floor (600 MiB packed W ÷ 10.4 GB/s;
`M_TILE=16` makes the rows=32 GEMMs stream weights twice). **Context contention is
the single largest recoverable term left.**

8-core attention is BLOCKED by the shim DMA budget (~2 MM2S/column; each worker
needs its own Q and KV stream) and would buy only ~30 ms — not the next move.

**The "~32–42 ms" projection this brief originally carried was WRONG** — it came
from an aggregate bandwidth figure (15.14 GB/s) mistaken for the weight path
(actually 10.0–10.7). Of the measured 123.7 ms, ~60 ms is the genuine bandwidth
floor (600 MiB packed W ÷ 10.4 GB/s — M_TILE=16 makes the rows=32 GEMMs stream
weights twice) and ~62 ms is **hardware-context contention**: npu1 admits 6 hw
contexts, the r14 array pins one, leaving 4 LRU slots for 8 primitive kernels →
36 context misses/block. Isolated probes hit 0.80–0.84 ms/dispatch; in-body the
same dispatches cost 1.24–3.30 ms. That contention is a NEW actionable term.

**Target 27B/31B, NOT 9B.** The verify budget scales with target size; the draft
cost does not. A 9B prototype measures a permanently-negative result.

## Tasks, in order

1. ~~**Wire the multi-core W4A8 GEMM into the body.**~~ **DONE** (98bbce9b6). Needs (a) an **oq4 DFlash
   sidecar** — only OQ8 exists. `dflash_convert` has `--oq4.<bits>`; use **pure W4
   (qt=33/34)**, since qt=36 mixed expands to dense int8 at upload and buys no
   bandwidth. (b) a **host-side stripe packer** matching r14's layout. Kernel
   artifacts already built and validated: `~/.hipfire/npu/r14_1x2x128_nb128`
   (recommended) and `r14_1x4x64_nb128`.
   **Trap:** `NpuGemmMp::load_cached` REJECTS an `r14_…` basename — its `_r{N}`
   guard matches the `r14` token itself. Use `load_with_tile`.
2. ~~**Multi-core the attention kernel.**~~ **DONE** (326971468, 4-core).
3. **Attack host glue** (67.9 ms) **and the ~62 ms of GEMM context contention** — the two remaining levers to reach the 27B budget.
4. **Re-measure the block wall after each.** Report cold and warm separately.
5. **Then re-run Phase F** — acceptance rate across f16 / oq8 / oq4.25+ / mq4
   drafters. Valid for the first time (see traps).

## Gates

**Parity.** Full-body cosine > 0.99 vs the f16 golden AND vs the int4/bf16
precision reference. Do NOT loosen. The F32 sidecar is a bug repro only —
`gemm_f32_batched` has a batch>1 transpose bug and a pure-F32 drafter scores τ=0.

**Losslessness (must not regress).** At temp 0, all drafters commit
BYTE-IDENTICAL tokens to `--ar-baseline` while differing in accepted counts:

```
T=~/.hipfire/models/qwen3.5-9b-mq4.hfq
./target/release/examples/dflash_spec_demo --target $T --draft <D> \
  --prompt "Explain how a four-stroke engine works." --max 96
```

md5 over `| tail -20` must be `02e621bd56b5` for AR and all four drafters.

## Traps that cost the most last time

- **The verify forward WAS nondeterministic**; single-run md5 comparison was
  measuring noise and produced 4+ wrong eliminations. Deterministic now
  (`6ca303af8`), but ALWAYS use ≥3 repeats and assert cross-run identity first.
- **`./tests/coherence-gate-dflash.sh` compares single runs** and structurally
  CANNOT catch that bug class. Not sufficient on its own.
- **A hypothesis must explain the PRIMARY SYMPTOM.** "Baked scalar" was proposed
  for a variance bug; a baked scalar is stale but DETERMINISTIC. Check first.
- **Check the claimed-CORRECT side of a comparison**, not just the claimed-broken
  side. One filed bug asserted the serial path "honors" a value it does not.
- **SNR is the WRONG gate for a drafter weight format.** Spec decode is lossless,
  so quality costs ACCEPTANCE RATE, not correctness. `oq4.25+` fails on SNR
  (cos 0.9606 / 11.04 dB vs int8's 33.18) — that may not matter. **Phase F
  decides.**

## Dead ends — do not repeat

- NPU weight path saturates **~10 GB/s per routing topology**, ~13 GB/s across two
  orthogonal routes. **EIGHT knobs measured null**: channel count, consumer count,
  buffer depth, compute load, activation traffic, shape, burst length,
  buffer-region layout. **Weight BYTES are the only lever left.**
- `opus`/`NpuOpusExecutor` is aie2p/npu2-only — unusable on npu1.
- Mixed-precision overlays are second-order for quality: n_out 3→63
  (4.25→8.00 b/w) buys **1 dB**; FWHT rotation buys **0.2 dB**. int4 loses ~22 dB
  and that is textbook (5.5 dB/bit), not a bug.
- **DeltaNet state must NEVER be Q8** (policy, `51e1ac078`). It is FP32 now.

## Guardrails

GPU/NPU work holds the lock: `./target/release/hipfire lock acquire <name>` /
`release`. rustup cargo (`export PATH="$HOME/.cargo/bin:$PATH"`). `graphify
query` before grepping repo source (hook-enforced) — include this in subagent
prompts. Do NOT touch `.agents/scheduled_tasks.lock`, `third_party/`, or
`benchmarks/npu_gemm_tuning/` except to add a new round dir. Commit validated
work with evidence in the message; report failures plainly with the numbers.
