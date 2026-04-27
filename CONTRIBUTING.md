# Contributing to hipfire

hipfire is alpha. Real-world testing on cards we don't have, kernel work
on archs we don't ship for, bug reports with full reproduction, and new
model architecture support — all welcome.

## Two ways to help, no Rust required

Both paths below use only the installer-provided binaries (the
`hipfire` wrapper, daemon, and quantizer dropped into `~/.hipfire/bin/`
by `scripts/install.sh`). No `cargo`, no ROCm SDK, no source build.

### 1. Run the bench matrix on your GPU

If you have an RDNA card the maintainer doesn't, the highest-leverage
thing is running the standard tester workflow and posting numbers.

```bash
hipfire diag                            # ROCm + arch detection sanity check
hipfire pull qwen3.5:0.8b               # ~0.5 GB; fits any RDNA card
hipfire pull qwen3.5:9b                 # ~5.3 GB; needs 6 GB+ VRAM
hipfire bench qwen3.5:0.8b --runs 5     # decode + prefill tok/s over 5 runs
hipfire bench qwen3.5:9b   --runs 5
```

For 16 GB+ cards, also pull and bench `qwen3.5:27b`. For mixed-arch
or non-Linux setups, see `hipfire diag --help` for environment-
specific guidance.

Open an issue titled `Benchmarks: <your GPU>` and paste the `diag`
output + each `bench` block. Results land in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). The `hipfire-tester` skill
in `.skills/hipfire-tester/` walks an AI agent through this end-to-
end if you want help.

### 2. Diagnose and report a bug

```
hipfire diag                            # capture everything first
```

Open an issue with: GPU + ROCm version, exact command, full error
output (not just the last line), and the diag output. The
`hipfire-autoheal` skill (in `.skills/hipfire-autoheal/`) is a
fix-catalog walkthrough that an agent can apply on your behalf for
common runtime issues; if it doesn't resolve cleanly, that's exactly
the case we want filed.

---

## Developer workflow

### Setup

```bash
git clone https://github.com/Kaden-Schutt/hipfire
cd hipfire
cargo build --release --features deltanet --example daemon -p engine
cargo build --release --features deltanet --example test_kernels -p engine
cargo build --release -p hipfire-quantize
```

Requires Rust 1.75+ and ROCm 6+ (the dev workflow needs `hipcc` for
kernel JIT). Pre-compiled kernel blobs ship for gfx1010 / gfx1030 /
gfx1100 / gfx1200; other arches JIT-compile on first load.

### GPU kernel correctness check

```bash
./target/release/examples/test_kernels      # ~30s, no model needed
```

Validates every dispatched kernel against a CPU reference on the
detected arch. This is the load-bearing correctness gate for any
arch port; if it fails on your hardware we want to hear about it
(see issue template / autoheal skill).

### The three gates

Any change to kernels, dispatch, fusion, rotation, rmsnorm, sampling,
the spec-decode path, or the forward pass MUST pass the relevant gates
before commit. The pre-commit hook runs them automatically when staged
files match the hotspot regex.

```bash
./scripts/coherence-gate.sh             # AR coherence (panic / zero-tokens / timeout = hard fail)
./scripts/coherence-gate-dflash.sh      # spec-decode token-attractor detection
./scripts/speed-gate.sh --fast          # 4B prefill+decode regression vs tests/speed-baselines/<arch>.txt
```

**Don't bypass with `--no-verify`.** A regression the gate catches is
information. Authorized exceptions need explicit written sign-off from
the maintainer for that specific change. Read
[docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md)
before claiming any perf win — within-session A/B noise is ±10–15% on
gfx1100, and the bench harness has a stale-binary trap that's bitten
us before.

### New kernel files

```bash
# Kernel source: kernels/src/<name>.hip
# Per-arch overrides: kernels/src/<name>.gfx12.hip       (family tag)
#                     kernels/src/<name>.gfx1100.hip     (chip tag)

# Register in: crates/rdna-compute/src/kernels.rs
# Wire dispatch in: crates/rdna-compute/src/dispatch.rs

# After editing any .hip file, regenerate hashes for the pre-compiled
# blob loader (otherwise the runtime falls back to JIT):
./scripts/write-kernel-hashes.sh

# Compile-check across the supported arch matrix:
./scripts/compile-kernels.sh gfx1010 gfx1030 gfx1100 gfx1200 gfx1201
```

Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Quantization design (MQ4 / HF4 / asym KV math):
[docs/QUANTIZATION.md](docs/QUANTIZATION.md).

### Porting to a new GPU arch

The `.skills/hipfire-arch-port/` directory is the canonical entry
point — playbook, WMMA matrix, validation procedure, contributor
onboarding. Don't write code before reading it; six-week silent-
corruption bugs from getting WMMA C-mappings wrong are how every
prior arch port has gone sideways.

Recent reference: PR #56 (RobinVanCauter, gfx1201 / 9070 XT) walked
the skill end-to-end and shipped a full validated 5-kernel port +
6 channel tests in one round. That's the bar.

### New model architectures

Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)'s "Two model
paths" section — `crates/engine/src/llama.rs` is the template for
dense Llama-style models, `qwen35.rs` is the Qwen 3.5 hybrid path
with DeltaNet linear attention. Add the architecture string to
`from_gguf` / `from_hfq` and patch the tensor-shape divergences.

For a new GGUF dequant type (Q5_K, IQ-quants, etc.), port from
llama.cpp's `ggml-quants.c` into
`crates/hipfire-quantize/src/gguf_input.rs`. ~150 lines per format.

### Branch naming

| Type | Pattern |
|---|---|
| Feature | `feature/<short-name>` |
| Bug fix | `fix/<short-name>` |
| Arch port | `port/<arch>-<kernel>` |
| Benchmark contribution | `bench/<gpu-name>` |

### PR template

Concise description, before/after numbers if perf-sensitive, mention
which gates passed. One logical change per PR. Run `cargo fmt` and
`cargo clippy` before submit; CI enforces both.

For perf claims: include the binary md5
(`md5sum target/release/examples/bench_qwen35_mq4`) and the prompt
md5 if the bench is prompt-dependent. Without these, the result is
unreproducible — see
[docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).

### Code style

- `cargo fmt` — required, CI-enforced.
- `cargo clippy` — no new warnings.
- **No Python in the inference hot path.** Python is fine for
  tooling, benchmarks, and offline analysis; never in the engine.
- Comment HIP kernel parameters: VGPR budget, wave occupancy, LDS
  usage, K-tile depth — anything a reader needs to understand the
  perf shape without inspecting `--save-temps` output.

---

## Skills (agent-driven workflows)

| Skill | When to use |
|---|---|
| [`hipfire-tester`](.skills/hipfire-tester/) | First-time bringup + bench submission on a new GPU. |
| [`hipfire-diag`](.skills/hipfire-diag/) | "Hipfire isn't working — what's wrong?" Captures GPU/HIP/kernel state. |
| [`hipfire-autoheal`](.skills/hipfire-autoheal/) | Runtime issue triage: daemon hangs, JIT failures, port conflicts, OOM. |
| [`hipfire-arch-port`](.skills/hipfire-arch-port/) | Porting hipfire to a new GPU arch (gfx12, gfx1152, gfx94x, …). |

Each skill has a `SKILL.md` (or `skill.json` + sibling `.md` files)
that any agent framework can load. Designed for Claude Code / Cursor /
Codex but framework-agnostic.

---

## Where the active asks are

These three are real open questions where contributor input would
land cleanly:

- **Issue #57** — gfx12 (RDNA4) WMMA dispatch wiring + perf
  measurement vs the dot2 fallback. Needs R9700 / 9070 XT hardware.
  PR #56 landed the kernels; #57 measures and flips the dispatch.
- **Issue #58** — multi-GPU support roadmap. Pipeline-parallel first
  cut design open for discussion. Mostly weighing in vs writing code,
  unless you have a multi-GPU rig and want to prototype the
  device-aware tensor allocator.
- **Issue #50** — gfx1152 (Strix Halo APU) crash — awaiting a
  reproducer + dmesg from the original reporter. If you have one,
  comment there.

For everything else, `hipfire list -r` + a benchmark issue on any
arch we don't have local numbers for is welcome.

---

## License

MIT. Files contributed under the same terms.
