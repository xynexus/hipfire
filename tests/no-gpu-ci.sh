#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${HIPFIRE_PYTHON:-python3}"

echo "== Rust check =="
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-D warnings" cargo check --workspace --examples

echo "== Eval harness check =="
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-D warnings" cargo check -p hipfire-eval

echo "== Rust no-GPU unit tests =="
cargo test -p hipfire-rdna --lib
cargo test -p hipfire-arch-qwen35 --lib moe_prefill
cargo test -p hipfire-eval --lib
cargo test -p hipfire-quantize xxh64_provenance_tests
cargo test -p hipfire-quantize fixture
cargo test -p hipfire-runtime quant_catalog_matches_derived_gemv_routes
# hipfire-daemon is BIN-ONLY (no src/lib.rs, only [[bin]]), so ci.yml's
# `cargo test --lib --workspace` selects zero targets from it — it does not fail,
# it silently tests nothing. `cargo check --workspace --examples` above does not
# type-check #[cfg(test)] code either. Its unit tests therefore ran in NO gate:
# the executor-v2 stream tests were added with a file-scope #![allow(dead_code)]
# justified by "the unit tests exercise every item", while nothing in CI ran them.
# --bin is required; -p alone inherits the same empty --lib selection.
cargo test -p hipfire-daemon --bin hipfire-daemon
cargo test -p hipfire-arch-api --lib
cargo test -p hipfire-arch-specs --lib
cargo test -p hipfire-arch-template-spec --lib
cargo test -p hipfire-arch-template --lib
cargo test -p hipfire-archs --lib
cargo test -p hipfire-arch-llama --lib caps

echo "== Tiny-fixture round-trip (CPU: emit → quantize, no GPU) =="
bash tests/fixture-roundtrip-nogpu.sh

echo "== Tiny affected-file selector (no GPU) =="
bash tests/tiny-affected-gate-nogpu.sh

echo "== Resident/streamed parity workflow (no GPU) =="
bash tests/resident-streamed-parity-nogpu.sh

echo "== ZAYA ragged-slice parity workflow (no GPU) =="
bash tests/zaya-ragged-slice-parity-nogpu.sh

echo "== Eval harness no-GPU smoke =="
cargo build -p hipfire-eval
HIPFIRE_EVAL_BIN="$ROOT/target/debug/hipfire-eval" bash tests/smoke/eval-harness-nogpu-smoke.sh

echo "== Installer link layout =="
bash tests/install-links.sh

echo "== Python CPU tests =="
"$PYTHON" -m ruff check .
"$PYTHON" -m mypy tests scripts benchmarks tools --config-file pyproject.toml
"$PYTHON" -m pytest tests

echo "== Env-var registry (declared in hipfire-env, per-crate enforcement) =="
./tests/env-registry-gate.sh

echo "== Env-var docs coverage (docs named HIPFIRE_* vars exist in source) =="
cargo run -q -p hipfire-cli -- gen-env-docs --check

echo "== CLI docs freshness (docs/CLI.md + man/ vs clap definition) =="
cargo run -q -p hipfire-cli -- gen-docs --check

echo "== Config schema freshness (docs/config-schema.* vs schema registry) =="
cargo run -q -p hipfire-cli -- gen-config-schema --format json --output docs/config-schema.json --check
cargo run -q -p hipfire-cli -- gen-config-schema --format toml --output docs/config-schema.toml --check

echo "== Model-support freshness (model_support_generated.rs + MODEL-SUPPORT.md vs docs/model-support.toml) =="
cargo run -q -p hipfire-cli -- gen-model-support --check
cargo run -q -p hipfire-cli -- gen-config-schema --format markdown --output docs/config-schema.md --check

echo "== Artifact naming check =="
bash scripts/check-artifact-names.sh

echo "== Arch capability-layer purity (no format tokens in arch-api / *-spec) =="
bash scripts/check-arch-spec-purity.sh

echo "== Eval smoke script syntax =="
bash -n tests/smoke/eval-harness-nogpu-smoke.sh
bash -n tests/smoke/eval-harness-gpu-smoke.sh
bash -n tests/smoke/eval-harness-model-eval-smoke.sh
bash -n tests/tiny-affected-gate.sh
bash -n tests/tiny-affected-gate-nogpu.sh
bash -n tests/tiny-quant-gate.sh
bash -n tests/tiny-state-gate.sh
bash -n tests/tiny-spec-gate.sh
bash -n tests/smoke/diffusion-sdapi-smoke.sh
bash -n tests/smoke/diffusion-tiny-sd-hfq-admission.sh

echo "== Legacy CLI checks =="
echo "Legacy CLI support has been removed; no script-runtime checks are run."
