# Model induction

`scripts/induct_model.py` is the resumable offline workflow for turning a
Hugging Face source checkpoint into the artifacts needed by hipfire. For the
Qwen3.5 397B path it produces:

- a BF16-preserving DFlash draft sidecar converted from the published z-lab
  safetensors;
- one unified calibration artifact containing Hessians, imatrices, routed
  expert statistics, router histograms, and matched-corpus KLDREF records;
- the calibrated target quant;
- a TriAttention band-center sidecar consumed by the CASK/TriAttention path;
- an induction manifest recording source identities, commands, stage state,
  and the evidence still required for admission.

The default artifact names match runtime registry discovery for the target
quant:

```text
~/.hipfire/models/Qwen3.5-397B-A17B.oq4.25++.hfq
~/.hipfire/drafts/Qwen3.5-397B-A17B-BF16.dflash.hfq
~/.hipfire/triattn/Qwen3.5-397B-A17B.triattn.hfq
~/.hipfire/calib/Qwen3.5-397B-A17B.calib.hfq
~/.hipfire/induction/Qwen3.5-397B-A17B.oq4.25++/manifest.json
~/.hipfire/induction/Qwen3.5-397B-A17B.oq4.25++/two-pass.json
```

## Qwen3.5 397B workflow

Inspect the resolved snapshots, compatibility contract, build steps, output
paths, and commands first:

```bash
python3 scripts/induct_model.py --dry-run
```

Run all stages:

```bash
python3 scripts/induct_model.py
```

The defaults resolve these local Hugging Face repositories:

```text
/srv/huggingface/models--Qwen--Qwen3.5-397B-A17B
/srv/huggingface/models--z-lab--Qwen3.5-397B-A17B-DFlash
```

The preflight requires the target and draft to agree on hidden size,
vocabulary size, and target-layer contract. Their attention geometries are
reported but deliberately not required to match: the published DFlash draft
uses its own head and KV-head layout.

The DFlash sidecar defaults to BF16 for the quality-first path. RDNA3/RDNA4
load those weights as native BF16; older cards convert the same BF16 payload to
F16 while loading, so the default artifact remains portable. Use
`--dflash-format f16` to create an explicitly F16 fallback artifact. MQ draft
formats are separate size/speed experiments and need their own acceptance
evidence. The target defaults to `oq4.25++`: mixed 4.25-bit Opus storage with
AWQ plus Hessian/LDLQ feedback. Additional target quantizer flags can be
supplied after `--`.

On the primary gfx1151 host, calibration defaults to 64 independent sequences
by 32 time positions (2,048 rows). Override these with `--batch-size`,
`--time-tile`, and `--max-rows`; all three are recorded in the two-pass recipe
fingerprint, so resume cannot silently change the geometry. The native engine
also performs live memory estimates and allocation probes before accepting an
automatic geometry on a different host.

Mixed-precision Opus targets are supported directly. For example,
`--format oq4.5++` keeps the `++` calibration recipe and emits canonically
named `oq4.5++` primary and sidecar artifacts. The bitwidth must satisfy the
quantizer's sparse-overlay grammar; see [QUANTIZE.md](QUANTIZE.md) for the
accepted increments.

## Passes and resume behavior

The expensive target source is read twice:

1. The layer-streamed teacher pass reads each BF16 tensor once and produces
   calibration statistics plus KLDREF. Calibration microbatches are processed
   while each layer is resident, and routed expert activations are recorded
   separately.
2. The quantizer reads the source once to apply those statistics and write the
   target HFQ.

TriAttention calibration then loads the quantized HFQ, not the BF16 source.
The DFlash conversion reads only the much smaller draft checkpoint. This keeps
the target induction at two source-checkpoint loads without conflating KLDREF
with Hessian collection: KLDREF is the matched teacher signal for candidate
evaluation, while Hessian/imatrix records drive AWQ and LDLQ.

The target stage is skipped only when both HFQ outputs are structurally valid
and `two-pass.json` matches the requested recipe. That atomic manifest embeds
the native calibration read ledger, source/run/sample fingerprints, the cheap
HFQ metadata/index fingerprints, and the quantizer's payload hash. A file with
the right magic but stale or missing provenance is regenerated. DFlash and
TriAttention stages additionally require their expected magic. Useful controls
are:

```bash
python3 scripts/induct_model.py --stage dflash
python3 scripts/induct_model.py --stage target
python3 scripts/induct_model.py --stage triattn
python3 scripts/induct_model.py --stage target --force
```

Repo-built tools are rebuilt when missing or older than their defining source.
Use `--no-auto-build` to require the tools to be current before starting. GPU
stages use the shared `hipfire lock` path. Native calibration acquires the lock
itself, the target wrapper scopes quantization once, and TriAttention
calibration is independently scoped.

## Admission is separate from generation

Successful artifact creation leaves the manifest at `admission.status =
"pending"`. It does not establish that the quant, DFlash, CASK, or their
combination should be promoted. Admission still requires finite-logit and
coherence checks, matched-corpus KLD/PPL evidence, DFlash acceptance and output
checks, long-context CASK/TriAttention recall, combined DFlash+CASK recall, and
Kernel Atlas AR/DFlash performance rows.

The TriAttention generator validates its own center statistics, but those
statistics alone are not a long-context quality result.
