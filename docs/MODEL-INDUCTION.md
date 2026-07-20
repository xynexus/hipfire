# Model induction

`scripts/induct_model.py` is the resumable offline workflow for turning a
Hugging Face source checkpoint into the artifacts needed by hipfire. For the
Qwen3.5 397B path it produces:

- BF16 and F16 DFlash draft sidecars converted from the published z-lab
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
~/.hipfire/drafts/Qwen3.5-397B-A17B-F16.dflash.hfq
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

The DFlash stage creates both BF16 and F16 sidecars by default. RDNA3/RDNA4 use
the BF16 artifact natively; older cards can select the explicit F16 artifact,
and the runtime can also convert a BF16 payload to F16 while loading. To build
only one representation, pass one `--dflash-format`; repeat the flag to request
a custom set. MQ draft formats are separate size/speed experiments and need
their own acceptance evidence. The target defaults to `oq4.25++`: mixed
4.25-bit Opus storage with
AWQ plus Hessian/LDLQ feedback. Additional target quantizer flags can be
supplied after `--`.

On the primary gfx1151 host, calibration defaults to 64 independent sequences
by 32 time positions (2,048 rows). Override these with `--batch-size`,
`--time-tile`, and `--max-rows`; all three are recorded in the two-pass recipe
fingerprint, so resume cannot silently change the geometry. The native engine
also performs live memory estimates and allocation probes before accepting an
automatic geometry on a different host.

The target pass defaults to a 16 GiB bounded next-layer source prefetch. While
one layer executes, the native engine reads the following layer's canonical
safetensor ranges into bounded resident host staging without consuming its
read-ledger entries. The next layer consumes complete tensor views directly
from staging, which is freed after synchronous GPU upload. It retains a 32 GiB
live host-memory reserve plus the following layer's upload footprint and
reduces the effective budget when required. A transition is disabled when
Linux reports recent full-memory PSI or less than 25% free swap, and no
mid-tensor prefix is retained because it cannot satisfy a direct view. Override
with `--layer-prefetch-bytes N`, or pass zero to disable lookahead; the chosen
operational budget is recorded in the two-pass recipe while pressure decisions
are recorded per layer.

Expert quality controls are also explicit induction inputs rather than hidden
calibrator defaults: `--min-expert-activations`,
`--expert-capture-target`, `--expert-capture-tile-rows`,
`--required-expert-fraction`, `--sampling-seed`, and
`--expert-coverage-policy`. They are forwarded to the native pass and included
in the two-pass recipe fingerprint. This makes strict/fallback runs and the
coverage/capture sweeps distinct resumable recipes even if native CLI defaults
later change. Use `astrea expert-sweep-plan` to freeze those sweeps before
execution; [QUANTIZE.md](QUANTIZE.md) records the family-neutral command and
held-out evidence contract.

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
the native calibration engine-build identity, read ledger,
source/run/sample fingerprints, the cheap HFQ metadata/index fingerprints, and
the quantizer's payload hash. A file with the right magic but stale or missing
provenance is regenerated. DFlash and TriAttention stages additionally require
their expected magic. Useful controls are:

```bash
python3 scripts/induct_model.py --stage dflash
python3 scripts/induct_model.py --stage target
python3 scripts/induct_model.py --stage triattn
python3 scripts/induct_model.py --stage target --force
```

When a complete calibration artifact exists but the target quant does not,
induction automatically asks the two-pass workflow to reuse it. Reuse first
runs the native calibrator's no-GPU dry plan and compares the artifact's family,
adapter, architecture, source/shard identities, tokenizer, corpus and sampled
tokens, microbatch geometry, F32 boundary mode, expert capture policy, and
KLDREF settings against the requested run. The native run fingerprint also
binds the complete adapter tensor plan, calibration job, and geometry. Any
mismatch fails before quantization; `--force` disables automatic reuse and
regenerates calibration.

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

Resident-oracle comparisons use the offline, family-neutral artifact gate:

```bash
hipfire-coexistence artifact compare-calibration \
  --reference <resident.calib.hfq> \
  --candidate <streamed.calib.hfq>
```

The command refuses to treat numerically similar artifacts as matched evidence
when either package lacks the frozen corpus and sample fingerprints. Its JSON
report records structural, provenance, non-finite, and tolerance failures for
the induction evidence ledger.
