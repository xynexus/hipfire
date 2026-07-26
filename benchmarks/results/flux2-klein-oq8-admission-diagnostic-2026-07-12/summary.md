# hipfire eval summary

- model: `/srv/huggingface/FLUX.2-klein-base-4B.oq8.hfq`
- model hash: `eacc300a88a98d586081d8605e9e96fb582f6120757f0565ddf8a08a59054b8d`
- baseline: `/srv/huggingface/FLUX.2-klein-base-4B.diffusers.p0.hfq`
- baseline hash: `52ea667f20a9e1ec6995afe30758cbf6296070b459eae7597e644c6b35d34c5c`
- tier: `extensive`
- tier target: `0` seconds (full native/evidence suite; no fixed wall-clock target)
- CI suitable: `false`
- hipfire version: `0.3.0`
- runner: `hipfire-eval 0.3.0`
- git commit: `12892469d1d68c331be59ec71768bfcd280219dc`
- git branch: `chaingun`
- git describe: `v0.1.0-alpha-3980-g12892469d-dirty`
- git dirty: `true`
- binary hash: `010bd4c3356b9ecd425b3f2341622d70e0e0acc418adf9381786d563456a1f0d`
- arch: `gfx1103`
- ROCm: `7.14.60850-d34cbb6409`
- hardware bucket: `apu_uma:gfx1103:0x15bf:12cu:1gib:ddr5:128bit:90gbps`
- host profile hash: `fnv64:cd988deb3c3da519`
- rows: 0 pass / 1 fail / 0 skip

## Models

| role | identifier | exists | file hash | tag hash | metadata | quantization hash |
|---|---|---|---|---|---|---|
| candidate | /srv/huggingface/FLUX.2-klein-base-4B.oq8.hfq | true | eacc300a88a98d586081d8605e9e96fb582f6120757f0565ddf8a08a59054b8d |  | pass |  |
| baseline | /srv/huggingface/FLUX.2-klein-base-4B.diffusers.p0.hfq | true | 52ea667f20a9e1ec6995afe30758cbf6296070b459eae7597e644c6b35d34c5c |  | pass |  |

## Datasets

| suite | status | source | repo | revision | digest | license | selected | selected items | cache | reason |
|---|---|---|---|---|---|---|---:|---|---|---|
| none | Skip | none |  |  |  |  | 0 |  |  | no dataset-backed suites selected |

## Comparisons

- status: `Skip`
- reason: `candidate rows exist, but matching baseline/reference metric rows are absent`
- cases: `0`

## Admission

- status: `Fail`
- verdict: `reject`
- reason: `1 frozen diffusion RGB baseline comparison(s) failed`
- required evidence: `1`
- findings: `0`

| evidence | status | rows | reason |
|---|---|---|---|
| diffusion | Skip | 0 | no passing diffusion evidence rows |

### Observed Evidence

| evidence | status | rows | reason |
|---|---|---|---|
| phase_timings | Skip | 0 | no observed phase_timings evidence rows |
| launch_counts | Skip | 0 | no observed launch_counts evidence rows |
| moe_router_histogram | Skip | 0 | no observed moe_router_histogram evidence rows |
| memory | Skip | 0 | no observed memory evidence rows |
| dflash_trace | Skip | 0 | no observed dflash_trace evidence rows |
| path_c_trace | Skip | 0 | no observed path_c_trace evidence rows |
| module_evidence | Skip | 0 | no observed module_evidence evidence rows |
| coherence | Skip | 0 | no observed coherence evidence rows |
| profiling | Skip | 0 | profiling disabled by --profile off |

## Evidence Artifacts

| artifact | status | path | detail |
|---|---|---|---|
| admission | fail | artifacts/admission.json | reject |
| coherence | not_collected | artifacts/coherence.json | 0 |
| comparisons | skip | artifacts/comparisons.json | 0 |
| dflash_trace | not_collected | artifacts/dflash_trace.json | 0 |
| launch_counts | not_collected | artifacts/launch_counts.json | 0 |
| memory | not_collected | artifacts/memory.json | 0 |
| module_evidence | not_collected | artifacts/module_evidence.json | 0 |
| moe_router_histogram | not_collected | artifacts/moe_router_histogram.json | 0 |
| path_c_trace | not_collected | artifacts/path_c_trace.json | 0 |
| performance | not_collected | artifacts/performance.json | 0 |
| phase_timings | not_collected | artifacts/phase_timings.json | 0 |
| profiling | disabled | artifacts/profiling.json | 0 |
| quality | not_collected | artifacts/quality.json | 0 |
| run_metadata | collected | artifacts/run_metadata.json |  |

## Rows

| battery | suite | case | item | model | model hash | prompt hash | status | reason |
|---|---|---|---|---|---|---|---|---|
| diffusion |  | rgb_baseline_0 | seed-7 | /srv/huggingface/FLUX.2-klein-base-4B.oq8.hfq | eacc300a88a98d586081d8605e9e96fb582f6120757f0565ddf8a08a59054b8d | 9bd9b1c6097c9bc836d096cd4a5ff240723c39e10f7480108c1b6dbd54c98fd5 | Fail | RGB drift mae=76.0064 max=255 exceeds frozen limits 1/4 |
