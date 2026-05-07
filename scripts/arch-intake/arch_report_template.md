# {{ARCH_NAME}} Architecture Intake Report

**Date:** {{DATE}}
**Branch (code):** {{CODE_BRANCH}}
**Branch (docs):** {{DOC_BRANCH}}
**Original branch (preserved):** {{ORIGINAL_BRANCH}} (if any)
**Rebase target:** `master` HEAD `{{MASTER_HEAD_SHA}}`

## TL;DR

{{1-3 sentences: arch released? branch state? where does the port land?
What's blocking a coherence-gate-passing forward?}}

## 1. Branch State

### 1.1. Released model lineup

| HF model ID | Params | Class | Status |
|-------------|--------|-------|--------|
| ... | ... | ... | ... |

Source: huggingface.co/{{ORG}}, verified {{DATE}}.

### 1.2. Original branch commits (if any)

```
{{git log --oneline ORIGINAL_BRANCH}}
```

The branch is **{{N}} commits behind master** at start of intake. The
intervening master changes that conflict:

- ...

### 1.3. Rebase strategy decision

{{Direct per-commit rebase vs. squash-port. State the choice and rationale.}}

### 1.4. Rebase outcome

Single port commit on `{{CODE_BRANCH}}`:
```
{{git log --oneline}}
```

Surface created:

| Path | Origin | Notes |
|------|--------|-------|
| ... | ... | ... |

### 1.5. Conflicts resolved

| Surface | Choice (master vs branch) | Rationale |
|---------|--------------------------|-----------|
| ... | ... | ... |

### 1.6. Build status

`cargo build --release` (full workspace, default features): {{GREEN/RED}}.
`cargo build --release -p hipfire-arch-{{NAME}} --examples`: {{GREEN/RED}}.

### 1.7. Smoke test status

{{Outcome: not run / run on what hardware / what happened.}}

## 2. Architecture Characterization

Source: `{{ArchName}}Config` parser in `crates/hipfire-arch-{{NAME}}/src/{{NAME}}.rs`,
backed by HuggingFace `config.json` keys.

| Field | Value | HF config key | Notes |
|-------|-------|---------------|-------|
| hidden_size | ... | hidden_size | ... |
| n_layers | ... | num_hidden_layers | ... |
| vocab_size | ... | vocab_size | ... |
| ... | ... | ... | ... |

### Distinctives vs. closest existing arch

1. ...
2. ...

## 3. Kernel-Fit Checklist

Coverage analysis for a single decode step.

| Step | Kernel needed | Status |
|------|---------------|--------|
| Embed lookup | ... | EXISTS / NEW / NEEDS-ADAPTION |
| Input RMSNorm | rmsnorm_f32 | EXISTS |
| Q/K/V projections | weight_gemv HFQ Q4/Q6/Q8 | ... |
| RoPE | ... | ... |
| KV cache write | kv_cache_write_* | ... |
| FlashAttention | attention_flash_* | ... |
| O-projection | weight_gemv | ... |
| Post-attention RMSNorm | rmsnorm_f32 | ... |
| FFN gate + up | weight_gemv_swiglu_residual | ... |
| FFN down | weight_gemv | ... |
| LM head | weight_gemv | ... |
| Final softcap | logit_softcap_f32 | ... |

Net new kernels in this port: **{{N}}**.

## 4. Remaining Gaps to a Coherence-Gate-Passing Forward

Listed in execution order; estimates assume one focused contributor.

### Gap 1: ...

{{description}}

**Effort:** {{hours/days}}.

**Impact:** {{what's blocked}}.

### Gap 2: ...

...

## Total effort estimate

{{X}} focused {{days/weeks}} to land a coherence-gate-passing forward.
Critical path is {{Gap N}} followed by {{Gap M}}. Other gaps are
parallelizable.

## Files

- `crates/hipfire-arch-{{NAME}}/src/{{NAME}}.rs` — primary forward path.
- `crates/hipfire-arch-{{NAME}}/src/arch.rs` — trait impl.
- `crates/hipfire-arch-{{NAME}}/examples/{{NAME}}_smoke_forward.rs` — smoke test.
- `kernels/src/...` — net-new kernels.
- `crates/rdna-compute/src/dispatch.rs` — net-new helpers.

## References

- Original branch: ...
- Rebased branch: ...
- HF model lineup: ...
- Architecture trait: `crates/hipfire-runtime/src/arch.rs`.
- Architecture-port reference templates: `crates/hipfire-arch-toy/`, `crates/hipfire-arch-llama/`.
- Arch-intake pipeline: `docs/plans/arch-intake-pipeline.md`.
