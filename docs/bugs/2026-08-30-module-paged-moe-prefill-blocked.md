# Module-paged MoE residency cannot prefill: three stacked gaps

Status: found 2026-08-30 on nix1, master `3351ec70d`, Qwen3.6-35B-A3B
`oq4.25++` on gfx1103. **NOT fixed** — the blocker is kernel correctness, not
wiring. Scope at the end.

## Symptom

With a VRAM budget small enough that `ResidencyMode::Auto` selects
`QwenMoeModules`, the model loads and paging arms:

```
paged experts enabled: registered 10240 routed expert modules, cache_budget=9.31 GiB
```

and then every generate fails:

```
prefill: HipError(0): paged grouped-MoE prefill requires grouped GEMM path2
support (hipError=0)
```

This was reachable only after PR #393 gave the budget a non-zero value; with the
0 default `Auto` always resolved to `Full`, so module residency had never been
selected and this path had never run.

## Cause

Paging routes through `routed_expert_buckets`, and the bucketed path requires
grouped GEMM path 2:

```rust
// crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs:978
if routed_expert_buckets.is_some() && !path2_eligible {
    return Err(HipError::new(
        0,
        "paged grouped-MoE prefill requires grouped GEMM path2 support",
    ));
}
```

`HIPFIRE_KERNEL_TRACE=1` names the decline precisely:

```
MoE grouped GEMM declined: expert_gate_up=Oq8G256 arch=gfx1103 use_path2=true
mixed=false experts_empty=true buckets=true n=218 k_top=8 n_exp=256
```

`n=218` clears the batch threshold and `use_path2=true`, so the refusal is the
dtype/arch predicate. Three separate gaps sit behind it.

### 1. The grouped kernel for this dtype is off because it is wrong

```rust
// crates/hipfire-arch-qwen35/src/qwen35/mod.rs:2745
// ⚠️ Oq8G256 grouped is OPT-IN and OFF by default: the kernel is FAST and
// WRONG. gemm_oq8g256_moe_grouped_wmma + its path-2 arms exist and route
// (Qwen3.6-35B-A3B 215.0 -> 391.0 tok/s, 1.8x), but the output degenerates
// into echoing the prompt, while the same model on path-1 answers 4/4.
// So the grouped DISPATCH is right and the OQ8 decode is not -- most
// likely the weight_byte_offset [...]
DType::Oq8G256 => {
    arch.starts_with("gfx11")
        && std::env::var("HIPFIRE_MOE_OQ8_GROUPED").as_deref() == Ok("1")
}
```

### 2. The bucketed path has no arm for this dtype either

Setting `HIPFIRE_MOE_OQ8_GROUPED=1` gets past (1) and lands on a second refusal:

```
prefill: HipError(0): bucketed grouped-MoE gate_up does not support expert 1
dtype Oq8G256
```

```rust
// crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs:1019
if !mixed_routed_quant_dtype_supported(dtype)
    && !matches!(dtype, DType::F16 | DType::BF16)
{
    return Err(HipError::new(0, &format!(
        "bucketed grouped-MoE gate_up does not support expert {expert} dtype {dtype:?}"
    )));
}
```

So the paged path accepts only the mixed-routed-quant dtypes plus F16/BF16, and
`Oq8G256` is neither — independent of whether the kernel in (1) is correct.

### 3. Paged MoE cannot serve short prompts at all

```rust
// crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs:884
const MOE_GROUPED_MIN_BATCH: usize = 64;
```

`path2_eligible` requires `n >= MOE_GROUPED_MIN_BATCH`. Paging *requires* path 2
(the check in `Cause` above), so a paged MoE model refuses any prompt under 64
tokens even once (1) and (2) are solved. First observed with an 8-token prompt,
which is what made this look like a dtype problem until a 218-token prompt failed
the same way.

## Scope of the fix

Reverse order of cost; each step is independently useful.

* **(3) first, ~1 day.** Give the bucketed path a path-1 fallback with
  `ensure_paged_experts_resident` already in hand (`prefill_chunk.rs:1010`), so a
  short prompt runs indexed GEMV over paged experts instead of refusing. Removes
  a hard failure without touching a kernel.
* **(1) next, the real work, ~1 week.** Validate `gemm_oq8g256_moe_grouped_wmma`
  against a reference. The hypothesis is already recorded — `weight_byte_offset`,
  passed as 0 on the strength of a path-1 kernel that addresses blocks directly,
  where the Oq4 sibling documents a nonzero offset for the combined resident
  layout. `parity_gemm_oq_compact_moe_grouped` is the shape to copy: the compact
  sibling was moved onto a bit-exact f32 path and verified at 0 mismatches of up
  to 2,097,152 values.
* **(2) falls out of (1)** — once the kernel is trusted, adding `Oq8G256` to the
  bucketed predicate is a one-line change that no longer risks silent garbage.

Until then, module-paged residency should not be selected for `Oq8G256` routed
experts. PR #393 keeps a DERIVED budget advisory precisely so it cannot select
it; an operator who sets `scheduler_vram_budget_bytes` explicitly still can, and
will hit this.
