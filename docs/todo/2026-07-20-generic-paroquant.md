# TODO: Make ParoQuant model-generic

Status: open

ParoQuant loading currently has a generic-looking primitive layer in
`crates/hipfire-runtime/src/hfq.rs`, but the complete and actively used path is
still specialized around Qwen3.5 in
`crates/hipfire-arch-qwen35/src/qwen35/loading.rs`, its Qwen35 weight layout,
and Qwen35-specific MoE decode dispatch. This makes direct safetensors support
look broader than the actual runtime contract.

## Target contract

- Put checkpoint-format parsing, AWQ INT4 repacking, Givens sidecar validation,
  and tensor-name resolution behind model-independent ParoQuant tooling APIs.
- Represent a ParoQuant projection as an architecture-neutral descriptor:
  logical role, `[M,K]`, group size, packed weight tensors, and rotation
  metadata. Do not return `Qwen35Weights` from the generic layer.
- Keep model-family code responsible only for mapping logical roles into its
  runtime weight layout and choosing the matching HIP dispatch operation.
- Support dense and fused/split MoE tensors without baking
  `model.language_model.layers.*` or Qwen expert naming into the format reader.
- Keep import/conversion work in `hipfire-coexistence` (or another dedicated
  tooling crate). Runtime may load the normalized native contract, but must not
  grow a general compatibility/conversion pipeline in the inference hot path.

## Proposed slices

1. Extract the duplicated `load_paroquant_weight*`, AWQ nibble reorder, and
   Givens validation code from the Qwen35 loader and runtime HFQ module into one
   tested format component.
2. Define a `ParoQuantTensorResolver` over `ModelSource` with configurable
   prefix/role aliases and unambiguous errors for missing or conflicting names.
3. Add a normalized per-projection result consumed by architecture adapters;
   migrate the existing LLaMA/Qwen3 path first, then Qwen3.5 dense attention and
   shared MLP, then routed MoE experts.
4. Move raw ParoQuant safetensors normalization/export into
   `hipfire-coexistence`; emit canonical HFQM/HFQ metadata so subsequent loads
   do not repeat source-format discovery.
5. Delete the Qwen35-local repacker and legacy name fallbacks once all callers
   use the shared resolver.

## Acceptance criteria

- One authoritative implementation of ParoQuant packing and rotation parsing.
- Table-driven unit fixtures cover at least LLaMA, Qwen3, Qwen3.5 dense, and
  Qwen3.5 fused MoE naming/layouts.
- Malformed group size, rotation dimensions, expert counts, and missing
  sidecars fail before GPU upload with tensor-specific diagnostics.
- Existing Qwen3.5 ParoQuant smoke output remains coherent, and a second model
  family passes the same loader-level fixture and runtime smoke.
- RDNA2, RDNA3, and RDNA4 dispatch behavior is unchanged unless a separately
  measured kernel change is intentionally included.
