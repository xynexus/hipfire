// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline spec for **Qwen3.8-Flash-Next** (`model_type: qwen4_exp`), arch id 26.
//!
//! Identity + the `Ingest` quant-policy. Deps only `hipfire-arch-api`, so the
//! quantizer links it without the runtime/GPU stack.
//!
//! The family is a hybrid of pieces this tree already serves separately —
//! Gated DeltaNet (as Qwen3.5/3.8), a 4-wide gated residual and a sparse-attention
//! indexer (as DeepSeek V4), a Qwen3-VL tower — plus one genuinely new component,
//! a hashed n-gram embedding table. Semantics below were read from the vendored
//! reference at `third_party/transformers-qwen4_exp/`, not inferred from the
//! checkpoint; see `docs/plans/2026-08-29-qwen4exp-flash-next-scope.md`.
//!
//! Tensor prefixes this classifies (from the real checkpoint's 1658 names):
//!
//! ```text
//! model.language_model.layers.{i}.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,
//!                                              in_proj_b,conv1d,A_log,dt_bias,norm,out_proj}
//! model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj / {q,k}_norm
//! model.language_model.layers.{i}.self_attn.indexer.{index_qk_proj,q_layernorm,k_layernorm}
//! model.language_model.layers.{i}.mlp.{experts.gate_up_proj,experts.down_proj,gate,
//!                                      shared_expert.*,shared_expert_gate}
//! model.language_model.layers.{i}.{attn,mlp}_hyper_connection.{input_mix_weight_down,
//!                                  input_mix_weight_up,block_inject_weight,hc_norm}
//! model.language_model.layers.{i}.ple.{conv1d,key_proj,value_proj,norm_*}
//! model.language_model.layers.{i}.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight
//! model.language_model.layers.{i}.ple.ple_embedding.{ngram_heads_offsets,
//!                                  ngram_heads_vocab_sizes,layer_multipliers}
//! mtp.*                     — an embedded multi-token-prediction block
//! model.visual.*            — the vision tower
//! ```
//!
//! NOTE this checkpoint has **no** `input_layernorm` and **no**
//! `post_attention_layernorm` anywhere, and no final `model.language_model.norm`:
//! `hc_norm` replaces them. Grepping all 1658 names for "layernorm" returns only
//! the two indexer norms.

use hipfire_arch_api::{
    default_importance, default_precision_class, default_requires, register_arch, transformer_role,
    Arch, ArchId, CapReq, Dt, ExpertLayout, Ingest, Init, PrecisionClass, TensorRole, TensorSpec,
    ToyFixture, ToyModel,
};

/// Qwen3.8-Flash-Next (`qwen4_exp`) header id.
pub const QWEN4EXP_ARCH_ID: ArchId = ArchId(26);

/// Lean identity marker for the Qwen3.8-Flash-Next offline spec.
pub struct Qwen4ExpSpec;

impl Arch for Qwen4ExpSpec {
    fn id(&self) -> ArchId {
        QWEN4EXP_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "qwen4-exp"
    }
    fn model_types(&self) -> &'static [&'static str] {
        &["qwen4_exp", "qwen4_exp_text"]
    }
}

impl Qwen4ExpSpec {
    /// The hashed n-gram embedding table: 128 shards of `[2500012, 160]`, one flat
    /// 320,001,536-row table, ~102 GB at source width — 41% of the model's
    /// parameters, injected at a single layer.
    ///
    /// It is a **row gather**: one token reads exactly 16 rows (16 hash heads x
    /// 160 dims). So it needs random access, which the `Embed` role already
    /// requires — but the shared prior would reach that role via a `contains
    /// ("embed")` substring match, which is luck rather than intent here, so this
    /// classifies it explicitly.
    fn is_ngram_table(tensor: &str) -> bool {
        tensor.contains("ngram_embedding.shard_")
    }

    /// The n-gram addressing metadata: per-head prefix offsets, per-head prime
    /// moduli, and the three hash multipliers. Integer indices, not weights — a
    /// lossy round-trip here does not degrade quality, it addresses the wrong row.
    fn is_ngram_index(tensor: &str) -> bool {
        tensor.ends_with("ngram_heads_offsets")
            || tensor.ends_with("ngram_heads_vocab_sizes")
            || tensor.ends_with("layer_multipliers")
    }

    /// Gated DeltaNet ingress and recurrent state controls. The convolution state
    /// and the decay/gate parameters feed a recurrence, so error here compounds
    /// across the sequence rather than averaging out — the same reason SSM ingress
    /// is pinned elsewhere in the tree.
    fn is_recurrent_ingress(tensor: &str) -> bool {
        // The PLE short convolution carries state too — it is DILATED by
        // `ngram_size`, giving a 9-slot history rather than the mixer's 3 — so it
        // belongs here for the same reason, even though it is not `linear_attn`.
        if tensor.contains("ple.conv1d") {
            return true;
        }
        if !tensor.contains("linear_attn.") {
            return false;
        }
        tensor.ends_with("A_log")
            || tensor.ends_with("dt_bias")
            || tensor.contains("conv1d")
            || tensor.contains("in_proj_a")
            || tensor.contains("in_proj_b")
    }

    /// Names carrying the substring "embed" that are NOT gathered lookup tables.
    /// The shared prior keys `TensorRole::Embed` off that substring, which here
    /// would put three unrelated kinds of tensor on a random-access codec for no
    /// benefit — found by running this policy over the checkpoint's real 1658
    /// names (`examples/classify.rs`):
    ///
    /// * `model.visual.patch_embed.proj.*` — a patch-embedding CONVOLUTION,
    ///   consumed as a flat linear, never indexed by row;
    /// * `model.visual.pos_embed.weight` — a learned position table that is
    ///   bilinearly INTERPOLATED to the image grid, so it is read as a whole
    ///   plane rather than gathered;
    /// * `mtp.fc_embedding.weight` — a square projection that consumes the
    ///   embedding; `mtp.pre_fc_norm_embedding.weight` — a norm over it.
    fn is_false_embed(tensor: &str) -> bool {
        tensor.contains("patch_embed")
            || tensor.contains("pos_embed")
            || tensor.contains("fc_embedding")
            || tensor.contains("norm_embedding")
    }

    /// The sparse-attention indexer. It does not produce activations — it produces
    /// a SELECTION, choosing which 512 micro-blocks of the KV cache the main
    /// attention may see. An error does not perturb a value slightly; it attends to
    /// the wrong tokens, and the damage is unbounded and non-local. Same standing
    /// as DeepSeek V4's indexer, which is kept at source fidelity for this reason.
    fn is_selection_stream(tensor: &str) -> bool {
        tensor.contains("self_attn.indexer.")
    }

    /// The gated-residual (hyper-connection) mixers. Every block reads and writes a
    /// 4-wide residual stream through these, twice per layer, so they sit directly
    /// on the residual path.
    ///
    /// Split by size, because the two halves have very different cost. The low-rank
    /// mix pair is ~634 M parameters read every token across the model and cannot
    /// be held at source width; the per-branch scalars and the norm are a few
    /// thousand values and are free to protect absolutely.
    fn is_residual_mix_bulk(tensor: &str) -> bool {
        tensor.contains("input_mix_weight_down") || tensor.contains("input_mix_weight_up")
    }
    fn is_residual_mix_tiny(tensor: &str) -> bool {
        tensor.contains("block_inject_weight") || tensor.contains("hc_norm")
    }
}

impl Ingest for Qwen4ExpSpec {
    fn role(&self, tensor: &str) -> TensorRole {
        if Self::is_ngram_table(tensor) {
            // A gathered lookup table, like a token embedding.
            return TensorRole::Embed;
        }
        if Self::is_ngram_index(tensor) {
            return TensorRole::Other;
        }
        if Self::is_false_embed(tensor) {
            // Classify by what it structurally IS, bypassing the substring.
            return if tensor.contains("norm") {
                TensorRole::Norm
            } else if tensor.contains("bias") {
                TensorRole::Other
            } else {
                TensorRole::AttnProj
            };
        }
        if Self::is_recurrent_ingress(tensor) {
            // `conv1d` has its own role; the scalar decay/gate vectors do not, and
            // reporting them as `Other` keeps them out of the projection buckets.
            return if tensor.contains("conv1d") {
                TensorRole::Conv1d
            } else {
                TensorRole::Other
            };
        }
        transformer_role(tensor)
    }

    fn importance(&self, tensor: &str) -> u8 {
        // Structural saliency only — never a format. The three overrides are the
        // tensors whose failure mode is categorical rather than a small numeric
        // perturbation.
        if Self::is_ngram_index(tensor) || Self::is_selection_stream(tensor) {
            return u8::MAX;
        }
        if Self::is_recurrent_ingress(tensor) || Self::is_residual_mix_tiny(tensor) {
            return u8::MAX;
        }
        default_importance(self.role(tensor))
    }

    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }

    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        // Categorical failure modes: wrong row, wrong token selection, or a
        // corrupted recurrence. None of these degrade gracefully.
        if Self::is_ngram_index(tensor)
            || Self::is_selection_stream(tensor)
            || Self::is_residual_mix_tiny(tensor)
        {
            return PrecisionClass::SourcePrecision;
        }
        // The n-gram table cannot be calibrated: with 320 M rows, any corpus
        // touches well under 1% of them, so a fitted per-row codec is extrapolated
        // from almost nothing. It also lives on disk rather than in memory, where
        // its width costs I/O blocks rather than residency — see the scope doc's
        // decision 3. Keep it at source fidelity.
        if Self::is_ngram_table(tensor) {
            return PrecisionClass::SourcePrecision;
        }
        // Recurrent ingress corrupts state when lossy; the residual mix bulk is too
        // large for source width but must not be spent down under a tight budget.
        if Self::is_recurrent_ingress(tensor) || Self::is_residual_mix_bulk(tensor) {
            return PrecisionClass::Pinned;
        }
        default_precision_class(self.role(tensor))
    }

    fn expert_layout(&self) -> ExpertLayout {
        // 512 experts stacked: `experts.gate_up_proj` is [E, 2*mi, hidden] and
        // `experts.down_proj` is [E, hidden, mi]; the quantizer splits per expert.
        ExpertLayout::StackedGateUpDown
    }
}

/// The `k`-th prime strictly greater than `start`, 1-indexed — the reference's
/// `_find_nth_prime_after`. The n-gram table gives each hash head its own prime
/// modulus so their collisions decorrelate, and the primes are DERIVED from
/// `ngram_vocab_size_base` rather than stored, so a loader that gets this wrong
/// addresses a different row of the table and fails silently.
pub fn nth_prime_after(start: u64, count: u64) -> u64 {
    fn is_prime(v: u64) -> bool {
        if v < 2 {
            return false;
        }
        if v % 2 == 0 {
            return v == 2;
        }
        let mut d = 3u64;
        while d.saturating_mul(d) <= v {
            if v % d == 0 {
                return false;
            }
            d += 2;
        }
        true
    }
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

/// Per-head vocabulary sizes and their prefix-sum offsets, plus the padded total.
/// Mirrors `Qwen4ExpTextNGramEmbedding.__init__`: head `h` takes the `(h+1)`-th
/// prime after `base - 1`, offsets are the running sum, and the total is padded up
/// to a multiple of `divisor` before being split into shards.
pub fn ngram_head_layout(base: u64, n_heads: usize, divisor: u64) -> (Vec<u64>, Vec<u64>, u64) {
    ngram_head_layout_at(base, n_heads, divisor, 0)
}

/// As [`ngram_head_layout`], for the `ple_index`-th PLE block.
///
/// The reference indexes the prime ladder GLOBALLY across PLE blocks —
/// `global_head_idx = ple_layer_index * ngram_heads + head_idx` — so a second PLE
/// block continues the ladder rather than restarting it. The shipped checkpoint
/// has one block (`ple_index = 0`), where this is the identity.
pub fn ngram_head_layout_at(
    base: u64,
    n_heads: usize,
    divisor: u64,
    ple_index: usize,
) -> (Vec<u64>, Vec<u64>, u64) {
    let mut sizes = Vec::with_capacity(n_heads);
    let mut offsets = Vec::with_capacity(n_heads);
    let mut total = 0u64;
    for h in 0..n_heads {
        let global = ple_index * n_heads + h;
        let size = nth_prime_after(base - 1, global as u64 + 1);
        offsets.push(total);
        sizes.push(size);
        total += size;
    }
    let padded = total.div_ceil(divisor) * divisor;
    (sizes, offsets, padded)
}

/// Tiny Qwen3.8-Flash-Next fixture. Mirrors the real checkpoint's tensor names and
/// structural relationships at fixture dims, so the loader, quantizer k-map and
/// ingest policy all see the shapes they will see in production.
struct Qwen4ExpTiny {
    hidden: usize,
    vocab: usize,
    layers: usize,
    full_attn_interval: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    // gated DeltaNet
    l_key_heads: usize,
    l_val_heads: usize,
    l_head_dim: usize,
    conv_kernel: usize,
    // sparse-attention indexer
    idx_heads: usize,
    idx_kv_heads: usize,
    idx_head_dim: usize,
    // MoE
    experts: usize,
    experts_per_tok: usize,
    moe_inter: usize,
    shared_inter: usize,
    // gated residual
    hc_count: usize,
    hc_lowrank: usize,
    // per-layer n-gram embedding
    ple_layer: usize,
    ple_embed_dim: usize,
    ngram_size: usize,
    heads_per_ngram: usize,
    ngram_base: u64,
    ngram_shards: usize,
}

impl Qwen4ExpTiny {
    /// ~15M params. `hidden = 512` is two G256 groups on the `gate_up` reduction
    /// dim, not one — a single group would stop covering multi-group accumulation,
    /// which is the trap the sibling Qwen3.5 MoE preset documents.
    ///
    /// 4 layers at `full_attn_interval = 4` reproduces the real 3:1 pattern
    /// (three Gated DeltaNet layers then one sparse-attention layer). The MoE here
    /// is deliberately SMALL and admissible (4 experts, top-2, `moe_inter = 256`)
    /// so this fixture is about the ARCHITECTURE; the production 512-expert /
    /// top-10 / `moe_inter = 640` geometry is its own axis and has its own probe.
    fn preset() -> Self {
        Self {
            hidden: 512,
            vocab: 4096,
            layers: 4,
            full_attn_interval: 4,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 256,
            l_key_heads: 2,
            l_val_heads: 6,
            l_head_dim: 128,
            conv_kernel: 4,
            idx_heads: 4,
            idx_kv_heads: 1,
            idx_head_dim: 128,
            experts: 4,
            experts_per_tok: 2,
            moe_inter: 256,
            shared_inter: 256,
            hc_count: 4,
            hc_lowrank: 64,
            ple_layer: 1,
            ple_embed_dim: 512,
            ngram_size: 3,
            heads_per_ngram: 8,
            ngram_base: 200,
            ngram_shards: 4,
        }
    }

    /// The production routed-MoE geometry: top-10 over 12 experts at
    /// `moe_inter = 640`. `moe_inter % 256 != 0` and `k != 8` are the two
    /// conditions the indexed decode path refuses, so this variant is what
    /// exercises the fallback the real model will take.
    fn moe_production_preset() -> Self {
        Self {
            experts: 12,
            experts_per_tok: 10,
            moe_inter: 640,
            shared_inter: 640,
            layers: 2,
            full_attn_interval: 2,
            ..Self::preset()
        }
    }

    fn hc_hidden(&self) -> usize {
        self.hidden * self.hc_count
    }
    fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }
    /// Rows per shard: the padded total split evenly, exactly as the checkpoint's
    /// 128 shards are a uniform slice of one flat table.
    fn ngram_shard_rows(&self) -> usize {
        let (_, _, padded) = ngram_head_layout(self.ngram_base, self.ngram_heads(), 128);
        (padded as usize).div_ceil(self.ngram_shards)
    }

    fn config_json(&self) -> String {
        let layer_types: Vec<&str> = (0..self.layers)
            .map(|i| {
                if (i + 1) % self.full_attn_interval == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let lt = layer_types
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(", ");
        // `ple_layer_ids` is ONE-BASED in this family: the reference matches on
        // `ple_layer_ids.index(layer_idx + 1)`, which is why the checkpoint's
        // `[2]` names its tensors `layers.1.ple.*`.
        format!(
            r#"{{
  "architectures": ["Qwen4ExpForConditionalGeneration"],
  "model_type": "qwen4_exp",
  "tie_word_embeddings": false,
  "text_config": {{
    "model_type": "qwen4_exp_text",
    "hidden_size": {hidden},
    "vocab_size": {vocab},
    "num_hidden_layers": {layers},
    "full_attention_interval": {interval},
    "layer_types": [{lt}],
    "num_attention_heads": {n_heads},
    "num_key_value_heads": {n_kv},
    "head_dim": {head_dim},
    "partial_rotary_factor": 0.25,
    "rope_parameters": {{ "rope_theta": 10000000, "rope_type": "default",
                          "mrope_interleaved": true, "mrope_section": [11, 11, 10],
                          "partial_rotary_factor": 0.25 }},
    "linear_num_key_heads": {lk},
    "linear_num_value_heads": {lv},
    "linear_key_head_dim": {lhd},
    "linear_value_head_dim": {lhd},
    "linear_conv_kernel_dim": {conv},
    "output_gate_type": "sigmoid",
    "indexer_n_heads": {ih},
    "indexer_kv_heads": {ikv},
    "indexer_head_dim": {ihd},
    "indexer_budget": 64,
    "indexer_compress_ratio": 4,
    "num_experts": {experts},
    "num_experts_per_tok": {k},
    "moe_intermediate_size": {mi},
    "shared_expert_intermediate_size": {smi},
    "hc_count": {hc},
    "hc_lowrank": {hcr},
    "ple_layer_ids": [{ple_one_based}],
    "ple_embed_dim": {ple_dim},
    "ple_conv_kernel_size": {conv},
    "ngram_size": {ng},
    "heads_per_ngram": {hpn},
    "ngram_vocab_size_base": {base},
    "make_ngram_vocab_size_divisible_by": 128,
    "split_ngram_parts": {shards},
    "seed": 0,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu"
  }}
}}"#,
            hidden = self.hidden,
            vocab = self.vocab,
            layers = self.layers,
            interval = self.full_attn_interval,
            n_heads = self.n_heads,
            n_kv = self.n_kv_heads,
            head_dim = self.head_dim,
            lk = self.l_key_heads,
            lv = self.l_val_heads,
            lhd = self.l_head_dim,
            conv = self.conv_kernel,
            ih = self.idx_heads,
            ikv = self.idx_kv_heads,
            ihd = self.idx_head_dim,
            experts = self.experts,
            k = self.experts_per_tok,
            mi = self.moe_inter,
            smi = self.shared_inter,
            hc = self.hc_count,
            hcr = self.hc_lowrank,
            ple_one_based = self.ple_layer + 1,
            ple_dim = self.ple_embed_dim,
            ng = self.ngram_size,
            hpn = self.heads_per_ngram,
            base = self.ngram_base,
            shards = self.ngram_shards,
        )
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let u = |s: f32| Init::Uniform(s);
        let hcw = self.hc_hidden();
        let mut t: Vec<TensorSpec> = Vec::new();
        let lm = "model.language_model";

        t.push(TensorSpec::new(
            format!("{lm}.embed_tokens.weight"),
            vec![self.vocab, self.hidden],
            u(0.02),
        ));
        // Untied head — `tie_word_embeddings` is false in this family.
        t.push(TensorSpec::new(
            "lm_head.weight".to_string(),
            vec![self.vocab, self.hidden],
            u(0.02),
        ));
        // Model-level gated-residual mixer: the ONLY pre-head norm, since this
        // family has no `model.norm` and no per-block input/post-attention norms.
        t.push(TensorSpec::new(
            format!("{lm}.hyper_connection_mixer.hc_norm.weight"),
            vec![hcw],
            Init::NormOnes,
        ));
        t.push(TensorSpec::new(
            format!("{lm}.hyper_connection_mixer.input_mix_weight_down.weight"),
            vec![self.hc_lowrank, hcw],
            u(0.02),
        ));
        t.push(TensorSpec::new(
            format!("{lm}.hyper_connection_mixer.input_mix_weight_up.weight"),
            vec![hcw, self.hc_lowrank],
            u(0.02),
        ));

        for l in 0..self.layers {
            let p = format!("{lm}.layers.{l}");
            let is_full = (l + 1) % self.full_attn_interval == 0;

            // Two gated-residual blocks per layer, one before each sub-block.
            for which in ["attn_hyper_connection", "mlp_hyper_connection"] {
                t.push(TensorSpec::new(
                    format!("{p}.{which}.hc_norm.weight"),
                    vec![hcw],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{p}.{which}.input_mix_weight_down.weight"),
                    vec![self.hc_lowrank, hcw],
                    u(0.02),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.{which}.input_mix_weight_up.weight"),
                    vec![hcw, self.hc_lowrank],
                    u(0.02),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.{which}.block_inject_weight.weight"),
                    vec![self.hc_count, hcw],
                    u(0.02),
                ));
            }

            if is_full {
                let q_out = self.n_heads * self.head_dim;
                let kv_out = self.n_kv_heads * self.head_dim;
                // Q is DOUBLED: it carries the attention output gate interleaved
                // with the queries, which is why `q_proj` is 2x the head span.
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.q_proj.weight"),
                    vec![q_out * 2, self.hidden],
                    u(0.02),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.k_proj.weight"),
                    vec![kv_out, self.hidden],
                    u(0.02),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.v_proj.weight"),
                    vec![kv_out, self.hidden],
                    u(0.02),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.o_proj.weight"),
                    vec![self.hidden, q_out],
                    u(0.02),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.q_norm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.k_norm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
                // One projection feeds both indexer queries and its single shared
                // key head, so its output span is (n_heads + kv_heads) * head_dim.
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.indexer.index_qk_proj.weight"),
                    vec![
                        (self.idx_heads + self.idx_kv_heads) * self.idx_head_dim,
                        self.hidden,
                    ],
                    u(0.02),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.indexer.q_layernorm.weight"),
                    vec![self.idx_head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{p}.self_attn.indexer.k_layernorm.weight"),
                    vec![self.idx_head_dim],
                    Init::NormOnes,
                ));
            } else {
                let k_span = self.l_key_heads * self.l_head_dim;
                let v_span = self.l_val_heads * self.l_head_dim;
                let qkv = k_span * 2 + v_span;
                t.push(TensorSpec::new(
                    format!("{p}.linear_attn.in_proj_qkv.weight"),
                    vec![qkv, self.hidden],
                    u(0.02),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.linear_attn.in_proj_z.weight"),
                    vec![v_span, self.hidden],
                    u(0.02),
                ));
                for ab in ["in_proj_a", "in_proj_b"] {
                    t.push(TensorSpec::new(
                        format!("{p}.linear_attn.{ab}.weight"),
                        vec![self.l_val_heads, self.hidden],
                        u(0.02),
                    ));
                }
                t.push(TensorSpec::new(
                    format!("{p}.linear_attn.conv1d.weight"),
                    vec![qkv, 1, self.conv_kernel],
                    u(0.1),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.linear_attn.A_log"),
                    vec![self.l_val_heads],
                    Init::ALog,
                ));
                t.push(TensorSpec::new(
                    format!("{p}.linear_attn.dt_bias"),
                    vec![self.l_val_heads],
                    Init::Zeros,
                ));
                t.push(TensorSpec::new(
                    format!("{p}.linear_attn.norm.weight"),
                    vec![self.l_head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{p}.linear_attn.out_proj.weight"),
                    vec![self.hidden, v_span],
                    u(0.02),
                ));
            }

            // MoE on EVERY layer, routed experts stacked.
            t.push(TensorSpec::new(
                format!("{p}.mlp.gate.weight"),
                vec![self.experts, self.hidden],
                u(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.experts.gate_up_proj"),
                vec![self.experts, self.moe_inter * 2, self.hidden],
                u(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.experts.down_proj"),
                vec![self.experts, self.hidden, self.moe_inter],
                u(0.02),
            ));
            for proj in ["gate_proj", "up_proj"] {
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.{proj}.weight"),
                    vec![self.shared_inter, self.hidden],
                    u(0.02),
                ));
            }
            t.push(TensorSpec::new(
                format!("{p}.mlp.shared_expert.down_proj.weight"),
                vec![self.hidden, self.shared_inter],
                u(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.shared_expert_gate.weight"),
                vec![1, self.hidden],
                u(0.02),
            ));

            // The n-gram / per-layer-embedding block sits on exactly one layer.
            if l == self.ple_layer {
                t.push(TensorSpec::new(
                    format!("{p}.ple.conv1d.weight"),
                    vec![hcw, 1, self.conv_kernel],
                    u(0.1),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.ple.key_proj.weight"),
                    vec![hcw, self.ple_embed_dim],
                    u(0.02),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.ple.value_proj.weight"),
                    vec![self.hidden, self.ple_embed_dim],
                    u(0.02),
                ));
                for n in ["norm_query", "norm_key", "norm_conv"] {
                    t.push(TensorSpec::new(
                        format!("{p}.ple.{n}.weight"),
                        vec![hcw],
                        Init::NormOnes,
                    ));
                }
                let rows = self.ngram_shard_rows();
                let per_head = self.ple_embed_dim / self.ngram_heads();
                for sh in 0..self.ngram_shards {
                    t.push(TensorSpec::new(
                        format!("{p}.ple.ple_embedding.ngram_embedding.shard_{sh}.weight"),
                        vec![rows, per_head],
                        u(0.02),
                    ));
                }
                // NOTE `ngram_heads_offsets` / `ngram_heads_vocab_sizes` /
                // `layer_multipliers` are deliberately ABSENT. They are integer
                // buffers the reference DERIVES in `__init__` from
                // `ngram_vocab_size_base` / `ngram_size` / `seed`, and the fixture
                // manifest carries floats only. A loader must derive them the same
                // way (see `ngram_head_layout`) and prefer the checkpoint's stored
                // copies when present.
            }
        }
        t.into_iter()
            .map(|s| TensorSpec { dt: Dt::Bf16, ..s })
            .collect()
    }
}

impl ToyModel for Qwen4ExpSpec {
    fn fixture(&self, seed: u64) -> ToyFixture {
        self.fixture_named("default", seed)
            .expect("default fixture")
    }
    fn fixture_names(&self) -> &'static [&'static str] {
        &["default", "moe-production"]
    }
    fn fixture_named(&self, name: &str, _seed: u64) -> Option<ToyFixture> {
        let m = match name {
            "default" => Qwen4ExpTiny::preset(),
            "moe-production" => Qwen4ExpTiny::moe_production_preset(),
            _ => return None,
        };
        Some(ToyFixture {
            config_json: m.config_json(),
            tensors: m.manifest(),
        })
    }
}

static QWEN4EXP_SPEC: Qwen4ExpSpec = Qwen4ExpSpec;
register_arch!(QWEN4EXP_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    const NGRAM_SHARD: &str =
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_37.weight";
    const NGRAM_OFFSETS: &str =
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets";
    const INDEXER: &str = "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight";
    const GDN_ALOG: &str = "model.language_model.layers.0.linear_attn.A_log";
    const GDN_CONV: &str = "model.language_model.layers.0.linear_attn.conv1d.weight";
    const HC_DOWN: &str =
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight";
    const HC_INJECT: &str =
        "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight";
    const EXPERT_GU: &str = "model.language_model.layers.0.mlp.experts.gate_up_proj";

    #[test]
    fn registers_ingest_under_arch_26() {
        let reg = ArchRegistry::build();
        let a = reg.get(QWEN4EXP_ARCH_ID).expect("qwen4exp spec registered");
        assert_eq!(a.family, "qwen4-exp");
        assert!(a.caps.ingest.is_some());
    }

    /// The n-gram table is 41% of the model's parameters and is READ BY ROW.
    /// Landing it anywhere but a random-access class would make the on-disk
    /// gather impossible, so this is the load-bearing classification in the file.
    #[test]
    fn ngram_table_is_a_random_access_lookup_at_source_fidelity() {
        let s = Qwen4ExpSpec;
        assert_eq!(s.role(NGRAM_SHARD), TensorRole::Embed);
        assert!(
            s.requires(NGRAM_SHARD).random_access,
            "a 16-row-per-token gather cannot use a sequential codec"
        );
        assert_eq!(
            s.precision_class(NGRAM_SHARD),
            PrecisionClass::SourcePrecision
        );
    }

    /// Addressing metadata, not weights. Quantizing these does not blur a value,
    /// it reads a different row of a 320 M-row table.
    #[test]
    fn ngram_index_tensors_are_never_compressed() {
        let s = Qwen4ExpSpec;
        assert_eq!(
            s.precision_class(NGRAM_OFFSETS),
            PrecisionClass::SourcePrecision
        );
        assert_eq!(s.importance(NGRAM_OFFSETS), u8::MAX);
        for t in [
            "x.ngram_heads_vocab_sizes",
            "x.layer_multipliers",
            "x.ngram_heads_offsets",
        ] {
            assert_eq!(s.precision_class(t), PrecisionClass::SourcePrecision, "{t}");
        }
    }

    /// The indexer picks WHICH tokens attention sees. Its failure is categorical,
    /// which is why it outranks ordinary attention projections.
    #[test]
    fn selection_stream_outranks_ordinary_attention() {
        let s = Qwen4ExpSpec;
        assert_eq!(s.precision_class(INDEXER), PrecisionClass::SourcePrecision);
        let ordinary = "model.language_model.layers.3.self_attn.q_proj.weight";
        assert!(
            s.precision_class(INDEXER) > s.precision_class(ordinary),
            "indexer must rank above a plain attention projection"
        );
    }

    /// Recurrent ingress compounds error across the sequence rather than
    /// averaging it, so it is pinned above the bulk.
    #[test]
    fn recurrent_ingress_is_pinned_above_bulk() {
        let s = Qwen4ExpSpec;
        for t in [GDN_ALOG, GDN_CONV] {
            assert_eq!(s.precision_class(t), PrecisionClass::Pinned, "{t}");
        }
        assert_eq!(s.role(GDN_CONV), TensorRole::Conv1d);
        assert!(s.precision_class(GDN_ALOG) > s.precision_class(EXPERT_GU));
    }

    /// The two halves of the gated residual are deliberately split: the low-rank
    /// mix pair is ~634 M params read every token and cannot sit at source width,
    /// while the per-branch scalars are a handful of values and are free to pin.
    #[test]
    fn residual_mix_splits_bulk_from_scalars() {
        let s = Qwen4ExpSpec;
        assert_eq!(s.precision_class(HC_DOWN), PrecisionClass::Pinned);
        assert_eq!(
            s.precision_class(HC_INJECT),
            PrecisionClass::SourcePrecision
        );
        assert!(
            s.precision_class(HC_INJECT) > s.precision_class(HC_DOWN),
            "scalars are cheap to protect absolutely; the mix pair is not"
        );
    }

    /// 512 experts ship stacked; the quantizer must split them per expert.
    #[test]
    fn routed_experts_are_stacked() {
        assert_eq!(
            Qwen4ExpSpec.expert_layout(),
            ExpertLayout::StackedGateUpDown
        );
    }

    /// Ordinary tensors must still fall through to the shared prior — the overrides
    /// above are exceptions, not a replacement classification.
    #[test]
    fn ordinary_tensors_take_the_shared_prior() {
        let s = Qwen4ExpSpec;
        for t in [
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            EXPERT_GU,
        ] {
            assert_eq!(s.role(t), transformer_role(t), "{t}");
            assert_eq!(
                s.precision_class(t),
                default_precision_class(transformer_role(t)),
                "{t}"
            );
        }
    }
}

#[cfg(test)]
mod real_checkpoint_name_tests {
    use super::*;

    /// Regressions for the three misclassifications that only appeared when this
    /// policy was run over the checkpoint's real 1658 tensor names. All three are
    /// the same trap: the shared prior keys `Embed` off a `contains("embed")`
    /// substring, and three unrelated kinds of tensor contain it.
    #[test]
    fn substring_embed_does_not_capture_non_tables() {
        let s = Qwen4ExpSpec;
        for t in [
            "model.visual.patch_embed.proj.weight",
            "model.visual.pos_embed.weight",
            "mtp.fc_embedding.weight",
        ] {
            assert_ne!(
                s.role(t),
                TensorRole::Embed,
                "{t} is not a gathered table; Embed would force random access"
            );
            assert!(
                !s.requires(t).random_access,
                "{t} must not demand a random-access codec"
            );
        }
        assert_eq!(
            s.role("mtp.pre_fc_norm_embedding.weight"),
            TensorRole::Norm,
            "a norm over the embedding is a norm"
        );
    }

    /// The real table still classifies as one — the guard above must not overreach.
    #[test]
    fn real_gathered_tables_keep_random_access() {
        let s = Qwen4ExpSpec;
        for t in [
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        ] {
            assert_eq!(s.role(t), TensorRole::Embed, "{t}");
            assert!(s.requires(t).random_access, "{t}");
        }
    }

    /// The PLE short conv is dilated by `ngram_size` (9-slot history) and carries
    /// state across positions, so it is pinned like the DeltaNet conv rather than
    /// left on the ordinary `Conv1d` default.
    #[test]
    fn ple_conv_is_pinned_like_the_deltanet_conv() {
        let s = Qwen4ExpSpec;
        let ple = "model.language_model.layers.1.ple.conv1d.weight";
        let gdn = "model.language_model.layers.0.linear_attn.conv1d.weight";
        assert_eq!(s.precision_class(ple), PrecisionClass::Pinned);
        assert_eq!(s.precision_class(ple), s.precision_class(gdn));
    }
}

#[cfg(test)]
mod fixture_tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    fn names(f: &ToyFixture) -> Vec<String> {
        f.tensors.iter().map(|t| t.name.clone()).collect()
    }

    #[test]
    fn registers_toy_model() {
        let reg = ArchRegistry::build();
        let a = reg.get(QWEN4EXP_ARCH_ID).unwrap();
        assert!(a.caps.toy_model.is_some(), "ToyModel must be registered");
    }

    /// The prime ladder is DERIVED, not stored, so getting it wrong addresses a
    /// different row of the table and fails silently. Pinned against the values
    /// the reference's `_find_nth_prime_after` produces.
    #[test]
    fn ngram_prime_ladder_matches_the_reference() {
        // The real checkpoint's first three, for base 20_000_000.
        let sizes: Vec<u64> = (1..=3).map(|k| nth_prime_after(19_999_999, k)).collect();
        assert_eq!(sizes, vec![20_000_003, 20_000_023, 20_000_033]);
        // Offsets are the running prefix sum, so head 1 starts where head 0 ends.
        let (s, o, _) = ngram_head_layout(20_000_000, 3, 128);
        assert_eq!(o, vec![0, 20_000_003, 40_000_026]);
        assert_eq!(s[0] + s[1], o[2]);
    }

    /// Every head gets a DISTINCT modulus — that is the whole point of the
    /// multi-hash: same key, different prime, so collisions decorrelate.
    #[test]
    fn ngram_head_moduli_are_distinct() {
        let (sizes, _, _) = ngram_head_layout(200, 16, 128);
        let mut u = sizes.clone();
        u.sort_unstable();
        u.dedup();
        assert_eq!(u.len(), 16, "16 heads need 16 distinct primes: {sizes:?}");
    }

    /// The shard split must tile the padded table exactly, as the checkpoint's
    /// 128 shards are a uniform slice of one flat 320,001,536-row table.
    #[test]
    fn ngram_shards_tile_the_padded_table() {
        let m = Qwen4ExpTiny::preset();
        let (_, _, padded) = ngram_head_layout(m.ngram_base, m.ngram_heads(), 128);
        assert_eq!(padded % 128, 0, "padded total must be divisible by 128");
        assert_eq!(
            m.ngram_shard_rows() * m.ngram_shards,
            padded as usize,
            "shards must tile the padded table with no remainder"
        );
    }

    /// Structural invariants the real checkpoint has and the fixture must mirror,
    /// because each one is a place a loader can silently go wrong.
    #[test]
    fn fixture_mirrors_the_real_structure() {
        let f = Qwen4ExpSpec.fixture(0);
        let n = names(&f);
        let has = |s: &str| n.iter().any(|x| x.contains(s));

        // No per-block norms and no final norm — hc_norm replaces them.
        assert!(
            !has("input_layernorm"),
            "this family has no input_layernorm"
        );
        assert!(!has("post_attention_layernorm"));
        assert!(
            !n.iter().any(|x| x.ends_with("language_model.norm.weight")),
            "no final model norm in this family"
        );
        // The 3:1 pattern: three DeltaNet layers then one sparse-attention layer.
        let gdn = n
            .iter()
            .filter(|x| x.contains("linear_attn.in_proj_qkv"))
            .count();
        let full = n.iter().filter(|x| x.contains("self_attn.q_proj")).count();
        assert_eq!(
            (gdn, full),
            (3, 1),
            "4 layers at interval 4 = 3 GDN + 1 full"
        );
        // The indexer rides only the full-attention layers.
        assert_eq!(
            n.iter()
                .filter(|x| x.contains("indexer.index_qk_proj"))
                .count(),
            full
        );
        // Gated residual: two blocks per layer plus one model-level mixer.
        let m = Qwen4ExpTiny::preset();
        assert_eq!(
            n.iter()
                .filter(|x| x.contains("block_inject_weight"))
                .count(),
            m.layers * 2
        );
        assert!(has("hyper_connection_mixer.hc_norm"));
        // The n-gram block sits on exactly one layer.
        assert_eq!(
            n.iter()
                .filter(|x| x.contains("ngram_embedding.shard_"))
                .count(),
            m.ngram_shards
        );
        assert_eq!(n.iter().filter(|x| x.contains("ple.key_proj")).count(), 1);
        // Untied head.
        assert!(has("lm_head.weight") && has("embed_tokens.weight"));
    }

    /// Q carries the attention output gate interleaved with the queries, so its
    /// projection is twice the head span — the same shape as the real checkpoint's
    /// `q_proj [12288, 2560]` against `head_dim 256 * 24 heads = 6144`.
    #[test]
    fn q_projection_is_doubled_for_the_output_gate() {
        let m = Qwen4ExpTiny::preset();
        let f = Qwen4ExpSpec.fixture(0);
        let q = f
            .tensors
            .iter()
            .find(|t| t.name.contains("self_attn.q_proj"))
            .expect("q_proj");
        assert_eq!(q.shape, vec![m.n_heads * m.head_dim * 2, m.hidden]);
        let o = f
            .tensors
            .iter()
            .find(|t| t.name.contains("self_attn.o_proj"))
            .unwrap();
        assert_eq!(o.shape, vec![m.hidden, m.n_heads * m.head_dim]);
    }

    /// The DeltaNet value span is 3x the key span (48 V heads to 16 QK in the real
    /// model), and the qkv projection packs Q+K at the key span with V at the value
    /// span — an asymmetry the loader must split correctly.
    #[test]
    fn deltanet_value_span_is_three_times_the_key_span() {
        let m = Qwen4ExpTiny::preset();
        assert_eq!(m.l_val_heads, m.l_key_heads * 3);
        let f = Qwen4ExpSpec.fixture(0);
        let qkv = f
            .tensors
            .iter()
            .find(|t| t.name.contains("in_proj_qkv"))
            .unwrap();
        let k_span = m.l_key_heads * m.l_head_dim;
        let v_span = m.l_val_heads * m.l_head_dim;
        assert_eq!(qkv.shape, vec![k_span * 2 + v_span, m.hidden]);
        let z = f
            .tensors
            .iter()
            .find(|t| t.name.contains("in_proj_z"))
            .unwrap();
        assert_eq!(z.shape, vec![v_span, m.hidden], "the output gate spans V");
    }

    /// The production variant is the geometry the indexed decode path refuses.
    #[test]
    fn moe_production_variant_carries_the_refused_geometry() {
        let m = Qwen4ExpTiny::moe_production_preset();
        assert_eq!(m.experts_per_tok, 10, "k != 8");
        assert_ne!(m.moe_inter % 256, 0, "moe_inter is not a multiple of 256");
        assert!(
            m.experts > m.experts_per_tok,
            "top-k must be a real selection"
        );
        assert!(Qwen4ExpSpec.fixture_named("moe-production", 0).is_some());
    }

    /// Every fixture tensor must classify through this crate's own Ingest policy —
    /// a name the manifest emits but the policy mishandles is the exact failure the
    /// `classify` example exists to catch, so pin it here too.
    #[test]
    fn every_fixture_tensor_classifies_sanely() {
        let s = Qwen4ExpSpec;
        for f in ["default", "moe-production"] {
            for t in &Qwen4ExpSpec.fixture_named(f, 0).unwrap().tensors {
                let n = &t.name;
                if n.contains("ngram_embedding.shard_") {
                    assert_eq!(s.precision_class(n), PrecisionClass::SourcePrecision, "{n}");
                    assert!(s.requires(n).random_access, "{n}");
                } else if n.contains("block_inject_weight") || n.contains("hc_norm") {
                    assert_eq!(s.precision_class(n), PrecisionClass::SourcePrecision, "{n}");
                } else if n.contains("indexer.") {
                    assert_eq!(s.precision_class(n), PrecisionClass::SourcePrecision, "{n}");
                } else if n.contains("input_mix_weight") || n.contains("conv1d") {
                    assert_eq!(s.precision_class(n), PrecisionClass::Pinned, "{n}");
                }
                assert!(!t.shape.is_empty() && t.shape.iter().all(|d| *d > 0), "{n}");
            }
        }
    }
}
