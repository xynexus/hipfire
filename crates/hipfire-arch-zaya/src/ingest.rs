// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline safetensors → canonical ZAYA1 ingest (family-specific).
//!
//! The raw Zyphra ZAYA1 checkpoint stores **80 alternating half-layers** (even
//! index = CCA attention, odd index = EDA/MoD MoE) under `zaya_block` /
//! `local_experts` / `res_scale` names, whereas hipfire's loader ([`crate::gpu`])
//! reads **40 hybrid decoder blocks** with canonical names. The mapping collapses
//! each `(attention 2k, MoE 2k+1)` half-layer pair into block `k`
//! (`block = raw_layer / 2`, matching upstream `convert_zaya_weights_to_hf.py`'s
//! `new_layer_idx = old // 2`).
//!
//! Per AGENTS.md this family-specific knowledge lives in the arch crate; the
//! generic conversion driver (open dir / read bytes / write `.hfq`) lives in
//! `hipfire-coexistence`.

/// HFQM arch id written into the converted container header.
pub const ZAYA_ARCH_ID: u32 = 16;

/// BF16 on-disk quant-type code (matches `hipfire-quant-format`'s `Bf16`).
pub const BF16_QUANT_TYPE: u8 = 16;

/// Map a raw ZAYA1 checkpoint tensor name to its canonical hipfire name (or
/// `None` for names outside the known layout). `num_blocks` is the collapsed
/// block count (`num_hidden_layers / 2`), needed to route the model-level
/// residual scale onto the last block.
///
/// **Residual scales are pre-half-layer scales**, applied to the *input* of each
/// half-layer, so they sit one half-layer ahead of that half-layer's weights:
/// - raw `layers.0.res_scale` = the model input scale (`input_hidden_states_*`);
/// - raw `layers.{2l+1}.res_scale` (odd) = block `l`'s **post-attention** residual;
/// - raw `layers.{2l+2}.res_scale` (even ≥2) = block `l`'s **post-MLP** residual;
/// - `model.res_scale` = the final (last block's post-MLP) residual scale.
///
/// Every other tensor stays with its own half-layer: block = `raw_layer / 2`,
/// attention side on even indices, MoE side on odd.
pub fn canonical_name(raw: &str, num_blocks: usize) -> Option<String> {
    // ── model-level ──────────────────────────────────────────────────────────
    match raw {
        "model.embed_tokens.weight" => return Some(raw.to_string()),
        "model.final_norm.weight" => return Some("model.norm.weight".to_string()),
        _ => {}
    }
    if let Some(sub) = raw.strip_prefix("model.res_scale.") {
        // Final residual scale → the last block's post-MLP residual.
        let last = num_blocks.saturating_sub(1);
        return Some(format!("model.layers.{last}.post_mlp_residual_scale.{sub}"));
    }

    // ── per-half-layer ───────────────────────────────────────────────────────
    let rest = raw.strip_prefix("model.layers.")?;
    let (idx, tail) = rest.split_once('.')?;
    let raw_layer: usize = idx.parse().ok()?;

    // Residual scale: pre-half-layer, shifted one half-layer ahead of weights.
    if let Some(sub) = tail.strip_prefix("res_scale.") {
        if raw_layer == 0 {
            // Input scale of layer 0 = the model input scale (hidden-states only;
            // layer 0 carries no residual sub-tensors).
            return match sub {
                "hidden_states_scale" => Some("model.input_hidden_states_scale".to_string()),
                "hidden_states_bias" => Some("model.input_hidden_states_bias".to_string()),
                _ => None,
            };
        }
        return if raw_layer % 2 == 1 {
            let l = (raw_layer - 1) / 2;
            Some(format!(
                "model.layers.{l}.post_attention_residual_scale.{sub}"
            ))
        } else {
            let l = raw_layer / 2 - 1;
            Some(format!("model.layers.{l}.post_mlp_residual_scale.{sub}"))
        };
    }

    let block = raw_layer / 2;
    let p = format!("model.layers.{block}");
    if raw_layer % 2 == 0 {
        // even index → CCA attention half-layer → the block's attention side.
        let qkv = format!("{p}.self_attn.qkv_proj");
        let c = match tail {
            "input_norm.weight" => format!("{p}.input_layernorm.weight"),
            "self_attn.o_proj.weight" => format!("{p}.self_attn.o_proj.weight"),
            "self_attn.qkv.linear_q.weight" => format!("{qkv}.q_proj.weight"),
            "self_attn.qkv.linear_k.weight" => format!("{qkv}.k_proj.weight"),
            "self_attn.qkv.val_proj1.weight" => format!("{qkv}.v_proj_current.weight"),
            "self_attn.qkv.val_proj2.weight" => format!("{qkv}.v_proj_delayed.weight"),
            "self_attn.qkv.conv_qk.0.weight" => format!("{qkv}.conv_qk_depthwise.weight"),
            "self_attn.qkv.conv_qk.0.bias" => format!("{qkv}.conv_qk_depthwise.bias"),
            "self_attn.qkv.conv_qk.1.weight" => format!("{qkv}.conv_qk_grouped.weight"),
            "self_attn.qkv.conv_qk.1.bias" => format!("{qkv}.conv_qk_grouped.bias"),
            "self_attn.qkv.temp" => format!("{p}.self_attn.qk_norm.temp"),
            _ => return None,
        };
        Some(c)
    } else {
        // odd index → EDA/MoD MoE half-layer → the block's MoE side.
        let g = format!("{p}.mlp.gate");
        let rmlp = format!("{g}.router_mlp");
        if let Some(er) = tail.strip_prefix("zaya_block.experts.local_experts.") {
            let (e, proj) = er.split_once('.')?;
            return match proj {
                "linear_fc1.weight" => Some(format!("{p}.mlp.experts.{e}.gate_up_proj.weight")),
                "linear_fc2.weight" => Some(format!("{p}.mlp.experts.{e}.down_proj.weight")),
                _ => None,
            };
        }
        // Router MLP linears sit at even sub-indices (GELU activations occupy the
        // odd slots): router_mlp.0 → fc1, .2 → fc2, .4 → out_proj (17-way logits).
        let c = match tail {
            "input_norm.weight" => format!("{p}.post_attention_layernorm.weight"),
            "zaya_block.router.down_proj.weight" => format!("{g}.down_proj.weight"),
            "zaya_block.router.down_proj.bias" => format!("{g}.down_proj.bias"),
            "zaya_block.router.rmsnorm_eda.weight" => format!("{rmlp}.norm.weight"),
            "zaya_block.router.router_mlp.0.weight" => format!("{rmlp}.fc1.weight"),
            "zaya_block.router.router_mlp.0.bias" => format!("{rmlp}.fc1.bias"),
            "zaya_block.router.router_mlp.2.weight" => format!("{rmlp}.fc2.weight"),
            "zaya_block.router.router_mlp.2.bias" => format!("{rmlp}.fc2.bias"),
            "zaya_block.router.router_mlp.4.weight" => format!("{rmlp}.out_proj.weight"),
            "zaya_block.router.balancing_biases" => format!("{g}.balancing_biases"),
            "zaya_block.router.router_states_scale" => format!("{g}.router_states_scale"),
            _ => return None,
        };
        Some(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NB: usize = 40;

    #[test]
    fn maps_model_level() {
        assert_eq!(
            canonical_name("model.final_norm.weight", NB).as_deref(),
            Some("model.norm.weight")
        );
        // model-level residual scale → last block's post-MLP residual
        assert_eq!(
            canonical_name("model.res_scale.residual_scale", NB).as_deref(),
            Some("model.layers.39.post_mlp_residual_scale.residual_scale")
        );
    }

    #[test]
    fn residual_scales_shift_one_half_layer() {
        // layer 0 res_scale (hidden only) = the model input scale
        assert_eq!(
            canonical_name("model.layers.0.res_scale.hidden_states_scale", NB).as_deref(),
            Some("model.input_hidden_states_scale")
        );
        assert_eq!(
            canonical_name("model.layers.0.res_scale.residual_scale", NB),
            None
        );
        // odd layer 1 res_scale → block 0 post-attention residual
        assert_eq!(
            canonical_name("model.layers.1.res_scale.residual_scale", NB).as_deref(),
            Some("model.layers.0.post_attention_residual_scale.residual_scale")
        );
        // even layer 2 res_scale → block 0 post-MLP residual
        assert_eq!(
            canonical_name("model.layers.2.res_scale.hidden_states_scale", NB).as_deref(),
            Some("model.layers.0.post_mlp_residual_scale.hidden_states_scale")
        );
        // odd layer 3 res_scale → block 1 post-attention residual
        assert_eq!(
            canonical_name("model.layers.3.res_scale.residual_bias", NB).as_deref(),
            Some("model.layers.1.post_attention_residual_scale.residual_bias")
        );
    }

    #[test]
    fn merges_half_layers_into_blocks() {
        // raw attention layer 2 → block 1 attention side
        assert_eq!(
            canonical_name("model.layers.2.self_attn.qkv.linear_q.weight", NB).as_deref(),
            Some("model.layers.1.self_attn.qkv_proj.q_proj.weight")
        );
        // raw MoE layer 3 → block 1 MoE side
        assert_eq!(
            canonical_name(
                "model.layers.3.zaya_block.experts.local_experts.7.linear_fc1.weight",
                NB
            )
            .as_deref(),
            Some("model.layers.1.mlp.experts.7.gate_up_proj.weight")
        );
        assert_eq!(
            canonical_name("model.layers.3.zaya_block.router.router_mlp.4.weight", NB).as_deref(),
            Some("model.layers.1.mlp.gate.router_mlp.out_proj.weight")
        );
        // even layer input_norm → input_layernorm; odd layer input_norm → post_attn
        assert_eq!(
            canonical_name("model.layers.0.input_norm.weight", NB).as_deref(),
            Some("model.layers.0.input_layernorm.weight")
        );
        assert_eq!(
            canonical_name("model.layers.1.input_norm.weight", NB).as_deref(),
            Some("model.layers.0.post_attention_layernorm.weight")
        );
    }
}
