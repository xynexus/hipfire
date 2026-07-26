// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Full Qwen3 embedding encoder assembled from resident AIE2P component images.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::qwen3_encoder_blob::Qwen3EncoderBlobWeights;
use crate::{
    NpuQwen3AttentionUnpack, NpuQwen3FinalPoolL2, NpuQwen3HeadNormRope, NpuQwen3KvPack,
    NpuQwen3Oq8Projection, NpuQwen3QueryPack, NpuQwen3ResidualRmsNorm, NpuQwen3SwiGlu,
    NpuSegmentedAttention, Qwen3HeadNormRopeGeometry, SegmentedAttentionGeometry, XdnaError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FullEmbeddingIoGeometry {
    pub sequence_bucket: usize,
    pub dispatch_batch: usize,
    pub hidden_size: usize,
    pub output_dimensions: usize,
}

#[derive(Debug)]
pub struct FullEmbeddingStageTrace {
    pub name: &'static str,
    pub columns: usize,
    pub token_major_bf16: Vec<u16>,
}

#[derive(Debug, Default)]
pub struct FullEmbeddingTrace {
    pub completed_layers_bf16: Vec<Vec<u16>>,
    pub stage_layer: usize,
    pub last_layer_stages: Vec<FullEmbeddingStageTrace>,
}

#[derive(Default)]
struct FullEmbeddingTimings {
    residual_norm: Duration,
    qkv_projection: Duration,
    headnorm_rope: Duration,
    attention_layout: Duration,
    attention: Duration,
    attention_output: Duration,
    gate_up_projection: Duration,
    swiglu: Duration,
    down_projection: Duration,
    final_pool: Duration,
}

impl FullEmbeddingTimings {
    fn report(&self) {
        for (name, elapsed) in [
            ("residual_norm", self.residual_norm),
            ("qkv_projection", self.qkv_projection),
            ("headnorm_rope", self.headnorm_rope),
            ("attention_layout", self.attention_layout),
            ("attention", self.attention),
            ("attention_output", self.attention_output),
            ("gate_up_projection", self.gate_up_projection),
            ("swiglu", self.swiglu),
            ("down_projection", self.down_projection),
            ("final_pool", self.final_pool),
        ] {
            eprintln!(
                "qwen3_embedding_stage_total stage={name} elapsed_ms={:.3}",
                elapsed.as_secs_f64() * 1e3
            );
        }
    }
}

impl FullEmbeddingIoGeometry {
    pub fn validate(self) -> Result<Self, XdnaError> {
        if !matches!(self.sequence_bucket, 128 | 256 | 512 | 1024 | 2048)
            || self.dispatch_batch == 0
            || self.hidden_size == 0
            || self.output_dimensions == 0
        {
            return Err(invalid("invalid full embedding encoder geometry"));
        }
        if self
            .sequence_bucket
            .checked_mul(self.dispatch_batch)
            .is_none_or(|rows| rows > 4096)
        {
            return Err(invalid(
                "full embedding encoder dispatch exceeds 4096 padded rows",
            ));
        }
        Ok(self)
    }

    fn rows(self) -> usize {
        self.sequence_bucket * self.dispatch_batch
    }

    fn input_elements(self) -> Result<usize, XdnaError> {
        self.rows()
            .checked_mul(self.hidden_size)
            .ok_or_else(|| invalid("full embedding encoder input size overflow"))
    }
}

pub struct NpuFullEmbeddingEncoder {
    geometry: FullEmbeddingIoGeometry,
    padded_rows: usize,
    layer_count: usize,
    residual_norm: NpuQwen3ResidualRmsNorm,
    query_projection: NpuQwen3Oq8Projection,
    kv_projection: NpuQwen3Oq8Projection,
    attention_output_projection: NpuQwen3Oq8Projection,
    gate_up_projection: NpuQwen3Oq8Projection,
    down_projection: NpuQwen3Oq8Projection,
    headnorm_rope: NpuQwen3HeadNormRope,
    query_pack: NpuQwen3QueryPack,
    kv_pack: NpuQwen3KvPack,
    attention: NpuSegmentedAttention,
    attention_unpack: NpuQwen3AttentionUnpack,
    swiglu: NpuQwen3SwiGlu,
    final_pool: NpuQwen3FinalPoolL2,
    zeros_hidden: Vec<u16>,
}

impl NpuFullEmbeddingEncoder {
    pub fn load_cached(
        bundle: impl AsRef<Path>,
        geometry: FullEmbeddingIoGeometry,
        weight_blob: &[u8],
    ) -> Result<Self, XdnaError> {
        let geometry = geometry.validate()?;
        let weights = Qwen3EncoderBlobWeights::parse(weight_blob)?;
        if geometry.hidden_size != weights.hidden_size
            || geometry.output_dimensions != weights.hidden_size
        {
            return Err(invalid(format!(
                "Qwen3 encoder blob hidden={} does not match IO hidden={} output={}",
                weights.hidden_size, geometry.hidden_size, geometry.output_dimensions
            )));
        }
        if weights.query_heads != 16 || weights.kv_heads != 8 || weights.head_dim != 128 {
            return Err(invalid(
                "the first Qwen3 full-encoder slice requires QH16/KVH8/D128",
            ));
        }
        let padded_rows = geometry.rows().next_multiple_of(256);
        let bundle = bundle.as_ref();
        let image = |name: &str| read_image(&bundle.join(name));

        let (xclbin, instructions) = image("residual-rmsnorm")?;
        let mut norm_weights = weights
            .layers
            .iter()
            .map(|layer| layer.input_norm.as_slice())
            .collect::<Vec<_>>();
        norm_weights.extend(
            weights
                .layers
                .iter()
                .map(|layer| layer.post_attention_norm.as_slice()),
        );
        norm_weights.push(weights.final_norm.as_slice());
        let residual_norm = NpuQwen3ResidualRmsNorm::load_bank(
            &xclbin,
            &instructions,
            padded_rows,
            weights.hidden_size,
            &norm_weights,
            weights.norm_epsilon,
        )?;

        let query_matrices = weights
            .layers
            .iter()
            .map(|layer| &layer.query)
            .collect::<Vec<_>>();
        let (xclbin, instructions) = image("projection-q")?;
        let query_projection = NpuQwen3Oq8Projection::load_bank(
            &xclbin,
            &instructions,
            padded_rows,
            weights.hidden_size,
            weights.query_heads * weights.head_dim,
            &query_matrices,
        )?;

        let kv_matrices = weights
            .layers
            .iter()
            .flat_map(|layer| [&layer.key, &layer.value])
            .collect::<Vec<_>>();
        let (xclbin, instructions) = image("projection-kv")?;
        let kv_projection = NpuQwen3Oq8Projection::load_bank(
            &xclbin,
            &instructions,
            padded_rows,
            weights.hidden_size,
            weights.kv_heads * weights.head_dim,
            &kv_matrices,
        )?;

        let attention_output_matrices = weights
            .layers
            .iter()
            .map(|layer| &layer.attention_output)
            .collect::<Vec<_>>();
        let (xclbin, instructions) = image("projection-o")?;
        let attention_output_projection = NpuQwen3Oq8Projection::load_bank(
            &xclbin,
            &instructions,
            padded_rows,
            weights.query_heads * weights.head_dim,
            weights.hidden_size,
            &attention_output_matrices,
        )?;

        let gate_up_matrices = weights
            .layers
            .iter()
            .flat_map(|layer| [&layer.gate, &layer.up])
            .collect::<Vec<_>>();
        let (xclbin, instructions) = image("projection-gate-up")?;
        let gate_up_projection = NpuQwen3Oq8Projection::load_bank(
            &xclbin,
            &instructions,
            padded_rows,
            weights.hidden_size,
            weights.intermediate_size,
            &gate_up_matrices,
        )?;

        let down_matrices = weights
            .layers
            .iter()
            .map(|layer| &layer.down)
            .collect::<Vec<_>>();
        let (xclbin, instructions) = image("projection-down")?;
        let down_projection = NpuQwen3Oq8Projection::load_bank(
            &xclbin,
            &instructions,
            padded_rows,
            weights.intermediate_size,
            weights.hidden_size,
            &down_matrices,
        )?;

        let query_norms = weights
            .layers
            .iter()
            .map(|layer| layer.query_norm.as_slice())
            .collect::<Vec<_>>();
        let key_norms = weights
            .layers
            .iter()
            .map(|layer| layer.key_norm.as_slice())
            .collect::<Vec<_>>();
        let (xclbin, instructions) = image("headnorm-rope")?;
        let headnorm_rope = NpuQwen3HeadNormRope::load_bank(
            &xclbin,
            &instructions,
            Qwen3HeadNormRopeGeometry {
                sequence_bucket: geometry.sequence_bucket,
                dispatch_batch: geometry.dispatch_batch,
                query_heads: weights.query_heads,
                kv_heads: weights.kv_heads,
                head_dim: weights.head_dim,
            },
            &query_norms,
            &key_norms,
            weights.rope_theta,
            weights.norm_epsilon,
        )?;

        let attention_geometry = SegmentedAttentionGeometry {
            sequence_bucket: geometry.sequence_bucket,
            dispatch_batch: geometry.dispatch_batch,
            query_heads: weights.query_heads,
            kv_heads: weights.kv_heads,
            head_dim: weights.head_dim,
        }
        .validate()
        .map_err(invalid)?;
        let (xclbin, instructions) = image("query-pack")?;
        let query_pack = NpuQwen3QueryPack::load(&xclbin, &instructions, attention_geometry)?;
        let (xclbin, instructions) = image("kv-pack")?;
        let kv_pack = NpuQwen3KvPack::load(&xclbin, &instructions, attention_geometry)?;
        let (xclbin, instructions) = image("attention")?;
        let attention = NpuSegmentedAttention::load(&xclbin, &instructions, attention_geometry)?;
        let (xclbin, instructions) = image("attention-unpack")?;
        let attention_unpack =
            NpuQwen3AttentionUnpack::load(&xclbin, &instructions, attention_geometry)?;
        let swiglu = if std::env::var("HIPFIRE_QWEN3_SWIGLU").is_ok_and(|value| value == "npu") {
            let (xclbin, instructions) = image("swiglu")?;
            NpuQwen3SwiGlu::load(
                &xclbin,
                &instructions,
                padded_rows,
                weights.intermediate_size,
            )?
        } else {
            NpuQwen3SwiGlu::load_host(padded_rows, weights.intermediate_size)?
        };
        let (xclbin, instructions) = image("final-pool-l2")?;
        let final_pool = NpuQwen3FinalPoolL2::load(
            &xclbin,
            &instructions,
            geometry.sequence_bucket,
            geometry.dispatch_batch,
            weights.hidden_size,
            &weights.final_norm,
            weights.norm_epsilon,
        )?;
        let layer_count = weights.layer_count;
        Ok(Self {
            geometry,
            padded_rows,
            layer_count,
            residual_norm,
            query_projection,
            kv_projection,
            attention_output_projection,
            gate_up_projection,
            down_projection,
            headnorm_rope,
            query_pack,
            kv_pack,
            attention,
            attention_unpack,
            swiglu,
            final_pool,
            zeros_hidden: vec![0; padded_rows * weights.hidden_size],
        })
    }

    pub fn geometry(&self) -> FullEmbeddingIoGeometry {
        self.geometry
    }

    pub fn run(
        &mut self,
        padded_hidden_bf16: &[u16],
        real_token_lengths: &[u32],
    ) -> Result<Vec<Vec<f32>>, XdnaError> {
        self.run_inner(padded_hidden_bf16, real_token_lengths, None)
    }

    /// Run the encoder and retain the residual stream after every transformer
    /// layer. This is intentionally an explicit diagnostic path: normal
    /// serving pays no allocation or copy cost for the trace.
    pub fn run_with_layer_trace(
        &mut self,
        padded_hidden_bf16: &[u16],
        real_token_lengths: &[u32],
    ) -> Result<(Vec<Vec<f32>>, FullEmbeddingTrace), XdnaError> {
        let mut layer_trace = FullEmbeddingTrace {
            completed_layers_bf16: Vec::with_capacity(self.layer_count),
            stage_layer: std::env::var("HIPFIRE_QWEN3_TRACE_LAYER")
                .ok()
                .and_then(|value| value.parse().ok())
                .filter(|&layer| layer < self.layer_count)
                .unwrap_or(self.layer_count - 1),
            last_layer_stages: Vec::new(),
        };
        let embeddings = self.run_inner(
            padded_hidden_bf16,
            real_token_lengths,
            Some(&mut layer_trace),
        )?;
        Ok((embeddings, layer_trace))
    }

    fn run_inner(
        &mut self,
        padded_hidden_bf16: &[u16],
        real_token_lengths: &[u32],
        mut layer_trace: Option<&mut FullEmbeddingTrace>,
    ) -> Result<Vec<Vec<f32>>, XdnaError> {
        let trace_timings =
            std::env::var("HIPFIRE_QWEN3_STAGE_TRACE").is_ok_and(|value| value != "0");
        let mut timings = FullEmbeddingTimings::default();
        macro_rules! timed {
            ($field:ident, $expression:expr) => {{
                let started = Instant::now();
                let result = $expression;
                timings.$field += started.elapsed();
                result
            }};
        }
        if padded_hidden_bf16.len() != self.geometry.input_elements()? {
            return Err(invalid(format!(
                "full embedding encoder input has {} BF16 values; expected {}",
                padded_hidden_bf16.len(),
                self.geometry.input_elements()?
            )));
        }
        validate_lengths(
            real_token_lengths,
            self.geometry.dispatch_batch,
            self.geometry.sequence_bucket,
        )?;
        let stop_after_stage_layer = layer_trace.is_some()
            && std::env::var_os("HIPFIRE_QWEN3_TRACE_STOP_AFTER_LAYER").is_some();
        let actual_rows = self.geometry.rows();
        let hidden = self.geometry.hidden_size;
        let mut padded_hidden = vec![0u16; self.padded_rows * hidden];
        padded_hidden[..padded_hidden_bf16.len()].copy_from_slice(padded_hidden_bf16);
        let (_, mut normalized) = timed!(
            residual_norm,
            self.residual_norm
                .run_index(0, &padded_hidden, &self.zeros_hidden)
        )?;
        let mut completed = padded_hidden;
        for layer in 0..self.layer_count {
            let trace_last_layer = layer_trace
                .as_deref()
                .is_some_and(|trace| layer == trace.stage_layer);
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "normalized_input",
                hidden,
                actual_rows,
                &normalized,
            );
            let query_projection = timed!(
                qkv_projection,
                self.query_projection.run_index(layer, &normalized)
            )?;
            let key_projection = timed!(
                qkv_projection,
                self.kv_projection.run_index(2 * layer, &normalized)
            )?;
            let value = timed!(
                qkv_projection,
                self.kv_projection.run_index(2 * layer + 1, &normalized)
            )?;
            let q_width = self.query_projection.output_columns();
            let kv_width = self.kv_projection.output_columns();
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "query_projection",
                q_width,
                actual_rows,
                &query_projection,
            );
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "key_projection",
                kv_width,
                actual_rows,
                &key_projection,
            );
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "value_projection",
                kv_width,
                actual_rows,
                &value,
            );
            let (query, key) = timed!(
                headnorm_rope,
                self.headnorm_rope.run_index(
                    layer,
                    &query_projection[..actual_rows * q_width],
                    &key_projection[..actual_rows * kv_width],
                )
            )?;
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "query_headnorm_rope",
                q_width,
                actual_rows,
                &query,
            );
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "key_headnorm_rope",
                kv_width,
                actual_rows,
                &key,
            );
            let value = &value[..actual_rows * kv_width];
            let packed_query = timed!(
                attention_layout,
                self.query_pack.run(&query, real_token_lengths)
            )?;
            let packed_kv = timed!(attention_layout, self.kv_pack.run(&key, value))?;
            let packed_attention = timed!(
                attention,
                self.attention.run_packed(&packed_query, &packed_kv)
            )?;
            let attention = timed!(
                attention_layout,
                self.attention_unpack.run(&packed_attention)
            )?;
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "attention",
                q_width,
                actual_rows,
                &attention,
            );
            let mut padded_attention = vec![0u16; self.padded_rows * q_width];
            padded_attention[..attention.len()].copy_from_slice(&attention);
            let attention_delta = timed!(
                attention_output,
                self.attention_output_projection
                    .run_index(layer, &padded_attention)
            )?;
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "attention_delta",
                hidden,
                actual_rows,
                &attention_delta,
            );
            let (post_attention, ffn_input) = timed!(
                residual_norm,
                self.residual_norm.run_index(
                    self.layer_count + layer,
                    &completed,
                    &attention_delta,
                )
            )?;
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "post_attention",
                hidden,
                actual_rows,
                &post_attention,
            );
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "ffn_input",
                hidden,
                actual_rows,
                &ffn_input,
            );
            let gate = timed!(
                gate_up_projection,
                self.gate_up_projection.run_index(2 * layer, &ffn_input)
            )?;
            let up = timed!(
                gate_up_projection,
                self.gate_up_projection.run_index(2 * layer + 1, &ffn_input)
            )?;
            let intermediate = self.gate_up_projection.output_columns();
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "gate_projection",
                intermediate,
                actual_rows,
                &gate,
            );
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "up_projection",
                intermediate,
                actual_rows,
                &up,
            );
            let activated = timed!(swiglu, self.swiglu.run(&gate, &up))?;
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "swiglu",
                intermediate,
                actual_rows,
                &activated,
            );
            let down = timed!(
                down_projection,
                self.down_projection.run_index(layer, &activated)
            )?;
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "down_projection",
                hidden,
                actual_rows,
                &down,
            );
            if layer + 1 < self.layer_count {
                (completed, normalized) = timed!(
                    residual_norm,
                    self.residual_norm
                        .run_index(layer + 1, &post_attention, &down)
                )?;
            } else {
                let final_norm_index = 2 * self.layer_count;
                (completed, normalized) = timed!(
                    residual_norm,
                    self.residual_norm
                        .run_index(final_norm_index, &post_attention, &down)
                )?;
            }
            if let Some(trace) = layer_trace.as_deref_mut() {
                trace
                    .completed_layers_bf16
                    .push(completed[..actual_rows * hidden].to_vec());
            }
            capture_stage(
                &mut layer_trace,
                trace_last_layer,
                "completed",
                hidden,
                actual_rows,
                &completed,
            );
            if trace_last_layer && stop_after_stage_layer {
                break;
            }
        }
        let final_hidden = &completed[..actual_rows * hidden];
        let embeddings = timed!(
            final_pool,
            self.final_pool.run(final_hidden, real_token_lengths)
        )?;
        if trace_timings {
            timings.report();
        }
        Ok(embeddings)
    }
}

fn capture_stage(
    trace: &mut Option<&mut FullEmbeddingTrace>,
    enabled: bool,
    name: &'static str,
    columns: usize,
    rows: usize,
    values: &[u16],
) {
    if enabled {
        if let Some(trace) = trace.as_deref_mut() {
            trace.last_layer_stages.push(FullEmbeddingStageTrace {
                name,
                columns,
                token_major_bf16: values[..rows * columns].to_vec(),
            });
        }
    }
}

fn read_image(directory: &Path) -> Result<(Vec<u8>, Vec<u8>), XdnaError> {
    Ok((
        std::fs::read(directory.join("final.xclbin")).map_err(XdnaError::Open)?,
        std::fs::read(directory.join("insts.bin")).map_err(XdnaError::Open)?,
    ))
}

fn validate_lengths(lengths: &[u32], expected: usize, bucket: usize) -> Result<(), XdnaError> {
    if lengths.len() != expected {
        return Err(invalid(format!(
            "full embedding encoder received {} lengths; expected {expected}",
            lengths.len()
        )));
    }
    for (index, &length) in lengths.iter().enumerate() {
        if length == 0 || length as usize > bucket {
            return Err(invalid(format!(
                "full embedding encoder length[{index}]={length} is outside 1..={bucket}"
            )));
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_encoder_geometry_enforces_padded_row_limit() {
        let valid = FullEmbeddingIoGeometry {
            sequence_bucket: 512,
            dispatch_batch: 8,
            hidden_size: 1024,
            output_dimensions: 1024,
        };
        assert_eq!(valid.validate().unwrap(), valid);
        assert!(FullEmbeddingIoGeometry {
            dispatch_batch: 9,
            ..valid
        }
        .validate()
        .is_err());
    }
}
