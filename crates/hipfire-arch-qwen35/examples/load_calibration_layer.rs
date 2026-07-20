// SPDX-License-Identifier: Apache-2.0
//! GPU smoke for loading exactly one BF16/F16 Qwen3.5 source layer.

use hipfire_arch_qwen35::calibration_stream::{
    free_qwen35_streamed_layer, inspect_qwen35_stream_source, load_qwen35_streamed_layer,
    qwen35_capture_registry, Qwen35StreamedCalibrationLayer,
};
use hipfire_arch_qwen35::qwen35::{config_from_safetensors, LayerWeights};
use hipfire_rdna::Gpu;
use hipfire_runtime::calibration::contracts::{
    CalibrationJob, CalibrationOptions, CalibrationSample, CaptureRegistry, ExpertCaptureQuota,
    ExpertCoveragePolicy, ExpertSamplingPolicy, SampleSet,
};
use hipfire_runtime::calibration::schedule::{MicrobatchGeometry, MicrobatchPlanner};
use hipfire_runtime::calibration::source::{
    PlannedTensorReader, ReadLedger, TensorLoadPlan, TensorOwner,
};
use hipfire_runtime::calibration::stream::CalibrationLayer;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: load_calibration_layer <safetensors-directory> [layer]")?;
    let layer: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "0".into())
        .parse()?;
    let mode = std::env::args().nth(3).unwrap_or_default();
    let execute = matches!(mode.as_str(), "execute" | "capture");
    let source = SafetensorsSource::open(Path::new(&path))?;
    let inspection = inspect_qwen35_stream_source(&source)?;
    let config = config_from_safetensors(&source).ok_or("invalid Qwen3.5 config")?;
    if layer >= inspection.num_layers {
        return Err(format!("layer {layer} is outside 0..{}", inspection.num_layers).into());
    }
    let plan = TensorLoadPlan::build(&source, inspection.tensor_requests.clone())?;
    let mut ledger = ReadLedger::new(&plan);
    let mut gpu = Gpu::init()?;
    if execute {
        let samples = SampleSet::new(
            vec![
                CalibrationSample::new("a", vec![1, 2], "smoke"),
                CalibrationSample::new("b", vec![3, 4], "smoke"),
            ],
            2,
            1,
        )?;
        let mut options = CalibrationOptions::default();
        options.max_rows = 2;
        options.sequence_batch = Some(2);
        options.time_tile = Some(1);
        options.kldref = false;
        if mode == "capture" {
            options.expert_quota = ExpertCaptureQuota {
                min_rows: 1,
                target_rows: 16,
                tile_rows: 16,
                sampling: ExpertSamplingPolicy::DeterministicFirst { seed: 1 },
            };
            options.expert_coverage_policy = ExpertCoveragePolicy::PreserveUndercovered;
        }
        let job = CalibrationJob::new("source", "tokenizer", samples, options)?;
        let mut streamed = {
            let mut reader =
                PlannedTensorReader::new(&source, &mut ledger, TensorOwner::Layer(layer));
            Qwen35StreamedCalibrationLayer::load(&mut reader, &mut gpu, &config, layer, &job)?
        };
        let batches = MicrobatchPlanner::new(MicrobatchGeometry {
            sequence_batch: 2,
            time_tile: 1,
            row_budget: 2,
        })?
        .plan(&job.samples);
        let capture = if mode == "capture" {
            qwen35_capture_registry(&config, job.options.expert_quota)?
        } else {
            CaptureRegistry::default()
        };
        for (batch_index, batch) in batches.iter().enumerate() {
            let input = (0..batch.rows.len() * config.dim)
                .map(|index| ((index % 31) as f32 - 15.0) * 1.0e-4)
                .collect::<Vec<_>>();
            let mut output = vec![0.0; input.len()];
            streamed.execute(&mut gpu, batch, &input, &mut output, &capture)?;
            if output.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "layer {layer} batch {batch_index} produced non-finite output"
                )
                .into());
            }
            println!(
                "executed layer={layer} batch={batch_index} rows={} max_abs={:.6}",
                batch.rows.len(),
                output.iter().copied().map(f32::abs).fold(0.0, f32::max)
            );
        }
        if mode == "capture" {
            let path = std::env::temp_dir().join(format!(
                "qwen35-stream-layer-{layer}-{}.calib.hfq",
                std::process::id()
            ));
            let summary = streamed.write_capture_part(
                &mut gpu,
                &path,
                inspection.arch_id,
                "{\"artifact_kind\":\"calibration-smoke\"}",
            )?;
            println!(
                "captured layer={layer} tensors={} consistency={:.3e} path={}",
                summary.descriptors.len(),
                summary.max_consistency,
                path.display()
            );
            std::fs::remove_file(path)?;
        }
        streamed.finish(&mut gpu)?;
        return Ok(());
    }
    let weights = {
        let mut reader = PlannedTensorReader::new(&source, &mut ledger, TensorOwner::Layer(layer));
        load_qwen35_streamed_layer(&mut reader, &mut gpu, &config, layer)?
    };
    let (kind, gate_up_dtype, down_dtype, routed_experts) = match &weights {
        LayerWeights::DeltaNetMoe(layer) => (
            "linear_attention",
            layer.ffn.expert_gate_up_dtype,
            layer.ffn.expert_down_dtype,
            layer.ffn.experts.len(),
        ),
        LayerWeights::FullAttnMoe(layer) => (
            "full_attention",
            layer.ffn.expert_gate_up_dtype,
            layer.ffn.expert_down_dtype,
            layer.ffn.experts.len(),
        ),
        _ => return Err("expected grouped-MoE layer".into()),
    };
    println!(
        "loaded layer={layer} kind={kind} experts={routed_experts} gate_up={gate_up_dtype:?} down={down_dtype:?} arch={}",
        gpu.arch
    );
    free_qwen35_streamed_layer(&mut gpu, weights);
    Ok(())
}
