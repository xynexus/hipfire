// SPDX-License-Identifier: Apache-2.0
//! GPU smoke for one source-streamed Gemma3 layer and its logical captures.

use hipfire_arch_gemma3::calibration_stream::{
    gemma3_capture_registry, inspect_gemma3_stream_source, Gemma3StreamedCalibrationLayer,
};
use hipfire_arch_gemma3::config_from_metadata_json;
use hipfire_model::ModelSource;
use hipfire_rdna::Gpu;
use hipfire_runtime::calibration::contracts::{
    CalibrationJob, CalibrationOptions, CalibrationSample, SampleSet,
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
        .ok_or("usage: load_calibration_layer <safetensors-snapshot> [layer]")?;
    let layer: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "0".into())
        .parse()?;
    let source = SafetensorsSource::open(Path::new(&path))?;
    let inspection = inspect_gemma3_stream_source(&source)?;
    let config =
        config_from_metadata_json(source.metadata_json()).ok_or("invalid Gemma3 config")?;
    if layer >= inspection.num_layers {
        return Err(format!("layer {layer} is outside 0..{}", inspection.num_layers).into());
    }
    let plan = TensorLoadPlan::build(&source, inspection.tensor_requests.clone())?;
    let mut ledger = ReadLedger::new(&plan);
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
    let job = CalibrationJob::new("source", "tokenizer", samples, options)?;
    let mut gpu = Gpu::init()?;
    let mut streamed = {
        let mut reader = PlannedTensorReader::new(&source, &mut ledger, TensorOwner::Layer(layer));
        Gemma3StreamedCalibrationLayer::load(&mut reader, &mut gpu, &config, layer, &job)?
    };
    let batches = MicrobatchPlanner::new(MicrobatchGeometry {
        sequence_batch: 2,
        time_tile: 1,
        row_budget: 2,
    })?
    .plan(&job.samples);
    let capture = gemma3_capture_registry(&config)?;
    for (batch_index, batch) in batches.iter().enumerate() {
        let input = (0..batch.rows.len() * config.hidden_size)
            .map(|index| ((index % 31) as f32 - 15.0) * 1.0e-4)
            .collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        streamed.execute(&mut gpu, batch, &input, &mut output, &capture)?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "Gemma3 layer {layer} batch {batch_index} produced non-finite output"
            )
            .into());
        }
        println!(
            "executed layer={layer} batch={batch_index} rows={} max_abs={:.6}",
            batch.rows.len(),
            output.iter().copied().map(f32::abs).fold(0.0, f32::max)
        );
    }
    let path = std::env::temp_dir().join(format!(
        "gemma3-stream-layer-{layer}-{}.calib.hfq",
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
    streamed.finish(&mut gpu)?;
    Ok(())
}
