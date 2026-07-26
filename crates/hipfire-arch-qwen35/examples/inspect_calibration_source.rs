// SPDX-License-Identifier: Apache-2.0
//! Read-only validation of a Qwen3.5 streamed-calibration source plan.

use hipfire_arch_qwen35::calibration_stream::inspect_qwen35_stream_source;
use hipfire_runtime::calibration::source::{TensorLoadPlan, TensorOwner};
use hipfire_runtime::safetensors_source::SafetensorsSource;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: inspect_calibration_source <safetensors-directory>")?;
    let source = SafetensorsSource::open(Path::new(&path))?;
    let model = inspect_qwen35_stream_source(&source)?;
    let plan = TensorLoadPlan::build(&source, model.tensor_requests.clone())?;
    println!(
        "family={} arch={} layers={} hidden={} vocab={} tensors={} unique_bytes={}",
        model.family,
        model.arch_id,
        model.num_layers,
        model.hidden_width,
        model.vocab_size,
        plan.entries().len(),
        plan.unique_source_bytes(),
    );
    for layer in 0..model.num_layers {
        println!(
            "layer={layer} tensors={} bytes={}",
            plan.entries()
                .iter()
                .filter(|entry| entry.owner == TensorOwner::Layer(layer))
                .count(),
            plan.bytes_for(TensorOwner::Layer(layer)),
        );
    }
    Ok(())
}
