//! End-to-end GPU smoke for the family-neutral routed-expert capture callback.
//!
//! Builds one synthetic E=2/K=2 routed batch, invokes the same gate/up and down
//! callbacks used by the generic MoE executor, and verifies telemetry,
//! quota-aligned reductions, and logical artifact descriptors.

use hipfire_dispatch::families::moe::{
    MoePrefillCapture, MoePrefillCaptureBatch, MoePrefillCapturePoint,
};
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::calibration::contracts::{
    CaptureDescriptor, CaptureId, CapturePolicy, CaptureRegistry, ExpertCaptureQuota,
    ExpertCaptureRole, ExpertTelemetry, ProjectionRole,
};
use hipfire_runtime::calibration::expert_capture::GroupedMoeCalibrationCapture;
use hipfire_runtime::moe::grouped::GroupedMoeRoutingPlan;
use std::sync::Arc;

fn upload_i32(gpu: &Gpu, values: &[i32]) -> GpuTensor {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    gpu.upload_raw(&bytes, &[bytes.len()])
        .expect("upload i32 routing tensor")
}

fn registry(quota: ExpertCaptureQuota) -> CaptureRegistry {
    let mut registry = CaptureRegistry::default();
    for expert in 0..2 {
        for (role, width, output) in [
            (ProjectionRole::GateUpInput, 3, "gate_up_proj"),
            (ProjectionRole::DownInput, 4, "down_proj"),
        ] {
            registry
                .register(CaptureDescriptor {
                    id: CaptureId::new(0, role, Some(expert)),
                    output_names: vec![format!("layers.0.experts.{expert}.{output}")],
                    input_width: width,
                    policy: CapturePolicy::ImatrixOnly,
                    layer: 0,
                    role,
                    expert: Some(expert),
                    expert_quota: Some(quota),
                })
                .unwrap();
        }
    }
    registry
}

fn main() {
    let mut gpu = Gpu::init().expect("initialize GPU");
    let quota = ExpertCaptureQuota {
        min_rows: 2,
        target_rows: 2,
        tile_rows: 2,
        ..ExpertCaptureQuota::default()
    };
    let telemetry = ExpertTelemetry::new(1, 2, 2, quota, 8).unwrap();
    let capture = GroupedMoeCalibrationCapture::new(Arc::new(registry(quota)), telemetry).unwrap();

    let indices = vec![0usize, 1, 1, 0];
    let routing = GroupedMoeRoutingPlan::build(&indices, 2, 2, 2).unwrap();
    let topk_indices = upload_i32(
        &gpu,
        &indices
            .iter()
            .map(|&expert| expert as i32)
            .collect::<Vec<_>>(),
    );
    let topk_weights = gpu
        .upload_f32(&[0.7, 0.3, 0.6, 0.4], &[4])
        .expect("upload route weights");
    let sorted = upload_i32(&gpu, &routing.sorted_slot_index);
    let gate_source = gpu
        .upload_f32(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0], &[2, 3])
        .expect("upload gate source");
    let down_source = gpu
        .upload_f32(
            &[
                1.0, 0.0, 2.0, 0.5, // token 0 rank 0
                0.5, 1.5, 0.0, 2.0, // token 0 rank 1
                2.0, 1.0, 0.5, 0.0, // token 1 rank 0
                1.0, 1.0, 1.0, 1.0, // token 1 rank 1
            ],
            &[4, 4],
        )
        .expect("upload down source");

    capture
        .capture(
            &mut gpu,
            &MoePrefillCaptureBatch {
                layer: 0,
                point: MoePrefillCapturePoint::GateUpInput,
                source: &gate_source,
                source_width: 3,
                source_row_div: 2,
                topk_indices: &topk_indices,
                topk_weights: &topk_weights,
                sorted_slot_index: &sorted,
                batch_size: 2,
                k_top: 2,
                num_experts: 2,
            },
        )
        .expect("capture gate/up rows");
    capture
        .capture(
            &mut gpu,
            &MoePrefillCaptureBatch {
                layer: 0,
                point: MoePrefillCapturePoint::DownInput,
                source: &down_source,
                source_width: 4,
                source_row_div: 1,
                topk_indices: &topk_indices,
                topk_weights: &topk_weights,
                sorted_slot_index: &sorted,
                batch_size: 2,
                k_top: 2,
                num_experts: 2,
            },
        )
        .expect("capture down rows");
    gpu.hip.device_synchronize().expect("synchronize capture");

    assert!(capture.finalize().unwrap().is_empty());
    let telemetry = capture.telemetry_snapshot();
    for expert in 0..2 {
        for role in [ExpertCaptureRole::GateUpInput, ExpertCaptureRole::DownInput] {
            let stats = telemetry.capture_stats(0, expert, role);
            assert_eq!((stats.seen_rows, stats.admitted_rows), (2, 2));
            assert_eq!(stats.quota_skipped_rows, 0);
        }
    }
    let collector = capture.collector();
    assert_eq!(collector.accumulator_len(), 4);
    assert!(collector
        .tensor_descriptors()
        .iter()
        .all(|descriptor| !descriptor.has_hessian && descriptor.n_tokens == 2));

    collector.free_gpu(&mut gpu);
    for tensor in [topk_indices, topk_weights, sorted, gate_source, down_source] {
        gpu.free_tensor(tensor).expect("free smoke tensor");
    }
    println!("family-neutral grouped calibration callback: PASS");
}
