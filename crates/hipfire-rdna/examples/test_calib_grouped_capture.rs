//! GPU/CPU parity for family-neutral grouped-expert calibration staging.
//!
//! Exercises both source layouts used by routed MoE capture:
//! - gate/up: `flat_slot / K_TOP` selects the original token row;
//! - down: `flat_slot` selects the flattened routed activation row.
//!
//! The two gather calls fill one persistent tile, matching the production
//! carry-across-microbatches contract, before a single sum-of-squares reduction.

use hipfire_rdna::{DType, Gpu, GpuTensor};

fn upload_i32(gpu: &Gpu, values: &[i32]) -> GpuTensor {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    gpu.upload_raw(&bytes, &[bytes.len()])
        .expect("upload sorted routed indices")
}

fn expected_sumsq(source: &[f32], rows: &[usize], width: usize) -> Vec<f32> {
    let mut expected = vec![0.0f32; width];
    for &row in rows {
        for column in 0..width {
            let value = source[row * width + column];
            expected[column] += value * value;
        }
    }
    expected
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 1.0e-6,
        "{label} max absolute error {max_error}: actual={actual:?} expected={expected:?}"
    );
}

fn capture_case(
    gpu: &mut Gpu,
    label: &str,
    source: &[f32],
    width: usize,
    sorted: &GpuTensor,
    x_row_div: usize,
    expected_rows: &[usize],
) {
    let source_gpu = gpu
        .upload_f32(source, &[source.len() / width, width])
        .expect("upload calibration source");
    let staging = gpu
        .zeros(&[expected_rows.len(), width], DType::F32)
        .expect("allocate calibration staging tile");
    let imatrix = gpu
        .zeros(&[width], DType::F32)
        .expect("allocate calibration imatrix");

    // The expert's first two routes came from one model microbatch. Padding in
    // the grouped layout separates the final route, which arrives in the next.
    gpu.calib_gather_rows_f32(&source_gpu, sorted, &staging, 0, 0, 2, width, x_row_div)
        .expect("gather first partial tile");
    gpu.calib_gather_rows_f32(&source_gpu, sorted, &staging, 4, 2, 1, width, x_row_div)
        .expect("complete staged tile");
    gpu.calib_sumsq_reduce_f32(&staging, &imatrix, expected_rows.len(), width)
        .expect("reduce completed tile");
    gpu.hip
        .device_synchronize()
        .expect("synchronize parity run");

    let actual = gpu.download_f32(&imatrix).expect("download imatrix");
    let expected = expected_sumsq(source, expected_rows, width);
    assert_close(label, &actual, &expected);

    gpu.free_tensor(source_gpu).expect("free source");
    gpu.free_tensor(staging).expect("free staging");
    gpu.free_tensor(imatrix).expect("free imatrix");
}

fn main() {
    const K_TOP: usize = 10;
    const WIDTH: usize = 7;
    let mut gpu = Gpu::init().expect("initialize GPU");
    let sorted = upload_i32(&gpu, &[9, 19, -1, -1, 29]);

    let gate_source = (0..3 * WIDTH)
        .map(|index| index as f32 * 0.125 - 0.75)
        .collect::<Vec<_>>();
    capture_case(
        &mut gpu,
        "gate-up K=10",
        &gate_source,
        WIDTH,
        &sorted,
        K_TOP,
        &[0, 1, 2],
    );

    let down_source = (0..30 * WIDTH)
        .map(|index| (index % 23) as f32 * 0.0625 - 0.5)
        .collect::<Vec<_>>();
    capture_case(
        &mut gpu,
        "down K=10",
        &down_source,
        WIDTH,
        &sorted,
        1,
        &[9, 19, 29],
    );

    gpu.free_tensor(sorted).expect("free sorted indices");
    println!("grouped expert calibration gather/reduction parity: PASS");
}
