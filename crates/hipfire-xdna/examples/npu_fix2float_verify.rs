#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::NpuKernel;

    let cache = std::env::args()
        .nth(1)
        .ok_or("usage: npu_fix2float_verify CACHE")?;
    let kernel = NpuKernel::load(
        &std::fs::read(format!("{cache}/final.xclbin"))?,
        &std::fs::read(format!("{cache}/insts.bin"))?,
    )?;
    let values: [i32; 16] = [
        0,
        1,
        -1,
        2,
        -2,
        127,
        -127,
        32_767,
        -32_768,
        65_535,
        -65_535,
        1_000_000,
        -1_000_000,
        i32::MAX,
        i32::MIN + 1,
        42,
    ];
    let scales: [f32; 16] = [
        1.0,
        0.5,
        -0.25,
        2.0,
        1.0 / 3.0,
        0.125,
        -0.0625,
        0.001,
        0.0001,
        1.5,
        -2.0,
        0.000_003,
        -0.000_003,
        0.25,
        -0.25,
        42.0,
    ];
    let mut input = kernel.alloc_arg(values.len() * 4)?;
    let mut scale_input = kernel.alloc_arg(scales.len() * 4)?;
    let mut f32_output = kernel.alloc_arg(values.len() * 4)?;
    let mut scaled_output = kernel.alloc_arg(values.len() * 4)?;
    input.as_mut_slice().copy_from_slice(as_bytes(&values));
    scale_input
        .as_mut_slice()
        .copy_from_slice(as_bytes_f32(&scales));
    f32_output.as_mut_slice().fill(0);
    scaled_output.as_mut_slice().fill(0);
    kernel.dispatch(&[&input, &scale_input, &f32_output, &scaled_output])?;
    let got_f32 = unsafe {
        std::slice::from_raw_parts(f32_output.as_slice().as_ptr().cast::<f32>(), values.len())
    };
    let got_scaled = unsafe {
        std::slice::from_raw_parts(
            scaled_output.as_slice().as_ptr().cast::<f32>(),
            values.len(),
        )
    };
    let mut mismatches = 0;
    for (index, &value) in values.iter().enumerate() {
        let expected = value as f32;
        let scaled_expected = expected * scales[index];
        let scaled_error = (got_scaled[index] - scaled_expected).abs();
        let scaled_tolerance = scaled_expected.abs().max(1.0) * 1e-6;
        if got_f32[index].to_bits() != expected.to_bits() || scaled_error > scaled_tolerance {
            mismatches += 1;
            println!(
                "lane={index} input={value} f32={} expected={expected} scaled={} expected_scaled={scaled_expected}",
                got_f32[index], got_scaled[index]
            );
        }
    }
    println!("fix2float AIE2P mismatches={mismatches}");
    if mismatches != 0 {
        return Err("AIE2P fix2float parity failed".into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn as_bytes(values: &[i32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(target_os = "linux")]
fn as_bytes_f32(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {}
