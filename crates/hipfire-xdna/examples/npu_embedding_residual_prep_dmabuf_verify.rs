//! Verify R48 residual preparation with AMDGPU-owned dma-buf arguments.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_rdna::Gpu;
    use hipfire_xdna::{NpuEmbeddingResidualPrep, XdnaDevice};

    const K: usize = 768;
    const ROW_BYTES: usize = 2 * K * size_of::<u16>();
    const RECORD_BYTES: usize = 16_384;

    let cache = std::env::args()
        .nth(1)
        .ok_or("usage: npu_embedding_residual_prep_dmabuf_verify CACHE")?;
    let mut completed = vec![0u8; NpuEmbeddingResidualPrep::completed_bytes()];
    for token in 0..256 {
        for hidden in 0..K {
            let value = f32_to_bf16_bits((token as f32 - 128.0) * 0.125 + hidden as f32 * 0.003);
            let offset = token * ROW_BYTES + hidden * size_of::<u16>();
            completed[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            let low = f32_to_bf16_bits((hidden as f32 % 7.0 - 3.0) * 0.0005);
            let low_offset = offset + K * size_of::<u16>();
            completed[low_offset..low_offset + 2].copy_from_slice(&low.to_le_bytes());
        }
    }

    let gpu = Gpu::init()?;
    let completed_shared = gpu.alloc_shared_gtt(NpuEmbeddingResidualPrep::completed_bytes())?;
    let output_shared = gpu.alloc_shared_gtt(NpuEmbeddingResidualPrep::output_bytes())?;
    let mut prep = NpuEmbeddingResidualPrep::load_cached(&cache)?;
    prep.attach_shared(
        completed_shared.dmabuf_fd(),
        completed_shared.len(),
        output_shared.dmabuf_fd(),
        output_shared.len(),
    )?;
    let observer_device = XdnaDevice::open_default()?;
    let observer =
        observer_device.import_dmabuf(output_shared.dmabuf_fd(), output_shared.len(), true)?;
    prep.write_completed_bf16x2(&completed)?;
    prep.fill_output(0)?;
    prep.run_shared()?;

    let mismatch_count = |output: &[u8]| {
        let records = &output[NpuEmbeddingResidualPrep::activation_bytes()..];
        let mut mismatches = 0usize;
        for col in 0..8 {
            for core_row in 0..4 {
                let wave = col / 4;
                let active_col = core_row;
                let source_core_row = col % 4;
                let record = ((wave * 4 + active_col) * 4 + source_core_row) * RECORD_BYTES;
                let token_base = col * 32 + core_row * 8;
                for row in 0..8 {
                    let source = (token_base + row) * ROW_BYTES;
                    let target = record + row * K * 2;
                    for hidden in 0..K {
                        let high = u16::from_le_bytes(
                            completed[source + hidden * 2..source + hidden * 2 + 2]
                                .try_into()
                                .unwrap(),
                        );
                        let low_offset = source + K * 2 + hidden * 2;
                        let low = u16::from_le_bytes(
                            completed[low_offset..low_offset + 2].try_into().unwrap(),
                        );
                        let expected =
                            f32_to_bf16_bits(bf16_bits_to_f32(high) + bf16_bits_to_f32(low));
                        let got_offset = target + hidden * 2;
                        let got = u16::from_le_bytes(
                            records[got_offset..got_offset + 2].try_into().unwrap(),
                        );
                        mismatches += usize::from(got != expected);
                    }
                }
            }
        }
        mismatches
    };
    let owner_mismatches = mismatch_count(output_shared.as_slice());
    let producer_mismatches = mismatch_count(prep.output());
    let observer_before = mismatch_count(observer.as_slice());
    observer_device.sync_bo(
        observer.handle(),
        hipfire_xdna::submit::SYNC_DIRECT_TO_DEVICE,
        observer.len(),
    )?;
    let observer_after = mismatch_count(observer.as_slice());
    if owner_mismatches != 0
        || producer_mismatches != 0
        || observer_before != 0
        || observer_after != 0
    {
        return Err(format!(
            "dma-buf residual prep mismatches owner={owner_mismatches} producer={producer_mismatches} observer-before={observer_before} observer-after={observer_after}"
        )
        .into());
    }
    prep.fill_output(0)?;
    prep.write_bootstrap_bf16x2(&completed)?;
    prep.run_bootstrap()?;
    let bootstrap_owner = mismatch_count(output_shared.as_slice());
    let bootstrap_producer = mismatch_count(prep.output());
    let bootstrap_observer = mismatch_count(observer.as_slice());
    if bootstrap_owner != 0 || bootstrap_producer != 0 || bootstrap_observer != 0 {
        return Err(format!(
            "bootstrap-to-dma-buf mismatches owner={bootstrap_owner} producer={bootstrap_producer} observer={bootstrap_observer}"
        )
        .into());
    }
    println!(
        "AMDGPU dma-buf residual prep: imported and bootstrap inputs exact across all mappings"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_residual_prep_dmabuf_verify is Linux-only");
}
