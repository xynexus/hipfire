//! GPU parity check for mixed low-bit + BF16/F16 routed-expert bucket dispatch.
//!
//! Run under the repository GPU lock:
//!   hipfire lock acquire -- cargo run --release -p hipfire-arch-qwen35 \
//!     --example test_mixed_moe_prefill

use hipfire_rdna::{DType, Gpu, OwnedTensor};
use hipfire_runtime::hfq_modules::{HfqModuleKind, HfqModuleRecord, HfqModuleTensor};
use hipfire_runtime::moe::grouped::build_paged_expert_buckets;
use hipfire_runtime::weight_pager::{ExpertModuleKey, PagerConfig, PreadH2DTransport, WeightPager};

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    let mut rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    let mut exponent = exponent as u16;
    if rounded & 0x80_0000 != 0 {
        rounded = 0;
        exponent += 1;
    }
    sign | (exponent << 10) | ((rounded >> 13) as u16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let bits = if exponent == 0 {
        sign
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

fn round_f16(value: f32) -> f32 {
    f16_to_f32(f32_to_f16_bits(value))
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

fn upload(gpu: &mut Gpu, bytes: &[u8]) -> OwnedTensor {
    let tensor = gpu
        .alloc_owned(&[bytes.len()], DType::Raw)
        .expect("allocate upload tensor");
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("upload tensor");
    tensor
}

fn upload_f32(gpu: &mut Gpu, values: &[f32]) -> OwnedTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * size_of::<f32>(),
        )
    };
    upload(gpu, bytes)
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> OwnedTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * size_of::<i32>(),
        )
    };
    upload(gpu, bytes)
}

fn upload_u64(gpu: &mut Gpu, values: &[u64]) -> OwnedTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * size_of::<u64>(),
        )
    };
    upload(gpu, bytes)
}

fn mq6_constant_rows(m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % 256, 0);
    let row_bytes = (k / 256) * 200;
    let mut bytes = vec![0u8; m * row_bytes];
    let mut decoded = vec![0.0f32; m * k];
    let scale = 0.01f32;
    let scale_h = round_f16(scale);
    for row in 0..m {
        let q = (row % 13 + 1) as u32;
        let value = round_f16(scale_h * q as f32);
        decoded[row * k..(row + 1) * k].fill(value);
        for group in 0..k / 256 {
            let offset = row * row_bytes + group * 200;
            bytes[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
            bytes[offset + 4..offset + 8].copy_from_slice(&0.0f32.to_le_bytes());
            let packed = q | (q << 6) | (q << 12) | (q << 18);
            for four in 0..64 {
                let out = offset + 8 + four * 3;
                bytes[out] = packed as u8;
                bytes[out + 1] = (packed >> 8) as u8;
                bytes[out + 2] = (packed >> 16) as u8;
            }
        }
    }
    (bytes, decoded)
}

fn oq4_constant_rows(m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % 256, 0);
    let row_bytes = (k / 256) * hipfire_runtime::oq_moe::OQ4_MOE_BLOCK_BYTES;
    let mut bytes = vec![0u8; m * row_bytes];
    let mut decoded = vec![0.0f32; m * k];
    let scale = 0.01f32;
    let scale_h = round_f16(scale);
    for row in 0..m {
        let q = (row % 7 + 1) as i8 * if row % 2 == 0 { 1 } else { -1 };
        let value = round_f16(scale_h * q as f32);
        decoded[row * k..(row + 1) * k].fill(value);
        let nibble = (q as u8) & 0x0f;
        let packed = nibble | (nibble << 4);
        for group in 0..k / 256 {
            let offset = row * row_bytes + group * hipfire_runtime::oq_moe::OQ4_MOE_BLOCK_BYTES;
            bytes[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
            bytes[offset + 4..offset + hipfire_runtime::oq_moe::OQ4_MOE_BLOCK_BYTES].fill(packed);
        }
    }
    (bytes, decoded)
}

fn oq4_canonical_constant_rows(m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % 256, 0);
    let row_bytes = (k / 256) * 130;
    let mut bytes = vec![0u8; m * row_bytes];
    let mut decoded = vec![0.0f32; m * k];
    let scale_bits = f32_to_f16_bits(0.01);
    let scale = f16_to_f32(scale_bits);
    for row in 0..m {
        let q = (row % 7 + 1) as i8 * if row % 2 == 0 { 1 } else { -1 };
        decoded[row * k..(row + 1) * k].fill(round_f16(scale * q as f32));
        let nibble = (q as u8) & 0x0f;
        let packed = nibble | (nibble << 4);
        for group in 0..k / 256 {
            let offset = row * row_bytes + group * 130;
            bytes[offset..offset + 2].copy_from_slice(&scale_bits.to_le_bytes());
            bytes[offset + 2..offset + 130].fill(packed);
        }
    }
    (bytes, decoded)
}

fn raw_constant_rows(m: usize, k: usize, dtype: DType) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(m * k * 2);
    let mut decoded = Vec::with_capacity(m * k);
    for row in 0..m {
        let value = 0.0025 * (row + 1) as f32;
        let bits = match dtype {
            DType::F16 => f32_to_f16_bits(value),
            DType::BF16 => {
                (value
                    .to_bits()
                    .wrapping_add(0x7fff + ((value.to_bits() >> 16) & 1))
                    >> 16) as u16
            }
            _ => unreachable!(),
        };
        let rounded = if dtype == DType::F16 {
            f16_to_f32(bits)
        } else {
            f32::from_bits((bits as u32) << 16)
        };
        for _ in 0..k {
            bytes.extend_from_slice(&bits.to_le_bytes());
            decoded.push(rounded);
        }
    }
    (bytes, decoded)
}

fn download_f32(gpu: &Gpu, tensor: &OwnedTensor, len: usize) -> Vec<f32> {
    let mut values = vec![0.0f32; len];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            values.len() * size_of::<f32>(),
        )
    };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("download tensor");
    values
}

fn run_case(gpu: &mut Gpu, quant_dtype: DType, full_dtype: DType, k_top: usize) {
    let (rows, experts, mi, k) = (2usize, k_top, 16usize, 256usize);
    let m = 2 * mi;
    let routes = (0..rows * k_top).map(|slot| slot % 2).collect::<Vec<_>>();
    let buckets = build_paged_expert_buckets(&routes, rows, k_top, experts).unwrap();

    let (quant_bytes, quant_weights) = match quant_dtype {
        DType::MQ6G256 => mq6_constant_rows(m, k),
        DType::Oq4G256 => oq4_constant_rows(m, k),
        _ => unreachable!(),
    };
    let (raw_bytes, raw_weights) = raw_constant_rows(m, k, full_dtype);
    let quant = upload(gpu, &quant_bytes);
    let raw = upload(gpu, &raw_bytes);
    let mut expert_ptrs = vec![0u64; experts];
    expert_ptrs[0] = quant.buf.as_ptr() as u64;
    expert_ptrs[1] = raw.buf.as_ptr() as u64;
    let ptrs = upload_u64(gpu, &expert_ptrs);

    let x_rot_host = (0..rows * k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    let x_norm_host = (0..rows * k)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.0234375)
        .collect::<Vec<_>>();
    let x_rot = upload_f32(gpu, &x_rot_host);
    let x_norm = upload_f32(gpu, &x_norm_host);
    let gate = gpu.zeros_owned(&[rows * k_top * mi], DType::F32).unwrap();
    let up = gpu.zeros_owned(&[rows * k_top * mi], DType::F32).unwrap();
    let grouped = gpu.zeros_owned(&[16 * m], DType::F32).unwrap();

    for bucket in &buckets {
        let sorted = upload_i32(gpu, &bucket.sorted_slot_index);
        let tile_ids = upload_i32(gpu, &bucket.expert_tile_ids);
        let dtype = if bucket.expert == 0 {
            quant_dtype
        } else {
            full_dtype
        };
        let x = if bucket.expert == 0 { &x_rot } else { &x_norm };
        hipfire_dispatch::pipeline::run_grouped_moe_gemm(
            gpu,
            dtype,
            &ptrs,
            &tile_ids,
            &sorted,
            x,
            &grouped,
            m,
            k,
            k_top,
            bucket.m_total,
            rows,
            false,
            false,
            !hipfire_runtime::oq_moe::moe_expert_blocks_repacked(),
        )
        .unwrap();
        gpu.moe_gate_up_unscatter_k8(&grouped, &sorted, &gate, &up, mi, k_top, bucket.m_total)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let got_gate = download_f32(gpu, &gate, rows * k_top * mi);
    let got_up = download_f32(gpu, &up, rows * k_top * mi);
    let round_full = |value| {
        if full_dtype == DType::F16 {
            round_f16(value)
        } else {
            round_bf16(value)
        }
    };
    let mut max_abs = 0.0f32;
    for flat in 0..rows * k_top {
        let expert = routes[flat];
        let token = flat / k_top;
        let (weights, input) = if expert == 0 {
            (
                &quant_weights,
                x_rot_host.iter().map(|&x| round_f16(x)).collect::<Vec<_>>(),
            )
        } else {
            (
                &raw_weights,
                x_norm_host
                    .iter()
                    .map(|&x| round_full(x))
                    .collect::<Vec<_>>(),
            )
        };
        for out_row in 0..m {
            let expected = (0..k)
                .map(|column| weights[out_row * k + column] * input[token * k + column])
                .sum::<f32>();
            let got = if out_row < mi {
                got_gate[flat * mi + out_row]
            } else {
                got_up[flat * mi + out_row - mi]
            };
            max_abs = max_abs.max((got - expected).abs());
        }
    }
    let tolerance = if full_dtype == DType::BF16 {
        0.08
    } else {
        0.03
    };
    assert!(
        max_abs <= tolerance,
        "mixed {quant_dtype:?}+{full_dtype:?} K={k_top} max_abs={max_abs:.6e} exceeds {tolerance}"
    );
    println!("mixed {quant_dtype:?}+{full_dtype:?} K={k_top}: max_abs={max_abs:.6e} PASS");
}

fn run_paged_oq4_case(gpu: &mut Gpu) {
    let (rows, experts, k_top, m, k) = (2usize, 8usize, 8usize, 32usize, 256usize);
    let (gate_bytes, weights) = oq4_canonical_constant_rows(m, k);
    let (down_bytes, _) = oq4_canonical_constant_rows(m, k);
    let down_rel = gate_bytes.len().div_ceil(32) * 32;
    let mut module_bytes = gate_bytes.clone();
    module_bytes.resize(down_rel, 0);
    module_bytes.extend_from_slice(&down_bytes);
    let path = std::env::temp_dir().join(format!(
        "hipfire-oq4-expert-pager-{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, &module_bytes).expect("write paged OQ4 fixture");

    let module = HfqModuleRecord {
        module_id: "layers.0.experts.0".to_string(),
        kind: HfqModuleKind::RoutedExpert,
        layer: Some(0),
        expert: Some(0),
        placement_policy: Some("lazy_lru".to_string()),
        data_offset: 0,
        data_size: module_bytes.len(),
        tensors: vec![
            HfqModuleTensor {
                name: "model.layers.0.mlp.experts.0.gate_up_proj.weight".to_string(),
                quant_type: 34,
                shape: vec![m as u32, k as u32],
                group_size: 256,
                rel_offset: 0,
                data_size: gate_bytes.len(),
            },
            HfqModuleTensor {
                name: "model.layers.0.mlp.experts.0.down_proj.weight".to_string(),
                quant_type: 34,
                shape: vec![m as u32, k as u32],
                group_size: 256,
                rel_offset: down_rel,
                data_size: down_bytes.len(),
            },
        ],
    };
    let transport = PreadH2DTransport::open(&path).expect("open paged OQ4 fixture");
    let mut pager = WeightPager::new(Box::new(transport), PagerConfig::default());
    pager
        .register_expert_module(module)
        .expect("register OQ4 expert module");
    pager
        .ensure_expert_module_resident(
            ExpertModuleKey {
                layer: 0,
                expert: 0,
            },
            gpu,
        )
        .expect("page and repack canonical OQ4 module");
    assert_eq!(
        pager.module_stats().resident_module_bytes,
        (2 * m * hipfire_runtime::oq_moe::OQ4_MOE_BLOCK_BYTES) as u64
    );

    let ptrs = upload_u64(gpu, &[0; 8]);
    let down_ptrs = upload_u64(gpu, &[0; 8]);
    pager
        .patch_expert_module_ptr_table(0, &[0], &ptrs, &down_ptrs, gpu)
        .expect("patch repacked OQ4 expert pointers");
    let routes = vec![0usize; rows * k_top];
    let bucket = &build_paged_expert_buckets(&routes, rows, k_top, experts).unwrap()[0];
    let sorted = upload_i32(gpu, &bucket.sorted_slot_index);
    let tile_ids = upload_i32(gpu, &bucket.expert_tile_ids);
    let x_host = (0..rows * k)
        .map(|index| ((index % 19) as f32 - 9.0) * 0.03125)
        .collect::<Vec<_>>();
    let x = upload_f32(gpu, &x_host);
    let grouped = gpu.zeros_owned(&[bucket.m_total * m], DType::F32).unwrap();
    hipfire_dispatch::pipeline::run_grouped_moe_gemm(
        gpu,
        DType::Oq4G256,
        &ptrs,
        &tile_ids,
        &sorted,
        &x,
        &grouped,
        m,
        k,
        k_top,
        bucket.m_total,
        rows,
        false,
        false,
        !hipfire_runtime::oq_moe::moe_expert_blocks_repacked(),
    )
    .expect("run paged grouped OQ4");
    gpu.hip.device_synchronize().unwrap();
    let got = download_f32(gpu, &grouped, bucket.m_total * m);
    let rounded_x = x_host.iter().copied().map(round_f16).collect::<Vec<_>>();
    let mut max_abs = 0.0f32;
    for slot in 0..bucket.m_total {
        let flat = bucket.sorted_slot_index[slot];
        if flat < 0 {
            continue;
        }
        let token = flat as usize / k_top;
        for row in 0..m {
            let expected = (0..k)
                .map(|column| weights[row * k + column] * rounded_x[token * k + column])
                .sum::<f32>();
            max_abs = max_abs.max((got[slot * m + row] - expected).abs());
        }
    }
    assert!(
        max_abs <= 0.03,
        "paged canonical OQ4 max_abs={max_abs:.6e} exceeds 0.03"
    );
    println!("paged canonical OQ4: max_abs={max_abs:.6e} PASS");
    let _ = std::fs::remove_file(path);
}

fn main() {
    let mut gpu = Gpu::init().expect("initialize GPU");
    if gpu.arch != "gfx1151" {
        println!(
            "SKIP: mixed routed-expert fast path is scoped to gfx1151, got {}",
            gpu.arch
        );
        return;
    }
    for k_top in [8, 10] {
        for quant_dtype in [DType::MQ6G256, DType::Oq4G256] {
            run_case(&mut gpu, quant_dtype, DType::F16, k_top);
            run_case(&mut gpu, quant_dtype, DType::BF16, k_top);
        }
    }
    run_paged_oq4_case(&mut gpu);
    gpu.reclaim_pending();
}
