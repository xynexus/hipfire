#[cfg(test)]
fn signed_i4(v: u32) -> i32 {
    let nib = (v & 0xF) as i32;
    if nib >= 8 {
        nib - 16
    } else {
        nib
    }
}

#[cfg(test)]
fn pack_signed_i4(vals: &[i32; 8]) -> u32 {
    vals.iter()
        .enumerate()
        .fold(0u32, |acc, (i, v)| acc | (((v & 0xF) as u32) << (i * 4)))
}

#[cfg(test)]
fn dot8_i4_i4(a: u32, b: u32) -> i32 {
    let mut sum = 0;
    for i in 0..8 {
        sum += signed_i4(a >> (i * 4)) * signed_i4(b >> (i * 4));
    }
    sum
}

#[cfg(test)]
fn quantize_i4_group(x: &[f32]) -> (f32, i32, Vec<u32>, Vec<i32>) {
    assert_eq!(x.len(), 256);
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let dx = if amax > 0.0 { amax / 7.0 } else { 0.0 };
    let inv_dx = if amax > 0.0 { 7.0 / amax } else { 0.0 };
    let mut packs = Vec::with_capacity(32);
    let mut vals = Vec::with_capacity(256);
    let mut sum_a = 0;
    for chunk in x.chunks_exact(8) {
        let mut q = [0i32; 8];
        for (i, v) in chunk.iter().enumerate() {
            q[i] = (v * inv_dx).round().clamp(-8.0, 7.0) as i32;
            sum_a += q[i];
            vals.push(q[i]);
        }
        packs.push(pack_signed_i4(&q));
    }
    (dx, sum_a, packs, vals)
}

#[cfg(test)]
fn dot8_formula_group(
    sc: f32,
    zp: f32,
    q_packs: &[u32],
    a_packs: &[u32],
    dx: f32,
    sum_a: i32,
) -> f32 {
    assert_eq!(q_packs.len(), 32);
    assert_eq!(a_packs.len(), 32);
    let shifted_dot: i32 = q_packs
        .iter()
        .zip(a_packs)
        .map(|(q, a)| dot8_i4_i4(q ^ 0x8888_8888, *a))
        .sum();
    sc * dx * shifted_dot as f32 + (zp + 8.0 * sc) * dx * sum_a as f32
}

#[cfg(test)]
fn expanded_quantized_group(sc: f32, zp: f32, q_packs: &[u32], a_vals: &[i32], dx: f32) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..256 {
        let q = ((q_packs[i / 8] >> ((i % 8) * 4)) & 0xF) as f32;
        acc += (sc * q + zp) * (dx * a_vals[i] as f32);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q_packs() -> Vec<u32> {
        (0..32)
            .map(|p| {
                let mut out = 0u32;
                for i in 0..8 {
                    let q = ((p * 7 + i * 3 + 5) & 0xF) as u32;
                    out |= q << (i * 4);
                }
                out
            })
            .collect()
    }

    #[test]
    fn shifted_signed_dot_matches_expanded_quantized_math() {
        let x: Vec<f32> = (0..256)
            .map(|i| {
                let centered = (i as f32 % 37.0) - 18.0;
                centered * 0.03125 + ((i * 17 % 11) as f32 - 5.0) * 0.004
            })
            .collect();
        let (dx, sum_a, a_packs, a_vals) = quantize_i4_group(&x);
        let q_packs = q_packs();

        let got = dot8_formula_group(0.0375, -0.21, &q_packs, &a_packs, dx, sum_a);
        let expected = expanded_quantized_group(0.0375, -0.21, &q_packs, &a_vals, dx);

        assert!(
            (got - expected).abs() < 1.0e-5,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn zero_group_has_zero_scale_and_zero_sum() {
        let x = vec![0.0f32; 256];
        let (dx, sum_a, packs, vals) = quantize_i4_group(&x);
        assert_eq!(dx, 0.0);
        assert_eq!(sum_a, 0);
        assert!(packs.iter().all(|p| *p == 0));
        assert!(vals.iter().all(|v| *v == 0));
    }

    #[test]
    fn quantization_clamps_to_signed_i4_range() {
        let mut x = vec![0.0f32; 256];
        x[0] = -100.0;
        x[1] = 100.0;
        x[2] = 99.0;
        let (_dx, _sum_a, _packs, vals) = quantize_i4_group(&x);
        assert_eq!(vals[0], -7);
        assert_eq!(vals[1], 7);
        assert_eq!(vals[2], 7);
    }
}
