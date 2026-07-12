//! Hardware parity and latency gate for resident Dense heads plus L2 norm.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::NpuEmbeddingDenseL2;

    const INPUT: usize = 768;
    const INTERMEDIATE: usize = 3072;
    const OUTPUT: usize = 768;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_dense_l2_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    let input = (0..INPUT)
        .map(|index| ((index * 17 % 97) as f32 - 48.0) / 64.0)
        .collect::<Vec<_>>();
    let head0 = (0..INTERMEDIATE * INPUT)
        .map(|index| ((index * 13 % 61) as f32 - 30.0) / 2048.0)
        .collect::<Vec<_>>();
    let head1 = (0..OUTPUT * INTERMEDIATE)
        .map(|index| ((index * 19 % 67) as f32 - 33.0) / 4096.0)
        .collect::<Vec<_>>();
    let rounded = |value: f32| bf16_bits_to_f32(f32_to_bf16_bits(value));
    let mut hidden = vec![0.0f32; INTERMEDIATE];
    for row in 0..INTERMEDIATE {
        hidden[row] = input
            .iter()
            .zip(&head0[row * INPUT..(row + 1) * INPUT])
            .map(|(input, weight)| input * rounded(*weight))
            .sum();
    }
    let mut expected = vec![0.0f32; OUTPUT];
    for row in 0..OUTPUT {
        expected[row] = hidden
            .iter()
            .zip(&head1[row * INTERMEDIATE..(row + 1) * INTERMEDIATE])
            .map(|(input, weight)| input * rounded(*weight))
            .sum();
    }
    let inverse = expected
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .recip();
    expected.iter_mut().for_each(|value| *value *= inverse);

    let mut kernel = NpuEmbeddingDenseL2::load_cached(&args[0])?;
    kernel.upload_weights(&head0, &head1)?;
    kernel.write_input(&input)?;
    kernel.run_shared()?;
    let got = kernel.read_embedding_f32();
    let max_abs = got
        .iter()
        .zip(&expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let cosine = got
        .iter()
        .zip(&expected)
        .map(|(left, right)| left * right)
        .sum::<f32>()
        / (got.iter().map(|value| value * value).sum::<f32>().sqrt()
            * expected
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt());
    if cosine < 0.999_99 || max_abs > 5.0e-4 {
        return Err(
            format!("Dense/L2 parity failed: cosine={cosine:.8} max_abs={max_abs:.8}").into(),
        );
    }
    let started = Instant::now();
    for _ in 0..iterations {
        kernel.run_shared()?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "embedding-dense-l2 768->3072->768: cosine={cosine:.8} max_abs={max_abs:.8} dispatch_ms={dispatch_ms:.4}"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_dense_l2_verify is Linux-only");
}
