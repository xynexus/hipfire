fn main() {
    use hipfire_kvquant::kvarn::*;
    let (r, c) = (8usize, 128usize);
    let mut tile = vec![0f32; r * c];
    for i in 0..r { for j in 0..c {
        let base = (((i * 31 + j * 17) % 13) as f32 - 6.0) * 0.05;
        tile[i * c + j] = if j == (i * 7) % c { 12.0 } else { base };
    }}
    let err = |q: &QuantTile, rf: &[f32]| -> f32 {
        let d = dequantize_tile(q);
        let num: f32 = d.iter().zip(rf).map(|(a,b)| (a-b)*(a-b)).sum();
        let den: f32 = rf.iter().map(|v| v*v).sum::<f32>().max(1e-12);
        (num/den).sqrt()
    };
    let mut rot_ref = tile.clone(); hadamard_rows(&mut rot_ref, c);
    println!("{:>6} {:>12} {:>12} {:>8}", "bits", "plain", "rotated", "gain");
    for (bits, qmax) in [(2usize, 3.0f32), (4, 15.0), (8, 255.0)] {
        let p = err(&quantize_tile_qmax(&tile, r, c, qmax), &tile);
        let q = err(&quantize_tile_rotated(&tile, r, c, qmax), &rot_ref);
        println!("{bits:>6} {p:>12.5} {q:>12.5} {:>7.2}x", p / q.max(1e-9));
    }
}
