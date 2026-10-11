// SPDX-License-Identifier: Apache-2.0
// hipfire — Idefics3 image tiling + pixel shuffle. See LICENSE / NOTICE.

//! Port of HF `Idefics3ImageProcessor` (`do_image_splitting`, Lanczos) and the
//! connector's pixel shuffle. Every frame the tower sees is exactly
//! `tile × tile`, so no padding / patch masks / position bucketing are needed.

use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use hipfire_arch_gemma3_vl::patchify_rgb8;

use crate::config::Idefics3Config;

/// One image, tiled: per-frame SigLIP patch tensors (row-major tiles, then the
/// global view) plus the tile grid (`0×0` = single global frame).
pub struct Tiled {
    pub frames: Vec<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
}

struct Rgb {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

fn resize(img: &Rgb, w: usize, h: usize) -> Result<Rgb, String> {
    if (w, h) == (img.w, img.h) {
        // PIL returns a copy for a same-size resize.
        return Ok(Rgb {
            w,
            h,
            px: img.px.clone(),
        });
    }
    let src = Image::from_vec_u8(img.w as u32, img.h as u32, img.px.clone(), PixelType::U8x3)
        .map_err(|e| format!("idefics3: src image: {e}"))?;
    let mut dst = Image::new(w as u32, h as u32, PixelType::U8x3);
    Resizer::new()
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3)),
        )
        .map_err(|e| format!("idefics3: resize: {e}"))?;
    Ok(Rgb {
        w,
        h,
        px: dst.into_vec(),
    })
}

fn crop(img: &Rgb, x0: usize, y0: usize, w: usize, h: usize) -> Rgb {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in y0..y0 + h {
        px.extend_from_slice(&img.px[(y * img.w + x0) * 3..(y * img.w + x0 + w) * 3]);
    }
    Rgb { w, h, px }
}

/// `_resize_output_size_rescale_to_max_len`: long side → `max_len` (up OR
/// down), short side by aspect, rounded up to even.
fn longest_edge_size(w: usize, h: usize, max_len: usize) -> (usize, usize) {
    let aspect = w as f64 / h as f64;
    let (mut w, mut h) = (w, h);
    if w >= h {
        w = max_len;
        h = (w as f64 / aspect) as usize;
        h += h % 2;
    } else {
        h = max_len;
        w = (h as f64 * aspect) as usize;
        w += w % 2;
    }
    (w.max(1), h.max(1))
}

/// `resize_for_vision_encoder`: both sides up to multiples of `tile`.
fn tile_multiple_size(w: usize, h: usize, tile: usize) -> (usize, usize) {
    let aspect = w as f64 / h as f64;
    let up = |x: f64| (x / tile as f64).ceil() as usize * tile;
    if w >= h {
        let w = up(w as f64);
        (w, up(((w as f64 / aspect) as usize) as f64))
    } else {
        let h = up(h as f64);
        (up(((h as f64 * aspect) as usize) as f64), h)
    }
}

pub fn preprocess_bytes(bytes: &[u8], cfg: &Idefics3Config) -> Result<Tiled, String> {
    let decoded = image::load_from_memory(bytes)
        .map_err(|e| format!("idefics3: decode image: {e}"))?
        .to_rgba8();
    let (w, h) = (decoded.width() as usize, decoded.height() as usize);
    // `convert_to_rgb`: alpha-composite onto opaque white.
    let px = decoded
        .pixels()
        .flat_map(|p| {
            let a = p[3] as u32;
            [0, 1, 2].map(|c| ((p[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8)
        })
        .collect();
    let img = Rgb { w, h, px };
    let tile = cfg.vision.image_size;

    let (w, h) = longest_edge_size(img.w, img.h, cfg.longest_edge);
    let img = resize(&img, w, h)?;
    let (w, h) = tile_multiple_size(img.w, img.h, tile);
    let img = resize(&img, w, h)?;

    let (rows, cols) = if h > tile || w > tile {
        (h / tile, w / tile)
    } else {
        (0, 0)
    };
    let mut frames = Vec::with_capacity(rows * cols + 1);
    for r in 0..rows {
        for c in 0..cols {
            frames.push(patchify_rgb8(
                &crop(&img, c * tile, r * tile, tile, tile).px,
                &cfg.vision,
            ));
        }
    }
    frames.push(patchify_rgb8(&resize(&img, tile, tile)?.px, &cfg.vision));
    Ok(Tiled { frames, rows, cols })
}

/// The processor's expansion of one `<image>` placeholder.
pub fn image_prompt(rows: usize, cols: usize, seq_len: usize) -> String {
    let img = "<image>".repeat(seq_len);
    let mut s = String::new();
    if rows > 0 {
        for r in 1..=rows {
            for c in 1..=cols {
                s.push_str(&format!("<fake_token_around_image><row_{r}_col_{c}>{img}"));
            }
            s.push('\n');
        }
        s.push('\n');
    }
    s.push_str(&format!(
        "<fake_token_around_image><global-img>{img}<fake_token_around_image>"
    ));
    s
}

/// Connector pixel shuffle: `[g·g, d]` → `[(g/s)², d·s²]`, where output row
/// `(i/s)·(g/s) + j/s` holds patch `(i, j)` at column `(i%s)·s·d + (j%s)·d`.
pub fn pixel_shuffle(x: &[f32], grid: usize, d: usize, s: usize) -> Vec<f32> {
    let g2 = grid / s;
    let mut out = vec![0.0f32; x.len()];
    for i in 0..grid {
        for j in 0..grid {
            let row = (i / s) * g2 + j / s;
            let col = (i % s) * s * d + (j % s) * d;
            let dst = row * d * s * s + col;
            out[dst..dst + d].copy_from_slice(&x[(i * grid + j) * d..(i * grid + j + 1) * d]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_hf() {
        // 1000×800 → long side 2048, short 1638 (even) → 2048×2048 tiles grid.
        assert_eq!(longest_edge_size(1000, 800, 2048), (2048, 1638));
        assert_eq!(tile_multiple_size(2048, 1638, 512), (2048, 2048));
        // Portrait A4-ish 1240×1754 → 1448×2048 → 1536×2048 (3×4 tiles).
        assert_eq!(longest_edge_size(1240, 1754, 2048), (1448, 2048));
        assert_eq!(tile_multiple_size(1448, 2048, 512), (1536, 2048));
    }

    #[test]
    fn pixel_shuffle_matches_torch_index_map() {
        // grid 4, d 1, s 2: 2×2 blocks become rows, each block read row-major.
        let x: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let y = pixel_shuffle(&x, 4, 1, 2);
        assert_eq!(&y[0..4], &[0.0, 1.0, 4.0, 5.0]);
        assert_eq!(&y[4..8], &[2.0, 3.0, 6.0, 7.0]);
        assert_eq!(&y[12..16], &[10.0, 11.0, 14.0, 15.0]);
    }

    #[test]
    fn prompt_shape() {
        let p = image_prompt(1, 2, 2);
        assert_eq!(
            p,
            "<fake_token_around_image><row_1_col_1><image><image>\
             <fake_token_around_image><row_1_col_2><image><image>\n\n\
             <fake_token_around_image><global-img><image><image><fake_token_around_image>"
        );
    }
}
