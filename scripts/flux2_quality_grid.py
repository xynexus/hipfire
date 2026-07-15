#!/usr/bin/env python3
"""Render a quality-inspection grid across quantization levels for human review.

Rows = prompts, columns = model variants (bf16 + a bf16 alt-seed for natural
variation + the quant levels). One model load per variant (multi-prompt batch).
Composes a single labeled PNG. Run under `hipfire lock run`.

    hipfire lock run qgrid -- python3 scripts/flux2_quality_grid.py <scratchpad> [size] [steps]
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HIPFIRE = "./target/release/hipfire"


def main() -> int:
    scratch = Path(sys.argv[1])
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    prompt_files = [
        "benchmarks/prompts/flux2_image_admission_object.txt",
        "benchmarks/prompts/flux2_image_admission_texture.txt",
    ]
    prompts = [Path(p).read_text().strip() for p in prompt_files]
    row_labels = ["object", "texture"]

    # (column label, model file, seed)
    variants = [
        ("bf16 (seed7)", scratch / "klein4b.p0.hfq", 7),
        ("bf16 (seed42)", scratch / "klein4b.p0.hfq", 42),
        ("cons oqf4 (10t)", scratch / "klein4b.oqf4cons.hfq", 7),
        ("data-driven (40t)", scratch / "klein4b.oqf4dd.hfq", 7),
        ("aggressive (109t)", scratch / "klein4b.oqf4agg.hfq", 7),
    ]

    grids = scratch / "qgrid"
    grids.mkdir(exist_ok=True)
    cell_imgs = {}  # (row, col) -> PIL.Image

    # max_batch=1 on this artifact ⇒ one prompt per call (one model load each).
    for col, (label, model, seed) in enumerate(variants):
        outdir = grids / f"col{col}"
        outdir.mkdir(exist_ok=True)
        print(f"[{col+1}/{len(variants)}] {label} ({model.name}, seed {seed})", flush=True)
        for row, pr in enumerate(prompts):
            png = outdir / f"r{row}.png"
            cmd = [HIPFIRE, "diffusion", "txt2img", "--model", str(model),
                   "--prompt", pr, "--seed", str(seed), "--output", str(png),
                   "--steps", str(steps), "--cfg-scale", "4.0",
                   "--width", str(size), "--height", str(size), "--rocm-device-id", "0"]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if r.returncode != 0:
                print(f"  row {row} FAILED: {r.stderr[-200:]}", flush=True)
            elif png.exists():
                cell_imgs[(row, col)] = Image.open(png).convert("RGB").resize((size, size))

    # Compose labeled grid.
    pad, top, left = 6, 26, 130
    W = left + len(variants) * (size + pad) + pad
    H = top + len(prompts) * (size + pad) + pad
    canvas = Image.new("RGB", (W, H), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    for col, (label, _, _) in enumerate(variants):
        x = left + col * (size + pad)
        draw.text((x + 4, 8), label, fill=(230, 230, 235))
    for row, rl in enumerate(row_labels):
        y = top + row * (size + pad)
        draw.text((6, y + size // 2), rl, fill=(230, 230, 235))
    for (row, col), img in cell_imgs.items():
        x = left + col * (size + pad)
        y = top + row * (size + pad)
        canvas.paste(img, (x, y))

    out = scratch / "quality_grid.png"
    canvas.save(out)
    print(f"grid -> {out} ({W}x{H})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
