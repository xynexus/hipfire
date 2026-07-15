#!/usr/bin/env python3
"""Optional LPIPS perceptual cross-check over the PNGs saved by the diffusion
admission battery (executor_diffusion.rs writes them to
`<run>/artifacts/diffusion/caseN_seedS_{baseline,candidate}.png`).

LPIPS needs a pretrained CNN (AlexNet/VGG), which does not belong in hipfire's
Rust admission path or CI, so it lives here as an out-of-band comparison baseline
(AGENTS.md: Python is allowed for tooling/benchmarks/comparison baselines).

    pip install lpips torch torchvision pillow numpy
    python scripts/flux2_lpips.py <run>/artifacts/diffusion
    python scripts/flux2_lpips.py <dir> --net vgg --json

Lower LPIPS = more perceptually similar. Repositioned-but-coherent renders score
moderate; garbage scores high. Use as a perceptual sanity check alongside the
in-Rust SSIM/early-latent gates, not as a hard threshold without calibration.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_dir", type=Path, help="artifacts/diffusion dir with the saved PNG pairs")
    ap.add_argument("--net", choices=["alex", "vgg", "squeeze"], default="alex")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        import lpips as lpips_lib
        from PIL import Image
    except ImportError as e:
        print(f"error: missing dependency ({e}). pip install lpips torch torchvision pillow numpy",
              file=sys.stderr)
        return 2

    d = args.image_dir
    if not d.is_dir():
        print(f"error: {d} is not a directory", file=sys.stderr)
        return 2

    pairs = {}
    pat = re.compile(r"^(case\d+_seed-?\d+)_(baseline|candidate)\.png$")
    for p in sorted(d.glob("*.png")):
        m = pat.match(p.name)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = p
    pairs = {k: v for k, v in pairs.items() if "baseline" in v and "candidate" in v}
    if not pairs:
        print(f"error: no baseline/candidate PNG pairs found in {d}", file=sys.stderr)
        return 2

    loss_fn = lpips_lib.LPIPS(net=args.net)

    def load(path):
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    results = {}
    for case, files in pairs.items():
        with torch.no_grad():
            v = float(loss_fn(load(files["baseline"]), load(files["candidate"])).item())
        results[case] = v

    if args.json:
        print(json.dumps({"net": args.net, "lpips": results,
                          "mean": sum(results.values()) / len(results)}, indent=2))
    else:
        print(f"LPIPS (net={args.net}, lower = more similar):")
        for case, v in results.items():
            print(f"  {case:24s} {v:.4f}")
        print(f"  {'mean':24s} {sum(results.values()) / len(results):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
