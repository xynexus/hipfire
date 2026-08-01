#!/usr/bin/env python3
"""Diff a device token sequence against the fp32 oracle, one step at a time.

    python3 decode_oracle.py --prompt 128000,791,4062,14198,39935 \\
                             --device 35308,927,279,16053,5679,627,32,11670

The oracle follows the DEVICE's token at every step rather than its own, so each
line is a one-step comparison on a SHARED prefix. Two greedy chains that diverge
once never rejoin, so "3 of 8 matched" between two independent chains says
nothing about where the disagreement is; this says exactly which step disagreed
and by how much the oracle preferred its own answer, which is the diagnostic. A
divergence at a margin of 0.05 logits is 4-bit quantization choosing between two
near-ties; one at 3.0 logits is a fault.

STANDALONE: `oracle_forward` imports torch, and torch after `aie.iron` segfaults
in this venv, so this cannot run in the same process as any device harness.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import oracle_forward as of  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True, help="comma-separated prompt ids")
    p.add_argument("--device", required=True,
                   help="comma-separated ids the device generated")
    o = p.parse_args()
    prompt = [int(t) for t in o.prompt.split(",")]
    dev = [int(t) for t in o.device.split(",")]

    cfg = json.loads(of.CFG.read_text())
    sd = of.load(cfg)
    toks, ours = list(prompt), []
    for want in dev:
        x16, _ = of.forward(toks, cfg, sd)
        lg = of.logits(x16[-1:], cfg, sd)[0]
        t = int(np.argmax(lg))
        ours.append(t)
        print(f"  pos {len(toks) - 1:2d}  oracle {t:6d}  device {want:6d}"
              f"  {'ok' if t == want else 'DIFFER'}"
              f"   top3 {np.argsort(lg)[::-1][:3].tolist()}"
              f"   oracle margin {lg[t] - np.sort(lg)[-2]:+.4f}", flush=True)
        toks.append(want)

    d = next((i for i, (a, b) in enumerate(zip(dev, ours)) if a != b), None)
    print(f"\n  device {dev}\n  oracle {ours}")
    print("  -> IDENTICAL" if d is None
          else f"  -> first divergence at generated token {d}")
    return 0 if d is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
