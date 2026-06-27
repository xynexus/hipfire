#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — compare hipfire vs HF LFM2.5 per-layer hidden states + final logits
# to locate the first diverging layer in the lfm2 forward pass.
import argparse, json, struct, sys
import numpy as np


def load_hidden(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"LFM2HID0", f"bad magic in {path}: {magic!r}"
        n_layers, n_pos, hidden, _ = struct.unpack("<IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype="<f4")
    arr = data.reshape(n_pos, n_layers, hidden)
    return arr  # [pos, layer, hidden]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="shared out-prefix for both dumps")
    args = ap.parse_args()

    H = load_hidden(f"{args.prefix}.hipfire.hidden.bin")
    R = load_hidden(f"{args.prefix}.hf.hidden.bin")
    assert H.shape == R.shape, f"shape mismatch hipfire {H.shape} vs hf {R.shape}"
    n_pos, n_layers, hidden = H.shape
    print(f"shape: pos={n_pos} layers={n_layers} hidden={hidden}\n")

    # Compare the LAST position (drives next-token logits) per layer.
    p = n_pos - 1
    print(f"per-layer @ last position (pos={p}):")
    print(f"{'layer':>5} {'cos':>9} {'max_abs':>11} {'mean_abs':>11} "
          f"{'|hf|':>9} {'|hip|':>9}")
    first_bad = None
    for l in range(n_layers):
        h, r = H[p, l], R[p, l]
        c = cos(h, r)
        mx = float(np.max(np.abs(h - r)))
        mn = float(np.mean(np.abs(h - r)))
        flag = ""
        if c < 0.99 and first_bad is None:
            first_bad = l
            flag = "  <-- FIRST DIVERGENCE"
        print(f"{l:>5} {c:>9.4f} {mx:>11.3f} {mn:>11.4f} "
              f"{np.linalg.norm(r):>9.2f} {np.linalg.norm(h):>9.2f}{flag}")

    print()
    if first_bad is None:
        print("hidden states track (all cos >= 0.99) — divergence is in final "
              "norm / lm_head, not the layer stack.")
    else:
        print(f"FIRST DIVERGING LAYER: {first_bad} "
              f"(cos {cos(H[p, first_bad], R[p, first_bad]):.4f})")
        if first_bad > 0:
            print(f"  layer {first_bad-1} still matches "
                  f"(cos {cos(H[p, first_bad-1], R[p, first_bad-1]):.4f}) — "
                  f"bug is in layer {first_bad}'s mixer/mlp.")
        else:
            print("  layer 0 already diverges — bug is in embedding or layer-0 "
                  "(conv) mixer / first rmsnorm.")

    # Final logits comparison.
    try:
        hl = json.load(open(f"{args.prefix}.hipfire.logits.json"))
        rl = json.load(open(f"{args.prefix}.hf.logits.json"))
        print("\nfinal logits:")
        print(f"  hipfire argmax={hl['argmax']} ({hl['argmax_text']!r}) "
              f"range=[{hl['logit_min']:.2f},{hl['logit_max']:.2f}] mean={hl['logit_mean']:.3f}")
        print(f"  hf      argmax={rl['argmax']} ({rl['argmax_text']!r}) "
              f"range=[{rl['logit_min']:.2f},{rl['logit_max']:.2f}] mean={rl['logit_mean']:.3f}")
        print(f"  argmax match: {hl['argmax'] == rl['argmax']}")
        print(f"  hf top5: {[(t['id'], round(t['logit'],2), t['text']) for t in rl['top20'][:5]]}")
        print(f"  hip top5: {[(t['id'], round(t['logit'],2), t['text']) for t in hl['top20'][:5]]}")
    except FileNotFoundError as e:
        print(f"(logits json missing: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
