#!/usr/bin/env python3
"""The EXTERNAL oracle: Llama-3.2-1B in fp32, straight from Meta's checkpoint.

Nothing in this file reads the q4nx container, `q4nx.py`, or any host reference
in this tree. It loads `original/consolidated.00.pth` and `config.json` and
computes a causal forward over a whole prompt. That is the only kind of check
worth having here: five faults in this project hid because a stage was compared
against a reference computed from the same wrong input.

**Meta format, not HF.** RoPE rotates the (2i, 2i+1) pairs and the q/k rows carry
Meta's order — the container is HF-*named* but its rows decode to Meta order, and
the device gets there by a pack-time row permutation (`qkv_verify.load_linear`).
None of that is repeated here; the oracle rotates pairwise on checkpoint rows,
which is what the model means.

The llama3 rope schedule is recomputed from `config.json` rather than read from
the container's `rope_freqs.weight` — the two were measured identical to bf16
(6.5e-04 relative), and taking it from the config keeps this file free of the
container.

## The arithmetic is torch's, deliberately

A numpy version of this file was written first and was **silently wrong for
prompts longer than one token**. Two independent numpy faults were measured in
the same process, neither reproducible standalone:

  * `(g / (1 + exp(-g))) * u` evaluated to `u` — `np.array_equal` True — so the
    SwiGLU gate was dropped, while the identical expression three lines later
    was correct;
  * with torch also imported, `h2 @ w1.T` gave three different answers in one
    scope, the second equal to the first plus the correct one, exactly.

Both vanish at T=1, where numpy takes gemv and elementwise paths of a different
shape — so the single-token oracle reproduced the recorded result (`16309`,
x16 1.76944 / 155.896) perfectly and the five-token one did not. That is the
worst possible failure mode for an oracle and it is why this file no longer does
its own linear algebra: torch is a second implementation, maintained elsewhere,
and it agrees with an independent per-position hand computation.

    python3 oracle_forward.py --tokens 128000,791,4062,14198,39935
    python3 oracle_forward.py --prompt "The quick brown fox" --save x16.npy

Do NOT import this alongside `aie.iron` — that combination segfaults in this
venv. The device harnesses shell out to it or read the .npy it saves.
"""

import argparse
import json
from pathlib import Path

import numpy as np

SNAP = Path("/srv/huggingface/models--meta-llama--Llama-3.2-1B-Instruct/"
            "snapshots/9213176726f574b556790deb65791e0c5aa438b6")
CKPT = SNAP / "original/consolidated.00.pth"
CFG = SNAP / "config.json"
TOKJSON = SNAP / "tokenizer.json"


def load(cfg=None):
    import torch
    return torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)


def llama3_inv_freq(cfg):
    """The llama3 rope schedule, from the config. Every frequency is rescaled,
    not just the long-range ones, which a plain 1/theta^(2i/d) misses."""
    hd, theta = cfg["head_dim"], cfg["rope_theta"]
    s = cfg["rope_scaling"]
    lo, hi = s["low_freq_factor"], s["high_freq_factor"]
    fac, old = s["factor"], s["original_max_position_embeddings"]
    base = 1.0 / (theta ** (np.arange(0, hd, 2) / hd))
    wl = 2 * np.pi / base
    out = np.where(wl > old / lo, base / fac, base)
    mid = (wl >= old / hi) & (wl <= old / lo)
    smooth = (old / wl - lo) / (hi - lo)
    return np.where(mid, (1 - smooth) * base / fac + smooth * base, out)


def _rope(v, co, si):
    """Meta convention: rotate the (2i, 2i+1) pairs. `v` is [T, heads, head_dim]."""
    import torch
    e, o = v[..., 0::2], v[..., 1::2]
    return torch.stack([e * co - o * si, o * co + e * si], -1).flatten(-2)


def forward(tokens, cfg=None, sd=None, keep=None, act_hook=None):
    """Causal fp32 forward over `tokens` -> (x16 [T, hidden] numpy, stash).

    x16 is the residual stream BEFORE `norm.weight`, which is what the device
    produces; the final norm and lm_head are the head's job. `keep` is an
    optional list of layer indices whose *input* hidden states are stashed.

    `act_hook`, if given, is applied to EVERY GEMV input just before the
    matmul -- the four of them: the attention-norm output feeding q/k/v, the
    attention output feeding wo, the ffn-norm output feeding w1/w3, and the
    SwiGLU product feeding w2. That is what makes a W4A**4** evaluation
    possible without reimplementing this forward: quantising the activations
    is the only thing an A4 format does that an A16 one does not, and every
    quality comparison of a ROTATED format needs it (the rotation cuts
    activation int4 error 2.1-2.5x and weight error only 1.1x, measured).
    Default None leaves this function bit-identical to what it was.
    """
    import torch
    cfg = cfg or json.loads(CFG.read_text())
    sd = load(cfg) if sd is None else sd
    W = lambda n: sd[n].float()

    NH, NKV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    HD, EPS = cfg["head_dim"], cfg["rms_norm_eps"]
    GQA, T = NH // NKV, len(tokens)
    ang = torch.from_numpy(np.outer(np.arange(T), llama3_inv_freq(cfg))
                           ).float()[:, None, :]        # [T, 1, HD/2]
    co, si = torch.cos(ang), torch.sin(ang)
    neg = torch.full((T, T), float("-inf")).triu(1)      # causal

    rms = lambda t, w: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * w

    with torch.no_grad():
        hook = act_hook or (lambda t: t)
        x = W("tok_embeddings.weight")[list(tokens)]
        stash = {}
        for L in range(cfg["num_hidden_layers"]):
            P = f"layers.{L}."
            if keep and L in keep:
                stash[L] = x.numpy().copy()
            h1 = hook(rms(x, W(P + "attention_norm.weight")))
            q = _rope((h1 @ W(P + "attention.wq.weight").T).view(T, NH, HD), co, si)
            k = _rope((h1 @ W(P + "attention.wk.weight").T).view(T, NKV, HD), co, si)
            v = (h1 @ W(P + "attention.wv.weight").T).view(T, NKV, HD)
            # GQA: query head h attends with KV head h // GQA.
            kx = k.repeat_interleave(GQA, 1).permute(1, 2, 0)      # [NH, HD, T]
            vx = v.repeat_interleave(GQA, 1).permute(1, 0, 2)      # [NH, T, HD]
            sc = torch.bmm(q.permute(1, 0, 2), kx) / HD ** 0.5      # [NH, T, T]
            p = torch.softmax(sc + neg, -1)
            a = torch.bmm(p, vx).permute(1, 0, 2).reshape(T, NH * HD)
            h = x + hook(a) @ W(P + "attention.wo.weight").T

            h2 = hook(rms(h, W(P + "ffn_norm.weight")))
            g = h2 @ W(P + "feed_forward.w1.weight").T
            u = h2 @ W(P + "feed_forward.w3.weight").T
            x = h + hook(torch.nn.functional.silu(g) * u) @ W(P + "feed_forward.w2.weight").T
        return x.numpy().copy(), stash


def logits(x16, cfg=None, sd=None):
    """Final RMSNorm then lm_head — the head, applied to a pre-norm hidden state."""
    import torch
    cfg = cfg or json.loads(CFG.read_text())
    sd = load(cfg) if sd is None else sd
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(x16, np.float32))
        xn = (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + cfg["rms_norm_eps"])
              * sd["norm.weight"].float())
        return (xn @ sd["output.weight"].float().T).numpy().copy()


def encode(text):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(TOKJSON))
    return [128000] + t.encode(text, add_special_tokens=False).ids


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", help="comma-separated token ids")
    p.add_argument("--prompt", help="text; BOS is prepended")
    p.add_argument("--save", help="write the last position's x16 (pre-norm) to .npy")
    p.add_argument("--top", type=int, default=5)
    o = p.parse_args()
    if o.tokens:
        toks = [int(t) for t in o.tokens.split(",")]
    elif o.prompt:
        toks = encode(o.prompt)
    else:
        toks = [128000]
    cfg = json.loads(CFG.read_text())
    sd = load(cfg)
    print(f"oracle: fp32 from consolidated.00.pth, {len(toks)} tokens {toks}")

    x16, _ = forward(toks, cfg, sd)
    if o.save:
        np.save(o.save, x16[-1].astype(np.float32))
    lg = logits(x16[-1:], cfg, sd)[0]
    top = np.argsort(lg)[::-1][:o.top]
    print(f"  x16 mean|.| {np.abs(x16[-1]).mean():.5f}  max {np.abs(x16[-1]).max():.5f}")
    print(f"  argmax {top[0]}")
    for t in top:
        print(f"    {t:6d}  {lg[t]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
