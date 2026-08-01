#!/usr/bin/env python3
"""A correct Llama layer on q4nx weights — the yardstick, rebuilt.

The per-phase host references in this tree were the target every device check was
measured against, and one of them was wrong: P5 added `x`, the layer input, where
a Llama layer adds `h`, the post-attention residual. Device and reference agreed
to the bit at thirteen of sixteen layers and both computed something that is not
a Llama layer (cosine 0.012 against an independent fp32 forward).

So before touching the device, rebuild the reference and prove it against an
oracle that shares no code with it:

    h = x + o_proj(attn(rmsnorm(x, w_in)))
    y = h + down(silu(gate(h2)) * up(h2)),  h2 = rmsnorm(h, w_post)

Position 0 only, which makes the attention exact and trivial: the softmax is over
one entry, so each q head's output is its KV group's v, and RoPE at position 0 is
the identity. That removes every term this file would otherwise have to get right
to test the one term it exists to test — the residual.

    python3 host_forward.py                 # BOS, prints the token
    python3 host_forward.py --token 9906

Needs the q4nx container; no NPU. `--compare` also runs the fp32 checkpoint.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
# NOT ffn_verify.load_linear: that returns the CONTAINER's row order, which is
# uncorrelated with the checkpoint (corr 0.001, relF 1.52 measured). This is the
# function written beside q4nx_decode_tensor for exactly this purpose.
def load_linear(c, name, N, K):
    return q4nx.q4nx_tensor_blocks(c, name, (N, K))
from head_verify import Q4NX, VOCAB, logits_for, rmsnorm  # noqa: E402
from qkv_verify import HEAD, K_DIM  # noqa: E402

D_FF, NLAY, NQ, NKV = 8192, 16, 32, 8


def layer(c, x, L):
    """One decoder layer at position 0, on q4nx weights."""
    P = f"model.layers.{L}."
    nw1 = c.bf16(P + "input_layernorm.weight").astype(np.float32)[:K_DIM]
    h1 = rmsnorm(x, nw1)

    # Only v is needed: at position 0 the softmax is over a single entry, so the
    # attention output for every q head is its KV group's v.
    vd, vm, vc = load_linear(c, P + "self_attn.v_proj.weight", NKV * HEAD, K_DIM)
    v = q4nx.gemv_reference_bf16(h1, vd, vm, vc)
    attn = np.repeat(v.reshape(NKV, HEAD), NQ // NKV, axis=0).reshape(-1)

    od, om, oc = load_linear(c, P + "self_attn.o_proj.weight", K_DIM, K_DIM)
    h = x + q4nx.gemv_reference_bf16(attn.astype(np.float32), od, om, oc)

    # THE FIX: the FFN's residual is h, not x. Adding x here drops the attention
    # contribution from the layer's output entirely.
    nw2 = c.bf16(P + "post_attention_layernorm.weight").astype(np.float32)[:K_DIM]
    h2 = rmsnorm(h.astype(np.float32), nw2)

    gd, gm, gc = load_linear(c, P + "mlp.gate_proj.weight", D_FF, K_DIM)
    ud, um, uc = load_linear(c, P + "mlp.up_proj.weight", D_FF, K_DIM)
    g = q4nx.gemv_reference_bf16(h2, gd, gm, gc)
    u = q4nx.gemv_reference_bf16(h2, ud, um, uc)
    sw = (g / (1.0 + np.exp(-g))) * u

    dd, dm, dc = load_linear(c, P + "mlp.down_proj.weight", K_DIM, D_FF)
    return h + q4nx.gemv_reference_bf16(sw.astype(np.float32), dd, dm, dc)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--token", type=int, default=128000)
    p.add_argument("--save", help="write the final hidden state to .npy")
    p.add_argument("--top", type=int, default=5)
    o = p.parse_args()

    c = q4nx.Q4nx(str(Q4NX))
    emb = c.bf16("model.embed_tokens.weight").astype(np.float32).reshape(-1, K_DIM)
    x = emb[o.token].astype(np.float64).copy()
    print(f"host forward, token {o.token}, position 0")
    for L in range(NLAY):
        x = layer(c, x, L)
        if L in (0, 1, 7, NLAY - 1):
            print(f"  x{L+1:<3d} mean|.| {np.abs(x).mean():.5f}  max {np.abs(x).max():.5f}")

    xf = x.astype(np.float32)
    if o.save:
        np.save(o.save, xf)
    nw = c.bf16("model.norm.weight").astype(np.float32)[:K_DIM]
    lg = logits_for(c, rmsnorm(xf, nw))
    top = np.argsort(lg)[::-1][:o.top]
    print(f"  argmax {top[0]}")
    for t in top:
        print(f"    {t:6d}  {lg[t]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
