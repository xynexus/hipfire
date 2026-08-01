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

`layer_parts(c, x, L, pos, prior)` is the one layer function and everything else
calls it, so the copy-and-diverge that produced those faults cannot recur.

    prior=None    the POSITION-0 shortcut: the softmax is over one entry, so each
                  q head's output is its KV group's v and RoPE is the identity.
                  Every term but the residual drops out, which is what made it the
                  right thing to validate against the oracle first. Unchanged, and
                  still reproduces the oracle's token (16309) bit for bit.
    prior=(K,V)   real multi-position attention: q and k rotated at `pos`, GQA over
                  8 KV groups, softmax over the cache. `prefill()` builds K and V
                  by running the model forward one token at a time, because layer
                  L's input at position p is layer L-1's output at position p.

    python3 host_forward.py                 # BOS, prints the token
    python3 host_forward.py --token 9906
    python3 host_forward.py --tokens 128000,791,4062,14198,39935

The multi-position path is externally validated: at pos 4 it produces the same
token (35308) as an fp32 forward from `consolidated.00.pth`, cosine 0.964990 at
4-bit weights. That is what `fused.py --tokens` measures the device against.

Needs the q4nx container; no NPU.
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
#
# Memoised, because a multi-position `prefill` walks every layer once per token
# and the container decode dominates. ~60 MB of blocks per layer, so a 16-layer
# model holds ~1 GB — fine here and nowhere near a hot path.
# ponytail: unbounded dict; give it an LRU if a model ever outgrows RAM.
_BLOCKS = {}


def load_linear(c, name, N, K):
    if name not in _BLOCKS:
        _BLOCKS[name] = q4nx.q4nx_tensor_blocks(c, name, (N, K))
    return _BLOCKS[name]
from head_verify import Q4NX, VOCAB, logits_for, rmsnorm  # noqa: E402
from qkv_verify import HEAD, K_DIM, EPS, ROPE_THETA, rope_ref  # noqa: E402
# q_proj/k_proj come back with each head's rows reordered to
# [0,2,...,62,1,3,...,63], which is what makes the kernel's HALF-SPLIT rotation
# compute the checkpoint's PAIRWISE one. Everything else uses the plain loader.
from qkv_verify import load_linear as load_rope_linear  # noqa: E402

D_FF, NLAY, NQ, NKV = 8192, 16, 32, 8
GQA = NQ // NKV
rnd = lambda v: q4nx.bf16_to_f32(q4nx.f32_to_bf16(np.asarray(v, np.float32)))


def rope_cs(c, pos):
    """-> (cs_q, cs_k), the cos|sin tables the broadcast carries at `pos`.

    `rope_freqs.weight` is the container's stored llama3 per-frequency divisor,
    measured identical to bf16 against the schedule computed from `config.json`.
    cs_q also carries attention's `head_dim^-0.5 * log2(e)` pre-scale, which is
    why the reference divides the scores by log2(e) rather than by sqrt(d).
    """
    divisor = c.bf16("rope_freqs.weight").astype(np.float64)[:HEAD // 2]
    inv = (1.0 / ROPE_THETA ** (np.arange(0, HEAD, 2) / HEAD)) / divisor
    ang = pos * inv
    cs_k = rnd(np.concatenate([np.cos(ang), np.sin(ang)]))
    return rnd(cs_k * (HEAD ** -0.5) * np.log2(np.e)), cs_k


def qk_at(c, h1, L, pos):
    """-> (q [NQ][HEAD], k [NKV][HEAD]) exactly as phase P1 emits them.

    GEMV, round to bf16, rotate, round again — the kernel stages the projection
    in a bf16 buffer and rotates *that*, so a reference that rotates full
    precision values measures its own extra precision (1.25%, `rope_ref`).
    """
    cs_q, cs_k = rope_cs(c, pos)
    out = []
    for name, n, cs in ((".self_attn.q_proj.weight", NQ, cs_q),
                        (".self_attn.k_proj.weight", NKV, cs_k)):
        full = f"model.layers.{L}" + name
        key = "rope:" + full
        if key not in _BLOCKS:
            _BLOCKS[key] = load_rope_linear(c, full, n * HEAD, K_DIM)
        d, m, q = _BLOCKS[key]
        raw = q4nx.gemv_reference_bf16(h1, d, m, q).reshape(n, HEAD)
        out.append(np.stack([rnd(rope_ref(rnd(raw[h]), cs)) for h in range(n)]))
    return out


def layer_parts(c, x, L, pos=0, prior=None):
    """One decoder layer on q4nx weights -> (attn, h, sw, y, k, v).

    `prior` is None for the position-0 shortcut: the softmax is over a single
    entry, so each q head's output is its KV group's v and RoPE is the identity.
    That path is unchanged and is what the fp32 oracle validated end to end.

    Otherwise `prior` is (K, V), each [pos][NKV][HEAD] — the ROTATED k' and the
    v' of every earlier position, in the same permuted row order the device's
    cache holds. The returned k/v are this position's, to append to it.
    """
    P = f"model.layers.{L}."
    nw1 = c.bf16(P + "input_layernorm.weight").astype(np.float32)[:K_DIM]
    h1 = rmsnorm(x, nw1)

    vd, vm, vc = load_linear(c, P + "self_attn.v_proj.weight", NKV * HEAD, K_DIM)
    v = q4nx.gemv_reference_bf16(h1, vd, vm, vc).reshape(NKV, HEAD)
    if prior is None:
        attn = np.repeat(v, GQA, axis=0).reshape(-1)
        k = None
    else:
        q, k = qk_at(c, h1, L, pos)
        v = rnd(v)                              # the cache holds bf16
        Kp, Vp = prior
        Kf = np.concatenate([np.asarray(Kp, np.float64).reshape(-1, NKV, HEAD),
                             k[None].astype(np.float64)])       # [pos+1][NKV][HEAD]
        Vf = np.concatenate([np.asarray(Vp, np.float64).reshape(-1, NKV, HEAD),
                             v[None].astype(np.float64)])
        # q' already carries head_dim^-0.5 * log2(e), so undo the log2(e) to get
        # a natural-log softmax. Query head h attends with KV group h // GQA.
        att = np.empty((NQ, HEAD), np.float64)
        for h in range(NQ):
            g = h // GQA
            sc = (Kf[:, g] @ q[h].astype(np.float64)) / np.log2(np.e)
            e = np.exp(sc - sc.max())
            att[h] = (e / e.sum()) @ Vf[:, g]
        attn = rnd(att).reshape(-1)

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
    y = h + q4nx.gemv_reference_bf16(sw.astype(np.float32), dd, dm, dc)
    return attn, h, sw, y, k, v


def layer(c, x, L):
    """One decoder layer at position 0, on q4nx weights."""
    return layer_parts(c, x, L)[3]


def prefill(c, tokens, nlay=NLAY, L0=0):
    """Sequential host prefill of positions 0..len(tokens)-1.

    -> (K, V), each a list of nlay arrays [T][NKV][HEAD]: the k'/v' every layer
    would have written had the device decoded those tokens one at a time. This
    is what a decode step at position T reads back, and building it needs a real
    forward — layer L's input at position p is layer L-1's output at position p,
    which itself attends over positions 0..p.
    """
    emb = c.bf16("model.embed_tokens.weight").astype(np.float32).reshape(-1, K_DIM)
    K = [[] for _ in range(nlay)]
    V = [[] for _ in range(nlay)]
    for p, tok in enumerate(tokens):
        x = rnd(emb[tok]).astype(np.float64)
        for L in range(nlay):
            prior = (np.array(K[L]).reshape(p, NKV, HEAD),
                     np.array(V[L]).reshape(p, NKV, HEAD))
            _, _, _, x, k, v = layer_parts(c, x, L0 + L, p, prior)
            K[L].append(k)
            V[L].append(v)
    return ([np.asarray(k, np.float64) for k in K],
            [np.asarray(v, np.float64) for v in V])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--token", type=int, default=128000)
    p.add_argument("--tokens", help="comma-separated prompt; positions 0..n-2 are "
                                    "prefilled and the decode step runs at n-1")
    p.add_argument("--save", help="write the final hidden state to .npy")
    p.add_argument("--top", type=int, default=5)
    o = p.parse_args()

    c = q4nx.Q4nx(str(Q4NX))
    emb = c.bf16("model.embed_tokens.weight").astype(np.float32).reshape(-1, K_DIM)
    if o.tokens:
        toks = [int(t) for t in o.tokens.split(",")]
        pos = len(toks) - 1
        print(f"host forward, {len(toks)} tokens {toks}, decode at position {pos}")
        Kc, Vc = prefill(c, toks[:pos])
        x = rnd(emb[toks[pos]]).astype(np.float64)
        for L in range(NLAY):
            x = layer_parts(c, x, L, pos, (Kc[L], Vc[L]))[3]
            if L in (0, 1, 7, NLAY - 1):
                print(f"  x{L+1:<3d} mean|.| {np.abs(x).mean():.5f}  "
                      f"max {np.abs(x).max():.5f}")
    else:
        x = emb[o.token].astype(np.float64).copy()
        print(f"host forward, token {o.token}, position 0")
        for L in range(NLAY):
            x = layer(c, x, L)
            if L in (0, 1, 7, NLAY - 1):
                print(f"  x{L+1:<3d} mean|.| {np.abs(x).mean():.5f}  "
                      f"max {np.abs(x).max():.5f}")

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
