#!/usr/bin/env python3
"""Multi-token decode: the device carries its own KV cache from token to token.

    embed -> 16 layers (ONE dispatch) -> final RMSNorm -> lm_head -> argmax -> feed back

Every earlier result in this tree is a SINGLE decode step onto a cache the host
built. This is the loop, and two things had to land together for it to exist:

  * `g_kprev` is per LAYER (`kernels/npu/flm_kv_pair.h`). A k' column pair is
    closed with the previous token's key out of core `.bss`, and the fused
    design runs all sixteen layers on that core -- so one array had layer 0 of
    the next token closing its pair with layer 15's key. Even positions never
    READ the carry, which is why pos 0/2/4 all verified and this stayed hidden.
  * Position is a RUNTIME value, three `ScratchpadParameter` byte offsets rather
    than three numbers interpolated into the design source. Each position used
    to be its own xclbin, and loading one clears core `.bss` -- destroying
    `g_kprev` regardless of the first fix.

    python3 decode.py --prompt "The quick brown fox" -n 6
    python3 decode.py --tokens 128000,791,4062,14198,39935 -n 4 --host-ref

## Comparing against FLM

FLM's server applies the Instruct chat template INCLUDING a system block, and
ignores `raw`, `template`, `system` and `num_predict` — 35 tokens of fixed
overhead, measured from its own `/api/generate` `context` array, which is the
sequence it fed the model. So do not tokenize a prompt and expect FLM's answer;
take its `context`, cut it at the assistant header, and pass that to `--tokens`.
The system block carries the DATE, so capture it the same day.

    prompt "Hi", 36 tokens, captured 2026-08-02
    device [4438, 649, 358, 1520]     FLM [4438, 649, 358, 1520, 499, 3432, 30]

Four is what a 40-column cache leaves after a 36-token prompt. TSEQ=40 is the
largest this design holds — the KV tile has to fit the one operand size every
fifo shares — and it costs 5% of throughput against TSEQ=32, all of it compute.

`decode_oracle.py` diffs a generated sequence against the fp32 oracle instead,
one step at a time on a shared prefix, for prompts that need no template.

## The head

Final RMSNorm + lm_head + argmax run on the HOST, exactly (`q4nx` q4_1, the same
arithmetic every reference here uses), because no device design in this tree runs
lm_head against the model's real rows -- `gemv_bench`'s lm_head-sized build
streams tiled repeats of one tensor, which measures bytes and not logits. So the
per-token time reported below is the LAYERS dispatch, measured; the token figure
adds lm_head's own measured held-run dispatch (3010.6 us) and the measured host
term (10.5 us), and says so.

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
import fused  # noqa: E402
from fused import BLK, K_DIM, Q4NX, TSEQ, rnd  # noqa: E402
from qkv_verify import EPS  # noqa: E402

VOCAB = 128256
LM_HEAD_US = 3010.6      # the exact q4_1 lm_head dispatch, run HELD, measured
HOST_US = 10.5           # embedding row 0.38 + final RMSNorm 5.25 + argmax 4.89
FLM_TOK_S = 61.18
CHAT = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


class Head:
    """final RMSNorm -> lm_head -> argmax, exact, decoded once.

    `head_verify.logits_for` re-decodes the whole 128256x2048 tensor on every
    call, which is where its ~50 s comes from. Held, the GEMV is what is left.
    """

    def __init__(self, c):
        self.nw = c.bf16("model.norm.weight").astype(np.float32)[:K_DIM]
        self.d, self.m, self.codes = q4nx.q4nx_tensor_blocks(
            c, "lm_head.weight", (VOCAB, K_DIM))

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        xn = (x / np.sqrt((x.astype(np.float64) ** 2).mean() + EPS)
              ).astype(np.float32) * self.nw
        lg = np.empty(VOCAB, np.float32)
        for lo in range(0, VOCAB, 8192):
            hi = min(lo + 8192, VOCAB)
            lg[lo:hi] = q4nx.gemv_reference_bf16(
                xn, self.d[lo:hi], self.m[lo:hi], self.codes[lo:hi])
        return int(np.argmax(lg)), lg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="text, tokenized with the shipped tokenizer")
    g.add_argument("--tokens", help="comma-separated token ids, BOS included")
    p.add_argument("--chat", action="store_true",
                   help="wrap --prompt in the Instruct user/assistant template. "
                        "NOT FLM's context -- see the module docstring.")
    p.add_argument("-n", "--ntok", type=int, default=6, help="tokens to generate")
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--host-ref", action="store_true",
                   help="run the same loop on the host reference and diff the "
                        "token sequences. Minutes per token.")
    p.add_argument("--oracle-at", type=int, default=None,
                   help="write the device hidden state at this generated step "
                        "to compare against oracle_forward.py")
    o = p.parse_args()

    if o.prompt:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(Q4NX.parent / "tokenizer.json"))
        text = CHAT.format(p=o.prompt) if o.chat else o.prompt
        toks = tk.encode(text, add_special_tokens=not o.chat).ids
    else:
        tk, toks = None, [int(t) for t in o.tokens.split(",")]
    if len(toks) + o.ntok - 1 > TSEQ:
        raise SystemExit(
            f"{len(toks)} prompt tokens + {o.ntok} generated needs "
            f"{len(toks) + o.ntok - 1} positions; the KV cache is TSEQ={TSEQ}. "
            f"See the module docstring for why TSEQ is not a free constant.")

    c = q4nx.Q4nx(str(Q4NX))
    emb = c.bf16("model.embed_tokens.weight").astype(np.float32).reshape(-1, K_DIM)
    print(f"prompt {len(toks)} tokens: {toks}")
    print("loading lm_head ...", flush=True)
    head = Head(c)

    s = fused.Session(c, nlay=o.layers)
    # The device decodes EVERY position itself, prompt included, so the cache it
    # attends over is one it wrote -- and `g_kprev[layer]` at position t holds
    # the k' the same core computed at t-1. Nothing is staged by the host.
    out, times = [], []
    cur, pos = toks[0], 0
    while True:
        x, us = s.step(rnd(emb[cur]), pos)
        times.append(us)
        if pos + 1 < len(toks):                 # still prefilling
            cur, pos = toks[pos + 1], pos + 1
            print(f"  pos {pos - 1:2d}  prefill  tok {cur:6d}  {us:8.1f} us")
            continue
        t0 = time.perf_counter_ns()
        tok, _ = head(x)
        head_us = (time.perf_counter_ns() - t0) / 1e3
        out.append(tok)
        if o.oracle_at is not None and len(out) - 1 == o.oracle_at:
            np.save("decode_x16.npy", x.astype(np.float32))
            print(f"  x16 at generated step {o.oracle_at} -> decode_x16.npy")
        piece = tk.decode([tok]) if tk else ""
        print(f"  pos {pos:2d}  GENERATE tok {tok:6d} {piece!r:16s} "
              f"{us:8.1f} us device  {head_us / 1e3:.2f} ms host head")
        if len(out) >= o.ntok or pos + 1 >= TSEQ:
            break
        cur, pos = tok, pos + 1

    # Only the GENERATED steps are the decode rate; the prefill steps run the
    # same dispatch but there is no head on them.
    gen = np.array(times[len(toks) - 1:])
    lay = float(np.median(gen))
    tok_us = lay + LM_HEAD_US + HOST_US
    print(f"\n  generated {out}")
    if tk:
        print(f"  text      {tk.decode(out)!r}")
    print(f"  layers dispatch, held run, n={len(gen)}: median {lay:.1f} us  "
          f"min {gen.min():.1f}  max {gen.max():.1f}")
    print(f"  token = {lay:.1f} + {LM_HEAD_US} (lm_head, measured) + {HOST_US} "
          f"(host) = {tok_us:.1f} us -> {1e6 / tok_us:.1f} tok/s  "
          f"(FLM {FLM_TOK_S})")

    if o.host_ref:
        # The same loop on the host reference -- q4nx weights, the same layer
        # arithmetic, its own KV cache. `prior` is passed as EMPTY arrays rather
        # than None even at position 0, so every position takes the same code
        # path and position 0 returns its real rotated k' for the next step to
        # attend over; None would take the pos-0 shortcut and return k = None.
        import host_forward as hf
        NKV, HEAD = hf.NKV, hf.HEAD
        ref = []
        K = [np.zeros((0, NKV, HEAD)) for _ in range(o.layers)]
        V = [np.zeros((0, NKV, HEAD)) for _ in range(o.layers)]
        cur = toks[0]
        for pos in range(len(toks) - 1 + len(out)):
            x = rnd(emb[cur]).astype(np.float64)
            for L in range(o.layers):
                _, _, _, x, k, v = hf.layer_parts(c, x, L, pos, (K[L], V[L]))
                K[L] = np.concatenate([K[L], np.asarray(k, np.float64)[None]])
                V[L] = np.concatenate([V[L], np.asarray(v, np.float64)[None]])
            if pos + 1 < len(toks):
                cur = toks[pos + 1]
            else:
                cur, _ = head(x)
                ref.append(cur)
            print(f"  host pos {pos:2d} -> {cur}", flush=True)
        print(f"\n  device {out}")
        print(f"  host   {ref}")
        d = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)
        if d is None:
            print(f"  -> IDENTICAL over {min(len(out), len(ref))} tokens")
        else:
            print(f"  -> DIVERGE at generated token {d} (position "
                  f"{len(toks) - 1 + d}): device {out[d]}, host {ref[d]}. "
                  f"The prefix is shared, so this is one step's disagreement.")
        return 0 if d is None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
