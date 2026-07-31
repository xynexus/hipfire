#!/usr/bin/env python3
"""Baseline FastFlowLM end to end on the NPU: decode tok/s, prefill tok/s, achieved bandwidth.

These are the only figures in the FLM reverse/forward-engineering work that come
from timing FLM itself rather than from our own kernels, so the method is
recorded here rather than in a shell history.

Phase 2 must match these; phase 3 must beat them. Plan:
`~/flm-re-fe-mutate-goal.md`, log: `docs/npu/flm-refe-log.md`.

Two workloads, because one number conflates them:

  decode   short prompt, `gen-lim` tokens generated. FLM's own "Decoding speed".
           This is the weight-streaming case -- every token reads the whole
           weight set, so tok/s x bytes-per-token is the achieved read bandwidth.
  prefill  a long prompt loaded via `/input`, `gen-lim 1` so generation cannot
           pollute it. FLM's "Prefill speed". Compute-bound, not bandwidth-bound.

Prefill must be measured with a LONG prompt. At 52 prompt tokens FLM reports
~121 tok/s, which is almost entirely the fixed ~400 ms TTFT overhead and says
nothing about prefill throughput.

    python3 flm_bench.py                          # both models, default reps
    python3 flm_bench.py --models llama3.2:1b
    python3 flm_bench.py --reps 5 --gen-lim 256

Bandwidth needs bytes-per-token, which is model-specific and NOT the container
size -- see BYTES_PER_TOKEN.

Each rep spawns a fresh `flm run`, so the model is reloaded every time. For the
23 GB MoE that dominates wall-clock. It does not bias the numbers -- FLM's own
"Decoding speed" / "Prefill speed" exclude load -- it just makes the sweep slow.
Reusing one process across reps would need the metrics to be resettable between
rounds, which `/status` does not appear to offer; measuring cold-start-per-rep is
also the more conservative choice.
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

# Streamed weight bytes per decoded token.
#
# NOT the container size. For llama3.2:1b the .q4nx container is 1.24 GB, but
# `model.embed_tokens.weight` is BF16 [128256, 2048] = 525.3 MB and is a per-token
# *gather* of one 4 KB row, not a stream. The streamed set is the 113 I8 tensors
# = 772.3 MB, which over the 1.236 B non-embedding weights is exactly 5.00
# bits/weight. lm_head is its own I8 tensor (164.2 MB) and IS streamed.
# Derivation: docs/npu/wire-in-r6-prefill-offload.md.
#
# Qwen3.6-35B-A3B, derived the same way from its safetensors manifest (733
# tensors, 23,235.3 MB container) plus config.json. Only 8 of 256 experts run
# per token, so the container size overstates the stream by ~8.5x:
#
#   routed experts (all 256)   20132.7 MB  -> active 8/256:   629.1 MB
#   shared experts               133.7 MB  -> always      :   133.7 MB
#   attention / router / norms  1411.5 MB  -> always      :  1411.5 MB
#   lm_head                      540.3 MB  -> always      :   540.3 MB
#   embed_tokens (BF16)         1017.1 MB  -> gathered    :     0.004 MB (1 row)
#                                                          ------------
#                                             per token    :  2714.7 MB
#
# Cross-checks that this decode is right: 40 `mlp.*_exps_proj` tensors matches
# num_hidden_layers=40; 30 `linear_attn.*` and 10 `self_attn.*` matches
# full_attention_interval=4 (40/4 = 10 full-attention layers); and one routed
# expert is 1.966 MB for 3x2048x512 weights = exactly 5.00 bits/weight, the same
# rate llama3.2:1b's streamed set works out to.
#
# Note attention (1411.5) and lm_head (540.3) together are 72% of the per-token
# stream -- for this model the experts are NOT the dominant traffic. That is
# directly relevant to phase 3c (two-stage lm_head).
BYTES_PER_TOKEN = {
    "llama3.2:1b": 772.3e6,
    "qwen3.6-moe:35b-a3b": 2714.7e6,
}

DECODE_PROMPT = ("Write a long, detailed essay about the history of maritime "
                 "navigation, covering celestial navigation, the marine "
                 "chronometer, and satellite positioning.")
PREFILL_INSTRUCTION = "Summarize the text above in one sentence."

METRIC_RE = {
    "total_tokens": re.compile(r"Total tokens:\s+(\d+)"),
    "ttft_ms": re.compile(r"TTFT:\s+([\d.]+)\s*ms"),
    "prefill_tps": re.compile(r"Prefill speed:\s+([\d.]+)\s*tokens/s"),
    "decode_tps": re.compile(r"Decoding speed:\s+([\d.]+)\s*tokens/s"),
}
CHUNK_RE = re.compile(r"Prefill chunk \d+/\d+ with (\d+) tokens")


def run_flm(model, script, timeout):
    """Feed `script` to `flm run <model>` on stdin, return (stdout, seconds)."""
    t0 = time.monotonic()
    p = subprocess.run(["flm", "run", model], input=script, capture_output=True,
                       text=True, timeout=timeout)
    return p.stdout + p.stderr, time.monotonic() - t0


def parse(out):
    m = {k: None for k in METRIC_RE}
    for k, rx in METRIC_RE.items():
        hit = rx.search(out)
        if hit:
            m[k] = float(hit.group(1))
    chunks = [int(c) for c in CHUNK_RE.findall(out)]
    m["prompt_tokens"] = sum(chunks) if chunks else None
    return m


def decode_run(model, gen_lim, timeout):
    script = f"/verbose\n/set gen-lim {gen_lim}\n{DECODE_PROMPT}\n"
    out, wall = run_flm(model, script, timeout)
    m = parse(out)
    m["wall_s"] = wall
    return m, out


def prefill_run(model, prompt_file, timeout):
    # gen-lim 1 so decoding cannot contaminate the prefill measurement.
    script = f"/verbose\n/set gen-lim 1\n/input {prompt_file} {PREFILL_INSTRUCTION}\n"
    out, wall = run_flm(model, script, timeout)
    m = parse(out)
    m["wall_s"] = wall
    return m, out


def make_prompt_file(path, approx_tokens):
    """A long, non-repetitive prompt. Repetitive text is a bad prefill probe --
    it is unrepresentative and can interact with caching."""
    words = []
    seed = ("The %d-th survey vessel charted %d fathoms near the %s shelf, "
            "recording salinity %d.%d and a current bearing %d degrees. ")
    places = ("northern", "southern", "eastern", "western", "polar", "equatorial",
              "coastal", "abyssal", "continental", "oceanic")
    i = 0
    # ~1.4 tokens/word is a rough English rate; overshoot slightly and let the
    # measured prompt_tokens be the number of record.
    while len(" ".join(words).split()) < approx_tokens:
        words.append(seed % (i, 100 + i * 7 % 900, places[i % len(places)],
                             30 + i % 6, i % 10, i * 13 % 360))
        i += 1
    with open(path, "w") as f:
        f.write(" ".join(words))
    return path


def median_of(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def main():
    p = argparse.ArgumentParser(description="Baseline FLM on the NPU")
    p.add_argument("--models", default="llama3.2:1b,qwen3.6-moe:35b-a3b")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--gen-lim", type=int, default=256,
                   help="tokens generated per decode run")
    p.add_argument("--prefill-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", default=None, help="CSV path")
    p.add_argument("--keep-logs", action="store_true")
    opts = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = opts.out or os.path.join(
        here, "results", f"flm-baseline-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    prompt_file = make_prompt_file(
        os.path.join(here, "results", "prefill_prompt.txt"), opts.prefill_tokens)

    rows = []
    print(f"{'model':22s} {'workload':9s} {'n':>3s} {'prompt_tok':>10s} "
          f"{'tok/s':>9s} {'TTFT ms':>9s} {'GB/s':>7s}")
    print("-" * 76)

    for model in [m.strip() for m in opts.models.split(",") if m.strip()]:
        for workload in ("decode", "prefill"):
            runs, logs = [], []
            for i in range(opts.reps):
                try:
                    if workload == "decode":
                        m, out = decode_run(model, opts.gen_lim, opts.timeout)
                    else:
                        m, out = prefill_run(model, prompt_file, opts.timeout)
                except subprocess.TimeoutExpired:
                    print(f"{model:22s} {workload:9s} {i:3d}  TIMEOUT")
                    continue
                runs.append(m)
                logs.append(out)
            if not runs:
                continue

            key = "decode_tps" if workload == "decode" else "prefill_tps"
            tps = median_of(runs, key)
            ttft = median_of(runs, "ttft_ms")
            ptok = median_of(runs, "prompt_tokens")

            gbs = ""
            bpt = BYTES_PER_TOKEN.get(model)
            if workload == "decode" and tps and bpt:
                gbs = f"{tps * bpt / 1e9:.1f}"

            print(f"{model:22s} {workload:9s} {len(runs):3d} "
                  f"{ptok if ptok is not None else '-':>10} "
                  f"{tps if tps is not None else '-':>9} "
                  f"{ttft if ttft is not None else '-':>9} {gbs:>7s}")
            rows.append(dict(utc=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()),
                             model=model, workload=workload, reps=len(runs),
                             gen_lim=opts.gen_lim, prompt_tokens=ptok,
                             tok_s=tps, ttft_ms=ttft, gb_s=gbs,
                             all_tps=";".join(str(r.get(key)) for r in runs)))
            if opts.keep_logs:
                with open(out_path + f".{model.replace(':','_')}.{workload}.log", "w") as f:
                    f.write("\n===RUN===\n".join(logs))

    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nresults -> {out_path}")
    else:
        print("\nno results")
        sys.exit(1)


if __name__ == "__main__":
    main()
