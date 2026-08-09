#!/usr/bin/env python3
"""M1b check for the v2 daemon plan: sampling no longer shares process state.

The plan's stated exit — two seeds sampled in one *batched* step matching their
solo runs — cannot be run yet, because batched decode is greedy-only precisely
BECAUSE the RNG was global. That is the circularity M1b breaks. What is testable
now, and what actually gates the later stages:

  1. greedy output is byte-identical before/after the change (no regression);
  2. temperature > 0 is reproducible across identical requests;
  3. a temperature > 0 request is unaffected by another generation running
     between it and its repeat — the property a shared RNG could not offer.

(3) is the real one. Under the old global, an interleaved request advanced the
same stream, so the repeat drew from a different point and diverged.

Diagnostics tooling, so Python is fine here (AGENTS.md rule 1). The daemon
self-locks; do NOT wrap in `hipfire lock`.
"""

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAEMON = os.path.join(REPO, "target", "release", "hipfire-daemon")
MODEL = os.path.expanduser("~/.hipfire/models/qwen3.5-0.8b--oq4++.hfq")
PROMPT = "Write one short sentence about the sea."


class Daemon:
    def __init__(self):
        self.proc = subprocess.Popen(
            [DAEMON], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def send(self, frame):
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()

    def collect(self, terminals, collect=frozenset()):
        got = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("daemon closed stdout")
            try:
                f = json.loads(line)
            except json.JSONDecodeError:
                continue
            if f.get("type") in collect:
                got.append(f)
            if f.get("type") == "error":
                raise RuntimeError(f"daemon error: {f.get('message')}")
            if f.get("type") in terminals:
                return got

    def generate(self, rid, temperature, max_tokens=48):
        self.send({
            "type": "generate", "id": rid, "prompt": PROMPT,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        toks = self.collect({"done"}, collect={"token"})
        return "".join(t.get("text", "") for t in toks)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


def main():
    if not os.path.exists(DAEMON):
        print(f"missing {DAEMON}", file=sys.stderr)
        return 2

    d = Daemon()
    failures = []
    try:
        d.send({"type": "load", "id": "l", "model": MODEL, "params": {"max_seq": 2048}})
        d.collect({"loaded"})

        # PRE-EXISTING, NOT AN RNG ISSUE: the first generation after a load
        # differs from every later one. Measured here as gen0 != gen1 while
        # gen1..gen4 are all byte-identical, and independently in the M0 trace
        # gate, where rep 0 emitted 254 tokens against 256 for every later rep.
        # Greedy decoding never draws from the sampler RNG, so this predates and
        # is unrelated to M1b — but it does mean any determinism check has to
        # warm up first or it measures the load, not the sampler.
        d.generate("warmup", 0.0)

        print("== 1. greedy is deterministic")
        g1 = d.generate("g1", 0.0)
        g2 = d.generate("g2", 0.0)
        print(f"   {g1[:70]!r}")
        if g1 != g2:
            failures.append("greedy output differs between identical requests")
        else:
            print("   OK")

        print("\n== 2. temperature>0 is reproducible across identical requests")
        t1 = d.generate("t1", 0.8)
        t2 = d.generate("t2", 0.8)
        print(f"   {t1[:70]!r}")
        print(f"   {t2[:70]!r}")
        if t1 != t2:
            failures.append(
                "temperature>0 not reproducible: each request seeds its own "
                "stream from a fixed constant, so identical requests must match"
            )
        else:
            print("   OK")

        print("\n== 3. an interleaved generation does not perturb the repeat")
        # This is the property the global RNG could not provide.
        t3 = d.generate("t3", 0.8)
        _ = d.generate("interleaved", 0.9, max_tokens=32)
        t4 = d.generate("t4", 0.8)
        if t3 != t4:
            failures.append(
                "a generation run in between changed a temperature>0 result — "
                "sampling state is still shared across requests"
            )
        else:
            print("   OK: identical across an interleaved request")
    finally:
        d.close()

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("M1b gate PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
