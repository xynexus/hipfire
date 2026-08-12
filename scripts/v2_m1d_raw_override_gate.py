#!/usr/bin/env python3
"""M1d regression gate: prompt framing must not leak between requests.

`RAW_OVERRIDE` was a thread_local Cell set ONLY by the plain-generate handler and
read by `qwen35_materialize_batch_prefill_prompt` via `effective_raw()`. The batch
prefill path never set it and nothing cleared it, so a batch prefill inherited
whatever the last unrelated `generate` left behind.

Measured on gfx1103 before the fix, for one identical `prefix_hash_preflight`
session:

  fresh daemon                      15 tokens, 3 boundaries, c8427f59...
  after one `generate` raw:true      7 tokens, 1 boundary,  12317032...

Those hashes are the KV-reuse cache keys, so this changed what a later request
matched in the prefix cache — not merely how a prompt was framed.

Exit: both preflights identical. Diagnostics tooling, so Python is fine here
(AGENTS.md rule 1). The daemon self-locks; do NOT wrap in `hipfire lock`.
"""
import json, os, subprocess, sys

DAEMON = "/home/sadara/hipfire/target/release/hipfire-daemon"
MODEL = os.path.expanduser("~/.hipfire/models/qwen3.5-0.8b--oq4++.hfq")

SESSION = {
    "id": "s0",
    "prompt": "Describe the ocean in one sentence.",
    "assistant_prefix": "",
    "max_think_tokens": 0,
    "semantic_boundary_checkpoints": False,
    "state_handle": {
        "state_kinds": ["attention_kv", "deltanet_recurrent"],
        "logical_position": 0,
        "cached_prefix_tokens": 0,
    },
}
PREFLIGHT = {"type": "prefix_hash_preflight", "id": "pf1", "session": SESSION,
             "model": MODEL}


class D:
    def __init__(self):
        self.p = subprocess.Popen([DAEMON], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)

    def send(self, f):
        self.p.stdin.write(json.dumps(f) + "\n"); self.p.stdin.flush()

    def until(self, kinds):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("daemon closed")
            try:
                f = json.loads(line)
            except json.JSONDecodeError:
                continue
            if f.get("type") in kinds or f.get("type") == "error":
                return f

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=15)
        except Exception:
            self.p.kill()


def preflight(with_raw_generate):
    d = D()
    try:
        d.send({"type": "load", "id": "l", "model": MODEL, "params": {"max_seq": 2048}})
        r = d.until({"loaded"})
        if r.get("type") == "error":
            return ("load-error", r.get("message"))
        if with_raw_generate:
            d.send({"type": "generate", "id": "g", "prompt": "hi",
                    "temperature": 0.0, "max_tokens": 1, "raw": True})
            d.until({"done"})
        d.send(PREFLIGHT)
        return ("ok", d.until({"prefix_hash_preflight_done", "prefix_hash_preflight",
                               "prefix_hash_candidates", "prefill_ready", "error"}))
    finally:
        d.close()


def main():
    clean = preflight(False)
    after = preflight(True)
    for label, r in (("fresh daemon", clean), ("after generate raw:true", after)):
        if r[0] != "ok" or r[1].get("type") == "error":
            print(f"FAIL: {label}: {r[1]}")
            return 2
        full = r[1].get("full", {})
        print(f"{label:26} prefix_len={full.get('prefix_len')} hash={full.get('value')}")
    if clean[1] != after[1]:
        print("\nFAIL: an unrelated `generate` changed a later batch prefill's framing")
        return 1
    print("\nM1d gate PASS: framing is per-request, not inherited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
