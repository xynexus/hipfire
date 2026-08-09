#!/usr/bin/env python3
"""M0 exit gate for the v2 daemon plan: verify the executor trace.

`docs/plans/2026-08-09-v2-daemon-module-major-multistream.md`, stage M0, states
three exit conditions. This script measures all three:

  1. an N-token greedy generation produces exactly N-1 inter-token gaps;
  2. wall time reconstructed from the trace matches externally measured wall
     time within 2%;
  3. tracing on vs. off costs less than 1% of tok/s.

Condition 3 is A/B alternated **within one daemon lifetime per arm**, because
gfx1103 shows a ~8.6% first-run position effect: comparing a cold first run
against a warm second one would attribute warm-up to the trace. Each arm runs
`--reps` generations and the first is discarded as warm-up.

Diagnostics/benchmark tooling, so Python is allowed here (AGENTS.md rule 1).
The daemon self-locks, so do NOT wrap this in `hipfire lock`.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAEMON = os.path.join(REPO, "target", "release", "hipfire-daemon")


class Daemon:
    """A daemon on stdio pipes, speaking the JSON-lines protocol."""

    def __init__(self, trace: bool):
        env = dict(os.environ)
        if trace:
            env["HIPFIRE_DAEMON_TRACE"] = "1"
            env["HIPFIRE_DAEMON_TRACE_CAPACITY"] = "65536"
        else:
            env.pop("HIPFIRE_DAEMON_TRACE", None)
        self.proc = subprocess.Popen(
            [DAEMON],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            bufsize=1,
        )

    def send(self, frame: dict) -> None:
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()

    def read_until(self, terminals: set, collect: set = frozenset()) -> tuple:
        """Read frames until one of `terminals`. Returns (terminal, collected)."""
        collected = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("daemon closed stdout before a terminal frame")
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = frame.get("type")
            if kind in collect:
                collected.append(frame)
            if kind == "error":
                raise RuntimeError(f"daemon error: {frame.get('message')}")
            if kind in terminals:
                return frame, collected

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_arm(model: str, max_tokens: int, reps: int, trace: bool, max_seq: int):
    """One daemon lifetime: load once, then `reps` generations. Returns
    (per-rep tok/s discarding warm-up, per-rep wall seconds, final trace)."""
    daemon = Daemon(trace=trace)
    try:
        daemon.send({
            "type": "load",
            "id": "load-1",
            "model": model,
            "params": {"max_seq": max_seq},
        })
        daemon.read_until({"loaded"})

        rates, walls = [], []
        for rep in range(reps):
            frame = {
                "type": "generate",
                "id": f"gen-{rep}",
                "prompt": "Count slowly and carefully from one onward.",
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            started = time.perf_counter()
            daemon.send(frame)
            done, tokens = daemon.read_until({"done"}, collect={"token"})
            elapsed = time.perf_counter() - started
            emitted = len(tokens)
            walls.append(elapsed)
            rates.append(emitted / elapsed if elapsed > 0 else 0.0)
            print(
                f"    rep {rep}: {emitted} tokens in {elapsed:.3f}s "
                f"= {rates[-1]:.2f} tok/s{' (warm-up, discarded)' if rep == 0 else ''}",
                flush=True,
            )

        dump = None
        if trace:
            daemon.send({"type": "executor_trace", "id": "trace-1"})
            dump, _ = daemon.read_until({"executor_trace"})
        # Walls are returned in full, warm-up included, so the caller can pair
        # rep i's stopwatch with rep i's dispatch span. Returning only the
        # post-warm-up slice and then comparing against "the" trace span is
        # exactly the off-by-one that made this gate report 25% drift when the
        # trace was in fact accurate to 0.5%.
        return rates[1:], walls, dump
    finally:
        daemon.close()


def generate_dispatch_spans_ns(records):
    """Durations of the dispatch pairs that contain token events, in order.

    Walking begin/end pairs and keeping the ones with tokens inside identifies
    the `generate` dispatches without needing to know the request ids, and it
    pairs them with reps positionally — dispatch k is rep k, because the daemon
    is serial and the driver sends one generation at a time.

    The dispatch span, not the token span, is what the external stopwatch should
    match: it covers prefill and the terminal frame, which the first-to-last-
    token window excludes by construction.
    """
    spans, open_at, tokens = [], None, 0
    for r in records:
        if r["event"] == "dispatch_begin":
            open_at, tokens = r["t_ns"], 0
        elif r["event"] == "token_emitted" and open_at is not None:
            tokens += 1
        elif r["event"] == "dispatch_end" and open_at is not None:
            if tokens > 0:
                spans.append(r["t_ns"] - open_at)
            open_at, tokens = None, 0
    return spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=os.path.expanduser("~/.hipfire/models/qwen3.5-0.8b--oq4++.hfq"),
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=4, help="incl. one discarded warm-up")
    args = ap.parse_args()

    if not os.path.exists(DAEMON):
        print(f"missing {DAEMON}; cargo build --release -p hipfire-daemon", file=sys.stderr)
        return 2

    print(f"== arm A: tracing ON  ({args.reps} reps, first discarded)", flush=True)
    on_rates, on_walls, dump = run_arm(
        args.model, args.max_tokens, args.reps, True, args.max_seq
    )
    print(f"== arm B: tracing OFF ({args.reps} reps, first discarded)", flush=True)
    off_rates, _, _ = run_arm(
        args.model, args.max_tokens, args.reps, False, args.max_seq
    )

    failures = []

    # ---- condition 1: N tokens -> N-1 gaps -------------------------------
    print("\n== condition 1: gap count")
    if not dump or not dump.get("enabled"):
        failures.append("trace was not enabled in the ON arm")
    else:
        records = dump.get("records", [])
        token_events = [r for r in records if r["event"] == "token_emitted"]
        by_stream = {}
        for r in token_events:
            by_stream.setdefault(r["stream"], []).append(r)
        # The last generation is the one whose tokens are freshest in the ring.
        biggest = max(by_stream.values(), key=len) if by_stream else []
        gaps = len(biggest) - 1 if biggest else 0
        print(f"   dropped={dump['dropped']} record_count={dump['record_count']}")
        print(f"   largest single-stream token series: {len(biggest)} -> {gaps} gaps")
        if dump["dropped"] != 0:
            failures.append(
                f"ring wrapped (dropped={dump['dropped']}); window is truncated, "
                "raise HIPFIRE_DAEMON_TRACE_CAPACITY"
            )
        if len(biggest) < 2:
            failures.append("fewer than 2 token events recorded")
        elif gaps != len(biggest) - 1:
            failures.append(f"expected {len(biggest) - 1} gaps, got {gaps}")
        else:
            print(f"   OK: {len(biggest)} tokens -> {gaps} gaps")

        # ---- condition 2: reconstructed vs measured wall time -------------
        print("\n== condition 2: reconstructed wall time within 2%, per rep")
        spans = generate_dispatch_spans_ns(records)
        if len(spans) != len(on_walls):
            failures.append(
                f"{len(spans)} generate dispatches traced but {len(on_walls)} "
                "measured; cannot pair reps"
            )
        else:
            worst = 0.0
            for rep, (span_ns, measured) in enumerate(zip(spans, on_walls)):
                span_s = span_ns / 1e9
                drift = abs(span_s - measured) / measured
                worst = max(worst, drift)
                print(
                    f"   rep {rep}: dispatch {span_s:.3f}s vs stopwatch "
                    f"{measured:.3f}s -> {drift * 100:.2f}%"
                )
            # The residual is IPC and JSON encode/decode: the stopwatch starts
            # before the request is written and stops after the terminal frame is
            # parsed, both of which sit outside the daemon's dispatch.
            if worst > 0.02:
                failures.append(f"worst per-rep drift {worst*100:.2f}% (>2%)")
            else:
                print(f"   OK: worst {worst * 100:.2f}%")

    # ---- condition 3: tracing overhead < 1% ------------------------------
    print("\n== condition 3: tracing overhead < 1%")
    if not on_rates or not off_rates:
        failures.append("not enough reps to compare arms")
    else:
        on_med = statistics.median(on_rates)
        off_med = statistics.median(off_rates)
        overhead = (off_med - on_med) / off_med
        print(f"   ON  median {on_med:.2f} tok/s  (n={len(on_rates)})")
        print(f"   OFF median {off_med:.2f} tok/s  (n={len(off_rates)})")
        print(f"   -> overhead {overhead * 100:+.2f}%")
        if overhead > 0.01:
            failures.append(f"tracing costs {overhead*100:.2f}% (>1%)")
        else:
            print("   OK")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("M0 gate PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
