#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — baseline speed/capability matrix driver.
#
# Resumable, fault-tolerant orchestrator: for each (model, format, kv) cell it
# quantizes fresh from the local HF source (skipping if the .hfq already exists),
# runs `hipfire eval --battery speed,quality` (which self-locks the GPU), and
# appends a row to the matrix. A failed quant/eval is logged and skipped so a
# single bad cell never wedges the run. Re-running resumes from the recorded
# state. Ordered smallest -> largest; formats grouped plain -> +awq -> ++ldlq.
#
# Usage: scripts/baseline_matrix.py --phase plain|awq|ldlq [--models a,b] [--max N]
#
# Tooling only (allowed per AGENTS.md: no Python in the inference hot path).
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HF = Path("/srv/huggingface")
WORK = Path(os.path.expanduser("~/.hipfire/baseline-matrix-work"))  # .hfq scratch on nvme (/tmp is a small RAM tmpfs)
REFS = REPO / "benchmarks/quality-baselines/refs"
OUTDIR = REPO / "benchmarks/results" / f"gfx1151-baseline-matrix"
QUANT = REPO / "target/release/hipfire-quantize"
HIPFIRE = "hipfire"  # eval via PATH (self-locks the GPU)
KV_MODES = ["q8", "asym4"]                     # f16/kvarn probed per-arch later

# Ordered smallest -> largest. `src` is the HF repo dir under /srv/huggingface;
# `arch` are extra hipfire-quantize flags (e.g. qwen2 must route to arch-id 7).
# `formats` is the plain (loader-supported) candidate set for the family.
MODELS = [
    dict(name="supra-50m",        src="models--SupraLabs--Supra-50M-Instruct", arch=[], formats=["mq4", "mq6", "mq3", "hfq4"]),
    dict(name="lfm2.5-350m",      src="models--LiquidAI--LFM2.5-350M",         arch=[], formats=["mq4", "mq6", "mq3", "hfq4"]),
    dict(name="qwen3-0.6b",       src="models--Qwen--Qwen3-0.6B",              arch=[], formats=["mq4", "mq6", "mq3", "hfq4", "q8f16"]),
    dict(name="qwen3.5-0.8b",     src="models--Qwen--Qwen3.5-0.8B",            arch=[], formats=["q8f16", "mq6", "mq4", "mq3", "hfq4"]),
    dict(name="lfm2.5-1.2b-inst", src="models--LiquidAI--LFM2.5-1.2B-Instruct",arch=[], formats=["mq4", "mq6", "mq3", "hfq4"]),
    dict(name="lfm2.5-1.2b-think",src="models--LiquidAI--LFM2.5-1.2B-Thinking",arch=[], formats=["mq4", "mq6", "mq3"]),
    dict(name="qwen3.5-2b",       src="models--Qwen--Qwen3.5-2B",              arch=[], formats=["q8f16", "mq6", "mq4", "mq3"]),
    dict(name="qwen3.5-4b",       src="models--Qwen--Qwen3.5-4B",              arch=[], formats=["q8f16", "mq6", "mq4", "mq3"]),
    dict(name="lfm2.5-8b-a1b",    src="models--LiquidAI--LFM2.5-8B-A1B",       arch=[], formats=["mq6", "mq4", "mq3"]),
    dict(name="qwen3.5-9b",       src="models--Qwen--Qwen3.5-9B",              arch=[], formats=["q8f16", "mq6", "mq4", "mq3"]),
    dict(name="llama-3.2-1b",      src="models--meta-llama--Llama-3.2-1B",          arch=[], formats=["q8f16", "hfq4", "mq4", "mq3"]),
    dict(name="llama-3.2-1b-inst", src="models--meta-llama--Llama-3.2-1B-Instruct", arch=[], formats=["q8f16", "hfq4", "mq4", "mq3"]),
    dict(name="llama-3.2-3b-inst", src="models--meta-llama--Llama-3.2-3B-Instruct", arch=[], formats=["q8f16", "hfq4", "mq4", "mq3"]),
    dict(name="llama-3.1-8b-inst", src="models--meta-llama--Llama-3.1-8B-Instruct", arch=[], formats=["hfq4", "mq4", "mq3"]),
    dict(name="nemotron-4b",       src="models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16", arch=[], formats=["mq6", "mq4", "mq3"]),
]

AWQ_FLAGS = ["--awq"]                           # the "+" pass
LDLQ_FLAGS = ["--ldlq"]                         # the "++" pass (needs --hessian sidecar)


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def snapshot(src: str) -> Path | None:
    hits = sorted(glob.glob(str(HF / src / "snapshots" / "*")))
    return Path(hits[0]) if hits else None


def kldref_for(name: str) -> str | None:
    for p in (REFS / f"{name}-bf16.kldref.hfq", REFS / f"{name}.kldref.hfq"):
        if p.exists():
            return str(p)
    return None


def state_path() -> Path:
    return OUTDIR / "state.json"


def load_state() -> dict:
    p = state_path()
    return json.loads(p.read_text()) if p.exists() else {"done": [], "failed": []}


def save_state(st: dict) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(st, indent=2))


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s\n{e.output or ''}"


def quantize(model: dict, fmt: str, extra: list[str], out: Path, timeout: int) -> tuple[bool, str]:
    if out.exists():
        return True, "cached"
    src = snapshot(model["src"])
    if src is None:
        return False, f"no local source: {model['src']}"
    out.parent.mkdir(parents=True, exist_ok=True)
    base_fmt = fmt.rstrip("+")
    cmd = [str(QUANT), "--input", str(src), "--output", str(out), "--format", base_fmt, *model["arch"], *extra]
    rc, log_out = run(cmd, timeout)
    ok = rc == 0 and out.exists()
    return ok, ("ok" if ok else f"rc={rc}: {log_out[-400:]}")


def latest_run_dir(stdout: str) -> Path | None:
    # eval prints the run dir as a line; fall back to newest under eval-results.
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if "/eval-results/runs/" in line and Path(line).exists():
            return Path(line)
    runs = sorted(glob.glob(os.path.expanduser("~/.hipfire/eval-results/runs/*")), key=os.path.getmtime)
    return Path(runs[-1]) if runs else None


def extract_metrics(run_dir: Path) -> dict:
    """Pull speed (prefill/decode tok/s) + quality (KLD/PPL) from results.jsonl."""
    out = {"prefill_tok_s": None, "decode_tok_s": None, "kld": None, "ppl": None,
           "quality_status": None, "speed_status": None, "speed_reason": None}
    jl = run_dir / "results.jsonl"
    if not jl.exists():
        return out
    for line in jl.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        m = d.get("metrics") or {}
        if d.get("battery") == "speed":
            out["speed_status"] = d.get("status")
            if d.get("reason"):
                out["speed_reason"] = d.get("reason")
            if m.get("decode_tok_s") is not None:
                out["prefill_tok_s"] = m.get("prefill_tok_s")
                out["decode_tok_s"] = m.get("decode_tok_s")
        if d.get("battery") == "quality":
            out["quality_status"] = d.get("status")
            for k in ("mean_kld", "kld", "ppl", "perplexity"):
                if m.get(k) is not None:
                    out["kld" if "kld" in k else "ppl"] = m[k]
    return out


def eval_cell(out_hfq: Path, kv: str, kldref: str | None, timeout: int) -> tuple[bool, dict, str]:
    cmd = [HIPFIRE, "eval", "--model", str(out_hfq), "--battery", "speed,quality",
           "--kv-mode", kv, "--tier", "fast"]
    if kldref:
        cmd += ["--kldref", kldref]
    rc, stdout = run(cmd, timeout)
    rd = latest_run_dir(stdout)
    if rd is None:
        return False, {}, f"rc={rc}: no run dir\n{stdout[-300:]}"
    return True, extract_metrics(rd), str(rd)


def write_table(rows: list[dict]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    hdr = "| Model | Format | KV | Status | Prefill tok/s | Decode tok/s | Mean KLD | PPL | hfq MB | Note |\n"
    hdr += "|---|---|---|---|---:|---:|---:|---:|---:|---|\n"
    def cell(v): return "" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))
    def esc(s): return (s or "").replace("|", "/").replace("\n", " ").strip()
    lines = []
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"], r["format"], r["kv"], r.get("status", ""),
            cell(r.get("prefill_tok_s")), cell(r.get("decode_tok_s")),
            cell(r.get("kld")), cell(r.get("ppl")), cell(r.get("hfq_mb")),
            esc(r.get("note")),
        ]) + " |")
    (OUTDIR / "result-table.md").write_text(
        f"# gfx1151 baseline matrix\n\nHost: apu_uma:gfx1151. Generated by scripts/baseline_matrix.py.\n\n"
        + hdr + "\n".join(lines) + "\n")
    with (OUTDIR / "result-data.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["plain", "awq", "ldlq"], default="plain")
    ap.add_argument("--models", default="", help="comma list to restrict")
    ap.add_argument("--max", type=int, default=0, help="stop after N new cells (0=all)")
    ap.add_argument("--quant-timeout", type=int, default=3600)
    ap.add_argument("--eval-timeout", type=int, default=1200)
    args = ap.parse_args()

    only = set(filter(None, args.models.split(",")))
    extra = {"plain": [], "awq": AWQ_FLAGS, "ldlq": LDLQ_FLAGS}[args.phase]
    suffix = {"plain": "", "awq": "+", "ldlq": "++"}[args.phase]

    st = load_state()
    done = set(tuple(c) for c in st["done"])
    failed = set(tuple(c) for c in st["failed"])
    rows = list(st.get("rows", []))
    new = 0

    # For +awq / ++ldlq, only target (model, base_format) pairs whose PLAIN
    # variant produced a working speed row — AWQ/Hessian calibration is slow and
    # a base format that faults at load will fault the same way after AWQ/LDLQ.
    plain_ok = {(r["model"], r["format"]) for r in rows
                if r.get("status") == "ok" and not str(r["format"]).endswith(("+", "++"))}

    for model in MODELS:
        if only and model["name"] not in only:
            continue
        if snapshot(model["src"]) is None:
            log(f"SKIP {model['name']}: no local source")
            continue
        kldref = kldref_for(model["name"])
        for fmt in model["formats"]:
            fmt_tag = fmt + suffix
            if args.phase in ("awq", "ldlq") and (model["name"], fmt) not in plain_ok:
                continue  # base format didn't work in plain; skip the slow +/++ pass
            out_hfq = WORK / f"{model['name']}.{fmt_tag}.hfq"
            quant_ok = None
            quant_why = ""
            for kv in KV_MODES:
                cell = (model["name"], fmt_tag, kv)
                if cell in done or cell in failed:
                    continue
                if args.phase == "ldlq":
                    # ++ needs a Hessian sidecar; deferred to the Hessian phase.
                    log(f"DEFER {cell}: ldlq/Hessian phase not yet wired")
                    continue
                empty = dict(prefill_tok_s=None, decode_tok_s=None, kld=None, ppl=None,
                             quality_status=None, speed_status=None, speed_reason=None)
                if quant_ok is None:
                    log(f"quantize {model['name']} {fmt_tag} ...")
                    quant_ok, quant_why = quantize(model, fmt_tag, extra, out_hfq, args.quant_timeout)
                    if not quant_ok:
                        log(f"  QUANT FAIL {model['name']} {fmt_tag}: {quant_why}")
                if not quant_ok:
                    rows.append(dict(model=model["name"], format=fmt_tag, kv=kv, hfq_mb=None,
                                     status="fail", note=("quant: " + quant_why)[:120], **empty))
                    done.add(cell); failed.add(cell); new += 1
                    st["done"] = [list(c) for c in done]; st["failed"] = [list(c) for c in failed]; st["rows"] = rows
                    save_state(st); write_table(rows)
                    continue
                hfq_mb = round(out_hfq.stat().st_size / 1e6, 1) if out_hfq.exists() else None
                log(f"eval {cell} ...")
                found, metrics, info = eval_cell(out_hfq, kv, kldref, args.eval_timeout)
                speed_ok = bool(found and metrics.get("speed_status") == "pass"
                                and metrics.get("decode_tok_s") is not None)
                status = "ok" if speed_ok else "fail"
                note = "" if speed_ok else (metrics.get("speed_reason") or info or "")[:120]
                rows.append(dict(model=model["name"], format=fmt_tag, kv=kv, hfq_mb=hfq_mb,
                                 status=status, note=note, **metrics))
                done.add(cell)
                if speed_ok:
                    log(f"  OK {cell}: prefill={metrics.get('prefill_tok_s')} decode={metrics.get('decode_tok_s')} q={metrics.get('quality_status')}")
                else:
                    failed.add(cell)
                    log(f"  FAIL {cell}: {note}")
                new += 1
                st["done"] = [list(c) for c in done]; st["failed"] = [list(c) for c in failed]; st["rows"] = rows
                save_state(st); write_table(rows)
                if args.max and new >= args.max:
                    log(f"reached --max {args.max}; stopping")
                    return 0
    log(f"phase '{args.phase}' complete: {len(done)} done, {len(failed)} failed, {new} new this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
