#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — baseline speed/capability matrix driver.
#
# Resumable, fault-tolerant orchestrator. For each model it generates a Hessian
# calibration sidecar (q8f16 reference -> collect_artifacts) once, then for each
# target format quantizes fresh and runs `hipfire eval --battery speed,quality`
# (which self-locks the GPU) across the KV modes. Records ok rows (prefill/decode
# tok/s) and fail rows (with reason — which combos load/fault is capability data).
# A failed cell is logged and skipped so one bad cell never wedges the run.
#
# Target formats: mq4, oq4 (plain) + mq4+, oq4+ (AWQ) + oq4++ (LDLQ/Hessian).
# Quantized .hfq scratch lives on nvme (/tmp is a small RAM tmpfs).
#
# Usage: scripts/baseline_matrix.py [--models a,b] [--formats mq4,oq4++] [--max N]
#
# Tooling only (allowed per AGENTS.md: no Python in the inference hot path).
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HF = Path("/srv/huggingface")
WORK = Path(os.path.expanduser("~/.hipfire/baseline-matrix-work"))   # .hfq scratch on nvme
CALIB_DIR = Path(os.path.expanduser("~/.hipfire/calib"))             # Hessian sidecars
def detect_arch() -> str:
    """gfx arch for this host — names the per-system matrix dir so each machine
    writes its own. Cheap (filesystem): the pre-compiled kernels dir is keyed by
    arch. Override with HIPFIRE_BENCH_ARCH."""
    v = os.environ.get("HIPFIRE_BENCH_ARCH")
    if v:
        return v
    hits = sorted(glob.glob(os.path.expanduser("~/.hipfire/kernels/gfx*")))
    return os.path.basename(hits[0]) if hits else "unknown"


ARCH = detect_arch()
OUTDIR = REPO / "benchmarks/results" / f"{ARCH}-baseline-matrix"
QUANT = REPO / "target/release/hipfire-quantize"
COLLECT = REPO / "target/release/examples/collect_artifacts"
CORPUS = REPO / "benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt"
HIPFIRE = "hipfire"  # eval via PATH (self-locks the GPU)

TARGET_FORMATS = ["mq4", "oq4", "mq4+", "oq4+", "oq4++"]
KV_MODES = ["q8", "asym4", "f16", "kvarn"]
KV_BASE = ("q8", "asym4")          # always attempted; f16/kvarn gated on a working q8 cell

# Ordered smallest -> largest. `arch` are extra hipfire-quantize flags
# (e.g. qwen2 routes to arch-id 7). All models get the TARGET_FORMATS set.
MODELS = [
    dict(name="supra-50m",        src="models--SupraLabs--Supra-50M-Instruct",        arch=[]),
    dict(name="lfm2.5-350m",      src="models--LiquidAI--LFM2.5-350M",                arch=[]),
    dict(name="qwen3-0.6b",       src="models--Qwen--Qwen3-0.6B",                     arch=[]),
    dict(name="qwen3.5-0.8b",     src="models--Qwen--Qwen3.5-0.8B",                   arch=[]),
    dict(name="llama-3.2-1b",     src="models--meta-llama--Llama-3.2-1B",             arch=[]),
    dict(name="llama-3.2-1b-inst",src="models--meta-llama--Llama-3.2-1B-Instruct",    arch=[]),
    dict(name="lfm2.5-1.2b-inst", src="models--LiquidAI--LFM2.5-1.2B-Instruct",       arch=[]),
    dict(name="qwen3.5-2b",       src="models--Qwen--Qwen3.5-2B",                     arch=[]),
    dict(name="llama-3.2-3b-inst",src="models--meta-llama--Llama-3.2-3B-Instruct",    arch=[]),
    dict(name="qwen3.5-4b",       src="models--Qwen--Qwen3.5-4B",                     arch=[]),
    dict(name="nemotron-4b",      src="models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16", arch=[]),
    dict(name="lfm2.5-8b-a1b",    src="models--LiquidAI--LFM2.5-8B-A1B",              arch=[]),
    dict(name="llama-3.1-8b-inst",src="models--meta-llama--Llama-3.1-8B-Instruct",    arch=[]),
    dict(name="qwen3.5-9b",       src="models--Qwen--Qwen3.5-9B",                     arch=[]),
]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def snapshot(src: str) -> Path | None:
    hits = sorted(glob.glob(str(HF / src / "snapshots" / "*")))
    return Path(hits[0]) if hits else None


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s\n{e.output or ''}"


def quant_spec(fmt: str, calib: str | None) -> tuple[str, list[str]]:
    """(base_format, extra_flags) for a target format token."""
    base = fmt.rstrip("+")
    if fmt.endswith("++"):
        return base, ["--ldlq", "--hessian", calib or ""]
    if fmt.endswith("+"):
        return base, ["--awq", "--hessian", calib or ""]
    return base, []


def ensure_calib(model: dict, quant_timeout: int, calib_timeout: int) -> str | None:
    """Build a q8f16 reference and a Hessian .calib.hfq for this model (cached).

    q8f16 keeps the embedding at Q8 (qt=3), which collect_artifacts handles, and
    is near-lossless so the Hessian is a high-quality reference.
    """
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    calib = CALIB_DIR / f"{model['name']}.calib.hfq"
    if calib.exists():
        return str(calib)
    src = snapshot(model["src"])
    if src is None:
        return None
    ref = WORK / f"{model['name']}.q8f16ref.hfq"
    if not ref.exists():
        log(f"  calib: build q8f16 reference for {model['name']} ...")
        rc, out = run([QUANT, "--input", src, "--output", ref, "--format", "q8f16", *model["arch"]], quant_timeout)
        if rc != 0 or not ref.exists():
            log(f"  calib: q8f16 ref FAILED: {out[-200:]}")
            return None
    log(f"  calib: collect_artifacts for {model['name']} ...")
    rc, out = run([COLLECT, "--model", ref, "--corpus", CORPUS, "--output", calib, "--max-tokens", "128"], calib_timeout)
    if calib.exists():
        return str(calib)
    log(f"  calib: collect_artifacts FAILED: {out[-300:]}")
    return None


def quantize(model: dict, fmt: str, calib: str | None, out: Path, timeout: int) -> tuple[bool, str]:
    if out.exists():
        return True, "cached"
    src = snapshot(model["src"])
    if src is None:
        return False, f"no local source: {model['src']}"
    base, extra = quant_spec(fmt, calib)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [QUANT, "--input", src, "--output", out, "--format", base, *model["arch"], *extra]
    rc, log_out = run(cmd, timeout)
    ok = rc == 0 and out.exists()
    return ok, ("ok" if ok else f"rc={rc}: {log_out[-300:]}")


def latest_run_dir(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if "/eval-results/runs/" in line and Path(line).exists():
            return Path(line)
    runs = sorted(glob.glob(os.path.expanduser("~/.hipfire/eval-results/runs/*")), key=os.path.getmtime)
    return Path(runs[-1]) if runs else None


def extract_metrics(run_dir: Path) -> dict:
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


def kldref_for(name: str) -> str | None:
    for p in (CALIB_DIR / f"{name}.calib.hfq",):  # calib bundles the kldref slice
        if p.exists():
            return str(p)
    return None


def state_path() -> Path:
    return OUTDIR / "state.json"


def load_state() -> dict:
    p = state_path()
    return json.loads(p.read_text()) if p.exists() else {"done": [], "failed": [], "rows": []}


def save_state(st: dict) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(st, indent=2))


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
        "# gfx1151 baseline matrix\n\nHost: apu_uma:gfx1151. Generated by scripts/baseline_matrix.py.\n"
        "Formats: mq4, oq4 (plain) + mq4+, oq4+ (AWQ) + oq4++ (LDLQ/Hessian).\n\n"
        + hdr + "\n".join(lines) + "\n")
    with (OUTDIR / "result-data.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_system_json(run_dir: Path) -> None:
    """Write OUTDIR/system.json (arch/hardware_bucket/rocm) from an eval manifest
    once — gives gen_benchmarks.py a rich per-system label."""
    sj = OUTDIR / "system.json"
    if sj.exists():
        return
    man = run_dir / "manifest.json"
    if not man.exists():
        return
    try:
        d = json.loads(man.read_text())
    except Exception:
        return
    hp = d.get("host_profile", {})
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sj.write_text(json.dumps({
        "arch": d.get("arch") or ARCH,
        "rocm": d.get("rocm"),
        "hardware_bucket": hp.get("hardware_bucket"),
        "cu_count": hp.get("cu_count"),
        "vram_bytes": hp.get("vram_bytes"),
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma list to restrict")
    ap.add_argument("--formats", default="", help="comma list to restrict (default: all 5)")
    ap.add_argument("--max", type=int, default=0, help="stop after N new cells (0=all)")
    ap.add_argument("--quant-timeout", type=int, default=3600)
    ap.add_argument("--calib-timeout", type=int, default=3600)
    ap.add_argument("--eval-timeout", type=int, default=1200)
    args = ap.parse_args()

    only_m = set(filter(None, args.models.split(",")))
    formats = list(filter(None, args.formats.split(","))) or TARGET_FORMATS

    st = load_state()
    done = set(tuple(c) for c in st["done"])
    failed = set(tuple(c) for c in st["failed"])
    rows = list(st.get("rows", []))
    new = 0

    empty = dict(prefill_tok_s=None, decode_tok_s=None, kld=None, ppl=None,
                 quality_status=None, speed_status=None, speed_reason=None)

    def record(model, fmt, kv, status, note, hfq_mb, metrics):
        rows.append(dict(model=model, format=fmt, kv=kv, hfq_mb=hfq_mb,
                         status=status, note=note, **metrics))
        done.add((model, fmt, kv))
        if status != "ok":
            failed.add((model, fmt, kv))
        st["done"] = [list(c) for c in done]
        st["failed"] = [list(c) for c in failed]
        st["rows"] = rows
        save_state(st)
        write_table(rows)

    for model in MODELS:
        if only_m and model["name"] not in only_m:
            continue
        if snapshot(model["src"]) is None:
            log(f"SKIP {model['name']}: no local source")
            continue
        calib = None
        calib_tried = False
        # which (model, fmt) already worked on q8 (for the f16/kvarn gate)
        working_q8 = {(r["model"], r["format"]) for r in rows
                      if r.get("status") == "ok" and r.get("kv") == "q8"}

        for fmt in formats:
            needs_calib = fmt.endswith("+")  # + or ++
            out_hfq = WORK / f"{model['name']}.{fmt}.hfq"
            quant_ok = None
            quant_why = ""
            for kv in KV_MODES:
                c = (model["name"], fmt, kv)
                if c in done or c in failed:
                    continue
                if kv not in KV_BASE and (model["name"], fmt) not in working_q8:
                    continue  # only probe f16/kvarn where the q8 cell worked
                if needs_calib and calib is None:
                    if not calib_tried:
                        log(f"ensure calib for {model['name']} ...")
                        calib = ensure_calib(model, args.quant_timeout, args.calib_timeout)
                        calib_tried = True
                    if calib is None:
                        record(model["name"], fmt, kv, "fail", "calib generation failed", None, dict(empty))
                        new += 1
                        continue
                if quant_ok is None:
                    log(f"quantize {model['name']} {fmt} ...")
                    quant_ok, quant_why = quantize(model, fmt, calib, out_hfq, args.quant_timeout)
                    if not quant_ok:
                        log(f"  QUANT FAIL {model['name']} {fmt}: {quant_why}")
                if not quant_ok:
                    record(model["name"], fmt, kv, "fail", ("quant: " + quant_why)[:120], None, dict(empty))
                    new += 1
                    continue
                hfq_mb = round(out_hfq.stat().st_size / 1e6, 1) if out_hfq.exists() else None
                log(f"eval {c} ...")
                found, metrics, info = eval_cell(out_hfq, kv, kldref_for(model["name"]), args.eval_timeout)
                if found:
                    write_system_json(Path(info))
                speed_ok = bool(found and metrics.get("speed_status") == "pass"
                                and metrics.get("decode_tok_s") is not None)
                status = "ok" if speed_ok else "fail"
                note = "" if speed_ok else (metrics.get("speed_reason") or info or "")[:120]
                record(model["name"], fmt, kv, status, note, hfq_mb, metrics)
                if speed_ok:
                    working_q8 = {(r["model"], r["format"]) for r in rows
                                  if r.get("status") == "ok" and r.get("kv") == "q8"}
                    log(f"  OK {c}: prefill={metrics.get('prefill_tok_s')} decode={metrics.get('decode_tok_s')}")
                else:
                    log(f"  FAIL {c}: {note}")
                new += 1
                if args.max and new >= args.max:
                    log(f"reached --max {args.max}; stopping")
                    return 0
    ok = sum(1 for r in rows if r.get("status") == "ok")
    log(f"complete: {len(done)} cells done, {len(failed)} failed, {ok} working, {new} new this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
