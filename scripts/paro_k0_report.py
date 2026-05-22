#!/usr/bin/env python3
"""paro_k0_report.py — reduce paro_k0_eval outputs into a comparison table.

Collects:
  * paro_plus_gptq cell (this run)
  * paro_only cell (this run)
  * Optional baselines (passed via --baseline NAME=PATH)

Outputs a JSON + a markdown table for the report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


HARNESS_DEFAULT = "/workspace/hipfire/benchmarks/quality-baselines/harness"


def reduce_kldseq(path: Path) -> tuple[float, float, int]:
    sys.path.insert(0, str(Path(HARNESS_DEFAULT)))
    from kldref_format import read_per_seq_kld  # type: ignore
    m, p, n = read_per_seq_kld(str(path))
    mk = sum(m) / len(m)
    vn = [x for x in n if not math.isnan(x)]
    mn = sum(vn) / len(vn) if vn else float("nan")
    ppl = math.exp(mn) if vn and not math.isnan(mn) else float("nan")
    return mk, ppl, len(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paro-cells-dir", type=Path, required=True,
                    help="Output dir from paro_k0_eval.sh (contains paro_plus_gptq/, paro_only/).")
    ap.add_argument("--baseline", action="append", default=[],
                    help="NAME=PATH for baseline .kldseq files to include. Repeatable.")
    ap.add_argument("--meta", type=Path, default=None,
                    help="paro_meta.json from the training run; surfaces hparams in the report.")
    ap.add_argument("--out", type=Path, required=True, help="JSON output path.")
    args = ap.parse_args()

    rows: list[dict] = []

    for base in args.baseline:
        if "=" not in base:
            print(f"WARN: skipping malformed --baseline {base!r}", file=sys.stderr)
            continue
        name, path = base.split("=", 1)
        path = Path(path)
        if not path.exists():
            print(f"WARN: baseline kldseq not found: {path}", file=sys.stderr)
            continue
        kld, ppl, nc = reduce_kldseq(path)
        rows.append({"cell": name, "kind": "baseline", "kld": kld, "ppl": ppl, "n_chunk": nc, "path": str(path)})

    for cell_name in ["paro_plus_gptq", "paro_only"]:
        kldseq = args.paro_cells_dir / cell_name / "eval.kldseq"
        if not kldseq.exists():
            print(f"WARN: paro cell kldseq not found: {kldseq}", file=sys.stderr)
            continue
        kld, ppl, nc = reduce_kldseq(kldseq)
        rows.append({"cell": cell_name, "kind": "paro", "kld": kld, "ppl": ppl, "n_chunk": nc, "path": str(kldseq)})

    # Compute deltas vs each baseline, anchored to "closed_form_paper_gptq".
    anchor = next((r for r in rows if r["cell"] == "closed_form_paper_gptq"), None)
    if anchor is None:
        anchor = next((r for r in rows if r["kind"] == "baseline"), None)
    if anchor is not None:
        for r in rows:
            if r["cell"] == anchor["cell"]:
                r["kld_delta_pct"] = 0.0
                continue
            r["kld_delta_pct"] = 100.0 * (r["kld"] - anchor["kld"]) / anchor["kld"]

    meta_payload = None
    if args.meta is not None and args.meta.exists():
        with args.meta.open() as f:
            meta_payload = json.load(f)

    report = {
        "rows": rows,
        "anchor": anchor["cell"] if anchor else None,
        "meta": meta_payload,
    }
    with args.out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}\n")

    # Markdown table.
    cols = ["cell", "kld", "ppl", "kld_delta_pct", "n_chunk"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---:" if c != "cell" else "---" for c in cols]) + "|")
    for r in rows:
        kld_d = r.get("kld_delta_pct", float("nan"))
        kld_d_str = f"{kld_d:+.2f}%" if not math.isnan(kld_d) else "n/a"
        print(f"| {r['cell']} | {r['kld']:.6f} | {r['ppl']:.4f} | {kld_d_str} | {r['n_chunk']} |")


if __name__ == "__main__":
    main()
