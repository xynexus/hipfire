#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""aiecost CLI.

    python -m aiecost probe                 # H-series provenance report
    python -m aiecost probe --json          # machine-readable
    python -m aiecost probe --csv out.csv   # durable row per claim

Run from tools/npu/ (or with tools/npu on PYTHONPATH).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from . import calib, device


def cmd_probe(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(device.provenance(args.device), indent=2, default=str))
        return 0
    if args.csv:
        rep = device.provenance(args.device)
        rows = rep["claims"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["device", "git_commit", "xrt_version"])
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        **r,
                        "value": json.dumps(r["value"]) if isinstance(r["value"], dict) else r["value"],
                        "device": rep["device"],
                        "git_commit": rep["git_commit"],
                        "xrt_version": rep["xrt"].get("xrt_version", ""),
                    }
                )
        print(f"wrote {len(rows)} claims -> {args.csv}")
        return 0
    print(device.render(args.device))
    return 0


def cmd_calib(args: argparse.Namespace) -> int:
    key = args.key or calib.current_key()
    consts = calib.load(key)
    print(f"calibration key: {key}")
    if not consts:
        print("  (none — run the benches; the model will refuse to predict)")
        return 0
    for name, c in sorted(consts.items()):
        flag = "" if c.admissible else "   [NOT ADMISSIBLE]"
        print(f"\n  {name} = {c.value:g} {c.unit}   [{c.bench}]{flag}")
        print(f"    method: {c.method}")
        for e in c.evidence:
            print(f"    evidence: {e}")
        for cv in c.caveats:
            print(f"    caveat:   {cv}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from . import model
    from .spec import ScheduleSpec

    raw = json.loads(Path(args.spec).read_text())
    specs = [ScheduleSpec(**s) for s in (raw if isinstance(raw, list) else [raw])]
    for s in specs:
        if isinstance(s.mmul_shape, list):
            s.mmul_shape = tuple(s.mmul_shape)
    if len(specs) == 1:
        print(model.predict(specs[0], args.key, args.device).render())
        return 0
    print(f"ranking {len(specs)} candidates (fastest first):\n")
    for i, (s, p) in enumerate(model.rank(specs, args.key, args.device), 1):
        head = f"{p.device_s * 1e6:10.3f} us" if (p.admissible and p.buildable) else "     ---   "
        print(f"{i:2}. {head}  {s.name}  [{p.limiter if p.admissible else 'refused'}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aiecost", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="H-series device facts + claims register")
    pr.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    pr.add_argument("--json", action="store_true", help="machine-readable report")
    pr.add_argument("--csv", metavar="PATH", help="durable one-row-per-claim CSV")
    pr.set_defaults(fn=cmd_probe)

    ca = sub.add_parser("calib", help="show calibration constants + their evidence")
    ca.add_argument("--key", help="version key (default: this device)")
    ca.set_defaults(fn=cmd_calib)

    pd = sub.add_parser("predict", help="predict a ScheduleSpec (JSON: one object or a list to rank)")
    pd.add_argument("spec", help="path to spec JSON")
    pd.add_argument("--key", help="version key (default: this device)")
    pd.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    pd.set_defaults(fn=cmd_predict)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
