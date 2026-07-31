#!/usr/bin/env python3
"""Scan a running process for AIE2P transaction binaries built at run time.

FLM's per-layer dispatch is not an embedded blob -- it is generated at run time
by `llama_npu_sequence::gen_layer_seq()` into an `npu_sequence` object (see
txn_scan.py for how that was established). So it can only be recovered by
reading it out of the live process.

This avoids reversing the XRT / amdxdna submit ABI entirely: the sequence is
materialised in ordinary process memory before submission, and a TXN header is
self-validating (the op walk must land exactly on `txn_size` AND produce exactly
`num_ops`). So walking anonymous memory with that same validator finds it.

    # terminal 1
    flm run llama3.2:1b
    # terminal 2
    python3 txn_memscan.py --name flm --dump captured/

Anything matching a transaction already present in FLM's shared objects is
flagged KNOWN (that is the generic DDR<->memtile staging ladder); anything else
is new and is what we are after.

Read-only. Requires ptrace permission on the target (same user is normally
enough; otherwise /proc/sys/kernel/yama/ptrace_scope).
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txn_scan import walk, classify, describe_bds, find_txns  # noqa: E402

# Skip only what cannot hold a freshly built sequence. Do NOT skip /dev/
# mappings: XRT buffer objects are mmap'd from /dev/accel/accel0, and the
# instruction buffer handed to the NPU is exactly such a BO -- excluding them
# was why the first version of this scan found only the staging blobs already
# present in the .so files.
SKIP_PATH_RE = re.compile(r"\.(so|so\.\d+.*)$|\[vvar|\[vdso|\[vsyscall")
MAX_REGION = 2 << 30


def regions(pid):
    out = []
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split(maxsplit=5)
            rng, perms = parts[0], parts[1]
            path = parts[5].strip() if len(parts) > 5 else ""
            if "r" not in perms:
                continue
            if path and SKIP_PATH_RE.search(path):
                continue
            lo, hi = (int(x, 16) for x in rng.split("-"))
            if hi - lo > MAX_REGION:
                continue
            out.append((lo, hi, perms, path or "[anon]"))
    return out


def read_region(mem, lo, hi):
    try:
        mem.seek(lo)
        return mem.read(hi - lo)
    except (OSError, ValueError, OverflowError):
        return None


def known_digests(lib_paths):
    """Digests of every transaction already embedded in FLM's shared objects."""
    known = {}
    for p in lib_paths:
        try:
            d = open(p, "rb").read()
        except OSError:
            continue
        for hdr, _ops in find_txns(d):
            blob = d[hdr["offset"]:hdr["offset"] + hdr["txn_size"]]
            known[hashlib.sha256(blob).hexdigest()] = (os.path.basename(p), hdr)
    return known


def pids_for(name):
    try:
        out = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
        return [int(x) for x in out.stdout.split()]
    except (OSError, ValueError):
        return []


def main():
    p = argparse.ArgumentParser(description="Scan a live process for AIE2P TXN binaries")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pid", type=int)
    g.add_argument("--name", help="process name pattern (pgrep -f)")
    g.add_argument("--launch", metavar="CMD",
                   help="spawn CMD and scan it. Needed when "
                        "/proc/sys/kernel/yama/ptrace_scope is 1: the tracer must be "
                        "an ANCESTOR of the tracee, so attaching to a sibling or to a "
                        "pre-existing `flm serve` fails with EPERM.")
    p.add_argument("--stdin", metavar="TEXT", default=None,
                   help="text to feed --launch on stdin (use \\n for newlines)")
    p.add_argument("--settle", type=float, default=25.0,
                   help="seconds to let the launched process load and start work")
    p.add_argument("--passes", type=int, default=3,
                   help="repeat scans, to catch buffers that are only transiently live")
    p.add_argument("--dump", metavar="DIR")
    p.add_argument("--libs", nargs="*", default=None,
                   help="shared objects whose transactions count as KNOWN")
    p.add_argument("-v", "--verbose", action="store_true")
    o = p.parse_args()

    child = None
    if o.launch:
        import shlex
        import time
        child = subprocess.Popen(shlex.split(o.launch), stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 text=True)
        if o.stdin:
            try:
                child.stdin.write(o.stdin.replace("\\n", "\n"))
                child.stdin.flush()
            except BrokenPipeError:
                pass
        print(f"launched pid {child.pid}; settling {o.settle}s")
        time.sleep(o.settle)
        pids = [child.pid]
    else:
        pids = [o.pid] if o.pid else pids_for(o.name)
    if not pids:
        sys.exit(f"no process matching {o.name!r}")

    libs = o.libs
    if libs is None:
        libdir = "/opt/fastflowlm/lib"
        libs = [os.path.join(libdir, f) for f in os.listdir(libdir)
                if f.endswith(".so")] if os.path.isdir(libdir) else []
    known = known_digests(libs)
    print(f"{len(known)} known transaction(s) across {len(libs)} shared object(s)")

    seen, new = set(), 0
    import time as _time
    for _p in range(max(1, o.passes)):
      if _p:
        _time.sleep(3)
        print(f"\n--- pass {_p + 1} ---")
      for pid in pids:
          try:
              mem = open(f"/proc/{pid}/mem", "rb", 0)
          except OSError as e:
              print(f"pid {pid}: {e}", file=sys.stderr)
              continue
          regs = regions(pid)
          print(f"\n=== pid {pid}: scanning {len(regs)} region(s) ===")
          print(f"{'addr':>16s} {'cols':>4s} {'ops':>5s} {'bytes':>7s}  "
                f"{'DMA programmed':<34s} {'DDR bytes':>12s}  status")
          with mem:
              for lo, hi, perms, path in regs:
                  d = read_region(mem, lo, hi)
                  if not d:
                      continue
                  for hdr, ops in find_txns(d):
                          if True:
                              i = hdr["offset"]
                              blob = d[i:i + hdr["txn_size"]]
                              dig = hashlib.sha256(blob).hexdigest()
                              if dig in seen:
                                  i += hdr["txn_size"]
                                  continue
                              seen.add(dig)
                              counts, rows, dma, bds = classify(d, hdr, ops)
                              info = describe_bds(bds)
                              ddr = sum(b["bytes"] for b in info
                                        if b["valid"] and b["kind"] == "shim")
                              dma_s = ", ".join(f"{t}:{dn}x{n}"
                                                for (t, dn), n in sorted(dma.items())) or "-"
                              if dig in known:
                                  status = f"KNOWN ({known[dig][0]})"
                              else:
                                  status = "*** NEW ***"
                                  new += 1
                              print(f"{lo + i:#16x} {hdr['cols']:4d} {hdr['num_ops']:5d} "
                                    f"{hdr['txn_size']:7d}  {dma_s:<34s} {ddr:12,d}  {status}")
                              if o.verbose:
                                  print(f"{'':16s} rows touched: "
                                        f"{','.join(str(x) for x in sorted(rows))}   "
                                        f"ops: " + ", ".join(f"{hex(k)}x{v}"
                                                             for k, v in sorted(counts.items())))
                              if o.dump and dig not in known:
                                  os.makedirs(o.dump, exist_ok=True)
                                  fn = f"pid{pid}_{lo + i:#x}_c{hdr['cols']}_{hdr['num_ops']}ops.txn"
                                  with open(os.path.join(o.dump, fn), "wb") as f:
                                      f.write(blob)
    if child is not None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
    print(f"\n{len(seen)} distinct transaction(s); {new} NEW")


if __name__ == "__main__":
    main()
