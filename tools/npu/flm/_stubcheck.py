"""Assert a stub build changes ONLY the kernel bindings.

The whole attribution rests on the DMA being identical between the real build
and the stub builds -- same fifos, sizes, counts, order, every acquire and
release. This diffs the design source `build()` would exec at each FUSED_STUB
setting against the unstubbed one, with the fifo-name tag normalised, and fails
if anything except the inserted `ExternalFunction(..._stub, ...)` lines differs.

    python3 _stubcheck.py
"""
import difflib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
# Print the generated design source instead of compiling it.
DUMP = ("import sys, pathlib; sys.path.insert(0, '.');"
        "s = pathlib.Path('fused.py').read_text()"
        ".replace('    exec(src, ns)', '    print(src); raise SystemExit(0)');"
        "g = {'__name__': '_d', '__file__': 'fused.py'};"
        "exec(compile(s, 'fused.py', 'exec'), g);"
        "g['build'](0, 16, 1)")


def gen(mode):
    r = subprocess.run([sys.executable, "-c", DUMP], cwd=HERE, text=True,
                       capture_output=True,
                       env={**os.environ, "FUSED_STUB": mode})
    if r.returncode:
        raise SystemExit(r.stderr)
    # the tag carries the mode into every fifo NAME (that is what busts iron's
    # AST cache); normalise it so the diff shows structure, not the tag.
    out = r.stdout.replace(f"S{mode}_", "_") if mode else r.stdout
    return out.splitlines()


base, bad = gen(""), 0
for mode in ("ab", "c", "all"):
    d = [ln for ln in difflib.unified_diff(base, gen(mode), lineterm="", n=0)
         if ln[:1] in "+-" and ln[:3] not in ("---", "+++") and ln[1:].strip()]
    gone = [ln for ln in d if ln[0] == "-"]
    new = [ln for ln in d if ln[0] == "+"]
    stub = [ln for ln in new if "_stub" in ln or ln[1:].strip().startswith("arg_types")]
    ok = not gone and len(new) == len(stub) and stub
    bad += not ok
    print(f"  FUSED_STUB={mode:<4} -{len(gone)} +{len(new)} "
          f"({len(stub)} kernel bindings)   {'OK' if ok else 'DIFFERS ELSEWHERE'}")
    for ln in gone + [ln for ln in new if ln not in stub]:
        print("     ", ln)
print("  -> stub builds change only the kernel bindings" if not bad else "  -> FAIL")
raise SystemExit(1 if bad else 0)
