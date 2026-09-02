#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — speculative decode must reproduce autoregressive output (GPU).
#
# Speculative decode's defining guarantee is that the verify makes it lossless:
# the emitted sequence is whatever the target would have produced on its own,
# whatever the draft block width. This gate asserts exactly that, by generating
# the same prompt greedily at several block widths and comparing token streams.
#
#   ./tests/spec-ar-equivalence-gate.sh [model.hfq]
#
# ⚠️ THIS GATE CURRENTLY FAILS. That is the point — it is the executable
# reproducer for docs/bugs/2026-09-01-spec-decode-not-output-equivalent-to-ar.md
# (the batched verify's slot 0 disagrees with the single-token forward). It is
# deliberately NOT wired into tests/no-gpu-ci.sh or the affected-gate, because a
# gate that is red for a known unfixed reason trains people to ignore gates.
# Wire it in when the bug is fixed; until then run it by hand to check progress.
#
# Why the existing gates cannot see this: tiny-state-gate.sh hashes decode
# output but never varies the block width, and until the spine-discard fix the
# drafter-free path never produced a block wider than 1, so no configuration in
# CI ever exercised a real speculative block.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/srv/hipfire/models/qwen3.5-0.8b--oq4++.hfq}"
BIN=./target/release/hipfire
TOKENS="${SPEC_AR_TOKENS:-300}"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

[ -x "$BIN" ] || { echo "build first: cargo build --release -p hipfire-cli"; exit 2; }
[ -f "$MODEL" ] || { echo "SKIP: model not found: $MODEL"; exit 0; }

# A code prompt: highly predictable, so the n-gram actually fills a block.
python3 - "$W" "$MODEL" "$TOKENS" <<'PY'
import json,sys
W,model,n = sys.argv[1], sys.argv[2], int(sys.argv[3])
src = open('crates/hipfire-specdecode-ngram/src/hot.rs').read()[:6151]
p = 'Here is a Rust module:\n\n' + src + '\n\nExplain what this module does, in detail.'
m = [{'type':'load','model':model,'params':{'max_seq':8192}},
     {'type':'generate','id':'g1','prompt':p,'max_tokens':n,'temperature':0.0},
     {'type':'unload'}]
open(f'{W}/req.jsonl','w').write('\n'.join(json.dumps(x) for x in m) + '\n')
PY

hash_of() {  # log -> sha of the emitted token stream
  python3 - "$1" <<'PY'
import json,sys,hashlib
out=[]
for line in open(sys.argv[1]):
    line=line.strip()
    if not line.startswith('{'): continue
    try: d=json.loads(line)
    except Exception: continue
    if d.get('type')=='token': out.append(d.get('text',''))
print(hashlib.sha256(''.join(out).encode()).hexdigest()[:16], len(out))
PY
}

echo "== spec/AR equivalence: $(basename "$MODEL"), $TOKENS tokens, greedy =="
HIPFIRE_NGRAM_SPEC=0 "$BIN" daemon < "$W/req.jsonl" > "$W/ar.log" 2>&1
read -r REF_H REF_N < <(hash_of "$W/ar.log")
printf '  %-10s %s (n=%s)  <- reference\n' "AR" "$REF_H" "$REF_N"

rc=0
for b in 2 4 16; do
  HIPFIRE_NGRAM_SPEC=1 HIPFIRE_SPEC_BLOCK=$b "$BIN" daemon < "$W/req.jsonl" > "$W/b$b.log" 2>&1
  read -r H N < <(hash_of "$W/b$b.log")
  if [ "$H" = "$REF_H" ]; then
    printf '  %-10s %s (n=%s)  OK\n' "b=$b" "$H" "$N"
  else
    printf '  %-10s %s (n=%s)  FAIL — differs from AR\n' "b=$b" "$H" "$N"
    rc=1
  fi
done

if [ $rc -ne 0 ]; then
  echo "spec-ar-equivalence-gate: FAIL — speculation changed the output."
  echo "  See docs/bugs/2026-09-01-spec-decode-not-output-equivalent-to-ar.md"
  echo "  Diagnose with HIPFIRE_DFLASH_VERIFY_DEBUG=1 and compare slot 0's argmax"
  echo "  at one start_pos between a b=1 window and a b>=2 window."
else
  echo "spec-ar-equivalence-gate: PASS"
fi
exit $rc
