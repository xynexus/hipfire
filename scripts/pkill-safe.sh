#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# Kill processes without killing yourself.
#
# THE TRAP THIS EXISTS FOR. `pkill -f <pattern>` matches the FULL COMMAND LINE of
# every process — including the shell running the pkill, whose command line
# contains the pattern by construction. It kills its own shell and surfaces as a
# bare `exit 144` with no output, which reads like the command silently did
# nothing. AGENTS.md documents it; it is still easy to walk into, because the
# same hazard appears in any hand-rolled `ps -eo args | grep | kill` — the
# `[p]attern` bracket trick does NOT save you there, since your shell's command
# line genuinely contains the pattern via its own arguments.
#
# This script excludes the caller AND its whole ancestor chain, always, in both
# modes. It is also LOUD: it names every pid it signals and says so when nothing
# matched, because silence is what makes the original failure confusing.
#
# BEST OPTION IS STILL NOT THIS SCRIPT. If you SPAWNED the process, record its
# pid and kill that pid — an explicit pid cannot match your own shell. See
# tests/agentic-gate.sh, which tracks the daemon it starts. Reach for this only
# for a process you did not spawn.
#
#   scripts/killproc.sh hipfire-daemon hipfire-eval   # exact names (pgrep -x)
#   scripts/killproc.sh --pattern 'python .*train\.py'
#   scripts/killproc.sh --dry-run hipfire-daemon
#   scripts/killproc.sh --kill hipfire-daemon         # SIGKILL after 5s grace
#   scripts/killproc.sh --self-check
set -uo pipefail

DRY=0; SIG=TERM; HARD=0; MODE=exact; PATTERN=""
usage() { sed -n '5,30p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }

# Everything it must never signal: the caller, every ancestor up to init, and
# the caller's whole PROCESS GROUP. The group matters because a selector also
# matches the caller's own DESCENDANTS -- the subshells and `pgrep` that the
# selection itself spawns carry the pattern in their command lines. Ancestor
# exclusion alone does not cover those; the self-check below caught exactly that
# (a self-matching pattern selected 3 pids on the first draft).
ancestors() {
    local p=$$
    while [ -n "$p" ] && [ "$p" != 0 ] && [ "$p" != 1 ]; do
        printf '%s\n' "$p"
        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    done
    local mypg; mypg=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    [ -n "$mypg" ] && ps -eo pid=,pgid= | awk -v g="$mypg" '$2==g {print $1}'
}

self_check() {
    local excl; excl=$(ancestors | tr '\n' ' ')
    case " $excl " in *" $$ "*) ;; *) echo "FAIL: ancestors omits self"; exit 1;; esac
    local n; n=$(ancestors | wc -l)
    [ "$n" -ge 2 ] || { echo "FAIL: ancestors found $n, expected >=2 (self+parent)"; exit 1; }
    # a pattern matching THIS shell must select nothing after exclusion
    local hits; hits=$(select_pids pattern 'killproc' | wc -l)
    [ "$hits" -eq 0 ] || { echo "FAIL: self-matching pattern selected $hits pid(s)"; exit 1; }
    # and a name that cannot exist selects nothing, without error
    [ -z "$(select_pids exact 'definitely-not-a-real-process-xyz')" ] || { echo "FAIL: bogus name matched"; exit 1; }
    echo "killproc self-check OK (excluded: $excl)"
}

select_pids() {         # $1=mode $2=target -> safe-to-signal pids
    local mode=$1 target=$2 raw
    if [ "$mode" = exact ]; then raw=$(pgrep -x -- "$target" 2>/dev/null)
    else raw=$(pgrep -f -- "$target" 2>/dev/null); fi
    [ -n "$raw" ] || return 0
    local mypg; mypg=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    local anc; anc=$(ancestors)
    local p pg
    for p in $raw; do
        case " $(printf '%s ' $anc) " in *" $p "*) continue ;; esac
        pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
        # Empty pgid = the process is already gone. `pgrep` lists short-lived
        # children of the selection itself (the pgrep/subshell), which exit
        # before we can read their group; signalling a dead pid is at best noise
        # and at worst hits a recycled pid, so drop them.
        [ -n "$pg" ] || continue
        [ "$pg" = "$mypg" ] && continue
        printf '%s\n' "$p"
    done
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --kill)    HARD=1; shift ;;
        --pattern) MODE=pattern; PATTERN=${2:?--pattern needs a regex}; shift 2 ;;
        --self-check) self_check; exit 0 ;;
        -h|--help) usage 0 ;;
        --) shift; break ;;
        -*) echo "killproc: unknown flag $1" >&2; usage 2 ;;
        *)  break ;;
    esac
done

targets=()
[ "$MODE" = pattern ] && targets=("$PATTERN") || targets=("$@")
[ ${#targets[@]} -gt 0 ] || { echo "killproc: nothing to do (no target given)" >&2; usage 2; }

total=0
for t in "${targets[@]}"; do
    pids=$(select_pids "$MODE" "$t")
    if [ -z "$pids" ]; then echo "killproc: no match for '$t'"; continue; fi
    for p in $pids; do
        cmd=$(ps -o args= -p "$p" 2>/dev/null | cut -c1-70)
        if [ "$DRY" = 1 ]; then echo "killproc: WOULD signal $p  ($cmd)"
        else echo "killproc: SIG$SIG -> $p  ($cmd)"; kill "-$SIG" "$p" 2>/dev/null; fi
        total=$((total+1))
    done
done

if [ "$HARD" = 1 ] && [ "$DRY" = 0 ] && [ "$total" -gt 0 ]; then
    sleep 5
    for t in "${targets[@]}"; do
        for p in $(select_pids "$MODE" "$t"); do
            echo "killproc: still alive, SIGKILL -> $p"; kill -9 "$p" 2>/dev/null
        done
    done
fi
echo "killproc: signalled $total process(es)"
exit 0
