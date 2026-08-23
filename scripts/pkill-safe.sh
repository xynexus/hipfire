#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# A drop-in `pkill` that cannot kill the shell running it.
#
# Install it as `pkill` ahead of /usr/bin on PATH (on this box, ~/.hipfire/bin)
# so it is what you get by reflex. It takes pkill's own arguments and returns
# pkill's exit codes, so existing muscle memory and existing scripts keep
# working -- they just stop being able to shoot the caller.
#
# SCOPE, MEASURED. A shell FUNCTION outranks PATH, and Claude Code installs a
# `pkill` function in its command shells that protects $CLAUDE_PID (the agent
# process). So:
#
#   * in a plain script or a fresh `bash -c`  -> this file is what runs. Verified.
#   * in a Claude Code command shell          -> the harness function runs instead.
#
# The two guards protect DIFFERENT things and neither subsumes the other: the
# harness protects the agent's own pid, this protects the CALLER SHELL and its
# process group. The caller shell is the one that actually dies with `exit 144`,
# and the harness does not cover it -- which is how this trap still landed twice
# in one session inside those very shells. Invoke this file explicitly
# (`scripts/pkill-safe.sh ...`) when you want the caller-shell guarantee.
#
# THE TRAP. `pkill -f <pat>` matches the FULL COMMAND LINE of every process,
# including the shell running the pkill, whose command line contains the pattern
# by construction. It kills its own shell and surfaces as a bare `exit 144` with
# no output, which reads like the command silently did nothing. The same hazard
# is in any hand-rolled `ps -eo args | grep | kill`, and the `[p]attern` bracket
# trick does NOT save you there -- your shell's command line genuinely contains
# the pattern via its own arguments.
#
# WHAT IT REFUSES TO SIGNAL: the caller, every ancestor up to init, and the
# caller's whole process group (which covers the subshells and `pgrep` that the
# selection itself spawns). It says so on stderr rather than skipping silently.
#
# STILL BETTER: if you SPAWNED the process, record its pid and kill that pid.
# An explicit pid cannot match your own shell. tests/agentic-gate.sh does this.
#
#   pkill hipfire-daemon              # exact-ish, same as system pkill
#   pkill -f 'python .*train\.py'     # the dangerous form, now safe
#   pkill -9 hipfire-daemon
#   pkill --self-check
set -uo pipefail

[ "${1:-}" = "--self-check" ] && { SELFCHECK=1; shift; } || SELFCHECK=0

SIG=TERM
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        # pkill's signal forms. Everything else is a pgrep selection flag and is
        # passed through untouched, so -f/-x/-u/-P/-g/-n/-o/--ns/... all work.
        -[0-9]|-[0-9][0-9]) SIG=${1#-}; shift ;;
        -SIG*|-[A-Z][A-Z]*)  SIG=${1#-}; SIG=${SIG#SIG}; shift ;;
        --signal) SIG=${2:?--signal needs a value}; SIG=${SIG#SIG}; shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done

forbidden() {           # caller + ancestors + caller's process group
    local p=$$
    while [ -n "$p" ] && [ "$p" != 0 ] && [ "$p" != 1 ]; do
        printf '%s\n' "$p"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    done
    local g; g=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    [ -n "$g" ] && ps -eo pid=,pgid= | awk -v g="$g" '$2==g {print $1}'
}

select_pids() {
    local raw; raw=$(pgrep "${args[@]}" 2>/dev/null) || true
    [ -n "$raw" ] || return 0
    local mypg; mypg=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    local bad; bad=" $(forbidden | tr '\n' ' ') "
    local p pg
    for p in $raw; do
        case "$bad" in *" $p "*) SKIPPED=$((SKIPPED+1)); continue ;; esac
        pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
        # empty pgid => already exiting (typically the pgrep we just spawned);
        # signalling a dead pid is noise at best, a recycled pid at worst.
        [ -n "$pg" ] || continue
        [ "$pg" = "$mypg" ] && { SKIPPED=$((SKIPPED+1)); continue; }
        printf '%s\n' "$p"
    done
}

if [ "$SELFCHECK" = 1 ]; then
    args=(-f "pkill-safe")
    SKIPPED=0; hits=$(select_pids | wc -l)
    [ "$hits" -eq 0 ] || { echo "FAIL: a pattern matching this very process selected $hits pid(s)"; exit 1; }
    args=(-x "definitely-not-a-real-process-xyz")
    SKIPPED=0; [ -z "$(select_pids)" ] || { echo "FAIL: bogus name matched"; exit 1; }
    echo "pkill-safe self-check OK"; exit 0
fi

[ ${#args[@]} -gt 0 ] || { echo "pkill-safe: no pattern given" >&2; exit 2; }

SKIPPED=0
mapfile -t pids < <(select_pids)
[ "$SKIPPED" -gt 0 ] && echo "pkill-safe: refused $SKIPPED self/ancestor/process-group match(es)" >&2
if [ ${#pids[@]} -eq 0 ]; then exit 1; fi     # pkill: 1 == nothing matched
for p in "${pids[@]}"; do kill "-$SIG" "$p" 2>/dev/null; done
exit 0
