#!/usr/bin/env bash
# hipfire-memguard — run a command, and kill IT (not the box) if free memory runs out.
#
# Why this exists, and why the two obvious fixes do not work on halo:
#
#   * `oom_score_adj` only chooses a victim once the kernel is ALREADY out of
#     memory. It was tried: the journal shows `hipfire-quantiz` killed twice at
#     oom_score_adj:1000 (total-vm 139 GB and 155 GB) and the box still went
#     down, because the OOM killer then walked the rest of the user session.
#   * `MemoryMax=` on a cgroup never fires. Measured 2026-09-04 on a 2B qtip3
#     build: cgroup memory.current sat at 7.36 GB for the whole run while system
#     used climbed 5.6 -> 106.3 GB. KFD/amdgpu buffer objects are charged to no
#     memcg and to no process RSS, so neither the cap nor the badness score can
#     see them.
#
# What does work is a threshold on MemAvailable: kill the child while several GB
# are still free, so the system never reaches global OOM and no cascade starts.
# The memory comes back on process exit (verified: 120.7 GB available after).
#
# Usage:  hipfire-memguard [-t GB] [-i SECS] -- <command> [args...]
# Exit:   the child's exit status, or 137 if the guard killed it.
set -uo pipefail

THRESHOLD_GB=${HIPFIRE_MEMGUARD_GB:-20}
INTERVAL=${HIPFIRE_MEMGUARD_INTERVAL:-2}
while [ $# -gt 0 ]; do
  case "$1" in
    -t) THRESHOLD_GB=$2; shift 2 ;;
    -i) INTERVAL=$2; shift 2 ;;
    --) shift; break ;;
    *)  break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: hipfire-memguard [-t GB] [-i SECS] -- <command>" >&2; exit 2; }

avail_gb() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }

# `<&0` is load-bearing: backgrounding with `&` in a non-interactive shell
# redirects the child's stdin to /dev/null, so any command that READS stdin
# (`hipfire-daemon < requests.jsonl`) saw EOF immediately and exited 0 having
# done nothing. Inherit the caller's stdin explicitly.
"$@" <&0 &
child=$!
# An explicit recorded pid, never a pattern: `pkill -f` matches this script's own
# command line via its arguments and kills the caller's shell (AGENTS.md).
killed=0
(
  while kill -0 "$child" 2>/dev/null; do
    a=$(avail_gb)
    if [ "$a" -lt "$THRESHOLD_GB" ]; then
      echo "[memguard] MemAvailable ${a}GB < ${THRESHOLD_GB}GB — killing pid $child" >&2
      kill -9 "$child" 2>/dev/null
      # Mark it for the parent; the child's own status will be 137.
      touch "/tmp/.memguard-killed.$child" 2>/dev/null
      break
    fi
    sleep "$INTERVAL"
  done
) &
watchdog=$!

wait "$child"; rc=$?
kill "$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
if [ -e "/tmp/.memguard-killed.$child" ]; then
  rm -f "/tmp/.memguard-killed.$child"
  echo "[memguard] child $child was killed to protect the system (low MemAvailable)" >&2
  exit 137
fi
exit $rc
