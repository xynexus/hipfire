# Plan: collapse hipfire's binaries into one

Status: plan, 2026-08-27. Directive from the user: *"There is no longer a reason
to have hipfire-eval, hipfire-daemon, hipfire-quantize, hipfire-host-profile etc
as separate binaries"*, plus *"the idea of keeping external code requirements out
of hipfire core was a failure"*.

Target: **one shipped executable, `hipfire`**, with today's binaries as
subcommands. One exception, `hipfire-priv-helper`, handled in §5.

Companion: `2026-08-27-non-disruptive-induction-scope.md`. The two are coupled —
§3 explains why the merge is a prerequisite for the induction goal, not a
tidy-up.

---

## 1. Inventory

| crate | lines | binaries | disposition |
|---|---|---|---|
| `hipfire-cli` | 12,709 | **`hipfire`** | the host — everything lands here |
| `hipfire-quantize` | 36,266 | `main` + 5 aux | `hipfire quantize …` |
| `hipfire-eval` | 25,073 | `main` | `hipfire eval …` (already a subcommand shape) |
| `hipfire-coexistence` | 12,202 | `main` | split — see §4 |
| `hipfire-daemon` | 11,711 | `main` | `hipfire daemon` |
| `hipfire-monitor` | 3,085 | `main` | `hipfire monitor` |
| `hipfire-steer-harness` | 2,184 | `main` + `hneurons_probe` | `hipfire steer …` |
| `hipfire-admin-ui` | 1,282 | `main` | `hipfire ui admin` |
| `hipfire-atlas` | 1,178 | `main` | `hipfire atlas` |
| `hipfire-chat-ui` | 1,004 | `main` | `hipfire ui chat` |
| `hipfire-runtime` | — | `hfq`, `hipfire_host_profile` | `hipfire hfq`, `hipfire host-profile` |
| `hipfire-priv-helper` | 211 | `main` | **stays a distinct executable** (§5) |

~107,000 lines across ~20 binary targets. Note the aux binaries in
`hipfire-quantize` (`dflash_convert`, `draft_to_mq4`, `dspark_convert`,
`mq4_merge_mtp`, `mtp_extract`) — five more executables that are really
subcommands wearing binary costumes.

**Crates stay crates.** This merges *binary targets*, not libraries: `hipfire-cli`
gains dependencies on the others and dispatches into them. Nothing requires
flattening 107k lines into one crate, and doing so would make the change
unreviewable.

## 2. What the split actually costs today

Not aesthetics — three concrete failures, all observed this session.

**Process boundaries force lock contention.** Non-daemon GPU binaries must be
wrapped in `hipfire lock`, while the daemon takes the lock itself. That is why
AGENTS.md has to warn that wrapping `hipfire-eval` deadlocks *and names your own
label as the blocker*. One process, one lock holder, and the whole class of
warning disappears.

**Binaries hunt for each other on the filesystem.** `hipfire-eval` fails with
*"daemon binary not found; build with `cargo build -p hipfire-daemon --bin
hipfire-daemon`"* at four separate call sites. `find_priv_helper()` probes three
locations. Every such lookup is a deployment failure mode that in-process
dispatch cannot have.

**Subprocesses cannot be scheduled.** `induction/orchestrate.rs:264` spawns via
`std::process::Command::new`. A spawned child cannot be preempted at a quantum
boundary, admitted against a drain budget, or share a resident model — so
non-disruptive induction is unreachable while induction is a pipeline of process
spawns. This is the item that makes the merge load-bearing.

## 3. Why this unblocks induction

`2026-08-27-non-disruptive-induction-scope.md` establishes that calibration and
KV-path QAT already interleave with serving, KLD-ref needs a chunk quantum, and
quantization is unwired. All of that assumes the work happens **inside the daemon
process**. Today it does not: induction shells out, and the spawned
`hipfire-quantize` blocks on a lock the daemon holds.

So merging is step zero for the induction goal. Sequence accordingly (§6).

## 4. `hipfire-coexistence` splits rather than merges

Its stated purpose — quarantining external-ecosystem dependencies — is not borne
out. Its complete non-hipfire dependency list is `twox-hash, tokio, serde,
serde_json, sha2, base64, blake3, indicatif, chrono, libc`: **no `safetensors`,
no `gguf`, no `hf-hub`, no `tokenizers`, no `pyo3`**. Meanwhile it depends on 20
hipfire crates and nothing depends on it — a boundary only one side can see.

The honest split is not the crate line:

- **`induction/` (3,189 L)** → daemon-reachable, quanta-based. The `Command::new`
  orchestration becomes direct calls; `hipfire-quantize` is already a library
  with GPU behind an optional feature, so its codecs are callable in-process.
- **safetensors import/export, `.hfa` repack** → stay offline, no GPU. This half
  of the original separation was correct and should survive as
  `hipfire convert …` subcommands that simply never touch the daemon.
- **everything else** (artifact inspect, router profile, calibration compare) →
  ordinary subcommands.

## 5. `hipfire-priv-helper` — embedded and emitted

Per the user: store the compiled helper **inside** the `hipfire` binary and write
it out when needed. It stays a distinct *executable*; it stops being a distinct
*shipped artifact*.

Today: 211 lines, a narrow allowlist (`/sys/fs/resctrl`,
`/proc/sys/kernel/perf_event_paranoid`, module `amd_uncore`), invoked by
`hipfire doctor --fix`, elevated with `pkexec` (sudo fallback), located by
`find_priv_helper()` which probes exe-neighbour → `~/.hipfire/bin/` → `PATH`.

**Mechanism:** build the helper first, embed with `include_bytes!`, and have
`doctor --fix` materialise it on demand (write, `chmod 0755`, verify, invoke).

### ⚠️ The security constraint that decides whether this is safe

**The emitted file must not be writable by an unprivileged user at the moment it
is executed with elevated privileges.** Otherwise emit-then-`pkexec` is a local
privilege-escalation primitive: anything that can replace the file between write
and exec gets root.

This is not hypothetical for the current code — `find_priv_helper()` already
accepts `~/.hipfire/bin/hipfire-priv-helper`, a **user-writable** path. Emitting
there and elevating it would make the escalation trivial. So the design must:

- emit into a **root-owned, non-world-writable** directory (`/usr/libexec/hipfire/`
  or equivalent), which means the *emit* step itself needs privilege — do it once
  at install/first-fix time under the same elevation prompt, not per invocation;
- **verify before executing**: embed the helper's hash, re-check after write, and
  refuse if the on-disk bytes or the file's ownership/permissions do not match;
- prefer `O_EXCL` creation and an explicit mode, never write-then-chmod into a
  pre-existing path;
- keep the polkit policy pinned to that absolute path.

**Alternative worth considering and rejecting deliberately:** make the helper a
hidden subcommand (`hipfire __priv-helper`) and `pkexec` the main binary. That
removes emission entirely — no file, no TOCTOU — but runs the *whole* ~100k-line
binary as root, including its argument parsing and env-driven behaviour. That is
precisely the surface the separate helper exists to keep small, so it should only
be adopted with eyes open.

## 6. Sequence

Ordered so each step is independently shippable and the load-bearing one comes
early.

1. **`hipfire daemon`** — fold the daemon in first. It removes the four
   "daemon binary not found" call sites and the eval→daemon spawn, and it is the
   process everything else needs to be inside.
2. **`hipfire quantize`** (+ the five aux binaries as subcommands) — needed
   before induction can call codecs in-process.
3. **`induction/` into a daemon-reachable crate**, replacing `Command::new` with
   direct calls. **This is where the induction goal unblocks.** Test: induction
   still produces a byte-identical artifact, now from inside one process.
4. **`hipfire eval`** — largest at 25k lines, but mostly mechanical once the
   daemon is in-process.
5. **The small ones** — monitor, atlas, UIs, steer-harness, `hfq`,
   `host-profile`.
6. **Coexistence's offline half** → `hipfire convert …`.
7. **`priv-helper` embedding** (§5) — last, because it is the only step with a
   security-review requirement and it blocks nothing else.

## 7. Risks

- **Binary size and link time.** One executable carrying quantize + eval + daemon
  + arch crates will be large and slow to link. Feature-gating subcommands is the
  escape hatch, but it partially recreates the split — decide deliberately rather
  than discovering it at 300 MB.
- **`tokio` from coexistence.** It appears in coexistence's deps; the daemon is
  not obviously async in the same way. Check before moving `induction/`, not
  during.
- **Six arch-crate dependencies in coexistence** suggest induction reaches into
  arch specifics. Whether that survives a move into `hipfire-runtime` without a
  dependency cycle is the main structural unknown, and it may force a different
  destination crate. Resolve on paper first.
- **AGENTS.md invariants change.** The "coexistence keeps index/bytes, zero GPU"
  rule and the non-daemon-GPU-binary lock rule both need rewriting. Those are
  stated contracts — they want sign-off, not a silent edit.
- **Gates and scripts** reference binaries by name (`cargo build -p
  hipfire-daemon --bin hipfire-daemon`, `scripts/two_pass_quantize.py` wrapping
  `hipfire lock run`). Each needs updating in the step that moves its binary, or
  CI breaks in a way that looks unrelated.

## 8. What this does not change

- **Crate structure.** Libraries stay separate; only binary targets collapse.
- **The offline/online distinction.** Container translation still never touches
  the daemon — it just stops being a separate *executable*.
- **Concurrency.** One process does not mean simultaneous kernels; induction
  interleaved with serving remains alternating quanta.
