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

**Binaries hunt for each other on the filesystem.** Counted rather than
estimated: the "daemon binary not found" string appears 12 times, behind 7
message-producing `find_daemon_bin*` sites, with 22 call sites across 8 crates.
The saving grace is that resolution has exactly ONE owner —
`find_daemon_bin_candidates()` — so the whole class has a single chokepoint.
Its last two candidates are a repo-root `target/` located by shelling out to
`git rev-parse`, which is why a deployed install outside a repo had one working
path and no fallback. `find_priv_helper()` probes three locations. Every such
lookup is a deployment failure mode that in-process dispatch cannot have.

*Closed in step 1* — see §6.

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

1. **`hipfire daemon`** — ✅ **DONE** (`6101e3f07`). What it took, versus what
   this section guessed:

   - The lib split was a *rename*, not the ~1,500-line code move it looked like:
     the submodules already `use crate::*`, which resolves against the crate root
     whether that root is `main.rs` or `lib.rs`. So `git mv src/main.rs
     src/lib.rs`, `fn main` → `pub fn main`, and a 3-line `src/bin/` shim. The
     standalone `hipfire-daemon` binary still builds and behaves identically.
   - The daemon still runs as its **own OS process**. That is what made this step
     small: the `process::exit` calls, the startup panics, the stdin lock and the
     stdout responder are all *correct* for a process whose whole job is to be
     the daemon, so none of them had to be converted to `Result`.
   - `current_exe()` went to the FRONT of the candidate chain, ahead of anything
     on disk, so a running `hipfire` can never spawn an older build of itself
     left in `~/.hipfire/bin` or a stale `target/`. Spawn dispatches on the file
     name (`hipfire` ⇒ pass `daemon`), the ordinary multi-call convention, which
     also makes `HIPFIRE_DAEMON_BIN=/path/to/hipfire` work. **Zero churn at the
     22 call sites** — they still receive a `PathBuf`.
   - Verified on halo/gfx1151 from a copy of `hipfire` in a non-repo directory,
     with no `HIPFIRE_DAEMON_BIN` and no `hipfire-daemon` beside it: `hipfire
     chat` spawned `<that path>/hipfire daemon` and generated 12 tokens at
     24.56 tok/s.

   **Size, measured rather than feared** (§7 asked for this deliberately):
   `hipfire` 36.9 → 59.1 MB, replacing the 36.9 + 33.0 = 69.9 MB pair. The merged
   binary is **10.8 MB smaller** than what it subsumes; shared code dedupes.

   **The trap this re-armed.** The daemon's 96 unit tests followed the code into
   the lib target, so `cargo test -p hipfire-daemon --bin hipfire-daemon` now
   matches **zero** tests and still exits 0 — the same silent-green failure the
   comment above that line was written to warn about, exactly inverted. Any step
   that moves a crate's target layout must re-check the test COUNT, because both
   spellings pass. `no-gpu-ci.sh` moved to `--lib` (96 pass) and `ci.yml`'s
   workspace-wide `cargo test --lib` now picks them up for free, which it never
   could while the crate was bin-only.

   **Deferred to rung 2 (not done here):** `hipfire-daemon-adapter` still spawns
   a child. Collapsing it to an in-process `DaemonTransport` is blocked on a real
   structural problem this section did not anticipate: the trait and all three
   impls are **private** to the adapter, and `hipfire-daemon` already depends on
   `hipfire-daemon-adapter` — so an in-process arm inside the adapter would need
   the reverse edge and close a dependency cycle. That needs the shared pieces
   (`fatal_startup_error`, `acquire_resource_lease_or_exit`, `default_socket_path`
   — all daemon-SERVER functions the client adapter should never have owned) to
   move out first.

   *Original text follows.* Fold the daemon in first. It removes the four
   "daemon binary not found" call sites and the eval→daemon spawn, and it is the
   process everything else needs to be inside.

   **This is also where `hipfire-daemon-adapter` collapses**, and that is the
   bulk of the step rather than a detail of it. The adapter is 2,506 lines whose
   whole job is talking to the daemon as a *child process*:
   `StdioTransport::spawn` at `lib.rs:193` runs `tokio::process::Command::new`
   and pipes JSONL over stdin/stdout. Seven crates depend on it — cli, server,
   eval, coexistence, coherence, steer-harness, and `hipfire-daemon` itself — so
   it is the widest blast radius in the sequence.

   The seam already exists. `DaemonTransport` is a trait with three
   implementations: `StdioTransport` (child process), `SocketTransport`
   (`lib.rs:345`, already not a spawn), and a test `MockTransport`. So in-process
   means adding a fourth peer, not rewriting seven call sites and not inventing
   an abstraction. Keep the spawn path for the cases that still want a separate
   process — crash isolation, a daemon outliving the CLI.

   The catch, and the reason this is step 1's real work: `DaemonTransport`'s
   methods return `BoxFuture`, while the daemon handlers behind them are
   synchronous. An in-process transport has to bridge that without blocking a
   runtime worker — the constraint §7 names.
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
- **The async/sync seam is the daemon's, not coexistence's.** The original
  worry — that `tokio` arrives as a new dependency from coexistence — does not
  survive checking, on either half. The merge host is already a tokio process:
  `hipfire-cli/src/main.rs:190` is `#[tokio::main]` with
  `tokio = { features = ["full"] }`, and it embeds `hipfire-server` (axum,
  tokio-stream, async-stream). Coexistence asks only for `rt-multi-thread`, a
  strict subset, so folding it in does not even move the feature union. Nor does
  its tokio travel with `induction/`: there is exactly one use site in the crate,
  `hipfire-coexistence/src/main.rs:643`, building a runtime for `hipfire hub`
  fetch/verify/repair — squarely the offline half of §4 — while
  `crates/hipfire-coexistence/src/induction/` contains no `async`, no `.await`,
  and no `tokio` at all.

  The real mismatch runs the other way and lands in **step 1**, not step 3.
  `hipfire-daemon/src/main.rs:1523` is a plain `fn main()`: no tokio, serial
  executor over process globals. Folding it into `hipfire` puts that blocking,
  GPU-owning loop on a tokio worker of the same runtime serving axum, where a
  multi-second kernel sweep starves request handling.

  It must get a **dedicated OS thread**, and `spawn_blocking` is not an
  alternative — an earlier draft of this line said it was, and that was wrong.
  `hipfire_rdna::Gpu` is `!Send` and `!Sync` (three raw `*mut c_void` fields, no
  `unsafe impl` anywhere in the tree), so the handle cannot be built on one
  thread and moved to another: the executor thread has to call `Gpu::init()`
  itself. Tokio's blocking pool does not pin work to a thread, so it cannot make
  that guarantee. Done this way in step 1.
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

---

## Status — executed 2026-08-27

All seven steps addressed on `feat/daemon-subcommand`, merged to master in #367
(`38c215e9d`). SHAs below are the merged ones.
`hipfire` is one executable carrying every former binary as a subcommand:
`daemon quantize convert eval monitor atlas steer hneurons-probe hfq
host-profile`. All 17 standalone bin targets still build and run, so nothing
that invokes them by name broke.

| step | state | commit |
|---|---|---|
| 1 daemon | done | `6101e3f07` |
| 2 quantize + 5 aux | done | `0fc1bb277` |
| — runtime/env fix | done | `198bd2c8f` |
| 3 induction | **partial** — see below | `89a496736` |
| 4 eval | done | `5e6ea9815` |
| 5 the small ones | done | `2b6247d53` |
| 6 coexistence | **partial** — see below | `78ff76077`, `acec8c2ce` |
| 7 priv-helper | **redirected** — see below | `39917c9e2` |

**Size, the §7 risk, settled by measurement:** 36.9 → 72.2 MB. It replaces
36.9 + 33.0 + 10.7 + 9.7 + 22.5 + 20.9 + … across 16 executables. The merged
binary is smaller than the set it subsumes; nothing approached 300 MB.

**End-to-end proof:** a lone `hipfire` copied into a directory that is not a git
repo, with no `HIPFIRE_DAEMON_BIN` and no sibling binaries, spawns itself as the
daemon and generates tokens (24.8 tok/s on halo/gfx1151).

### The recurring hazard was argv, not size

Two shapes, and the difference decides how much work a fold is:

- **Flag-scanners** (daemon, quantizer) search argv for `--flags` position-
  independently, so an extra leading subcommand token is invisible. Zero argv
  work.
- **Positional parsers** (the five conversion tools, atlas, hfq, steer, the
  probe, eval, coexistence) index argv absolutely or `.skip(1)`, and most
  reject the first token they do not recognise. Every one needed to be handed
  the argv it would have had as its own binary.

Guessing wrong is silent: `hipfire convert mtp-extract` failed with
`unknown arg: convert`, and `hipfire eval qwen` would have read `eval` as the
model name.

### Corrections to this plan, found by executing it

- §2's "four separate call sites" undercounted: 12 literal strings, 7
  message-producing sites, 22 call sites in 8 crates. But resolution has ONE
  owner, so closing the class needed zero call-site churn.
- §7's "six arch-crate dependencies … the main structural unknown" does not
  apply to induction. All 3,189 lines of `induction/` import exactly two
  hipfire crates. No cycle exists.
- §1 lists `hipfire-admin-ui` and `hipfire-chat-ui` as binaries. They are
  wasm32 Leptos apps in the workspace `exclude` list, built by `trunk`. Not
  native binaries; already embedded via hipfire-server's feature flags.
- The lib split is a rename, not a code move — but only where submodules say
  `use crate::*`. Where a `main.rs` names its own crate (`hipfire_quantize::`,
  ×37; coexistence ×14; steer, hfq, atlas) it needs
  `extern crate self as <name>;` or the paths rewritten.
- `#[tokio::main]` had to go. It is what would have made `hipfire eval` panic:
  that crate builds its own runtime and calls `block_on` at seven sites.

### Step 3 — what is left, precisely

Pass 1 is already in-process (M6). Pass 2 calls the quantizer, and
`hipfire_quantize::cli::main` has **68 `std::process::exit` calls**, so
in-process any failure kills the caller. A daemon-resident induction therefore
needs exactly one thing and nothing else: **make the quantizer fallible**. A
partial conversion would be worse than none. What is fixed: the orchestrator no
longer defaults to CWD-relative `target/release/...` paths, so induction works
from a deployed install.

### Step 7 — redirected, deliberately

Investigating §5's constraint surfaced a live vulnerability that predates this
plan: `install.sh` installs the helper into user-writable `~/.hipfire/bin` and
renders a polkit policy pinning `pkexec` to that path. Following the printed
instructions authorises running a user-writable file as root. `hipfire doctor`
now refuses to offer polkit elevation for any path that is not root-owned along
its whole chain, and install.sh refuses to suggest installing the policy until
the helper lives somewhere root owns. The embedding itself still wants the
security review §6 asked for — it needs a privileged emit target, an embedded
hash re-verified after write, and `O_EXCL`.


### Step 6 — corrected 2026-08-30: a forwarder, not the §4 split

Marked "done" above on the strength of `hipfire convert <group>` reaching every
coexistence group. It does. But what shipped is one
`#[command(external_subcommand)]` handing argv verbatim to a crate that still
routes on its own `args[0]`/`args[1]` — stacked, not merged. §4 of this plan
asked for something else entirely: induction becoming direct calls,
safetensors/repack becoming real `hipfire convert` subcommands, everything else
ordinary subcommands. None of those three happened.

The cost is not aesthetic, and it is what surfaced the error. An
`external_subcommand` gives clap a bare `Vec<String>`, so:

- `hipfire convert --help` listed five drafter tools and no groups;
- `gen-docs` rendered the same, leaving `man/hipfire-convert.1` with **zero**
  mentions of `download`;
- the CLI-docs freshness gate in `no-gpu-ci` could never catch it, because
  regenerating faithfully reproduces a definition that omits them.

A user reading `--help` could not discover that downloading a model was
possible. That is how the question "what happened to `hipfire download`?" gets
asked about a command that had shipped four days earlier.

**Done 2026-08-30 (`acec8c2ce` and follow-up):** `download`, `induct`, `import`,
`export` and `repack` are real subcommands. Their man pages now generate
themselves, which is the whole test of whether a promotion is real. `hub` was
retired rather than promoted — it was one spelling of two other commands. The
`repack` alias on `optimize` is gone: they are different operations (container
round-trip vs arch-optimal layout) that were one alias away from colliding.

**Still forwarded:** `artifact`, `lora`, `calibrate`, `two-pass`, `npu`. Tracked
in `docs/todo/2026-08-30-coexistence-subcommand-promotion.md`.
