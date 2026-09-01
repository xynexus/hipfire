# Decisions pending — todo items blocked on an owner call

Raised 2026-08-31 while working `docs/todo` unattended. Each item below is
blocked on a judgement that is not mine to make: a product semantic, a
compatibility break, or a scope/sequencing choice. Everything else in
`docs/todo` was either startable or closed.

The distinction used: an "open question" answerable by running an experiment is
NOT listed here — those are work, not decisions.

---

## 1. ~~SpinQuant R1/R2 — which path to a servable artifact~~ — WITHDRAWN 2026-09-01

**This was my error, not a real decision.** I grepped
`2026-07-01-spinquant-r1r2-future-work.md` for "Two clean paths (pick one)",
found it, and registered a fork. The surrounding line reads
`<details><summary>Original two-path plan (path (a) was chosen)</summary>` — the
choice was made long ago, and the section is collapsed history.

Explored 2026-09-01 and confirmed against the tree. **Path (a) is built and
serving:**

- `hipfire-train/examples/learn_r1_dump.rs` learns R1 and writes `HFR1` + `h:u32`
  + `h*h` f32, validating fp-invariance before it writes (max|Δlogit| 5.6e-5).
- `hipfire-quantize/src/rotate.rs` + `--rotate <M.r1>` consumes it — same magic,
  `h` checked against `hidden_size` — folds each RMSNorm into its readers and
  rotates residual readers/writers/embedding, installing the result as a global
  override so **every** codec branch quantizes rotated weights with no per-branch
  edit. Unit tests `r1_plan_is_orthonormal` and `bake_cancels_codec_fwht_leaving_m`
  (the bake proof) both live in `rotate.rs`.
- The doc records serving as DONE end-to-end, and a real Supra-50M -> 73 MB oq4
  artifact with R1 orthonormality 1.1e-6.

Path (b), the safetensors round-trip, was never needed and is not a live option.

**The genuine open item is R2 DEPLOYMENT, and it is a constraint rather than a
choice.** R2 *learning* is already DONE and measured (§1 of that doc: per-head
ctx-learned R2 beats the fixed rotation, +1.35). What blocks deploying it is
that the codec's 256-wide FWHT crosses head boundaries, so a per-head
`apply_r2` cannot be un-baked — the net transform would be
`F₂₅₆ ∘ blockdiag(R2)`, not `R2`. That needs a per-head codec variant, which is
separate work with no decision attached.

Worth knowing before anyone funds it: the same doc measured that on gfx1151,
**W4A4 ≈ W4A8 (~1.04–1.06x)** with equally-tuned kernels, and concludes W4A8
with no rotation is about as fast for far less risk. The SpinQuant throughput
payoff is marginal *on this APU*; revisit on CDNA/gfx12, which are core-bound.

## 2. n-gram topic tags — the write path

`2026-08-29-ngram-topic-tags.md`

With N attached tags, "write to topic" is ambiguous:

- write to **every** attached tag table — N-fold write amplification, and a gram
  learned in a `rust+gpu` session pollutes both single-topic tables;
- write to the **first** tag only;
- keep writing to the **user** table and build tag tables offline from curated
  corpora.

This decides what the stored data MEANS, so it is not a default I should pick.
The doc leans toward the third (keeps the write path single, keeps tag tables
clean, fits the "generate training data" motivation).

**Owner's position, 2026-09-01:** no good method yet. The one shape that seems
workable is a **treesitter or classification model detecting code/topic**, and
routing n-gram updates to the respective stores on that signal — i.e. neither
"write to all N tags" nor "first tag only", but *classify, then direct*.

That reframes the question rather than answering it, and it is worth saying how:

- It makes the write path **single per gram** again (one classified destination),
  which is what made option three attractive — without giving up online learning
  the way "build tag tables offline from curated corpora" does.
- It moves the cost from write amplification to **classification latency on the
  write path**, and adds a dependency the n-gram store does not currently have.
  Treesitter is cheap and deterministic for the code/not-code split; a
  classification model is neither, and would want to be async or batched.
- It introduces a **new failure mode**: a misclassified gram lands in the wrong
  table and is indistinguishable from a correct one afterwards. Option three's
  offline curation has no equivalent, because the corpus is chosen deliberately.
- Treesitter alone may cover the case that motivated topics. If the split that
  matters is code vs prose, a parse either succeeds or it does not, and that is a
  far stronger signal than a topic classifier — and cheap enough for the write
  path.

**Still open**, and now written into `2026-08-29-ngram-topic-tags.md` §4 as a
fourth option beside the original three, with what it changes and the narrowing
that looks right: treesitter, code vs prose, one destination per gram — a parse
either succeeds or it does not, which needs no model on the write path.

Secondary, same doc: probe order across tags (§3). Tags have no natural order;
the doc leans request order. Whatever is chosen must be stable within a session
or the per-tag marginal numbers are meaningless.

## 3. Coexistence flag spelling — a deliberate compatibility break

`2026-08-30-coexistence-subcommand-promotion.md`

`import gguf` spells its paths `--in`/`--out`; every other promoted command uses
`--input`/`--output`. Options: unify (breaks scripts that use the old spelling),
add a clap `alias` (accepts both, at the cost of two documented names for one
flag), or leave the split. The doc says explicitly this "wants its own
decision".

Note the promotion already made one behaviour change on purpose: `ArgBag`
silently ignored unknown flags, so `two-pass --typo 1` used to be accepted and
dropped; under clap it errors. Any script passing a flag the tool never read
will now fail loudly.

## 4. STEEL long-context NPU — productionize an IRON bridge?

`2026-07-15-steel-large-context-npu.md`

Step 5 is "decide whether a hipfire<->IRON bridge (author in IRON, compile
offline, dispatch via amdxdna) is worth productionizing for the long-context
path". That is a strategic commitment to a second authoring toolchain, not a
technical unknown.

## 5. n-gram cold store — merge on unload?

`2026-08-29-ngram-cold-store-merge-cadence.md`

Should unload trigger a merge? It would flush the backlog and leave the file
tidy for the next load; today `merge_backlog` is simply dropped with the
`NgramSpec`. Cheap either way, but it is a policy call about when work happens.

The crash-safety item raised beside it is NOT a decision — `merge` zeroes the
data region before rewriting, so a crash mid-merge leaves a partly zeroed body;
tmp-file + rename makes it atomic and is the pattern `hipfire-vision-cache`
already uses for its manifest. That one is just work.

---

## 6. Quant benchmark queue — restart the 35B pipeline, or drop it?

`2026-08-08-quant-benchmark-queue-handoff.md`

Re-checked 2026-08-31: the two "in flight" jobs are dead, their scratchpad and
scripts are gone, and every expensive Qwen3.5-35B-A3B output (bf16 43 GB, calib
1.8 GB, `oq4.25++` 17.9 GB) is absent from `/srv`. The source `.hfa` (47.7 GB)
survives, so it is restartable from §"Recipe" — but it is a restart, not a
resume, and the doc's "three stages banked" is no longer true.

That is hours of GPU on a box whose daemon is serving. Whether the benchmark
queue is still worth it is a priority call. Not restarted unattended.

The one durable result needs no rerun and is recorded in place: the per-expert
LDLQ factorization is irreducible for calibrated recipes (AWQ rebases the
Hessian per tensor, damping is applied after the rebase).

---

## 7. MoE routed experts — one allocation per expert, and who owns it?

`2026-09-01-moe-expert-pair-allocation.md`

gfx1151 rounds every GTT allocation, and the qwen35 loader allocates each
expert's two projections separately, so some models pay the rounding twice.
Computed from the real artifacts through `gtt_alloc_cost`:

| model | separate | one/expert | saving |
|---|---|---|---|
| Qwen3.6-35B-A3B oq4.25++ | 30.0 GiB | 20.0 GiB | **10.0 GiB (33.3%)** |
| Qwen3.5-122B-A10B oq4.25++ | 75.8 GiB | 74.5 GiB | 1.3 GiB (1.7%) |

Model-shaped, not universal — it depends where per-expert size sits on the 2 MiB
grid, and it is **not** a 122B unlock (that model's amplification was the
compact->Oq8 expansion, already closed by compact-resident being default).

**The decision is ownership, not effort.** Packing after load is the low-risk
shape — loaders untouched, correctness is a device-to-device copy — but
`GpuTensor::sub_offset` yields a `NonOwning` buffer, and `dispose()` carries

```rust
BufferOrigin::NonOwning => {
    debug_assert!(false, "dispose() on a non-owning buffer — aliasing bug");
}
```

That assert exists because of #262, where an alias went back to the pool and came
back as scratch over live weights. So packing means `ExpertWeights` must learn
that `down` is a view and must not be freed — a lifetime change on the load path
of every MoE model, and one that deliberately routes around a guard added after a
memory-corruption bug.

The fork:

- **one buffer with two views** — smallest diff, but needs the ownership change
  above and an explicit exemption from the alias invariant;
- **a per-layer arena** for all of a layer's experts — amortises the rounding
  further (512 experts in one allocation rather than 512 allocations), but changes
  how `expert_gate_up_ptrs` / `expert_down_ptrs` are built, from independent
  addresses to offsets into one base, so it wants measuring against the indexed
  MoE kernels first.

Not startable unattended: either branch is a deliberate break of an invariant
that was added in response to memory corruption, which is an owner call.

Ruled out and not worth re-testing: routing expert loads through the GPU pool.
`pool.rs::alloc` recycles whole buffers rather than sub-allocating from slabs, so
the per-tensor rounding applies either way. The allocation SHAPE is the only lever.

---

## 8. The `*-DFlash--bf16.hfq` drafters on `/srv` are unusable — re-cut, or delete?

Found 2026-09-01 while measuring speculative decode.

**Every DFlash artifact on `/srv/hipfire/models` is `arch 1` (qwen2), not
`arch 20`**, so `DflashConfig::from_hfq` rejects all seven and no DFlash path can
load from them. They do carry drafter provenance
(`config.dflash_config` with `mask_token_id` + `target_layer_ids`), just under an
older key than the loader's top-level `dflash`, and with the wrong arch tag:

| artifact | size | matching HF source on `/srv`? |
|---|---|---|
| `Qwen3.6-27B-DFlash--bf16.hfq` | 2.29 GB | **yes** — already re-cut (see below) |
| `Qwen3.5-122B-A10B-DFlash--bf16.hfq` | 0.68 GB | no |
| `Qwen3.5-397B-A17B-DFlash--bf16.hfq` | 1.64 GB | no |
| `Qwen3.5-9B-DFlash--bf16.hfq` | 1.39 GB | no |
| `Qwen3.6-35B-A3B-DFlash--bf16.hfq` | 0.50 GB | no (the source is Qwen3.**5**-35B-A3B) |
| `gemma-4-26B-A4B-it-DFlash--bf16.hfq` | 0.57 GB | no |
| `gemma-4-31B-it-DFlash--bf16.hfq` | 2.04 GB | no |

**Only one of the seven can be regenerated locally.** That one is done —
`dflash_convert --input /srv/huggingface/models--z-lab--Qwen3.6-27B-DFlash/snapshots/*/`
produced `~/.hipfire/models/Qwen3.6-27B--dflash.bf16.hfq` (3.46 GB, arch 20), and
it is what the DDTree-vs-chain numbers in `docs/perf/ddtree-vs-chain-opus.md` were
measured with.

The decision, and why it is not mine:

- these live on **`/srv`**, the shared NFS mount, which the overnight brief treats
  as read-only precisely because its contents are often the only copy;
- **six of the seven have no local source**, so "delete the unusable ones" may be
  destructive in a way no re-cut can undo. Whether a copy exists elsewhere — an
  upstream HF repo still published, another machine — is not something I can
  determine from here.

Options: re-cut the one that can be re-cut and leave the rest (status quo, but the
directory keeps advertising six drafters that cannot load); fetch the missing
sources and re-cut all seven; or delete the six as dead weight. The middle option
depends on whether those upstream repos still exist.

Worth fixing either way, and it is plain work rather than a decision: the loader
could **name the mismatch** instead of failing with `parse DflashConfig`. It has
enough information to say "this artifact is arch 1 with a `config.dflash_config`
block — it looks like a pre-`arch 20` drafter; re-cut it with `dflash_convert`."

---

## Go / no-go, rather than a design answer

Five documents are unapproved by their own status line, so starting them is a
scope decision rather than a technical one:

| doc | status |
|---|---|
| `bug-reporting.md` | "idea, not yet approved for build" |
| `npu-gpu-heterogeneous-prefill.md` | "research idea, not yet approved for build"; also depends on prereqs that do not exist |
| `kvarn-hot-bitwidth.md` | "idea. Owner: KV" |
| `merge-sidecar.md` | "idea. Owner: KV" |
| `2026-07-22-embedding-quant-improvements.md` | "open / research" |
