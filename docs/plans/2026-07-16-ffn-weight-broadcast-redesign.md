# Plan: FFN-dataflow redesign for weight-broadcast (batched NPU encode)

Date: 2026-07-16. Scopes the redesign needed to land weight-broadcast on the
resident dense-W8 FFN, after the incremental r26 attempts hit a hard channel wall.
Prereq context: `docs/todo/2026-07-15-batched-npu-encode.md` (obstacle tree) and
memory `project_npu_encode_dispatch_floor`.

## Objective + proven mechanism + payoff

Make batching amortize. Today the FFN is un-pinned to arbitrary batch (M512=2×M256
bit-exact, committed `8b8ffb1a8`) but naive batching is only ~1.07× because
`upload_weights` **replicates the weight tile per row-macro**, so weight-DMA
(~960 MB/s, the FFN's bottleneck) scales with rows.

The fix — **weight-broadcast**: load ONE macro's weights into the memtile once,
replay across all M_MACROS row-tiles on-chip via `iter_count` (IRON
`set_iter_count(N)` → memtile MM2S `repeat_count=N-1`; replays the resident BD
chain, sequence order, NO new L3 transfer). **Mechanism PROVEN** (probe
`scratchpad/iter_probe.py`, lowered `input_with_addresses.mlir`: S2MM loads once,
MM2S replays). Payoff: weight-DMA 11 MB → 3.67 MB/layer (~56% traffic cut), plus
— if T goes on-chip — eliminating the ~1.5 MB/layer T DRAM round-trip.

## The channel wall (why incremental r26 patching failed)

r26 tiling: 8 cols × 4 core-rows. `%mt{col}` (col-indexed) hosts WEIGHTS
(@wsh→@wbc, per column) AND — for col 0–3 == row 0–3 — ACTIVATIONS (@xsh→@xbc, per
row) AND core outputs (@oc). aie2p limits: **shim = 2 DMA ch/dir, memtile = 6
DMA ch/dir.**

| memtile | inputs (S2MM) | outputs (MM2S) |
|---|---|---|
| mt0–3 (weights+activations+out) | @wsh + @xsh + 4×@oc = **6 (AT LIMIT)** | @wbc + @xbc + @osh = 3 |
| mt4–7 (weights+out) | @wsh + 4×@oc = 5 | @wbc + @osh = 2 |

**Key fact:** the `iter_count` replay needs an extra memtile **S2MM (input)**
channel (the buffer-loopback the lowered probe shows). mt0–3 are already at 6
inputs → any broadcast form overflows there. Confirmed empirically:
- two shim→memtile loads (@wshg+@wshd) → **shim output limit (2)** exceeded;
- one shim load + memtile split-link → **memtile input limit (6)** exceeded.

## Why simple re-tiling alone does NOT work

8 weight-broadcasts (one per column) each need a spare memtile channel, but 4 of
the 8 memtiles must also host the 4 activation streams. Moving activations to
mt4–7 just shifts the saturation there (8 broadcasts > 4 free memtiles). There is
no channel-neutral re-assignment: the total channel demand (8 weight replays + 4
activation streams + outputs) exceeds what 8 memtiles at these limits provide,
**as long as the 4×@oc core-output fan-in stays.**

## Two strategic paths

### Path A — retrofit r26 (free a channel, then one-fifo broadcast + interleave)

Keeps the hand-tuned kernel. Three coupled changes:

1. **Free a memtile input channel** — reduce the **4×@oc core-output fan-in** to
   ≤3 (or 2). The 4 cores/column each DMA their output tile to the memtile (4
   S2MM). Combine via the existing core-to-core cascade (`aie.flow … "Core":0`,
   already used for activation sharing) so fewer cores write to the memtile.
   *Frees 1–2 S2MM on every memtile — the enabling move.* (Core-output restructure;
   the riskiest piece.)
2. **One weight fifo + iter_count** — @wsh loads 28 blocks single-copy; @wbc
   (`iter_count=M_MACROS`) replays the [gate18,down10] sequence. Uses the freed
   S2MM. No split (avoids the 2-fifo channel doubling).
3. **Core interleave `[gate,down]×M_MACROS`** — merge the gate `for outblock` and
   down `for mblock` loops into one `for macro` loop (gate→down per macro) so the
   consumption order matches the single-fifo replay. The per-macro intermediate T
   (hidden, 96×1152) must be available to down-m right after gate-m: keep it
   **on-chip** (memtile-resident, ~221 KB < 512 KB) — which also removes the T
   DRAM round-trip — or pipeline the DRAM round-trip per macro.
4. Host: single-copy `%W` in `upload_weights` (28 blocks/col, drop replication).

Effort: high (hand-written 700-line MLIR generator + core kernel + host).
Risk: high (deadlocks, channel budget, T restructure). Multi-day, iterative HW.
Upside: keeps r26's proven performance; smallest conceptual change to what ships.

### Path B — greenfield IRON FFN with broadcast designed in

Re-author the resident FFN in the IRON high-level API (like the STEEL/pilot work).
IRON does **auto tile/channel placement** and supports `iter_count` natively (the
probe used it), so the channel wall is the compiler's problem, not ours. Weight
reuse is expressed once (`.forward().set_iter_count(M_MACROS)`), not fought.

Effort: high (full rewrite of a hand-optimized kernel).
Risk: medium-per-step (IRON handles channels/locks; less deadlock risk) but must
match r26's numerics AND performance, then replace the resident path — a real
bake-off. Two FFN impls until it wins.
Upside: aligns with the strategic IRON direction (STEEL long-context lives there);
pilot already validated IRON FFN primitives (gemm/gelu/elementwise); broadcast is
free rather than a channel fight.

## Recommendation

**Spike Path A step 1 first** (the @oc fan-in reduction) as a cheap go/no-go: if
combining core outputs frees a memtile S2MM without wrecking the output schedule,
Path A is viable and cheapest-to-ship. If step 1 proves as gnarly as the earlier
obstacles (likely — it's hand-tuned cascade logic), **switch to Path B** — the
IRON rewrite is more work up front but far lower per-step risk and is where NPU
FFN/attention is heading anyway. Do NOT resume two-fifo/split-link patching on
r26's current tiling; that ceiling is proven.

## Validation gates (both paths)

- `npu_resident_ffn_w8_canonical_verify` — absolute reference (cosine > 0.999).
- `npu_resident_ffn_w8_batched_verify` — M512 `[X;X]` → doc0==doc1==M256(X)
  bit-exact (self-consistency is NOT enough; gate on the absolute reference too).
- M256 vs M512 timing — the payoff metric; expect the ~1.07× to flatten toward
  ~2× rows-per-ms once weight-DMA stops scaling with rows.

## Risks

- Deadlocks from objectfifo acquire/release imbalance (Path A) — bounded by build
  timeouts; slow to debug.
- On-chip T may itself need channels (gate-out→memtile→down-in) — re-check the
  budget after the @oc reduction.
- Path B perf parity with the hand-tuned r26 is unproven; needs a bake-off before
  replacing the resident path.

## Effort

Either path is a multi-day, iterative-hardware effort — not a single edit. This is
a dedicated kernel-redesign task, correctly deferred out of the batch un-pin
milestone (which is committed and bit-exact).
