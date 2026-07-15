# Gemma 4 Phase 3 layered attention and KV state

Date: 2026-07-15. Status: passed.

## Exit-gate evidence

- `hipfire-runtime::layered_kv` now describes every logical layer with an
  explicit physical group and slot, full or sliding-window storage, and an
  optional shared producer. Compatible owned layers are grouped around the
  existing `KvCache` implementation; shared logical layers own no cache.
- The mixed-geometry test plan combines 256- and 512-wide heads, distinct
  local/global KV-head counts, and shared layers. Its exact byte accounting is
  the sum of the two owned K/V allocations rather than the four logical layers
  or the maximum geometry.
- `LayeredAttentionScratch` allocates one maximum Q/K/V/attention workspace and
  returns checked, non-owning per-layer subviews.
- `KvSequenceCursor` and the arena expose explicit reset, growth, physical-ring,
  and visible-logical-range behavior.
- Gemma 3's homogeneous unquantized path now enters through
  `LayeredKvArena::homogeneous_fp32_cache`, which validates the generalized plan
  and returns the same legacy `KvCache` allocation used by the family before
  this phase.

CPU gates:

```text
$ cargo test -p hipfire-runtime --lib
test result: ok. 167 passed; 0 failed; 4 ignored

$ cargo test -p hipfire-arch-gemma3
test result: ok. 7 passed; 0 failed
```

The six new layered-KV unit tests cover mixed geometry and exact owned bytes,
shared-producer resolution and zero storage, local/global boundary mapping,
cursor growth/reset/second request, the homogeneous adapter, and invalid
sharing/geometry rejection. The pre-existing `kv::index_math_tests` all passed
unchanged in the full runtime suite.

The locked GPU parity example passed on gfx1103:

```text
$ cargo run --release -p hipfire-runtime --example layered_kv_parity
GPU: gfx1103
layered_kv_parity: PASS (owned=2 logical=4 bytes=98304)
```

It writes and reads the local ring at window-1, window, and window+1, checks a
global layer at the same positions, compares the downloaded device values with
the CPU storage model, verifies that a shared layer aliases its producer's
device buffers, resets the arena, and verifies a clean position-zero overwrite
for a second request.

The existing Gemma 3 consumer also passed all six committed tiny-quant rows
after the adapter change on gfx1103: collect, Q8F16, HFQ4, OQ4, OQ8, and
calibrated OQ4+. The gate reported zero findings and zero admission findings.

## Reuse and cleanup ledger

- Existing primitives reused: `KvCache`, its full and capped constructors,
  `DeviceBuffer`, `DeviceBufferView`, and the existing Gemma 3 forward/cache
  lifecycle.
- Duplicate removed or retained: no KV codec or storage implementation was
  copied. Existing homogeneous constructors remain available as compatibility
  adapters; quantized and hierarchical cache paths remain intentionally
  unchanged.
- Generic seam added or changed: `LayeredKvPlan`, `LayeredKvArena`,
  `KvSequenceCursor`, and `LayeredAttentionScratch` provide only geometry,
  ownership, mapping, and checked-view policy around the physical caches.
- Generic abstraction consumers: the Phase 3 parity example uses mixed groups;
  Gemma 3 uses the homogeneous adapter in its real forward path.
- Stale assumption removed: a model no longer has to imply one KV geometry and
  storage policy for every logical layer, nor allocate physical storage for a
  logical layer that shares a producer.
- Oracle retained: the old `KvCache` constructors and all old KV unit tests stay
  intact, while Gemma 3's committed tiny-quant results continue to provide the
  homogeneous end-to-end regression anchor.
