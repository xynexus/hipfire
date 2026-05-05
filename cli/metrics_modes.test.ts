// Bun-native tests for the metrics-mode expansion helpers.
//
// Run: bun test cli/metrics_modes.test.ts
//
// We don't import from index.ts to keep the test runnable without spinning up
// the full CLI module graph (it has top-level side effects). Keep in sync with
// cli/index.ts:normalizeMetricsKMode / normalizeMetricsVMode / kvPairToRuntimeMode.

import { test, expect } from "bun:test";

function normalizeMetricsKMode(mode: string): string {
  const m = mode.toLowerCase();
  if (m === "tqv1" || m === "tq1") return "asym4_tqv1";
  if (m === "f32") return "fp32";
  if (m === "turbo2") return "asym2";
  if (m === "turbo3" || m === "turbo") return "asym3";
  if (m === "turbo4") return "asym4";
  return m;
}

function normalizeMetricsVMode(mode: string): string {
  const m = mode.toLowerCase();
  if (m === "f32") return "fp32";
  if (m === "asym2") return "tqv2";
  if (m === "asym3") return "tqv3";
  if (m === "asym4") return "tqv4";
  return m;
}

function kvPairToRuntimeMode(kRaw: string, vRaw: string): string | null {
  const k = normalizeMetricsKMode(kRaw);
  const v = normalizeMetricsVMode(vRaw);
  if (k === "fp32" && v === "fp32") return "fp32";
  if (k === "q8" && v === "q8") return "q8";
  if ((k === "asym2" || k === "asym3" || k === "asym4") && v === "q8") return k;
  if (k === "asym4" && (v === "tqv1" || v === "tqv2" || v === "tqv3" || v === "tqv4")) return `asym4_${v}`;
  return null;
}

function expandMetricsModes(kModes: string[], vModes: string[]): { modes: string[]; skipped: string[] } {
  const modes: string[] = [];
  const skipped: string[] = [];
  for (const k of kModes) {
    for (const v of vModes) {
      const runtime = kvPairToRuntimeMode(k, v);
      if (!runtime) {
        skipped.push(`${k}/${v}`);
        continue;
      }
      if (!modes.includes(runtime)) modes.push(runtime);
    }
  }
  const fp32Idx = modes.indexOf("fp32");
  if (fp32Idx >= 0) modes.unshift(...modes.splice(fp32Idx, 1));
  else modes.unshift("fp32");
  return { modes, skipped };
}

test("metrics mode aliases normalize to runtime KV modes", () => {
  expect(normalizeMetricsKMode("f32")).toBe("fp32");
  expect(normalizeMetricsKMode("turbo")).toBe("asym3");
  expect(normalizeMetricsKMode("turbo2")).toBe("asym2");
  expect(normalizeMetricsKMode("turbo4")).toBe("asym4");
  expect(normalizeMetricsKMode("tq1")).toBe("asym4_tqv1");
  expect(normalizeMetricsKMode("tqv1")).toBe("asym4_tqv1");
  expect(normalizeMetricsVMode("asym2")).toBe("tqv2");
  expect(normalizeMetricsVMode("asym3")).toBe("tqv3");
  expect(normalizeMetricsVMode("asym4")).toBe("tqv4");
});

test("metrics K/V matrix expands only supported runtime pairs", () => {
  const expanded = expandMetricsModes(
    ["fp32", "q8", "asym4", "asym3", "tq1"],
    ["fp32", "q8", "tqv4", "tqv3", "tqv2", "tqv1"],
  );
  expect(expanded.modes).toEqual([
    "fp32",
    "q8",
    "asym4",
    "asym4_tqv4",
    "asym4_tqv3",
    "asym4_tqv2",
    "asym4_tqv1",
    "asym3",
  ]);
  expect(expanded.skipped).toContain("q8/fp32");
  expect(expanded.skipped).toContain("asym3/tqv4");
  expect(expanded.skipped).toContain("tq1/tqv1");
});
