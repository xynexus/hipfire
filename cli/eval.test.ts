import { test, expect } from "bun:test";
import { resolveEvalBinary } from "./eval";

test("HIPFIRE_EVAL_BIN has priority when it exists", () => {
  const bin = resolveEvalBinary({ HIPFIRE_EVAL_BIN: process.execPath }, process.platform);
  expect(bin).toBe(process.execPath);
});

test("missing eval binary returns null", () => {
  const bin = resolveEvalBinary({ HIPFIRE_EVAL_BIN: "/definitely/missing/hipfire-eval" }, process.platform);
  expect(bin).toBeNull();
});
