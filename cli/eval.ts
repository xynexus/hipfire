// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

import { existsSync } from "fs";
import { join, resolve } from "path";
import { homedir } from "os";
import { spawn } from "bun";

export function resolveEvalBinary(env = process.env, platform = process.platform): string | null {
  const exe = platform === "win32" ? ".exe" : "";
  const candidates = [
    env.HIPFIRE_EVAL_BIN,
    resolve(__dirname, `../target/release/hipfire-eval${exe}`),
    join(homedir(), ".hipfire", "bin", `hipfire-eval${exe}`),
  ].filter((p): p is string => !!p);
  return candidates.find(p => existsSync(p)) ?? null;
}

export async function runEvalCommand(args: string[]): Promise<void> {
  const bin = resolveEvalBinary();
  if (!bin) {
    console.error("hipfire-eval not found.");
    console.error("  Build: cargo build --release -p hipfire-runtime --bin hipfire-eval");
    console.error("  Or set HIPFIRE_EVAL_BIN=/path/to/hipfire-eval");
    process.exit(1);
  }
  const proc = spawn([bin, ...args], {
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
    env: { ...process.env },
  });
  const code = await proc.exited;
  process.exit(code ?? 1);
}
