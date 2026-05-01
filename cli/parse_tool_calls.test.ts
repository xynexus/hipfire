// Bun-native test for the defensive tool-call parser (#111).
//
// Focuses on shapes captured from MQ4 quantization drift: the "flat" form
// (sibling args instead of nested `arguments`) and the "XML-tag" form
// (`<plain>NAME</param>`). Strict OpenAI-spec input is the control case.
//
// Run: bun test cli/parse_tool_calls.test.ts
//
// We don't import from index.ts to keep the test runnable without spinning up
// the full CLI module graph (it has top-level side effects). Instead the
// parser is duplicated here. Keep in sync with cli/index.ts:parseToolCalls.

import { test, expect } from "bun:test";

function parseOneToolCall(raw: string): { name: string; arguments: any; repaired: boolean } | null {
  try {
    const tc = JSON.parse(raw);
    if (tc && typeof tc === "object" && typeof tc.name === "string") {
      if (tc.arguments !== undefined) {
        return { name: tc.name, arguments: tc.arguments, repaired: false };
      }
      const drop = new Set(["name", "type", "id", "function"]);
      const args: Record<string, any> = {};
      let coerced = false;
      for (const [k, v] of Object.entries(tc)) {
        if (drop.has(k)) continue;
        args[k] = v;
        coerced = true;
      }
      if (coerced) return { name: tc.name, arguments: args, repaired: true };
      return { name: tc.name, arguments: {}, repaired: false };
    }
  } catch {}

  const xmlPatterns = [
    /^<\s*plain\s*>\s*([A-Za-z_][\w.]*)\s*<\s*\/\s*param\s*>/,
    /^<\s*function\s*=\s*([A-Za-z_][\w.]*)\s*>/,
    /^<\s*tool\s*name\s*=\s*"?([A-Za-z_][\w.]*)"?\s*>/,
  ];
  for (const pat of xmlPatterns) {
    const nm = raw.match(pat);
    if (!nm) continue;
    const after = raw.slice(nm[0].length).trim();
    const args = extractFirstJsonObject(after);
    if (args !== null) return { name: nm[1], arguments: args, repaired: true };
    return { name: nm[1], arguments: {}, repaired: true };
  }
  return null;
}

function extractFirstJsonObject(s: string): any | null {
  const start = s.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let escape = false;
  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (escape) { escape = false; continue; }
      if (ch === "\\") { escape = true; continue; }
      if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') { inStr = true; continue; }
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        try { return JSON.parse(s.slice(start, i + 1)); }
        catch { return null; }
      }
    }
  }
  return null;
}

test("strict OpenAI form parses without repair flag", () => {
  const r = parseOneToolCall('{"name": "write", "arguments": {"path": "/tmp/x", "content": "y"}}');
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({ path: "/tmp/x", content: "y" });
  expect(r!.repaired).toBe(false);
});

test("zero-arg tool call passes through with empty args", () => {
  const r = parseOneToolCall('{"name": "list_files"}');
  expect(r).not.toBeNull();
  expect(r!.name).toBe("list_files");
  expect(r!.arguments).toEqual({});
  expect(r!.repaired).toBe(false);
});

test("flat form (sibling args, no `arguments` wrapper) is repaired", () => {
  // Captured from qwen3.6:27b MQ4 multi-tool stream on 2026-05-01 (#111).
  const raw = '{"name": "write", "path": "/tmp/rate_limiter.py", "content": "print(1)"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({ path: "/tmp/rate_limiter.py", content: "print(1)" });
  expect(r!.repaired).toBe(true);
});

test("flat form with extra metadata keys (id, type) drops them", () => {
  const raw = '{"id": "abc", "type": "function", "name": "bash", "command": "ls -la"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("bash");
  expect(r!.arguments).toEqual({ command: "ls -la" });
  expect(r!.repaired).toBe(true);
});

test("XML-corruption form <plain>NAME</param> {ARGS} is repaired", () => {
  // Captured from reporter Fluorax (#111 issue body).
  const raw = '<plain>write</param> {"path": "/home/mike/rate_limiter.py", "content": "y"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({ path: "/home/mike/rate_limiter.py", content: "y" });
  expect(r!.repaired).toBe(true);
});

test("XML <function=NAME> variant is repaired", () => {
  const raw = '<function=read>{"path": "/etc/passwd"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("read");
  expect(r!.arguments).toEqual({ path: "/etc/passwd" });
  expect(r!.repaired).toBe(true);
});

test("XML form with unparseable JSON tail emits empty args, preserves name", () => {
  const raw = '<plain>write</param> {"path": "/tmp/x", "content": "broken';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({});
  expect(r!.repaired).toBe(true);
});

test("totally unparseable garbage returns null", () => {
  const r = parseOneToolCall("totally not a tool call");
  expect(r).toBeNull();
});

test("nested arguments object survives extraction with strings containing braces", () => {
  // Common content shape: code with `{` inside; balanced-brace walker must
  // not be tricked by braces inside JSON strings.
  const raw = '<plain>write</param> {"path": "/tmp/x.py", "content": "def f(): return {1: 2}"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({ path: "/tmp/x.py", content: "def f(): return {1: 2}" });
});

test("escaped quotes inside JSON strings are handled", () => {
  const raw = '<plain>write</param> {"path": "/tmp/x", "content": "say \\"hi\\""}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.arguments).toEqual({ path: "/tmp/x", content: 'say "hi"' });
});
