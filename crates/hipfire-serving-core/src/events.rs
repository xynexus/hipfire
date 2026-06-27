// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! JSONL stream-event emitters for the daemon IPC protocol.
//!
//! Every event the daemon writes to stdout is built through `serde_json` so a
//! user-controlled `id` or error string can't desync the line protocol by
//! injecting an embedded `"`, `\`, or newline. Extracted verbatim from the
//! former `main.rs` monolith (no behavior change).

use std::io::Write;

use hipfire_generate::eos_filter::FilterAction;

/// Cap on the *encoded* base64 string length the daemon will accept on the
/// IPC. ~40 MB encoded → ~30 MB raw image bytes (4/3 expansion).
pub const MAX_BASE64_ENCODED_LEN: usize = 40 * 1024 * 1024;

/// Emit an id-tagged `{"type":"error","id","message"}` line and flush. Use this
/// (not raw `writeln!`) so a user-controlled `id`/message can't desync the JSONL
/// protocol via an embedded `"`, `\`, or newline.
pub fn emit_error_with_id(stdout: &mut std::io::Stdout, id: &str, message: impl std::fmt::Display) {
    let envelope = serde_json::json!({
        "type": "error",
        "id": id,
        "message": format!("{}", message),
    });
    let _ = writeln!(stdout, "{}", envelope);
    let _ = stdout.flush();
}

/// Id-less variant of [`emit_error_with_id`] — emits `{"type":"error","message"}`.
/// Currently has no caller (pre-existing dead code, relocated as-is); kept for
/// protocol completeness.
#[allow(dead_code)]
pub fn emit_error_no_id(stdout: &mut std::io::Stdout, message: impl std::fmt::Display) {
    let envelope = serde_json::json!({
        "type": "error",
        "message": format!("{}", message),
    });
    let _ = writeln!(stdout, "{}", envelope);
    let _ = stdout.flush();
}

/// Emit a parsed `deepseek4::dsml::StreamEvent` to the JSONL stream.
/// Maps:
///   - Token(text)        → `{type:"token",   id, text}`
///   - Reasoning(text)    → `{type:"reasoning", id, text}`
///   - ToolCalls(calls)   → `{type:"tool_calls", id, calls:[{name, arguments}]}`
///
/// The CLI / OpenAI HTTP layer translates these into the corresponding
/// SSE chunks (`content`, `reasoning_content`, `tool_calls.delta`).
pub fn emit_stream_event(
    stdout: &mut std::io::Stdout,
    id: &str,
    ev: hipfire_arch_deepseek4::dsml::StreamEvent,
) {
    use hipfire_arch_deepseek4::dsml::StreamEvent;
    // The request id is user-supplied. Build the envelope through
    // `serde_json` so any embedded `"` / `\` / control chars are
    // escaped — otherwise a malformed id corrupts every subsequent
    // line of the JSONL stream and the cli/serve loop dies with a
    // `JSON Parse error: Expected '}'`.
    let envelope = match ev {
        StreamEvent::Token(text) => serde_json::json!({
            "type": "token",
            "id": id,
            "text": text,
        }),
        StreamEvent::Reasoning(text) => serde_json::json!({
            "type": "reasoning",
            "id": id,
            "text": text,
        }),
        StreamEvent::ToolCalls(calls) => {
            let arr: Vec<serde_json::Value> = calls
                .into_iter()
                .map(|c| {
                    serde_json::json!({
                        "name": c.name,
                        "arguments": c.arguments,
                    })
                })
                .collect();
            serde_json::json!({
                "type": "tool_calls",
                "id": id,
                "calls": serde_json::Value::Array(arr),
            })
        }
    };
    let _ = writeln!(stdout, "{}", envelope);
}

/// Emit a probe-only `{"type":"committed",id,tok_id,pos,t_ms}` event for every
/// committed token — a parallel raw-token-id stream alongside the text `token`
/// events. Gated on `HIPFIRE_EMIT_TOKEN_IDS=1` (read once); a no-op otherwise, so
/// existing JSONL clients see no change. The probe binary sets the env on the
/// daemon child it spawns.
pub fn emit_committed_event(
    stdout: &mut std::io::Stdout,
    id: &str,
    tok_id: u32,
    pos: usize,
    t_ms: u64,
) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let on = *ENABLED
        .get_or_init(|| std::env::var("HIPFIRE_EMIT_TOKEN_IDS").ok().as_deref() == Some("1"));
    if !on {
        return;
    }
    // Build through `serde_json::json!` for the same reason
    // `emit_error_with_id` does: `id` is user-supplied and a single `"`
    // or `\` in it would corrupt the line, breaking the client's JSONL
    // parser for every subsequent event on the same connection.
    let envelope = serde_json::json!({
        "type": "committed",
        "id": id,
        "tok_id": tok_id,
        "pos": pos,
        "t_ms": t_ms,
    });
    let _ = writeln!(stdout, "{}", envelope);
}

/// Emit a single-line `{"type":"error","id":"...","message":"..."}` JSON
/// line on the IPC stream. Uses `serde_json` so user-controlled error
/// strings (image decoder messages, base64 errors) can't desync the
/// protocol by injecting embedded `"`, `\`, or newline bytes.
pub fn write_error(stdout: &mut std::io::Stdout, id: &str, message: &str) {
    let line = serde_json::json!({
        "type": "error",
        "id": id,
        "message": message,
    });
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

/// Act on an [`EosFilter`](hipfire_generate::eos_filter) decision: stream any
/// emitted/held bytes as a `token` event and return `true` when generation
/// should stop (`Stop`/`StopEmit`), `false` to continue (`Emit`/`Hold`).
pub fn emit_filter_action(stdout: &mut std::io::Stdout, id: &str, action: FilterAction) -> bool {
    match action {
        FilterAction::Emit(text_bytes) => {
            emit_text_bytes(stdout, id, &text_bytes);
            false
        }
        FilterAction::StopEmit(text_bytes) => {
            emit_text_bytes(stdout, id, &text_bytes);
            true
        }
        FilterAction::Hold => false,
        FilterAction::Stop => true,
    }
}

/// Canonical generate-loop timing — the single source of truth for the `tok_s`,
/// `prefill_tok_s`, `decode_tok_s`, and `ttft_ms` fields every `done` event
/// carries. These are pure tokens/second arithmetic with no dependence on the
/// architecture, so no per-arch generate path should recompute (or, as happened
/// with the lfm2/minimax loops, forget to emit) them.
#[derive(Debug, Clone, Copy)]
pub struct GenTiming {
    /// Tokens produced during the decode phase.
    pub generated: usize,
    /// Prompt tokens consumed during prefill (for `prefill_tok_s`).
    pub prefill_tokens: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
}

fn tok_rate(n: usize, secs: f64) -> f64 {
    if secs > 0.0 {
        n as f64 / secs
    } else {
        0.0
    }
}

impl GenTiming {
    /// Build from the millisecond clocks the arch generate loops already keep
    /// (`prefill_ms`/`decode_ms` as `u128` elapsed millis).
    pub fn from_millis(
        generated: usize,
        prefill_tokens: usize,
        prefill_ms: u128,
        decode_ms: u128,
    ) -> Self {
        Self {
            generated,
            prefill_tokens,
            prefill_s: prefill_ms as f64 / 1000.0,
            decode_s: decode_ms as f64 / 1000.0,
        }
    }

    pub fn total_s(&self) -> f64 {
        self.prefill_s + self.decode_s
    }
    pub fn tok_s(&self) -> f64 {
        tok_rate(self.generated, self.total_s())
    }
    pub fn decode_tok_s(&self) -> f64 {
        tok_rate(self.generated, self.decode_s)
    }
    pub fn prefill_tok_s(&self) -> f64 {
        tok_rate(self.prefill_tokens, self.prefill_s)
    }
    pub fn ttft_ms(&self) -> f64 {
        self.prefill_s * 1000.0
    }

    /// The timing fields shared by every `done` event, rendered as a
    /// comma-separated JSON fragment with NO surrounding braces and NO leading
    /// comma — ready to splice into a `done` envelope after the `id`.
    pub fn done_fields(&self) -> String {
        format!(
            r#""tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}"#,
            self.generated,
            self.tok_s(),
            self.prefill_tokens,
            self.prefill_s * 1000.0,
            self.prefill_tok_s(),
            self.decode_tok_s(),
            self.ttft_ms(),
        )
    }
}

/// Emit the canonical `{"type":"done",...}` event for a generate loop. `extra`
/// is an optional pre-built JSON fragment for arch-/mode-specific fields and
/// MUST begin with a comma and carry no surrounding braces (e.g.
/// `,"dflash":true,"cycles":7`); pass `""` when there are none.
pub fn emit_done(stdout: &mut std::io::Stdout, id: &str, timing: &GenTiming, extra: &str) {
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}",{}{}}}"#,
        id,
        timing.done_fields(),
        extra,
    );
    let _ = stdout.flush();
}

#[cfg(test)]
mod timing_tests {
    use super::GenTiming;

    #[test]
    fn rates_are_tokens_over_seconds() {
        let t = GenTiming {
            generated: 100,
            prefill_tokens: 256,
            prefill_s: 0.5,
            decode_s: 2.0,
        };
        assert!((t.decode_tok_s() - 50.0).abs() < 1e-9); // 100 / 2.0
        assert!((t.prefill_tok_s() - 512.0).abs() < 1e-9); // 256 / 0.5
        assert!((t.tok_s() - 40.0).abs() < 1e-9); // 100 / (0.5 + 2.0)
        assert!((t.ttft_ms() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn zero_duration_yields_zero_not_nan() {
        // The lfm2/minimax loops can report a 0ms prefill (prefill_already_done);
        // the rate must be a finite 0.0, never a NaN/inf that breaks JSON parse.
        let t = GenTiming::from_millis(0, 0, 0, 0);
        assert_eq!(t.decode_tok_s(), 0.0);
        assert_eq!(t.prefill_tok_s(), 0.0);
        assert_eq!(t.tok_s(), 0.0);
    }

    #[test]
    fn done_fields_carries_the_canonical_keys() {
        let f = GenTiming::from_millis(64, 257, 318, 192).done_fields();
        for key in [
            "\"tokens\":",
            "\"tok_s\":",
            "\"prefill_tokens\":",
            "\"prefill_ms\":",
            "\"prefill_tok_s\":",
            "\"decode_tok_s\":",
            "\"ttft_ms\":",
        ] {
            assert!(f.contains(key), "missing {key} in {f}");
        }
        // No surrounding braces / leading comma — splices straight after `id`.
        assert!(!f.starts_with(','));
        assert!(!f.starts_with('{'));
    }
}

/// Emit a `{"type":"token","id","text"}` event for `text_bytes`. No-op on empty
/// input or non-UTF-8 bytes (a partial multibyte fragment is dropped rather than
/// emitting mojibake; the filter re-presents it once the codepoint completes).
pub fn emit_text_bytes(stdout: &mut std::io::Stdout, id: &str, text_bytes: &[u8]) {
    if text_bytes.is_empty() {
        return;
    }
    if let Ok(text) = std::str::from_utf8(text_bytes) {
        let _ = writeln!(
            stdout,
            r#"{{"type":"token","id":"{}","text":{}}}"#,
            id,
            serde_json::to_string(text).unwrap_or_default()
        );
        let _ = stdout.flush();
    }
}
