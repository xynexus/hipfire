// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Dialect-agnostic grammar-guided decoding for tool calls.
//!
//! Constrained decoding was built for DeepSeek V4's DSML format and lived inside
//! `hipfire-arch-deepseek4`, but only the *state machine* is dialect-specific. The
//! engine around it — "does this token keep us on a legal path", the vocab mask, the
//! logit zeroing — is the same for every family. This module holds that engine so a
//! second dialect costs a state machine, not a second copy of the machinery.
//!
//! Each family declares tool calls in its own syntax, so the state machines genuinely
//! differ:
//!
//! - DeepSeek V4 — `<｜DSML｜tool_calls>` / `<｜DSML｜invoke name="…">`
//! - MiniCPM5    — `<function name="f"><param name="p">v</param></function>`
//! - Qwen3.5     — `<tool_call>\n<function=f>\n<parameter=p>\nv\n</parameter>…`
//! - zaya1       — as Qwen3.5, wrapped in `<zyphra_tool_call>`
//!
//! A [`ToolGrammar`] reports the legal byte-string continuations from its current
//! position; [`token_mask`] turns that into a vocab bitmask and [`apply_mask_to_logits`]
//! zeroes the disallowed logits before sampling. A model physically cannot emit a
//! malformed call — which is a stronger guarantee than finetuning it to prefer valid
//! ones.

/// Schema for the tools available to a request, built from the OpenAI-format tools
/// array. A grammar uses this to constrain tool names and parameter names at the
/// positions where they appear.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    /// Parameter names in schema order. Order isn't enforced at parse time — params
    /// may be emitted in any order — but this is the authoritative set of legal names.
    pub params: Vec<String>,
    /// Subset of `params` that MUST appear before the call may close. A grammar drops
    /// its close alternatives until every required param has been seen, which stops a
    /// model emitting an empty call that the downstream client then rejects.
    pub required: Vec<String>,
}

/// A tool-call dialect as a byte-level state machine.
///
/// Implementors track where they are in their own syntax and report what may legally
/// come next. Everything else — masking, logit zeroing — is provided here.
pub trait ToolGrammar {
    /// No constraint applies right now: every token is legal. Free emission is the
    /// common case (prose, code, parameter bodies), so this is also the fast path that
    /// skips the per-token scan entirely.
    fn is_free(&self) -> bool;

    /// Bytes committed since the last firm state transition. A continuation is matched
    /// against `partial() + decode(token)`, so a marker may span several tokens.
    fn partial(&self) -> &str;

    /// Full legal continuations from the position after the last firm transition.
    /// Empty when [`Self::is_free`] is true.
    fn allowed_continuations(&self) -> Vec<String>;

    /// Commit decoded bytes, advancing state when a continuation completes. Must be
    /// idempotent at the byte level: single bytes or larger chunks reach the same state.
    fn advance(&mut self, text: &str);

    /// Whether leading whitespace may precede the next literal in this state, so a
    /// token like `\n<fun` is accepted when the newline will be consumed on transition.
    fn allows_leading_ws(&self) -> bool {
        false
    }
}

/// Whether `text` keeps `grammar` on a legal path — i.e. `partial + text` is a prefix
/// of (or extends past) at least one allowed continuation.
pub fn is_token_allowed<G: ToolGrammar + ?Sized>(grammar: &G, text: &str) -> bool {
    if grammar.is_free() {
        return true;
    }
    let combined = format!("{}{}", grammar.partial(), text);
    let conts = grammar.allowed_continuations();
    if matches_any(&combined, &conts) {
        return true;
    }
    if grammar.allows_leading_ws() {
        return matches_any(combined.trim_start_matches(['\n', ' ']), &conts);
    }
    false
}

/// A candidate is viable if it is a prefix of a continuation, or already extends past
/// one (the surplus belongs to the next state).
fn matches_any(s: &str, conts: &[String]) -> bool {
    conts
        .iter()
        .any(|cont| cont.starts_with(s) || s.starts_with(cont.as_str()))
}

/// Populate a boolean mask over `vocab`: which tokens are legal at the current
/// position. `out` must be at least `vocab.len()` long; entries past that are untouched.
///
/// Fast path: when the grammar is free the whole mask is set true and the caller can
/// skip masking. Hot path is an O(vocab) scan — with ~129k entries and a handful of
/// alternatives per state it stays sub-millisecond per sample step.
pub fn token_mask<G: ToolGrammar + ?Sized>(grammar: &G, vocab: &[String], out: &mut [bool]) {
    debug_assert!(out.len() >= vocab.len());
    if grammar.is_free() {
        for slot in out.iter_mut().take(vocab.len()) {
            *slot = true;
        }
        return;
    }
    for (id, text) in vocab.iter().enumerate() {
        out[id] = is_token_allowed(grammar, text);
    }
}

/// Apply a mask in-place to logits: disallowed tokens become `-inf`, allowed ones are
/// untouched. Split from [`token_mask`] so a caller can reuse one `Vec<bool>` across
/// the decode loop.
pub fn apply_mask_to_logits(mask: &[bool], logits: &mut [f32]) {
    let n = mask.len().min(logits.len());
    for (i, allowed) in mask.iter().enumerate().take(n) {
        if !allowed {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal grammar: `<a>` then `<b>`, free afterwards. Enough to exercise the
    /// engine without dragging a real dialect's state machine into these tests.
    struct TwoStep {
        seen: usize,
        partial: String,
    }

    impl ToolGrammar for TwoStep {
        fn is_free(&self) -> bool {
            self.seen >= 2
        }
        fn partial(&self) -> &str {
            &self.partial
        }
        fn allowed_continuations(&self) -> Vec<String> {
            match self.seen {
                0 => vec!["<a>".to_string()],
                1 => vec!["<b>".to_string()],
                _ => Vec::new(),
            }
        }
        fn advance(&mut self, text: &str) {
            self.partial.push_str(text);
            while let Some(want) = self.allowed_continuations().first().cloned() {
                if self.partial.starts_with(&want) {
                    self.partial = self.partial[want.len()..].to_string();
                    self.seen += 1;
                } else {
                    break;
                }
            }
        }
    }

    fn two_step() -> TwoStep {
        TwoStep {
            seen: 0,
            partial: String::new(),
        }
    }

    #[test]
    fn a_token_is_allowed_only_while_it_stays_on_a_legal_path() {
        let g = two_step();
        assert!(is_token_allowed(&g, "<"), "prefix of <a>");
        assert!(is_token_allowed(&g, "<a>"), "exact continuation");
        assert!(!is_token_allowed(&g, "x"), "off-path token is rejected");
        assert!(!is_token_allowed(&g, "<b>"), "right marker, wrong position");
    }

    #[test]
    fn a_marker_may_span_several_tokens() {
        let mut g = two_step();
        g.advance("<");
        assert_eq!(g.partial(), "<", "partial marker is held, not discarded");
        assert!(is_token_allowed(&g, "a>"), "completes across the split");
        assert!(!is_token_allowed(&g, "z>"));
        g.advance("a>");
        assert_eq!(g.seen, 1, "firm transition once the marker completes");
    }

    #[test]
    fn a_free_grammar_allows_everything_and_short_circuits_the_mask() {
        let mut g = two_step();
        g.advance("<a><b>");
        assert!(g.is_free());
        assert!(is_token_allowed(&g, "literally anything"));
        let vocab: Vec<String> = ["x", "<a>", "!"].iter().map(|s| s.to_string()).collect();
        let mut mask = vec![false; vocab.len()];
        token_mask(&g, &vocab, &mut mask);
        assert!(mask.iter().all(|&m| m), "free -> whole vocab allowed");
    }

    #[test]
    fn the_mask_selects_exactly_the_viable_vocabulary() {
        let g = two_step();
        let vocab: Vec<String> = ["<a>", "<", "nope", "<b>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut mask = vec![true; vocab.len()];
        token_mask(&g, &vocab, &mut mask);
        assert_eq!(mask, vec![true, true, false, false]);
    }

    #[test]
    fn masking_zeroes_disallowed_logits_and_leaves_the_rest_alone() {
        let mask = [true, false, true];
        let mut logits = [1.0_f32, 2.0, 3.0];
        apply_mask_to_logits(&mask, &mut logits);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
        assert_eq!(logits[2], 3.0);
        // a mask shorter than the logits must not panic or touch the tail
        let mut logits = [1.0_f32, 2.0, 3.0];
        apply_mask_to_logits(&[false], &mut logits);
        assert!(logits[0].is_infinite());
        assert_eq!(logits[2], 3.0);
    }
}

// ── MiniCPM5 XML dialect ────────────────────────────────────────────────

/// Grammar for MiniCPM5's tool-call syntax:
///
/// ```text
/// <function name="read_file"><param name="path">src/lib.rs</param></function>
/// ```
///
/// Parameter values are free text (they carry code, so they must be), optionally
/// wrapped in `<![CDATA[…]]>` when they contain `<`, `&` or newlines — which is what
/// makes multi-param writes safe in this dialect where JSON needs escaping the model
/// reliably gets wrong.
///
/// Constrained: the opener, tool names, param names, and the closing tags. Free: the
/// parameter bodies. That keeps the constraint surface tight — the model writes code
/// however it likes but cannot invent a tag or misspell a tool.
#[derive(Debug, Clone)]
pub struct MiniCpmXmlGrammar {
    state: XmlState,
    partial: String,
    tools: Vec<ToolSchema>,
    /// Index into `tools` for the call being emitted.
    tool_idx: Option<usize>,
    /// Param names already emitted for the current call, so required-param tracking
    /// can withhold the close.
    emitted: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum XmlState {
    /// Free prose. Watching for `<` to begin a call.
    Out,
    /// After `<function name="` — emitting a tool name.
    InToolName,
    /// Between tags inside a call: next is a param open or the function close.
    InBody,
    /// After `<param name="` — emitting a param name.
    InParamName,
    /// Inside a param value. Free until `</` appears.
    InParamBody,
}

const FN_OPEN: &str = "<function name=\"";
const FN_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<param name=\"";
const PARAM_CLOSE: &str = "</param>";

/// Trim `buf` down to the longest suffix that could still be the beginning of
/// `marker`, discarding everything before it. Without this a `<` occurring naturally
/// inside a parameter value (`if a < b`) would be mistaken for the start of a close
/// tag and wrongly constrain the rest of the body.
fn keep_marker_prefix(buf: &mut String, marker: &str) {
    let max = buf.len().min(marker.len().saturating_sub(1));
    for len in (1..=max).rev() {
        let start = buf.len() - len;
        if buf.is_char_boundary(start) && marker.starts_with(&buf[start..]) {
            buf.drain(..start);
            return;
        }
    }
    buf.clear();
}

impl MiniCpmXmlGrammar {
    pub fn new(tools: Vec<ToolSchema>) -> Self {
        Self {
            state: XmlState::Out,
            partial: String::new(),
            tools,
            tool_idx: None,
            emitted: Vec::new(),
        }
    }

    fn current_tool(&self) -> Option<&ToolSchema> {
        self.tool_idx.and_then(|i| self.tools.get(i))
    }

    /// Every required param emitted, so the call may close.
    fn required_satisfied(&self) -> bool {
        self.current_tool().is_some_and(|tool| {
            tool.required
                .iter()
                .all(|req| self.emitted.iter().any(|e| e == req))
        })
    }

    /// Consume one firm transition if the buffer completes a marker. Returns true when
    /// state changed, so the caller can look for another in the same buffer.
    fn transition_once(&mut self) -> bool {
        match self.state {
            XmlState::Out => {
                if let Some(at) = self.partial.find(FN_OPEN) {
                    self.partial.drain(..at + FN_OPEN.len());
                    self.state = XmlState::InToolName;
                    self.tool_idx = None;
                    self.emitted.clear();
                    return true;
                }
                keep_marker_prefix(&mut self.partial, FN_OPEN);
                false
            }
            XmlState::InToolName => {
                let Some(quote) = self.partial.find("\">") else {
                    return false;
                };
                let name = self.partial[..quote].to_string();
                self.tool_idx = self.tools.iter().position(|t| t.name == name);
                self.partial.drain(..quote + 2);
                self.state = XmlState::InBody;
                true
            }
            XmlState::InBody => {
                if self.partial.starts_with(PARAM_OPEN) {
                    self.partial.drain(..PARAM_OPEN.len());
                    self.state = XmlState::InParamName;
                    return true;
                }
                if self.partial.starts_with(FN_CLOSE) {
                    self.partial.drain(..FN_CLOSE.len());
                    self.state = XmlState::Out;
                    self.tool_idx = None;
                    return true;
                }
                false
            }
            XmlState::InParamName => {
                let Some(quote) = self.partial.find("\">") else {
                    return false;
                };
                let name = self.partial[..quote].to_string();
                self.emitted.push(name);
                self.partial.drain(..quote + 2);
                self.state = XmlState::InParamBody;
                true
            }
            XmlState::InParamBody => {
                if let Some(at) = self.partial.find(PARAM_CLOSE) {
                    self.partial.drain(..at + PARAM_CLOSE.len());
                    self.state = XmlState::InBody;
                    return true;
                }
                keep_marker_prefix(&mut self.partial, PARAM_CLOSE);
                false
            }
        }
    }
}

impl ToolGrammar for MiniCpmXmlGrammar {
    fn is_free(&self) -> bool {
        match self.state {
            // Prose is free until the buffer looks like the start of `<function`.
            // A bare `<` is far too common in prose and code to constrain on.
            XmlState::Out => !self.partial.starts_with("<f"),
            // Param values carry code, so `<` alone means nothing; only `</` can be
            // the close tag starting.
            // ponytail: a literal `</...` inside a value (e.g. HTML in a string) is
            // forced to `</param>`. Same ceiling DSML accepts; escape via CDATA.
            XmlState::InParamBody => !self.partial.starts_with("</"),
            _ => false,
        }
    }

    fn partial(&self) -> &str {
        &self.partial
    }

    fn allowed_continuations(&self) -> Vec<String> {
        match self.state {
            XmlState::Out => vec![FN_OPEN.to_string()],
            // Only names from the schema — the model cannot invent a tool.
            XmlState::InToolName => self
                .tools
                .iter()
                .map(|t| format!("{}\">", t.name))
                .collect(),
            XmlState::InBody => {
                let mut out = vec![PARAM_OPEN.to_string()];
                // Withhold the close until every required param has been emitted.
                if self.required_satisfied() {
                    out.push(FN_CLOSE.to_string());
                }
                out
            }
            XmlState::InParamName => self
                .current_tool()
                .map(|tool| {
                    tool.params
                        .iter()
                        .filter(|p| !self.emitted.iter().any(|e| &e == p))
                        .map(|p| format!("{p}\">"))
                        .collect()
                })
                .unwrap_or_default(),
            XmlState::InParamBody => vec![PARAM_CLOSE.to_string()],
        }
    }

    fn advance(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.partial.push_str(text);
        while self.transition_once() {}
    }
}

#[cfg(test)]
mod minicpm_tests {
    use super::*;

    fn tools() -> Vec<ToolSchema> {
        vec![
            ToolSchema {
                name: "read_file".to_string(),
                params: vec!["path".to_string()],
                required: vec!["path".to_string()],
            },
            ToolSchema {
                name: "write_file".to_string(),
                params: vec!["path".to_string(), "contents".to_string()],
                required: vec!["path".to_string(), "contents".to_string()],
            },
        ]
    }

    fn feed(g: &mut MiniCpmXmlGrammar, s: &str) {
        g.advance(s);
    }

    #[test]
    fn a_well_formed_call_walks_the_states_and_returns_to_free() {
        let mut g = MiniCpmXmlGrammar::new(tools());
        assert!(g.is_free(), "prose is unconstrained");
        feed(&mut g, "<function name=\"");
        assert_eq!(g.state, XmlState::InToolName);
        feed(&mut g, "read_file\">");
        assert_eq!(g.state, XmlState::InBody);
        feed(&mut g, "<param name=\"");
        assert_eq!(g.state, XmlState::InParamName);
        feed(&mut g, "path\">");
        assert_eq!(g.state, XmlState::InParamBody);
        feed(&mut g, "src/lib.rs</param>");
        assert_eq!(g.state, XmlState::InBody);
        feed(&mut g, "</function>");
        assert_eq!(g.state, XmlState::Out);
        assert!(g.is_free());
    }

    #[test]
    fn only_schema_tool_names_are_reachable() {
        let mut g = MiniCpmXmlGrammar::new(tools());
        feed(&mut g, "<function name=\"");
        assert!(is_token_allowed(&g, "read_file\">"));
        assert!(is_token_allowed(&g, "read"), "prefix of a real tool");
        assert!(!is_token_allowed(&g, "delete_everything\">"));
        assert!(!is_token_allowed(&g, "zzz"));
    }

    #[test]
    fn the_close_is_withheld_until_required_params_are_emitted() {
        let mut g = MiniCpmXmlGrammar::new(tools());
        feed(&mut g, "<function name=\"write_file\">");
        // contents is still missing -> the model cannot close the call
        assert!(!is_token_allowed(&g, FN_CLOSE));
        feed(&mut g, "<param name=\"path\">x</param>");
        assert!(!is_token_allowed(&g, FN_CLOSE), "contents still missing");
        feed(&mut g, "<param name=\"contents\">y</param>");
        assert!(is_token_allowed(&g, FN_CLOSE), "now closable");
    }

    #[test]
    fn a_param_name_cannot_be_invented_or_repeated() {
        let mut g = MiniCpmXmlGrammar::new(tools());
        feed(&mut g, "<function name=\"write_file\"><param name=\"");
        assert!(is_token_allowed(&g, "path\">"));
        assert!(is_token_allowed(&g, "contents\">"));
        assert!(!is_token_allowed(&g, "sudo\">"), "not in the schema");
        feed(&mut g, "path\">v</param><param name=\"");
        assert!(!is_token_allowed(&g, "path\">"), "already emitted");
        assert!(is_token_allowed(&g, "contents\">"));
    }

    #[test]
    fn parameter_bodies_stay_free_so_code_and_cdata_pass_through() {
        let mut g = MiniCpmXmlGrammar::new(tools());
        feed(
            &mut g,
            "<function name=\"write_file\"><param name=\"path\">",
        );
        assert!(g.is_free(), "value bodies are unconstrained");
        // CDATA-wrapped multi-line code with < and & must survive verbatim
        feed(&mut g, "<![CDATA[if a < b && c { }\nlet x = 1;");
        assert!(is_token_allowed(&g, "anything at all"));
        feed(&mut g, "]]></param>");
        assert_eq!(g.state, XmlState::InBody);
    }

    // Regression: a `<` inside a value (comparison, generic, HTML) must not be read as
    // the start of `</param>` and constrain the rest of the body.
    #[test]
    fn a_bare_angle_bracket_inside_a_value_does_not_constrain() {
        let mut g = MiniCpmXmlGrammar::new(tools());
        feed(
            &mut g,
            "<function name=\"write_file\"><param name=\"contents\">",
        );
        feed(&mut g, "if a < b && c > d { }");
        assert!(g.is_free(), "comparison operators keep the body free");
        assert!(is_token_allowed(&g, "\nlet v: Vec<u8> = vec![];"));
        feed(&mut g, "</param>");
        assert_eq!(g.state, XmlState::InBody, "the real close still fires");
    }

    #[test]
    fn markers_split_across_tokens_still_transition() {
        let mut g = MiniCpmXmlGrammar::new(tools());
        for frag in ["<fun", "ction", " name", "=\"", "read", "_file", "\">"] {
            feed(&mut g, frag);
        }
        assert_eq!(g.state, XmlState::InBody, "reassembled across 7 tokens");
    }
}
