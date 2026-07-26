# Goal Prompt: API Access, Usage, and Rate Limits

Implement the approved API access, usage accounting, rate limiting, durable
Responses sessions, and admin user-management plan in `/home/sadara/hipfire`.

The canonical specification is:

`docs/plans/2026-07-10-api-access-usage-rate-limits.md`

Start from the `chaingun` branch at or after orchestrator baseline commit
`0477058b8`. Preserve unrelated user changes and never commit
`.agents/scheduled_tasks.lock`. Follow all `AGENTS.md` instructions, query the
existing Graphify graph before code exploration, and run `graphify update .`
after code changes.

## Objective

Deliver admin-managed API users and scoped tokens, user-owned durable Responses
contexts, workload-aware rate limiting, per-token usage monitoring, scheduler
ownership and fairness, admin APIs, and the canonical Leptos Access/Usage UI.

## Required Implementation Order

1. Add the versioned `hipfire-auth` redb store, public types, migrations,
   HMAC-SHA256 token hashing, credential cache, and tests.
2. Add API principal middleware and the `auto/off/optional/required` rollout
   policy without changing the existing admin-auth boundary.
3. Add deterministic request, token, image-cost, and concurrency limiters with
   reservation/settlement and OpenAI-shaped `429` responses.
4. Add privacy-preserving hourly usage rollups, audit events, retention, and
   batched durable writes.
5. Scope and persist Responses contexts by user while keeping Chat Completions
   stateless and anonymous-local Responses memory-only.
6. Carry user/token ownership through the continuous scheduler, preserve fair
   admission, and allow compatible cross-user microbatching with separate
   metering.
7. Add the typed admin user/token/usage/rate-limit APIs and one-time token
   disclosure.
8. Add `PRODUCT.md` and `DESIGN.md`, then build the Leptos Overview/Access/Usage
   console with login, loading, empty, error, responsive, keyboard, and
   destructive-action states.
9. Update operator/configuration documentation, Graphify, and generated config
   or environment documentation required by the repository.

## Constraints

- Keep inference Rust and HIP/ROCm-direct; do not add Python to the hot path.
- Use `redb` as the only new persistent auth/usage store.
- Never persist raw tokens, prompts, generated text, or images in usage data.
- Keep API tokens separate from admin cookies and the local admin bearer secret.
- Keep token verification and rate-limit checks in memory on the request hot
  path; isolate blocking database work behind a storage actor.
- Preserve the existing `hipfire-lock` contract; scheduler accounting is not a
  replacement lock primitive.
- Keep remote access fail-closed and loopback behavior backward-compatible.
- Add no hard user deletion; disabling users and revoking tokens must preserve
  audit and usage history.

## Acceptance and Verification

- Auth, storage, migrations, rate limits, Responses ownership, scheduler
  attribution, usage retention, and admin API tests pass.
- Cross-user Responses access is rejected without revealing whether another
  user's response ID exists.
- Raw token secrets appear only in the single token-creation response and are
  absent from logs and persisted records.
- Remote API calls without credentials fail under `auto`; loopback anonymous
  calls retain compatibility.
- Rate-limited work receives `429` before model load or scheduler enqueue, and
  reservations are released on completion, error, disconnect, and cancellation.
- The Leptos UI builds with `trunk`; desktop/mobile and keyboard workflows for
  login, user creation, token generation/copy, revocation, limits, and usage
  filtering are verified.
- Run focused crate tests, `cargo check -p hipfire-server`,
  `./tests/no-gpu-ci.sh`, `git diff --check`, and `graphify update .`. Report any
  unrelated pre-existing gate failure with exact evidence rather than changing
  unrelated files.

Implement and validate the work in reviewable commits. Do not stop at policy
types or mock UI: carry the feature through middleware, persistence, scheduler,
admin API, and browser workflows end to end.
