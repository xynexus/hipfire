# API Access, Usage, and Rate-Limit Plan

## Summary

Add admin-managed API users, scoped tokens, workload-aware rate limiting,
durable Responses sessions, and token-level usage monitoring. Use
[`redb`](https://github.com/cberner/redb), a stable pure-Rust ACID key/value
store, behind a new `hipfire-auth` crate.

Authentication identifies a user; it never doubles as a conversation ID. Chat
Completions remains stateless, while Responses state is owned by
`(user_id, response_id)`.

## Core Architecture

- Add `~/.hipfire/access.redb` with private permissions and versioned tables
  for users, name indexes, tokens, user-token indexes, hourly usage, Responses
  contexts, expiry indexes, and audit events.
- Store only HMAC-SHA256 token digests using a separate `0600` pepper file.
  Token shape is `hfr_<token-id>_<32-byte-secret>` and the secret is returned
  once.
- Users are admin-managed, soft-disabled rather than deleted, and may own
  multiple tokens. New tokens expire after 90 days by default and carry
  `text`, `embeddings`, `images`, and `training` scopes.
- Add `api_auth_mode = auto | off | optional | required`; `auto` means optional
  on loopback and required on non-loopback. Unauthenticated remote binding
  requires an explicit unsafe override.
- Keep admin cookies/secrets separate from API credentials. Admin routes
  continue using `admin_gate`; API tokens never grant admin access.
- Put an immutable credential cache and live rate-limit state in memory. Use a
  blocking storage actor for redb transactions and batched usage flushes so
  request handling never blocks on file I/O.

## Request, Session, and Limit Integration

- Add `RequestPrincipal { user_id, token_id, scopes, auth_kind }` to request
  extensions. Missing credentials in optional local mode become
  `anonymous-local`; malformed, expired, revoked, or disabled credentials
  return `401`, and missing scopes return `403`.
- Protect `/v1/*` and `/sdapi/*`. Leave basic `/health`, static UI assets, and
  admin login public. Any valid token may list models.
- Extend orchestrator workloads with opaque user/token ownership. Schedule
  fairly across users within priority classes while still microbatching
  compatible work across users. Meter each result separately.
- Enforce user aggregate limits plus optional stricter token overrides:
  - 60 requests/minute with burst 15.
  - 120k text tokens/minute with burst 30k.
  - Four in-flight text jobs.
  - One image job and 80 megapixel-steps/minute with burst 40.
  - One training job; training remains exclusive.
- Reserve estimated text/image cost before scheduler admission, then settle
  against actual completion metrics. Return OpenAI-shaped `429` errors with
  `Retry-After` and rate-limit headers.
- Record requests, errors, rate-limit hits, input/output/cache tokens, images,
  megapixel-steps, and training seconds. Never record prompts, generated text,
  or images in usage telemetry. Flush hourly rollups asynchronously and retain
  90 days.
- Scope Responses lookup by user, not token, so token rotation preserves
  conversations. Persist authenticated response deltas and parent links for
  30 days, limited to 128 responses per user, 2 MiB per context, and chain
  depth 128. Anonymous local contexts remain memory-only.
- Token revocation or user disable blocks new work and removes queued workloads
  owned by that credential; active GPU work finishes unless explicitly
  cancelled.

## Admin API and UI

- Add admin-gated endpoints:
  - `GET/POST /admin/access/users`
  - `GET/PATCH /admin/access/users/{id}`
  - `GET/POST /admin/access/users/{id}/tokens`
  - `DELETE /admin/access/tokens/{id}`
  - `GET /admin/access/usage`
  - `GET /admin/access/rate-limits`
  - `GET /admin/access/audit`
- Use cursor pagination and typed contracts in `hipfire-admin-types`. Token
  creation returns the secret once with `Cache-Control: no-store`; revocation
  is idempotent. User patches support status and rate-policy overrides.
- Expand the Leptos `/admin/ui` console with Overview, Access, and Usage tabs.
  Do not duplicate this UI in the legacy inline console; retain a
  legacy-controls link until broader parity allows `/admin` to redirect.
- Access view: searchable user table, enabled/disabled filter, create-user
  command, user detail, limit editor, token list, generate-token flow,
  one-time copy confirmation, and explicit revoke/disable confirmations.
- Usage view: 24h/7d/30d/90d filters, user/token/workload filters, hourly trend,
  totals, current bucket remainder, active concurrency, rate-limit hits, and
  breakdown table.
- Provide loading skeletons, empty/error states, keyboard-complete controls,
  responsive tables, visible focus, and WCAG 2.2 AA contrast. Use the confirmed
  "quiet technical trust" product direction.
- Before UI implementation, add `PRODUCT.md` and `DESIGN.md` capturing the
  confirmed product register, operator audience, accessibility target, current
  teal/neutral visual system, and anti-patterns.
- Extend `hipfire-web-ui` with authenticated PATCH/DELETE helpers and consistent
  structured error handling. Add a proper Leptos admin login state for `401`
  responses.

## Testing and Rollout

- Unit-test redb migrations, indexes, token hashing, one-time secrets, expiry,
  revocation, user disable, scope checks, retention, and corruption fail-closed
  behavior.
- Test deterministic rate-limit clocks, reservation/refund, aggregate user
  limits, token overrides, concurrency release on cancellation,
  anonymous-local behavior, and `429` headers.
- Test cross-user Responses isolation, token rotation, restart recovery, expiry
  pruning, missing-parent behavior, and bounded chains.
- Integration-test all protected route groups, streaming completion accounting,
  embedding/image/training costs, batch-item scopes, queued cancellation, and
  microbatch accounting across users.
- Test admin CRUD, pagination, audit records, no-store token responses,
  same-origin mutation checks, and absence of secrets in logs/database exports.
- Build the WASM UI with `trunk`, run browser workflows for
  login/user/token/limits/usage, capture desktop and mobile screenshots, and
  verify keyboard and contrast behavior.
- Run targeted crate tests, `cargo check -p hipfire-server`,
  `./tests/no-gpu-ci.sh`, `git diff --check`, and `graphify update .`.
- Roll out backward-compatibly on loopback. Non-loopback deployments switch to
  required authentication under `auto`; document the upgrade and bootstrap
  flow through the existing admin password or local admin secret.
