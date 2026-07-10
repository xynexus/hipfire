# API access and usage controls

Hipfire separates administrator authentication from inference credentials.
Administrator sessions and `~/.hipfire/admin.secret` may manage the server;
they are not API tokens. API tokens may call only the workload scopes granted
to them and can never authorize `/admin/*`.

## Default rollout behavior

`api_auth_mode` defaults to `auto`:

- A loopback bind (`127.0.0.1`, `::1`, or localhost) accepts anonymous local
  API calls for backward compatibility. Presented credentials are still
  validated; a malformed, expired, revoked, or disabled token is not silently
  treated as anonymous.
- A non-loopback bind requires a valid API token. Startup refuses `off` or
  `optional` on a non-loopback address unless
  `unsafe_allow_unauthenticated_remote = true` is also set.

Explicit modes are `off`, `optional`, and `required`. The unsafe remote
override is an acknowledgement, not a recommended deployment mode.

## Bootstrap a remote server

Set an admin password on the server host before exposing the listener:

```sh
hipfire admin set-password
hipfire serve
```

Open `http://SERVER:11435/admin/ui/`, sign in, select **Access**, create an API
user, and generate a token with the required scopes. The token secret is shown
once. Copy it before confirming the panel; Hipfire stores only its
HMAC-SHA256 digest.

The local admin secret is a non-browser recovery path. It is created with mode
`0600` at `~/.hipfire/admin.secret` and should not leave the server host:

```sh
admin_secret=$(< ~/.hipfire/admin.secret)
user_id=$(
  curl -fsS http://127.0.0.1:11435/admin/access/users \
    -H "Authorization: Bearer $admin_secret" \
    -H 'Content-Type: application/json' \
    -d '{"name":"production-client","rate_policy":{}}' | jq -r .id
)
curl -fsS "http://127.0.0.1:11435/admin/access/users/$user_id/tokens" \
  -H "Authorization: Bearer $admin_secret" \
  -H 'Content-Type: application/json' \
  -d '{"label":"initial","scopes":["text"],"rate_policy":{}}'
```

The last response contains the only copy of the token secret and carries
`Cache-Control: no-store`.

## Use and rotate tokens

API tokens have the form `hfr_<token-id>_<secret>` and default to a 90-day
expiry. Send one as an ordinary bearer credential:

```sh
curl http://SERVER:11435/v1/models \
  -H "Authorization: Bearer $HIPFIRE_API_TOKEN"

curl http://SERVER:11435/v1/chat/completions \
  -H "Authorization: Bearer $HIPFIRE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hello"}]}'
```

Scopes are `text`, `embeddings`, `images`, and `training`. Any valid token may
list models. Missing scope returns `403`; invalid, expired, revoked, or
disabled credentials return `401`. Create a replacement token before revoking
the old one. Responses conversations belong to the API user rather than the
token, so rotation preserves `previous_response_id` chains.

Disabling a user or revoking a token blocks new work and removes queued work
owned by that credential. Active accelerator work is allowed to finish.

## Limits and telemetry

User policies apply in aggregate across their tokens. A token policy may only
make its user's limits stricter. Defaults are 60 requests/minute (burst 15),
120k text tokens/minute (burst 30k), four text jobs, one image job, 80
megapixel-steps/minute (burst 40), and one exclusive training job. Rejections
use OpenAI-shaped `429` responses with `Retry-After` and rate-limit headers.

The Usage view provides hourly request, error, rate-limit, token, image,
megapixel-step, and training-duration rollups plus live bucket/concurrency
state. Rollups are retained for 90 days. Hipfire does not put prompts,
generated text, or images in usage records.

Credentials, usage rollups, authenticated Responses contexts, and audit events
live in `~/.hipfire/access.redb`; the separate token pepper is mode `0600`.
Authenticated Responses contexts are retained for 30 days with per-user size,
count, and chain-depth bounds. Anonymous loopback contexts remain memory-only.
