# hipfire admin UI

The Leptos/WASM console served by `hipfire-server` at `/admin/ui/`. It owns the
Overview, API Access, and Usage workflows; the legacy `/admin` page remains the
home for controls that have not moved yet.

Build the deployable assets:

```sh
NO_COLOR=true trunk build --release
```

Run the mocked same-origin browser workflows (install Chromium once with
`npx playwright install chromium`):

```sh
npm ci
npm run test:browser
```

The browser suite covers an expired-session login, keyboard tab activation,
user/token management, one-time token display, Usage rendering, narrow-screen
overflow, and baseline text contrast. It refreshes the review screenshots in
`docs/screenshots/`.
