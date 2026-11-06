### Added: passkey sign-in for the dashboard; every `/api` route and socket needs a session

The dashboard now answers `/api/*` and `/ws/*` only to a signed-in session. You sign in with a passkey (WebAuthn:
Touch ID, Windows Hello, a phone or a security key). The first start prints a one-time setup link, and the first
passkey can only be created with it. More passkeys need a signed-in session, and the last one cannot be removed. The
**Passkeys** tab adds, renames and removes them and signs out. Sessions are an HttpOnly, SameSite=Strict cookie that
lasts 12 hours (`Auth(ttl=...)`, `STRANDS_ROBOTS_DASHBOARD_SESSION_TTL`). Scripts, notebooks and agents use
`dash.auth.token()` (or `--print-token`) as `Authorization: Bearer <token>`.

Before, any local process that could reach the port could send `X-Confirm: yes` and release an e-stop or move a real
robot. `X-Confirm` still asks the operator, but now only a signed-in session can send it. Passkeys need a hostname,
so the dashboard's address is now `http://localhost:<port>`, and the page on `127.0.0.1` redirects there.
`Dashboard(sim, auth=False)` (`--no-auth`) turns sign-in off with a warning and is refused while a real robot is
attached unless `unauthenticated_hardware=True`. The `dashboard` extra now installs `webauthn`.
`Dashboard.revision` is removed: use `dash.sim.revision`.
