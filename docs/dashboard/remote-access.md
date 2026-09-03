# Remote access: your home lab from a phone

The dashboard commands real motors. Putting it on the internet is therefore an
ordering problem, not a networking one: **prove the guard refuses a remote
client before you open the door.** This page does it in that order.

## What the guard actually does

Every `/api/*` and `/ws/*` route goes through one ASGI middleware (raw ASGI, not
`BaseHTTPMiddleware`, because a WebSocket scope never reaches an HTTP middleware -
and `/ws/mesh`, `/ws/chat`, `/ws/voice` are exactly the routes that drive motors
and spend money). It accepts, in order:

1. the static `security.auth_token` (env `DASHBOARD_AUTH_TOKEN`), or
2. a WebAuthn session JWT minted by a registered passkey.

With **neither** configured the posture is **local-only, never open**: loopback
passes, everything else is refused. And "loopback" is judged honestly - a
request carrying `cf-connecting-ip`, `x-forwarded-for` or `x-real-ip` is treated
as remote no matter what the socket peer says, because a tunnel connects *from*
localhost on behalf of the whole internet.

Verify it on your own machine before you trust it:

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:8090/api/fleet
# 200  <- you are local

curl -s -H 'x-forwarded-for: 203.0.113.9' localhost:8090/api/fleet
# {"detail":"unauthorized"}   <- 401, exactly what a tunnelled client would get
```

The static page still returns 200 - the app shell is not secret. The data,
the commands and the sockets are.

Only `/api/auth/status`, `/api/auth/register/{begin,finish}` and
`/api/auth/login/{begin,finish}` are exempt, so a browser can ask "do I need to
log in?" and complete a ceremony. Registration gates *itself*: once a
credential exists, open enrollment is over.

## Path A: a token and a link (works today)

```bash
DASHBOARD_AUTH_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')" \
  python -m strands_robots dashboard --port 8090
```

Now the 401 above applies to *everyone*, including you - so hand the browser the
token once:

```
https://robots.example.com/?token=<the token>
```

The frontend stores it in `localStorage`, sends `Authorization: Bearer …` on
every API call, and appends `?token=` to WebSocket URLs (a browser cannot set
headers on a WS handshake, so the query string is the only channel; the server
accepts it for `/ws` only). The link is also what makes a **QR code that puts a
phone straight on one robot** - `?backend=` picks the host, `?token=` unlocks it.

A URL-borne secret lands in history and in logs. Treat the link as the password
it is: generate a long token, use it once per device, and rotate it by
restarting with a new value. For anything beyond your own household, put an
identity proxy (Cloudflare Access and friends) in front of the tunnel as well.

## Path B: passkeys (shipped, end to end)

`strands_robots/dashboard/auth.py` implements the full WebAuthn rail - the
private key never leaves your Touch ID / Face ID enclave, the dashboard stores
only public keys in `~/.strands_dashboard/auth.json` (chmod 600, hot-reloaded on
mtime change), and a signed challenge mints a short-lived HS256 JWT.

```bash
STRANDS_DASH_AUTH_ENABLED=true \
STRANDS_DASH_AUTH_RP_ID=robots.example.com \
STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(24))')" \
  python -m strands_robots dashboard --port 8090

curl -s localhost:8090/api/auth/status
# {"enabled":false,"setup_required":true,"credentials":[],"bootstrap_required":true,
#  "rp_id":"localhost","secure_context":true,"rpid_usable":true,"authenticated":false}
```

`setup_required` means no credential exists yet; `bootstrap_required` means the
first enrollment must present that one-time token, so the open-enrollment window
cannot be walked through by whoever finds the URL first.

Enrollment happens in the browser. `AuthGate` probes `/api/fleet` alongside
`/api/auth/status` on mount and decides:

- **200 from `/api/fleet`** -> the door is already open (local dev, a static
  token, or a live session) and the gate shows nothing. The UI does not
  second-guess a server that just said yes.
- **401/403** -> the gate appears: the enroll form when `setup_required`, the
  login prompt otherwise. The enroll form asks for the bootstrap token only when
  the server demands one.

The minted JWT rides the same token plumbing as Path A (localStorage -> `Bearer`
header, `?token=` on sockets), so nothing below the gate knows passkeys exist.

**Enroll from the device you will actually use remotely** - your phone - because
the private key lives in *that* device's enclave and cannot be copied to it
later. A second device means a second enrollment, and `GET
/api/auth/credentials` shows both.

Two rules WebAuthn imposes that will otherwise waste your evening:

- **`rp_id` cannot be a raw IP.** `https://192.168.1.50` is refused by the
  browser before the ceremony starts; `/api/auth/status` says so in `warning`.
  Use a hostname, or force `STRANDS_DASH_AUTH_RP_ID`.
- **The origin must be secure**: HTTPS, or `http://localhost`. A LAN hostname
  over plain HTTP is not a secure context. The tunnel gives you HTTPS for free,
  which is why passkeys and tunnels arrive together.

## The tunnel

```bash
cloudflared tunnel create robots
# ~/.cloudflared/robots.yml
#   tunnel: <id>
#   credentials-file: /Users/you/.cloudflared/<id>.json
#   ingress:
#     - hostname: robots.example.com
#       service: http://localhost:8090
#     - service: http_status:404
cloudflared tunnel route dns robots robots.example.com
cloudflared tunnel run robots
```

Order of operations, and the reason for it:

1. Configure a credential (Path A token, or a passkey once you can enroll one).
2. **Restart the dashboard** - the guard is loaded at startup, so a running
   process that predates your change proves nothing. A 200 from an old process
   is stale code, not evidence.
3. Re-run the `x-forwarded-for` probe above and confirm the 401.
4. *Then* `cloudflared tunnel run`.
5. Load the site on the phone and check that a **wrong** token is refused.

## Once you are remote

- **E-stop from a phone is the feature**, not a curiosity - it is the reason
  this is worth exposing. Set the **same** `STRANDS_MESH_OVERRIDE_CODE` on every
  peer, or a broadcast stop locks each robot until its process restarts.
- Cameras are the bandwidth. Lower `STRANDS_MESH_CAMERA_HZ` on the publishing
  peer rather than fighting the uplink.
- The mesh is a separate trust domain from the web session. `--local-dev`
  disables mesh TLS for single-machine work; do not leave it on for a fleet
  that spans machines. See [Multi-robot Mesh](../mesh.md).
- A session lasts `STRANDS_DASH_AUTH_TOKEN_TTL` seconds (default 86400).
  `GET /api/auth/credentials` lists enrolled passkeys and
  `DELETE /api/auth/credentials/{id}` revokes a lost phone.

## Who may start real motion, once the dashboard is on the internet

The web guard decides who gets *in*. It says nothing about what a caller who is
in may do, and the loudest thing in this system is an arm that starts moving in
a room nobody is standing in. Two separate switches, deliberately separate:

| variable | default | what it changes |
|---|---|---|
| `STRANDS_DASH_AGENT_PHYSICAL_MOTION` | unset (refuse) | Lets the **agent** start a task on a real robot by itself — from a chat sentence or a voice command. Unset, it refuses and offers you the ▶ button instead. |
| `STRANDS_DASH_TASK_REQUIRES_CONFIRM` | unset (allow) | Set it, and `POST /api/robots/{peer}/task` refuses a real-motion request that does not carry the browser's confirmation. The ▶ button sends it; `curl` does not. |

Both are visible on **Settings → permissions**, the second one with a one-tap
toggle, so neither has to be flipped by hand.

Read the defaults honestly:

- The **agent** is refused by default, because a sentence is easy to say by
  accident and a policy that does not fit the arm it is pointed at is a
  collision, not an error message.
- **You** are not, because the API token *is* the operator: the deploy snippet,
  your own scripts and every test post to that route. Enforcing a confirmation
  that a client can simply assert would not stop an attacker — it would only
  break the callers you meant to keep.

So `STRANDS_DASH_TASK_REQUIRES_CONFIRM` is an **anti-accident** lock, not a
security boundary. It is worth turning on once the dashboard is tunnelled,
because that is the moment the token stops living only on your desk: after a
leak, a single `curl` is real motion. If the token *has* leaked, this lock is
not the fix — rotate the token and re-enroll the passkey.

Stopping is never gated by either variable. A guard that can trap a moving arm
would be worse than no guard, so `stop`, `stop_all` and the e-stop path ignore
both.
