---
description: The operator dashboard - one process on your machine that shows the fleet, runs a simulated robot you can watch, and puts a Strands Agent behind consent cards.
---

# Dashboard

One command, one browser tab, on the machine the robots are reachable from.

```bash
uv pip install 'strands-robots[dashboard,sim-mujoco]'
python -m strands_robots dashboard --open
```

It binds `127.0.0.1:8090` and opens the page. Nothing is exposed to the network
until a passkey guards it.

## The first minute

1. **This machine, no passkey yet.** The dashboard is usable from a browser on
   the same machine, at `http://127.0.0.1:8090` or `localhost`, and refuses
   every other caller - a same-host proxy or `ssh -L` forward, a request whose
   `Host` is any other name (DNS rebinding), and a page from any other origin
   (cross-site `fetch` or `WebSocket`). None of those is presence at the
   machine.
2. **Enrol the owner passkey.** The login screen asks for a bootstrap token. It
   is in the `0600` file `enrol_token` beside `~/.strands_dashboard/auth.json`
   (or the `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` you set). Paste it, name the key,
   let the browser create the passkey. From that moment the API is sealed: every
   route but the login screen and `/api/health` answers `401` without a session.
   The login screen's own route says whether setup is required and which proof
   it needs - never the enrolled passkeys, which only a session may list.
3. **Bind a LAN address if you want to.** `--host 0.0.0.0` is refused until a
   passkey or a static `DASHBOARD_AUTH_TOKEN` exists, and says so.

```bash
python -m strands_robots dashboard --host 0.0.0.0 --port 8090
```

## What is on the page

| Tab | What it shows | Where the rules live |
|---|---|---|
| Fleet | every robot the registry knows, sim and real, and the mesh peers when the `[mesh]` extra is installed | `strands_robots.registry` |
| Sim | a MuJoCo robot stepping in this process - an MJPEG stream and the same model in your browser | `strands_robots.simulation` |
| Agent | a Strands Agent with the robot tool; anything that would move hardware pauses on a consent card | `dashboard.agent_hitl`, `dashboard.consent` |
| Settings | the file `~/.strands_robots/dashboard/settings.json` - agent model, mesh endpoints, static token (shown only as set / unset) | `dashboard.settings` |

Every string a route serves is rendered as text, never as markup: a Fleet row can
carry a mesh peer's name, and script running in this page would be same-origin -
it carries the session cookie and names this origin as its own, so it is behind
every guard above by construction. `tests/test_dashboard_static_renders_data_as_text.py`
reads that rule off the files the wheel ships.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `STRANDS_DASH_AUTH_STORE` | `~/.strands_dashboard/auth.json` | the passkey store; auth is on the moment it holds a credential |
| `STRANDS_DASH_AUTH_ENABLED` | read from the store | force on (`1`) or off (`0`); anything else is ignored with a warning |
| `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` | minted into `enrol_token` | the proof the first enrolment needs |
| `DASHBOARD_AUTH_TOKEN` | unset | a static bearer for scripts; a passkey session is still needed to remove a passkey |
| `DASHBOARD_SETTINGS_FILE` | `~/.strands_robots/dashboard/settings.json` | where Settings are written |

Every auth duration knob (`STRANDS_DASH_AUTH_TOKEN_TTL`, `SESSION_MAX_AGE`,
`HANDOFF_TTL`) is documented in the [configuration reference](reference/configuration.md);
none can be widened past its cap.

## See also

- [Security](security.md) - the threat model the dashboard is built against.
- [Mesh](mesh.md) - how a fleet e-stop reaches every peer.
