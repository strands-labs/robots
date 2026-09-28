# Dashboard

At the end of this page `strands-robots dashboard` is serving, and you know what each tab does, who may click, and what the e-stops do.

```bash
pip install 'strands-robots[dashboard,sim-mujoco]'      # fastapi, uvicorn, webauthn, PyJWT + MuJoCo for the twin
strands-robots dashboard --open                          # http://127.0.0.1:8090
```

Flags: `--host` (default `127.0.0.1`), `--port` (default `8090`), `--open`, `--log-level`. A non-loopback host is refused until a passkey or static `security.auth_token` guards the API.

## What it serves

The process joins the Zenoh mesh as a robot-less gateway: one page drives hardware, simulators, or a mix. The UI is a built React SPA under `strands_robots/dashboard/static/`; no node at runtime. Each tab's rules live in the `strands_robots.dashboard` module named for it.

| tab | shows |
|---|---|
| Fleet | every live mesh peer (joints, cameras, task, lockout) and every registry robot; teleop pairing, a task form |
| Devices | this machine's serial ports and cameras; spawn a robot process per port (a managed mesh child), assign cameras, read its log, despawn |
| Record | a LeRobot dataset session: arms, cameras, start / stop / redo / discard episodes, thumbnails, labels, upload |
| Train | datasets, trainers, a graded job form, live loss, checkpoints, validation on a robot, a deploy snippet |
| Calibrate | the LeRobot calibration wizard, with a confirm before the arm moves |
| Sim | a MuJoCo robot stepping in this process, streamed to the browser; or a **mirror**, the twin posed from a real arm's servo bus, never written |
| Agent | a Strands Agent over the fleet and the simulations; anything that moves a robot pauses on a consent card; a microphone opens voice |
| Settings | agent model and prompt, mesh endpoints, voice provider, editable `.env` keys (a closed set; gates read-only), static token shown only as set / unset |

Every path a client names is resolved and must sit under its home (`HF_LEROBOT_HOME`, `STRANDS_TRAIN_OUTPUT_DIR`, the Hub cache); anything else gets one refusal that never says whether the path exists. A port must be a `/dev/...` path, a robot id one segment, a camera name `[A-Za-z0-9._-]`: each becomes a file name or argv.

Two e-stops: `POST /api/safety/estop` stops this process's simulations; `POST /api/mesh/safety/estop` is the signed fleet stop, whose answer names the peers that did not reply. The page fires both.

## Who may click

One dependency, `access.require_session`, guards every route but the login screen and `/api/health`. Three ways in, first match wins:

1. A passkey session token, as `Authorization: Bearer` or the `strands_dash` cookie; never a query string, which would land in access logs.
2. The static `security.auth_token`, compared in constant time.
3. Nothing, but only while no passkey is enrolled and the request is this machine's own browser: loopback socket, no proxy header, a loopback `Host`, an `Origin` naming the same host. A DNS-rebound page or cross-site fetch fails there.

The first passkey closes the third door. Its enrollment must present `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN`, or the token the process minted into a `0600` file beside the credential store; loopback alone is not presence. `STRANDS_DASH_AUTH_ORIGIN` and `STRANDS_DASH_AUTH_RP_ID` pin the WebAuthn origin and relying party behind a proxy; `STRANDS_DASH_AUTH_TOKEN_TTL` (86400 s), `..._SESSION_MAX_AGE` (2592000 s) and `..._HANDOFF_TTL` (300 s) bound a session; a value that is not a whole number of seconds is refused, not defaulted: every substituted default is the wider one.

## The e-stop button

`POST /api/safety/estop` stops every sim session in this process and locks; every route that would move a sim answers `423` until it clears. `POST /api/safety/resume` sets the state to `unknown` on purpose: a resume is a request, not proof; the first command a session accepts is, and only then does `GET /api/safety` say `clear`. The signed fleet stop ([safety and e-stop](mesh/safety-and-estop.md)) is a separate rail; one reaching this process engages the same lockout.

## The agent in the browser

`/ws/agent` takes `{"type": "say", "text": ...}` and streams the console's events back (text, tool_use, tool_result, interrupt, done, error). Its tools share the HTTP routes' `Safety` object, so the e-stop refuses the agent like a button. `sim_set_joints` raises the real-hardware hook's interrupt (`MotionInterruptHook`, see [agents](agents.md)); the browser shows a consent card and `{"type": "resume", "id": ..., "approve": true, "always": false}` resumes the turn; `always` lasts the conversation and dies with the socket. One turn per socket; a second `say` is refused, not queued.

Two switches, off by default, matter once a physical peer is reachable: `STRANDS_DASH_AGENT_PHYSICAL_MOTION=1` lets the agent's tools move metal at all, and `STRANDS_DASH_TASK_REQUIRES_CONFIRM=1` makes a real-motion task or teleop POST carry an explicit boolean confirmation (strings are refused). Neither touches a simulated peer; both are granted and revoked from a consent card, never from Settings.

## Logs

Every log line passes `log_redaction`: tokens, cookies and credential ids are masked first.
