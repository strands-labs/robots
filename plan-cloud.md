# Phase 3: move the control plane to the cloud

Goal: run Strands (dashboard, mesh hub, agent, sims, GPU policies) in the cloud as the main dev/test
environment. Only hardware stays home: the Pi (follower) and the desktop leader arms. No Tailscale.
Draft 2026-11-08, nothing built yet. Open decisions at the bottom.

## Target

```
 home (outbound only, no open ports)          cloud
 ┌──────────────────────┐   tls/mTLS :7447   ┌───────────────────────────────────────────┐
 │ Pi: xlerobot-host    │──────────────────▶ │ CPU VM (control plane, docker compose)    │
 │     pi_control.py    │                    │  caddy      HTTPS, dash.<domain>          │
 ├──────────────────────┤                    │  dashboard  REST + WS API + UI            │
 │ desktop: leader_peer │──────────────────▶ │  zenohd     broker, TLS, mTLS client certs│
 └──────────────────────┘                    │  sim worker MuJoCo twins (CPU render)     │
                                             └───────────────┬───────────────────────────┘
                                                             │ HTTPS API (per action chunk)
                                             ┌───────────────▼───────────────────────────┐
                                             │ GPU: policy inference (serverless, 0→1)   │
                                             │      training jobs (on-demand pods)       │
                                             └───────────────────────────────────────────┘
```

## Recommended stack (cost first)

| Piece | Choice | ~Cost |
|---|---|---|
| Control plane | DigitalOcean droplet, 4 vCPU / 8 GB, region nearest home | ~$48/mo |
| GPU inference | RunPod Serverless (scale to zero, pay per second); same account + volume as today | ~$0 idle, ~$0.0004-0.0007/s busy |
| GPU training | RunPod on-demand pod on the existing `strands-xlerobot` volume, started by API, auto-stops | $0.59/hr A40 while running |
| Frontend previews | Cloudflare Pages (already set up, demo mode per PR) | $0 |
| DNS / TLS | Cloudflare DNS + Caddy (Let's Encrypt) on the droplet | $0 |
| Images | GHCR | $0 |

Why not AWS: more moving parts and cost for a single-operator dev box. The AWS IoT mesh transport in
`strands_robots/mesh/transport/iot_transport.py` stays an option if a managed broker is ever wanted.

## "API-centric, not P2P"

- Zenoh used in **client mode against one cloud router** is a broker (like MQTT), not P2P: every peer
  dials out to `tls/<host>:7447`, no multicast, no scouting, no inbound ports at home.
- mTLS is already in the code (`STRANDS_MESH_AUTH_MODE=mtls`, `_zenoh_config.tls_block`). Turning it on
  also fixes "no publisher session id" (commands become attributable), so agent motion starts working.
- Operator/agent/CI talk to the dashboard's HTTPS API only. GPU work is a plain HTTPS API.
- Replacing Zenoh outright with REST/WebSocket would mean rewriting `mesh/` (presence, state, cmd,
  teleop, cameras, safety). Not recommended unless the broker model proves painful.

## CI/CD

- PR: lint + unit tests (`test-lint.yml`), frontend build, Cloudflare Pages preview (exists).
- Merge to `main`: build `deploy/Dockerfile` → push `ghcr.io/jacob-brokloff/strands-xlerobot:<sha>`
  → SSH deploy to droplet (`docker compose pull && up -d`) → health check `/api/health` → auto
  rollback to previous tag on failure.
- Frontend-only changes: same pipeline (UI is baked into the image); target < 5 min to live.
- GPU image: separate workflow, builds only when `deploy/gpu/**` or `strands_robots/inference/**`
  changes, pushes to GHCR, updates the RunPod Serverless template via API.
- Secrets in GitHub Actions: `DEPLOY_SSH_KEY`, `RUNPOD_API_KEY`, `DASHBOARD_AUTH_TOKEN`.

## Code changes (expected)

1. `deploy/cloud/`: compose (caddy, zenohd, dashboard, agent), Caddyfile, `.env.example`.
2. mesh: TLS endpoint for `ZENOH_CONNECT`, cert issuing script (small CA, one cert per peer).
3. Dashboard: drop LAN assumptions (`lan_hint.py`, loopback-only passkey origin → `dash.<domain>`).
4. Inference: HTTP endpoint next to the WebSocket `PolicyServer` (serverless platforms are
   request/response); client sends JPEG-compressed frames (raw frames cost ~1 s/call today).
5. Policy runner on the Pi calls the HTTPS endpoint per action chunk (ACT / SmolVLA / pi0 all emit
   chunks, so ~1-3 calls/s is enough; 30 Hz control stays local on the Pi).
6. Training: `scripts/train_*` launched as a RunPod job from the dashboard/CI, checkpoints to HF Hub.
7. Sims on the CPU droplet render with OSMesa (slow); fall back to lower camera Hz or a GPU sim worker.
8. Retire: VM 102, Tailscale on all hosts, `runpod-vllm-tunnel`, `deploy/vm/`.

## Steps (each ends in something that runs)

1. Droplet + Caddy + dashboard from GHCR image, passkey login on `https://dash.<domain>`.
2. CI/CD to that droplet on merge to `main`.
3. zenohd with TLS; Pi and desktop peers connect over the internet with mTLS; teleop works via cloud.
4. GPU inference API on RunPod Serverless; one ACT rollout on the real robot.
5. Training job launcher; retire LAN VM and Tailscale.

## Risks

- Teleop round trip goes leader → cloud → Pi. Nearest region keeps it ~20-40 ms; measure in step 3.
- Serverless cold start (model load) 10-60 s: keep one warm worker during a session, 0 otherwise.
- Pi WiFi upload bandwidth for cameras; JPEG + lower resolution for the fleet view.

## Open decisions

- Home location / cloud region.
- Domain to use for `dash.<domain>`.
- Keep Zenoh as a cloud broker (recommended) vs rewrite the mesh as REST/WebSocket.
