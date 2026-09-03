# Fleet dashboard

`strands-robots dashboard` starts a fleet cockpit: a FastAPI server plus a
mobile-first PWA that shows every robot on the mesh - real arms, simulated
twins, their cameras, joints, policies and training jobs - and lets an agent
drive them in natural language.

It is **a mesh peer, not a hub**. It publishes presence like any other peer and
subscribes to the rest, so a robot started from your own Python script appears
in it unchanged (see [Defining robots in code](#defining-robots-in-code)).

```bash
pip install "strands-robots[all]"          # mesh + lerobot + sim extras
python -m strands_robots dashboard --port 8090 --local-dev
# open http://localhost:8090
```

New here with two SO-101 arms and a Mac? Start with the
[10-minute quickstart](quickstart.md) - install, calibrate, spawn both arms,
identify the cameras, first dataset on disk. From there:
[Collect, train, deploy](collect-train-deploy.md) closes the data loop,
[Remote access](remote-access.md) reaches it from a phone, and
[When it does not work](troubleshooting.md) is the error-string index.

## Command surface

Every flag below is from `python -m strands_robots dashboard --help` on this
build - not aspirational:

| flag | meaning |
|---|---|
| `--host` / `--port` | bind address (default `0.0.0.0:8090`) |
| `--peer-id` | mesh peer id for the dashboard itself (auto if omitted) |
| `--local-dev` | sets `STRANDS_MESH_LOCAL_DEV=1` - no TLS, one machine |
| `--force` | start even if the port is already bound (see below) |
| `--auth-token TOKEN` | shared-token guard for `/api/*` and `/ws/*` |
| `--cors-origin ORIGIN` | extra allowed browser origin |
| `--camera-hz` | camera publish rate the dashboard asks peers for |
| `--zenoh-connect` / `--zenoh-listen` / `--mesh-port` | join a mesh running elsewhere (`tls/robot.lan:7447`) |
| `--mesh-backend {zenoh,iot,bridge}` | LAN/VPN, AWS IoT Core, or bridge |
| `--log-level` | uvicorn/py logging level |

Without `--force`, a second dashboard on a taken port is **refused with the pid
of the process that owns it**. That refusal is deliberate: the mesh session is
opened before the socket bind, so a duplicate launch would already have joined
the mesh as a second hub by the time the bind failed.

## HTTP API

The UI is a client of this API, so anything the UI does you can do with `curl`.
Verified live against a running dashboard:

| area | endpoints |
|---|---|
| health / fleet | `GET /api/health`, `GET /api/fleet`, `GET /api/activity` |
| devices | `GET /api/devices`, `POST /api/devices/spawn`, `POST /api/devices/despawn`, `GET /api/devices/profiles`, `GET /api/devices/logs/{peer_id}` |
| cameras | `GET /api/frame/{peer_id}/{cam}` (single JPEG), camera frames also stream over `/ws` |
| control | `POST /api/robots/{peer_id}/task`, `POST /api/robots/{peer_id}/stop`, `POST /api/robots/{peer_id}/twin` |
| teleop | `GET /api/robots/{peer_id}/teleop`, `POST .../teleop/receive`, `POST .../teleop/stop` |
| safety | `POST /api/safety/estop`, `POST /api/safety/resume` |
| policies | `GET /api/policies`, `POST /api/policies/validate`, `GET /api/checkpoints/search`, `GET /api/checkpoints/families`, `GET /api/checkpoints/features?repo_id=…` (what a checkpoint declares it was trained on — the raw rail behind `GET /api/robots/{peer_id}/policy-fit`, read-only, local cache only, `{}` when unknown; useful for asking by hand why a fit verdict came out the way it did) |
| data | `POST /api/collect`, `POST /api/replay`, `GET /api/training/datasets` |
| training | `GET /api/training/{trainers,jobs,status}`, `POST /api/training/{validate,submit,export}` |
| calibration | `GET /api/calibration`, `GET /api/calibration/{name}` |
| registry | `GET /api/robots/registry` (every robot the factory knows) |
| agent | `GET /api/agent/status`, `POST /api/agent/reset`, chat over `/ws/chat`, voice over `/ws/voice` |
| config / mesh | `GET,POST /api/config`, `GET,POST /api/mesh/config`, `POST /api/mesh/restart` |
| auth | `GET /api/auth/status`, `POST /api/auth/{register,login}/{begin,finish}`, `GET/DELETE /api/auth/credentials...` |

`GET /openapi.json` is served too, so `curl -s localhost:8090/openapi.json` is
always the current truth.

## Defining robots in code

A robot you start yourself is a first-class citizen of the same fleet:

```python
from strands_robots import Robot

robot = Robot(
    "so101",
    mode="real",
    port="/dev/tty.usbmodem5AB0181806",   # macOS: /dev/cu.* also works
    cameras={"top": 1, "wrist": 0},
    id="follower_arm",                     # lerobot calibration identity
    mesh=True,
    peer_id="so101-arm-1",
)
```

With `mesh=True` (or `STRANDS_MESH=true`) this process publishes presence,
joint state and camera frames on the mesh, and the dashboard renders it beside
its own managed spawns. Dashboard-spawned and code-defined peers differ only in
an origin badge. The camera argument is a mapping of options per camera, not a
bare index - full shape and the deployment story in
[Robots defined in code](code-defined-robots.md).

## Security posture

- **Local by default.** With no auth configured, the guard only serves clients
  it can prove are loopback. `GET /api/auth/status` reports
  `{"enabled": false, "setup_required": true, ...}` in that state.
- **Passkeys for remote access.** Enrol a WebAuthn credential (a phone, a
  YubiKey, Touch ID) before exposing the dashboard beyond localhost.
- The mesh has its own, separate security story - mTLS and ACLs are on by
  default and `--local-dev` turns them off for single-machine work. See
  [Multi-robot Mesh](../mesh.md).

## Known rough edges

Measured on this build, so you do not lose an evening to them:

- `strands-robots doctor` needs `sysctl` on `PATH`. In a stripped shell
  (`PATH=/usr/bin:/bin`) its MuJoCo and sim checks fail with
  `[Errno 2] No such file or directory: 'sysctl'`; add `/usr/sbin` and they
  pass.
- On macOS, doctor's `MUJOCO_GL not set and no display detected` FAIL is
  advisory - `export MUJOCO_GL=egl` is Linux advice, and the sim smoke test
  passes without it (`Robot('so100') works`).
- Camera indices are **not stable identities**. `GET /api/devices` only probes
  indices nothing has claimed, so an index used by a running peer (or by
  Photo Booth) simply will not be listed, and adding an iPhone Continuity
  camera renumbers the rest. Identify a camera by looking at a frame before you
  map it to an arm; never trust the number alone
  ([how](quickstart.md#6-identify-the-cameras-1-min-and-the-one-that-saves-an-evening)).
