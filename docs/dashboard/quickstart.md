# Quickstart: two SO-101 arms and a Mac, in 10 minutes

Zero to a fleet you can see, with a first dataset on disk. Everything below was
run on the machine this page was written on; where the software has a sharp
edge, the page says so instead of pretending.

You need: two SO-101 arms (a leader and a follower), their USB cables, one or
two USB cameras, Python 3.10+.

## 1. Install (2 min)

```bash
pip install "strands-robots[dashboard,lerobot,mesh,sim-mujoco]"
python -m strands_robots doctor          # sanity check
```

`dashboard` brings the FastAPI + WebAuthn server this page drives, `lerobot`
the hardware + dataset stack, `mesh` the peer-to-peer transport the dashboard
speaks, `sim-mujoco` the twin you can rehearse on.

If `doctor` fails on MuJoCo with `No such file or directory: 'sysctl'`, your
`PATH` is missing `/usr/sbin` - that is the whole bug. On macOS its
`MUJOCO_GL not set and no display detected` line is Linux advice you can
ignore: the sim smoke test on the next line is the real verdict.

## 2. Find the arms (1 min)

```bash
python -c "import serial.tools.list_ports as p; [print(x.device, x.serial_number, x.description) for x in p.comports()]"
```

Both SO-101 boards show up as CH340-family USB serial (`vid 1a86`), e.g.
`/dev/cu.usbmodem5AB0181806`. The board serial number is worth writing down -
the dashboard keys its remembered device profiles on it, not on the port path,
so an arm keeps its identity across reboots and re-plugs.

## 3. Calibrate each arm once (3 min)

Skip this and your joint limits are wrong - the arm will fight its own range.
Calibration is per physical arm and is remembered by the `id` you give it:

```bash
lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5AB0181806 --robot.id=follower_arm

lerobot-calibrate --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5AB01584281 --teleop.id=leader_arm
```

Follow the prompts on screen (move each joint to its limits). Files land under
`~/.cache/huggingface/lerobot/calibration/{robots,teleoperators}/so101_*/<id>.json`,
and `curl -s localhost:8090/api/calibration` lists every one the machine knows
once the dashboard is up.

> Which arm is which? The follower runs 12 V servos, the leader 7.4 V. If the
> two arms are labelled backwards in the UI, that is the mapping to double
> check - the calibration `id` is what binds a port to a role.

## 4. Start the dashboard (30 s)

```bash
python -m strands_robots dashboard --port 8090 --local-dev
# -> http://localhost:8090
curl -s localhost:8090/api/health
# {"status":"ok","mesh_online":true,"dashboard_peer_id":"dashboard-...","peers":5,...}
```

`--local-dev` turns off mesh TLS for single-machine work. Without auth
configured the API only answers clients it can prove are loopback, so nothing
is exposed yet.

A second dashboard on a taken port is refused **with the pid that owns it** -
pass `--force` only if you mean it.

If the browser shows a passkey gate instead of the fleet, a credential is
configured on this server: the UI probes `/api/fleet` first and only shows the
door when it gets a 401/403. That is [remote access](remote-access.md) working,
not a broken build.

## 5. Add the arms to the fleet (2 min)

**From the UI:** *Devices* panel -> **Spawn**: pick the robot (`so101`), Mode
`real hardware`, the **Servo bus** port, the **Calibration id** you used above,
and a camera index. Ports and cameras already claimed by a running peer are
shown greyed out, so you cannot double-open one.

The form attaches a single camera, named `main`. A two-camera rig (top + wrist)
goes through the API or your own script - and note that each camera is a
**mapping of options**, never a bare index:

```bash
curl -sX POST localhost:8090/api/devices/spawn -H 'content-type: application/json' -d '{
  "robot_name": "so101", "mode": "real",
  "peer_id": "so101-arm-1",
  "port": "/dev/cu.usbmodem5AB0181806",
  "robot_id": "follower_arm",
  "cameras": {
    "top":   {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30},
    "wrist": {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30}
  }
}'
```

`{"top": 2}` is refused with `Camera 'top' config must be a mapping of option
name to value, got int: 2` - see
[Robots defined in code](code-defined-robots.md#the-cameras-argument-is-a-mapping-of-mappings).

Each spawn is a **child process that joins the mesh as its own peer** - it is
not a thread inside the dashboard. `GET /api/devices/logs/{peer_id}` is its
stdout, and that is where a hardware problem confesses.

## 6. Identify the cameras (1 min, and the one that saves an evening)

A camera index is a **position, not an identity**:

- `GET /api/devices` only probes indices nothing has claimed. A camera held by
  a running peer (or by Photo Booth) is simply absent from the list.
- Plugging in an iPhone Continuity camera renumbers the rest.
- The order the OS enumerates cameras in is not the order other tools
  (`ffmpeg -f avfoundation -list_devices`) print.

So do not reason about which index is the wrist - **look**:

```bash
curl -s localhost:8090/api/frame/so101-arm-1/wrist -o /tmp/wrist.jpg && open /tmp/wrist.jpg
```

`/api/frame/{peer_id}/{cam}` returns one JPEG of whatever that name is pointing
at. If `wrist` shows the room and `top` shows the gripper, despawn
(`POST /api/devices/despawn {"peer_id": ...}`) and respawn with the two indices
swapped. Once it is right, the mapping is remembered per board serial in
`~/.strands_dashboard/profiles.json` and replayed when you plug that arm back
in - `GET /api/devices/profiles` shows what is stored.

## 7. Your first dataset (1 min)

The fastest honest first dataset is a **scripted rollout in the sim twin** - no
hands needed, and it proves the whole record -> discover chain:

```bash
curl -sX POST localhost:8090/api/collect -H 'content-type: application/json' -d '{
  "dataset_root": "/tmp/so101-demo", "dataset_repo_id": "local/so101-demo",
  "robot_name": "so101", "policy_provider": "mock",
  "instruction": "wave the arm", "n_episodes": 2, "duration": 5, "fps": 30
}'

curl -s localhost:8090/api/training/datasets      # the dataset now appears here
strands-robots verify-dataset /tmp/so101-demo --expected 2
```

`/api/collect` spawns a one-shot sim peer (you will see it appear in the grid),
drives the policy for `n_episodes`, verifies episode boundaries against the
parquet files, and exits. Counts come from the parquet, not from a counter that
hopes.

**Human teleop demos on the real arms are a different rail.** Teleop itself is
live over the API - point the follower at the leader's stream with
`POST /api/robots/<follower>/teleop/receive` and watch the counters on
`GET /api/robots/<follower>/teleop`. Recording those demos from the UI is also
live: the Record panel probes `/api/record/session` at load and drives the
session state machine on the server (target episodes, per-episode discard,
thumbnails, measured fps). The mock fallback only applies if the probe 404s,
which is a server-side wiring problem rather than an expected state. Details:
[Collect, train, deploy](collect-train-deploy.md#1-get-episodes-onto-disk).

## Where things go wrong

| symptom | cause |
|---|---|
| `Port is in use!` in a peer's log | usually nothing to do: a read that died mid-exchange leaves the bus flagged, and the next read clears it by itself. If it persists, two processes are on one servo bus (the arm cannot be shared) - `/usr/sbin/lsof /dev/cu.usbmodem*` names every holder; despawn the other peer. One holder that keeps stranding is a cable or hub, not software - see the card's `bus healed ×N` count |
| a camera tile shows nothing | another app owns that index, or the index moved; check `GET /api/devices` and re-map |
| joints read but the arm fights its range | calibration `id` at spawn does not match the one you calibrated |
| dashboard refuses to start | a dashboard already owns the port; its pid is in the message |
| `{"detail":"unauthorized"}` from a browser that is not on this Mac | working as designed - see [Remote access](remote-access.md) |
| anything else | [Troubleshooting](troubleshooting.md) quotes the real messages |
| `doctor` fails on MuJoCo/sysctl | `/usr/sbin` missing from `PATH` |

## Next

- [Fleet dashboard overview](index.md) - every CLI flag and HTTP route
- [Robots defined in code](code-defined-robots.md) - the same peer from your
  own script, and how to deploy it to an edge device
- [Collect, train, deploy](collect-train-deploy.md) - episodes to a checkpoint
  running on the arm, every step as an HTTP call
- [Remote access](remote-access.md) - reach the fleet from a phone, guard
  first and tunnel second
- [When it does not work](troubleshooting.md) - every refusal and error string,
  with what it is protecting
- [Multi-robot Mesh](../mesh.md) - what the peers are actually saying
- [Teleoperation](../hardware/teleoperation.md) - leader/follower and the
  teleoperator matrix
