# Collect, train, deploy

The loop the dashboard exists for: get episodes onto disk, turn them into a
checkpoint, run that checkpoint on the arm. Every step is an HTTP call, so the
UI and `curl` are interchangeable - and every step below was checked against a
running dashboard rather than transcribed from a design doc.

## 1. Get episodes onto disk

There are two working paths today, and they are good at different things.

### Policy-driven collection (`POST /api/collect`)

One call spawns a **one-shot sim peer**, drives exactly `n_episodes` rollouts,
writes a LeRobotDataset with per-episode parquet boundaries, and exits. It
appears in the fleet grid with live cameras while it runs.

```bash
curl -sX POST localhost:8090/api/collect -H 'content-type: application/json' -d '{
  "dataset_root": "'"$HOME"'/datasets/so101-cubes",
  "dataset_repo_id": "you/so101-cubes",
  "robot_name": "so101",
  "policy_provider": "mock",
  "instruction": "pick up the red cube",
  "n_episodes": 5, "duration": 10.0, "fps": 30
}'
```

`dataset_root` is required (422 without it) and is **remembered**, so the
dataset shows up in `/api/training/datasets` even when it lives outside the
default scan paths like `HF_LEROBOT_HOME`. Counts come back from parquet truth,
not from what the loop intended to write.

Use this to shake out the whole pipeline before your arms are calibrated:
`policy_provider: "mock"` needs no GPU, no checkpoint and no hardware.

### Teleop recording with the leader arm

For real demonstrations - the leader arm drives the follower, and the
**follower** is what gets recorded - the working rail today is the agent's own
recording actions, driven through chat or `POST /api/robots/{peer}/task`, which
is what the README's first example does. Teleop itself is wired end to end:

```bash
# point the follower (real, or a sim twin) at the leader's input stream
curl -sX POST localhost:8090/api/robots/so101-arm-1/teleop/receive \
  -H 'content-type: application/json' \
  -d '{"source_peer_id": "so101-arm-2", "device_name": "leader"}'

curl -s localhost:8090/api/robots/so101-arm-1/teleop   # rates, drops, slew rejections
curl -sX POST localhost:8090/api/robots/so101-arm-1/teleop/stop
```

The first `teleop_receive` can take over 15 seconds (zenoh subscriber declaration
plus gossip); the route budgets 45s. That is slow, not stuck.

Because a sim twin can follow a **real** leader arm, you can rehearse a
recording session with no metal moving at all.

## 2. See what you actually recorded

```bash
curl -s localhost:8090/api/training/datasets | python -m json.tool
```

Each row carries `root`, `repo_id`, `total_episodes`, `total_frames`, `fps` and
`robot_type` - enough to catch the two classic mistakes (a dataset that recorded
30 frames because the episode ended early, or one whose `robot_type` says
`unknown` because it was recorded against a robot the registry never saw).

Replay an episode as real physics before you spend GPU hours on it:

```bash
curl -sX POST localhost:8090/api/replay -H 'content-type: application/json' \
  -d '{"repo_id": "you/so101-cubes", "episode": 0, "speed": 1.0, "robot_name": "so101"}'
```

The replay peer joins the fleet with live cameras, drives MuJoCo with the
recorded actions, and exits when the episode ends. If the arm in the twin does
something your arm never did, the dataset is the problem, not the policy.

## 3. Train

```bash
curl -s localhost:8090/api/training/trainers
# {"trainers":["cosmos3","fast_sac","groot","lerobot_local","mock","ppo"]}
```

**Validate before you submit** - same body, different route, no job created:

```bash
curl -sX POST localhost:8090/api/training/validate -H 'content-type: application/json' \
  -d '{"provider":"lerobot_local","dataset_root":"'"$HOME"'/datasets/so101-cubes", ...}'
curl -sX POST localhost:8090/api/training/submit   -H 'content-type: application/json' -d '{...}'
curl -s "localhost:8090/api/training/jobs"
curl -s "localhost:8090/api/training/status?provider=lerobot_local&job_id=<id>"
```

Then turn the run into something loadable:

```bash
curl -sX POST localhost:8090/api/training/export -H 'content-type: application/json' -d '{
  "provider": "lerobot_local",
  "output_dir": "'"$HOME"'/checkpoints/so101-cubes",
  "dataset_root": "'"$HOME"'/datasets/so101-cubes"
}'
```

## 4. Deploy the checkpoint

Find one (local cache first, marked `local`, then public LeRobot checkpoints
ranked by downloads), then dry-run the provider config **without touching a
robot**:

```bash
curl -s "localhost:8090/api/checkpoints/search?q=smolvla&limit=10"
curl -s localhost:8090/api/checkpoints/families        # policy_type values lerobot accepts
curl -sX POST localhost:8090/api/policies/validate -H 'content-type: application/json' \
  -d '{"policy_provider":"lerobot_local","policy_config":{"pretrained_name_or_path":"you/so101-cubes"}}'
```

Then run it:

```bash
curl -sX POST localhost:8090/api/robots/so101-arm-1/task -H 'content-type: application/json' -d '{
  "instruction": "pick up the red cube",
  "policy_provider": "lerobot_local",
  "pretrained_name_or_path": "you/so101-cubes",
  "duration": 30
}'
```

Three behaviours of this route that will otherwise confuse you:

- **Only allowlisted keys reach the peer.** `policy_host`, `policy_port`,
  `policy_type`, `server_address`, `model_path`, `pretrained_name_or_path`,
  `robot_name`, `target_pose`, `target_joints`, `world_update`,
  `control_frequency`, `action_horizon`, `fast_mode`, `n_steps`. Anything else -
  a nested `policy_config`, for instance - is **silently dropped** by the wire
  validator, so the command looks accepted and arrives empty.
- **The reply waits for the policy to finish.** The timeout is forced to at
  least `duration + 10s`, because a shorter one reports "timeout" on a task
  that is running perfectly.
- **A live twin gets the same instruction**, fire-and-forget, when
  `<peer>-twin` is on the mesh. The response says `mirrored_to_twin`.

The response carries one honest boolean: `ok` is computed from the peer's reply,
because a command can come back successfully and still have failed.

## 5. Safety refusals are part of the loop

Two guards will stop a first deployment, on purpose:

- `STRANDS_TRUST_REMOTE_CODE=1` - loading a HuggingFace policy executes code
  from that repo. The refusal names the variable; set it only for orgs you
  trust.
- `STRANDS_MESH_HF_REPO_ALLOW=org/,other-org/repo` - the checkpoint you asked
  for must be in the allowlist. `pretrained_name_or_path=...' not in allowlist`
  is this guard, not a missing file.

Stop everything with `POST /api/safety/estop`; `POST /api/safety/resume` needs
the `STRANDS_MESH_OVERRIDE_CODE` shared by every peer.

## Where to go next

- [Quickstart](quickstart.md) - if you do not yet have two arms spawned
- [Fleet dashboard reference](index.md) - every route and panel
- [Remote access](remote-access.md) - drive this loop from a phone
