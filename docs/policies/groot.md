---
description: NVIDIA GR00T (N1.5 / N1.6 / N1.7) - ZMQ service or local inference, 27 embodiment data_configs, full container lifecycle.
---

# GR00T

```bash
uv pip install "strands-robots[groot-service]"
```

```python
from strands_robots.policies import create_policy

# Service mode (container running separately)
policy = create_policy("groot", port=5555, data_config="so100_dualcam")

# Local mode (load model in-process)        # requires GPU
policy = create_policy("groot", model_path="/checkpoint", data_config="so100_dualcam", device="cuda")
```

## Parameters

```python
Gr00tPolicy(
    data_config="so100_dualcam",    # embodiment config (required)
    host="localhost",
    port=5555,
    model_path=None,                # set for local mode; None = service mode
    embodiment_tag="NEW_EMBODIMENT",
    device="cuda",                  # local mode only
    groot_version=None,             # override auto-detection (N1.5/N1.6/N1.7)
    strict=False,
    api_token=None,                 # fallback: GROOT_API_TOKEN env var
    observation_mapping=None,
    action_mapping=None,
    language_key=None,
    strict_keys=False,             # raise instead of positional key-guessing
)
```

## Strict key matching

When no explicit `observation_mapping`/`action_mapping` is given, a
**local-mode** policy auto-infers the robot<->model key mapping: exact name
matches first, then positional fallback for any leftover keys (with a log
line). On a multi-camera or multi-DOF rig, positional fallback can silently
bind the wrong camera or action column. Pass `strict_keys=True` to raise a
`ValueError` (listing the unmatched robot keys vs available model keys)
instead of guessing:

```python
policy = create_policy("groot", data_config="so100_dualcam",
                       model_path="nvidia/GR00T-N1.6-3B", strict_keys=True)
```

`strict_keys` defaults to `False` (positional fallback preserved) and is a
no-op when an explicit mapping is supplied.

Auto-inference is local-mode only, because it reads the checkpoint's modality
configs and service mode cannot introspect the remote server. A service-mode
policy given no mapping therefore infers nothing and sends the keys its
`data_config` declares, and `strict_keys` has nothing to be strict about
there. An *explicit* mapping needs no model metadata - the video/state split
comes from the `video.` / `state.` prefixes of the caller's own values - so it
is parsed and honoured in **either** mode. What service mode cannot do is
cross-check it: a mapping naming a key the server does not have surfaces as a
server-side error rather than a constructor refusal.

## Action chunk shape

Both unpack paths turn the model's / server's `{action.<key>: array}` chunk into
one dict per timestep, reading every value at the same index. Two properties are
therefore required of the chunk and refused with a `ValueError` naming the
offending keys and their shapes:

| Chunk | Result |
|-------|--------|
| every value `(horizon,)` or `(horizon, action_dim)`, same `horizon` | unpacked into `horizon` per-step dicts |
| a 0-D / scalar value (no time axis) | refused - `scalar (0-D) action value(s)` |
| values whose leading axes disagree (e.g. `(16, 7)` and `(8, 1)`) | refused - `time axes disagree` |

A disagreeing chunk is refused rather than truncated to the shortest value: the
steps a longer value carries are commands the model produced, so dropping them
would execute part of a trajectory and re-query as if the whole chunk had run.
The refusal is the same whichever key the producer serialized first.

## Versions

| Version | Transport | Notes |
|---------|-----------|-------|
| GR00T N1.5 | ZMQ | `(K, ...)` observation shape |
| GR00T N1.6 | ZMQ | `(K, ...)` observation shape |
| GR00T N1.7 | ZMQ | `(B, T, ...)` float32; auto-detected |

## 27 data_configs

```
so100               so100_dualcam          so100_4cam
so101               so101_dualcam          so101_tricam
bimanual_panda_gripper                     single_panda_gripper
libero_panda        oxe_droid              oxe_widowx
oxe_google          fourier_gr1_arms_only  fourier_gr1_arms_waist
fourier_gr1_full_upper_body
unitree_g1          unitree_g1_full_body   unitree_g1_locomanip
unitree_g1_real     unitree_g1_sonic
agibot_*            galaxea_r1_pro
```

Every name above is also a robot identifier: the registry declares each
embodiment's `data_config` spellings as aliases of the robot they name, so
`unitree_g1_sonic` resolves the same Unitree G1 that `unitree_g1` does.

```python
Robot("unitree_g1_sonic", mode="sim")          # same robot as Robot("unitree_g1")
sim.add_robot(name="g1", data_config="unitree_g1_sonic")
```

That is what lets `add_robot(data_config=...)` find the model to load, lets the
Isaac IK solve resolve an MJCF for the robot, and lets `move_to` read the
registry `gripper` block instead of guessing the gripper heuristically.

## Container lifecycle

```python
from strands_robots.tools import gr00t_inference

# The image name is operator config, not an agent parameter: set
# STRANDS_GR00T_IMAGE (default "gr00t:latest") and it must pass the
# STRANDS_GR00T_IMAGE_ALLOW allowlist.
gr00t_inference(action="build_image")
gr00t_inference(action="download_checkpoint", hf_repo="nvidia/GR00T-N1.7-3B")
gr00t_inference(action="start_container",     data_config="so100_dualcam")
# ... run policy ...
gr00t_inference(action="stop", container_name="gr00t-inference")  # stop only
gr00t_inference(action="lifecycle", lifecycle="teardown",
                container_name="gr00t-inference",
                remove_volumes=True)  # stop + remove container (and volumes)
```

`action="stop"` escalates SIGTERM then SIGKILL over every process serving the
port, inside the GR00T container first and on the host as a fallback. A process
that had already exited is not a failure, but a port still held after both
signals is: the result is then `{"status": "error", ...}` naming the port and the
surviving pid, because reporting success there would send the next `start` into a
bind that cannot succeed. Check the status before rebinding the same port.

## See also

- [Policy providers](../policies/overview.md)
- [Real hardware](../hardware/robot-control.md)
- [LeRobot Local](lerobot-local.md)
- [Cosmos 3](cosmos3.md)
- [cuRobo](curobo.md)
- [Isaac-GR00T project](https://github.com/NVIDIA/Isaac-GR00T)
