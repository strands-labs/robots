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

## Container lifecycle

```python
from strands_robots.tools import gr00t_inference

gr00t_inference(action="build_image",         tag="gr00t-n1.7:latest")
gr00t_inference(action="download_checkpoint", model_id="nvidia/GR00T-N1.7-3B")
gr00t_inference(action="start_container",     tag="gr00t-n1.7:latest",
                                              model_id="nvidia/GR00T-N1.7-3B",
                                              data_config="so100_dualcam")
# ... run policy ...
gr00t_inference(action="stop", container_name="gr00t-inference")  # stop only
gr00t_inference(action="lifecycle", lifecycle="teardown",
                container_name="gr00t-inference",
                remove_volumes=True)  # stop + remove container (and volumes)
```

## See also

- [Policy providers](../policies/overview.md)
- [Real hardware](../hardware/robot-control.md)
- [LeRobot Local](lerobot-local.md)
- [Cosmos 3](cosmos3.md)
- [cuRobo](curobo.md)
- [Isaac-GR00T project](https://github.com/NVIDIA/Isaac-GR00T)
