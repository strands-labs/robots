---
description: NVIDIA GR00T (N1.5 / N1.6 / N1.7) — ZMQ service or local inference, 27 embodiment data_configs, full container lifecycle.
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
)
```

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
- [Isaac-GR00T project](https://github.com/NVIDIA/Isaac-GR00T)
