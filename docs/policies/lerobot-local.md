---
description: HuggingFace LeRobot direct inference — ACT, Pi0, SmolVLA, Diffusion Policy, MolmoAct2. RTC + processor bridge.
---

# LeRobot Local

```bash
uv pip install "strands-robots[lerobot]"
export STRANDS_TRUST_REMOTE_CODE=1        # required; raises UntrustedRemoteCodeError otherwise
```

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",   # HF model_id or local path
    device="cuda",
)
```

## Parameters

```python
LerobotLocalPolicy(
    pretrained_name_or_path="",          # HF model_id or local checkpoint dir (required)
    policy_type=None,                    # override auto-detected class
    device=None,                         # "cuda" | "cpu" | "mps"
    actions_per_step=1,
    use_processor=True,                  # observation/action processor bridge
    processor_overrides=None,
    tokenizer_max_length=48,
    tokenizer_padding_side="right",
    rtc_enabled=None,                    # Real-Time Chunk smoothing (NOT rtc=)
    rtc_execution_horizon=None,
    rtc_max_guidance_weight=None,
    inference_kwargs=None,
    embodiment=None,
    norm_tag=None,                       # MolmoAct2 normalisation tag
    image_keys=None,                     # MolmoAct2 camera key override
    inference_action_mode="continuous",  # "continuous" | "discrete"
)
```

## Supported models

| Model | Notes |
|-------|-------|
| ACT | Action Chunking Transformer |
| Pi0 / Pi0.5 | Physical Intelligence VLA |
| SmolVLA | HuggingFace small VLA |
| Diffusion Policy | flow-matching |
| VQ-BeT | discrete action tokenisation |
| MolmoAct2 | transformers-native SO100/SO101; use `norm_tag` + `image_keys` |

## MolmoAct2

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="your-org/molmoact2-so101",
    device="cuda",
    norm_tag="so101",
    image_keys=["wrist_camera", "front_camera"],
    inference_action_mode="continuous",
)
# see examples/molmoact2_so101_pickplace.py
```

## RTC

```python
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/pi0_so100",
                        rtc_enabled=True, rtc_execution_horizon=16, rtc_max_guidance_weight=1.0)
```

## See also

- [Policy providers](../policies/overview.md)
- [Training](../training/overview.md)
- [GR00T](groot.md)
- [LeRobot project](https://github.com/huggingface/lerobot)
