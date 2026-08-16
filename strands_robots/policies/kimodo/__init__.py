"""Kimodo policy - NVIDIA text-to-motion diffusion for the Unitree G1.

:class:`KimodoPolicy` wraps NVIDIA's Kimodo generative motion model
(``nvidia/Kimodo-G1-RP-v1`` on HuggingFace). Kimodo is a *text-conditioned
kinematic motion generator*: given a natural-language prompt (e.g.
``"a person walking forward with confident strides"``) it synthesises per-frame
full-body ``qpos`` sequences for the Unitree G1 via a diffusion sampler.

Where it sits (the same seat as :class:`~strands_robots.policies.motionbricks.MotionBricksPolicy`):

* ``requires_images = False`` - text prompt drives synthesis, never cameras.
* ``get_actions`` reads the goal from the well-known ``**kwargs`` keys
  (``text_prompt`` / ``instruction``, ``diffusion_steps``, ``guidance_scale``).
* The output is the G1's 29 leg+waist+arm joint targets, keyed by the
  canonical WBC joint ordering, so a downstream tracker names the same joints
  without a remapping table. Tracking is a cascade (the 29 targets are the
  tracker's input), NOT a
  :class:`~strands_robots.policies.composite.CompositePolicy` layer - that
  class merges disjoint joint groups. Standalone in sim the targets are applied
  directly, which is the faithful kinematic reference.

Kimodo differs from MotionBricks in ONE dimension: Kimodo is **prompt-driven
generative** (any English motion description, one-shot diffusion sample),
MotionBricks is **style-driven** (a fixed vocabulary of walk/stealth/boxing
clip modes plus a heading command). They both emit the same ``qpos`` contract
so downstream trackers see the same signal.

Requires the ``[kimodo]`` extra:

* ``torch>=2.0.0``, ``diffusers>=0.30.0``, ``transformers>=4.40.0``,
  ``huggingface_hub``, ``accelerate``, ``scipy``.

Model weights are fetched on demand from HuggingFace; no checkpoints bundled.
See :doc:`docs/policies/kimodo`.
"""

from strands_robots.policies.kimodo.config import KimodoConfig

# Hardware action-key helpers. Importing them does not import lerobot: the
# joint rename table is built on the first get_joint_map() call, so a pure-sim
# import path never looks for the driver.
from strands_robots.policies.kimodo.hardware import (
    build_lerobot_g1_action_dict,
    get_joint_map,
    kimodo_action_to_lerobot_g1,
)
from strands_robots.policies.kimodo.policy import (
    KIMODO_G1_JOINTS,
    KimodoMotionAgent,
    KimodoPolicy,
)

__all__ = [
    "KimodoPolicy",
    "KimodoConfig",
    "KimodoMotionAgent",
    "KIMODO_G1_JOINTS",
    "build_lerobot_g1_action_dict",
    "get_joint_map",
    "kimodo_action_to_lerobot_g1",
]
