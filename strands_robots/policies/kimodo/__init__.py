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
  canonical WBC joint ordering, so it composes with
  :class:`~strands_robots.policies.wbc.WBCPolicy` (Kimodo emits motion
  targets, WBC tracks them) via
  :class:`~strands_robots.policies.composite.CompositePolicy` and (in the sim)
  a PD tracker at 1kHz physics / 50Hz control.

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
]
