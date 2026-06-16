"""WBC policy - GR00T Whole-Body-Control (SONIC) locomotion for the Unitree G1.

The :class:`WBCPolicy` wraps NVIDIA's
`GR00T-WholeBodyControl <https://github.com/NVlabs/GR00T-WholeBodyControl>`_
ONNX locomotion controllers. Like :class:`~strands_robots.policies.curobo.CuroboPolicy`
it runs **in process** (ONNX Runtime, no sidecar), and like the rest of the
non-VLA family:

* ``requires_images = False`` - locomotion controls from joint state + base IMU,
  never camera frames.
* ``get_actions`` reads the goal from the well-known locomotion ``**kwargs``
  (``target_velocity = [vx, vy, omega]``, optional ``target_orientation``).
* The controller drives the **15 leg+waist DOFs** of the Unitree G1; the arm
  joints are held at their nominal defaults. Layering an upper-body policy on
  top is the job of a future ``CompositePolicy`` (#468), out of scope here.

Requires the ``[wbc]`` extra (``onnxruntime``); no model weights are bundled
(fetched at runtime under the NVIDIA Open Model License). See issue #466.
"""

from strands_robots.policies.wbc.config import WBCConfig
from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS, WBC_G1_LEG_WAIST_JOINTS, WBCPolicy

__all__ = [
    "WBCPolicy",
    "WBCConfig",
    "WBC_G1_LEG_WAIST_JOINTS",
    "WBC_G1_ALL_JOINTS",
]
