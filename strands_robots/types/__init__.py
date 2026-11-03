"""The data that crosses every boundary: a JointState, an Action, a Pose, an Observation.

Everything here is a frozen dataclass with numpy inside, and a value: ``==`` compares the arrays,
``hash`` agrees with it. The same object comes back whether it was measured in MuJoCo, in Isaac
or off a servo bus. Types owned by one concept live with it (Physics/Contact in ``simulation``,
Requirements in ``policy``, Calibration in ``robot.drivers``, RunResult in ``robot.run``).
"""

from __future__ import annotations

from .joints import Action, JointState
from .observation import Observation
from .pose import Pose

__all__ = ["JointState", "Action", "Pose", "Observation"]
