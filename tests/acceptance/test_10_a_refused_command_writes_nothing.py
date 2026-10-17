"""Check 10: refusals come before the bus. A real body (a feetech driver over a recorded serial line)
refuses an agent's command without the operator, when the operator says no, and for a joint it does not
have; every refusal names its cause, and no frame that could move a servo reaches the line.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_drivers import make_driver
from test_operator_gate import goal_writes


@pytest.mark.parametrize(
    "confirm, action, error, match",
    [
        (None, None, PermissionError, "pass confirm="),
        (lambda targets: False, None, PermissionError, "refused"),
        (lambda targets: True, {"elbow": 0.1}, KeyError, r"unknown joint\(s\) \['elbow'\]"),
        (lambda targets: True, [0.0, 0.0], ValueError, "action has 2 values"),
    ],
    ids=["no-operator", "operator-says-no", "unknown-joint", "wrong-length"],
)
def test_a_refused_command_writes_nothing(confirm, action, error, match):
    from strands_robots import Robot

    d, serial = make_driver()
    real = Robot("so101", driver=d, confirm=confirm, twin=False)
    before = real.observe(images=False).joints.positions
    serial.frames.clear()
    with pytest.raises(error, match=match):
        real.confirm_act(before if action is None else action)
    assert goal_writes(serial) == []
    np.testing.assert_array_equal(real.observe(images=False).joints.positions, before)
    real.close()
