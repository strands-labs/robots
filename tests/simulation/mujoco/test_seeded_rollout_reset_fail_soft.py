"""Regression test: a seeded rollout tolerates a policy whose reset fails.

When ``run_policy`` is given a ``seed`` it reseeds the client RNGs and forwards
the seed to the policy via ``policy.reset(seed=...)`` for reproducibility. That
policy-side reseed is best-effort: a policy that cannot reseed (e.g. a remote
inference client that raises, or a policy with no seedable state) must NOT abort
the rollout. The runner catches the failure, logs a warning, and continues with
only the client-side reseed applied.

This pins that fail-soft contract so a raising ``reset`` can never regress into
a hard rollout failure.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation import Simulation  # noqa: E402

_ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""


class _ResetRaisingPolicy(MockPolicy):
    """A MockPolicy whose seeded reset always raises (unseedable client)."""

    def reset(self, seed: int | None = None) -> None:
        raise RuntimeError("policy cannot reseed")


@pytest.fixture
def sim(tmp_path):
    path = tmp_path / "arm.xml"
    path.write_text(_ROBOT_XML)
    s = Simulation()
    s.create_world()
    s.add_robot("arm", urdf_path=str(path))
    yield s
    s.destroy()


def test_seeded_rollout_survives_reset_failure(sim, caplog):
    """A seeded rollout completes even when ``policy.reset(seed=)`` raises."""
    import logging

    with caplog.at_level(logging.WARNING):
        result = sim.run_policy(
            robot_name="arm",
            policy_object=_ResetRaisingPolicy(),
            seed=123,
            n_steps=5,
        )

    # The rollout ran to completion despite the reset failure...
    assert result["status"] == "success", result
    # ...and the best-effort failure was surfaced as a warning, not swallowed
    # silently (no warn-and-continue without a log).
    assert any("reset" in rec.message and "123" in rec.message for rec in caplog.records)
