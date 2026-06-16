"""Smoke test for the torque-control deploy harness (examples/wbc_g1_torque_deploy.py).

Runs WITHOUT real SONIC weights: it drives the harness's ``simulate_rollout``
loop with a STUB WBCPolicy session (real ``compute_torques`` / config / joint
mapping, fake ONNX) on the real torque-actuated G1 model. This guards the
harness MECHANICS that bit us during development:

* the bare robot_descriptions G1 MJCF has no ground plane -> the robot fell
  through space (z -> -inf); _build_torque_g1 must add a floor.
* the observation fed to the policy must be the whole-body per-joint dict with
  velocities + base IMU, in the model's joint order.
* arms are held, only the 15 leg+waist joints are driven.

The real-gait validation (does it actually WALK) lives in the gated integration
test ``tests_integ/policies/wbc/test_wbc_live.py`` (needs real weights).

Requires mujoco; skips cleanly otherwise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed - pip install 'strands-robots[sim-mujoco]'")

from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBCConfig, WBCPolicy  # noqa: E402

# Import the example module by path (examples/ is not an installed package).
_HARNESS_PATH = Path(__file__).resolve().parents[3] / "examples" / "wbc_g1_torque_deploy.py"


def _load_harness():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("wbc_g1_torque_deploy", _HARNESS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("wbc_g1_torque_deploy", mod)
    spec.loader.exec_module(mod)
    return mod


class _StubSession:
    """ONNX stand-in: returns a tiny constant 15-dim raw action."""

    class _In:
        name = "obs"

    def __init__(self, fill: float = 0.0) -> None:
        self.fill = fill

    def get_inputs(self):  # type: ignore[no-untyped-def]
        return [self._In()]

    def run(self, output_names, feed):  # type: ignore[no-untyped-def]
        return [np.full((1, 15), self.fill, dtype=np.float32)]


def _stub_policy(fill: float = 0.0) -> WBCPolicy:
    """A WBCPolicy with stub ONNX sessions but the real config + PD law.

    ``fill=0`` makes the raw action zero, so target_q == default_angles every
    tick: the policy commands a static hold at the nominal stance via the real
    compute_torques PD law - the cleanest mechanics check (no learned gait).
    """
    cfg = WBCConfig(
        policy_path="x.onnx",
        default_angles=[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0],
        kps=[150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250],
        kds=[2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5],
    )
    p = WBCPolicy(config=cfg, walk=False, allow_missing_models=True)
    p.policy_session = _StubSession(fill=fill)
    return p


def test_build_torque_g1_has_ground_and_torque_actuators() -> None:
    """REGRESSION: the harness model must have a ground plane (bare G1 MJCF
    lacks one -> falls through space) and torque (motor) actuators."""
    import mujoco

    harness = _load_harness()
    mj, model, data, joint_names = harness._build_torque_g1()
    geom_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}
    assert "wbc_ground" in geom_names, "harness must add a ground plane"
    assert len(joint_names) == 29, f"expected 29 actuated joints, got {len(joint_names)}"
    assert tuple(joint_names[:15]) == WBC_G1_ALL_JOINTS[:15]
    # Every actuator converted to a FIXED-gain / NONE-bias motor (pure torque).
    for i in range(model.nu):
        assert model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE


def test_model_observation_shape_and_velocity() -> None:
    """The harness observation is the whole-body per-joint dict with .vel keys
    plus base quat/ang-vel - the inputs a balance controller actually needs."""
    harness = _load_harness()
    mj, model, data, joint_names = harness._build_torque_g1()
    obs = harness._model_observation(data, joint_names, len(joint_names))
    for name in WBC_G1_ALL_JOINTS:
        assert name in obs, f"missing joint position {name}"
        assert f"{name}.vel" in obs, f"missing joint velocity {name}.vel"
    assert len(obs["base_quat"]) == 4
    assert len(obs["base_ang_vel"]) == 3


def test_rollout_static_hold_does_not_fall_through_floor() -> None:
    """With a zero-action stub (static hold at the nominal stance via the real
    PD law), a short rollout must keep the base ABOVE the floor - i.e. the model
    has a ground and the torque path is wired. (We don't assert balance: a
    zero-policy humanoid is not statically stable; we assert it does not fall
    THROUGH the world, the bug the missing ground plane caused: z -> -18.)"""
    harness = _load_harness()
    policy = _stub_policy(fill=0.0)
    result = harness.simulate_rollout(policy, vx=0.0, duration=0.2, physics_dt=0.005, control_decimation=4)
    # 0.2s is short enough that even an unbalanced stance hasn't toppled; the key
    # property is the base is near its start height, not at z = -18 (fell through).
    assert result["z1"] > 0.5, f"base fell through the floor (z1={result['z1']}); ground plane missing?"
    assert result["steps"] >= 1
    assert not result["fell"]


def test_rollout_returns_expected_metric_keys() -> None:
    harness = _load_harness()
    policy = _stub_policy(fill=0.0)
    result = harness.simulate_rollout(policy, vx=0.0, duration=0.1, physics_dt=0.005)
    assert set(result) >= {"x0", "z0", "x1", "z1", "forward", "fell", "steps", "frames"}
    assert isinstance(result["frames"], list) and result["frames"] == []  # no renderer_dims
