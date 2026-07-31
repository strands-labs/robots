"""Backend-agnostic EEF state source: get_body_state fallback + Isaac plumbing (#1802).

On backends without a compiled MuJoCo model (Isaac), ``LiberoAdapter`` can
only source the ``state.x/y/z/roll/pitch/yaw`` keys from the sim's
``get_body_state`` duck-typed contract, and ``state.gripper`` from the
observation dict itself. #1802 added:

* ``eef_pos_offset`` / ``eef_quat_offset`` - site-equivalent corrections
  applied to the body-state fallback (the wrist body sits ~9.7 cm behind
  the gripper tip RoboSuite's ``state.x/y/z`` was trained on);
* ``_read_gripper_qpos_from_obs`` + ``state_gripper_signs`` - both finger
  qpos read from obs keys, with the RoboSuite opposite-sign convention
  restorable for backends whose gripper drives both fingers positive;
* a loud ERROR (not a DEBUG skip) when injection is enabled but produces
  no state keys - previously the failure surfaced only as the GR00T
  server's cryptic ``State key 'state.x' must be in observation``.

These tests run against a pure-Python fake sim (no MuJoCo world, no Isaac
Kit), pinning exactly the code paths the Isaac backend exercises.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from strands_robots.benchmarks.libero import LiberoAdapter
from strands_robots.benchmarks.libero.adapter import (
    _quat_wxyz_multiply,
    _quat_wxyz_rotate_vec,
    _quat_wxyz_to_rpy_xyz,
)
from strands_robots.simulation.base import SimEngine

_PICK_CUBE_BDDL = """
(define (problem libero_eef_probe)
  (:language "pick up the cube")
  (:objects cube_1 - object)
  (:goal (on cube_1 table_1)))
"""

# 90 degrees about z - large enough that a missed rotation of the local
# offset shows up as centimetres of position error.
_HAND_QUAT = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
_HAND_POS = [0.4, -0.1, 0.9]


class _FakeSim(SimEngine):
    """Non-MuJoCo sim exposing only the ``get_body_state`` contract.

    ``_world`` has no ``_model`` / ``_data``, so the adapter's direct-MuJoCo
    reads no-op exactly as they do on the Isaac backend, forcing the
    body-state fallback.
    """

    def __init__(self, body_states: dict[str, dict[str, Any]] | None = None) -> None:
        self._world = None
        self._body_states = body_states or {}
        self.body_state_calls: list[str] = []

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        self.body_state_calls.append(body_name)
        state = self._body_states.get(body_name)
        if state is None:
            return {"status": "error", "content": [{"text": f"Body '{body_name}' not found"}]}
        return {"status": "success", "content": [{"text": "ok"}, {"json": state}]}

    # --- SimEngine stubs ---------------------------------------------------
    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self):
        return []

    def robot_joint_names(self, robot_name):
        return []

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {}

    def send_action(self, action, robot_name=None, n_substeps=1):
        return {"status": "success"}

    def physics_timestep(self):
        return 0.002

    def render(self, camera_name="default", width=640, height=480):
        return {"status": "success", "content": []}


class _NoBodyStateSim(_FakeSim):
    """A sim WITHOUT get_body_state - the pre-#1802 Isaac situation."""

    get_body_state = None  # type: ignore[assignment]

    def __getattribute__(self, name):
        if name == "get_body_state":
            raise AttributeError(name)
        return super().__getattribute__(name)


def _isaac_like_sim() -> _FakeSim:
    return _FakeSim(body_states={"panda_hand": {"position": list(_HAND_POS), "quaternion": list(_HAND_QUAT)}})


def _make_adapter(**kwargs) -> LiberoAdapter:
    return LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        **kwargs,
    )


class TestFallbackOffsets:
    def test_pos_offset_rotated_into_body_frame(self):
        """eef_pos_offset is expressed in the body frame: rotated by the
        body quaternion, not added in world coordinates."""
        adapter = _make_adapter(eef_body_name="panda_hand", eef_pos_offset=[0.0, 0.0, 0.097])
        pos, quat = adapter._read_eef_pose(_isaac_like_sim())

        expected = np.array(_HAND_POS) + np.array(_quat_wxyz_rotate_vec(_HAND_QUAT, [0.0, 0.0, 0.097]))
        np.testing.assert_allclose(pos, expected, atol=1e-12)
        np.testing.assert_allclose(quat, _HAND_QUAT, atol=1e-12)
        # 90 deg about z maps the local +z offset to world +z here; the
        # rotation matters for the general case - pin that the offset was
        # NOT a plain world-frame add for a non-z-preserving quat too.
        tilted_quat = [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]  # 90 deg about x
        sim = _FakeSim(body_states={"panda_hand": {"position": [0, 0, 0], "quaternion": tilted_quat}})
        pos2, _ = adapter._read_eef_pose(sim)
        np.testing.assert_allclose(pos2, [0.0, -0.097, 0.0], atol=1e-12)

    def test_quat_offset_right_multiplied(self):
        offset = [0.9238795325112867, 0.0, 0.0, 0.3826834323650898]  # 45 deg about z
        adapter = _make_adapter(eef_body_name="panda_hand", eef_quat_offset=offset)
        _, quat = adapter._read_eef_pose(_isaac_like_sim())
        np.testing.assert_allclose(quat, _quat_wxyz_multiply(_HAND_QUAT, offset), atol=1e-12)

    def test_pos_offset_uses_raw_body_quat_before_quat_offset(self):
        """The local-frame position offset must be expressed with the RAW body
        orientation; the quaternion correction applies afterwards."""
        quat_offset = [0.7071067811865476, 0.0, 0.0, -0.7071067811865476]  # -90 deg about z
        adapter = _make_adapter(
            eef_body_name="panda_hand",
            eef_pos_offset=[0.1, 0.0, 0.0],
            eef_quat_offset=quat_offset,
        )
        pos, quat = adapter._read_eef_pose(_isaac_like_sim())

        expected_pos = np.array(_HAND_POS) + np.array(_quat_wxyz_rotate_vec(_HAND_QUAT, [0.1, 0.0, 0.0]))
        np.testing.assert_allclose(pos, expected_pos, atol=1e-12)
        np.testing.assert_allclose(quat, _quat_wxyz_multiply(_HAND_QUAT, quat_offset), atol=1e-12)

    def test_no_offsets_leaves_fallback_pose_unchanged(self):
        adapter = _make_adapter(eef_body_name="panda_hand")
        pos, quat = adapter._read_eef_pose(_isaac_like_sim())
        np.testing.assert_allclose(pos, _HAND_POS, atol=1e-12)
        np.testing.assert_allclose(quat, _HAND_QUAT, atol=1e-12)

    def test_augment_observation_rpy_reflects_corrected_quat(self):
        offset = [0.9238795325112867, 0.0, 0.0, 0.3826834323650898]
        adapter = _make_adapter(eef_body_name="panda_hand", eef_quat_offset=offset)
        merged = adapter.augment_observation(_isaac_like_sim(), {"panda_finger_joint1": 0.02})
        expected_rpy = _quat_wxyz_to_rpy_xyz(_quat_wxyz_multiply(_HAND_QUAT, offset))
        assert (merged["roll"], merged["pitch"], merged["yaw"]) == pytest.approx(expected_rpy)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"eef_pos_offset": [1.0, 2.0]}, "eef_pos_offset"),
            ({"eef_pos_offset": "not-a-vector"}, "eef_pos_offset"),
            ({"eef_quat_offset": [1.0, 0.0, 0.0]}, "eef_quat_offset"),
            ({"eef_quat_offset": [2.0, 0.0, 0.0, 0.0]}, "unit quaternion"),
            ({"state_gripper_signs": [1.0]}, "state_gripper_signs"),
        ],
    )
    def test_malformed_offsets_raise_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _make_adapter(**kwargs)


class TestGripperFromObs:
    def test_both_fingers_read_from_obs_with_signs(self):
        """Isaac path: both finger qpos come from obs keys; [1, -1] restores
        the RoboSuite opposite-sign convention."""
        adapter = _make_adapter(
            eef_body_name="panda_hand",
            state_gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
            state_gripper_signs=[1.0, -1.0],
        )
        obs = {"panda_finger_joint1": 0.021, "panda_finger_joint2": 0.019}
        merged = adapter.augment_observation(_isaac_like_sim(), obs)
        assert merged["gripper"] == pytest.approx([0.021, -0.019])

    def test_namespaced_obs_keys_match_by_suffix(self):
        adapter = _make_adapter(
            eef_body_name="panda_hand",
            state_gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
        )
        obs = {"arm/panda_finger_joint1": 0.02, "arm/panda_finger_joint2": 0.02}
        merged = adapter.augment_observation(_isaac_like_sim(), obs)
        assert merged["gripper"] == pytest.approx([0.02, 0.02])

    def test_missing_finger_falls_back_to_single_joint_duplicate(self):
        """Whole-vector-or-nothing: one missing finger -> the legacy
        single-joint duplicate path (with signs still applied)."""
        adapter = _make_adapter(
            eef_body_name="panda_hand",
            gripper_joint_name="panda_finger_joint1",
            state_gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
            state_gripper_signs=[1.0, -1.0],
        )
        obs = {"panda_finger_joint1": 0.02}  # joint2 missing
        merged = adapter.augment_observation(_isaac_like_sim(), obs)
        assert merged["gripper"] == pytest.approx([0.02, -0.02])

    def test_signs_default_to_identity(self):
        adapter = _make_adapter(
            eef_body_name="panda_hand",
            state_gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
        )
        obs = {"panda_finger_joint1": 0.021, "panda_finger_joint2": 0.019}
        merged = adapter.augment_observation(_isaac_like_sim(), obs)
        assert merged["gripper"] == pytest.approx([0.021, 0.019])


class TestLoudErrorOnMissingState:
    def test_error_logged_when_nothing_injectable(self, caplog):
        """No get_body_state + no gripper obs keys -> ERROR naming the missing
        keys and the backend fix, not a silent DEBUG skip."""
        adapter = _make_adapter()
        with caplog.at_level(logging.ERROR, logger="strands_robots.benchmarks.libero.adapter"):
            merged = adapter.augment_observation(_NoBodyStateSim(), {})
        assert all(k not in merged for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"))
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        message = errors[0].getMessage()
        assert "could not inject EEF state" in message
        assert "state.x" in message
        assert "eef_body_name" in message
        assert "panda_hand" in message  # the actionable Isaac hint

    def test_error_logged_once_per_adapter(self, caplog):
        adapter = _make_adapter()
        with caplog.at_level(logging.ERROR, logger="strands_robots.benchmarks.libero.adapter"):
            adapter.augment_observation(_NoBodyStateSim(), {})
            adapter.augment_observation(_NoBodyStateSim(), {})
        assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1

    def test_no_error_when_injection_succeeds(self, caplog):
        adapter = _make_adapter(
            eef_body_name="panda_hand",
            state_gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
        )
        obs = {"panda_finger_joint1": 0.02, "panda_finger_joint2": 0.02}
        with caplog.at_level(logging.ERROR, logger="strands_robots.benchmarks.libero.adapter"):
            merged = adapter.augment_observation(_isaac_like_sim(), obs)
        assert all(k in merged for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"))
        assert not [r for r in caplog.records if r.levelno == logging.ERROR]

    def test_no_error_when_injection_disabled(self, caplog):
        adapter = _make_adapter(inject_eef_state=False)
        with caplog.at_level(logging.ERROR, logger="strands_robots.benchmarks.libero.adapter"):
            adapter.augment_observation(_NoBodyStateSim(), {})
        assert not [r for r in caplog.records if r.levelno == logging.ERROR]


class TestDirectMuJoCoPathUnaffected:
    """The offsets are fallback-only: a resolving site read is never corrected."""

    def test_site_read_ignores_offsets(self):
        mujoco = pytest.importorskip("mujoco")

        xml = """
        <mujoco model="eef_probe">
          <worldbody>
            <body name="robot0_right_hand" pos="0.3 0.0 0.5" quat="0 1 0 0">
              <geom type="box" size="0.02 0.02 0.02"/>
              <body name="tip" pos="0 0 -0.097">
                <site name="gripper0_grip_site" pos="0 0 0" size="0.005"/>
                <geom type="box" size="0.01 0.01 0.01"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        class _World:
            _model = model
            _data = data
            robots: dict[str, object] = {}

        sim = _FakeSim()
        sim._world = _World()

        adapter = _make_adapter(
            eef_body_name="robot0_right_hand",
            eef_state_site_name="gripper0_grip_site",
            eef_pos_offset=[0.0, 0.0, 0.5],  # would move the pose half a metre if misapplied
        )
        pos, _ = adapter._read_eef_pose(sim)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        np.testing.assert_allclose(pos, np.array(data.site_xpos[sid]), atol=1e-9)
