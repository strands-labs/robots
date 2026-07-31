"""IsaacSimulation.get_body_state - MuJoCo-envelope-compatible body pose reads.

The method is the read primitive under BOTH consumers #1802 unblocked on the
Isaac backend: the predicate DSL (``_body_position`` / ``_body_quaternion``)
and ``LiberoAdapter._read_eef_pose``'s body-state fallback (the source of the
``state.x/y/z/roll/pitch/yaw`` keys the ``libero_panda`` GR00T data-config
requires). These unit tests exercise it through a skeleton ``IsaacSimulation``
built with ``__new__`` (same pattern as ``test_dataset_recording.py``) so the
resolution order, envelope contract, quaternion conventions, and error shapes
are pinned without the Isaac Sim Kit runtime. The USD prim-walk path (which
needs a live Kit stage) is covered by
``tests_integ/simulation/test_isaac_body_state_gpu.py``.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import (
    IsaacSimulation,
    _ObjectState,
    _quat_wxyz_to_rotmat,
    _RobotState,
    _rotmat_to_quat_wxyz,
)


class _FakeRigidHandle:
    """Stub rigid-prim handle: world pose + optional velocities."""

    def __init__(
        self,
        pos: list[float],
        quat_wxyz: list[float],
        linear_velocity: list[float] | None = None,
        angular_velocity: list[float] | None = None,
    ) -> None:
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat_wxyz, dtype=np.float64)
        self._lin = linear_velocity
        self._ang = angular_velocity

    def get_world_pose(self):
        return self._pos, self._quat

    def get_linear_velocity(self):
        if self._lin is None:
            raise AttributeError("no velocity on this handle")
        return np.asarray(self._lin, dtype=np.float64)

    def get_angular_velocity(self):
        if self._ang is None:
            raise AttributeError("no velocity on this handle")
        return np.asarray(self._ang, dtype=np.float64)


def _make_engine(
    objects: dict[str, _ObjectState] | None = None,
    robots: dict[str, _RobotState] | None = None,
) -> IsaacSimulation:
    """Skeleton IsaacSimulation with exactly the state get_body_state reads."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = robots if robots is not None else {}
    engine._objects = objects if objects is not None else {}
    engine._pump_running = False
    engine._main_tid = threading.get_ident()
    return engine


def _object(name: str, handle: Any) -> _ObjectState:
    obj = _ObjectState(name=name, prim_path=f"/World/Objects/{name}", shape="box", is_static=False)
    obj.handle = handle
    return obj


def _json_payload(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if "json" in c)


def test_no_world_errors() -> None:
    engine = _make_engine()
    engine._world_created = False
    result = engine.get_body_state(body_name="anything")
    assert result["status"] == "error"
    assert "No world" in result["content"][0]["text"]


@pytest.mark.parametrize("bad_name", ["", "   ", None, 42])
def test_invalid_body_name_errors(bad_name) -> None:
    engine = _make_engine()
    result = engine.get_body_state(body_name=bad_name)
    assert result["status"] == "error"
    assert "non-empty string" in result["content"][0]["text"]


def test_object_pose_success_envelope() -> None:
    """Registered object resolves to the MuJoCo-compatible envelope shape."""
    quat = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]  # 90 deg about z
    handle = _FakeRigidHandle([0.1, -0.2, 0.3], quat)
    engine = _make_engine(objects={"cube_main": _object("cube_main", handle)})

    result = engine.get_body_state(body_name="cube_main")

    assert result["status"] == "success"
    assert any("text" in c for c in result["content"])
    payload = _json_payload(result)
    assert payload["position"] == pytest.approx([0.1, -0.2, 0.3])
    assert payload["quaternion"] == pytest.approx(quat)
    # rotation_matrix must agree with the quaternion (90 deg about z).
    expected_r = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert np.asarray(payload["rotation_matrix"]) == pytest.approx(np.asarray(expected_r), abs=1e-12)
    # No velocities were exposed -> keys OMITTED, never zero-filled.
    assert "linear_velocity" not in payload
    assert "angular_velocity" not in payload


def test_object_velocities_included_when_handle_provides_them() -> None:
    handle = _FakeRigidHandle([0, 0, 1], [1, 0, 0, 0], linear_velocity=[0.5, 0, 0], angular_velocity=[0, 0, 2.0])
    engine = _make_engine(objects={"ball": _object("ball", handle)})

    payload = _json_payload(engine.get_body_state(body_name="ball"))

    assert payload["linear_velocity"] == pytest.approx([0.5, 0.0, 0.0])
    assert payload["angular_velocity"] == pytest.approx([0.0, 0.0, 2.0])


def test_unknown_body_error_is_actionable() -> None:
    """The error names known objects/robots and the namespaced-link form."""
    engine = _make_engine(
        objects={"mug_main": _object("mug_main", _FakeRigidHandle([0, 0, 0], [1, 0, 0, 0]))},
        robots={"robot": _RobotState(name="robot", prim_path="/World/Robots/robot", joint_names=[])},
    )

    result = engine.get_body_state(body_name="does_not_exist")

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "does_not_exist" in text
    assert "mug_main" in text
    assert "robot" in text
    assert "<robot>/<link>" in text


def test_broken_object_handle_falls_through_to_error() -> None:
    """A handle whose pose read raises degrades to the structured error."""

    class _BrokenHandle:
        def get_world_pose(self):
            raise RuntimeError("physics view torn down")

    engine = _make_engine(objects={"cube": _object("cube", _BrokenHandle())})
    result = engine.get_body_state(body_name="cube")
    assert result["status"] == "error"
    assert "cube" in result["content"][0]["text"]


def test_prim_path_resolution_used_for_robot_links(monkeypatch) -> None:
    """Non-object names route to the USD prim read (mocked: no Kit runtime)."""
    engine = _make_engine(robots={"robot": _RobotState(name="robot", prim_path="/World/Robots/robot", joint_names=[])})
    rotmat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def _fake_prim_state(body_name: str):
        assert body_name == "robot/panda_hand"
        return {
            "position": [0.4, 0.0, 0.5],
            "quaternion": _rotmat_to_quat_wxyz(rotmat),
            "rotation_matrix": rotmat,
            "source": "prim",
            "prim_path": "/World/Robots/robot/panda_hand",
        }

    monkeypatch.setattr(engine, "_prim_body_state", _fake_prim_state)

    result = engine.get_body_state(body_name="robot/panda_hand")

    assert result["status"] == "success"
    payload = _json_payload(result)
    assert payload["position"] == pytest.approx([0.4, 0.0, 0.5])
    assert payload["quaternion"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_prim_read_unavailable_without_kit_runtime() -> None:
    """The real USD path degrades to None (-> error envelope) off-Kit.

    On a host without omni.usd/pxr importable, _prim_body_state must not
    raise; get_body_state reports the structured unknown-body error.
    """
    engine = _make_engine(robots={"robot": _RobotState(name="robot", prim_path="/World/Robots/robot", joint_names=[])})
    result = engine.get_body_state(body_name="robot/panda_hand")
    # Either the Kit runtime is present (then the stage lookup itself fails
    # gracefully) or it is not importable; both must yield the error envelope.
    assert result["status"] == "error"


class TestQuaternionHelpers:
    """The wxyz/rotation-matrix conversions get_body_state relies on."""

    @pytest.mark.parametrize(
        "quat",
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],  # 180 deg about x (Shepperd branch)
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.9238795325112867, 0.0, 0.3826834323650898, 0.0],
        ],
    )
    def test_roundtrip(self, quat) -> None:
        recovered = _rotmat_to_quat_wxyz(_quat_wxyz_to_rotmat(np.asarray(quat)))
        q = np.asarray(quat)
        r = np.asarray(recovered)
        # Double cover: q and -q encode the same rotation.
        assert min(np.linalg.norm(r - q), np.linalg.norm(r + q)) < 1e-12

    def test_rotmat_to_quat_is_normalized_and_w_canonical(self) -> None:
        q = _rotmat_to_quat_wxyz([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert q[0] >= 0.0
