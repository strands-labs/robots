"""Regression tests for the fresh-clone SO101/SO100 customer workflow.

Each test pins a behavior that a first-run customer relies on, so a future
refactor can't silently reintroduce a setup paper cut. Behaviors covered:

* The Feetech servo SDK ships with the ``[lerobot]`` extra.
* ``Robot("so100")`` is callable and dispatches actions (README contract).
* Pre-0.5 SO-family calibration files auto-migrate to the new path.
* README parameter names work as action-dispatch aliases.
* The mesh exposes dict-style peer lookup, not just a list.

The localhost mesh dev preset (``STRANDS_MESH_LOCAL_DEV``) is covered in
``tests/mesh/test_zenoh_config.py::TestLocalDevAuthPreset``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_sim(tmp_path):
    """Construct a minimal MuJoCo sim from inline MJCF (no asset download)."""
    from strands_robots.robot import Robot

    mjcf_xml = """<mujoco model="test_arm">
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="1 1 0.1"/>
        <body name="link0" pos="0 0 0.1">
          <joint name="joint0" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.02" fromto="0 0 0  0 0 0.2"/>
          <body name="link1" pos="0 0 0.2">
            <joint name="joint1" type="hinge" axis="0 1 0"/>
            <geom type="capsule" size="0.02" fromto="0 0 0  0 0 0.2"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor joint="joint0" ctrlrange="-1 1"/>
        <motor joint="joint1" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>"""
    mjcf_path = tmp_path / "test_arm.xml"
    mjcf_path.write_text(mjcf_xml)
    return Robot("so100", mode="sim", backend="mujoco", urdf_path=str(mjcf_path), mesh=False)


class TestLerobotExtraIncludesFeetech:
    """The ``[lerobot]`` extra must pull ``lerobot[feetech]`` for SO/Koch arms."""

    def test_lerobot_extra_includes_feetech(self):
        with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        lerobot_extra = data["project"]["optional-dependencies"]["lerobot"]
        # Exactly one lerobot pin, and it must carry the [feetech] marker so
        # scservo_sdk is installed for Feetech-based customers' first run.
        joined = " ".join(lerobot_extra)
        assert "lerobot[feetech]" in joined, f"[lerobot] extra must request lerobot[feetech]; got {lerobot_extra!r}"


class TestSimulationIsCallable:
    """The Simulation returned by ``Robot()`` must be callable and dispatch actions."""

    def test_sim_is_callable(self, tmp_path):
        pytest.importorskip("mujoco")
        sim = _build_sim(tmp_path)
        try:
            assert callable(sim), "Robot('so100') must be callable per README"
        finally:
            sim.destroy()

    def test_call_dispatches_action(self, tmp_path):
        pytest.importorskip("mujoco")
        sim = _build_sim(tmp_path)
        try:
            # list_robots is a safe, side-effect-free dispatch.
            result = sim(action="list_robots")
            assert isinstance(result, dict)
            assert result.get("status") == "success"
        finally:
            sim.destroy()

    def test_call_empty_action_returns_error_dict(self, tmp_path):
        pytest.importorskip("mujoco")
        sim = _build_sim(tmp_path)
        try:
            result = sim(action="")
            assert result["status"] == "error"
            # No raise — structured error per AgentTool contract.
            assert "action" in result["content"][0]["text"].lower()
        finally:
            sim.destroy()

    def test_call_matches_direct_method(self, tmp_path):
        pytest.importorskip("mujoco")
        sim = _build_sim(tmp_path)
        try:
            via_call = sim(action="list_robots")
            via_method = sim.list_robots_info()
            assert via_call["status"] == via_method["status"] == "success"
        finally:
            sim.destroy()


class TestReadmeParamAliases:
    """README parameter names must work as action-dispatch field aliases."""

    def test_joint_positions_alias(self, tmp_path):
        pytest.importorskip("mujoco")
        sim = _build_sim(tmp_path)
        try:
            # README used joint_positions=; canonical is positions=.
            # Both must reach set_joint_positions without a kwarg error.
            result = sim(action="set_joint_positions", joint_positions=[0.1, 0.2])
            assert result["status"] == "success", result
        finally:
            sim.destroy()

    def test_camera_names_alias_accepted(self, tmp_path):
        pytest.importorskip("mujoco")
        sim = _build_sim(tmp_path)
        try:
            # README used camera_names=; canonical is cameras=. The alias must
            # not produce an "unknown parameter" error. (We don't assert the
            # recording fully succeeds — just that the alias is routed.)
            result = sim(action="start_cameras_recording", camera_names=["default"], output_dir=str(tmp_path))
            text = result["content"][0]["text"].lower()
            assert "unknown parameter" not in text, result
            assert "unexpected keyword" not in text, result
        finally:
            with __import__("contextlib").suppress(Exception):
                sim(action="stop_cameras_recording")
            sim.destroy()


class TestCalibrationAutoMigration:
    """Pre-0.5 SO-family calibration files must auto-migrate to the new path."""

    def _hw_stub(self, calibration_fpath):
        """A HardwareRobot instance with only what the migration needs."""
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)

        class _FakeRobot:
            pass

        fake = _FakeRobot()
        fake.calibration_fpath = calibration_fpath
        hw.robot = fake
        return hw

    def test_migrates_single_legacy_file(self, tmp_path):
        calib_root = tmp_path / "robots"
        old = calib_root / "so101_follower" / "myarm.json"
        new = calib_root / "so_follower" / "myarm.json"
        old.parent.mkdir(parents=True)
        old.write_text('{"calib": true}')

        hw = self._hw_stub(new)
        hw._migrate_legacy_calibration()

        assert new.is_file(), "legacy cal file should have been copied to new path"
        assert new.read_text() == '{"calib": true}'
        # Copy, not move — old path still present for old lerobot installs.
        assert old.is_file()

    def test_noop_when_new_path_exists(self, tmp_path):
        calib_root = tmp_path / "robots"
        old = calib_root / "so101_follower" / "myarm.json"
        new = calib_root / "so_follower" / "myarm.json"
        old.parent.mkdir(parents=True)
        new.parent.mkdir(parents=True)
        old.write_text('{"old": true}')
        new.write_text('{"new": true}')

        hw = self._hw_stub(new)
        hw._migrate_legacy_calibration()

        # Existing new file must NOT be clobbered.
        assert new.read_text() == '{"new": true}'

    def test_ambiguous_multiple_legacy_files_skips(self, tmp_path):
        calib_root = tmp_path / "robots"
        new = calib_root / "so_follower" / "myarm.json"
        for variant in ("so100_follower", "so101_follower"):
            p = calib_root / variant / "myarm.json"
            p.parent.mkdir(parents=True)
            p.write_text("{}")

        hw = self._hw_stub(new)
        hw._migrate_legacy_calibration()

        # Two candidates -> refuse to guess -> new path stays absent.
        assert not new.is_file()

    def test_no_calibration_fpath_is_safe(self):
        hw = self._hw_stub(None)
        # Must not raise even when the robot exposes no calibration_fpath.
        hw._migrate_legacy_calibration()

    def test_unrelated_robot_not_touched(self, tmp_path):
        # A non-SO robot (calibration_fpath not under so_follower/so_leader)
        # must be left entirely alone.
        calib_root = tmp_path / "robots"
        new = calib_root / "koch_follower" / "arm.json"
        hw = self._hw_stub(new)
        hw._migrate_legacy_calibration()
        assert not new.is_file()


class TestMeshPeerDictLookup:
    """The mesh must expose dict-style peer lookup, not just a list."""

    def _make_mesh(self):
        from strands_robots.mesh.core import Mesh

        m = Mesh.__new__(Mesh)
        m.peer_id = "self-1"
        return m

    def test_peers_by_id_keys_on_peer_id(self, monkeypatch):
        from strands_robots.mesh import core as core_mod

        fake = [
            {"peer_id": "self-1", "hostname": "me"},
            {"peer_id": "peer-a", "hostname": "a"},
            {"peer_id": "peer-b", "hostname": "b"},
        ]
        monkeypatch.setattr(core_mod, "_session_get_peers", lambda: fake)

        m = self._make_mesh()
        by_id = m.peers_by_id
        assert isinstance(by_id, dict)
        # self excluded; others keyed by id.
        assert set(by_id) == {"peer-a", "peer-b"}
        assert by_id["peer-a"]["hostname"] == "a"

    def test_get_peer_none_safe(self, monkeypatch):
        from strands_robots.mesh import core as core_mod

        fake = [{"peer_id": "peer-a", "hostname": "a"}]
        monkeypatch.setattr(core_mod, "_session_get_peers", lambda: fake)

        m = self._make_mesh()
        assert m.get_peer("peer-a")["hostname"] == "a"
        assert m.get_peer("missing") is None
