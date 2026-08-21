"""Tests for strands_robots.robot - Robot() factory and list_robots()."""

import importlib
import os
import sys
import types

import pytest

from strands_robots.registry import (
    get_robot,
    list_aliases,
    list_robots,
    resolve_name,
)
from strands_robots.robot import (
    Robot,
    _attach_device_connect,
    _auto_detect_mode,
    _run_device_connect_foreground,
)


class TestResolveNames:
    def test_canonical(self):
        assert resolve_name("so100") == "so100"

    def test_alias(self):
        assert resolve_name("franka") == "panda"
        assert resolve_name("g1") == "unitree_g1"
        assert resolve_name("h1") == "unitree_h1"

    def test_case_insensitive(self):
        assert resolve_name("SO100") == "so100"
        assert resolve_name("Panda") == "panda"

    def test_hyphen_to_underscore(self):
        assert resolve_name("reachy-mini") == "reachy_mini"


class TestListRobots:
    def test_list_all(self):
        robots = list_robots("all")
        assert len(robots) > 0
        names = [r["name"] for r in robots]
        assert "so100" in names
        assert "panda" in names

    def test_list_sim(self):
        robots = list_robots("sim")
        for r in robots:
            assert r["has_sim"] is True

    def test_list_real(self):
        robots = list_robots("real")
        for r in robots:
            assert r["has_real"] is True

    def test_list_both(self):
        robots = list_robots("both")
        for r in robots:
            assert r["has_sim"] is True
            assert r["has_real"] is True

    def test_robot_has_fields(self):
        robots = list_robots()
        for r in robots:
            assert "name" in r
            assert "description" in r
            assert "has_sim" in r
            assert "has_real" in r


class TestRobotRegistry:
    def test_so100_exists(self):
        info = get_robot("so100")
        assert info is not None
        assert "asset" in info
        assert info["asset"]["dir"] == "trs_so_arm100"

    def test_all_aliases_point_to_valid_robots(self):
        aliases = list_aliases()
        for alias, canonical in aliases.items():
            info = get_robot(canonical)
            assert info is not None, f"Alias '{alias}' points to unknown robot '{canonical}'"

    def test_robot_count(self):
        """Ensure we have a reasonable number of robots."""
        robots = list_robots()
        assert len(robots) >= 30

    def test_all_robots_have_description(self):
        robots = list_robots()
        for r in robots:
            assert "description" in r, f"Robot '{r['name']}' missing description"
            assert len(r["description"]) > 0


class TestAutoDetectMode:
    @pytest.fixture(autouse=True)
    def _no_usb_hardware(self, monkeypatch):
        """Pin the port list empty so the verdict is the code's, not the host's.

        These tests assert the no-hardware default, but ``_auto_detect_mode``
        reads the host's real USB serial ports - so on a dev machine with a
        servo bus attached (an SO-10x CH34x bridge, vid 0x1a86) the detection
        correctly answers "real" and the assertion fails for a reason that has
        nothing to do with the code under test. Same shape as the shared-cache
        rule in AGENTS.md #15: a test that reads shared host state has a
        verdict that depends on what the host already holds.
        """
        import serial.tools.list_ports

        monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: [])

    def test_defaults_to_sim(self):
        """No hardware plugged in → sim."""
        assert _auto_detect_mode("so100") == "sim"

    def test_env_override_real(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROBOT_MODE", "real")
        assert _auto_detect_mode("so100") == "real"

    def test_env_override_sim(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROBOT_MODE", "sim")
        assert _auto_detect_mode("so100") == "sim"

    def test_env_override_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROBOT_MODE", "REAL")
        assert _auto_detect_mode("so100") == "real"

    def test_unrecognized_env_value_falls_through(self, monkeypatch):
        """Unrecognized STRANDS_ROBOT_MODE value is ignored with warning."""
        monkeypatch.setenv("STRANDS_ROBOT_MODE", "foo")
        # Falls through to default sim (logs warning)
        assert _auto_detect_mode("so100") == "sim"


class TestRobotFactory:
    def test_robot_is_callable(self):
        """Robot is a factory function, not a class."""
        import inspect

        assert callable(Robot)
        assert not inspect.isclass(Robot)

    def test_default_mode_is_sim(self):
        """Robot() defaults to sim mode - never accidentally sends to hardware."""
        import inspect

        sig = inspect.signature(Robot)
        assert sig.parameters["mode"].default == "sim"

    def test_isaac_backend_routes_to_factory_install_hint(self):
        """``isaac`` is now a vendored built-in backend (#1145), resolved through
        ``create_simulation`` like ``newton``. When Isaac Sim itself is not
        installed (it ships out-of-band via Omniverse / Isaac Lab / the NGC
        docker image, never via pip), constructing the sim succeeds but
        ``create_world()`` surfaces the backend's own actionable install hint -
        a ``RuntimeError`` naming Isaac Sim + its install paths - not a blanket
        NotImplementedError or a dead-end "on the roadmap" message.

        Skipped when Isaac Sim IS importable (a GPU box with Omniverse), where
        world creation would actually proceed."""
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        ok, _ = IsaacSimulation.is_available()
        if ok:
            pytest.skip("Isaac Sim is installed; the not-available install-hint path is not exercised.")
        with pytest.raises(RuntimeError, match="Isaac Sim") as exc_info:
            Robot("so100", mode="sim", backend="isaac")
        msg = str(exc_info.value)
        # The misleading legacy framing must be gone.
        assert "not yet implemented" not in msg
        assert "roadmap" not in msg

    def test_newton_backend_surfaces_dependency_install_hint(self, monkeypatch):
        """``newton`` is a built-in (warp-lang GPU) backend. When its optional
        deps (``warp``/``newton``) are absent, ``Robot(backend="newton")``
        surfaces the backend's own ``ImportError`` naming the ``sim-newton``
        pip extra - the same error ``create_simulation("newton")`` raises -
        instead of masking it behind a generic NotImplementedError.

        The optional deps may or may not be installed in a given environment
        (they are present on GPU dev boxes, absent in CPU CI), so simulate
        their absence deterministically rather than relying on the ambient
        install state: clear both lazy-import caches and make importing
        ``warp``/``newton`` fail, then assert the install hint surfaces. This
        keeps the test green wherever the ``sim-newton`` backend is installed."""
        import strands_robots.simulation.newton.backend as newton_backend
        from strands_robots import utils as sr_utils

        # Defeat both memoization layers so the import is genuinely re-attempted:
        # the backend module's ``_modules`` cache and ``require_optional``'s
        # ``_lazy_modules`` cache (which may already hold ``warp``/``newton``).
        monkeypatch.setattr(newton_backend, "_modules", {})
        for name in ("warp", "newton"):
            sr_utils._lazy_modules.pop(name, None)

        real_import_module = importlib.import_module

        def _blocked_import(name, *args, **kwargs):
            if name in ("warp", "newton") or name.startswith(("warp.", "newton.")):
                raise ImportError(f"No module named {name!r}")
            return real_import_module(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", _blocked_import)

        with pytest.raises(ImportError, match="sim-newton"):
            Robot("so100", mode="sim", backend="newton")

    def test_unknown_backend_lists_available(self):
        """A genuinely unknown backend name yields the factory's ``ValueError``
        that lists the available backends so the caller can self-correct."""
        with pytest.raises(ValueError, match="Unknown simulation backend"):
            Robot("so100", mode="sim", backend="totally_not_a_backend")

    def test_mjwarp_backend_gives_plugin_install_hint(self):
        """The GPU-parallel warp/MuJoCo path is the built-in ``newton`` backend;
        ``mjwarp`` is a name users reach for. ``Robot(backend="mjwarp")`` must
        point them at the in-tree ``strands-robots[sim-newton]`` extra rather
        than dead-ending on an unhelpful unknown-backend error with no remedy.
        The hint must not reference the deprecated ``strands-robots-sim``
        sibling package."""
        with pytest.raises(ValueError, match=r"strands-robots\[sim-newton\]") as exc_info:
            Robot("so100", mode="sim", backend="mjwarp", num_envs=128)
        msg = str(exc_info.value)
        assert "pip install" in msg
        assert "strands-robots-sim" not in msg

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            Robot("so100", mode="invalid")

    def test_cameras_rejected_in_sim_mode(self):
        """Passing cameras= in sim mode raises ValueError."""
        with pytest.raises(ValueError, match="cameras= is only supported in mode='real'"):
            Robot("so100", mode="sim", cameras={"wrist": {"type": "opencv"}})

    def test_sim_with_urdf_path(self):
        """Robot() with explicit urdf_path should work (if file exists)."""
        pytest.importorskip("mujoco")
        with pytest.raises(RuntimeError):
            Robot("test_bot", mode="sim", urdf_path="/nonexistent/robot.xml")

    def test_sim_happy_path_mujoco(self, tmp_path):
        """Happy-path: create a MuJoCo sim, step physics, destroy.

        Uses a minimal inline MJCF so the test works without downloaded assets.
        """
        mujoco = pytest.importorskip("mujoco")

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

        sim = Robot("so100", mode="sim", backend="mujoco", urdf_path=str(mjcf_path))
        try:
            assert sim._world is not None
            assert sim._world._model is not None
            assert sim._world._data is not None
            mujoco.mj_step(sim._world._model, sim._world._data)
            assert sim._world._data.time > 0
        finally:
            sim.destroy()

    def test_import_from_top_level(self):
        """Robot and list_robots importable from strands_robots."""
        from strands_robots import Robot as R
        from strands_robots import list_robots as lr

        assert R is Robot
        assert callable(lr)


class TestFactoryForwardsPosition:
    """``Robot(position=...)`` reaches the backend exactly as the caller wrote it.

    The factory is a thin wrapper over ``add_robot``, which validates a base
    pose up front: a wrong-length, non-numeric or non-finite vector is refused
    with an actionable message, an omitted pose spawns at the origin, and a
    NumPy pose is accepted. Reading the parameter by truthiness in the factory
    made the wrapper both less capable and less safe than the method it wraps -
    an empty vector read as "omitted" so the origin was substituted for a caller
    mistake ``add_robot`` refuses, and a NumPy pose raised a bare
    ``ValueError: truth value of an array ... is ambiguous``.
    """

    MJCF = """<mujoco model="test_arm">
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="1 1 0.1"/>
        <body name="link0" pos="0 0 0.1">
          <joint name="joint0" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.02" fromto="0 0 0  0 0 0.2"/>
        </body>
      </worldbody>
      <actuator><motor joint="joint0" ctrlrange="-1 1"/></actuator>
    </mujoco>"""

    @classmethod
    def _model(cls, tmp_path):
        """Write the minimal inline arm so no downloaded asset is needed."""
        path = tmp_path / "test_arm.xml"
        path.write_text(cls.MJCF)
        return str(path)

    def _spawned_position(self, tmp_path, **kwargs):
        """Return the base position the factory actually gave the backend."""
        sim = Robot("so100", mode="sim", urdf_path=self._model(tmp_path), mesh=False, **kwargs)
        try:
            return list(sim._world.robots["so100"].position)
        finally:
            sim.destroy()

    def test_omitted_position_spawns_at_the_origin(self, tmp_path):
        """The documented default survives: no position means the origin."""
        pytest.importorskip("mujoco")
        assert self._spawned_position(tmp_path) == [0.0, 0.0, 0.0]

    def test_list_position_is_forwarded_unchanged(self, tmp_path):
        pytest.importorskip("mujoco")
        assert self._spawned_position(tmp_path, position=[0.4, 0.2, 0.1]) == [0.4, 0.2, 0.1]

    def test_numpy_position_is_forwarded_unchanged(self, tmp_path):
        """A pose computed with NumPy is what pose arithmetic produces, and
        ``add_robot`` accepts it; the factory must not reject it."""
        pytest.importorskip("mujoco")
        np = pytest.importorskip("numpy")
        assert self._spawned_position(tmp_path, position=np.array([0.4, 0.2, 0.1])) == [0.4, 0.2, 0.1]

    def test_empty_position_is_refused_not_replaced_by_the_origin(self, tmp_path):
        """An empty vector is a caller mistake, not a request for the default.

        Substituting the origin would place the robot somewhere the caller never
        asked for while reporting success.
        """
        pytest.importorskip("mujoco")
        with pytest.raises(RuntimeError, match="3-element vector"):
            Robot("so100", mode="sim", urdf_path=self._model(tmp_path), mesh=False, position=[])

    def test_non_finite_position_is_refused(self, tmp_path):
        """nan/inf would propagate through the whole physics state."""
        pytest.importorskip("mujoco")
        with pytest.raises(RuntimeError, match="finite numbers"):
            Robot(
                "so100",
                mode="sim",
                urdf_path=self._model(tmp_path),
                mesh=False,
                position=[float("nan"), 0.0, 0.0],
            )

    def test_factory_and_add_robot_agree_on_every_position(self, tmp_path):
        """Parity: the wrapper accepts a pose if and only if the backend does.

        Any value where the two verdicts differ is a pose the caller can only
        express through one of the two entry points.
        """
        pytest.importorskip("mujoco")
        np = pytest.importorskip("numpy")
        from strands_robots import Simulation

        model = self._model(tmp_path)
        cases = [[], [0.5], [0.4, 0.2, 0.1], [float("inf"), 0.0, 0.0], np.array([0.4, 0.2, 0.1])]

        for position in cases:
            sim = Simulation(tool_name="parity", mesh=False)
            try:
                sim.create_world()
                backend_accepts = sim.add_robot(name="so100", urdf_path=model, position=position)["status"] != "error"
            finally:
                sim.destroy()

            try:
                factory = Robot("so100", mode="sim", urdf_path=model, mesh=False, position=position)
                factory.destroy()
                factory_accepts = True
            except RuntimeError:
                factory_accepts = False

            assert factory_accepts is backend_accepts, f"verdicts differ for position={position!r}"


class TestFactorySpawnParameterForwarding:
    """``Robot(...)`` must express every spawn parameter ``add_robot`` accepts.

    The factory is a thin front door to ``SimEngine.add_robot``, and ``position``
    is documented as forwarded verbatim so the backend's contract governs both
    its default and its refusals. ``orientation`` and ``keyframe`` are the same
    kind of parameter on the same method - both declared by the ABC and by every
    backend - but were not forwarded. The backend constructor's ``**kwargs``
    absorbed them, so a requested rotation or keyframe spawn was silently dropped
    and the robot came up unrotated in the zero configuration while the factory
    reported success. The refusals ``add_robot`` documents did not reach the
    caller either, including the one it states outright for a bad keyframe: "a
    hard error that names the available keyframes; it never silently falls back
    to zeros".
    """

    # A named actuator and a <keyframe> carrying BOTH halves of a keyframe: the
    # pose and the actuator command that holds it. The actuator is named because
    # a keyed ctrl is applied by actuator name, so an unnamed one is skipped and
    # could not show whether the command travelled.
    MJCF = """<mujoco model="test_arm">
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="1 1 0.1"/>
        <body name="link0" pos="0 0 0.1">
          <joint name="joint0" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.02" fromto="0 0 0  0 0 0.2"/>
        </body>
      </worldbody>
      <actuator><position name="act0" joint="joint0" kp="10" ctrlrange="-2 2"/></actuator>
      <keyframe><key name="home" qpos="0.6" ctrl="0.6"/></keyframe>
    </mujoco>"""

    ROTATION = [0.7071068, 0.0, 0.0, 0.7071068]  # 90 deg about x, wxyz

    @classmethod
    def _model(cls, tmp_path):
        """Write the inline arm so no downloaded asset is needed."""
        path = tmp_path / "keyframed_arm.xml"
        path.write_text(cls.MJCF)
        return str(path)

    def _spawn(self, tmp_path, **kwargs):
        """Return the spawn state the factory actually gave the backend."""
        sim = Robot("so100", mode="sim", urdf_path=self._model(tmp_path), mesh=False, **kwargs)
        try:
            return {
                "orientation": list(sim._world.robots["so100"].orientation),
                "qpos": [float(v) for v in sim._world._data.qpos],
                "ctrl": [float(v) for v in sim._world._data.ctrl],
            }
        finally:
            sim.destroy()

    def _add_robot(self, tmp_path, **kwargs):
        """The same spawn state reached through ``add_robot`` directly."""
        from strands_robots import Simulation

        sim = Simulation(tool_name="direct", mesh=False)
        try:
            sim.create_world()
            result = sim.add_robot(name="so100", urdf_path=self._model(tmp_path), **kwargs)
            assert result["status"] != "error", result
            return {
                "orientation": list(sim._world.robots["so100"].orientation),
                "qpos": [float(v) for v in sim._world._data.qpos],
                "ctrl": [float(v) for v in sim._world._data.ctrl],
            }
        finally:
            sim.destroy()

    def test_factory_can_express_every_add_robot_spawn_parameter(self):
        """Root cause: a parameter the factory cannot name is one a caller cannot reach.

        ``Robot(**kwargs)`` forwards leftovers to the *backend constructor*, not
        to ``add_robot``, and that constructor takes ``**kwargs`` of its own - so
        an unforwarded spawn parameter is absorbed with no error and no warning
        rather than surfacing as an unexpected-keyword failure.
        """
        import inspect

        from strands_robots.simulation.base import SimEngine

        backend = set(inspect.signature(SimEngine.add_robot).parameters) - {"self"}
        factory = set(inspect.signature(Robot).parameters) - {"kwargs"}
        unreachable = sorted(backend - factory)
        assert not unreachable, (
            f"Robot() cannot express add_robot parameter(s) {unreachable}; a caller passing "
            "one gets it absorbed by the backend constructor's **kwargs and silently dropped. "
            "Forward it in the factory's add_robot call, or give it a factory parameter that "
            "documents why the front door omits it."
        )

    def test_keyframe_reaches_the_pose_the_backend_would_spawn(self, tmp_path):
        """The whole point of the parameter: a non-zero canonical start pose."""
        pytest.importorskip("mujoco")
        assert self._spawn(tmp_path, keyframe="home")["qpos"] == pytest.approx([0.6])

    def test_keyframe_also_carries_the_actuator_command_that_holds_the_pose(self, tmp_path):
        """A MuJoCo ``<key>`` pairs a pose with the command that holds it.

        Forwarding only the pose would leave a gravity-loaded arm standing at its
        home configuration with its actuators commanded to zero, so the pose is
        not self-holding and collapses as soon as the world steps.
        """
        pytest.importorskip("mujoco")
        assert self._spawn(tmp_path, keyframe="home")["ctrl"] == pytest.approx([0.6])

    def test_keyframe_by_index_is_forwarded_too(self, tmp_path):
        """``add_robot`` documents a name *or* an index; both must reach it."""
        pytest.importorskip("mujoco")
        assert self._spawn(tmp_path, keyframe=0)["qpos"] == pytest.approx([0.6])

    def test_orientation_reaches_the_backend(self, tmp_path):
        pytest.importorskip("mujoco")
        assert self._spawn(tmp_path, orientation=self.ROTATION)["orientation"] == pytest.approx(self.ROTATION)

    def test_factory_and_add_robot_reach_the_same_spawn_state(self, tmp_path):
        """Parity: the front door and the method it wraps must agree.

        A difference here is a spawn state the caller can only reach by dropping
        out of the factory and calling the backend directly.
        """
        pytest.importorskip("mujoco")
        kwargs = {"keyframe": "home", "orientation": self.ROTATION}
        through_factory = self._spawn(tmp_path, **kwargs)
        through_backend = self._add_robot(tmp_path, **kwargs)
        for field in ("orientation", "qpos", "ctrl"):
            assert through_factory[field] == pytest.approx(through_backend[field]), (
                f"{field}: factory gave {through_factory[field]} but add_robot gives {through_backend[field]}"
            )

    def test_unknown_keyframe_is_refused_not_silently_spawned_at_zero(self, tmp_path):
        """``add_robot`` promises this is a hard error naming the available keyframes.

        Absorbed instead, the caller gets a robot in the zero configuration -
        out-of-distribution for a policy trained from the home pose - and nothing
        says the requested pose was never applied.
        """
        pytest.importorskip("mujoco")
        with pytest.raises(RuntimeError, match="home"):
            Robot(
                "so100",
                mode="sim",
                urdf_path=self._model(tmp_path),
                mesh=False,
                keyframe="not_a_keyframe",
            )

    def test_malformed_orientation_is_refused(self, tmp_path):
        """A 3-vector is a caller mistake, not a request for the identity rotation."""
        pytest.importorskip("mujoco")
        with pytest.raises(RuntimeError, match="4-element vector"):
            Robot(
                "so100",
                mode="sim",
                urdf_path=self._model(tmp_path),
                mesh=False,
                orientation=[0.0, 0.0, 1.0],
            )

    def test_non_finite_orientation_is_refused(self, tmp_path):
        """nan/inf in a base quaternion would propagate through the physics state."""
        pytest.importorskip("mujoco")
        with pytest.raises(RuntimeError, match="finite numbers"):
            Robot(
                "so100",
                mode="sim",
                urdf_path=self._model(tmp_path),
                mesh=False,
                orientation=[float("nan"), 0.0, 0.0, 0.0],
            )

    def test_omitting_them_keeps_the_historical_spawn(self, tmp_path):
        """No-overreach: forwarding must not change what an existing caller gets.

        Omitted, both parameters keep the documented defaults - the zero joint
        configuration, its actuators commanded to zero, and no rotation.
        """
        pytest.importorskip("mujoco")
        spawned = self._spawn(tmp_path)
        assert spawned["qpos"] == pytest.approx([0.0])
        assert spawned["ctrl"] == pytest.approx([0.0])
        assert spawned["orientation"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


class TestRobotRealMode:
    """Tests for mode='real' path (mocked - no physical hardware)."""

    def test_real_mode_requires_lerobot(self):
        """mode='real' imports lerobot hardware classes."""
        from unittest.mock import MagicMock, patch

        # Mock the hardware import to avoid needing lerobot installed
        with patch("strands_robots.robot.get_hardware_type", return_value="so100_follower"):
            with patch("strands_robots.hardware_robot.Robot") as mock_hw:
                mock_hw.return_value = MagicMock()
                try:
                    Robot("so100", mode="real")
                    mock_hw.assert_called_once()
                except ImportError:
                    # lerobot not installed - acceptable in unit CI
                    pass


class TestAutoDetectUSB:
    """Tests for USB-found-hardware branch in _auto_detect_mode."""

    def test_usb_detection_finds_feetech(self, monkeypatch):
        """Servo controller detected → returns 'real'."""
        pytest.importorskip("serial")
        from unittest.mock import MagicMock, patch

        mock_port = MagicMock()
        mock_port.description = "Feetech STS3215 Servo Controller"
        mock_port.device = "/dev/ttyUSB0"
        mock_port.manufacturer = "Feetech"

        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            assert _auto_detect_mode("so100") == "real"

    def test_usb_detection_excludes_bluetooth(self, monkeypatch):
        """Bluetooth device not treated as robot hardware."""
        pytest.importorskip("serial")
        from unittest.mock import MagicMock, patch

        mock_port = MagicMock()
        mock_port.description = "Bluetooth Internal Feetech"
        mock_port.device = "/dev/ttyBT0"
        mock_port.manufacturer = None

        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            assert _auto_detect_mode("so100") == "sim"

    def test_usb_detection_import_error(self, monkeypatch):
        """pyserial not installed → falls back to sim."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"serial": None, "serial.tools": None, "serial.tools.list_ports": None}):
            assert _auto_detect_mode("so100") == "sim"

    def test_usb_detection_no_robot_hardware(self, monkeypatch):
        """Robot without hardware support → skips USB scan."""
        from strands_robots.robot import _auto_detect_mode

        # "panda" may not have hardware support - defaults to sim
        result = _auto_detect_mode("panda")
        assert result == "sim"


class TestModeNormalization:
    """Mode parameter and STRANDS_ROBOT_MODE env var should agree on case/whitespace."""

    def test_mode_param_uppercase_accepted(self):
        """Robot('so100', mode='SIM') should work - env var path is case-insensitive,
        the direct param should be too."""
        pytest.importorskip("mujoco")
        sim = Robot("so100", mode="SIM")
        try:
            from strands_robots.simulation import Simulation

            assert isinstance(sim, Simulation)
        finally:
            sim.destroy()

    def test_mode_param_with_whitespace(self):
        """mode=' sim ' should be normalized like the env var is."""
        pytest.importorskip("mujoco")
        sim = Robot("so100", mode=" sim ")
        try:
            from strands_robots.simulation import Simulation

            assert isinstance(sim, Simulation)
        finally:
            sim.destroy()

    def test_env_var_with_whitespace(self, monkeypatch):
        """STRANDS_ROBOT_MODE='  sim  ' should resolve cleanly without firing the
        'ignored' warning."""
        from strands_robots.robot import _auto_detect_mode

        monkeypatch.setenv("STRANDS_ROBOT_MODE", "  sim  ")
        assert _auto_detect_mode("so100") == "sim"

    def test_env_var_auto_is_no_op(self, monkeypatch):
        """STRANDS_ROBOT_MODE=auto means 'do detection' - same as not setting it.
        Should not warn."""
        from strands_robots.robot import _auto_detect_mode

        monkeypatch.setenv("STRANDS_ROBOT_MODE", "auto")
        # Pin the port list empty: with a servo bus attached to the host the
        # detection correctly answers "real" and this asserts the wrong layer.
        import serial.tools.list_ports

        monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: [])
        # Auto-detect with no USB hardware → falls back to sim
        assert _auto_detect_mode("so100") == "sim"


class TestUnknownNameRejected:
    """Empty / whitespace / unknown robot names should raise ValueError before
    we descend into the sim or hardware backend, so the user sees one clean
    error instead of a confusing two-stage stderr+exception."""

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="Invalid robot name"):
            Robot("")

    def test_whitespace_name_rejected(self):
        with pytest.raises(ValueError, match="Invalid robot name"):
            Robot("  ")

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="Unknown robot"):
            Robot("definitely_not_a_robot_xyz")

    def test_unknown_name_rejected_in_real_mode(self):
        with pytest.raises(ValueError, match="Unknown robot"):
            Robot("definitely_not_a_robot_xyz", mode="real")

    def test_unknown_name_with_urdf_path_does_not_raise(self):
        """Explicit urdf_path bypasses the registry check - user knows what they
        want, we don't second-guess."""
        pytest.importorskip("mujoco")
        # Use a clearly-bogus path so the underlying load fails (as RuntimeError),
        # not a ValueError from validation. Cleanup is also covered separately.
        with pytest.raises(RuntimeError):
            Robot("my_custom_arm", urdf_path="/nonexistent/foo.xml")


class TestCleanupOnDispatchRaise:
    """If a backend world/robot-population call itself raises (vs returns
    status=error), the Simulation must still be destroyed. ``Robot()`` populates
    the world through the backend-agnostic ``SimEngine`` ABC methods
    (``create_world`` / ``add_robot``), so the cleanup path is pinned against
    those."""

    def test_destroy_called_when_create_world_raises(self):
        """OSError (or any exception) from create_world must trigger destroy()."""
        pytest.importorskip("mujoco")
        from unittest.mock import patch

        from strands_robots.simulation.mujoco.simulation import Simulation as SimImpl

        destroyed = []
        real_destroy = SimImpl.destroy

        def track(self):
            destroyed.append(self)
            return real_destroy(self)

        def raising_create_world(self, *args, **kwargs):
            raise OSError("simulated disk full")

        with (
            patch.object(SimImpl, "create_world", raising_create_world),
            patch.object(SimImpl, "destroy", track),
        ):
            with pytest.raises(OSError, match="simulated disk full"):
                Robot("so100")

        assert len(destroyed) == 1, f"destroy() should have been called once, was {len(destroyed)}x"

    def test_destroy_called_when_add_robot_raises(self):
        """RuntimeError from add_robot must trigger destroy()."""
        pytest.importorskip("mujoco")
        from unittest.mock import patch

        from strands_robots.simulation.mujoco.simulation import Simulation as SimImpl

        destroyed = []
        real_destroy = SimImpl.destroy

        def track(self):
            destroyed.append(self)
            return real_destroy(self)

        def raising_add_robot(self, *args, **kwargs):
            raise RuntimeError("simulated MJCF compile error")

        with (
            patch.object(SimImpl, "add_robot", raising_add_robot),
            patch.object(SimImpl, "destroy", track),
        ):
            with pytest.raises(RuntimeError, match="simulated MJCF compile error"):
                Robot("so100")

        assert len(destroyed) == 1, f"destroy() should have been called once, was {len(destroyed)}x"


class TestUSBProbeFallsBackOnRuntimeError:
    """libusb hub glitches can surface as RuntimeError from comports().
    _auto_detect_mode must fall back to sim, not propagate the exception."""

    def test_runtime_error_during_usb_probe(self):
        pytest.importorskip("serial")
        from unittest.mock import patch

        from strands_robots.robot import _auto_detect_mode

        def raise_runtime(*a, **kw):
            raise RuntimeError("simulated libusb hub glitch")

        with patch("serial.tools.list_ports.comports", side_effect=raise_runtime):
            # Must return "sim" (safe fallback), not raise.
            assert _auto_detect_mode("so100") == "sim"


class TestDashedNameAlias:
    """Common typo: users write 'so-100' (matches marketing). Should resolve to
    canonical 'so100' rather than producing a confusing 'Unknown robot' error."""

    def test_dashed_name_resolves_to_canonical(self):
        from strands_robots.registry import resolve_name

        assert resolve_name("so-100") == "so100"
        assert resolve_name("so_100") == "so100"
        assert resolve_name("SO-100") == "so100"


class TestCameraErrorMessage:
    """The cameras-in-sim error must NOT recommend the private _dispatch_action
    method - that's been a recurring review request."""

    def test_camera_error_does_not_leak_private_api(self):
        with pytest.raises(ValueError) as excinfo:
            Robot("so100", cameras={"wrist": {"type": "opencv"}})
        assert "_dispatch_action" not in str(excinfo.value), (
            "Error message should not mention the private _dispatch_action method"
        )


class TestRealModeConfigDiscovery:
    """Regression tests for `_create_minimal_config` switching from a
    hand-rolled mapping to lerobot's draccus ChoiceRegistry discovery.

    These tests use `pytest.importorskip("lerobot")` so they noop on
    machines without lerobot installed.
    """

    @pytest.fixture(autouse=True)
    def _clear_discovery_cache(self):
        """Reset ``_ensure_lerobot_robots_registered``'s
        ``@functools.cache`` around each test in the class so test
        ordering (``--last-failed``, ``pytest-xdist``, random-order
        plugins) cannot leave stale registry state behind. Without
        this, any test that booby-traps the walker (the OSError /
        decorator-failure pins) would have to remember to clear the
        cache on the way in AND out, and a future test in this class
        that forgets would inherit a half-populated registry from the
        last booby-trap and fail in a debugger-hostile way.
        """
        try:
            from strands_robots.hardware_robot import _ensure_lerobot_robots_registered
        except ImportError:
            yield
            return
        _ensure_lerobot_robots_registered.cache_clear()
        yield
        _ensure_lerobot_robots_registered.cache_clear()

    def test_lerobot_registry_discovery_finds_all_subpackages(self):
        """Walking ``lerobot.robots`` with pkgutil registers every robot
        config without any hard-coded type→module mapping. This is the
        future-proof path: any robot lerobot ships in
        ``lerobot/robots/<X>/`` (regardless of whether its ``robot_type``
        matches ``X``, e.g. ``hope_jr_arm`` lives in ``hope_jr/`` and
        ``lekiwi_client`` lives in ``lekiwi/``) automatically becomes
        constructible via ``Robot(...)`` mode='real'."""
        pytest.importorskip("lerobot")
        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

        _ensure_lerobot_robots_registered()
        registered = set(RobotConfig.get_known_choices().keys())

        # Pin only a single canonical entry to avoid upstream-coupled flake
        # risk (lerobot may rename/drop any of these in future releases).
        # The discovery contract: walking populates the registry from > 0
        # entries that include at least one driver from a standard subpackage.
        expected_min = {"so100_follower"}
        missing = expected_min - registered
        assert not missing, f"Discovery missed lerobot built-in: {missing}. Registered: {sorted(registered)}"
        # Sanity: the walk should discover more than just one
        assert len(registered) >= 3, f"Expected >= 3 registered types, got {len(registered)}: {sorted(registered)}"

    def test_subpackage_with_multiple_robots_picked_up(self):
        """Some lerobot subpackages register MULTIPLE robot_types (e.g.
        ``hope_jr/`` registers both ``hope_jr_arm`` and ``hope_jr_hand``;
        ``lekiwi/`` registers both ``lekiwi`` and ``lekiwi_client``).
        pkgutil-walking handles this naturally - a hand-rolled
        type→module map would have to special-case each."""
        pytest.importorskip("lerobot.robots.hope_jr")
        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

        _ensure_lerobot_robots_registered()
        registered = set(RobotConfig.get_known_choices().keys())
        # Multiple types from one subpackage:
        assert "hope_jr_arm" in registered
        assert "hope_jr_hand" in registered

    def test_so101_config_build_uses_RobotConfig_subclass(self):
        """Regression: lerobot 0.5.x's bare ``SOFollowerConfig`` has no
        ``id`` field - discovery picks the registered ``SOFollowerRobotConfig``
        subclass that does. (Original SO-100/SO-101 real-mode regression.)"""
        pytest.importorskip("lerobot.robots.so_follower")
        from strands_robots.hardware_robot import Robot as HwRobot

        # Build the config directly via the helper - no hardware connect.
        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "so101_smoke"
        cfg = hw._create_minimal_config("so101_follower", cameras={}, port="/dev/null", use_degrees=True)
        # Must be the registered subclass (has ``id``), not the bare config.
        assert hasattr(cfg, "id"), "config has no 'id' - used the wrong subclass"
        assert cfg.id == "so101_smoke"
        assert cfg.port == "/dev/null"
        # The registered subclass for so101 inherits both RobotConfig and
        # SOFollowerConfig - its name typically ends with `RobotConfig`.
        assert "RobotConfig" in type(cfg).__name__

    def test_unitree_g1_config_build_via_discovery(self):
        """Regression: ``unitree_g1`` was missing from the old hand-rolled
        config_mapping despite the registry advertising ``has_real=True``.
        Discovery via ChoiceRegistry picks it up automatically."""
        pytest.importorskip("lerobot.robots.unitree_g1")
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "g1_smoke"
        cfg = hw._create_minimal_config(
            "unitree_g1",
            cameras={},
            robot_ip="192.168.123.164",
            kp=[100.0] * 29,
            kd=[2.0] * 29,
            default_positions=[0.0] * 29,
            is_simulation=False,
        )
        assert cfg.id == "g1_smoke"
        assert cfg.robot_ip == "192.168.123.164"
        assert len(cfg.kp) == 29
        assert cfg.is_simulation is False

    def test_unknown_robot_type_raises_clean(self):
        """Unknown types produce an error listing the *actual* known types
        (not a stale hard-coded list)."""
        pytest.importorskip("lerobot.robots.config")
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "x"
        with pytest.raises(ValueError, match="Unsupported robot type"):
            hw._create_minimal_config("totally_made_up_robot", cameras={})

    def test_extra_kwargs_filtered_against_dataclass_fields(self):
        """Forwarded kwargs that the target dataclass doesn't declare are
        dropped silently, so callers can pass union-of-all known kwargs
        without breaking on simpler robots."""
        pytest.importorskip("lerobot.robots.so_follower")
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "so101_smoke"
        # `robot_ip` and `kp` are G1-only - must not raise on so101.
        cfg = hw._create_minimal_config(
            "so101_follower",
            cameras={},
            port="/dev/null",
            robot_ip="192.168.0.1",
            kp=[1.0] * 29,
            kd=[1.0] * 29,
        )
        assert cfg.port == "/dev/null"
        # Filtered out:
        assert not hasattr(cfg, "robot_ip")
        assert not hasattr(cfg, "kp")

    def test_cameras_dict_converted_to_opencv_config_with_defaults(self):
        """Camera dicts are converted to lerobot ``OpenCVCameraConfig`` objects.

        ``_create_minimal_config`` accepts a plain ``cameras`` dict (the shape an
        agent/tool passes) and must turn each ``opencv`` entry into a real
        ``lerobot.cameras.opencv.OpenCVCameraConfig`` on the resolved robot
        config. Only ``index_or_path`` is required; ``fps``/``width``/``height``/
        ``rotation``/``color_mode`` default. Pre-fix this conversion branch was
        never exercised (every config test passed ``cameras={}``), so a
        regression in the dict->config mapping would go unnoticed.
        """
        pytest.importorskip("lerobot.robots.so_follower")
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "so101_cam"
        cfg = hw._create_minimal_config(
            "so101_follower",
            cameras={
                "wrist": {"type": "opencv", "index_or_path": 0},
                "front": {
                    "index_or_path": "/dev/video2",
                    "fps": 60,
                    "width": 1280,
                    "height": 720,
                },
            },
            port="/dev/null",
        )
        assert set(cfg.cameras) == {"wrist", "front"}
        wrist = cfg.cameras["wrist"]
        assert isinstance(wrist, OpenCVCameraConfig)
        assert wrist.index_or_path == 0
        # Unspecified fields fall back to the documented defaults.
        assert (wrist.fps, wrist.width, wrist.height) == (30, 640, 480)
        # Explicit overrides are forwarded verbatim.
        front = cfg.cameras["front"]
        assert front.index_or_path == "/dev/video2"
        assert (front.fps, front.width, front.height) == (60, 1280, 720)

    def test_unsupported_camera_type_raises_value_error(self):
        """A non-opencv camera ``type`` is rejected with an actionable error.

        ``opencv`` is the only backend ``_create_minimal_config`` knows how to
        build; any other ``type`` (e.g. a typo or an unimplemented backend) must
        fail loudly at config-build time rather than be silently dropped, so the
        operator learns immediately the camera will not be wired up.
        """
        pytest.importorskip("lerobot.robots.so_follower")
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "so101_badcam"
        with pytest.raises(ValueError, match="Unsupported camera type: realsense"):
            hw._create_minimal_config(
                "so101_follower",
                cameras={"depth": {"type": "realsense", "index_or_path": 0}},
                port="/dev/null",
            )

    def test_mesh_attrs_set_before_initialize_robot_no_attribute_error_in_cleanup(self, caplog):
        """Pin the cleanup-AttributeError fix with the actual symptom.

        Pre-fix, when ``_initialize_robot`` raised partway through ``__init__``
        the secondary cleanup path ran ``cleanup()`` -> ``self.mesh`` and
        produced an ``AttributeError: 'Robot' object has no attribute
        'mesh'``. ``__del__`` swallows exceptions so the user never saw it
        directly, but ``cleanup()`` has its own ``except`` that calls
        ``logger.error(f"Cleanup error for {self.tool_name_str}: {e}")``.

        The fix moves ``self.mesh = None`` / ``self.peer_id = None`` to
        before ``_initialize_robot``, so that error log entry no longer
        appears. We assert on that absence; if a future refactor undoes
        the ordering swap (e.g. moves the mesh init back to its original
        spot), this test fails.
        """
        from unittest.mock import patch

        from strands_robots.hardware_robot import Robot as HwRobot

        with caplog.at_level("ERROR", logger="strands_robots.hardware_robot"):
            with patch.object(HwRobot, "_initialize_robot", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    HwRobot(tool_name="x", robot="so101_follower")

        # Pre-fix code logged either of:
        #   "Cleanup error for x: 'Robot' object has no attribute 'mesh'"
        # depending on whether peer_id or mesh was probed first. The fix
        # eliminates BOTH because both attrs are now initialised before
        # _initialize_robot runs.
        offenders = [
            r.message
            for r in caplog.records
            if "AttributeError" in r.message and "mesh" in r.message and "Cleanup error" in r.message
        ]
        assert not offenders, (
            f"cleanup() logged AttributeError for missing 'mesh': {offenders}. "
            "Did the mesh/peer_id init move back below _initialize_robot?"
        )

    def test_bi_so100_follower_resolves_via_discovery_shim(self):
        """Regression test for the lazy-import shim: ``bi_so100_follower``
        is registered by ``lerobot.robots.bi_so_follower`` (the directory
        name does NOT match the robot_type), so a hand-rolled
        ``import_module(f"lerobot.robots.{robot_type}")`` would miss it.
        Discovery via ``pkgutil.iter_modules`` walks the directory and
        catches it.

        This pins the discovery contract -- if a future cleanup PR drops
        the pkgutil walker (e.g. believing lerobot's __init__ has become
        eager), this test will fail before the breakage hits users.
        """
        pytest.importorskip("lerobot.robots.bi_so_follower")
        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

        # Cache is cleared by the class-level ``_clear_discovery_cache``
        # autouse fixture so the first call here is the FIRST call after
        # a fresh import -- exactly the scenario this test pins.
        _ensure_lerobot_robots_registered()

        # `hope_jr_arm` lives in `lerobot.robots.hope_jr` (the directory
        # name does NOT match the robot_type). Same with `hope_jr_hand`,
        # `lekiwi_client`, and `so100_follower`/`so101_follower` (both in
        # `so_follower`). A hand-rolled
        # `import_module(f"lerobot.robots.{robot_type}")` would miss all of
        # these. Discovery via pkgutil.iter_modules catches them.
        for robot_type, expected_pkg_prefix in [
            ("hope_jr_arm", "lerobot.robots.hope_jr"),
            ("hope_jr_hand", "lerobot.robots.hope_jr"),
            ("lekiwi_client", "lerobot.robots.lekiwi"),
            ("so101_follower", "lerobot.robots.so_follower"),
            ("so100_follower", "lerobot.robots.so_follower"),
        ]:
            try:
                ConfigClass = RobotConfig.get_choice_class(robot_type)
            except KeyError:
                pytest.fail(f"discovery missed {robot_type!r} (expected from {expected_pkg_prefix})")
            assert ConfigClass.__module__.startswith(expected_pkg_prefix), (
                f"Expected {robot_type} to come from {expected_pkg_prefix}, got {ConfigClass.__module__}"
            )

    def test_unsupported_type_error_has_no_chained_keyerror_traceback(self):
        """``raise ValueError(...) from None`` must suppress the chained
        KeyError traceback. Otherwise users see "During handling of the
        above exception (KeyError), another exception occurred" which
        leaks lerobot's draccus internals."""
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "x"
        try:
            hw._create_minimal_config("totally_made_up_robot", cameras={})
        except ValueError as e:
            assert e.__cause__ is None, f"ValueError carries chained cause {e.__cause__!r}; should be `from None`."
            assert e.__suppress_context__, (
                "ValueError does not suppress its context; users will see the internal KeyError traceback."
            )
        else:
            pytest.fail("expected ValueError")

    def test_robot_factory_real_mode_so101_runs_create_minimal_config_and_pins_id_override(self):
        """Public-API end-to-end pin for the ENTIRE `mode='real'` path
        the previous helper-only tests (which poke
        `_create_minimal_config` via `__new__`) cannot exercise.

        This test patches lerobot's own `make_robot_from_config`
        (inside `lerobot.robots.utils`, the only import site
        `_initialize_robot` uses) rather than `_initialize_robot`
        itself. That keeps `_create_minimal_config` -- the entire new
        discovery path this PR is about -- ON the call chain, so the
        test actually exercises:

          - the discovery walk,
          - the draccus `get_choice_class` lookup,
          - dataclass-field filtering of forwarded kwargs,
          - the `id=` override semantics advertised in the PR
            description.

        Pins:

          1. The factory dispatches `Robot('so101', mode='real')` to
             the hardware path (returns a HardwareRobot).
          2. The constructed lerobot config carries the user's
             `id="left_arm"` rather than the default `tool_name_str`.
             Per AGENTS.md > Review Learnings (#85) > "Pin regression
             tests for reviewed fixes", this advertised behaviour was
             previously unpinned: the prior version of this test
             patched `_initialize_robot` so `_create_minimal_config`
             never ran and the `id=` override was never asserted on.
        """
        pytest.importorskip("lerobot.robots.so_follower")

        from unittest.mock import MagicMock, patch

        from strands_robots import Robot

        # Sentinel returned by the patched lerobot factory: pretend it's
        # a built lerobot Robot instance so HardwareRobot.__init__
        # completes happily without any serial-port traffic.
        fake_lerobot_robot = MagicMock(name="lerobot_robot_instance")
        fake_lerobot_robot.name = "so_follower"
        fake_lerobot_robot.config = MagicMock()
        fake_lerobot_robot.config.cameras = {}

        # `make_robot_from_config` is imported function-locally inside
        # `_initialize_robot` from `lerobot.robots.utils`; patch it at
        # the source module so the patched callable is what
        # `_initialize_robot` resolves at call time.
        with patch(
            "lerobot.robots.utils.make_robot_from_config",
            return_value=fake_lerobot_robot,
        ) as make_cfg:
            r = Robot(
                "so101",
                mode="real",
                port="/dev/null",
                use_degrees=True,
                id="left_arm",
            )

        # Pin 1: factory dispatch shape.
        from strands_robots.hardware_robot import Robot as HwRobot

        assert isinstance(r, HwRobot)
        assert r.robot is fake_lerobot_robot
        assert hasattr(r, "mesh")
        assert hasattr(r, "peer_id")

        # Pin 2: `_create_minimal_config` actually ran and produced a
        # config with the user's `id=` override winning over the
        # default `tool_name_str`. The advertised behaviour in the PR
        # description ("Users may now override the lerobot `id` ...
        # Default remains the strands tool name") is now actually
        # asserted on.
        assert make_cfg.called, (
            "lerobot.robots.utils.make_robot_from_config was not invoked; "
            "_initialize_robot must have taken a different code path. "
            "The discovery + config-build chain is no longer covered."
        )
        cfg = make_cfg.call_args.args[0]
        assert cfg.id == "left_arm", (
            f"id= kwarg must win over tool_name_str; got cfg.id={cfg.id!r}. "
            "Did a refactor swap kwargs.get('id', self.tool_name_str) for "
            "self.tool_name_str unconditionally?"
        )
        assert cfg.port == "/dev/null"
        assert cfg.use_degrees is True

    def test_id_kwarg_overrides_tool_name_directly(self):
        """Helper-level pin for the same advertised `id=` override
        behaviour, without going through `Robot()`. Belt-and-suspenders
        with the end-to-end test above: if some refactor breaks the
        factory dispatch, the helper-level pin still catches a
        regression in `_create_minimal_config`'s `id=` handling.
        """
        pytest.importorskip("lerobot.robots.so_follower")

        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "default_tool_name"

        # With an explicit id=, the user's value must win.
        cfg = hw._create_minimal_config("so101_follower", cameras={}, port="/dev/null", id="left_arm")
        assert cfg.id == "left_arm"

        # With no id=, default to the strands tool name (the
        # backwards-compatible fallthrough advertised in the docstring).
        cfg2 = hw._create_minimal_config("so101_follower", cameras={}, port="/dev/null")
        assert cfg2.id == "default_tool_name"

    def test_walk_continues_when_driver_raises_oserror_at_import(self):
        """Pin AGENTS.md > Review Learnings (#86) > "Exception Clauses Must
        Be Narrow" / hardware-probing pattern. A driver subpackage whose
        ``__init__`` raises a non-``ImportError`` (e.g. ``OSError`` from a
        USB probe in ``unitree_sdk2py``) must not abort the entire
        ``_ensure_lerobot_robots_registered`` walk -- subsequent driver
        imports must still happen.

        Pre-fix code used ``except ImportError`` only; an ``OSError``
        would propagate out of ``importlib.import_module``, abort the
        for-loop, and silently skip every later driver. This is the same
        silent-degradation mode the surrounding comment claims to guard
        against.

        Pinning technique: capture the *call sequence* of
        ``importlib.import_module`` and assert that at least one
        ``lerobot.robots.*`` import was attempted AFTER the booby-trapped
        target raised. We deliberately do NOT inspect ``RobotConfig``
        state -- that registry (draccus ``ChoiceRegistry``) and
        ``sys.modules`` are both process-global, so prior tests in the
        session may have already populated them; the @functools.cache
        clear by the autouse fixture is not enough to neutralise that
        layer of state. The behavioural contract being pinned ("the loop
        kept going past the OSError") is directly observable as
        subsequent ``import_module`` calls regardless of whether those
        modules were already cached at the Python-import level.
        """
        pytest.importorskip("lerobot")

        from unittest.mock import patch

        from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

        real_import = importlib.import_module
        booby_target = "lerobot.robots.so_follower"
        import_calls: list[str] = []

        def fake_import(name, *args, **kwargs):
            import_calls.append(name)
            if name == booby_target:
                raise OSError("simulated USB probe failure during driver __init__")
            return real_import(name, *args, **kwargs)

        # Cache is cleared by the autouse fixture so the walk runs.
        with patch(
            "strands_robots.hardware_robot.importlib.import_module",
            side_effect=fake_import,
        ):
            # Must not raise -- the OSError from so_follower must be caught
            # and the walk must continue past it.
            _ensure_lerobot_robots_registered()

        # Sanity: the booby-trap actually fired.  If lerobot ever drops
        # ``so_follower`` upstream, this fails loudly with a setup-error
        # message rather than a misleading regression-error message.
        assert booby_target in import_calls, (
            f"Booby target {booby_target!r} was never attempted by the walk; "
            f"test setup is stale (lerobot may have renamed/dropped it). "
            f"Full call sequence: {import_calls}"
        )

        # The contract: at least one ``lerobot.robots.*`` driver import
        # was attempted AFTER the booby-trap raised.  Pre-fix code
        # (``except ImportError`` only) would re-raise the OSError,
        # break out of the for-loop, and ``import_calls`` would contain
        # NO ``lerobot.robots.*`` entries after ``booby_target``.
        booby_index = import_calls.index(booby_target)
        later_lerobot_drivers = [
            n for n in import_calls[booby_index + 1 :] if n.startswith("lerobot.robots.") and n != booby_target
        ]
        assert later_lerobot_drivers, (
            f"Walk aborted at {booby_target}; OSError was not caught by the "
            f"per-driver except clause. No further ``lerobot.robots.*`` "
            f"import attempts after the booby-trap. Full call sequence: "
            f"{import_calls}"
        )

    def test_unknown_kwarg_typo_raises_value_error(self):
        """Pin AGENTS.md > Review Learnings (#86) > "Reject silently-dropped
        kwargs". A user typo like ``prot=`` (instead of ``port=``) must
        surface as a clear ``ValueError`` at config-build time, not be
        silently dropped and surface hours later as a misleading
        connection failure.

        The cross-robot polymorphism case -- forwardable kwargs that
        belong to a sibling robot, e.g. ``kp`` to so101 -- is NOT what
        this test pins (that case is handled by
        ``test_extra_kwargs_filtered_against_dataclass_fields``). This
        test is specifically about kwargs that are unknown to the entire
        ``forwardable`` allowlist (typos, kwargs from a different
        subsystem entirely).
        """
        pytest.importorskip("lerobot.robots.so_follower")

        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "so101_typo"

        with pytest.raises(ValueError, match=r"Unknown kwarg.*prot"):
            hw._create_minimal_config(
                "so101_follower",
                cameras={},
                prot="/dev/ttyACM0",  # typo: should be `port`
            )

    def test_known_cross_robot_kwarg_is_silently_filtered_not_rejected(self):
        """Companion to ``test_unknown_kwarg_typo_raises_value_error``:
        a kwarg that IS in ``forwardable`` but does NOT belong to the
        target robot's dataclass (e.g. ``kp`` to so101) must NOT raise.
        That deliberate tolerance is the only reason
        ``Robot('so101', mode='real', kp=[...])`` doesn't blow up when a
        caller is iterating over a heterogeneous fleet -- the strict
        rejection only applies to kwargs no robot in the family knows.
        """
        pytest.importorskip("lerobot.robots.so_follower")

        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "so101_polymorphism"

        # Must not raise -- ``kp`` is a unitree_g1 kwarg, in ``forwardable``,
        # but not on so101_follower's dataclass. Silent filter is correct.
        cfg = hw._create_minimal_config(
            "so101_follower",
            cameras={},
            port="/dev/null",
            kp=[1.0] * 29,
        )
        assert cfg.port == "/dev/null"
        assert not hasattr(cfg, "kp")

    def test_dataclass_declared_field_accepted_without_forwardable_entry(self):
        """A kwarg that is NOT in the cross-robot forwardable tuple but IS
        declared on the target dataclass should be accepted and forwarded.
        This future-proofs new lerobot fields without requiring a
        strands_robots release to add them to the forwardable tuple."""
        pytest.importorskip("lerobot.robots.so_follower")

        import dataclasses

        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import (
            _FORWARDABLE_KWARGS,
            _ensure_lerobot_robots_registered,
        )
        from strands_robots.hardware_robot import (
            Robot as HwRobot,
        )

        _ensure_lerobot_robots_registered()
        ConfigClass = RobotConfig.get_choice_class("so101_follower")
        fields = list(dataclasses.fields(ConfigClass))
        real_fields = {f.name for f in fields}

        # Import from production code -- single source of truth (no drift).
        forwardable_set = set(_FORWARDABLE_KWARGS)
        dataclass_only_fields = real_fields - forwardable_set - {"id", "cameras"}
        if not dataclass_only_fields:
            pytest.skip("No dataclass-only fields found on SO101 config")

        target_field = sorted(dataclass_only_fields)[0]

        # Satisfy every required (no-default) field so construction fails ONLY
        # if the forwarding of ``target_field`` is broken -- never because of an
        # unrelated required field. lerobot #4142 added optional PID-gain fields
        # (position_p/i/d_coefficient) to SO-follower configs; when one of those
        # sorts first as ``target_field`` the config still declares a required
        # ``port``, so a fixed ``cameras``/``target_field`` kwarg pair alone left
        # ``port`` unset and the construction raised a ValueError unrelated to
        # the forwarding contract under test. Fill required fields dynamically so
        # the test stays pinned on forwarding as lerobot evolves its dataclass.
        required = {
            f.name: "placeholder"
            for f in fields
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
            and f.name not in {"cameras", "id"}
        }
        kwargs = {**required, target_field: "test_value"}

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "test_forward"
        # Pass the dataclass-only field -- should NOT raise ValueError, and the
        # value must round-trip verbatim (forwarded, not dropped).
        cfg = hw._create_minimal_config("so101_follower", cameras={}, **kwargs)
        assert getattr(cfg, target_field) == "test_value"


class TestMinimalConfigContractBranches:
    """Pin the kwarg-resolution contract branches of ``_create_minimal_config``
    that the happy-path tests never reach.

    ``_create_minimal_config`` resolves a lerobot ``RobotConfig`` subclass and
    decides, per kwarg, whether to forward / drop / reject it. Four documented
    branches were previously unexercised: future-proof forwarding of a
    dataclass-declared field outside the cross-robot allowlist (#294), the
    explicit-``id`` diagnostic when a config lacks an ``id`` field (#292), the
    fail-loud TypeError when lerobot hands back a non-dataclass config class,
    and the ValueError wrap when the resolved dataclass refuses construction.
    """

    def test_future_proof_forwards_dataclass_field_not_in_allowlist(self):
        """A kwarg outside the cross-robot allowlist but declared on the target
        dataclass is forwarded verbatim (#294 -- "zero strands_robots changes
        for new lerobot fields").

        ``earthrover_mini_plus`` declares ``sdk_url``, which is NOT in
        ``_FORWARDABLE_KWARGS``. Pre-existing coverage of this branch relied on
        SO101, whose dataclass has no allowlist-external field, so the test
        skipped and the branch went unverified. Using a robot that actually
        carries such a field pins the forward (value preserved, not dropped).
        """
        pytest.importorskip("lerobot.robots.earthrover_mini_plus")

        import dataclasses

        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import (
            _FORWARDABLE_KWARGS,
            _ensure_lerobot_robots_registered,
        )
        from strands_robots.hardware_robot import Robot as HwRobot

        _ensure_lerobot_robots_registered()
        cfg_cls = RobotConfig.get_choice_class("earthrover_mini_plus")
        fields = {f.name for f in dataclasses.fields(cfg_cls)}
        # Guard the premise: sdk_url must be a real, allowlist-external field.
        assert "sdk_url" in fields
        assert "sdk_url" not in set(_FORWARDABLE_KWARGS) | {"id", "cameras"}

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "earthrover_fwd"
        cfg = hw._create_minimal_config("earthrover_mini_plus", cameras={}, sdk_url="http://example:9000")
        assert cfg.sdk_url == "http://example:9000"

    def test_explicit_id_kwarg_warns_when_config_lacks_id_field(self, monkeypatch, caplog):
        """#292: every lerobot RobotConfig declares ``id`` today, so an explicit
        ``id=`` is normally consumed to namespace calibration files. If a future
        config drops the field, silently discarding the operator's ``id=`` would
        misnamespace calibration with no signal -- so the branch emits a named
        warning. Pin it with a synthetic id-less config.
        """
        import dataclasses
        import logging

        pytest.importorskip("lerobot")
        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import Robot as HwRobot

        @dataclasses.dataclass
        class _NoIdConfig:
            cameras: dict = dataclasses.field(default_factory=dict)

        monkeypatch.setattr(RobotConfig, "get_choice_class", lambda robot_type: _NoIdConfig)

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "no_id_robot"
        with caplog.at_level(logging.WARNING, logger="strands_robots.hardware_robot"):
            cfg = hw._create_minimal_config("fake_robot", cameras={}, id="left_arm")

        # Config still builds (the explicit id is simply not applied).
        assert isinstance(cfg, _NoIdConfig)
        warnings = [r.getMessage() for r in caplog.records if "does not declare an 'id'" in r.getMessage()]
        assert any("left_arm" in m for m in warnings), (
            f"expected a warning naming the unusable explicit id; got {warnings}"
        )

    def test_non_dataclass_config_class_raises_typeerror(self, monkeypatch):
        """If lerobot returns a non-dataclass config class, kwarg filtering via
        ``dataclasses.fields`` is impossible -- fail loud with a TypeError that
        names the offending class rather than blindly forwarding every kwarg
        (AGENTS.md > Key Conventions: no silent defaults on error).
        """
        pytest.importorskip("lerobot")
        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import Robot as HwRobot

        class _NotADataclass:
            pass

        monkeypatch.setattr(RobotConfig, "get_choice_class", lambda robot_type: _NotADataclass)

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "weird_robot"
        with pytest.raises(TypeError, match="non-dataclass config class"):
            hw._create_minimal_config("weird_robot", cameras={})

    def test_config_construction_failure_wrapped_as_value_error(self, monkeypatch):
        """A dataclass that refuses construction (e.g. a required field that no
        forwarded kwarg satisfies) must surface as a ValueError naming the
        config class and the assembled config_data, not a raw TypeError leaking
        the dataclass internals.
        """
        import dataclasses

        pytest.importorskip("lerobot")
        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import Robot as HwRobot

        @dataclasses.dataclass
        class _RequiresArg:
            required_field: int  # no default -> construction fails when absent
            id: str = "default"

        monkeypatch.setattr(RobotConfig, "get_choice_class", lambda robot_type: _RequiresArg)

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "needs_arg_robot"
        with pytest.raises(ValueError, match="Failed to construct _RequiresArg"):
            hw._create_minimal_config("needs_arg_robot", cameras={})


class TestHardwareConfigV040Followups:
    """v0.4.0 hardware_robot follow-up bundle (#389) - PR #276 review trail."""

    def test_cross_robot_kwarg_drop_emits_debug_signal(self, caplog):
        """#294/#297: dropping a forwardable kwarg the target dataclass does
        not declare is tolerated (polymorphism), but must now emit a DEBUG
        signal naming the kwarg so operators can audit why it had no effect."""
        import logging

        pytest.importorskip("lerobot.robots.so_follower")
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "so101_drop_signal"

        with caplog.at_level(logging.DEBUG, logger="strands_robots.hardware_robot"):
            cfg = hw._create_minimal_config(
                "so101_follower",
                cameras={},
                port="/dev/null",
                kp=[1.0] * 29,  # forwardable, not on so101 dataclass -> dropped
            )
        assert not hasattr(cfg, "kp")
        drop_msgs = [r.getMessage() for r in caplog.records if "dropping cross-robot kwarg" in r.getMessage()]
        assert any("'kp'" in m for m in drop_msgs), (
            f"expected a DEBUG signal naming the dropped 'kp' kwarg; got {drop_msgs}"
        )

    def test_register_third_party_plugins_exception_is_narrow(self):
        """#291: the register_third_party_plugins() guard must NOT be a bare
        except Exception. Pin the narrowed (ImportError, AttributeError,
        OSError) tuple by source inspection so the BLE001 pattern cannot
        silently return."""
        import inspect

        from strands_robots import hardware_robot

        src = inspect.getsource(hardware_robot._ensure_lerobot_robots_registered)
        assert "except (ImportError, AttributeError, OSError)" in src, (
            "register_third_party_plugins must be guarded by a narrow exception tuple (#291)"
        )
        assert "except Exception as exc:  # noqa: BLE001 -- third-party plugin" not in src, (
            "the bare except Exception on plugin registration must be gone (#291)"
        )

    def test_lerobot_extra_provides_aarch64_video_decoder(self):
        """#378: a `pip install strands-robots[lerobot]` on Thor/Jetson must get a
        working aarch64 video decoder (torchcodec), not the removed
        `torchvision.io.VideoReader`.

        The [lerobot] extra used to carry an explicit aarch64 torchcodec pin
        because lerobot 0.5.1's own marker excluded aarch64. lerobot 0.6 fixed
        that upstream (its `torchcodec>=0.11,<0.12; aarch64` marker pulls the
        torch-ABI-matched decoder), so the strands override was dropped and the
        guarantee now rides on the `lerobot>=0.6` floor (the extra declares
        >=0.6.1, which bucket streaming additionally needs). This pins that
        floor so a revert below 0.6 -- which would resurrect the
        missing-decoder bug without the removed override -- fails here."""
        import tomllib
        from pathlib import Path

        from packaging.requirements import Requirement
        from packaging.version import Version

        root = Path(__file__).resolve().parents[1]
        data = tomllib.load(open(root / "pyproject.toml", "rb"))
        lerobot_extra = data["project"]["optional-dependencies"]["lerobot"]
        lerobot_req = next(Requirement(d) for d in lerobot_extra if Requirement(d).name == "lerobot")
        # Assert the declared lower BOUND, not membership of one version: a
        # floor raise (e.g. to >=0.6.1 for bucket streaming) excludes 0.6.0
        # while the >=0.6 requirement this guards still holds a fortiori.
        lower = min(Version(s.version) for s in lerobot_req.specifier if s.operator == ">=")
        assert lower >= Version("0.6") and Version("0.5.9") not in lerobot_req.specifier, (
            f"[lerobot] extra must floor lerobot at >=0.6 so aarch64 gets torchcodec (#378); "
            f"got {lerobot_req.specifier}"
        )


class TestRobotNamePreservesUserInput:
    """Robot('h1') should register the robot under the user's input name,
    not the canonical resolved name (unitree_h1). The user should be able
    to use the name they passed in all subsequent API calls."""

    @pytest.fixture(autouse=True)
    def _mujoco(self):
        pytest.importorskip("mujoco")

    def test_alias_preserved_as_instance_name(self):
        """Robot('h1') registers robot as 'h1', not 'unitree_h1'."""
        from strands_robots import Robot

        sim = Robot("h1", mesh=False)
        try:
            robots = sim.list_robots()
            assert "h1" in robots, f"Expected 'h1' in {robots}"
            assert "unitree_h1" not in robots, f"Unexpected 'unitree_h1' in {robots}"
        finally:
            sim.destroy()

    def test_get_robot_state_works_with_user_name(self):
        """get_robot_state(robot_name='h1') succeeds after Robot('h1')."""
        from strands_robots import Robot

        sim = Robot("h1", mesh=False)
        try:
            state = sim.get_robot_state(robot_name="h1")
            assert state["status"] == "success"
        finally:
            sim.destroy()

    def test_robot_joint_names_works_with_user_name(self):
        """robot_joint_names('g1') returns joints after Robot('g1')."""
        from strands_robots import Robot

        sim = Robot("g1", mesh=False)
        try:
            joints = sim.robot_joint_names("g1")
            assert len(joints) > 0, "Expected non-empty joint list"
        finally:
            sim.destroy()

    def test_canonical_name_still_resolves_model(self):
        """The model is still loaded from the canonical asset directory."""
        from strands_robots import Robot

        sim = Robot("go2", mesh=False)
        try:
            robots = sim.list_robots()
            assert "go2" in robots
            joints = sim.robot_joint_names("go2")
            assert len(joints) == 12, f"go2 should have 12 joints, got {len(joints)}"
        finally:
            sim.destroy()

    def test_so100_unchanged(self):
        """Robot('so100') still works identically (name == canonical)."""
        from strands_robots import Robot

        sim = Robot("so100", mesh=False)
        try:
            assert "so100" in sim.list_robots()
            state = sim.get_robot_state(robot_name="so100")
            assert state["status"] == "success"
        finally:
            sim.destroy()

    def test_tool_name_uses_user_input(self):
        """The tool_name should reflect the user's input, not canonical."""
        from strands_robots import Robot

        sim = Robot("h1", mesh=False)
        try:
            assert sim.tool_name == "h1_sim"
        finally:
            sim.destroy()


def _stub_device_connect(monkeypatch, runtime):
    """Substitute the Device Connect integration module for the foreground loop.

    *runtime* is what ``init_device_connect_sync`` returns, or ``None`` to make
    importing the module fail the way an uninstalled ``[device-connect]`` extra
    does. Substituting the module is what makes the two branches selectable:
    unstubbed, the shipped code brings up a real Zenoh session where the extra
    happens to be installed and raises ``ImportError`` where it is not, so which
    branch a test exercised was decided by the environment rather than the test.
    """
    if runtime is None:
        monkeypatch.setitem(sys.modules, "strands_robots.device_connect", None)
        return
    module = types.ModuleType("strands_robots.device_connect")
    module.init_device_connect_sync = lambda robot, peer_id=None, peer_type=None: runtime
    monkeypatch.setitem(sys.modules, "strands_robots.device_connect", module)


def _drive_foreground(monkeypatch, capsys, peer_id="so100-test", runtime="runtime", mesh=None, instance=None):
    """Run the blocking foreground loop once and return its stdout.

    ``time.sleep`` is patched to raise ``KeyboardInterrupt`` (the operator's
    Ctrl+C) so the loop exits on the first tick, and ``os._exit`` is patched to
    a sentinel raise so the test process survives. Pass *instance* to inspect
    what the loop did to it.
    """
    if instance is None:
        instance = types.SimpleNamespace(_peer_id=peer_id, _peer_type="sim", mesh=mesh)
    _stub_device_connect(monkeypatch, runtime)
    monkeypatch.setattr("time.sleep", lambda _s: (_ for _ in ()).throw(KeyboardInterrupt()))

    class _ExitCalled(Exception):
        pass

    def _fake_exit(code):
        raise _ExitCalled()

    monkeypatch.setattr(os, "_exit", _fake_exit)

    with pytest.raises(_ExitCalled):
        _run_device_connect_foreground(instance)
    return capsys.readouterr().out


class TestRunDeviceConnectAsciiOutput:
    """Regression: the foreground ``.run()`` device-connect loop must print
    ASCII-only status lines.

    ``Robot(...).run()`` brings the device online and prints lifecycle messages
    straight to the operator's terminal. Those messages previously embedded
    emoji ("robot", "stop", "wave"), which violates the project's ASCII-only
    rule for logs and user-facing output (the same class of fix applied to
    serial_tool, pose_tool, and lerobot_camera). Non-ASCII bytes on a terminal
    or in a captured CI log can also raise UnicodeEncodeError under a non-UTF-8
    locale (``LC_ALL=C``). These tests pin the output to ASCII and exercise the
    otherwise-uncovered foreground loop without blocking.
    """

    def test_foreground_output_is_ascii_only(self, monkeypatch, capsys):
        out = _drive_foreground(monkeypatch, capsys)
        assert out, "foreground loop produced no output"
        offenders = [
            (i, ch, hex(ord(ch))) for i, line in enumerate(out.splitlines(), 1) for ch in line if ord(ch) > 0x7F
        ]
        assert not offenders, f"non-ASCII characters in run() output: {offenders}"
        # Output encodes cleanly under a non-UTF-8 locale (no UnicodeEncodeError).
        out.encode("ascii")

    def test_a_failed_bring_ups_report_is_ascii_too(self, monkeypatch, capsys):
        """The status line the failed branch prints is longer, and also ASCII."""
        out = _drive_foreground(monkeypatch, capsys, runtime=None)
        assert out.strip(), "foreground loop produced no output"
        out.encode("ascii")

    def test_foreground_output_reports_lifecycle(self, monkeypatch, capsys):
        """The ASCII messages still convey online + shutdown + peer id."""
        out = _drive_foreground(monkeypatch, capsys, peer_id="franka-7")
        assert "franka-7 is online" in out
        assert "Shutting down franka-7" in out
        assert "franka-7 stopped" in out


class TestTheStatusLineReportsWhatCameUp:
    """``run()`` prints one status line, and it has to match the runtime.

    The foreground runner keeps the process alive when a bring-up fails, which
    is deliberate. What it may not do is announce the device online on that
    path: the built-in mesh has already been stopped to make way for Device
    Connect, so a process whose bring-up failed is reachable over nothing at
    all, and the "is online" line was the last thing the operator was told.
    A warning was added beside it, but a warning next to a contradicting claim
    still leaves the claim.
    """

    def test_a_started_runtime_is_reported_online(self, monkeypatch, capsys):
        """The unchanged half: a runtime that came up is announced as before."""
        out = _drive_foreground(monkeypatch, capsys, peer_id="arm-1", runtime="runtime")

        assert "arm-1 is online. Ctrl+C to stop." in out
        assert "NOT online" not in out

    def test_a_bring_up_that_never_started_is_not_reported_online(self, monkeypatch, capsys):
        """An absent extra must not produce an "is online" line."""
        out = _drive_foreground(monkeypatch, capsys, peer_id="arm-1", runtime=None)

        assert "arm-1 is NOT online" in out
        assert "arm-1 is online" not in out
        assert "serves no transport" in out

    def test_the_failed_report_names_the_mesh_that_was_stopped_for_it(self, monkeypatch, capsys):
        """A robot that had a mesh has lost it, and the line says so.

        Which transport the process no longer has is the operator's next step,
        so the two cases are distinguished rather than collapsed into one
        message that is wrong for one of them.
        """
        stopped = []

        class _Mesh:
            def stop(self):
                stopped.append(True)

        out = _drive_foreground(monkeypatch, capsys, peer_id="arm-1", runtime=None, mesh=_Mesh())

        assert stopped == [True], "the mesh is still stopped for a bring-up that then failed"
        assert "The built-in mesh was stopped for it" in out

        without_mesh = _drive_foreground(monkeypatch, capsys, peer_id="arm-2", runtime=None, mesh=None)
        assert "The built-in mesh was stopped for it" not in without_mesh

    def test_an_absent_extra_is_reported_with_the_command_that_installs_it(self, monkeypatch, capsys, caplog):
        """The ImportError names an internal module; the warning names the extra."""
        with caplog.at_level("WARNING", logger="strands_robots.robot"):
            _drive_foreground(monkeypatch, capsys, runtime=None)

        failures = [r.getMessage() for r in caplog.records if "Device Connect init failed" in r.getMessage()]
        assert failures, [r.getMessage() for r in caplog.records]
        assert "strands-robots[device-connect]" in failures[0], failures[0]

    def test_a_broker_failure_is_reported_without_an_install_remedy(self, monkeypatch, capsys, caplog):
        """An unreachable broker is not fixed by installing anything."""
        module = types.ModuleType("strands_robots.device_connect")

        def _no_broker(robot, peer_id=None, peer_type=None):
            raise RuntimeError("no broker at tcp://127.0.0.1:7447")

        module.init_device_connect_sync = _no_broker
        monkeypatch.setitem(sys.modules, "strands_robots.device_connect", module)
        monkeypatch.setattr("time.sleep", lambda _s: (_ for _ in ()).throw(KeyboardInterrupt()))
        monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(RuntimeError("exited")))

        instance = types.SimpleNamespace(_peer_id="arm-1", _peer_type="sim", mesh=None)
        with caplog.at_level("WARNING", logger="strands_robots.robot"), pytest.raises(RuntimeError, match="exited"):
            _run_device_connect_foreground(instance)

        failures = [r.getMessage() for r in caplog.records if "Device Connect init failed" in r.getMessage()]
        assert failures, [r.getMessage() for r in caplog.records]
        assert "no broker" in failures[0]
        assert "pip install" not in failures[0], failures[0]
        assert "arm-1 is NOT online" in capsys.readouterr().out

    def test_built_in_mesh_is_stopped_before_device_connect(self, monkeypatch, capsys):
        """Device Connect supersedes the auto-started mesh in run() mode.

        A running built-in mesh must be stopped and detached so two Zenoh
        presence systems do not run in one process.
        """
        stopped = []

        class _Mesh:
            def stop(self):
                stopped.append(True)

        instance = types.SimpleNamespace(_peer_id="m1", _peer_type="sim", mesh=_Mesh())
        out = _drive_foreground(monkeypatch, capsys, instance=instance)

        assert stopped == [True], "built-in mesh was not stopped"
        assert instance.mesh is None, "mesh reference not detached"
        assert "m1 is online" in out, "the substituted runtime did come up"


class TestAttachDeviceConnectBindsRun:
    """``_attach_device_connect`` wires a callable ``.run()`` onto the instance."""

    def test_run_is_bound_and_callable(self):
        instance = types.SimpleNamespace()
        _attach_device_connect(instance, "so100", "sim", peer_id="p1")
        assert callable(instance.run)
        assert instance._peer_id == "p1"
        assert instance._peer_type == "sim"

    def test_real_mode_marks_peer_type_robot(self):
        instance = types.SimpleNamespace()
        _attach_device_connect(instance, "so100", "real", peer_id=None)
        assert instance._peer_type == "robot"
        # A peer id is synthesized from the canonical name when none is given.
        assert instance._peer_id.startswith("so100-")


class TestRobotFactoryErrorBranches:
    """Pin the error/enrichment branches of the ``Robot()`` factory.

    These exercise paths that the happy-path tests skip: a non-string mode
    reaching the final ``ValueError``, ``create_world`` reporting an in-band
    ``status=error`` (vs raising), the sim mesh-attach success assignment, and
    the best-effort mesh/Device-Connect enrichment on the hardware path.
    """

    def test_non_string_mode_raises_value_error(self):
        """A non-string mode passes through ``_normalize_mode`` unchanged and
        lands on the final ``ValueError`` rather than crashing earlier."""
        pytest.importorskip("mujoco")
        with pytest.raises(ValueError, match="Invalid mode"):
            Robot("so100", mode=123)

    def test_create_world_status_error_raises_with_message(self):
        """When ``create_world`` returns ``status=error`` (not an exception),
        the factory raises ``RuntimeError`` surfacing the backend message."""
        pytest.importorskip("mujoco")
        from unittest.mock import patch

        from strands_robots.simulation.mujoco.simulation import Simulation as SimImpl

        def fake_create_world(self, *args, **kwargs):
            return {"status": "error", "content": [{"text": "mjcf compile failed"}]}

        with patch.object(SimImpl, "create_world", fake_create_world):
            with pytest.raises(RuntimeError, match="Failed to create sim world.*mjcf compile failed"):
                Robot("so100")

    def test_create_world_status_error_empty_content_falls_back_to_repr(self):
        """An error result with no content still raises, using the raw result
        as the message rather than crashing on the empty content list."""
        pytest.importorskip("mujoco")
        from unittest.mock import patch

        from strands_robots.simulation.mujoco.simulation import Simulation as SimImpl

        def fake_create_world(self, *args, **kwargs):
            return {"status": "error", "content": []}

        with patch.object(SimImpl, "create_world", fake_create_world):
            with pytest.raises(RuntimeError, match="Failed to create sim world"):
                Robot("so100")

    def test_sim_mesh_success_assigns_mesh_and_peer_id(self, monkeypatch):
        """When ``init_mesh`` returns a mesh, the factory attaches it and copies
        the peer id onto the returned Simulation."""
        pytest.importorskip("mujoco")
        import types as _types

        fake_mesh = _types.SimpleNamespace(peer_id="sim-peer-xyz", stop=lambda: None)
        # init_mesh is imported inside the factory from strands_robots.mesh.
        monkeypatch.setattr("strands_robots.mesh.init_mesh", lambda *a, **k: fake_mesh)

        sim = Robot("so100", mode="sim")
        try:
            assert sim.mesh is fake_mesh
            assert sim.peer_id == "sim-peer-xyz"
        finally:
            sim.destroy()

    def test_real_mode_non_mujoco_backend_is_accepted(self):
        """In ``mode='real'`` a non-mujoco ``backend`` is ignored (hardware uses
        direct servo control), so construction proceeds to the hardware class."""
        from unittest.mock import MagicMock, patch

        sentinel = MagicMock()
        with (
            patch("strands_robots.robot.get_hardware_type", return_value="so100_follower"),
            patch("strands_robots.hardware_robot.Robot", return_value=sentinel) as mock_hw,
        ):
            result = Robot("so100", mode="real", backend="isaac")
        mock_hw.assert_called_once()
        assert result is sentinel

    def test_real_mode_mesh_failure_is_best_effort(self):
        """A mesh-init failure on the hardware path must be swallowed (logged),
        not propagated - the user asked for a hardware robot, mesh is enrichment."""
        from unittest.mock import MagicMock, patch

        sentinel = MagicMock()
        with (
            patch("strands_robots.robot.get_hardware_type", return_value="so100_follower"),
            patch("strands_robots.hardware_robot.Robot", return_value=sentinel),
            patch("strands_robots.mesh.init_mesh", side_effect=RuntimeError("zenoh router down")),
        ):
            # Must not raise despite the mesh failure.
            result = Robot("so100", mode="real")
        assert result is sentinel

    def test_device_connect_init_failure_keeps_process_alive(self, monkeypatch, capsys):
        """A failure inside ``init_device_connect_sync`` is logged and the
        foreground loop keeps running - it reports the failure rather than
        exiting, and the status line says the device is not online.

        The integration module is substituted rather than patched through, so
        this resolves without the ``[device-connect]`` extra installed.
        """
        instance = types.SimpleNamespace(_peer_id="so100-dc", _peer_type="sim", mesh=None)

        module = types.ModuleType("strands_robots.device_connect")
        module.init_device_connect_sync = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no broker"))
        monkeypatch.setitem(sys.modules, "strands_robots.device_connect", module)
        monkeypatch.setattr("time.sleep", lambda _s: (_ for _ in ()).throw(KeyboardInterrupt()))

        class _ExitCalled(Exception):
            pass

        monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(_ExitCalled()))

        with pytest.raises(_ExitCalled):
            _run_device_connect_foreground(instance)

        out = capsys.readouterr().out
        assert "so100-dc is NOT online" in out
        assert "Shutting down so100-dc" in out

    def test_auto_mode_without_hardware_resolves_to_sim(self, monkeypatch):
        """``mode='auto'`` with no env override and no detected hardware routes
        through ``_auto_detect_mode`` and lands on the safe sim default."""
        pytest.importorskip("mujoco")
        monkeypatch.delenv("STRANDS_ROBOT_MODE", raising=False)
        monkeypatch.setattr("strands_robots.robot._auto_detect_mode", lambda _c: "sim")

        sim = Robot("so100", mode="auto")
        try:
            from strands_robots.simulation import Simulation

            assert isinstance(sim, Simulation)
        finally:
            sim.destroy()

    def test_sim_mesh_failure_is_best_effort(self, monkeypatch):
        """A mesh-init failure on the sim path is swallowed; the Simulation is
        still returned (mesh is enrichment, not a precondition)."""
        pytest.importorskip("mujoco")
        monkeypatch.setattr(
            "strands_robots.mesh.init_mesh",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("zenoh router down")),
        )

        sim = Robot("so100", mode="sim")
        try:
            assert sim is not None
            # The failed mesh attach leaves no mesh wired on the instance.
            assert getattr(sim, "mesh", None) is None
        finally:
            sim.destroy()

    def test_real_mode_mesh_success_assigns_mesh_and_peer_id(self):
        """When ``init_mesh`` returns a mesh on the hardware path, the factory
        attaches it and copies the peer id onto the hardware instance."""
        import types as _types
        from unittest.mock import MagicMock, patch

        sentinel = MagicMock()
        fake_mesh = _types.SimpleNamespace(peer_id="hw-peer-7", stop=lambda: None)
        with (
            patch("strands_robots.robot.get_hardware_type", return_value="so100_follower"),
            patch("strands_robots.hardware_robot.Robot", return_value=sentinel),
            patch("strands_robots.mesh.init_mesh", return_value=fake_mesh),
        ):
            result = Robot("so100", mode="real")
        assert result.mesh is fake_mesh
        assert result.peer_id == "hw-peer-7"


class TestHardwareSendActionTeleopContract:
    """Pin the TeleopMixin host contract on hardware Robot.

    Regression: ``TeleopMixin._teleop_loop`` calls ``self.send_action(action,
    robot_name=...)``. The MuJoCo Simulation host defines ``send_action`` but
    the hardware ``Robot`` originally did NOT (it only wrapped an inner lerobot
    robot whose ``send_action`` is ``self.robot.send_action``). So
    ``Robot('so101', mode='real').attach_teleop(...).teleoperate()`` would have
    raised ``AttributeError`` mid-loop. This pins that hardware Robot now
    satisfies the contract directly: ensures connect, delegates to the inner
    robot, ignores ``robot_name`` (single device), and returns a status dict
    (never raises) so the hot teleop loop stays alive.

    Per AGENTS.md > Review Learnings (#85): pin regression tests for fixes.
    """

    def _make_hw_robot(self, *, connected: bool):
        from unittest.mock import MagicMock, patch

        from strands_robots import Robot
        from strands_robots.hardware_robot import Robot as HwRobot

        inner = MagicMock(name="lerobot_robot_instance")
        inner.name = "so_follower"
        inner.config = MagicMock()
        inner.config.cameras = {}
        inner.is_connected = connected
        inner.send_action = MagicMock(return_value=None)
        inner.connect = MagicMock(name="connect")

        with patch(
            "lerobot.robots.utils.make_robot_from_config",
            return_value=inner,
        ):
            r = Robot("so101", mode="real", port="/dev/null")
        assert isinstance(r, HwRobot)
        return r, inner

    def test_send_action_delegates_when_connected(self):
        pytest.importorskip("lerobot.robots.so_follower")
        r, inner = self._make_hw_robot(connected=True)
        res = r.send_action({"shoulder.pos": 0.5}, robot_name="ignored")
        assert res["status"] == "success"
        inner.send_action.assert_called_once_with({"shoulder.pos": 0.5})
        # Already connected -> no lazy connect call.
        inner.connect.assert_not_called()

    def test_send_action_lazy_connects_first(self):
        pytest.importorskip("lerobot.robots.so_follower")
        r, inner = self._make_hw_robot(connected=False)
        res = r.send_action({"j.pos": 1.0})
        assert res["status"] == "success"
        inner.connect.assert_called_once_with(False)  # calibrate=False
        inner.send_action.assert_called_once_with({"j.pos": 1.0})

    def test_send_action_returns_error_status_never_raises(self):
        pytest.importorskip("lerobot.robots.so_follower")
        r, inner = self._make_hw_robot(connected=True)
        inner.send_action.side_effect = RuntimeError("motor stalled")
        res = r.send_action({"j.pos": 1.0})  # must NOT raise
        assert res["status"] == "error"
        assert "motor stalled" in res["content"][0]["text"]

    def test_hardware_robot_satisfies_teleop_mixin(self):
        """Hardware Robot is a TeleopMixin and exposes the loop entrypoints."""
        pytest.importorskip("lerobot.robots.so_follower")
        from strands_robots.teleop_mixin import TeleopMixin

        r, _ = self._make_hw_robot(connected=True)
        assert isinstance(r, TeleopMixin)
        for m in ("attach_teleop", "teleoperate", "stop_teleoperate", "send_action"):
            assert callable(getattr(r, m, None)), f"missing {m}"


class TestRealModeAccountsForEverySpawnParameter:
    """In ``mode="real"`` every declared parameter either arrives or says it did not.

    ``Robot()`` is one front door onto two destinations, and its signature is the
    union of what they accept. Two parameters already handle the mismatch: a
    real-only ``cameras`` supplied in ``mode="sim"`` is refused with the route
    that does work, and a sim-only ``backend`` supplied in ``mode="real"`` is
    reported at debug level - the hardware path has no backend, so ignoring it is
    right, and saying so is what keeps it from looking honoured.

    The five remaining parameters the sim path hands to ``add_robot`` -
    ``urdf_path``, ``data_config``, ``position``, ``orientation``, ``keyframe`` -
    had neither. They are bound by name, so they never reach the hardware
    constructor's ``**kwargs``, and the real branch never read them: a caller
    asking a physical arm to come up at its ``home`` keyframe, or at a base
    position, got a working robot that discarded the request with nothing emitted
    at any log level.

    ``data_config`` was the one with a consequence beyond silence.
    ``hardware_robot.Robot`` declares it and carries it into the ``policy_config``
    a policy is built with, so the schema selector its own docstring tells a
    multi-camera user to "specify explicitly" was dropped between the two, and
    the policy came up on its default embodiment. It is forwarded now. The four
    that describe a pose in a simulated world cannot be forwarded anywhere - a
    physical arm is already where it is - so they take the mechanism ``backend``
    already uses.

    The contract graded here is the union of both: a parameter the sim path
    forwards must, in ``mode="real"``, either have its value reach the hardware
    class or be named in a debug record. That is derived from the ``add_robot``
    call itself rather than from a list here, so a sixth spawn parameter is held
    to it on arrival.
    """

    # Sentinels per parameter. ``name`` has to be a registry name because the
    # factory resolves it before either branch; the rest are only carried.
    SENTINELS: dict[str, object] = {
        "name": "so100",
        "urdf_path": "/tmp/sentinel-model.xml",
        "data_config": "so100_dualcam",
        "position": [1.0, 2.0, 0.5],
        "orientation": [0.0, 1.0, 0.0, 0.0],
        "keyframe": "home",
    }

    @staticmethod
    def _sim_forwarded_params() -> list[str]:
        """Parameters the sim branch hands to ``add_robot``, read from the source.

        The graded set is the ``add_robot`` call's own keyword arguments whose
        value is a bare factory parameter, so it tracks that call instead of a
        second copy of it maintained here.
        """
        import ast
        import inspect

        from strands_robots import robot as robot_mod

        tree = ast.parse(inspect.getsource(robot_mod))
        impl = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "Robot"][-1]
        declared = {a.arg for a in impl.args.args} | {a.arg for a in impl.args.kwonlyargs}
        for node in ast.walk(impl):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_robot":
                found = [kw.arg for kw in node.keywords if kw.arg in declared]
                if found:
                    return found
        raise AssertionError("no sim-path add_robot call found in the Robot() implementation")

    def _call_real(self, caplog, **kwargs):
        """Build in ``mode="real"``; return (hardware kwargs, debug records).

        ``caplog`` is cleared per call so a record left by an earlier build in the
        same test cannot answer for this one.
        """
        import logging
        from unittest.mock import MagicMock, patch

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="strands_robots.robot"):
            with (
                patch("strands_robots.robot.get_hardware_type", return_value="so100_follower"),
                patch("strands_robots.hardware_robot.Robot", return_value=MagicMock()) as mock_hw,
                patch("strands_robots.mesh.init_mesh", return_value=None),
            ):
                Robot(kwargs.pop("name", "so100"), mode="real", **kwargs)
        return mock_hw.call_args.kwargs, [r.getMessage() for r in caplog.records]

    def test_every_parameter_the_sim_path_forwards_is_carried_or_reported(self, caplog):
        """The headline contract, graded over the parameters the sim path forwards."""
        graded = self._sim_forwarded_params()
        assert len(graded) >= 5, f"premise: the sim path should forward several spawn parameters, got {graded}"
        assert set(graded) <= set(self.SENTINELS), (
            f"premise: no sentinel for {sorted(set(graded) - set(self.SENTINELS))}; add one so it is graded"
        )

        unaccounted = {}
        for param in graded:
            value = self.SENTINELS[param]
            hw_kwargs, records = self._call_real(caplog, **{param: value})
            carried = any(v == value for v in hw_kwargs.values())
            reported = any(param in m and "ignored in mode='real'" in m for m in records)
            if not (carried or reported):
                unaccounted[param] = f"carried={carried} reported={reported}"
        assert not unaccounted, (
            "mode='real' neither carried nor reported these parameters, so a caller who "
            f"supplied one got a working robot that discarded it in silence: {unaccounted}"
        )

    @pytest.mark.parametrize("param", ["urdf_path", "position", "orientation", "keyframe"])
    def test_a_supplied_spawn_pose_parameter_is_reported(self, param, caplog):
        """Each pose parameter the hardware path cannot honour names itself."""
        value = self.SENTINELS[param]
        _, records = self._call_real(caplog, **{param: value})
        named = [m for m in records if param in m and "ignored in mode='real'" in m]
        assert named, f"{param}={value!r} was ignored with nothing said; records: {records}"

    def test_a_falsy_but_supplied_keyframe_index_is_reported(self, caplog):
        """``keyframe=0`` is a valid keyframe index, so supply is read by identity.

        A truthiness test would drop it, and index 0 is the first keyframe a model
        declares - the value a caller reaches for when naming one by position.
        """
        _, records = self._call_real(caplog, keyframe=0)
        assert [m for m in records if "keyframe=0" in m and "ignored in mode='real'" in m], (
            f"keyframe=0 was supplied and ignored without a word; records: {records}"
        )

    def test_supplying_several_reports_all_of_them_once(self, caplog):
        """One record names every ignored parameter, rather than one record each."""
        _, records = self._call_real(
            caplog,
            urdf_path="/tmp/a.xml",
            position=[1.0, 2.0, 3.0],
            orientation=[1.0, 0.0, 0.0, 0.0],
            keyframe=2,
        )
        spawn = [m for m in records if "spawn pose applies to a simulated world" in m]
        assert len(spawn) == 1, f"expected exactly one spawn-pose record, got {spawn}"
        for param in ("urdf_path", "position", "orientation", "keyframe"):
            assert param in spawn[0], f"{param} missing from {spawn[0]!r}"

    def test_supplying_none_of_them_reports_nothing(self, caplog):
        """A bare hardware build stays quiet - the report is about a supplied value."""
        _, records = self._call_real(caplog)
        assert not [m for m in records if "spawn pose applies to a simulated world" in m], (
            f"nothing was supplied, so nothing should be reported; records: {records}"
        )

    def test_data_config_reaches_the_hardware_class(self, caplog):
        """The schema selector arrives, because the hardware class reads it.

        ``hardware_robot.Robot`` declares ``data_config`` and puts it into the
        ``policy_config`` a policy is built with, so dropping it here selected the
        policy's default embodiment for a caller who named one.
        """
        hw_kwargs, _ = self._call_real(caplog, data_config="so100_dualcam")
        assert hw_kwargs.get("data_config") == "so100_dualcam", (
            f"data_config did not reach the hardware class; got kwargs {sorted(hw_kwargs)}"
        )

    def test_data_config_is_not_reported_as_ignored(self, caplog):
        """It is forwarded, so it must not also claim to have been dropped."""
        _, records = self._call_real(caplog, data_config="so100_dualcam")
        assert not [m for m in records if "data_config" in m and "ignored in mode='real'" in m], (
            f"data_config is forwarded and must not be reported as ignored; records: {records}"
        )

    def test_an_unnamed_data_config_forwards_the_hardware_default(self, caplog):
        """Omitting it forwards ``None`` - the hardware class's own default.

        The sim path defaults it to the canonical robot name; applying that here
        would change what every existing hardware caller's policy is built with,
        so the value is passed verbatim instead.
        """
        hw_kwargs, _ = self._call_real(caplog)
        assert hw_kwargs.get("data_config", "<absent>") is None, (
            f"an unnamed data_config must forward None, got {hw_kwargs.get('data_config', '<absent>')!r}"
        )

    def test_the_sibling_backend_report_is_unchanged(self, caplog):
        """The parameter that already had this mechanism keeps it."""
        _, records = self._call_real(caplog, backend="isaac")
        assert [m for m in records if "backend='isaac'" in m and "ignored in mode='real'" in m], (
            f"the backend report regressed; records: {records}"
        )

    def test_a_real_only_parameter_is_still_refused_in_sim(self):
        """The opposite direction is untouched: ``cameras`` still refuses in sim.

        ``cameras`` is refused rather than reported because sim cameras exist and
        are added another way, so the caller is pointed at the route that works.
        Reporting it instead would be a regression.
        """
        with pytest.raises(ValueError, match="cameras= is only supported in mode='real'"):
            Robot("so100", mode="sim", cameras={"wrist": {"type": "opencv", "index_or_path": 0}})

    @pytest.mark.parametrize("param", ["urdf_path", "position", "orientation", "keyframe"])
    def test_each_ignored_parameter_documents_its_scope(self, param):
        """The docstring says so too - the rule is stated as documentation and a log.

        A reader deciding whether ``keyframe`` means anything on hardware reads the
        parameter's own entry, not a debug stream they would have to enable first.
        """
        import inspect
        import re

        doc = inspect.getdoc(Robot) or ""
        lines = doc.splitlines()
        start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
        entries: dict[str, str] = {}
        current: str | None = None
        indent: int | None = None
        for line in lines[start + 1 :]:
            if line.strip() in ("Returns:", "Raises:", "Example:", "Examples:", "Note:"):
                break
            match = re.match(r"^(\s+)(\*{0,2}\w+):\s?(.*)$", line)
            width = len(line) - len(line.lstrip()) if line.strip() else None
            if match and (indent is None or width == indent):
                indent = len(match.group(1))
                current = match.group(2)
                entries[current] = match.group(3)
            elif current is not None:
                entries[current] += " " + line.strip()
        assert param in entries, f"premise: no Args entry parsed for {param}; parsed {sorted(entries)}"
        text = " ".join(entries[param].split())
        assert 'mode="real"' in text, f"the {param} entry does not say what mode='real' does with it: {text!r}"
