"""Unit tests for the vendored Isaac Sim backend.

These tests deliberately do NOT require NVIDIA Isaac Sim to be installed.
They pin the parts that must work on any host (CI, dev box, GPU node
without an Omniverse install):

  * ``IsaacConfig`` validation and env-var resolution.
  * The lazy-import contract: importing ``strands_robots.simulation`` (or
    the ``isaac`` subpackage) must never trigger an ``omni`` / ``isaacsim``
    import (AGENTS.md lazy-import rule + issue #1145 acceptance criteria).
  * Factory registration + aliasing of the ``isaac`` backend.
  * ``IsaacSimulation`` is a concrete ``SimEngine`` subclass that can be
    constructed, and degrades to a structured error (never raises, never a
    silent zero-action default) when Isaac Sim is absent.
  * The zero-dependency procedural builders and description-file loaders.

The GPU-only rollout / rendering paths live in ``tests_integ`` and require
real Isaac Sim + a CUDA device.
"""

from __future__ import annotations

import sys

import pytest


class TestLazyImport:
    """Importing the sim package must not pull in Isaac Sim / Omniverse."""

    def test_importing_simulation_does_not_import_isaac(self):
        # Fresh interpreter so a previously-imported omni from another test
        # cannot mask an eager import regression here.
        import subprocess

        code = (
            "import sys, importlib\n"
            "importlib.import_module('strands_robots.simulation')\n"
            "bad = [m for m in sys.modules if m.startswith('omni') or m.startswith('isaacsim')]\n"
            "assert not bad, f'eager Isaac import: {bad}'\n"
            "assert 'strands_robots.simulation.isaac.simulation' not in sys.modules\n"
            "print('ok')\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

    def test_importing_isaac_subpackage_does_not_import_isaac(self):
        import subprocess

        code = (
            "import sys, importlib\n"
            "importlib.import_module('strands_robots.simulation.isaac')\n"
            "bad = [m for m in sys.modules if m.startswith('omni') or m.startswith('isaacsim')]\n"
            "assert not bad, f'eager Isaac import: {bad}'\n"
            "print('ok')\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

    def test_config_and_class_import_without_isaac(self):
        # Importing the config module and the simulation class (which defines
        # IsaacSimulation) must not import omni/isaacsim - all heavy imports
        # are inside methods, not at module scope.
        from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: F401
        from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: F401

        assert not any(m.startswith("omni") or m.startswith("isaacsim") for m in sys.modules)


class TestIsaacConfig:
    def test_defaults(self):
        from strands_robots.simulation.isaac.config import IsaacConfig

        c = IsaacConfig()
        assert c.num_envs == 1
        assert c.device == "cuda:0"
        assert c.headless is True
        assert c.render_mode == "headless"
        assert c.gravity == (0.0, 0.0, -9.81)

    def test_rejects_unknown_render_mode(self):
        from strands_robots.simulation.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="render_mode"):
            IsaacConfig(render_mode="bogus")

    def test_rejects_non_cuda_device(self):
        from strands_robots.simulation.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="CUDA device"):
            IsaacConfig(device="cpu")

    def test_rejects_zero_envs(self):
        from strands_robots.simulation.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="num_envs"):
            IsaacConfig(num_envs=0)

    def test_rejects_nonpositive_physics_dt(self):
        from strands_robots.simulation.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="physics_dt"):
            IsaacConfig(physics_dt=0.0)

    def test_env_headless_override(self, monkeypatch):
        from strands_robots.simulation.isaac.config import IsaacConfig

        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "false")
        assert IsaacConfig().headless is False
        monkeypatch.setenv("STRANDS_ISAAC_HEADLESS", "1")
        assert IsaacConfig().headless is True

    def test_env_rtx_pathtracing_override(self, monkeypatch):
        from strands_robots.simulation.isaac.config import IsaacConfig

        monkeypatch.setenv("STRANDS_ISAAC_RTX_PATHTRACING", "yes")
        assert IsaacConfig().render_mode == "rtx_pathtracing"

    def test_env_nucleus_url_resolution(self, monkeypatch):
        from strands_robots.simulation.isaac.config import IsaacConfig

        monkeypatch.setenv("STRANDS_ISAAC_NUCLEUS_URL", "omniverse://example")
        assert IsaacConfig().nucleus_url == "omniverse://example"

    def test_from_kwargs_rejects_unknown_key(self):
        from strands_robots.simulation.isaac.config import IsaacConfig

        with pytest.raises(TypeError):
            IsaacConfig.from_kwargs(headles=False)  # typo


class TestFactoryRegistration:
    def test_isaac_registered_as_builtin(self):
        from strands_robots.simulation import list_backends

        assert "isaac" in list_backends()

    def test_isaac_aliases(self):
        from strands_robots.simulation import list_backends

        backends = list_backends()
        for alias in ("isaac_sim", "isaacsim", "nvidia"):
            assert alias in backends

    def test_aliases_resolve_to_isaac(self):
        from strands_robots.simulation.factory import _resolve_name

        for alias in ("isaac_sim", "isaacsim", "nvidia"):
            assert _resolve_name(alias) == "isaac"

    def test_import_backend_class_is_isaac_simulation(self):
        from strands_robots.simulation.base import SimEngine
        from strands_robots.simulation.factory import _import_backend_class

        cls = _import_backend_class("isaac")
        assert cls.__name__ == "IsaacSimulation"
        assert issubclass(cls, SimEngine)


class TestIsaacSimulationConstruction:
    def test_is_subclass_of_simengine(self):
        from strands_robots.simulation.base import SimEngine
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        assert issubclass(IsaacSimulation, SimEngine)

    def test_construct_via_factory(self):
        from strands_robots.simulation import create_simulation

        sim = create_simulation("isaac", num_envs=1, headless=True)
        assert type(sim).__name__ == "IsaacSimulation"
        assert sim.list_robots() == []

    def test_unknown_kwarg_rejected_eagerly(self):
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        with pytest.raises(TypeError):
            IsaacSimulation(headles=False)  # typo -> not a silent default

    def test_is_available_returns_tuple(self):
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        ok, reason = IsaacSimulation.is_available()
        assert isinstance(ok, bool)
        # On a host without Isaac Sim, reason is a non-empty install hint.
        if not ok:
            assert reason and "Isaac Sim" in reason

    def test_create_world_without_isaac_is_structured_error(self):
        # Acceptance: no silent zero-valued default, no bare exception -
        # a structured {"status": "error", ...} dict per the SimEngine
        # error-handling contract (AGENTS.md). Skipped if Isaac IS present.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        ok, _ = IsaacSimulation.is_available()
        if ok:
            pytest.skip("Isaac Sim is installed; error path not exercised here.")
        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.create_world()
        assert result["status"] == "error"
        assert result["content"] and "text" in result["content"][0]


class TestProceduralBuilders:
    def test_list_procedural_robots(self):
        from strands_robots.simulation.isaac.procedural import list_procedural_robots

        names = list_procedural_robots()
        assert {"so100", "panda", "unitree_g1"} <= set(names)

    def test_get_procedural_robot_so100(self):
        from strands_robots.simulation.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("so100")
        assert robot is not None
        assert robot.num_joints > 0
        assert len(robot.joint_names) == robot.num_joints

    def test_get_procedural_robot_unknown_is_none(self):
        from strands_robots.simulation.isaac.procedural import get_procedural_robot

        assert get_procedural_robot("does_not_exist") is None


class TestLoaders:
    def test_load_urdf_missing_file_raises(self):
        from strands_robots.simulation.isaac.loaders import load_urdf

        with pytest.raises(FileNotFoundError):
            load_urdf("/nonexistent/robot.urdf")

    def test_load_urdf_parses_minimal_tree(self, tmp_path):
        from strands_robots.simulation.isaac.loaders import load_urdf

        urdf = tmp_path / "arm.urdf"
        urdf.write_text(
            """<?xml version="1.0"?>
<robot name="mini">
  <link name="base"/>
  <link name="link1"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0"/>
  </joint>
</robot>
"""
        )
        robot = load_urdf(str(urdf))
        assert robot.num_joints == 1
        assert "j1" in robot.joint_names

    def test_load_urdf_empty_document_raises(self, tmp_path):
        from strands_robots.simulation.isaac.loaders import load_urdf

        urdf = tmp_path / "empty.urdf"
        urdf.write_text('<?xml version="1.0"?><robot name="empty"></robot>')
        with pytest.raises(ValueError):
            load_urdf(str(urdf))


class TestInstallMetadata:
    def test_pip_extra_points_at_in_tree_extra(self):
        from strands_robots.simulation.isaac import _install

        assert _install.PIP_EXTRA == "pip install 'strands-robots[sim-isaac]'"

    def test_not_importable_reason_mentions_install_paths(self):
        from strands_robots.simulation.isaac import _install

        reason = _install.not_importable_reason()
        assert "Omniverse" in reason
        assert _install.ISAAC_SIM_DOCKER_IMAGE in reason
