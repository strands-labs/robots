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

import concurrent.futures
import sys
import threading

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


class TestPackageLazyExport:
    """The package-level public export surface resolves via PEP 562 ``__getattr__``.

    ``strands_robots/simulation/isaac/__init__.py`` documents

        from strands_robots.simulation.isaac import IsaacSimulation, IsaacConfig

    as the public entry point and re-exports both names lazily through
    ``__getattr__`` (so importing the subpackage never pulls omni/isaacsim).
    The other tests reach the classes through their defining submodules
    (``...isaac.config`` / ``...isaac.simulation``), which bypasses the
    package accessor - these pin the accessor itself so a regression in the
    lazy re-export (wrong name check, wrong target) is caught.
    """

    def test_isaac_config_resolves_through_package_getattr(self):
        import strands_robots.simulation.isaac as isaac_pkg
        from strands_robots.simulation.isaac.config import IsaacConfig as ConfigViaSubmodule

        # Attribute access on the package triggers __getattr__ -> _lazy_isaac_config.
        assert isaac_pkg.IsaacConfig is ConfigViaSubmodule
        assert isaac_pkg.IsaacConfig.__module__ == "strands_robots.simulation.isaac.config"

    def test_isaac_simulation_resolves_through_package_getattr(self):
        import strands_robots.simulation.isaac as isaac_pkg
        from strands_robots.simulation.isaac.simulation import IsaacSimulation as SimViaSubmodule

        # Attribute access triggers __getattr__ -> _lazy_isaac_simulation; still
        # no omni/isaacsim import (heavy imports live inside methods).
        assert isaac_pkg.IsaacSimulation is SimViaSubmodule
        assert not any(m.startswith("omni") or m.startswith("isaacsim") for m in sys.modules)

    def test_public_names_match_all(self):
        import strands_robots.simulation.isaac as isaac_pkg

        # Everything promised by __all__ is resolvable through the package.
        assert set(isaac_pkg.__all__) == {"IsaacSimulation", "IsaacConfig", "IsaacDeltaEEFController"}
        for name in isaac_pkg.__all__:
            assert getattr(isaac_pkg, name) is not None

    def test_unknown_attribute_raises_attribute_error(self):
        import strands_robots.simulation.isaac as isaac_pkg

        with pytest.raises(AttributeError, match="no attribute 'NotAName'"):
            _ = isaac_pkg.NotAName


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

    def test_rejects_nonpositive_rendering_dt(self):
        from strands_robots.simulation.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="rendering_dt"):
            IsaacConfig(rendering_dt=0.0)

    def test_rejects_nonpositive_camera_dimensions(self):
        from strands_robots.simulation.isaac.config import IsaacConfig

        with pytest.raises(ValueError, match="camera dimensions"):
            IsaacConfig(camera_width=0)
        with pytest.raises(ValueError, match="camera dimensions"):
            IsaacConfig(camera_height=-1)

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

    def test_create_world_signature_parity_with_base(self):
        # The base SimEngine.create_world abstractmethod declares
        # ``terrain`` / ``difficulty`` (the terrain-curriculum contract). Every
        # backend override must accept them so a caller / the tool router can
        # pass them uniformly; a narrower override raises a bare TypeError on a
        # documented parameter instead of the contract's actionable error.
        import inspect

        from strands_robots.simulation.base import SimEngine
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        base = set(inspect.signature(SimEngine.create_world).parameters)
        override = set(inspect.signature(IsaacSimulation.create_world).parameters)
        assert {"terrain", "difficulty"} <= override, f"Isaac create_world drops base params: missing {base - override}"

    def test_create_world_terrain_rejected_with_actionable_error(self):
        # Base contract (SimEngine.create_world docstring): a backend without
        # heightfield support rejects a non-None ``terrain`` with an actionable
        # error - NOT a bare TypeError, NOT a silent ignore. The rejection is
        # exercised before Isaac Sim boots, so it holds on any host.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.create_world(terrain="rough")
        assert result["status"] == "error"
        text = result["content"][0]["text"].lower()
        assert "terrain" in text and "mujoco" in text, text

    def test_create_world_difficulty_accepted_for_signature_parity(self):
        # ``difficulty`` is accepted (signature parity with the base contract);
        # passing it must not raise TypeError. A default ``difficulty=1.0`` is a
        # no-op that falls through to the world build (here the structured
        # Isaac-Sim-absent error on a host without Isaac Sim), not a crash.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.create_world(difficulty=1.0)
        assert result["status"] == "error"  # Isaac Sim absent -> structured error
        assert "difficulty" not in result["content"][0]["text"].lower()

    def test_create_world_difficulty_without_terrain_rejected(self):
        # Base create_world contract: a non-default ``difficulty`` with no
        # ``terrain`` is rejected with an actionable error rather than silently
        # having no effect. Isaac has no heightfield terrain for difficulty to
        # scale, so any != 1.0 value is doubly inert here; the reject fires
        # before Isaac Sim boots, so it holds on any host (and takes precedence
        # over the Isaac-Sim-absent error). Was a status=success / silent no-op
        # on a host with Isaac Sim before this contract landed.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.create_world(difficulty=2.0)
        assert result["status"] == "error"
        text = result["content"][0]["text"].lower()
        assert "difficulty" in text and "mujoco" in text, text

    def test_add_object_signature_parity_with_base(self):
        # The base SimEngine.add_object declares ``mesh_path`` / ``material``
        # (the "reject a non-None material loudly rather than silently ignore
        # it" contract). The Isaac override must accept them so the tool
        # router / a caller can pass them uniformly instead of hitting a bare
        # TypeError or, worse, a silent **kwargs swallow.
        import inspect

        from strands_robots.simulation.base import SimEngine
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        base = set(inspect.signature(SimEngine.add_object).parameters)
        override = set(inspect.signature(IsaacSimulation.add_object).parameters)
        assert {"mesh_path", "material"} <= override, f"Isaac add_object drops base params: missing {base - override}"

    def test_add_object_material_rejected_with_actionable_error(self):
        # Base contract (SimEngine.add_object docstring): a backend that does
        # not support ``material`` rejects a non-None value loudly rather than
        # silently ignoring it. The Isaac add_object only sets a flat color, so
        # a material spec is rejected with an actionable error before the stage
        # boots (holds on any host, Isaac Sim present or not).
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.add_object("block", material={"albedo": [0.2, 0.2, 0.2]})
        assert result["status"] == "error"
        text = result["content"][0]["text"].lower()
        assert "material" in text and "mujoco" in text, text

    def test_add_object_mesh_path_on_primitive_shape_rejected(self):
        # A ``mesh_path`` is only consumed by ``shape="mesh"`` (#2459); on a
        # primitive shape it would be silently ignored, so it is refused with
        # an error naming the remedy rather than dropped. Fires before the
        # world check, so it holds on any host.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.add_object("part", mesh_path="/tmp/widget.obj")
        assert result["status"] == "error"
        text = result["content"][0]["text"].lower()
        assert "shape='mesh'" in text, text

    def test_add_object_mesh_shape_requires_mesh_path(self):
        # MuJoCo-parity contract: shape='mesh' without a mesh_path names the
        # missing parameter rather than degrading to a default primitive.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.add_object("part", shape="mesh")
        assert result["status"] == "error"
        assert "mesh_path" in result["content"][0]["text"]

    def test_add_object_mesh_missing_file_rejected(self):
        # Fail-loud: a mesh asset that does not exist is refused by name
        # before anything touches the stage - never a silent fallback box.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.add_object("part", shape="mesh", mesh_path="/no/such/widget.obj")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "not found" in text and "/no/such/widget.obj" in text

    def test_add_object_mesh_unsupported_format_rejected(self, tmp_path):
        # A format the converter cannot parse (e.g. a COLLADA .dae) is
        # refused with the supported set named.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        asset = tmp_path / "part.dae"
        asset.write_bytes(b"\x00\x01")
        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.add_object("part", shape="mesh", mesh_path=str(asset))
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert ".dae" in text and ".obj" in text

    def test_add_object_mesh_with_valid_asset_reaches_world_check(self, tmp_path):
        # The behaviour test replacing the old explicit-refusal pin (#2459):
        # a usable mesh asset passes the mesh validation and proceeds to the
        # ordinary lifecycle gates - on a host without Isaac Sim that is the
        # "No world created" error, NOT the retired "not supported on the
        # Isaac backend yet" refusal. The live placement half runs on GPU in
        # tests_integ/.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        asset = tmp_path / "widget.obj"
        asset.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.add_object("part", shape="mesh", mesh_path=str(asset))
        assert result["status"] == "error"
        text = result["content"][0]["text"].lower()
        assert "no world" in text, text
        assert "not supported" not in text

    def test_add_robot_signature_parity_with_base(self):
        # The base SimEngine.add_robot declares ``keyframe`` (spawn at a
        # canonical <keyframe> pose). Every backend override must accept it
        # so a caller / the tool router can pass it uniformly; a narrower
        # override raises a bare TypeError on a documented parameter instead
        # of the contract's actionable error (cf. create_world terrain and
        # add_object material/mesh_path parity).
        import inspect

        from strands_robots.simulation.base import SimEngine
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        base = set(inspect.signature(SimEngine.add_robot).parameters)
        override = set(inspect.signature(IsaacSimulation.add_robot).parameters)
        assert {"keyframe"} <= override, f"Isaac add_robot drops base params: missing {base - override}"

    def test_add_robot_keyframe_rejected_with_actionable_error(self):
        # Base contract (SimEngine.add_robot docstring): an unknown/unsupported
        # keyframe is a hard error that never silently falls back to zeros. The
        # Isaac backend does not parse the MuJoCo <keyframe> block, so a non-None
        # keyframe is rejected with an actionable error - NOT a bare TypeError,
        # NOT a silent zero-pose spawn. The rejection fires before Isaac Sim
        # boots, so it holds on any host.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.add_robot("panda", keyframe="home")
        assert result["status"] == "error"
        text = result["content"][0]["text"].lower()
        assert "keyframe" in text and "mujoco" in text, text

    def test_add_robot_keyframe_none_not_rejected(self):
        # keyframe=None (the default) must NOT hit the reject path - a plain
        # add_robot call still degrades to the structured Isaac-Sim-absent /
        # no-world error rather than the keyframe rejection.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = sim.add_robot("panda")
        assert result["status"] == "error"
        assert "keyframe" not in result["content"][0]["text"].lower()


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


class TestMainThreadAffinityGuard:
    """Worker-thread calls into Isaac's main-thread-affine entry points.

    Isaac's kit runtime only pumps updates on the thread that created
    ``SimulationApp``, so ``reset()`` / ``step()`` called from a worker thread
    used to block **forever** with zero signal (#1896: a Strands ``Agent``
    ran ``evaluate_benchmark`` on a tool-executor thread while the main
    thread waited on the tool future; ``SimulationContext.stop()`` never
    returned). These tests pin the guard that replaces the deadlock:

      * worker thread + NO pump -> loud ``RuntimeError`` naming the
        run_on_main / run_pump_forever recipe, world untouched;
      * worker thread + pump running -> auto-marshalled onto the owning
        thread via ``run_on_main`` (the recording facade's schema-probe
        pattern);
      * owning thread -> runs inline, unchanged.

    Stub-based - no Isaac Sim required. The world stub records which thread
    each call landed on so the marshalling is asserted, not inferred.
    """

    class _StubWorld:
        def __init__(self):
            self.reset_calls: list[int] = []
            self.step_calls: list[int] = []

        def reset(self):
            self.reset_calls.append(threading.get_ident())

        def step(self, render=False):
            self.step_calls.append(threading.get_ident())

    @pytest.fixture()
    def sim_and_world(self):
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        world = self._StubWorld()
        # The minimal live-world state reset()/step() consult. No SimulationApp
        # boots here: the stub stands in for the kit-affine World handle.
        sim._world_created = True
        sim._world = world
        yield sim, world
        # Detach the stub before GC so SimEngine.__del__ -> cleanup() ->
        # destroy() never runs real teardown against it.
        sim._world_created = False
        sim._world = None

    @staticmethod
    def _future_from_worker(fn):
        """Run ``fn`` on a worker thread; return the completed Future.

        ``__exit__`` joins the worker (``shutdown(wait=True)``), so the future
        is done on return and ``future.result()`` re-raises whatever the worker
        raised - ``SystemExit`` included - with identity preserved. No
        exception-translating handler exists here on purpose (review on #1899):
        the assertions do the catching via ``pytest.raises``. A genuine
        regression to the #1896 deadlock blocks in ``__exit__`` until the
        suite-wide pytest-timeout (``--timeout=120`` in addopts) kills it.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(fn)

    def test_worker_thread_reset_without_pump_raises_actionable_error(self, sim_and_world):
        sim, world = sim_and_world
        future = self._future_from_worker(sim.reset)
        # The error must carry the recipe, not just a refusal.
        with pytest.raises(RuntimeError, match=r"run_pump_forever.*run_on_main"):
            future.result()
        assert world.reset_calls == []  # world never touched off-thread

    def test_worker_thread_step_without_pump_raises_actionable_error(self, sim_and_world):
        sim, world = sim_and_world
        future = self._future_from_worker(lambda: sim.step(3))
        with pytest.raises(RuntimeError, match="run_pump_forever"):
            future.result()
        assert world.step_calls == []

    def test_worker_thread_reset_with_pump_running_is_marshalled_to_main(self, sim_and_world):
        sim, world = sim_and_world
        sim._pump_running = True  # what run_pump_forever sets while it owns the renderer
        main_ident = threading.get_ident()

        worker_box = {}

        def _worker():
            worker_box["result"] = sim.reset()

        t = threading.Thread(target=_worker)
        t.start()
        # Play the pump's role: execute the job the worker enqueued via
        # run_on_main on this (owning) thread.
        job = sim._main_jobs.get(timeout=10)
        job()
        t.join(timeout=10)
        assert not t.is_alive()
        assert worker_box["result"]["status"] == "success"
        assert world.reset_calls == [main_ident]  # ran HERE, not on the worker

    def test_main_thread_reset_and_step_run_inline(self, sim_and_world):
        sim, world = sim_and_world
        main_ident = threading.get_ident()
        assert sim.reset()["status"] == "success"
        assert sim.step(2)["status"] == "success"
        assert world.reset_calls == [main_ident]
        assert world.step_calls == [main_ident, main_ident]

    def test_worker_thread_reset_without_world_keeps_structured_error(self):
        # No world -> nothing kit-affine can run, so the documented
        # structured error dict wins over the threading lecture.
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        sim = IsaacSimulation(num_envs=1, headless=True)
        result = self._future_from_worker(sim.reset).result()  # re-raises if the worker raised
        assert result["status"] == "error"
        assert "No world created" in result["content"][0]["text"]


class TestInstallMetadata:
    def test_pip_extra_points_at_in_tree_extra(self):
        from strands_robots.simulation.isaac import _install

        assert _install.PIP_EXTRA == "pip install 'strands-robots[sim-isaac]'"

    def test_not_importable_reason_mentions_install_paths(self):
        from strands_robots.simulation.isaac import _install

        reason = _install.not_importable_reason()
        assert "Omniverse" in reason
        assert _install.ISAAC_SIM_DOCKER_IMAGE in reason

    def test_not_importable_reason_lists_pip_route_with_caveats(self):
        """#1803: cp312 pip wheels made ``isaacsim`` pip-installable; the
        availability probe's guidance must list the pip route first, with
        the coverage-reinstall and EULA caveats that the pip route needs."""
        from strands_robots.simulation.isaac import _install

        reason = _install.not_importable_reason()
        # The stale "not via pip" guidance must be gone.
        assert "not via pip" not in reason
        # Pip route present, listed before the Launcher option.
        assert _install.ISAAC_SIM_PIP_INSTALL in reason
        assert "isaacsim[all,extscache]" in reason
        assert reason.index("pypi.nvidia.com") < reason.index("Omniverse")
        # Pip-route caveats: coverage reinstall + EULA env var.
        assert "coverage>=7.6" in reason
        assert "OMNI_KIT_ACCEPT_EULA=YES" in reason

    def test_inline_install_options_mention_pip_route(self):
        from strands_robots.simulation.isaac import _install

        inline = _install.install_options_inline()
        assert "pip" in inline
        assert "isaacsim[all,extscache]" in inline
        assert _install.ISAAC_SIM_DOCKER_IMAGE in inline
